"""The realtime companion session — Vyra's social brain.

One instance per websocket connection. It owns the conversation state
machine:

::

    LISTENING ──(endpoint + STT)──► THINKING ──(LLM reply)──► SPEAKING
        ▲                                │                        │
        │◄──────── barge-in / TTS finished / timeout ◄────────────┘

Friend-like behaviours implemented here:

* **She waits for you to finish** — endpointing via the VAD, not a button.
* **You can talk over her** — sustained voice while she speaks sends
  ``tts.interrupt`` (barge-in) and she yields the floor.
* **She can start the conversation** — a greeting shortly after you appear
  (or connect), and gentle re-engagement nudges after a lull, capped so she
  never nags.
* **She notices you** — on-device vision labels (present/smiling) feed her
  context and can trigger the greeting when you sit down.
"""

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

from ..config import Settings
from ..conversation import SessionHistory, build_messages
from ..personality import (
    GREETING_INSTRUCTION,
    PROACTIVE_INSTRUCTION,
    parse_emotion,
)
from ..providers.base import LLMProvider, ProviderError
from ..schemas import VisionContext
from . import protocol
from .stt import SttEngine
from .vad import SPEECH_END, SPEECH_START, VadStream

logger = logging.getLogger("vyra.session")

SendFn = Callable[[dict], Awaitable[None]]

_FALLBACK_LINE = (
    "Hmm, I couldn't reach my brain just now. Give me a second and try again?"
)


class RealtimeSession:
    def __init__(
        self,
        send: SendFn,
        settings: Settings,
        provider: LLMProvider,
        stt: Optional[SttEngine],
        user_name: Optional[str] = None,
        sample_rate: int = 16000,
        greet: bool = True,
        client_stt: bool = False,
    ) -> None:
        self._send = send
        self._settings = settings
        self._provider = provider
        self._stt = None if client_stt else stt
        self._user_name = user_name
        self._sample_rate = sample_rate
        self._greet_enabled = greet and settings.greet_on_connect

        self.history = SessionHistory(settings.max_history_turns)
        self.state = protocol.LISTENING
        self.muted = False

        self._vad = VadStream(
            sample_rate=sample_rate,
            frame_ms=settings.vad_frame_ms,
            start_ms=settings.vad_start_ms,
            end_silence_ms=settings.vad_end_silence_ms,
            min_utterance_ms=settings.vad_min_utterance_ms,
            pre_roll_ms=settings.vad_pre_roll_ms,
            min_rms=settings.vad_min_rms,
            noise_multiplier=settings.vad_noise_multiplier,
            barge_start_ms=settings.barge_min_ms,
            barge_noise_multiplier=settings.barge_noise_multiplier,
        )

        self.vision: Optional[VisionContext] = None
        self._say_id = 0
        self._greeted = False
        self._nudges = 0
        self._started_at = time.monotonic()
        self._last_activity = time.monotonic()

        self._think_task: Optional[asyncio.Task] = None
        self._speak_timeout_task: Optional[asyncio.Task] = None
        self._companion_task: Optional[asyncio.Task] = None
        self._closed = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        await self._send(
            protocol.event(
                "session.ready",
                provider=self._provider.name,
                model=self._provider.model,
                stt="server" if self._stt else "client",
                sample_rate=self._sample_rate,
            )
        )
        await self._set_state(protocol.LISTENING)
        self._companion_task = asyncio.ensure_future(self._companion_loop())

    async def close(self) -> None:
        self._closed = True
        for task in (self._think_task, self._speak_timeout_task, self._companion_task):
            if task is not None:
                task.cancel()

    # ------------------------------------------------------------------ #
    # Inbound events (called by the websocket endpoint)
    # ------------------------------------------------------------------ #
    async def on_audio(self, pcm: bytes) -> None:
        if self.muted or self._stt is None:
            return
        for kind, payload in self._vad.feed(pcm):
            if kind == SPEECH_START:
                await self._on_speech_start()
            elif kind == SPEECH_END:
                await self._on_utterance(payload)

    async def on_user_text(self, text: str) -> None:
        """Final transcript from the phone (client-STT mode) or typed text."""
        text = text.strip()
        if text:
            await self._handle_user_final(text)

    async def on_vision(self, vision: VisionContext) -> None:
        arrived = vision.present and not (self.vision.present if self.vision else False)
        self.vision = vision
        # Their friend just sat down and nothing has been said yet → greet.
        if (
            arrived
            and self._greet_enabled
            and not self._greeted
            and self.history.user_turn_count == 0
            and self.state == protocol.LISTENING
        ):
            self._greeted = True
            await self._say_instruction(GREETING_INSTRUCTION, proactive=True)

    async def on_tts_state(self, playing: bool) -> None:
        if playing:
            self._restart_speak_timeout()
        else:
            await self._end_speaking()

    async def on_mic_state(self, muted: bool) -> None:
        self.muted = muted
        if muted:
            await self._set_state(protocol.IDLE)
        else:
            self._last_activity = time.monotonic()
            if self.state == protocol.IDLE:
                await self._set_state(protocol.LISTENING)

    # ------------------------------------------------------------------ #
    # Speech events
    # ------------------------------------------------------------------ #
    async def _on_speech_start(self) -> None:
        self._last_activity = time.monotonic()
        if self.state == protocol.SPEAKING:
            await self._barge_in()
        elif self.state == protocol.THINKING:
            # They kept talking — drop the pending reply and keep listening.
            self._cancel_think()
            await self._set_state(protocol.LISTENING)

    async def _on_utterance(self, pcm: bytes) -> None:
        if self._stt is None:
            return
        try:
            text = await self._stt.transcribe(pcm, self._sample_rate)
        except Exception as exc:  # noqa: BLE001 - STT must never kill a session
            logger.error("STT failed: %s", exc)
            await self._send(protocol.event("error", message=f"STT failed: {exc}"))
            return
        text = text.strip()
        if not text:
            logger.info(
                "utterance (%.1fs audio) transcribed to empty text — ignored",
                len(pcm) / 2 / self._sample_rate,
            )
            return
        logger.info("heard (%.1fs): %r", len(pcm) / 2 / self._sample_rate, text)
        await self._send(protocol.event("user.final", text=text))
        await self._handle_user_final(text)

    async def _handle_user_final(self, text: str) -> None:
        self._nudges = 0
        self._greeted = True  # they spoke first; skip the scripted greeting
        self._last_activity = time.monotonic()
        if self.state == protocol.SPEAKING:
            await self._barge_in()
        self._cancel_think()
        self.history.add_user(text)
        self._think_task = asyncio.ensure_future(self._think())

    # ------------------------------------------------------------------ #
    # Thinking / speaking
    # ------------------------------------------------------------------ #
    async def _think(self, extra_instruction: Optional[str] = None) -> None:
        await self._set_state(protocol.THINKING)
        messages = build_messages(
            self.history.turns,
            user_name=self._user_name,
            vision=self.vision,
            max_history_turns=self._settings.max_history_turns,
            extra_instruction=extra_instruction,
        )
        try:
            raw = await self._provider.chat(messages)
        except asyncio.CancelledError:
            raise
        except ProviderError as exc:
            logger.error("LLM failed: %s", exc)
            await self._send(protocol.event("error", message=str(exc)))
            await self._say(_FALLBACK_LINE, "sad", remember=False)
            return
        logger.info("llm raw (%d chars): %r", len(raw), raw[:160])
        text, emotion = parse_emotion(raw)
        if not text:
            logger.warning(
                "LLM reply parsed to empty text (raw was %r) — asking to repeat",
                raw[:200],
            )
            text, emotion = "Sorry, say that again?", "thinking"
        await self._say(text, emotion, proactive=extra_instruction is not None)

    async def _say(
        self,
        text: str,
        emotion: str,
        proactive: bool = False,
        remember: bool = True,
    ) -> None:
        if remember:
            self.history.add_assistant(text)
        self._say_id += 1
        self._last_activity = time.monotonic()
        await self._set_state(protocol.SPEAKING)
        self._vad.set_barge_mode(True)
        await self._send(
            protocol.event(
                "assistant.say",
                id=self._say_id,
                text=text,
                emotion=emotion,
                proactive=proactive,
            )
        )
        self._restart_speak_timeout()

    async def _say_instruction(self, instruction: str, proactive: bool) -> None:
        self._cancel_think()
        self._think_task = asyncio.ensure_future(
            self._think(extra_instruction=instruction)
        )

    async def _barge_in(self) -> None:
        """The user started talking over Vyra — she yields the floor."""
        self._cancel_think()
        await self._send(protocol.event("tts.interrupt", id=self._say_id))
        await self._end_speaking()

    async def _end_speaking(self) -> None:
        if self._speak_timeout_task is not None:
            self._speak_timeout_task.cancel()
            self._speak_timeout_task = None
        self._vad.set_barge_mode(False)
        self._last_activity = time.monotonic()
        if self.state == protocol.SPEAKING:
            await self._set_state(
                protocol.IDLE if self.muted else protocol.LISTENING
            )

    def _restart_speak_timeout(self) -> None:
        if self._speak_timeout_task is not None:
            self._speak_timeout_task.cancel()

        async def _timeout() -> None:
            await asyncio.sleep(self._settings.speak_timeout_seconds)
            logger.warning("speak timeout — phone never reported TTS finish")
            await self._end_speaking()

        self._speak_timeout_task = asyncio.ensure_future(_timeout())

    def _cancel_think(self) -> None:
        if self._think_task is not None and not self._think_task.done():
            self._think_task.cancel()
        self._think_task = None

    # ------------------------------------------------------------------ #
    # Proactivity — Vyra can start the conversation
    # ------------------------------------------------------------------ #
    async def _companion_loop(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(0.25)
                if self.muted or self.state != protocol.LISTENING:
                    continue
                if self._vad.in_speech:
                    continue
                now = time.monotonic()
                if (
                    self._greet_enabled
                    and not self._greeted
                    and self.history.user_turn_count == 0
                    and now - self._started_at >= self._settings.greeting_delay_seconds
                ):
                    self._greeted = True
                    await self._say_instruction(GREETING_INSTRUCTION, proactive=True)
                    continue
                if (
                    self._nudges < self._settings.proactive_max_nudges
                    and now - self._last_activity
                    >= self._settings.proactive_idle_seconds
                ):
                    self._nudges += 1
                    await self._say_instruction(PROACTIVE_INSTRUCTION, proactive=True)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------ #
    async def _set_state(self, value: str) -> None:
        self.state = value
        await self._send(protocol.event("state", value=value))
