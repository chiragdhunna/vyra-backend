"""Server-side neural text-to-speech.

``edge`` uses Microsoft Edge's free neural voices (no key needed) — a
genuinely natural female voice instead of the robotic device engine. The
MP3 is synthesized here and shipped to the phone over the websocket; if
synthesis fails (offline, throttled) the app falls back to device TTS for
that utterance, so she never goes mute.
"""

import asyncio
import logging
from typing import Optional

from ..config import Settings

logger = logging.getLogger("vyra.tts")

try:  # pragma: no cover - optional dependency
    import edge_tts  # type: ignore

    _EDGE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    edge_tts = None  # type: ignore
    _EDGE_AVAILABLE = False


class TtsEngine:
    """Interface: synthesize speech, returning MP3 bytes (or None to skip)."""

    name = "base"

    async def synthesize(self, text: str) -> Optional[bytes]:
        raise NotImplementedError


class EdgeTts(TtsEngine):
    name = "edge"

    def __init__(self, voice: str, rate: str, pitch: str, timeout: float) -> None:
        self._voice = voice
        self._rate = rate
        self._pitch = pitch
        self._timeout = timeout

    async def synthesize(self, text: str) -> Optional[bytes]:
        try:
            return await asyncio.wait_for(self._run(text), timeout=self._timeout)
        except Exception as exc:  # noqa: BLE001 - voice must never kill a turn
            logger.warning(
                "edge-tts failed (%s) — phone will fall back to device TTS", exc
            )
            return None

    async def _run(self, text: str) -> Optional[bytes]:
        communicate = edge_tts.Communicate(
            text, voice=self._voice, rate=self._rate, pitch=self._pitch
        )
        chunks = []
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                chunks.append(chunk["data"])
        audio = b"".join(chunks)
        return audio or None


class FakeTts(TtsEngine):
    """Deterministic engine for tests."""

    name = "fake"

    def __init__(self) -> None:
        self.spoken: list = []

    async def synthesize(self, text: str) -> Optional[bytes]:
        self.spoken.append(text)
        return b"ID3FAKEMP3" + text[:8].encode()


def tts_mode(settings: Settings) -> str:
    """What session.ready reports: 'server' or 'device'."""
    kind = settings.tts_provider.strip().lower()
    if kind == "edge" and _EDGE_AVAILABLE:
        return "server"
    if kind == "fake":
        return "server"
    return "device"


def build_tts(settings: Settings) -> Optional[TtsEngine]:
    kind = settings.tts_provider.strip().lower()
    if kind == "edge":
        if not _EDGE_AVAILABLE:
            logger.warning(
                "TTS_PROVIDER=edge but edge-tts is not installed "
                "(pip install -r requirements.txt). Using device TTS."
            )
            return None
        return EdgeTts(
            voice=settings.edge_tts_voice,
            rate=settings.edge_tts_rate,
            pitch=settings.edge_tts_pitch,
            timeout=settings.tts_timeout_seconds,
        )
    if kind == "fake":
        return FakeTts()
    return None
