"""Speech-to-text engines for the realtime pipeline.

``whisper`` uses faster-whisper on this machine (install via
``requirements-stt.txt``). If it isn't installed, sessions gracefully fall
back to *client mode* — the phone does on-device STT and sends text — so
the backend always runs, just with turn-based ears instead of streaming
ones. ``fake`` is for tests.
"""

import asyncio
import logging
from typing import Optional

from ..config import Settings

logger = logging.getLogger("vyra.stt")

try:  # pragma: no cover - optional dependency
    from faster_whisper import WhisperModel  # type: ignore

    _WHISPER_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    WhisperModel = None  # type: ignore
    _WHISPER_AVAILABLE = False


class SttEngine:
    """Interface: transcribe one PCM16 mono utterance to text."""

    name = "base"

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class WhisperStt(SttEngine):
    name = "whisper"

    def __init__(self, model: str, device: str = "auto", compute_type: str = "int8") -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._model = None
        self._lock = asyncio.Lock()

    async def _ensure_model(self):
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    logger.info(
                        "Loading whisper model '%s' (%s/%s)…",
                        self._model_name, self._device, self._compute_type,
                    )
                    loop = asyncio.get_event_loop()
                    self._model = await loop.run_in_executor(
                        None,
                        lambda: WhisperModel(
                            self._model_name,
                            device=self._device,
                            compute_type=self._compute_type,
                        ),
                    )
                    logger.info("Whisper model ready.")
        return self._model

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        model = await self._ensure_model()
        loop = asyncio.get_event_loop()

        def _run(m) -> str:
            import numpy as np  # faster-whisper depends on numpy

            audio = (
                np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            )
            segments, _info = m.transcribe(
                audio, language="en", beam_size=1, vad_filter=False
            )
            return " ".join(seg.text.strip() for seg in segments).strip()

        try:
            return await loop.run_in_executor(None, lambda: _run(model))
        except Exception as exc:  # noqa: BLE001 - inspect for GPU issues
            # `device=auto` picks CUDA on machines with an NVIDIA GPU, but
            # the CUDA runtime (cuBLAS/cuDNN DLLs) is often not installed.
            # Fall back to CPU transparently — base/int8 is realtime on CPU.
            message = str(exc).lower()
            gpu_issue = any(
                key in message for key in ("cublas", "cudnn", "cuda", "hip")
            )
            if gpu_issue and self._device != "cpu":
                logger.warning(
                    "Whisper GPU path failed (%s) — falling back to CPU.", exc
                )
                self._device = "cpu"
                self._model = None
                model = await self._ensure_model()
                return await loop.run_in_executor(None, lambda: _run(model))
            raise


class FakeStt(SttEngine):
    """Deterministic engine for tests: pops queued transcripts."""

    name = "fake"

    def __init__(self) -> None:
        self.queue: "asyncio.Queue[str]" = asyncio.Queue()
        self.received: list = []

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        self.received.append((len(pcm), sample_rate))
        try:
            return self.queue.get_nowait()
        except asyncio.QueueEmpty:
            return "hello vyra"


def stt_mode(settings: Settings) -> str:
    """What the /config endpoint reports: 'server' or 'client'."""
    kind = settings.stt_provider.strip().lower()
    if kind == "whisper" and _WHISPER_AVAILABLE:
        return "server"
    if kind == "fake":
        return "server"
    return "client"


def build_stt(settings: Settings) -> Optional[SttEngine]:
    """Returns an engine, or None → sessions run in client-STT mode."""
    kind = settings.stt_provider.strip().lower()
    if kind == "whisper":
        if not _WHISPER_AVAILABLE:
            logger.warning(
                "STT_PROVIDER=whisper but faster-whisper is not installed "
                "(pip install -r requirements-stt.txt). Falling back to "
                "client-side STT."
            )
            return None
        return WhisperStt(
            model=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
    if kind == "fake":
        return FakeStt()
    return None
