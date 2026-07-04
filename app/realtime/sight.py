"""Situational sight: periodic camera glimpses through a local vision LLM.

Presence/smile/eyes (from on-device ML Kit) tell her *that* you're there.
This layer tells her *what's going on*: every N seconds the phone sends one
downscaled JPEG; a local multimodal model (via Ollama — e.g. ``moondream``,
``llava``) turns it into one short factual sentence, which is injected into
her conversation context. She doesn't narrate the feed — she simply *knows*
("you're holding a mug", "you look dressed up today") and uses it when it's
natural, exactly like a person in the room.

Privacy: frames go only to YOUR machine over the LAN, are described, and
are immediately discarded. Disabled unless VISION_LLM_MODEL is set.
"""

import base64
import logging
import time
from typing import Optional

import httpx

from ..config import Settings

logger = logging.getLogger("vyra.sight")

_DESCRIBE_PROMPT = (
    "In one short factual sentence, describe the person and what they are "
    "doing, holding, or wearing in this image. If no person is visible, "
    "say what the scene shows. No preamble, no speculation."
)


class SightEngine:
    """Interface: turn one JPEG into a one-line scene description."""

    name = "base"

    async def describe(self, jpeg: bytes) -> Optional[str]:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class OllamaSight(SightEngine):
    name = "ollama"

    def __init__(self, host: str, model: str, timeout: float = 45.0) -> None:
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=host.rstrip("/"), timeout=timeout
        )

    async def describe(self, jpeg: bytes) -> Optional[str]:
        try:
            response = await self._client.post(
                "/api/generate",
                json={
                    "model": self.model,
                    "prompt": _DESCRIBE_PROMPT,
                    "images": [base64.b64encode(jpeg).decode()],
                    "stream": False,
                },
            )
            response.raise_for_status()
            text = (response.json().get("response") or "").strip()
            return text or None
        except Exception as exc:  # noqa: BLE001 - sight must never kill a turn
            logger.warning("vision glimpse failed: %s", exc)
            return None

    async def aclose(self) -> None:
        await self._client.aclose()


class FakeSight(SightEngine):
    """Deterministic engine for tests."""

    name = "fake"

    def __init__(self) -> None:
        self.frames: list = []

    async def describe(self, jpeg: bytes) -> Optional[str]:
        self.frames.append(len(jpeg))
        return "a person at a desk holding a red mug"


class Glimpse:
    """The latest scene description, with freshness tracking."""

    def __init__(self) -> None:
        self.text: Optional[str] = None
        self._at = 0.0

    def update(self, text: str) -> None:
        self.text = text
        self._at = time.monotonic()

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self._at

    def fresh(self, max_age: float = 120.0) -> Optional[str]:
        if self.text and self.age_seconds <= max_age:
            return self.text
        return None


def build_sight(settings: Settings) -> Optional[SightEngine]:
    model = settings.vision_llm_model.strip()
    if not model:
        return None
    if model == "fake":
        return FakeSight()
    return OllamaSight(host=settings.ollama_host, model=model)
