"""The provider contract every LLM backend implements.

Messages use the OpenAI-style shape ``{"role": ..., "content": ...}`` with
roles ``system`` / ``user`` / ``assistant``; each provider maps that to its
own wire format. The rest of the codebase never knows which provider it got.
"""

import abc
from typing import AsyncIterator, Dict, List

Message = Dict[str, str]


class ProviderError(RuntimeError):
    """Raised when a provider call fails (network, auth, bad response)."""


class LLMProvider(abc.ABC):
    name: str = "base"
    model: str = ""

    @abc.abstractmethod
    async def chat(self, messages: List[Message]) -> str:
        """Return the full completion for ``messages``."""

    @abc.abstractmethod
    def chat_stream(self, messages: List[Message]) -> AsyncIterator[str]:
        """Yield completion deltas for ``messages``."""

    async def aclose(self) -> None:  # pragma: no cover - trivial default
        """Release any underlying resources (HTTP clients)."""
        return None
