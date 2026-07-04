"""Ollama provider — the fully local, private brain.

Talks to an Ollama server (usually on this same machine) via its native
``/api/chat`` endpoint. Non-streaming returns one JSON object; streaming
returns newline-delimited JSON.
"""

import json
import re
from typing import AsyncIterator, List, Optional

import httpx

from .base import LLMProvider, Message, ProviderError

# Reasoning models (qwen3, deepseek-r1, ...) may wrap chain-of-thought in
# <think> blocks inside content; that must never reach the voice pipeline.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        host: str,
        model: str,
        temperature: float = 0.9,
        timeout: float = 120.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._client = client or httpx.AsyncClient(
            base_url=host.rstrip("/"), timeout=timeout
        )

    def _payload(self, messages: List[Message], stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": self._temperature},
        }

    async def chat(self, messages: List[Message]) -> str:
        try:
            response = await self._client.post(
                "/api/chat", json=self._payload(messages, stream=False)
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            # Ollama's error bodies are the useful part — e.g. a 404 body is
            # `model 'x' not found, try pulling it first`.
            detail = ""
            try:
                detail = (exc.response.text or "").strip()[:300]
            except Exception:  # noqa: BLE001
                pass
            hint = f" — {detail}" if detail else ""
            if exc.response.status_code == 404:
                hint += (
                    f" (is the model pulled? try: ollama pull {self.model})"
                )
            raise ProviderError(f"Ollama request failed: {exc}{hint}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc
        try:
            message = data["message"]
            content = message.get("content") or ""
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"Unexpected Ollama response: {data!r}") from exc
        content = _THINK_BLOCK.sub("", content).strip()
        if not content:
            # Newer Ollama can put reasoning-model output in message.thinking;
            # and some model/template combos emit an instant end-of-turn.
            # Surface it loudly instead of letting an empty reply flow on.
            thinking = (message.get("thinking") or "").strip()
            raise ProviderError(
                f"Ollama returned an empty reply from '{self.model}'"
                + (" (only 'thinking' output — try a non-reasoning model "
                   "like llama3.1)" if thinking else
                   f" — sanity-check it with: ollama run {self.model} \"say hi\"")
            )
        return content

    async def chat_stream(self, messages: List[Message]) -> AsyncIterator[str]:
        try:
            async with self._client.stream(
                "POST", "/api/chat", json=self._payload(messages, stream=True)
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    delta = (chunk.get("message") or {}).get("content", "")
                    if delta:
                        yield delta
                    if chunk.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama stream failed: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()
