"""OpenAI-compatible provider.

Covers the official OpenAI API *and* everything that speaks its dialect —
LM Studio, Groq, Together, OpenRouter, vLLM, llama.cpp server — just point
``OPENAI_BASE_URL`` at it. Streaming uses standard SSE ``data:`` chunks.
"""

import json
from typing import AsyncIterator, List, Optional

import httpx

from .base import LLMProvider, Message, ProviderError


class OpenAICompatProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.9,
        timeout: float = 120.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.model = model
        self._temperature = temperature
        headers = {}
        if api_key:  # local OpenAI-compatible servers often need no key
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, headers=headers
        )

    def _payload(self, messages: List[Message], stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature,
            "stream": stream,
        }

    async def chat(self, messages: List[Message]) -> str:
        try:
            response = await self._client.post(
                "/chat/completions", json=self._payload(messages, stream=False)
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI-compatible request failed: {exc}") from exc
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Unexpected response: {data!r}") from exc

    async def chat_stream(self, messages: List[Message]) -> AsyncIterator[str]:
        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=self._payload(messages, stream=True)
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[len("data:") :].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    try:
                        delta = chunk["choices"][0]["delta"].get("content", "")
                    except (KeyError, IndexError, TypeError):
                        delta = ""
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI-compatible stream failed: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()
