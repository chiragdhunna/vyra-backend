"""Google Gemini provider (REST, no SDK).

Uses the ``generativelanguage.googleapis.com`` v1beta REST API directly so
the backend stays dependency-light. System messages map to
``systemInstruction``; assistant turns map to role ``model``. Streaming uses
``:streamGenerateContent?alt=sse``.
"""

import json
from typing import AsyncIterator, List, Optional, Tuple

import httpx

from .base import LLMProvider, Message, ProviderError


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        temperature: float = 0.9,
        timeout: float = 120.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if not api_key:
            raise ProviderError(
                "AI_PROVIDER=gemini but GEMINI_API_KEY is empty. "
                "Add it to the backend .env."
            )
        self.model = model
        self._api_key = api_key
        self._temperature = temperature
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout
        )

    def _payload(self, messages: List[Message]) -> Tuple[dict, str]:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        contents = []
        for message in messages:
            if message["role"] == "system":
                continue
            role = "user" if message["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": message["content"]}]})
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": self._temperature},
        }
        if system_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }
        return payload, self.model

    @staticmethod
    def _extract_text(data: dict) -> str:
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError):
            return ""
        return "".join(part.get("text", "") for part in parts)

    async def chat(self, messages: List[Message]) -> str:
        payload, model = self._payload(messages)
        try:
            response = await self._client.post(
                f"/models/{model}:generateContent",
                params={"key": self._api_key},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Gemini request failed: {exc}") from exc
        text = self._extract_text(data)
        if not text:
            raise ProviderError(f"Gemini returned no text: {data!r}")
        return text

    async def chat_stream(self, messages: List[Message]) -> AsyncIterator[str]:
        payload, model = self._payload(messages)
        try:
            async with self._client.stream(
                "POST",
                f"/models/{model}:streamGenerateContent",
                params={"key": self._api_key, "alt": "sse"},
                json=payload,
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
                    delta = self._extract_text(chunk)
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise ProviderError(f"Gemini stream failed: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()
