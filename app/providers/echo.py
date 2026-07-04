"""A deterministic, dependency-free provider.

Lets you run and demo the whole stack — REST, streaming, realtime voice —
with no model installed and no API key. Also the workhorse of the test
suite. Replies acknowledge the user's words so conversations visibly flow.
"""

import asyncio
from itertools import cycle
from typing import AsyncIterator, List

from .base import LLMProvider, Message

_EMOTIONS = cycle(["happy", "caring", "excited", "thinking"])


class EchoProvider(LLMProvider):
    name = "echo"
    model = "echo-1"

    def __init__(self, delay: float = 0.0) -> None:
        self._delay = delay

    def _reply(self, messages: List[Message]) -> str:
        last_user = ""
        for message in reversed(messages):
            if message["role"] == "user":
                last_user = message["content"]
                break
        if not last_user:
            return "Hey, I'm here! What's on your mind?\n[emotion: happy]"
        snippet = last_user.strip()
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        return (
            f"You said “{snippet}” — tell me more?\n"
            f"[emotion: {next(_EMOTIONS)}]"
        )

    async def chat(self, messages: List[Message]) -> str:
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._reply(messages)

    async def chat_stream(self, messages: List[Message]) -> AsyncIterator[str]:
        text = await self.chat(messages)
        # Emit in small chunks so streaming consumers get exercised for real.
        for i in range(0, len(text), 8):
            yield text[i : i + 8]
