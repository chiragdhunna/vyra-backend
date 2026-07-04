"""Conversation assembly: system prompt + context + windowed history.

One place decides what the model sees. Used by both the stateless REST
endpoints (history arrives with the request) and realtime sessions
(history lives in the session).
"""

from collections import deque
from typing import Deque, List, Optional

from .personality import VYRA_SYSTEM_PROMPT
from .providers.base import Message
from .schemas import ChatTurn, VisionContext


def context_note(
    user_name: Optional[str] = None, vision: Optional[VisionContext] = None
) -> str:
    """A short situational note appended to the system prompt."""
    lines = []
    if user_name:
        lines.append(f"Your friend's name is {user_name}.")
    if vision is not None:
        if vision.present and vision.smiling:
            lines.append("Camera sense: your friend is here right now, and they're smiling.")
        elif vision.present:
            lines.append("Camera sense: your friend is here right now.")
        else:
            lines.append("Camera sense: you can't see your friend at the moment.")
    return "\n".join(lines)


def build_messages(
    turns: List[ChatTurn],
    user_name: Optional[str] = None,
    vision: Optional[VisionContext] = None,
    max_history_turns: int = 24,
    extra_instruction: Optional[str] = None,
) -> List[Message]:
    """Assemble the provider payload from history + context."""
    system = VYRA_SYSTEM_PROMPT
    note = context_note(user_name, vision)
    if note:
        system = f"{system}\nContext:\n{note}"

    window = turns[-max_history_turns:] if max_history_turns > 0 else turns

    # Greetings / proactive nudges arrive as an extra instruction. When the
    # conversation already has turns, it rides along in the system prompt.
    # But with ZERO turns (fresh session greeting), many instruct models —
    # llama3.1 included — emit an immediate end-of-turn for a system-only
    # prompt, producing an EMPTY reply. So with no history the instruction
    # becomes the sole user-role message instead, which reliably generates.
    if extra_instruction and window:
        system = f"{system}\n\nRight now:\n{extra_instruction}"

    messages: List[Message] = [{"role": "system", "content": system}]
    for turn in window:
        messages.append({"role": turn.role, "content": turn.content})
    if extra_instruction and not window:
        messages.append({"role": "user", "content": extra_instruction})
    return messages


class SessionHistory:
    """Bounded rolling history for a realtime session."""

    def __init__(self, max_turns: int = 24) -> None:
        self._turns: Deque[ChatTurn] = deque(maxlen=max(2, max_turns))

    def add_user(self, text: str) -> None:
        self._turns.append(ChatTurn(role="user", content=text))

    def add_assistant(self, text: str) -> None:
        self._turns.append(ChatTurn(role="assistant", content=text))

    @property
    def turns(self) -> List[ChatTurn]:
        return list(self._turns)

    @property
    def user_turn_count(self) -> int:
        return sum(1 for t in self._turns if t.role == "user")
