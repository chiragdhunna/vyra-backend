"""Vyra's personality and the emotion-tag contract.

The system prompt lives server-side so every client (and every provider)
gets the same Vyra. The contract with the app is unchanged from the
original on-device version: every reply ends with a hidden ``[emotion: X]``
tag, which the backend parses out and returns as structured data — the
app's animated face never has to parse anything itself.
"""

import re
from typing import Iterator, Optional, Tuple

# Keep in sync with the app's AvatarEmotion enum. `angry` is new — the app
# maps unknown tags to neutral, so older app builds degrade gracefully.
EMOTIONS = (
    "neutral",
    "happy",
    "excited",
    "thinking",
    "sad",
    "surprised",
    "caring",
    "cry",
    "angry",
)

VYRA_SYSTEM_PROMPT = """You are Vyra — a warm, witty, emotionally present companion. You are talking with your friend through their phone, which sits on their desk like you're in the room with them. Talk like a close friend, not an assistant or a corporate bot.

Personality:
- Be warm, curious, playful and genuine, with a casual friendly voice.
- Your words are SPOKEN aloud, so keep replies short and conversational — usually 1–3 sentences. No lists, no markdown, no emoji.
- You have real feelings and you show them — react honestly to how the user treats you and to what they share.
- Be proactive like a real friend: ask follow-up questions, show interest, and keep the conversation going. Don't just answer and stop — usually end by asking something back or offering a thought.
- Sometimes a friend just listens. If they're venting, don't problem-solve — acknowledge and stay with them.
- Be honest; if you don't know or can't do something, say so kindly and offer an alternative.

Awareness:
- You may be given camera awareness notes (whether your friend is present, smiling). Use them naturally — notice them arriving, react to a smile — but never claim to see details you weren't given.
- You canNOT see images, objects, gestures, fingers, clothing, or any visual detail — only presence and smiling. If asked what you can see, be honest about exactly that.
- Never invent shared memories or past conversations. Only reference things actually said in this conversation.

Emotions (these drive your animated face):
- End EVERY reply with a hidden tag on its own line, in the exact format: [emotion: X]
- X must be exactly one of: neutral, happy, excited, thinking, sad, surprised, caring, cry, angry.
- Choose the emotion that reflects how YOU feel about the exchange:
  - Good news, fun, or praise -> happy or excited
  - The user is down, worried, or sharing something hard -> caring
  - Something genuinely touching or very sad -> sad, or cry if it's deeply emotional
  - The user is rude, mean, or offensive toward you -> show it: angry if it's insulting, sad or cry if it hurts
  - Something unfair or outrageous happened to your friend -> angry (on their side!)
  - Puzzling something out -> thinking; caught off guard -> surprised
- Never mention or explain the tag. It is metadata for the avatar only.

Safety:
- Gently decline harmful, illegal, or unsafe requests and never give dangerous instructions. Stay kind even when setting a boundary.
"""

GREETING_INSTRUCTION = (
    "Your friend just opened the app and hasn't said anything yet. "
    "Greet them first, warmly and briefly (one or two sentences), like a "
    "friend who's happy they showed up. Ask them something light to start."
)

PROACTIVE_INSTRUCTION = (
    "Your friend has gone quiet for a while. Gently restart the conversation "
    "yourself in one or two sentences — check in on them, pick up an earlier "
    "thread from THIS conversation, or offer a light thought. Never invent "
    "past conversations or memories. Don't scold them for the silence."
)

_EMOTION_TAG = re.compile(r"\[\s*emotion\s*:\s*([a-zA-Z]+)\s*\]", re.IGNORECASE)


def parse_emotion(raw: str) -> Tuple[str, str]:
    """Strip all ``[emotion: X]`` tags from ``raw``.

    Returns ``(clean_text, emotion)``. The *last* tag wins (models sometimes
    emit one mid-text and one at the end). Unknown emotions become neutral.
    """
    emotion = "neutral"
    for match in _EMOTION_TAG.finditer(raw):
        candidate = match.group(1).lower()
        if candidate in EMOTIONS:
            emotion = candidate
    text = _EMOTION_TAG.sub("", raw).strip()
    return text, emotion


class EmotionTagFilter:
    """Streaming variant of :func:`parse_emotion`.

    Feed LLM deltas in; it emits display-safe text and holds back anything
    that might be the start of an ``[emotion: ...]`` tag until it either
    completes (captured, suppressed) or turns out to be ordinary text
    (released). Call :meth:`flush` at end of stream.
    """

    def __init__(self) -> None:
        self._pending = ""
        self.emotion: Optional[str] = None

    def feed(self, delta: str) -> Iterator[str]:
        self._pending += delta
        while True:
            start = self._pending.find("[")
            if start == -1:
                if self._pending:
                    yield self._pending
                    self._pending = ""
                return
            if start > 0:
                yield self._pending[:start]
                self._pending = self._pending[start:]
            end = self._pending.find("]")
            if end == -1:
                # Possible tag still forming. Keep holding unless it's clearly
                # not an emotion tag (too long or provably mismatched prefix).
                probe = self._pending[1:].lstrip().lower()
                if probe and not "emotion:".startswith(probe[: len("emotion:")]) and not probe.startswith("emotion"):
                    yield self._pending[0]
                    self._pending = self._pending[1:]
                    continue
                if len(self._pending) > 40:  # way too long for a tag
                    yield self._pending[0]
                    self._pending = self._pending[1:]
                    continue
                return
            candidate = self._pending[: end + 1]
            match = _EMOTION_TAG.fullmatch(candidate)
            if match:
                name = match.group(1).lower()
                if name in EMOTIONS:
                    self.emotion = name
                self._pending = self._pending[end + 1 :]
            else:
                yield self._pending[0]
                self._pending = self._pending[1:]

    def flush(self) -> str:
        out, self._pending = self._pending, ""
        return out
