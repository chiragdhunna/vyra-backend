"""The `/realtime` websocket protocol.

Transport
---------
* **Text frames** — JSON events, ``{"type": ..., ...}`` both directions.
* **Binary frames** — raw microphone audio, PCM16 little-endian mono at the
  sample rate announced in ``session.start`` (16 kHz recommended).

Client → server events
----------------------
``session.start``   {user_name?, sample_rate=16000, greet=true, client_stt=false}
``vision.state``    {present: bool, smiling: bool}   (on-device ML Kit labels)
``user.text``       {text}      final transcript when the CLIENT does STT
``tts.state``       {playing: bool}   device TTS started / finished a say-id
``mic.state``       {muted: bool}
``ping``            {}

Server → client events
----------------------
``session.ready``   {provider, model, stt: "server"|"client", version}
``state``           {value: "listening"|"thinking"|"speaking"|"idle"}
``user.final``      {text}                    what the server heard (server STT)
``assistant.say``   {id, text, emotion, proactive: bool}  speak this + animate
``tts.interrupt``   {id}                      barge-in: stop speaking NOW
``error``           {message}
``pong``            {}
"""

from typing import Any, Dict

# States
LISTENING = "listening"
THINKING = "thinking"
SPEAKING = "speaking"
IDLE = "idle"


def event(type_: str, **fields: Any) -> Dict[str, Any]:
    payload = {"type": type_}
    payload.update(fields)
    return payload
