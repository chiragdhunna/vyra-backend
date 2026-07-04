"""End-to-end websocket tests: audio → VAD → STT → LLM → say → barge-in.

Uses the echo provider + fake STT, so the full realtime loop runs with no
model, key or network. Audio is synthesized PCM16 (sine = speech, zeros =
silence) at 16 kHz — the same wire format the phone sends.
"""

import math
import struct

RATE = 16000


def tone(ms: int, amplitude: int = 9000, freq: float = 220.0) -> bytes:
    n = RATE * ms // 1000
    return struct.pack(
        f"<{n}h",
        *[int(amplitude * math.sin(2 * math.pi * freq * i / RATE)) for i in range(n)],
    )


def silence(ms: int) -> bytes:
    return b"\x00\x00" * (RATE * ms // 1000)


def recv_until(ws, type_, limit=40):
    """Receive events until one of `type_` arrives; returns (event, seen)."""
    wanted = {type_} if isinstance(type_, str) else set(type_)
    seen = []
    for _ in range(limit):
        event = ws.receive_json()
        seen.append(event)
        if event["type"] in wanted:
            return event, seen
    raise AssertionError(f"never saw {wanted}; saw {[e['type'] for e in seen]}")


def start_session(client, **overrides):
    ws = client.websocket_connect("/realtime")
    ws.__enter__()
    payload = {"type": "session.start", "sample_rate": RATE, "greet": False}
    payload.update(overrides)
    ws.send_json(payload)
    ready = ws.receive_json()
    assert ready["type"] == "session.ready"
    state = ws.receive_json()
    assert state == {"type": "state", "value": "listening"}
    return ws, ready


def test_full_voice_turn_server_stt(make_client):
    client = make_client()
    ws, ready = start_session(client)
    try:
        assert ready["stt"] == "server"

        # Speak, then pause long enough to endpoint.
        ws.send_bytes(tone(400))
        ws.send_bytes(silence(300))

        final, _ = recv_until(ws, "user.final")
        assert final["text"] == "hello vyra"  # FakeStt default transcript

        say, seen = recv_until(ws, "assistant.say")
        kinds = [e["type"] for e in seen]
        assert "state" in kinds  # thinking/speaking transitions were sent
        assert "hello vyra" in say["text"]
        assert "[emotion:" not in say["text"]
        assert say["emotion"] != ""
        assert say["proactive"] is False

        # Phone plays TTS, reports finish → floor returns to the user.
        ws.send_json({"type": "tts.state", "playing": True})
        ws.send_json({"type": "tts.state", "playing": False})
        state, _ = recv_until(ws, "state")
        assert state["value"] == "listening"
    finally:
        ws.__exit__(None, None, None)


def test_barge_in_interrupts_speaking(make_client):
    client = make_client()
    ws, _ = start_session(client)
    try:
        ws.send_bytes(tone(400))
        ws.send_bytes(silence(300))
        recv_until(ws, "assistant.say")

        # She's speaking (phone confirms) — now talk over her, sustained.
        ws.send_json({"type": "tts.state", "playing": True})
        ws.send_bytes(tone(600, amplitude=16000))

        interrupt, _ = recv_until(ws, "tts.interrupt")
        assert interrupt["id"] >= 1

        # After yielding the floor she's listening again.
        state, _ = recv_until(ws, "state")
        assert state["value"] == "listening"

        # And the barge utterance itself becomes the next turn.
        ws.send_bytes(silence(300))
        final, _ = recv_until(ws, "user.final")
        assert final["text"] == "hello vyra"
    finally:
        ws.__exit__(None, None, None)


def test_client_stt_mode_text_turns(make_client):
    client = make_client()
    ws, ready = start_session(client, client_stt=True)
    try:
        assert ready["stt"] == "client"
        ws.send_json({"type": "user.text", "text": "good morning"})
        say, _ = recv_until(ws, "assistant.say")
        assert "good morning" in say["text"]
    finally:
        ws.__exit__(None, None, None)


def test_greeting_fires_without_user_input(make_client):
    client = make_client(GREETING_DELAY_SECONDS="0.1")
    ws = client.websocket_connect("/realtime")
    ws.__enter__()
    try:
        ws.send_json({"type": "session.start", "sample_rate": RATE, "greet": True})
        say, _ = recv_until(ws, "assistant.say")
        assert say["proactive"] is True
    finally:
        ws.__exit__(None, None, None)


def test_proactive_nudge_after_lull_is_capped(make_client):
    client = make_client(PROACTIVE_IDLE_SECONDS="0.3", PROACTIVE_MAX_NUDGES="1")
    ws, _ = start_session(client)  # greet=False → silence means a lull
    try:
        say, _ = recv_until(ws, "assistant.say")
        assert say["proactive"] is True
        ws.send_json({"type": "tts.state", "playing": True})
        ws.send_json({"type": "tts.state", "playing": False})
        # Cap is 1: no second nudge should arrive; the next event we force
        # instead is a pong, proving the line stayed quiet.
        import time

        time.sleep(0.7)
        ws.send_json({"type": "ping"})
        event, seen = recv_until(ws, "pong")
        assert all(e["type"] != "assistant.say" for e in seen[:-1])
    finally:
        ws.__exit__(None, None, None)


def test_mute_pauses_ears(make_client):
    client = make_client()
    ws, _ = start_session(client)
    try:
        ws.send_json({"type": "mic.state", "muted": True})
        state, _ = recv_until(ws, "state")
        assert state["value"] == "idle"

        # Audio while muted is ignored entirely.
        ws.send_bytes(tone(400))
        ws.send_bytes(silence(300))

        ws.send_json({"type": "mic.state", "muted": False})
        state, seen = recv_until(ws, "state")
        assert state["value"] == "listening"
        assert all(e["type"] != "user.final" for e in seen)
    finally:
        ws.__exit__(None, None, None)


def test_greeting_includes_wave_gesture(make_client):
    client = make_client(GREETING_DELAY_SECONDS="0.1")
    ws = client.websocket_connect("/realtime")
    ws.__enter__()
    try:
        ws.send_json({"type": "session.start", "sample_rate": RATE, "greet": True})
        say, _ = recv_until(ws, "assistant.say")
        assert say["proactive"] is True
        assert say["gesture"] == "wave"
    finally:
        ws.__exit__(None, None, None)


def test_welcome_back_reaction_after_absence(make_client):
    import time as _time

    client = make_client(
        WELCOME_BACK_AFTER_SECONDS="0.2",
        VISION_REACT_COOLDOWN_SECONDS="0.05",
        PROACTIVE_IDLE_SECONDS="30",
    )
    ws, _ = start_session(client)  # greet=False
    try:
        ws.send_json({"type": "vision.state", "present": True, "smiling": False})
        ws.send_json({"type": "vision.state", "present": False, "smiling": False})
        _time.sleep(0.3)  # away longer than the threshold
        ws.send_json({"type": "vision.state", "present": True, "smiling": False})
        say, _ = recv_until(ws, "assistant.say")
        assert say["proactive"] is True
        assert say["gesture"] == "wave"
    finally:
        ws.__exit__(None, None, None)


def test_tired_eyes_trigger_stretch_checkin(make_client):
    client = make_client(
        TIRED_AFTER_SECONDS="0.2",
        TIRED_REACT_COOLDOWN_SECONDS="0.05",
        VISION_REACT_COOLDOWN_SECONDS="0.05",
        PROACTIVE_IDLE_SECONDS="30",
    )
    ws, _ = start_session(client)
    try:
        ws.send_json({"type": "vision.state", "present": True,
                      "smiling": False, "eyes_open": 0.1})
        say, _ = recv_until(ws, "assistant.say")
        assert say["proactive"] is True
        assert say["gesture"] == "stretch"
    finally:
        ws.__exit__(None, None, None)


def test_laugh_gesture_detection():
    from app.realtime.session import pick_gesture

    assert pick_gesture("Hahaha that's amazing!") == "laugh"
    assert pick_gesture("lol you got me") == "laugh"
    assert pick_gesture("hehe okay okay") == "laugh"
    assert pick_gesture("That is wonderful news!") is None
    assert pick_gesture("What a hat.") is None


def test_ws_auth_rejects_bad_key(make_client):
    import pytest
    from starlette.websockets import WebSocketDisconnect

    client = make_client(VYRA_API_KEY="secret")
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/realtime") as ws:
            ws.receive_json()

    # Correct key via query param works.
    with client.websocket_connect("/realtime?key=secret") as ws:
        ws.send_json({"type": "session.start", "greet": False})
        assert ws.receive_json()["type"] == "session.ready"
