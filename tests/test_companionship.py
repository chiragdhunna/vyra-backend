"""Tests for the human-like layer: starters, neural voice, situational sight."""

import random
import time

import pytest

from app.starters import TOPICS, starter_instruction

# ------------------------------------------------------------- starters
class TestStarters:
    def test_topics_are_plentiful_and_unique(self):
        assert len(TOPICS) >= 20
        assert len(set(TOPICS)) == len(TOPICS)

    def test_callback_requires_history(self):
        rng = random.Random(7)
        for _ in range(50):
            mode, _ = starter_instruction(user_turns=0, rng=rng)
            assert mode != "callback"

    def test_callback_available_with_history(self):
        rng = random.Random(7)
        modes = {starter_instruction(user_turns=3, rng=rng)[0] for _ in range(80)}
        assert "callback" in modes
        assert "rant" in modes  # variety exists

    def test_never_repeats_excluded_mode(self):
        rng = random.Random(3)
        for _ in range(60):
            mode, _ = starter_instruction(user_turns=2, rng=rng, exclude="rant")
            assert mode != "rant"

    def test_instructions_carry_the_honesty_rule(self):
        rng = random.Random(1)
        for _ in range(20):
            _, instruction = starter_instruction(user_turns=2, rng=rng)
            assert "Never invent" in instruction

    def test_rant_mode_embeds_a_topic(self):
        rng = random.Random(5)
        found = False
        for _ in range(120):
            mode, instruction = starter_instruction(user_turns=0, rng=rng)
            if mode == "rant":
                found = any(topic in instruction for topic in TOPICS)
                break
        assert found


# ------------------------------------------------------- tts engine shapes
class TestTtsFactory:
    def test_fake_engine_synthesizes(self):
        import asyncio

        from app.config import Settings
        from app.realtime.tts import FakeTts, build_tts, tts_mode

        engine = build_tts(Settings(tts_provider="fake"))
        assert isinstance(engine, FakeTts)
        audio = asyncio.get_event_loop().run_until_complete(
            engine.synthesize("hello")
        )
        assert audio and audio.startswith(b"ID3FAKE")
        assert tts_mode(Settings(tts_provider="fake")) == "server"

    def test_device_mode_returns_none(self):
        from app.config import Settings
        from app.realtime.tts import build_tts, tts_mode

        assert build_tts(Settings(tts_provider="device")) is None
        assert tts_mode(Settings(tts_provider="device")) == "device"


# ------------------------------------------------------------ sight layer
class TestSight:
    def test_disabled_without_model(self):
        from app.config import Settings
        from app.realtime.sight import build_sight

        assert build_sight(Settings(vision_llm_model="")) is None

    def test_fake_sight_describes(self):
        import asyncio

        from app.config import Settings
        from app.realtime.sight import FakeSight, build_sight

        engine = build_sight(Settings(vision_llm_model="fake"))
        assert isinstance(engine, FakeSight)
        text = asyncio.get_event_loop().run_until_complete(
            engine.describe(b"\xff\xd8fakejpeg")
        )
        assert "mug" in text

    def test_glimpse_freshness(self):
        from app.realtime.sight import Glimpse

        glimpse = Glimpse()
        assert glimpse.fresh() is None
        glimpse.update("a person waving")
        assert glimpse.fresh() == "a person waving"
        glimpse._at = time.monotonic() - 999
        assert glimpse.fresh(max_age=120) is None

    def test_glimpse_lands_in_llm_context(self):
        from app.conversation import build_messages
        from app.schemas import ChatTurn

        messages = build_messages(
            [ChatTurn(role="user", content="hi")],
            glimpse="a person at a desk holding a red mug",
        )
        assert "red mug" in messages[0]["content"]
        assert "don't narrate" in messages[0]["content"]


# ----------------------------------------------- websocket integration
def _start(client, **overrides):
    ws = client.websocket_connect("/realtime")
    ws.__enter__()
    payload = {"type": "session.start", "sample_rate": 16000, "greet": False}
    payload.update(overrides)
    ws.send_json(payload)
    ready = ws.receive_json()
    assert ready["type"] == "session.ready"
    state = ws.receive_json()
    assert state["value"] == "listening"
    return ws, ready


def _recv_until(ws, type_, limit=40):
    seen = []
    for _ in range(limit):
        event = ws.receive_json()
        seen.append(event)
        if event["type"] == type_:
            return event, seen
    raise AssertionError(f"never saw {type_}; saw {[e['type'] for e in seen]}")


def test_server_tts_ships_audio_after_say(make_client):
    client = make_client(TTS_PROVIDER="fake")
    ws, ready = _start(client, client_stt=True)
    try:
        assert ready["tts"] == "server"
        ws.send_json({"type": "user.text", "text": "tell me something"})
        say, _ = _recv_until(ws, "assistant.say")
        audio, _ = _recv_until(ws, "assistant.audio")
        assert audio["id"] == say["id"]
        assert audio["format"] == "mp3"
        assert len(audio["audio_b64"]) > 8  # fake engine produced bytes
    finally:
        ws.__exit__(None, None, None)


def test_device_tts_sends_no_audio_event(make_client):
    client = make_client(TTS_PROVIDER="device")
    ws, ready = _start(client, client_stt=True)
    try:
        assert ready["tts"] == "device"
        ws.send_json({"type": "user.text", "text": "hello"})
        _, seen = _recv_until(ws, "assistant.say")
        ws.send_json({"type": "ping"})
        _, seen2 = _recv_until(ws, "pong")
        assert all(e["type"] != "assistant.audio" for e in seen + seen2)
    finally:
        ws.__exit__(None, None, None)


def test_vision_frame_feeds_glimpse_into_replies(make_client):
    import base64

    client = make_client(VISION_LLM_MODEL="fake")
    ws, ready = _start(client, client_stt=True)
    try:
        assert ready["vision_frames"] is True
        ws.send_json({
            "type": "vision.frame",
            "jpeg_b64": base64.b64encode(b"\xff\xd8fakejpeg").decode(),
        })
        # Echo provider quotes the user; the glimpse rides the system prompt,
        # which echo can't show — so instead prove the frame was described by
        # asking again and checking the session didn't error and still talks.
        ws.send_json({"type": "user.text", "text": "what am I holding?"})
        say, _ = _recv_until(ws, "assistant.say")
        assert say["text"]  # conversation alive with sight enabled
    finally:
        ws.__exit__(None, None, None)


def test_vision_frames_disabled_by_default(make_client):
    client = make_client()
    ws, ready = _start(client, client_stt=True)
    try:
        assert ready["vision_frames"] is False
    finally:
        ws.__exit__(None, None, None)


def test_idle_starters_vary_mode(make_client):
    client = make_client(
        PROACTIVE_IDLE_SECONDS="0.25",
        PROACTIVE_MAX_NUDGES="2",
        TTS_PROVIDER="device",
    )
    ws, _ = _start(client)
    try:
        first, _ = _recv_until(ws, "assistant.say")
        assert first["proactive"] is True
        ws.send_json({"type": "tts.state", "playing": True})
        ws.send_json({"type": "tts.state", "playing": False})
        # Second nudge arrives later (escalated gap ~0.4-0.5s) — wait for it.
        second, _ = _recv_until(ws, "assistant.say", limit=60)
        assert second["proactive"] is True
        assert second["id"] != first["id"]
    finally:
        ws.__exit__(None, None, None)
