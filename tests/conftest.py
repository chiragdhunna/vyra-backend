"""Shared fixtures: env manipulation + a TestClient wired to fresh caches."""

import pytest
from fastapi.testclient import TestClient

from app.config import reset_settings_cache
from app.main import create_app
from app.providers import reset_provider_cache


@pytest.fixture()
def make_client(monkeypatch):
    """Returns a factory: make_client(ENV_VAR=value, ...) -> TestClient.

    Defaults to the echo provider and fake STT so no network, model or key
    is ever needed. Fast VAD/companion timings keep websocket tests snappy.
    """

    created = []

    def _make(**env) -> TestClient:
        defaults = {
            "AI_PROVIDER": "echo",
            "STT_PROVIDER": "fake",
            "GREET_ON_CONNECT": "true",
            "GREETING_DELAY_SECONDS": "0.15",
            "PROACTIVE_IDLE_SECONDS": "0.6",
            "PROACTIVE_MAX_NUDGES": "1",
            "VAD_FRAME_MS": "20",
            "VAD_START_MS": "40",
            "VAD_END_SILENCE_MS": "100",
            "VAD_MIN_UTTERANCE_MS": "40",
            "VAD_PRE_ROLL_MS": "40",
            "BARGE_MIN_MS": "60",
            "SPEAK_TIMEOUT_SECONDS": "5",
            "VYRA_API_KEY": "",
        }
        defaults.update(env)
        for key, value in defaults.items():
            monkeypatch.setenv(key, str(value))
        reset_settings_cache()
        reset_provider_cache()
        client = TestClient(create_app())
        created.append(client)
        return client

    yield _make

    for client in created:
        client.close()
    reset_settings_cache()
    reset_provider_cache()
