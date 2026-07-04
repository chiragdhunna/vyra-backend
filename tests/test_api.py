import json


def test_healthz(make_client):
    client = make_client()
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_config_reports_provider_and_emotions(make_client):
    client = make_client()
    data = client.get("/config").json()
    assert data["provider"] == "echo"
    assert data["stt"] == "server"  # fake counts as server-side
    assert "angry" in data["emotions"]
    assert data["auth_required"] is False


def test_chat_round_trip_parses_emotion(make_client):
    client = make_client()
    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "I got the job!"}],
            "user_name": "Chirag",
            "vision": {"present": True, "smiling": True},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "I got the job!" in data["text"]
    assert "[emotion:" not in data["text"]  # tag stripped server-side
    assert data["emotion"] in {
        "neutral", "happy", "excited", "thinking", "sad",
        "surprised", "caring", "cry", "angry",
    }
    assert data["provider"] == "echo"


def test_chat_stream_sse_deltas_then_done(make_client):
    client = make_client()
    with client.stream(
        "POST",
        "/chat/stream",
        json={"messages": [{"role": "user", "content": "hello"}]},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = []
        for line in response.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):]))
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "done"
    assert "delta" in kinds
    done = events[-1]
    streamed = "".join(e["text"] for e in events if e["type"] == "delta")
    assert done["text"] == streamed.strip()
    assert "[emotion:" not in streamed  # filter suppressed the tag mid-stream
    assert done["emotion"] != ""


def test_auth_enforced_when_key_set(make_client):
    client = make_client(VYRA_API_KEY="secret")
    assert client.get("/config").status_code == 401
    assert (
        client.get("/config", headers={"X-Vyra-Key": "wrong"}).status_code == 401
    )
    ok = client.get("/config", headers={"X-Vyra-Key": "secret"})
    assert ok.status_code == 200
    assert ok.json()["auth_required"] is True
    # /healthz stays open for probes
    assert client.get("/healthz").status_code == 200


def test_chat_provider_error_maps_to_502(make_client):
    # Point the ollama provider at a closed port → ProviderError → 502.
    client = make_client(
        AI_PROVIDER="ollama",
        OLLAMA_HOST="http://127.0.0.1:9",  # discard port, guaranteed refused
        LLM_TIMEOUT_SECONDS="2",
    )
    response = client.post(
        "/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 502
    assert "Ollama" in response.json()["detail"]
