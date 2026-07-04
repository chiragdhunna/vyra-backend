"""Provider wire-format tests against httpx.MockTransport — no network."""

import json

import httpx
import pytest

from app.providers.echo import EchoProvider
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider
from app.providers.openai_compat import OpenAICompatProvider

MESSAGES = [
    {"role": "system", "content": "You are Vyra."},
    {"role": "user", "content": "hey"},
]


@pytest.fixture()
def anyio_backend():
    return "asyncio"


# --------------------------------------------------------------------- echo
@pytest.mark.anyio
async def test_echo_replies_with_emotion_tag():
    provider = EchoProvider()
    raw = await provider.chat(MESSAGES)
    assert "hey" in raw
    assert "[emotion:" in raw


@pytest.mark.anyio
async def test_echo_stream_matches_full_reply():
    provider = EchoProvider()
    chunks = [c async for c in provider.chat_stream(MESSAGES)]
    assert "".join(chunks)  # streams something, in >1 chunk for real content
    assert len(chunks) >= 2


# ------------------------------------------------------------------- ollama
def _ollama_transport():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode())
        captured["url"] = str(request.url)
        if captured["json"].get("stream"):
            lines = [
                json.dumps({"message": {"content": "Hel"}, "done": False}),
                json.dumps({"message": {"content": "lo!"}, "done": False}),
                json.dumps({"message": {"content": ""}, "done": True}),
            ]
            return httpx.Response(200, text="\n".join(lines))
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "Hello!"}}
        )

    return httpx.MockTransport(handler), captured


@pytest.mark.anyio
async def test_ollama_chat_payload_and_parse():
    transport, captured = _ollama_transport()
    client = httpx.AsyncClient(transport=transport, base_url="http://ollama.test")
    provider = OllamaProvider(
        host="http://ollama.test", model="llama3.1", client=client
    )
    text = await provider.chat(MESSAGES)
    assert text == "Hello!"
    assert captured["json"]["model"] == "llama3.1"
    assert captured["json"]["messages"][0]["role"] == "system"
    assert captured["url"].endswith("/api/chat")
    await provider.aclose()


@pytest.mark.anyio
async def test_ollama_stream_parses_ndjson():
    transport, _ = _ollama_transport()
    client = httpx.AsyncClient(transport=transport, base_url="http://ollama.test")
    provider = OllamaProvider(
        host="http://ollama.test", model="llama3.1", client=client
    )
    chunks = [c async for c in provider.chat_stream(MESSAGES)]
    assert "".join(chunks) == "Hello!"
    await provider.aclose()


# ------------------------------------------------------------------- gemini
def _gemini_transport():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode())
        captured["url"] = str(request.url)
        if ":streamGenerateContent" in str(request.url):
            chunk = {
                "candidates": [
                    {"content": {"parts": [{"text": "Hi there"}], "role": "model"}}
                ]
            }
            return httpx.Response(200, text=f"data: {json.dumps(chunk)}\n\n")
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": "Hi there"}], "role": "model"}}
                ]
            },
        )

    return httpx.MockTransport(handler), captured


@pytest.mark.anyio
async def test_gemini_maps_roles_and_system_instruction():
    transport, captured = _gemini_transport()
    client = httpx.AsyncClient(transport=transport, base_url="http://gemini.test")
    provider = GeminiProvider(api_key="k", model="gemini-2.5-flash", client=client)
    history = MESSAGES + [
        {"role": "assistant", "content": "yo"},
        {"role": "user", "content": "again"},
    ]
    text = await provider.chat(history)
    assert text == "Hi there"
    payload = captured["json"]
    assert payload["systemInstruction"]["parts"][0]["text"] == "You are Vyra."
    roles = [c["role"] for c in payload["contents"]]
    assert roles == ["user", "model", "user"]  # assistant → model
    assert "key=k" in captured["url"]
    await provider.aclose()


@pytest.mark.anyio
async def test_gemini_stream_sse():
    transport, _ = _gemini_transport()
    client = httpx.AsyncClient(transport=transport, base_url="http://gemini.test")
    provider = GeminiProvider(api_key="k", model="gemini-2.5-flash", client=client)
    chunks = [c async for c in provider.chat_stream(MESSAGES)]
    assert "".join(chunks) == "Hi there"
    await provider.aclose()


@pytest.mark.anyio
async def test_ollama_404_surfaces_model_hint():
    from app.providers.base import ProviderError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404, json={"error": "model 'llama3.1' not found, try pulling it first"}
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama.test"
    )
    provider = OllamaProvider(
        host="http://ollama.test", model="llama3.1", client=client
    )
    with pytest.raises(ProviderError) as excinfo:
        await provider.chat(MESSAGES)
    message = str(excinfo.value)
    assert "try pulling it first" in message
    assert "ollama pull llama3.1" in message
    await provider.aclose()


@pytest.mark.anyio
async def test_ollama_strips_think_blocks_and_rejects_empty():
    from app.providers.base import ProviderError

    responses = iter([
        {"message": {"role": "assistant",
                     "content": "<think>hmm reasoning</think>Hey there!"}},
        {"message": {"role": "assistant", "content": "",
                     "thinking": "endless pondering"}},
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(responses))

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama.test"
    )
    provider = OllamaProvider(
        host="http://ollama.test", model="qwen3", client=client
    )
    assert await provider.chat(MESSAGES) == "Hey there!"
    with pytest.raises(ProviderError) as excinfo:
        await provider.chat(MESSAGES)
    assert "empty reply" in str(excinfo.value)
    await provider.aclose()


def test_gemini_requires_key():
    import pytest as _pytest

    from app.providers.base import ProviderError

    with _pytest.raises(ProviderError):
        GeminiProvider(api_key="", model="gemini-2.5-flash")


# ------------------------------------------------------- openai-compatible
def _openai_transport():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode())
        captured["auth"] = request.headers.get("authorization", "")
        if captured["json"].get("stream"):
            chunks = [
                {"choices": [{"delta": {"content": "Hey"}}]},
                {"choices": [{"delta": {"content": " you"}}]},
            ]
            body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
            body += "data: [DONE]\n\n"
            return httpx.Response(200, text=body)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "Hey you"}}]}
        )

    return httpx.MockTransport(handler), captured


@pytest.mark.anyio
async def test_openai_chat_and_bearer_header():
    transport, captured = _openai_transport()
    client = httpx.AsyncClient(
        transport=transport,
        base_url="http://openai.test/v1",
        headers={"Authorization": "Bearer sk-test"},
    )
    provider = OpenAICompatProvider(api_key="sk-test", model="gpt-4o-mini", client=client)
    text = await provider.chat(MESSAGES)
    assert text == "Hey you"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["json"]["model"] == "gpt-4o-mini"
    await provider.aclose()


@pytest.mark.anyio
async def test_openai_stream_handles_done_marker():
    transport, _ = _openai_transport()
    client = httpx.AsyncClient(transport=transport, base_url="http://openai.test/v1")
    provider = OpenAICompatProvider(api_key="", model="gpt-4o-mini", client=client)
    chunks = [c async for c in provider.chat_stream(MESSAGES)]
    assert "".join(chunks) == "Hey you"
    await provider.aclose()
