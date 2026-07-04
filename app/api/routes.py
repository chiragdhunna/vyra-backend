"""REST surface: health, config, and the stateless chat endpoints.

The Flutter app's classic (turn-based) mode calls ``POST /chat``. The SSE
variant ``POST /chat/stream`` exists for lower perceived latency and for
non-mobile clients; both share the same conversation assembly and emotion
parsing, so every client gets identical Vyra behaviour.
"""

import json
import logging

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse

from .. import __version__
from ..config import get_settings
from ..conversation import build_messages
from ..personality import EMOTIONS, EmotionTagFilter, parse_emotion
from ..providers import ProviderError, get_llm_provider
from ..schemas import ChatRequest, ChatResponse, ConfigResponse, HealthResponse

logger = logging.getLogger("vyra.api")

router = APIRouter()


def require_api_key(x_vyra_key: str = Header(default="")) -> None:
    settings = get_settings()
    if settings.auth_enabled and x_vyra_key != settings.vyra_api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid X-Vyra-Key")


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(provider=get_settings().ai_provider)


@router.get("/config", response_model=ConfigResponse)
async def config(_: None = Depends(require_api_key)) -> ConfigResponse:
    settings = get_settings()
    provider = get_llm_provider()
    from ..realtime.stt import stt_mode  # local import avoids optional deps at boot

    return ConfigResponse(
        version=__version__,
        provider=provider.name,
        model=provider.model,
        stt=stt_mode(settings),
        emotions=list(EMOTIONS),
        auth_required=settings.auth_enabled,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, _: None = Depends(require_api_key)) -> ChatResponse:
    settings = get_settings()
    provider = get_llm_provider()
    messages = build_messages(
        request.messages,
        user_name=request.user_name,
        vision=request.vision,
        max_history_turns=settings.max_history_turns,
    )
    try:
        raw = await provider.chat(messages)
    except ProviderError as exc:
        logger.error("chat failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
    text, emotion = parse_emotion(raw)
    return ChatResponse(
        text=text or "…", emotion=emotion, provider=provider.name, model=provider.model
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, _: None = Depends(require_api_key)):
    """Server-sent events: ``delta`` events, then one final ``done`` event.

    The emotion tag never reaches the wire as display text — a streaming
    filter captures it and reports it in the ``done`` event.
    """
    settings = get_settings()
    provider = get_llm_provider()
    messages = build_messages(
        request.messages,
        user_name=request.user_name,
        vision=request.vision,
        max_history_turns=settings.max_history_turns,
    )

    async def gen():
        tag_filter = EmotionTagFilter()
        collected = []
        try:
            async for delta in provider.chat_stream(messages):
                for clean in tag_filter.feed(delta):
                    if clean:
                        collected.append(clean)
                        yield _sse({"type": "delta", "text": clean})
            leftover = tag_filter.flush()
            if leftover:
                # Anything left could still contain a complete tag.
                clean, emotion = parse_emotion(leftover)
                if clean:
                    collected.append(clean)
                    yield _sse({"type": "delta", "text": clean})
                if emotion != "neutral" and tag_filter.emotion is None:
                    tag_filter.emotion = emotion
        except ProviderError as exc:
            logger.error("chat stream failed: %s", exc)
            yield _sse({"type": "error", "message": str(exc)})
            return
        full = "".join(collected).strip()
        yield _sse(
            {
                "type": "done",
                "text": full or "…",
                "emotion": tag_filter.emotion or "neutral",
                "provider": provider.name,
                "model": provider.model,
            }
        )

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
