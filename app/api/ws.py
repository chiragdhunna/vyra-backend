"""Websocket endpoint: wires a connection to a :class:`RealtimeSession`."""

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import get_settings
from ..providers import get_llm_provider
from ..realtime.session import RealtimeSession
from ..realtime.stt import build_stt
from ..schemas import VisionContext

logger = logging.getLogger("vyra.ws")

ws_router = APIRouter()


@ws_router.websocket("/realtime")
async def realtime(ws: WebSocket) -> None:
    settings = get_settings()

    # Auth: ?key=... or X-Vyra-Key header, only when a key is configured.
    if settings.auth_enabled:
        supplied = ws.query_params.get("key", "") or ws.headers.get("x-vyra-key", "")
        if supplied != settings.vyra_api_key:
            await ws.close(code=4401, reason="invalid key")
            return

    await ws.accept()

    # First frame must be session.start (tolerate a leading ping).
    start = None
    try:
        while start is None:
            first = await ws.receive_json()
            if first.get("type") == "session.start":
                start = first
            elif first.get("type") == "ping":
                await ws.send_json({"type": "pong"})
            else:
                await ws.close(code=4400, reason="expected session.start")
                return
    except WebSocketDisconnect:
        return
    except Exception:
        await ws.close(code=4400, reason="expected session.start JSON")
        return

    session = RealtimeSession(
        send=ws.send_json,
        settings=settings,
        provider=get_llm_provider(),
        stt=build_stt(settings),
        user_name=start.get("user_name") or None,
        sample_rate=int(start.get("sample_rate") or 16000),
        greet=bool(start.get("greet", True)),
        client_stt=bool(start.get("client_stt", False)),
    )
    await session.start()

    try:
        while True:
            frame = await ws.receive()
            if frame.get("type") == "websocket.disconnect":
                break
            if frame.get("bytes") is not None:
                await session.on_audio(frame["bytes"])
                continue
            text = frame.get("text")
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "user.text":
                await session.on_user_text(str(event.get("text", "")))
            elif kind == "vision.state":
                await session.on_vision(
                    VisionContext(
                        present=bool(event.get("present", False)),
                        smiling=bool(event.get("smiling", False)),
                    )
                )
            elif kind == "tts.state":
                await session.on_tts_state(bool(event.get("playing", False)))
            elif kind == "mic.state":
                await session.on_mic_state(bool(event.get("muted", False)))
            elif kind == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - log and drop the connection
        logger.error("realtime session error: %s", exc, exc_info=True)
    finally:
        await session.close()
