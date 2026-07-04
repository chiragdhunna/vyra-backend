"""FastAPI application factory + uvicorn entrypoint.

Run it:

    python -m app.main            # reads host/port from .env
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .api.routes import router
from .api.ws import ws_router
from .config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="vyra-backend",
        version=__version__,
        description=(
            "The brain behind the Vyra companion app: swappable local "
            "(Ollama) or cloud LLMs, plus a realtime voice websocket with "
            "server-side turn-taking, barge-in and proactivity."
        ),
    )
    # The phone talks to us across the LAN; keep CORS permissive.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(ws_router)
    return app


app = create_app()


def run() -> None:  # pragma: no cover - thin wrapper
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)


if __name__ == "__main__":  # pragma: no cover
    run()
