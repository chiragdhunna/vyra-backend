#!/usr/bin/env bash
# One-command dev bootstrap: venv → deps → .env → server.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -q -r requirements.txt
# Server-side STT is optional; keep going if it fails (app falls back to
# on-device STT automatically).
pip install -q -r requirements-stt.txt || \
  echo "⚠️  faster-whisper install failed — realtime will use on-device STT."

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example — edit it to pick your AI provider."
fi

python -m app.main
