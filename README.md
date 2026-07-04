# 🧠 vyra-backend

The brain behind **[Vyra](https://github.com/chiragdhunna/vyra)** — the AI companion app.

Run this on the computer at your desk, put your phone on the same Wi-Fi, and Vyra stops being "an app with screens" and becomes *someone in the room*: she listens while you talk, waits for you to finish, answers out loud, lets you interrupt her mid-sentence, and sometimes starts the conversation herself.

```text
┌────────────────────── PHONE (Vyra app) ───────────────────────┐
│  🎤 mic ──────► streams PCM16 audio ─────────────┐            │
│  📷 camera ──► on-device ML Kit → tiny labels ───┤ WebSocket  │
│  🖥️ renders ONLY the animated face               │ /realtime  │
│      ◄── say {text, emotion} · interrupt · state ┘            │
└───────────────────────────────┬───────────────────────────────┘
                                │ ws://<this-machine>:8000
┌───────────────────────────────▼──────────────── vyra-backend ──┐
│  VAD + endpointing   →  "they finished talking"                │
│  Whisper STT         →  what they said        (optional)       │
│  Conversation brain  →  turn-taking · barge-in · proactivity   │
│  LLM  (the .env switch)                                        │
│    ├── ollama  → local, private, free                          │
│    ├── gemini  → Google cloud                                  │
│    ├── openai  → OpenAI or anything OpenAI-compatible          │
│    └── echo    → no model at all (demo/tests)                  │
│  Emotion parser      →  [emotion: X] → structured, incl. angry │
└────────────────────────────────────────────────────────────────┘
```

**Why a backend at all?** Three things a phone can't do alone: run a real local LLM (Ollama lives here), keep API keys out of the APK, and stream-listen *while speaking* so you can interrupt her like a real friend (on-device speech recognizers are strictly turn-based).

---

## 🚀 Quickstart

```bash
git clone https://github.com/chiragdhunna/vyra-backend.git
cd vyra-backend
./run.sh          # venv → deps → .env → server on 0.0.0.0:8000
```

or manually:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # core
pip install -r requirements-stt.txt      # optional: server-side Whisper STT
cp .env.example .env                     # then pick your AI_PROVIDER
python -m app.main
```

Check it's alive:

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/config
curl -X POST http://localhost:8000/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hey vyra!"}]}'
```

### Pick a brain (`.env`)

| Mode | Set | Needs |
| --- | --- | --- |
| **Local / private** | `AI_PROVIDER=ollama` | [Ollama](https://ollama.com) + `ollama pull llama3.1` |
| **Google Gemini** | `AI_PROVIDER=gemini` + `GEMINI_API_KEY` | [AI Studio key](https://aistudio.google.com/app/apikey) |
| **OpenAI** | `AI_PROVIDER=openai` + `OPENAI_API_KEY` | OpenAI key |
| **LM Studio / Groq / vLLM…** | `AI_PROVIDER=openai` + `OPENAI_BASE_URL` | any OpenAI-compatible server |
| **Nothing (demo)** | `AI_PROVIDER=echo` | — |

### Connect the phone

1. Both devices on the **same Wi-Fi**.
2. Find this machine's LAN IP (`ipconfig` on Windows, `ip a` / `ifconfig` on Linux/macOS) — e.g. `192.168.1.42`.
3. Allow TCP port `8000` through your OS firewall.
4. In the Vyra app's `.env`: `VYRA_BACKEND_URL=http://192.168.1.42:8000`.

Optionally set `VYRA_API_KEY` in both `.env` files to require a shared secret.

---

## 🔌 API

### REST

| Endpoint | What |
| --- | --- |
| `GET /healthz` | liveness (always open) |
| `GET /config` | active provider/model, STT mode, emotion list |
| `POST /chat` | stateless chat: `{messages:[{role,content}...], user_name?, vision?}` → `{text, emotion, provider, model}` |
| `POST /chat/stream` | same, as SSE: `delta` events then one `done` |
| `GET /docs` | interactive OpenAPI docs |

The emotion tag contract (`[emotion: X]`, X ∈ neutral, happy, excited, thinking, sad, surprised, caring, cry, **angry**) is parsed server-side — clients receive clean text + a structured emotion.

### WebSocket `/realtime`

Text frames = JSON events; binary frames = raw mic audio (PCM16 mono LE, 16 kHz). Full event tables live in [`app/realtime/protocol.py`](app/realtime/protocol.py).

A session in one glance:

```text
you → session.start {user_name, greet, client_stt}
      ← session.ready {provider, model, stt}
      ← state listening
you → (binary mic audio, continuously)
      ← user.final "so how was your day?"     (server Whisper heard you)
      ← state thinking → state speaking
      ← assistant.say {text, emotion: "happy"}   → phone speaks + animates
you → tts.state {playing:false}                  → floor returns to you
      ← state listening
   …you talk over her mid-reply…
      ← tts.interrupt                            → phone stops TTS instantly
```

Friend-like behaviours, all server-side: **endpointing** (she waits for your pause), **barge-in** (sustained voice while she speaks wins the floor; thresholds are stricter so her own speaker echo doesn't trigger it), **greeting** (she says hi first when you connect or when the camera sees you arrive), **re-engagement** (after a lull she checks in — capped by `PROACTIVE_MAX_NUDGES` so she never nags), and **vision context** (tiny `present`/`smiling` labels from on-device ML Kit — raw frames never leave the phone).

If `faster-whisper` isn't installed the socket still works: `session.ready` reports `stt: "client"` and the phone falls back to on-device turn-based STT, sending `user.text` events.

---

## 🗂️ Layout

```text
app/
├── main.py            FastAPI factory + uvicorn entrypoint
├── config.py          .env-driven settings (the provider switch lives here)
├── personality.py     Vyra's system prompt + emotion-tag parsing/streaming filter
├── conversation.py    prompt assembly, history windowing, vision context
├── schemas.py         REST/WS pydantic models
├── providers/         base contract + ollama · gemini · openai-compat · echo
├── api/
│   ├── routes.py      /healthz /config /chat /chat/stream
│   └── ws.py          /realtime websocket endpoint
└── realtime/
    ├── vad.py         energy VAD + endpointing + barge-in mode (pure python)
    ├── stt.py         faster-whisper engine (optional) / client fallback
    ├── session.py     the conversation state machine
    └── protocol.py    event reference
tests/                 pytest suite: unit + REST + full websocket voice loops
```

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite runs the entire realtime loop over a real websocket — synthesized PCM in, VAD endpointing, fake STT, echo LLM, barge-in, proactivity, auth — no network or models needed. CI runs it on every push.

## 🗺️ Roadmap

- [ ] Streaming server TTS option (Piper / edge-tts) with viseme timing for lip-sync
- [ ] Silero VAD as an optional upgrade over the energy detector
- [ ] Long-term memory across sessions
- [ ] Wake-word ("Vyra?") support
