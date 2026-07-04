"""Environment-driven configuration.

Everything the backend does is steered from `.env` (see `.env.example`).
The headline switch is ``AI_PROVIDER`` — flip it between ``ollama`` (local
model on this machine) and ``gemini`` / ``openai`` (cloud) without touching
code. ``openai`` is OpenAI-API-*compatible*: point ``OPENAI_BASE_URL`` at
LM Studio, Groq, Together, etc. and it works the same way.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Server ---
    host: str = "0.0.0.0"  # listen on the LAN so the phone can reach us
    port: int = 8000
    # Optional shared secret. When set, REST needs header `X-Vyra-Key` and the
    # websocket needs `?key=` (or the same header). Empty = open (home LAN).
    vyra_api_key: str = ""

    # --- LLM provider switch ---
    ai_provider: str = "ollama"  # ollama | gemini | openai | echo
    llm_temperature: float = 0.9
    llm_timeout_seconds: float = 120.0  # first token on CPU Ollama can be slow
    max_history_turns: int = 24  # user+assistant messages kept per request

    # Ollama (local model on this machine)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # Google Gemini (cloud)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # OpenAI-compatible (cloud, or LM Studio/Groq/etc. via base_url)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    # --- Realtime voice ---
    # whisper = transcribe streamed mic audio on this machine (needs
    # `pip install -r requirements-stt.txt`). client = the phone does STT and
    # sends text. Anything unavailable falls back to client gracefully.
    stt_provider: str = "whisper"  # whisper | client | fake (tests)
    stt_timeout_seconds: float = 20.0  # a wedged native call must never block the session
    whisper_model: str = "base.en"
    whisper_device: str = "cpu"  # cpu | cuda | auto (auto can HANG on broken CUDA installs)
    whisper_compute_type: str = "int8"

    # Voice-activity detection (energy-based; tuned for a phone on a desk).
    vad_frame_ms: int = 20
    vad_start_ms: int = 100  # sustained voice before we call it speech
    vad_end_silence_ms: int = 700  # pause length that ends the user's turn
    vad_min_utterance_ms: int = 300  # discard blips shorter than this
    vad_pre_roll_ms: int = 200  # audio kept from just before speech started
    vad_min_rms: float = 0.010  # absolute floor (0..1 of full scale)
    vad_noise_multiplier: float = 3.0  # speech must exceed noise floor by this
    # Barge-in (user talks over Vyra) needs to be stricter than normal speech
    # so she doesn't hear the phone's own speaker as an interruption.
    barge_min_ms: int = 300
    barge_noise_multiplier: float = 5.0

    # --- Her voice (server-side neural TTS) ---
    # edge = Microsoft Edge neural voices (free, no key, natural female
    # voices — night-and-day vs device TTS). Needs internet on THIS machine.
    # device = the phone's TTS engine (fully offline).
    tts_provider: str = "edge"  # edge | device | fake (tests)
    edge_tts_voice: str = "en-US-JennyNeural"  # warm female; try en-US-AriaNeural, en-GB-MaisieNeural
    edge_tts_rate: str = "+4%"
    edge_tts_pitch: str = "+18Hz"  # slightly brighter — she's a young woman
    tts_timeout_seconds: float = 12.0

    # --- Vision LLM (situational sight) ---
    # A local multimodal model (via Ollama) that periodically glimpses a
    # downscaled camera frame so she knows WHAT you're doing — not just that
    # you're present. e.g. `ollama pull moondream` (tiny) or llava.
    # Empty = disabled. Frames never leave your LAN.
    vision_llm_model: str = ""  # moondream | llava | ... | fake (tests)
    vision_frame_interval_seconds: float = 20.0

    # --- Vision awareness (Jarvis mode) ---
    # She reacts to what the camera senses: welcomes you back after you step
    # away, notices smiles, checks in when you look tired. Cooldowns keep her
    # observant, not creepy.
    vision_reactions: bool = True
    welcome_back_after_seconds: float = 45.0
    vision_react_cooldown_seconds: float = 90.0
    tired_eyes_threshold: float = 0.4
    tired_after_seconds: float = 15.0
    tired_react_cooldown_seconds: float = 600.0

    # --- Companion behaviour ---
    greet_on_connect: bool = True
    greeting_delay_seconds: float = 3.0
    proactive_idle_seconds: float = 75.0  # base; jittered and escalating
    proactive_max_nudges: int = 3
    speak_timeout_seconds: float = 30.0  # safety if the phone never reports TTS end

    @property
    def auth_enabled(self) -> bool:
        return bool(self.vyra_api_key)


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Used by tests to re-read settings after env changes."""
    get_settings.cache_clear()
