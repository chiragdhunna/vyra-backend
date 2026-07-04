"""Builds the configured LLM provider from settings (the `.env` switch)."""

import logging
from typing import Optional

from ..config import Settings, get_settings
from .base import LLMProvider, ProviderError
from .echo import EchoProvider
from .gemini import GeminiProvider
from .ollama import OllamaProvider
from .openai_compat import OpenAICompatProvider

logger = logging.getLogger("vyra.providers")

_provider: Optional[LLMProvider] = None


def build_provider(settings: Settings) -> LLMProvider:
    kind = settings.ai_provider.strip().lower()
    if kind == "ollama":
        return OllamaProvider(
            host=settings.ollama_host,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )
    if kind == "gemini":
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            base_url=settings.gemini_base_url,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )
    if kind == "openai":
        return OpenAICompatProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )
    if kind == "echo":
        return EchoProvider()
    raise ProviderError(
        f"Unknown AI_PROVIDER '{settings.ai_provider}'. "
        "Use one of: ollama, gemini, openai, echo."
    )


def get_llm_provider() -> LLMProvider:
    """Singleton accessor used by routes and realtime sessions."""
    global _provider
    if _provider is None:
        _provider = build_provider(get_settings())
        logger.info(
            "LLM provider ready: %s (%s)", _provider.name, _provider.model
        )
    return _provider


def reset_provider_cache() -> None:
    """Used by tests (and future hot-reload) to rebuild the provider."""
    global _provider
    _provider = None
