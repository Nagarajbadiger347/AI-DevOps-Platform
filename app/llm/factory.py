"""LLMFactory — returns the best available provider with automatic fallback.

Priority order (configurable via LLM_PROVIDER env var):
  claude  →  openai  →  (error)

Usage:
    from app.llm.factory import LLMFactory
    llm = LLMFactory.get()
    response = llm.complete("Summarise this incident …")
"""
from __future__ import annotations

import os
from app.llm.base import BaseLLM
from app.core.logging import get_logger

logger = get_logger(__name__)

# Preferred provider name — can be overridden per-call
_DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "claude").lower()


def _build_chain(preferred: str) -> list[BaseLLM]:
    """Import providers lazily to avoid hard failures at import time."""
    from app.llm.claude import ClaudeProvider
    from app.llm.openai import OpenAIProvider

    providers: dict[str, BaseLLM] = {
        "claude": ClaudeProvider(),
        "openai": OpenAIProvider(),
    }
    # Put preferred first, then the rest in fallback order
    order = [preferred] + [k for k in providers if k != preferred]
    return [providers[k] for k in order if k in providers]


class LLMFactory:
    """Stateless factory — call get() anywhere in the codebase."""

    @staticmethod
    def get(preferred: str | None = None) -> BaseLLM:
        """Return the first available provider in the fallback chain.

        Args:
            preferred: override the default provider for this call only
                       (e.g. "openai" to force GPT for a specific agent).

        Raises:
            RuntimeError: if no provider is configured.
        """
        target = (preferred or _DEFAULT_PROVIDER).lower()
        chain  = _build_chain(target)

        for provider in chain:
            if provider.is_available():
                name = provider.__class__.__name__
                logger.info("llm_provider_selected", provider=name, requested=target)
                return provider

        raise RuntimeError(
            "No LLM provider available.\n"
            "Configure at least one of:\n"
            "  • ANTHROPIC_API_KEY  (Claude)\n"
            "  • OPENAI_API_KEY     (OpenAI)\n"
            "  • GROQ_API_KEY       (Groq/Llama via Claude provider)\n"
            "  • Ollama running locally at OLLAMA_HOST"
        )
