"""LLMFactory — returns the best available provider with automatic fallback.

Priority order (configurable via LLM_PROVIDER env var):
  claude  →  openai  →  groq  →  ollama  →  (error)

Providers that return HTTP 429 (rate-limited) are automatically cooled
down and skipped for the duration specified in the error response.

Usage:
    from app.llm.factory import LLMFactory
    llm = LLMFactory.get()
    response = llm.complete("Summarise this incident …")
"""
from __future__ import annotations

import os
import re
import time
from app.llm.base import BaseLLM
from app.core.logging import get_logger

logger = get_logger(__name__)

# Preferred provider name — can be overridden per-call
_DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "claude").lower()

# Rate-limit cooldown: provider_key -> epoch time when usable again
_rate_limited_until: dict[str, float] = {}


def _parse_retry_seconds(error_str: str) -> int:
    """Extract retry-after seconds from a rate limit error message."""
    m = re.search(r"try again in (\d+)m(\d+)", error_str)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.search(r"try again in (\d+)s", error_str)
    if m:
        return int(m.group(1))
    return 180  # default 3-minute cooldown


def mark_rate_limited(provider_key: str, error_str: str = "") -> None:
    """Mark a provider as rate-limited so the factory skips it temporarily."""
    seconds = _parse_retry_seconds(error_str)
    _rate_limited_until[provider_key] = time.monotonic() + seconds
    logger.warning(
        "llm_rate_limited",
        extra={"provider": provider_key, "cooldown_seconds": seconds},
    )


def _is_rate_limited(provider_key: str) -> bool:
    until = _rate_limited_until.get(provider_key, 0)
    return time.monotonic() < until


def _build_chain(preferred: str) -> list[tuple[str, BaseLLM]]:
    """Import providers lazily to avoid hard failures at import time."""
    from app.llm.claude import ClaudeProvider
    from app.llm.openai import OpenAIProvider

    _alias = {"anthropic": "claude"}
    preferred = _alias.get(preferred, preferred)

    providers: dict[str, BaseLLM] = {
        "claude":  ClaudeProvider(),
        "openai":  OpenAIProvider(),
        "groq":    ClaudeProvider(force_provider="groq"),
        "ollama":  ClaudeProvider(force_provider="ollama"),
    }
    order = [preferred] + [k for k in providers if k != preferred]
    return [(k, providers[k]) for k in order if k in providers]


class LLMFactory:
    """Stateless factory — call get() anywhere in the codebase."""

    @staticmethod
    def get(preferred: str | None = None) -> BaseLLM:
        """Return the first available, non-rate-limited provider.

        Args:
            preferred: override the default provider for this call only.

        Raises:
            RuntimeError: if no provider is configured or all are rate-limited.
        """
        target = (preferred or os.getenv("LLM_PROVIDER", _DEFAULT_PROVIDER)).lower()
        chain  = _build_chain(target)

        for key, provider in chain:
            if _is_rate_limited(key):
                remaining = int(_rate_limited_until[key] - time.monotonic())
                logger.info(
                    "llm_provider_skipped_rate_limited",
                    extra={"provider": key, "retry_in_seconds": remaining},
                )
                continue
            if provider.is_available():
                logger.info("llm_provider_selected", extra={"provider": key, "requested": target})
                return provider

        # All providers unavailable — surface a helpful message
        cooled = [k for k, _ in chain if _is_rate_limited(k)]
        unconfigured = [k for k, p in chain if not _is_rate_limited(k) and not p.is_available()]
        parts = []
        if cooled:
            parts.append(f"Rate-limited (cooling down): {', '.join(cooled)}")
        if unconfigured:
            parts.append(f"Not configured (no API key): {', '.join(unconfigured)}")
        raise RuntimeError(
            "No LLM provider available.\n"
            + "\n".join(parts) + "\n"
            "Add ANTHROPIC_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY to your .env file."
        )
