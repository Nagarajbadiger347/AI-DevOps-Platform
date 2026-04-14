"""BaseAgent — every agent in the multi-agent pipeline inherits from this.

Contract:
  - Input:  shared PipelineState dict
  - Output: updated state dict (or partial context dict merged by the graph)
  - Agents are DECISION ONLY — they read state and return structured diffs
  - Agents MUST NOT call integrations directly
  - Agents MUST NOT mutate infrastructure
  - All integration calls go through the Executor

Provides:
  - _call_llm()          : LLMFactory wrapper with retry + backoff + LLM cache
  - _parse_json()        : robust JSON extraction from LLM text
  - _validate()          : assert required fields are present in a response
  - _parse_and_validate(): parse + validate in one call
  - _log/_warn/_error    : structured logging helpers with trace_id pre-filled
"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional

from app.core.logging import get_logger, trace_id_var, incident_id_var
from app.core.llm_cache import llm_cache, LLMCache

logger = get_logger(__name__)

# Exponential backoff delays for LLM retries (seconds): attempt 0, 1, 2
_LLM_RETRY_DELAYS: tuple[float, ...] = (1.0, 3.0, 8.0)
_LLM_MAX_RETRIES = 3

# Agent subclasses that should NEVER cache their LLM responses.
# Decision and Validator agents depend on fresh infra state — caching is unsafe.
_NO_CACHE_AGENTS = frozenset({"DecisionAgent", "ValidatorAgent"})


class AgentError(Exception):
    """Raised when an agent fails to produce a valid response after all retries."""


class BaseAgent(ABC):
    """Abstract base for all pipeline agents.

    Subclasses implement run(state) and optionally override:
      - SYSTEM_PROMPT   : str            — fixed system prompt for LLM calls
      - REQUIRED_FIELDS : tuple[str,...] — validated in every LLM response
      - USE_LLM_CACHE   : bool           — set False to bypass cache (default True)
    """

    SYSTEM_PROMPT:   ClassVar[str]           = ""
    REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = ()
    USE_LLM_CACHE:   ClassVar[bool]          = True

    @abstractmethod
    def run(self, state: dict) -> dict:
        """Execute the agent's responsibility and return updated state or partial dict."""
        ...

    # ── LLM call with cache + retry + backoff ─────────────────────────────────

    def _call_llm(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1500,
        retries: int = _LLM_MAX_RETRIES,
        temperature: float = 0.2,
        cache: bool = True,
    ) -> str:
        """Call LLMFactory with cache lookup, automatic retry, and exponential backoff.

        Cache behaviour:
          - Enabled when cache=True AND self.USE_LLM_CACHE is True AND
            the agent class is not in _NO_CACHE_AGENTS.
          - Cache key = SHA-256(system_prompt + user_prompt).
          - TTL = 5 minutes (configured in llm_cache module).
          - Cache is bypassed on retry attempts (only first attempt checks cache).

        Returns the raw response text.
        Raises AgentError if all retries are exhausted.
        """
        system_prompt = system or self.SYSTEM_PROMPT
        agent_name    = type(self).__name__
        trace_id      = trace_id_var.get("")
        incident_id   = incident_id_var.get("")

        use_cache = (
            cache
            and self.USE_LLM_CACHE
            and agent_name not in _NO_CACHE_AGENTS
            and temperature <= 0.3   # only cache deterministic-ish calls
        )

        # Cache lookup — only on first attempt
        cache_key: Optional[str] = None
        if use_cache:
            cache_key = LLMCache.make_key(system_prompt, prompt)
            cached    = llm_cache.get(cache_key)
            if cached is not None:
                self._log("llm_cache_hit", cache_key=cache_key[:12])
                return cached

        last_exc: Optional[Exception] = None

        for attempt in range(retries):
            try:
                from app.llm.factory import LLMFactory
                llm      = LLMFactory.get()
                response = llm.complete(
                    prompt,
                    system=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.content

                # Store in cache on success
                if use_cache and cache_key and content:
                    llm_cache.set(cache_key, content)

                if attempt > 0:
                    logger.info("llm_retry_succeeded", extra={
                        "agent":       agent_name,
                        "attempt":     attempt,
                        "trace_id":    trace_id,
                        "incident_id": incident_id,
                    })
                return content

            except Exception as exc:
                last_exc = exc
                logger.warning("llm_call_failed", extra={
                    "agent":       agent_name,
                    "attempt":     attempt,
                    "error":       str(exc),
                    "trace_id":    trace_id,
                    "incident_id": incident_id,
                })
                if attempt < retries - 1:
                    delay = _LLM_RETRY_DELAYS[min(attempt, len(_LLM_RETRY_DELAYS) - 1)]
                    logger.info("llm_retry_backoff", extra={
                        "agent":   agent_name,
                        "delay_s": delay,
                        "next":    attempt + 1,
                    })
                    time.sleep(delay)

        raise AgentError(
            f"{agent_name}: LLM call failed after {retries} attempts. "
            f"Last error: {last_exc}"
        )

    # ── JSON parsing ──────────────────────────────────────────────────────────

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Robustly extract and parse a JSON object from LLM output.

        Handles raw JSON, ```json fenced blocks, and mixed text with embedded JSON.
        Returns {} on parse failure — never raises.
        """
        text = text.strip()

        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            candidate = fence.group(1).strip()
        else:
            first = text.find("{")
            last  = text.rfind("}")
            candidate = text[first: last + 1] if first != -1 and last > first else text

        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            logger.warning("agent_json_parse_failed", extra={
                "agent":       type(self).__name__,
                "error":       str(exc),
                "raw_snippet": text[:300],
            })
            return {}

    # ── Response validation ───────────────────────────────────────────────────

    def _validate(self, data: dict, required: tuple[str, ...] | None = None) -> list[str]:
        """Return a list of missing required field names.

        Uses REQUIRED_FIELDS class var if ``required`` is not provided.
        """
        check = required if required is not None else self.REQUIRED_FIELDS
        return [f for f in check if f not in data or data[f] is None]

    def _parse_and_validate(
        self,
        text: str,
        required: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Parse JSON and validate required fields in one call.

        Logs a warning for missing fields but still returns the partial dict
        so the pipeline can degrade gracefully rather than hard-fail.
        """
        data    = self._parse_json(text)
        missing = self._validate(data, required)
        if missing:
            logger.warning("agent_response_missing_fields", extra={
                "agent":   type(self).__name__,
                "missing": missing,
                "snippet": text[:200],
            })
        return data

    # ── Structured logging helpers ────────────────────────────────────────────

    def _log(self, event: str, **kwargs: Any) -> None:
        logger.info(event, extra={
            "agent":       type(self).__name__,
            "trace_id":    trace_id_var.get(""),
            "incident_id": incident_id_var.get(""),
            **kwargs,
        })

    def _warn(self, event: str, **kwargs: Any) -> None:
        logger.warning(event, extra={
            "agent":       type(self).__name__,
            "trace_id":    trace_id_var.get(""),
            "incident_id": incident_id_var.get(""),
            **kwargs,
        })

    def _error(self, event: str, **kwargs: Any) -> None:
        logger.error(event, extra={
            "agent":       type(self).__name__,
            "trace_id":    trace_id_var.get(""),
            "incident_id": incident_id_var.get(""),
            **kwargs,
        })
