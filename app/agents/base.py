"""BaseAgent — every agent in the multi-agent system inherits from this.

Production additions over the original:
  - _call_llm()  : wraps LLMFactory with retry + exponential backoff
  - _parse_json(): robust JSON extraction from LLM text (unchanged)
  - _log/_warn   : structured logging helpers (unchanged)

Subclasses that want managed retry just call self._call_llm(prompt, system)
instead of calling the LLM factory directly. Existing agents that implement
their own run() continue to work without modification.
"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

# Exponential backoff delays for LLM call retries: 1s, 3s, 8s
_LLM_RETRY_DELAYS: tuple[float, ...] = (1.0, 3.0, 8.0)
_LLM_MAX_RETRIES = 3


class BaseAgent(ABC):
    """Agents are pure decision/collection units.

    Contract:
      - Input:  shared PipelineState dict
      - Output: updated PipelineState dict (or partial context dict merged by the graph)
      - NO direct calls to other agents
      - NO mutation of infrastructure (collection agents only read)
    """

    # Subclasses may override for a fixed system prompt
    SYSTEM_PROMPT: ClassVar[str] = ""

    @abstractmethod
    def run(self, state: dict) -> dict:
        """Execute the agent's responsibility and return updated state."""
        ...

    # ── LLM helper with retry + backoff ──────────────────────────────────────

    def _call_llm(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1500,
        retries: int = _LLM_MAX_RETRIES,
    ) -> str:
        """
        Call the LLM via LLMFactory with automatic retry and exponential backoff.

        Returns the raw response text.
        Raises RuntimeError if all retries are exhausted.
        """
        system_prompt = system or self.SYSTEM_PROMPT
        last_exc: Optional[Exception] = None

        for attempt in range(retries):
            try:
                from app.llm.factory import LLMFactory
                llm = LLMFactory.get()
                response = llm.complete(prompt, system=system_prompt, max_tokens=max_tokens)
                if attempt > 0:
                    logger.info("llm_retry_succeeded", extra={
                        "agent":   type(self).__name__,
                        "attempt": attempt,
                    })
                return response.content

            except Exception as exc:
                last_exc = exc
                logger.warning("llm_call_failed", extra={
                    "agent":   type(self).__name__,
                    "attempt": attempt,
                    "error":   str(exc),
                })
                if attempt < retries - 1:
                    delay = _LLM_RETRY_DELAYS[min(attempt, len(_LLM_RETRY_DELAYS) - 1)]
                    logger.info("llm_retry_backoff", extra={
                        "agent":   type(self).__name__,
                        "delay_s": delay,
                        "attempt": attempt + 1,
                    })
                    time.sleep(delay)

        raise RuntimeError(
            f"{type(self).__name__}: LLM call failed after {retries} attempts. "
            f"Last error: {last_exc}"
        )

    # ── JSON parsing ─────────────────────────────────────────────────────────

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Robustly extract and parse JSON from an LLM response."""
        text = text.strip()

        # Try fenced code block first
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
                "raw_snippet": text[:200],
            })
            return {}

    # ── Logging helpers ───────────────────────────────────────────────────────

    def _log(self, event: str, **kwargs: Any) -> None:
        logger.info(event, extra={"agent": type(self).__name__, **kwargs})

    def _warn(self, event: str, **kwargs: Any) -> None:
        logger.warning(event, extra={"agent": type(self).__name__, **kwargs})
