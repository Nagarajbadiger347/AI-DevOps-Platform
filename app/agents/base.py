"""BaseAgent — every agent in the multi-agent system inherits from this."""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Agents are pure decision/collection units.

    Contract:
      - Input:  shared PipelineState dict
      - Output: updated PipelineState dict (or a partial context dict for
                collection agents that are merged by the graph node)
      - NO direct calls to other agents
      - NO mutation of infrastructure (collection agents only read)
    """

    @abstractmethod
    def run(self, state: dict) -> dict:
        """Execute the agent's responsibility and return updated state."""
        ...

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Robustly extract and parse JSON from an LLM response."""
        text = text.strip()

        # Try fenced code block first
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            candidate = fence.group(1).strip()
        else:
            # Fall back to first { … last }
            first = text.find("{")
            last  = text.rfind("}")
            candidate = text[first: last + 1] if first != -1 and last > first else text

        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            logger.warning(
                "agent_json_parse_failed",
                agent=self.__class__.__name__,
                error=str(exc),
                raw_snippet=text[:200],
            )
            return {}

    def _log(self, event: str, **kwargs: Any) -> None:
        logger.info(event, agent=self.__class__.__name__, **kwargs)

    def _warn(self, event: str, **kwargs: Any) -> None:
        logger.warning(event, agent=self.__class__.__name__, **kwargs)
