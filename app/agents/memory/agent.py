"""MemoryAgent — reads similar past incidents before planning, stores outcome after.

Performance:
  - retrieve_similar() is called ONCE in collect_context node and the result
    stored in state["similar_incidents"]. The planner reads from state, not
    from ChromaDB directly, so there is no second vector DB query per pipeline run.
  - store() only writes high-confidence, non-junk results to avoid polluting
    future similarity searches.
"""
from __future__ import annotations

import datetime as _dt
import json as _json
from typing import Optional

from app.agents.base import BaseAgent
from app.core.logging import get_logger

logger = get_logger(__name__)

# Only store incidents whose plan confidence meets this threshold
_MIN_STORE_CONFIDENCE = 0.6

_JUNK_PHRASES = frozenset({
    "planning failed", "rate limit", "rate_limit", "429",
    "error code", "tokens per day",
})


def _is_useful(item: dict) -> bool:
    payload = item.get("payload", "")
    if isinstance(payload, str):
        try:
            payload = _json.loads(payload)
        except Exception:
            pass
    if isinstance(payload, dict):
        rc   = (payload.get("root_cause") or "").lower()
        conf = float(payload.get("confidence", 1.0))
        if conf < _MIN_STORE_CONFIDENCE or not rc:
            return False
        if any(p in rc for p in _JUNK_PHRASES):
            return False
        desc = (payload.get("description") or "").strip()
        if not desc or desc in ("test", "test auth fix", "auth check"):
            return False
    elif isinstance(payload, str) and payload in ("{}", "test"):
        return False
    return True


class MemoryAgent(BaseAgent):
    """Wraps app.memory.vector_db for the multi-agent pipeline."""

    # Never cache memory agent LLM calls — it doesn't make LLM calls anyway
    USE_LLM_CACHE = False

    def run(self, state: dict) -> dict:
        """Store the completed incident to ChromaDB — only if high quality."""
        plan        = state.get("plan", {})
        incident_id = state.get("incident_id", "unknown")
        root_cause  = plan.get("root_cause", "")
        confidence  = float(plan.get("confidence", 0.0))

        is_junk = (
            confidence < _MIN_STORE_CONFIDENCE
            or not root_cause
            or any(p in root_cause.lower() for p in _JUNK_PHRASES)
        )
        if is_junk:
            self._warn("incident_not_stored_low_quality",
                       incident_id=incident_id, confidence=confidence)
            state["status"] = state.get("status") or "completed"
            return state

        try:
            from app.memory.vector_db import store_incident
            now = _dt.datetime.now(_dt.timezone.utc).isoformat()
            store_incident({
                "id":          incident_id,
                "type":        "pipeline_v2",
                "source":      "langgraph_orchestrator",
                "created_at":  now,
                "description": state.get("description", ""),
                "risk":        plan.get("risk", ""),
                "status":      state.get("status", ""),
                "confidence":  confidence,
                "root_cause":  root_cause,
                "resolution_time": _resolution_secs(state),
                "severity":    state.get("severity", ""),
                "tenant_id":   state.get("tenant_id", ""),
                "payload": {
                    "description":       state.get("description", ""),
                    "root_cause":        root_cause,
                    "summary":           plan.get("summary", ""),
                    "risk":              plan.get("risk", ""),
                    "confidence":        confidence,
                    "actions_executed":  len(state.get("executed_actions", [])),
                    "validation_passed": state.get("validation_passed", False),
                    "status":            state.get("status", ""),
                    "created_at":        now,
                },
            })
            self._log("incident_stored", incident_id=incident_id, confidence=confidence)
        except Exception as exc:
            self._warn("memory_store_failed", incident_id=incident_id, error=str(exc))

        state["status"] = state.get("status") or "completed"
        return state

    @staticmethod
    def retrieve_similar(query: str, n: int = 5) -> list[dict]:
        """Fetch similar past incidents from ChromaDB.

        Called ONCE per pipeline run in collect_context — the result is stored
        in state["similar_incidents"] and reused by the planner. Do not call
        this again inside the planner or memory store nodes.
        """
        try:
            from app.memory.vector_db import search_similar_incidents
            # Fetch n+10 extra so we have headroom after quality filtering
            results = search_similar_incidents(query, n_results=n + 10)
            if results and isinstance(results[0], list):
                results = results[0]
            useful = [r for r in (results or []) if _is_useful(r)]
            return useful[:n]
        except Exception as exc:
            logger.warning("memory_retrieve_failed", extra={"error": str(exc)})
            return []


def _resolution_secs(state: dict) -> Optional[float]:
    """Compute resolution time in seconds from state timestamps."""
    try:
        import time as _t
        started   = state.get("started_at", "")
        completed = state.get("completed_at", "")
        if started and completed:
            from datetime import datetime, timezone
            fmt = "%Y-%m-%dT%H:%M:%S%z"
            s = datetime.fromisoformat(started.replace("Z", "+00:00"))
            c = datetime.fromisoformat(completed.replace("Z", "+00:00"))
            return round((c - s).total_seconds(), 1)
    except Exception:
        pass
    return None
