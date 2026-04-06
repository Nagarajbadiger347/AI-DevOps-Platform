"""MemoryAgent — reads similar past incidents before planning, stores outcome after."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.core.logging import get_logger

logger = get_logger(__name__)


class MemoryAgent(BaseAgent):
    """Wraps app.memory.vector_db for the multi-agent pipeline."""

    def run(self, state: dict) -> dict:
        """Store the completed incident — only if it has a real root cause."""
        plan        = state.get("plan", {})
        incident_id = state.get("incident_id", "unknown")
        root_cause  = plan.get("root_cause", "")
        confidence  = float(plan.get("confidence", 0.0))

        # Skip storing failed / low-quality runs — they pollute future similarity searches
        junk_phrases = ("planning failed", "rate limit", "rate_limit", "429", "error code")
        is_junk = (
            confidence < 0.6  # only store high-confidence results
            or not root_cause
            or any(p in root_cause.lower() for p in junk_phrases)
        )
        if is_junk:
            self._warn("incident_not_stored_low_quality",
                       incident_id=incident_id, confidence=confidence)
            state["status"] = state.get("status") or "completed"
            return state

        try:
            from app.memory.vector_db import store_incident
            import datetime as _dt
            _now = _dt.datetime.now(_dt.timezone.utc).isoformat()
            store_incident({
                "id":          incident_id,
                "type":        "pipeline_v2",
                "source":      "langgraph_orchestrator",
                "created_at":  _now,
                "description": state.get("description", ""),
                "risk":        plan.get("risk", ""),
                "status":      state.get("status", ""),
                "confidence":  confidence,
                "payload": {
                    "description":       state.get("description", ""),
                    "root_cause":        root_cause,
                    "summary":           plan.get("summary", ""),
                    "risk":              plan.get("risk", ""),
                    "confidence":        confidence,
                    "actions_executed":  len(state.get("executed_actions", [])),
                    "validation_passed": state.get("validation_passed", False),
                    "status":            state.get("status", ""),
                    "created_at":        _now,
                },
            })
            self._log("incident_stored", incident_id=incident_id, confidence=confidence)
        except Exception as exc:
            self._warn("memory_store_failed", incident_id=incident_id, error=str(exc))

        state["status"] = state.get("status") or "completed"
        return state

    @staticmethod
    def retrieve_similar(query: str, n: int = 5) -> list[dict]:
        """Fetch similar past incidents — called by the graph's collect_context node.

        Filters out failed/junk entries before returning so they never
        influence the planner's analysis.
        """
        import json as _json

        junk_phrases = ("planning failed", "rate limit", "rate_limit", "429",
                        "error code", "tokens per day")

        def _is_useful(item: dict) -> bool:
            payload = item.get("payload", "")
            if isinstance(payload, str):
                try:
                    payload = _json.loads(payload)
                except Exception:
                    pass
            if isinstance(payload, dict):
                rc = (payload.get("root_cause") or "").lower()
                conf = float(payload.get("confidence", 1.0))
                # Apply the same 0.6 minimum confidence threshold used when storing
                if conf < 0.6 or not rc:
                    return False
                if any(p in rc for p in junk_phrases):
                    return False
                # Must have a real description
                desc = (payload.get("description") or "").strip()
                if not desc or desc in ("test", "test auth fix", "auth check"):
                    return False
            elif isinstance(payload, str) and payload in ("{}", "test"):
                return False
            return True

        try:
            from app.memory.vector_db import search_similar_incidents
            results = search_similar_incidents(query, n_results=n + 10)  # fetch more to filter
            if results and isinstance(results[0], list):
                results = results[0]
            useful = [r for r in (results or []) if _is_useful(r)]
            return useful[:n]
        except Exception as exc:
            logger.warning("memory_retrieve_failed", extra={"error": str(exc)})
            return []
