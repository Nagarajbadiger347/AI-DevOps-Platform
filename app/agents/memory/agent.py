"""MemoryAgent — reads similar past incidents before planning, stores outcome after."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.core.logging import get_logger

logger = get_logger(__name__)


class MemoryAgent(BaseAgent):
    """Wraps app.memory.vector_db for the multi-agent pipeline."""

    def run(self, state: dict) -> dict:
        """Store the completed incident and return state unchanged."""
        plan      = state.get("plan", {})
        incident_id = state.get("incident_id", "unknown")

        try:
            from app.memory.vector_db import store_incident
            store_incident({
                "id":     incident_id,
                "type":   "pipeline_v2",
                "source": "langgraph_orchestrator",
                "payload": {
                    "description":     state.get("description", ""),
                    "root_cause":      plan.get("root_cause", ""),
                    "summary":         plan.get("summary", ""),
                    "risk":            plan.get("risk", ""),
                    "confidence":      plan.get("confidence", 0.0),
                    "actions_executed": len(state.get("executed_actions", [])),
                    "validation_passed": state.get("validation_passed", False),
                    "status":          state.get("status", ""),
                },
            })
            self._log("incident_stored", incident_id=incident_id)
        except Exception as exc:
            # Memory failures must never block the pipeline
            self._warn("memory_store_failed", incident_id=incident_id, error=str(exc))

        state["status"] = state.get("status") or "completed"
        return state

    @staticmethod
    def retrieve_similar(query: str, n: int = 5) -> list[dict]:
        """Fetch similar past incidents — called by the graph's collect_context node."""
        try:
            from app.memory.vector_db import search_similar_incidents
            results = search_similar_incidents(query, n_results=n)
            # Flatten nested list returned by ChromaDB
            if results and isinstance(results[0], list):
                return results[0]
            return results or []
        except Exception as exc:
            logger.warning("memory_retrieve_failed", error=str(exc))
            return []
