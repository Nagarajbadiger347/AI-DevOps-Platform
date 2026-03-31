"""AWSAgent — collects AWS observability data into pipeline state.

Read-only: no mutations. All infra changes go through the Executor.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.core.logging import get_logger

logger = get_logger(__name__)


class AWSAgent(BaseAgent):
    """Wraps app.integrations.aws_ops for the multi-agent pipeline."""

    def run(self, state: dict) -> dict:
        """Return a dict of AWS context (merged into state by the graph node)."""
        aws_cfg = state.get("metadata", {}).get("aws_cfg") or {}
        hours   = state.get("metadata", {}).get("hours", 2)

        if not aws_cfg or not aws_cfg.get("resource_type"):
            self._warn("aws_agent_skipped", reason="no aws_cfg in metadata")
            return {"_data_available": False, "_reason": "no aws_cfg provided"}

        try:
            from app.integrations.aws_ops import collect_diagnosis_context
            result = collect_diagnosis_context(
                resource_type=aws_cfg.get("resource_type", ""),
                resource_id=aws_cfg.get("resource_id", ""),
                log_group=aws_cfg.get("log_group", ""),
                hours=hours,
            )
            if isinstance(result, dict) and result.get("error"):
                return {"_data_available": False, "_reason": result["error"]}
            self._log("aws_context_collected",
                      incident_id=state.get("incident_id", ""),
                      resource_type=aws_cfg.get("resource_type"))
            return {"_data_available": True, **result}
        except Exception as exc:
            self._warn("aws_agent_error", error=str(exc))
            return {"_data_available": False, "_reason": str(exc)}
