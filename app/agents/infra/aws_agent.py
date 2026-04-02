"""AWSAgent — collects AWS observability data into pipeline state.

Read-only: no mutations. All infra changes go through the Executor.

When no specific resource is provided, collects a general AWS health snapshot:
  - All firing CloudWatch alarms
  - Recent CloudTrail events
  - EC2 instance list with states
  - Active ECS / Lambda error summary (if available)
"""
from __future__ import annotations
import os

from app.agents.base import BaseAgent
from app.core.logging import get_logger

logger = get_logger(__name__)


def _aws_credentials_configured() -> bool:
    """Return True if AWS credentials are available in the environment."""
    return bool(
        os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")
    )


class AWSAgent(BaseAgent):
    """Wraps app.integrations.aws_ops for the multi-agent pipeline."""

    def run(self, state: dict) -> dict:
        """Return a dict of AWS context (merged into state by the graph node)."""
        aws_cfg = state.get("metadata", {}).get("aws_cfg") or {}
        hours   = state.get("metadata", {}).get("hours", 2)

        if not _aws_credentials_configured():
            self._warn("aws_agent_skipped", reason="AWS credentials not configured")
            return {"_data_available": False, "_reason": "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set"}

        # If a specific resource is requested, do a targeted collection
        if aws_cfg.get("resource_type"):
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
                          resource_type=aws_cfg.get("resource_type"),
                          mode="targeted")
                return {"_data_available": True, **result}
            except Exception as exc:
                self._warn("aws_agent_error", error=str(exc))
                return {"_data_available": False, "_reason": str(exc)}

        # No specific resource — collect a general AWS health snapshot
        try:
            from app.integrations.aws_ops import (
                list_cloudwatch_alarms,
                get_cloudtrail_events,
                list_ec2_instances,
            )
            ctx: dict = {
                "_data_available":  True,
                "_mode":            "general_snapshot",
                "region":           os.getenv("AWS_REGION", "us-east-1"),
                "hours":            hours,
            }

            # Firing alarms — always useful for any incident
            try:
                ctx["active_alarms"] = list_cloudwatch_alarms(state="ALARM")
            except Exception as exc:
                ctx["active_alarms"] = {"error": str(exc)}

            # Recent CloudTrail events — catch recent config changes
            try:
                ctx["cloudtrail_events"] = get_cloudtrail_events(hours=hours)
            except Exception as exc:
                ctx["cloudtrail_events"] = {"error": str(exc)}

            # EC2 instance list — shows stopped/running state
            try:
                ctx["ec2_instances"] = list_ec2_instances()
            except Exception as exc:
                ctx["ec2_instances"] = {"error": str(exc)}

            self._log("aws_context_collected",
                      incident_id=state.get("incident_id", ""),
                      mode="general_snapshot",
                      alarms=len(ctx.get("active_alarms") or []) if isinstance(ctx.get("active_alarms"), list) else 0)
            return ctx

        except Exception as exc:
            self._warn("aws_agent_error", error=str(exc))
            return {"_data_available": False, "_reason": str(exc)}
