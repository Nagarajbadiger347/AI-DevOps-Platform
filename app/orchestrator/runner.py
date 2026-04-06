"""Runner — public entry point for the LangGraph pipeline.

Usage (from FastAPI or monitoring loop):
    from app.orchestrator.runner import run_pipeline
    result = run_pipeline(
        incident_id="INC-001",
        description="API pods are crash-looping",
        auto_remediate=False,
        metadata={"user": "nagaraj", "role": "admin",
                  "k8s_cfg": {"namespace": "prod"}, "hours": 2},
    )
"""
from __future__ import annotations

import datetime

from app.orchestrator.graph import pipeline_graph
from app.orchestrator.state import PipelineState
from app.core.logging import get_logger, new_correlation_id

logger = get_logger(__name__)


def run_pipeline(
    incident_id: str,
    description: str,
    auto_remediate: bool = False,
    dry_run: bool = False,
    metadata: dict | None = None,
) -> dict:
    """Invoke the LangGraph pipeline synchronously and return the final state.

    Args:
        incident_id:     Unique ID for this incident run.
        description:     Human-readable description of the incident.
        auto_remediate:  If True and risk is low/medium, execute without approval.
        metadata:        Extra context — user, role, aws_cfg, k8s_cfg, hours, etc.

    Returns:
        Final PipelineState as a plain dict.
    """
    cid = new_correlation_id()
    logger.info("pipeline_started", extra={
        "incident_id": incident_id,
        "auto_remediate": auto_remediate,
        "correlation_id": cid,
    })

    initial_state: PipelineState = {
        "incident_id":    incident_id,
        "description":    description,
        "auto_remediate": auto_remediate,
        "dry_run":        dry_run,
        "metadata":       metadata or {},
        "errors":         [],
        "retry_count":    0,
        "status":         "running",
        "correlation_id": cid,
        "started_at":     datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    try:
        final_state: dict = pipeline_graph.invoke(initial_state)
    except Exception as exc:
        logger.error("pipeline_unhandled_error", extra={
            "incident_id": incident_id, "error": str(exc),
        })
        final_state = {**initial_state,
                       "status": "failed",
                       "errors": [str(exc)],
                       "completed_at": datetime.datetime.now(
                           datetime.timezone.utc).isoformat()}

    logger.info("pipeline_finished", extra={
        "incident_id": incident_id,
        "status": final_state.get("status"),
        "validation_passed": final_state.get("validation_passed"),
        "actions_executed": len(final_state.get("executed_actions", [])),
        "actions_blocked": len(final_state.get("blocked_actions", [])),
    })

    try:
        from app.integrations.vscode import write_output, notify
        status = final_state.get("status", "unknown")
        root_cause = final_state.get("root_cause") or ""
        executed = final_state.get("executed_actions", [])
        risk = final_state.get("risk_level", "")
        icon = "✅" if status == "completed" else ("⏳" if status == "awaiting_approval" else "❌")
        write_output(
            f"{icon} PIPELINE DONE  [{incident_id}]  status={status}  risk={risk}  "
            f"actions={len(executed)}  root_cause={root_cause[:80] if root_cause else 'n/a'}"
        )
        if status == "awaiting_approval":
            notify(f"[{incident_id}] Awaiting approval — {root_cause[:60]}", level="warning")
        elif status == "failed":
            notify(f"[{incident_id}] Pipeline failed — {description[:60]}", level="error")
    except Exception:
        pass

    return final_state
