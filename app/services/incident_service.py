"""
Incident service — owns the business logic for running, approving,
and completing incident pipelines.

Routes call this. This calls the orchestrator + integrations.
No HTTP concerns here.
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

from app.orchestrator.runner import run_pipeline
from app.memory.vector_db import store_incident

logger = logging.getLogger(__name__)


def run(
    incident_id: str,
    description: str,
    user: str,
    role: str,
    auto_remediate: bool = False,
    dry_run: bool = False,
    aws_cfg: Optional[dict] = None,
    k8s_cfg: Optional[dict] = None,
    hours: int = 2,
    slack_channel: str = "#incidents",
    llm_provider: str = "",
    metadata: Optional[dict] = None,
) -> dict:
    """
    Execute the full incident pipeline.
    Returns the pipeline result dict.
    """
    extra = dict(metadata or {})
    extra.update({
        "user":          user,
        "role":          role,
        "aws_cfg":       aws_cfg or {},
        "k8s_cfg":       k8s_cfg or {},
        "hours":         hours,
        "slack_channel": slack_channel,
    })
    if llm_provider:
        extra["llm_provider"] = llm_provider

    result = run_pipeline(
        incident_id    = incident_id,
        description    = description,
        auto_remediate = auto_remediate,
        dry_run        = dry_run,
        metadata       = extra,
    )

    _persist(result, description)
    return result


def _persist(result: dict, description: str) -> None:
    """Store pipeline result to long-term memory. Best-effort."""
    inc_id = result.get("incident_id", "")
    if not inc_id:
        return
    try:
        plan = result.get("plan") or {}
        store_incident({
            "incident_id": inc_id,
            "description": description,
            "severity":    plan.get("risk", "medium"),
            "root_cause":  plan.get("root_cause", ""),
            "status":      result.get("status", "completed"),
            "confidence":  float(plan.get("confidence", 0.0)),
            "created_at":  datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
    except Exception as exc:
        logger.warning("incident_persist_failed incident=%s error=%s", inc_id, exc)


def handle_approval_flow(result: dict, slack_channel: str) -> dict:
    """
    Create approval token, persist state, notify via Slack + email.
    Extracted from route handler so it's testable independently.
    """
    from app.api.deps import _PENDING_PIPELINE_STATES

    cid = result.get("correlation_id")
    if not cid:
        return result

    _PENDING_PIPELINE_STATES[cid] = result

    try:
        from app.incident.approval import create_approval_request, post_approval_to_slack
        plan = result.get("plan") or {}
        approval = create_approval_request(
            incident_id  = result.get("incident_id", "unknown"),
            actions      = plan.get("actions", []),
            plan         = plan.get("summary") or result.get("approval_reason") or "AI-generated plan",
            risk_score   = float(result.get("risk_score") or plan.get("risk_score") or 0.5),
            cost_report  = result.get("cost_report"),
            requested_by = (result.get("metadata") or {}).get("user", "pipeline"),
        )
        _PENDING_PIPELINE_STATES[approval.correlation_id] = result
        _PENDING_PIPELINE_STATES.pop(cid, None)
        result["correlation_id"] = approval.correlation_id

        _persist_approval(result, plan, approval.correlation_id)

        if slack_channel:
            try:
                post_approval_to_slack(approval, slack_channel)
            except Exception as exc:
                logger.warning("approval_slack_failed channel=%s error=%s", slack_channel, exc)

        _email_approval(result, plan)

    except Exception as exc:
        logger.error("approval_flow_failed incident=%s error=%s",
                     result.get("incident_id"), exc)

    return result


def _persist_approval(result: dict, plan: dict, correlation_id: str) -> None:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        store_incident({
            "id":          result.get("incident_id", "unknown"),
            "type":        "pipeline_v2",
            "source":      "langgraph_orchestrator",
            "created_at":  now,
            "description": result.get("description", ""),
            "risk":        plan.get("risk", ""),
            "status":      "awaiting_approval",
            "confidence":  float(plan.get("confidence", 0.5)),
            "payload": {
                "description":       result.get("description", ""),
                "root_cause":        plan.get("root_cause", ""),
                "summary":           plan.get("summary", ""),
                "risk":              plan.get("risk", ""),
                "confidence":        float(plan.get("confidence", 0.5)),
                "actions_executed":  0,
                "validation_passed": False,
                "status":            "awaiting_approval",
                "created_at":        now,
                "correlation_id":    correlation_id,
            },
        })
    except Exception as exc:
        logger.warning("approval_persist_failed error=%s", exc)


def _email_approval(result: dict, plan: dict) -> None:
    try:
        from app.integrations.email import send_approval_required
        send_approval_required(
            incident_id     = result.get("incident_id", "unknown"),
            description     = result.get("description", ""),
            risk            = plan.get("risk", "unknown"),
            confidence      = float(plan.get("confidence", 0)),
            actions         = plan.get("actions", []),
            approval_reason = result.get("approval_reason", ""),
        )
    except Exception as exc:
        logger.warning("approval_email_failed incident=%s error=%s",
                       result.get("incident_id"), exc)


def send_completion_notification(result: dict) -> None:
    """Notify on pipeline completion. Best-effort."""
    try:
        from app.integrations.email import send_incident_completed
        plan = result.get("plan") or {}
        send_incident_completed(
            incident_id       = result.get("incident_id", "unknown"),
            description       = result.get("description", ""),
            risk              = plan.get("risk", "unknown"),
            status            = result.get("status", "completed"),
            root_cause        = plan.get("root_cause", ""),
            summary           = plan.get("summary", ""),
            actions_executed  = len(result.get("executed_actions", [])),
            validation_passed = bool(result.get("validation_passed", False)),
        )
    except Exception as exc:
        logger.warning("completion_email_failed incident=%s error=%s",
                       result.get("incident_id"), exc)
