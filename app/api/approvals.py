"""
Approval workflow routes.
Paths: /approvals/*
"""
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import (
    require_developer, require_viewer, AuthContext,
    _PENDING_PIPELINE_STATES, _cache_result,
)

router = APIRouter(prefix="/approvals", tags=["approvals"])


class QuickActionRequest(BaseModel):
    incident_id: str
    action: dict
    risk_score: float = 0.5


class ApprovalDecision(BaseModel):
    approved_action_indices: List[int] = []
    reason: Optional[str] = None


@router.post("/action")
def create_quick_action_approval(req: QuickActionRequest, auth: AuthContext = Depends(require_developer)):
    """Create a single-action approval request from an incident result card."""
    try:
        from app.incident.approval import create_approval_request
        action_type = req.action.get("type", "unknown")
        plan_summary = req.action.get("description") or f"Execute {action_type} on {req.incident_id}"
        approval = create_approval_request(
            incident_id=req.incident_id,
            actions=[req.action],
            plan=plan_summary,
            risk_score=req.risk_score,
            cost_report=None,
            requested_by=auth.username,
        )
        _PENDING_PIPELINE_STATES[approval.correlation_id] = {
            "incident_id":    req.incident_id,
            "correlation_id": approval.correlation_id,
            "plan":           {"actions": [req.action], "risk": "medium", "confidence": 1.0, "summary": plan_summary},
            "metadata":       {"user": auth.username, "role": auth.role},
            "auto_remediate": True,
            "dry_run":        False,
            "errors":         [],
            "retry_count":    0,
            "status":         "awaiting_approval",
        }
        from app.core.audit import audit_log as _al
        _al(user=auth.username, action="approval_requested",
            params={"incident_id": req.incident_id, "action_type": action_type,
                    "risk_score": req.risk_score, "correlation_id": approval.correlation_id},
            result={"success": True}, source="approvals")
        return {"success": True, "correlation_id": approval.correlation_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/pending")
def list_pending_approvals_endpoint(auth: AuthContext = Depends(require_viewer)):
    try:
        from app.incident.approval import _pending_approvals, STATUS_PENDING, STATUS_APPROVED, cleanup_expired
        cleanup_expired()
        approvals = [a for a in _pending_approvals.values() if a.status in (STATUS_PENDING, STATUS_APPROVED)]
        return {"approvals": [vars(a) for a in approvals]}
    except Exception as e:
        return {"approvals": [], "error": str(e)}


@router.get("/history")
def list_approval_history_endpoint(auth: AuthContext = Depends(require_viewer)):
    try:
        from app.incident.approval import _pending_approvals
        return {"approvals": [vars(a) for a in _pending_approvals.values()]}
    except Exception as e:
        return {"approvals": [], "error": str(e)}


@router.delete("/{correlation_id}")
def delete_approval_endpoint(correlation_id: str, auth: AuthContext = Depends(require_developer)):
    from app.incident.approval import _pending_approvals, _save_approvals
    if correlation_id not in _pending_approvals:
        raise HTTPException(status_code=404, detail="Approval not found")
    del _pending_approvals[correlation_id]
    _PENDING_PIPELINE_STATES.pop(correlation_id, None)
    _save_approvals()
    return {"ok": True, "deleted": correlation_id}


@router.post("/{correlation_id}/approve")
def approve_actions_endpoint(correlation_id: str, req: ApprovalDecision, auth: AuthContext = Depends(require_developer)):
    try:
        from app.incident.approval import approve_actions
        result = approve_actions(correlation_id, req.approved_action_indices, auth.username)
        from app.core.audit import audit_log as _al
        _al(user=auth.username, action="approval_approved",
            params={"correlation_id": correlation_id,
                    "approved_indices": req.approved_action_indices,
                    "requested_by": getattr(result, "requested_by", "pipeline")},
            result={"success": True}, source="approvals")
        return {"success": True, "approval": vars(result)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{correlation_id}/reject")
def reject_approval_endpoint(correlation_id: str, req: ApprovalDecision, auth: AuthContext = Depends(require_developer)):
    try:
        from app.incident.approval import reject_approval
        result = reject_approval(correlation_id, req.reason or "Rejected by user", auth.username)

        # Update the incident result cache so the UI reflects the rejection immediately
        saved_state = _PENDING_PIPELINE_STATES.pop(correlation_id, None)
        if saved_state:
            incident_id = saved_state.get("incident_id", "")
            if incident_id:
                saved_state["status"]      = "rejected"
                saved_state["rejected_by"] = auth.username
                saved_state["reject_reason"] = req.reason or "Rejected by user"
                _cache_result(incident_id, saved_state)

        from app.core.audit import audit_log as _al
        _al(user=auth.username, action="approval_rejected",
            params={"correlation_id": correlation_id, "reason": req.reason or "Rejected by user",
                    "requested_by": getattr(result, "requested_by", "pipeline")},
            result={"success": True}, source="approvals")
        return {"success": True, "approval": vars(result)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{correlation_id}/resume")
def resume_approved_pipeline(
    correlation_id: str,
    auth: AuthContext = Depends(require_developer),
):
    """Resume an awaiting-approval pipeline after human approval."""
    from app.incident.approval import get_approval_request, STATUS_APPROVED
    from app.execution.executor import Executor
    from app.execution.validator import Validator
    from app.agents.memory.agent import MemoryAgent
    import datetime as _dt

    approval = get_approval_request(correlation_id)
    if not approval:
        raise HTTPException(status_code=404, detail=f"Approval {correlation_id} not found")
    if approval.status != STATUS_APPROVED:
        raise HTTPException(
            status_code=400,
            detail=f"Approval is in state '{approval.status}', not approved",
        )

    saved_state = _PENDING_PIPELINE_STATES.get(correlation_id)
    if not saved_state:
        saved_state = {
            "incident_id":    approval.incident_id,
            "correlation_id": correlation_id,
            "plan": {
                "actions":    approval.actions,
                "risk":       "medium",
                "confidence": 1.0,
                "summary":    approval.plan_summary,
            },
            "metadata":       {"user": approval.requested_by, "role": "admin"},
            "auto_remediate": True,
            "dry_run":        False,
            "errors":         [],
            "retry_count":    0,
            "status":         "awaiting_approval",
        }

    resume_state = dict(saved_state)
    resume_state["auto_remediate"]          = True
    resume_state["requires_human_approval"] = False
    resume_state["approval_reason"]         = f"Manually approved by {auth.username}"
    resume_state["status"]                  = "running"

    plan = resume_state.get("plan") or {}
    approved_actions = getattr(approval, "approved_actions", None)
    if approved_actions:
        resume_state["plan"] = {**plan, "actions": approved_actions}

    try:
        executor = Executor()
        resume_state = executor.run(resume_state)
        validator = Validator()
        resume_state = validator.run(resume_state)
        memory_agent = MemoryAgent()
        resume_state = memory_agent.run(resume_state)
        resume_state["status"]       = "completed"
        resume_state["completed_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        resume_state["resumed_by"]   = auth.username
        resume_state["approved_by"]  = auth.username
    except Exception as exc:
        resume_state["status"] = "failed"
        resume_state["errors"] = resume_state.get("errors", []) + [str(exc)]

    _PENDING_PIPELINE_STATES.pop(correlation_id, None)
    _cache_result(resume_state.get("incident_id", ""), resume_state)

    # Auto-create war room after successful resume
    if resume_state.get("status") == "completed":
        try:
            from app.incident.war_room_store import create as _create_war_room
            incident_id = resume_state.get("incident_id", "unknown")
            plan = resume_state.get("plan") or {}
            _create_war_room(
                incident_id    = incident_id,
                description    = resume_state.get("description", ""),
                pipeline_state = {
                    "root_cause":    plan.get("root_cause", ""),
                    "summary":       plan.get("summary", ""),
                    "severity":      resume_state.get("severity", "medium"),
                    "status":        "active",
                    "actions_taken": resume_state.get("executed_actions", []),
                    "approved_by":   auth.username,
                },
                slack_channel  = (resume_state.get("metadata") or {}).get("slack_channel", ""),
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("auto_warroom_failed incident=%s error=%s",
                                                resume_state.get("incident_id"), exc)

    return resume_state
