"""Approval workflow system for incident response actions.

High-risk or high-cost actions must be approved by a human before execution.
Approval requests are held in memory with a 30-minute TTL and persisted to
``approvals.json`` so they survive process restarts.

Usage:
    from app.incident.approval import create_approval_request, approve_actions

    req = create_approval_request(
        incident_id="INC-001",
        actions=[{"action_type": "k8s_scale", ...}],
        plan="Scale payment-service from 2 → 6 replicas",
        risk_score=0.7,
        cost_report=report,
        requested_by="pipeline",
    )
    approve_actions(req.correlation_id, approved_action_indices=[0], approved_by="alice")
"""
from __future__ import annotations

import json
import os
import uuid
import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from pathlib import Path

try:
    from app.cost.analyzer import CostReport, format_cost_report
    _COST_AVAILABLE = True
except ImportError:
    _COST_AVAILABLE = False
    CostReport = Any  # type: ignore

try:
    from app.integrations.slack import post_message
    _SLACK_AVAILABLE = True
except ImportError:
    _SLACK_AVAILABLE = False

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_TTL_MINUTES = int(os.getenv("APPROVAL_TTL_MINUTES", "30"))
_APPROVALS_FILE = os.getenv(
    "APPROVALS_FILE",
    str(Path(__file__).resolve().parents[2] / "approvals.json"),
)

# Status constants
STATUS_PENDING  = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_EXPIRED  = "expired"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ApprovalRequest:
    """A single approval request for one or more incident actions."""
    incident_id:       str
    correlation_id:    str
    actions:           list[dict]
    plan_summary:      str
    risk_score:        float
    cost_report:       Optional[Any]          # CostReport or None
    requested_by:      str
    requested_at:      str
    expires_at:        str
    status:            str = STATUS_PENDING   # pending | approved | rejected | expired
    approved_by:       Optional[str] = None
    approved_at:       Optional[str] = None
    approved_actions:  list[dict] = field(default_factory=list)
    rejection_reason:  Optional[str] = None


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------
_pending_approvals: dict[str, ApprovalRequest] = {}


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------



def _log(level_fn, msg: str, **kwargs) -> None:
    """Emit a structured log message compatible with stdlib logging."""
    if kwargs:
        level_fn(msg, extra=kwargs)
    else:
        level_fn(msg)

def _serialize_approval(req: ApprovalRequest) -> dict:
    """Convert an ApprovalRequest to a JSON-serialisable dict."""
    d = asdict(req)
    # CostReport is a dataclass too — asdict handles it; if it's None, fine.
    return d


def _save_approvals() -> None:
    """Persist current approvals to disk."""
    try:
        path = Path(_APPROVALS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: _serialize_approval(v) for k, v in _pending_approvals.items()}
        path.write_text(json.dumps(data, indent=2, default=str))
    except Exception as exc:
        _log(logger.warning, "approvals_save_failed", error=str(exc))


def _load_approvals() -> None:
    """Load approvals from disk on startup."""
    try:
        path = Path(_APPROVALS_FILE)
        if not path.exists():
            return
        raw = json.loads(path.read_text())
        for cid, d in raw.items():
            status = d.get("status", STATUS_PENDING)
            # Skip expired/rejected — no need to restore
            if status in (STATUS_EXPIRED, STATUS_REJECTED):
                continue
            # Check pending items for expiry
            if status == STATUS_PENDING:
                expires_at = d.get("expires_at", "")
                if expires_at:
                    try:
                        exp_dt = datetime.datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                        if exp_dt < datetime.datetime.now(datetime.timezone.utc):
                            d["status"] = STATUS_EXPIRED
                            continue  # skip expired
                    except Exception:
                        pass
            # Restore pending AND approved (approved = approved but not yet resumed)
            _pending_approvals[cid] = ApprovalRequest(**{
                k: d[k] for k in ApprovalRequest.__dataclass_fields__ if k in d
            })
        _log(logger.info, "approvals_loaded", count=len(_pending_approvals))
    except Exception as exc:
        _log(logger.warning, "approvals_load_failed", error=str(exc))


# Load on import
_load_approvals()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _expires_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(minutes=_TTL_MINUTES)
    ).isoformat()


def _is_expired(req: ApprovalRequest) -> bool:
    try:
        exp = datetime.datetime.fromisoformat(req.expires_at.replace("Z", "+00:00"))
        return exp < datetime.datetime.now(datetime.timezone.utc)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_approval_request(
    incident_id: str,
    actions: list[dict],
    plan: str,
    risk_score: float,
    cost_report: Optional[Any],
    requested_by: str,
) -> ApprovalRequest:
    """Create a new approval request and store it.

    Args:
        incident_id:   Incident identifier (e.g. "INC-001").
        actions:       List of action dicts to be approved.
        plan:          Human-readable summary of what will happen.
        risk_score:    Float 0–1 representing assessed risk.
        cost_report:   CostReport instance (or None).
        requested_by:  Identifier of the requesting system/user.

    Returns:
        New ApprovalRequest with status=pending.
    """
    correlation_id = str(uuid.uuid4())
    req = ApprovalRequest(
        incident_id=incident_id,
        correlation_id=correlation_id,
        actions=actions,
        plan_summary=plan,
        risk_score=risk_score,
        cost_report=cost_report,
        requested_by=requested_by,
        requested_at=_now_iso(),
        expires_at=_expires_iso(),
    )
    _pending_approvals[correlation_id] = req
    _save_approvals()
    _log(logger.info, 
        "approval_request_created",
        correlation_id=correlation_id,
        incident_id=incident_id,
        actions=len(actions),
    )
    return req


def approve_actions(
    correlation_id: str,
    approved_action_indices: list[int],
    approved_by: str,
) -> ApprovalRequest:
    """Approve a subset of actions in a pending approval request.

    Args:
        correlation_id:          The request's correlation ID.
        approved_action_indices: 0-based indices of actions to approve.
        approved_by:             Username/identifier of the approver.

    Returns:
        Updated ApprovalRequest with status=approved.

    Raises:
        KeyError:   If the correlation_id is not found.
        ValueError: If the request is not in pending state or has expired.
    """
    req = _pending_approvals.get(correlation_id)
    if req is None:
        raise KeyError(f"No approval request found for correlation_id={correlation_id}")
    if _is_expired(req):
        req.status = STATUS_EXPIRED
        _save_approvals()
        raise ValueError(f"Approval request {correlation_id} has expired.")
    if req.status != STATUS_PENDING:
        raise ValueError(
            f"Approval request {correlation_id} is already in status={req.status}"
        )

    approved = [
        req.actions[i]
        for i in approved_action_indices
        if 0 <= i < len(req.actions)
    ]

    req.status           = STATUS_APPROVED
    req.approved_by      = approved_by
    req.approved_at      = _now_iso()
    req.approved_actions = approved

    _save_approvals()
    _log(logger.info, 
        "approval_granted",
        correlation_id=correlation_id,
        approved_by=approved_by,
        approved_count=len(approved),
    )
    return req


def reject_approval(
    correlation_id: str,
    reason: str,
    rejected_by: str,
) -> ApprovalRequest:
    """Reject an approval request.

    Args:
        correlation_id: The request's correlation ID.
        reason:         Human-readable rejection reason.
        rejected_by:    Username/identifier of the rejector.

    Returns:
        Updated ApprovalRequest with status=rejected.

    Raises:
        KeyError:   If the correlation_id is not found.
        ValueError: If the request is not in pending state.
    """
    req = _pending_approvals.get(correlation_id)
    if req is None:
        raise KeyError(f"No approval request found for correlation_id={correlation_id}")
    if req.status != STATUS_PENDING:
        raise ValueError(
            f"Approval request {correlation_id} is already in status={req.status}"
        )

    req.status           = STATUS_REJECTED
    req.approved_by      = rejected_by
    req.approved_at      = _now_iso()
    req.rejection_reason = reason

    _save_approvals()
    _log(logger.info, 
        "approval_rejected",
        correlation_id=correlation_id,
        rejected_by=rejected_by,
        reason=reason,
    )
    return req


def get_pending_approval(correlation_id: str) -> Optional[ApprovalRequest]:
    """Retrieve an approval request by correlation ID.

    Marks it as expired if the TTL has elapsed.

    Args:
        correlation_id: The request's correlation ID.

    Returns:
        ApprovalRequest or None if not found.
    """
    req = _pending_approvals.get(correlation_id)
    if req is None:
        return None
    if req.status == STATUS_PENDING and _is_expired(req):
        req.status = STATUS_EXPIRED
        _save_approvals()
    return req


def get_approval_request(correlation_id: str) -> Optional[ApprovalRequest]:
    """Return an ApprovalRequest by correlation_id, or None if not found."""
    return _pending_approvals.get(correlation_id)


def list_pending_approvals() -> list[ApprovalRequest]:
    """Return all approval requests currently in pending status.

    Expired requests are automatically transitioned.

    Returns:
        List of pending ApprovalRequest objects.
    """
    cleanup_expired()
    return [r for r in _pending_approvals.values() if r.status == STATUS_PENDING]


def cleanup_expired() -> int:
    """Expire and remove requests that have exceeded their TTL.

    Returns:
        Number of requests removed.
    """
    to_expire = [
        cid for cid, req in _pending_approvals.items()
        if req.status == STATUS_PENDING and _is_expired(req)
    ]
    for cid in to_expire:
        _pending_approvals[cid].status = STATUS_EXPIRED

    # Optionally remove fully expired ones from memory (keep rejected/approved for audit)
    removed = 0
    to_delete = [
        cid for cid, req in _pending_approvals.items()
        if req.status == STATUS_EXPIRED
    ]
    for cid in to_delete:
        del _pending_approvals[cid]
        removed += 1

    if removed:
        _save_approvals()
        _log(logger.info, "approvals_cleaned_up", removed=removed)
    return removed


def post_approval_to_slack(approval: ApprovalRequest, slack_channel: str) -> None:
    """Post a rich Slack message describing the pending approval request.

    Since we do not use interactive Slack buttons in this version, the
    message includes curl commands the approver can run to approve or reject.

    Args:
        approval:      The ApprovalRequest to announce.
        slack_channel: Slack channel ID or name to post to.
    """
    cid = approval.correlation_id

    action_lines = "\n".join(
        f"  [{i}] `{a.get('action_type', a.get('type', 'action'))}` — "
        f"{a.get('description', str(a)[:80])}"
        for i, a in enumerate(approval.actions)
    )

    all_indices = list(range(len(approval.actions)))

    cost_summary = ""
    if _COST_AVAILABLE and approval.cost_report is not None:
        try:
            cr = approval.cost_report
            cost_summary = (
                f"\n*Cost delta:* ${getattr(cr, 'total_estimated_monthly_delta', 0):+.2f}/month"
            )
        except Exception:
            pass

    expires_in = ""
    try:
        exp = datetime.datetime.fromisoformat(approval.expires_at.replace("Z", "+00:00"))
        remaining = exp - datetime.datetime.now(datetime.timezone.utc)
        mins = max(0, int(remaining.total_seconds() // 60))
        expires_in = f"\n*Expires in:* {mins} minutes"
    except Exception:
        pass

    text = (
        f":warning: *Approval Required* — Incident `{approval.incident_id}`\n\n"
        f"*Plan:* {approval.plan_summary}\n"
        f"*Risk score:* {approval.risk_score:.2f}\n"
        f"*Requested by:* {approval.requested_by}"
        f"{cost_summary}"
        f"{expires_in}\n\n"
        f"*Actions to approve:*\n{action_lines}\n\n"
        f"*To approve all actions:*\n"
        f"```\ncurl -X POST http://localhost:8000/approvals/{cid}/approve \\\n"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  -d '{{\"approved_action_indices\": {all_indices}, \"approved_by\": \"YOUR_NAME\"}}'\n```\n\n"
        f"*To reject:*\n"
        f"```\ncurl -X POST http://localhost:8000/approvals/{cid}/reject \\\n"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  -d '{{\"reason\": \"YOUR_REASON\", \"rejected_by\": \"YOUR_NAME\"}}'\n```"
    )

    if _SLACK_AVAILABLE:
        result = post_message(channel=slack_channel, text=text)
        if not result.get("success"):
            _log(logger.warning, "approval_slack_post_failed", error=result.get("error"))
        else:
            _log(logger.info, "approval_slack_posted", channel=slack_channel, correlation_id=cid)
    else:
        _log(logger.warning, "slack_unavailable_approval_not_posted", correlation_id=cid)
        print(f"[APPROVAL REQUIRED]\n{text}")
