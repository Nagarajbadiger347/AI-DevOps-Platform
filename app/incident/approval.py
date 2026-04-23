"""Approval workflow system — PostgreSQL backed.

High-risk or high-cost actions must be approved by a human before execution.
All approvals are stored in PostgreSQL with tenant isolation.

Usage:
    from app.incident.approval import create_approval_request, approve_actions

    req = create_approval_request(
        incident_id="INC-001",
        actions=[{"action_type": "k8s_scale", ...}],
        plan="Scale payment-service from 2 → 6 replicas",
        risk_score=0.7,
        cost_report=report,
        requested_by="pipeline",
        tenant_id="acme",
    )
    approve_actions(req.correlation_id, approved_action_indices=[0], approved_by="alice", tenant_id="acme")
"""
from __future__ import annotations

import json
import os
import uuid
import datetime
from dataclasses import dataclass, field
from typing import Optional, Any

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

_TTL_MINUTES = int(os.getenv("APPROVAL_TTL_MINUTES", "30"))

STATUS_PENDING  = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_EXPIRED  = "expired"
STATUS_EXECUTED = "executed"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ApprovalRequest:
    incident_id:       str
    correlation_id:    str
    actions:           list[dict]
    plan_summary:      str
    risk_score:        float
    cost_report:       Optional[Any]
    requested_by:      str
    requested_at:      str
    expires_at:        str
    tenant_id:         str = "default"
    status:            str = STATUS_PENDING
    approved_by:       Optional[str] = None
    approved_at:       Optional[str] = None
    approved_actions:  list[dict] = field(default_factory=list)
    rejection_reason:  Optional[str] = None
    executed_at:       Optional[str] = None
    executed_by:       Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _db():
    from app.core.database import execute, execute_one
    return execute, execute_one

def _log(level_fn, msg: str, **kwargs) -> None:
    if kwargs:
        level_fn(msg, extra=kwargs)
    else:
        level_fn(msg)

def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def _expires_iso() -> str:
    return (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=_TTL_MINUTES)).isoformat()

def _is_expired(req: ApprovalRequest) -> bool:
    try:
        exp = datetime.datetime.fromisoformat(req.expires_at.replace("Z", "+00:00"))
        return exp < datetime.datetime.now(datetime.timezone.utc)
    except Exception:
        return False

def _row_to_approval(row: dict) -> ApprovalRequest:
    # psycopg2 auto-parses JSONB columns into dicts
    meta = row.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    return ApprovalRequest(
        incident_id=row["incident_id"] or "",
        correlation_id=row["approval_id"],
        actions=meta.get("actions", []),
        plan_summary=row.get("description") or "",
        risk_score=float(row.get("estimated_cost") or 0),
        cost_report=None,
        requested_by=row.get("requested_by") or "",
        requested_at=str(row.get("created_at") or ""),
        expires_at=meta.get("expires_at", ""),
        tenant_id=row.get("tenant_id", "default"),
        status=row.get("status", STATUS_PENDING),
        approved_by=row.get("approved_by"),
        approved_at=str(row.get("resolved_at") or "") if row.get("resolved_at") else None,
        approved_actions=meta.get("approved_actions", []),
        rejection_reason=meta.get("rejection_reason"),
    )


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
    tenant_id: str = "default",
) -> ApprovalRequest:
    execute, _ = _db()
    correlation_id = str(uuid.uuid4())
    expires_at = _expires_iso()

    metadata = json.dumps({
        "actions": actions,
        "approved_actions": [],
        "expires_at": expires_at,
    })

    execute(
        """
        INSERT INTO approvals (approval_id, tenant_id, incident_id, action_type,
                               description, requested_by, status, estimated_cost, metadata, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """,
        (correlation_id, tenant_id, incident_id, "multi_action", plan,
         requested_by, STATUS_PENDING, risk_score, metadata)
    )

    req = ApprovalRequest(
        incident_id=incident_id,
        correlation_id=correlation_id,
        actions=actions,
        plan_summary=plan,
        risk_score=risk_score,
        cost_report=cost_report,
        requested_by=requested_by,
        requested_at=_now_iso(),
        expires_at=expires_at,
        tenant_id=tenant_id,
    )
    _log(logger.info, "approval_request_created",
         correlation_id=correlation_id, incident_id=incident_id, actions=len(actions))
    return req


def approve_actions(
    correlation_id: str,
    approved_action_indices: list[int],
    approved_by: str,
    tenant_id: str = "default",
) -> ApprovalRequest:
    execute, execute_one = _db()
    row = execute_one(
        "SELECT * FROM approvals WHERE approval_id = %s AND tenant_id = %s",
        (correlation_id, tenant_id)
    )
    if not row:
        raise KeyError(f"No approval request found for correlation_id={correlation_id}")

    req = _row_to_approval(row)
    if _is_expired(req):
        execute("UPDATE approvals SET status = %s WHERE approval_id = %s", (STATUS_EXPIRED, correlation_id))
        raise ValueError(f"Approval request {correlation_id} has expired.")
    if req.status != STATUS_PENDING:
        raise ValueError(f"Approval request {correlation_id} is already in status={req.status}")

    approved = [req.actions[i] for i in approved_action_indices if 0 <= i < len(req.actions)]
    raw_meta = row["metadata"] or "{}"
    meta = raw_meta if isinstance(raw_meta, dict) else json.loads(raw_meta)
    meta["approved_actions"] = approved
    execute(
        "UPDATE approvals SET status = %s, approved_by = %s, resolved_at = NOW(), metadata = %s WHERE approval_id = %s",
        (STATUS_APPROVED, approved_by, json.dumps(meta), correlation_id)
    )
    req.status = STATUS_APPROVED
    req.approved_by = approved_by
    req.approved_actions = approved
    _log(logger.info, "approval_granted", correlation_id=correlation_id,
         approved_by=approved_by, approved_count=len(approved))
    return req


def reject_approval(correlation_id: str, reason: str, rejected_by: str, tenant_id: str = "default") -> ApprovalRequest:
    execute, execute_one = _db()
    row = execute_one(
        "SELECT * FROM approvals WHERE approval_id = %s AND tenant_id = %s",
        (correlation_id, tenant_id)
    )
    if not row:
        raise KeyError(f"No approval request found for correlation_id={correlation_id}")
    req = _row_to_approval(row)
    if req.status != STATUS_PENDING:
        raise ValueError(f"Approval request {correlation_id} is already in status={req.status}")

    raw_meta = row["metadata"] or "{}"
    meta = raw_meta if isinstance(raw_meta, dict) else json.loads(raw_meta)
    meta["rejection_reason"] = reason
    execute(
        "UPDATE approvals SET status = %s, approved_by = %s, resolved_at = NOW(), metadata = %s WHERE approval_id = %s",
        (STATUS_REJECTED, rejected_by, json.dumps(meta), correlation_id)
    )
    req.status = STATUS_REJECTED
    req.rejection_reason = reason
    _log(logger.info, "approval_rejected", correlation_id=correlation_id,
         rejected_by=rejected_by, reason=reason)
    return req


def get_pending_approval(correlation_id: str, tenant_id: str = "default") -> Optional[ApprovalRequest]:
    _, execute_one = _db()
    row = execute_one(
        "SELECT * FROM approvals WHERE approval_id = %s AND tenant_id = %s",
        (correlation_id, tenant_id)
    )
    if not row:
        return None
    req = _row_to_approval(row)
    if req.status == STATUS_PENDING and _is_expired(req):
        execute, _ = _db()
        execute("UPDATE approvals SET status = %s WHERE approval_id = %s", (STATUS_EXPIRED, correlation_id))
        req.status = STATUS_EXPIRED
    return req


def get_approval_request(correlation_id: str, tenant_id: str = "default") -> Optional[ApprovalRequest]:
    return get_pending_approval(correlation_id, tenant_id)


def list_pending_approvals(tenant_id: str = "default") -> list[ApprovalRequest]:
    execute, _ = _db()
    rows = execute(
        "SELECT * FROM approvals WHERE tenant_id = %s AND status = %s ORDER BY created_at DESC",
        (tenant_id, STATUS_PENDING)
    )
    result = []
    for row in rows:
        req = _row_to_approval(row)
        if _is_expired(req):
            execute("UPDATE approvals SET status = %s WHERE approval_id = %s", (STATUS_EXPIRED, req.correlation_id))
            continue
        result.append(req)
    return result


def cleanup_expired(tenant_id: str = "default") -> int:
    execute, _ = _db()
    rows = execute(
        "UPDATE approvals SET status = %s WHERE tenant_id = %s AND status = %s AND created_at < NOW() - INTERVAL '%s minutes' RETURNING approval_id",
        (STATUS_EXPIRED, tenant_id, STATUS_PENDING, _TTL_MINUTES)
    )
    return len(rows)


def post_approval_to_slack(approval: ApprovalRequest, slack_channel: str) -> None:
    cid = approval.correlation_id
    action_lines = "\n".join(
        f"  [{i}] `{a.get('action_type', 'action')}` — {a.get('description', str(a)[:80])}"
        for i, a in enumerate(approval.actions)
    )
    all_indices = list(range(len(approval.actions)))
    text = (
        f":warning: *Approval Required* — Incident `{approval.incident_id}`\n\n"
        f"*Plan:* {approval.plan_summary}\n"
        f"*Risk score:* {approval.risk_score:.2f}\n"
        f"*Requested by:* {approval.requested_by}\n\n"
        f"*Actions:*\n{action_lines}\n\n"
        f"*To approve:*\n"
        f"```\ncurl -X POST http://localhost:8000/approvals/{cid}/approve \\\n"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  -d '{{\"approved_action_indices\": {all_indices}, \"approved_by\": \"YOUR_NAME\"}}'\n```"
    )
    if _SLACK_AVAILABLE:
        result = post_message(channel=slack_channel, text=text)
        if not result.get("success"):
            _log(logger.warning, "approval_slack_post_failed", error=result.get("error"))
    else:
        print(f"[APPROVAL REQUIRED]\n{text}")
