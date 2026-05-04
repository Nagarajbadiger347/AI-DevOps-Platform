"""Tenant, Workspace, Plan, and Usage persistence."""
from __future__ import annotations

import datetime
import json
import logging
from typing import Optional, List

from app.tenants.models import Tenant, Workspace, WorkspaceMember, Plan, UsageSummary, TrainingExample
from app.core.database import execute, execute_one

logger = logging.getLogger(__name__)

_CURRENT_PERIOD = lambda: datetime.datetime.utcnow().strftime("%Y-%m")


# ── Plans ──────────────────────────────────────────────────────────────────

def get_plan(name: str) -> Optional[Plan]:
    row = execute_one("SELECT * FROM plans WHERE name = %s AND active = true", (name,), cached=True)
    return _row_to_plan(row) if row else None


def list_plans() -> List[Plan]:
    rows = execute("SELECT * FROM plans WHERE active = true ORDER BY price_monthly_cents")
    return [_row_to_plan(r) for r in rows]


def _row_to_plan(row: dict) -> Plan:
    features = row.get("features") or {}
    if isinstance(features, str):
        features = json.loads(features)
    return Plan(
        name=row["name"],
        display_name=row["display_name"],
        price_monthly_cents=row["price_monthly_cents"],
        price_yearly_cents=row["price_yearly_cents"],
        max_seats=row.get("max_seats"),
        max_actions_per_mo=row.get("max_actions_per_mo"),
        max_agent_runs_per_mo=row.get("max_agent_runs_per_mo"),
        max_llm_tokens_per_mo=row.get("max_llm_tokens_per_mo"),
        max_training_examples=row.get("max_training_examples"),
        features=features,
    )


# ── Workspaces ─────────────────────────────────────────────────────────────

def get_workspace(workspace_id: str) -> Optional[Workspace]:
    row = execute_one("SELECT * FROM workspaces WHERE id = %s AND active = true", (workspace_id,), cached=True)
    return _row_to_workspace(row) if row else None


def get_workspace_by_slug(slug: str) -> Optional[Workspace]:
    row = execute_one("SELECT * FROM workspaces WHERE slug = %s AND active = true", (slug,), cached=True)
    return _row_to_workspace(row) if row else None


def create_workspace(slug: str, name: str, owner_user_id: str, plan_name: str = "starter") -> Workspace:
    rows = execute(
        """
        INSERT INTO workspaces (slug, name, owner_user_id, plan_name)
        VALUES (%s, %s, %s, %s)
        RETURNING *
        """,
        (slug, name, owner_user_id, plan_name)
    )
    ws = _row_to_workspace(rows[0])
    # Add owner as admin member
    add_workspace_member(ws.id, owner_user_id, role="admin", invited_by="system")
    return ws


def update_workspace(workspace_id: str, updates: dict) -> Optional[Workspace]:
    allowed = {"name", "plan_name", "stripe_customer_id", "stripe_subscription_id",
               "subscription_status", "trial_ends_at", "current_period_start",
               "current_period_end", "logo_url", "metadata", "active"}
    filtered = {k: v for k, v in updates.items() if k in allowed}
    if not filtered:
        return get_workspace(workspace_id)
    set_clause = ", ".join(f"{k} = %s" for k in filtered)
    values = list(filtered.values()) + [workspace_id]
    execute(f"UPDATE workspaces SET {set_clause}, updated_at = NOW() WHERE id = %s", tuple(values))
    return get_workspace(workspace_id)


def _row_to_workspace(row: dict) -> Workspace:
    meta = row.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    return Workspace(
        id=str(row["id"]),
        slug=row["slug"],
        name=row["name"],
        owner_user_id=row["owner_user_id"],
        plan_name=row.get("plan_name", "starter"),
        stripe_customer_id=row.get("stripe_customer_id"),
        stripe_subscription_id=row.get("stripe_subscription_id"),
        subscription_status=row.get("subscription_status", "trialing"),
        trial_ends_at=str(row["trial_ends_at"]) if row.get("trial_ends_at") else None,
        current_period_start=str(row["current_period_start"]) if row.get("current_period_start") else None,
        current_period_end=str(row["current_period_end"]) if row.get("current_period_end") else None,
        logo_url=row.get("logo_url"),
        metadata=meta,
        active=row.get("active", True),
        created_at=str(row["created_at"]) if row.get("created_at") else None,
    )


# ── Workspace members ──────────────────────────────────────────────────────

def add_workspace_member(workspace_id: str, username: str, role: str = "developer",
                          email: str = "", invited_by: str = "") -> WorkspaceMember:
    rows = execute(
        """
        INSERT INTO workspace_members (workspace_id, username, email, role, invited_by)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (workspace_id, username) DO UPDATE SET role = EXCLUDED.role, active = true
        RETURNING *
        """,
        (workspace_id, username, email or None, role, invited_by or None)
    )
    return _row_to_member(rows[0])


def list_workspace_members(workspace_id: str) -> List[WorkspaceMember]:
    rows = execute(
        "SELECT * FROM workspace_members WHERE workspace_id = %s AND active = true ORDER BY joined_at",
        (workspace_id,)
    )
    return [_row_to_member(r) for r in rows]


def get_member_role(workspace_id: str, username: str) -> Optional[str]:
    row = execute_one(
        "SELECT role FROM workspace_members WHERE workspace_id = %s AND username = %s AND active = true",
        (workspace_id, username), cached=True
    )
    return row["role"] if row else None


def _row_to_member(row: dict) -> WorkspaceMember:
    return WorkspaceMember(
        id=row.get("id"),
        workspace_id=str(row["workspace_id"]),
        username=row["username"],
        email=row.get("email"),
        role=row.get("role", "developer"),
        invited_by=row.get("invited_by"),
        joined_at=str(row["joined_at"]) if row.get("joined_at") else None,
        active=row.get("active", True),
    )


# ── Tenants ────────────────────────────────────────────────────────────────

def get_tenant(tenant_id: str) -> Optional[Tenant]:
    row = execute_one(
        "SELECT * FROM tenants WHERE tenant_id = %s AND active = true",
        (tenant_id,), cached=True,
    )
    return _row_to_tenant(row) if row else None


def list_tenants() -> List[Tenant]:
    rows = execute("SELECT * FROM tenants ORDER BY created_at DESC")
    return [_row_to_tenant(r) for r in rows]


def create_tenant(tenant: Tenant) -> Tenant:
    execute(
        """
        INSERT INTO tenants (tenant_id, name, workspace_id, plan_name, aws_role_arn,
                             aws_region, slack_channel, llm_provider, active, metadata, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (tenant_id) DO NOTHING
        """,
        (
            tenant.tenant_id, tenant.name, tenant.workspace_id, tenant.plan_name,
            tenant.aws_role_arn, tenant.aws_region, tenant.slack_channel,
            tenant.llm_provider, tenant.active,
            json.dumps(tenant.metadata or {}),
            tenant.created_at or datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
    )
    return tenant


def update_tenant(tenant_id: str, updates: dict) -> Optional[Tenant]:
    allowed = {"name", "workspace_id", "plan_name", "aws_role_arn", "aws_region",
               "slack_channel", "llm_provider", "active", "metadata"}
    filtered = {k: v for k, v in updates.items() if k in allowed}
    if not filtered:
        return get_tenant(tenant_id)
    set_clause = ", ".join(f"{k} = %s" for k in filtered)
    values = list(filtered.values()) + [tenant_id]
    execute(f"UPDATE tenants SET {set_clause}, updated_at = NOW() WHERE tenant_id = %s", tuple(values))
    return get_tenant(tenant_id)


def _row_to_tenant(row: dict) -> Tenant:
    meta = row.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    return Tenant(
        tenant_id=row["tenant_id"],
        name=row["name"],
        workspace_id=str(row["workspace_id"]) if row.get("workspace_id") else None,
        plan_name=row.get("plan_name", "starter"),
        aws_role_arn=row.get("aws_role_arn"),
        aws_region=row.get("aws_region"),
        slack_channel=row.get("slack_channel", "#incidents"),
        llm_provider=row.get("llm_provider", ""),
        active=row.get("active", True),
        metadata=meta,
        created_at=str(row["created_at"]) if row.get("created_at") else None,
    )


# ── Usage metering ─────────────────────────────────────────────────────────

def record_usage(tenant_id: str, event_type: str, quantity: int = 1,
                 workspace_id: str = "", meta: dict = None) -> None:
    """Fire-and-forget usage event. Never raises."""
    try:
        period = _CURRENT_PERIOD()
        execute(
            """
            INSERT INTO usage_events (tenant_id, workspace_id, event_type, quantity, meta, billed_period)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (tenant_id, workspace_id or None, event_type, quantity,
             json.dumps(meta or {}), period)
        )
    except Exception as exc:
        logger.warning("usage_record_failed tenant=%s event=%s error=%s", tenant_id, event_type, exc)


def get_usage_summary(tenant_id: str, period: str = "") -> UsageSummary:
    period = period or _CURRENT_PERIOD()
    row = execute_one(
        "SELECT * FROM usage_monthly WHERE tenant_id = %s AND billed_period = %s",
        (tenant_id, period)
    )
    tenant = get_tenant(tenant_id)
    plan = get_plan(tenant.plan_name if tenant else "starter") if tenant else None

    return UsageSummary(
        tenant_id=tenant_id,
        workspace_id=str(row["workspace_id"]) if row and row.get("workspace_id") else None,
        billed_period=period,
        llm_tokens=int(row["llm_tokens"]) if row else 0,
        agent_runs=int(row["agent_runs"]) if row else 0,
        actions=int(row["actions"]) if row else 0,
        api_calls=int(row["api_calls"]) if row else 0,
        plan_name=tenant.plan_name if tenant else "starter",
        limits={
            "max_actions_per_mo":    plan.max_actions_per_mo if plan else None,
            "max_agent_runs_per_mo": plan.max_agent_runs_per_mo if plan else None,
            "max_llm_tokens_per_mo": plan.max_llm_tokens_per_mo if plan else None,
        } if plan else {},
    )


def check_limit(tenant_id: str, resource: str) -> dict:
    """Returns {"allowed": bool, "current": int, "limit": int|None}."""
    summary = get_usage_summary(tenant_id)
    current = getattr(summary, resource.replace("max_", "").replace("_per_mo", "s"), 0)
    limit = summary.limits.get(f"max_{resource}_per_mo") if "_per_mo" not in resource else summary.limits.get(resource)
    allowed = limit is None or current < limit
    return {"allowed": allowed, "current": current, "limit": limit}


# ── Training examples ──────────────────────────────────────────────────────

def add_training_example(example: TrainingExample) -> TrainingExample:
    rows = execute(
        """
        INSERT INTO training_examples
            (tenant_id, workspace_id, trigger_text, correct_plan, correct_action,
             correct_params, outcome, tags, source, quality_score, approved, created_by)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id, created_at
        """,
        (
            example.tenant_id, example.workspace_id, example.trigger_text,
            json.dumps(example.correct_plan), example.correct_action,
            json.dumps(example.correct_params), example.outcome,
            example.tags, example.source, example.quality_score,
            example.approved, example.created_by,
        )
    )
    if rows:
        example.id = rows[0]["id"]
        example.created_at = str(rows[0]["created_at"])
    return example


def list_training_examples(tenant_id: str, approved_only: bool = True,
                           limit: int = 100) -> List[TrainingExample]:
    sql = "SELECT * FROM training_examples WHERE tenant_id = %s"
    params: list = [tenant_id]
    if approved_only:
        sql += " AND approved = true"
    sql += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    rows = execute(sql, tuple(params))
    return [_row_to_example(r) for r in rows]


def approve_training_example(example_id: int, tenant_id: str) -> bool:
    rows = execute(
        "UPDATE training_examples SET approved = true, updated_at = NOW() "
        "WHERE id = %s AND tenant_id = %s RETURNING id",
        (example_id, tenant_id)
    )
    return len(rows) > 0


def delete_training_example(example_id: int, tenant_id: str) -> bool:
    rows = execute(
        "DELETE FROM training_examples WHERE id = %s AND tenant_id = %s RETURNING id",
        (example_id, tenant_id)
    )
    return len(rows) > 0


def _row_to_example(row: dict) -> TrainingExample:
    for field in ("correct_plan", "correct_params"):
        if isinstance(row.get(field), str):
            row[field] = json.loads(row[field])
    return TrainingExample(
        id=row.get("id"),
        tenant_id=row["tenant_id"],
        workspace_id=str(row["workspace_id"]) if row.get("workspace_id") else None,
        trigger_text=row["trigger_text"],
        correct_plan=row.get("correct_plan") or {},
        correct_action=row.get("correct_action"),
        correct_params=row.get("correct_params") or {},
        outcome=row.get("outcome", "resolved"),
        tags=list(row.get("tags") or []),
        source=row.get("source", "manual"),
        quality_score=float(row.get("quality_score") or 1.0),
        approved=row.get("approved", True),
        created_by=row.get("created_by"),
        created_at=str(row["created_at"]) if row.get("created_at") else None,
    )


# ── Action outcome collection ──────────────────────────────────────────────

def record_action_outcome(tenant_id: str, action: str, params: dict,
                          success: bool, error_msg: str = "",
                          duration_ms: int = 0, session_id: str = "",
                          workspace_id: str = "") -> None:
    """Auto-collect every action result as potential training data."""
    try:
        execute(
            """
            INSERT INTO action_outcomes
                (tenant_id, workspace_id, session_id, action, params, success, error_msg, duration_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (tenant_id, workspace_id or None, session_id or None, action,
             json.dumps(params), success, error_msg or None, duration_ms or None)
        )
    except Exception as exc:
        logger.warning("action_outcome_record_failed tenant=%s action=%s error=%s", tenant_id, action, exc)
