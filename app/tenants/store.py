"""Tenant persistence — PostgreSQL backed store (replaces tenants.json)."""
from __future__ import annotations

import datetime
import logging
from typing import Optional, List

from app.tenants.models import Tenant
from app.core.database import execute, execute_one

logger = logging.getLogger(__name__)


def _row_to_tenant(row: dict) -> Tenant:
    return Tenant(
        tenant_id=row["tenant_id"],
        name=row["name"],
        aws_role_arn=row.get("aws_role_arn"),
        aws_region=row.get("aws_region"),
        slack_channel=row.get("slack_channel", "#incidents"),
        llm_provider=row.get("llm_provider", ""),
        active=row.get("active", True),
        metadata=row.get("metadata") or {},
        created_at=str(row["created_at"]) if row.get("created_at") else None,
    )


def get_tenant(tenant_id: str) -> Optional[Tenant]:
    row = execute_one(
        "SELECT * FROM tenants WHERE tenant_id = %s AND active = true",
        (tenant_id,),
        cached=True,   # cache for 30s — tenants change rarely
    )
    return _row_to_tenant(row) if row else None


def list_tenants() -> List[Tenant]:
    rows = execute("SELECT * FROM tenants ORDER BY created_at DESC")
    return [_row_to_tenant(r) for r in rows]


def create_tenant(tenant: Tenant) -> Tenant:
    execute(
        """
        INSERT INTO tenants (tenant_id, name, aws_role_arn, aws_region,
                             slack_channel, llm_provider, active, metadata, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (tenant_id) DO NOTHING
        """,
        (
            tenant.tenant_id,
            tenant.name,
            tenant.aws_role_arn,
            tenant.aws_region,
            tenant.slack_channel,
            tenant.llm_provider,
            tenant.active,
            __import__("json").dumps(tenant.metadata or {}),
            tenant.created_at or datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
    )
    return tenant


def update_tenant(tenant_id: str, updates: dict) -> Optional[Tenant]:
    allowed = {"name", "aws_role_arn", "aws_region", "slack_channel", "llm_provider", "active", "metadata"}
    filtered = {k: v for k, v in updates.items() if k in allowed}
    if not filtered:
        return get_tenant(tenant_id)

    set_clause = ", ".join(f"{k} = %s" for k in filtered)
    values = list(filtered.values()) + [tenant_id]
    execute(f"UPDATE tenants SET {set_clause}, updated_at = NOW() WHERE tenant_id = %s", tuple(values))
    return get_tenant(tenant_id)


def delete_tenant(tenant_id: str) -> bool:
    rows = execute("DELETE FROM tenants WHERE tenant_id = %s RETURNING tenant_id", (tenant_id,))
    return len(rows) > 0
