"""Persistent storage for pre-approval pipeline state.

Pre-approval pipeline state used to live only in `_PENDING_PIPELINE_STATES`,
an in-process dict in app/api/deps.py. That meant:

  * Resume after worker restart was impossible — state was gone, the resume
    handler fell back to a synthesised stub missing aws_context, k8s_context,
    severity, and the original metadata.
  * Two simultaneous /resume calls both passed the status check and executed
    twice.

This module persists the same state to the existing `approvals.metadata` JSONB
column under the key `pipeline_state`, and offers `lock_for_resume()` which
takes a row-level `FOR UPDATE` lock so concurrent resumes serialize.

The in-memory dict is still used as a write-through cache for the happy path
(low latency, no DB round-trip on read). The DB is the source of truth on
worker restart.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def save(correlation_id: str, state: dict) -> None:
    """Write pipeline state to BOTH the in-memory cache and the
    approvals row. Best-effort: a DB failure does not break the caller."""
    from app.api.deps import _PENDING_PIPELINE_STATES
    _PENDING_PIPELINE_STATES[correlation_id] = state
    try:
        from app.core.database import execute
        execute(
            """
            UPDATE approvals
               SET metadata = jsonb_set(
                   COALESCE(metadata, '{}'::jsonb),
                   '{pipeline_state}',
                   %s::jsonb,
                   true)
             WHERE approval_id = %s
            """,
            (json.dumps(state, default=str), correlation_id),
        )
    except Exception as exc:
        logger.warning(
            "pending_state_persist_failed correlation_id=%s error=%s",
            correlation_id, exc,
        )


def load(correlation_id: str) -> Optional[dict]:
    """Look up pipeline state for a correlation id. Memory first, DB second."""
    from app.api.deps import _PENDING_PIPELINE_STATES
    cached = _PENDING_PIPELINE_STATES.get(correlation_id)
    if cached:
        return cached
    try:
        from app.core.database import execute_one
        row = execute_one(
            "SELECT metadata FROM approvals WHERE approval_id = %s",
            (correlation_id,),
        )
        if not row:
            return None
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        ps = meta.get("pipeline_state")
        if ps:
            _PENDING_PIPELINE_STATES[correlation_id] = ps
            return ps
    except Exception as exc:
        logger.warning(
            "pending_state_load_failed correlation_id=%s error=%s",
            correlation_id, exc,
        )
    return None


def delete(correlation_id: str) -> None:
    """Remove pipeline state from cache and DB."""
    from app.api.deps import _PENDING_PIPELINE_STATES
    _PENDING_PIPELINE_STATES.pop(correlation_id, None)
    try:
        from app.core.database import execute
        execute(
            "UPDATE approvals SET metadata = metadata - 'pipeline_state' "
            "WHERE approval_id = %s",
            (correlation_id,),
        )
    except Exception as exc:
        logger.warning(
            "pending_state_delete_failed correlation_id=%s error=%s",
            correlation_id, exc,
        )


def lock_for_resume(correlation_id: str, expected_status: str = "approved") -> tuple[bool, str]:
    """Atomically check that the approval is in `expected_status` and mark
    it as `executing` so concurrent resumes lose the race.

    Implementation: SELECT … FOR UPDATE inside a single transaction, then
    UPDATE in the same transaction.  Two callers race on the row lock; the
    second sees status='executing' and returns (False, 'already executing').

    Returns (acquired, reason).
    """
    from app.core.database import get_conn
    from psycopg2.extras import RealDictCursor
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT status FROM approvals WHERE approval_id = %s FOR UPDATE",
                    (correlation_id,),
                )
                row = cur.fetchone()
                if not row:
                    return False, f"approval {correlation_id} not found"
                status = (row.get("status") or "").lower()
                if status == "executing":
                    return False, "already executing"
                if status != expected_status:
                    return False, f"approval is in state '{status}', not '{expected_status}'"
                cur.execute(
                    "UPDATE approvals SET status = 'executing' WHERE approval_id = %s",
                    (correlation_id,),
                )
        return True, "locked"
    except Exception as exc:
        logger.error(
            "pending_state_lock_failed correlation_id=%s error=%s",
            correlation_id, exc,
        )
        return False, f"lock error: {exc}"
