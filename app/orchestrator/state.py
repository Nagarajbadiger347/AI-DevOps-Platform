"""PipelineState — the single shared state object passed through LangGraph nodes.

Changes from original:
  - Added: trace_id, user, role, tenant_id (first-class fields, not buried in metadata)
  - Added: approval_deadline (unix timestamp — used by approval timeout cron)
  - Added: initial_state() factory — ensures required fields are always present
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Optional
from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    # ── Identity ─────────────────────────────────────────────────────────────
    incident_id:    str
    correlation_id: str
    trace_id:       str          # injected by runner, threads through all log lines
    user:           str          # resolved from JWT — never from request body
    role:           str          # resolved from JWT
    tenant_id:      str

    # ── Input ─────────────────────────────────────────────────────────────────
    description:    str
    auto_remediate: bool
    dry_run:        bool
    # caller-supplied extras: aws_cfg, k8s_cfg, hours, slack_channel, llm_provider
    metadata:       dict[str, Any]

    # ── Context collection ────────────────────────────────────────────────────
    aws_context:       dict[str, Any]
    k8s_context:       dict[str, Any]
    github_context:    dict[str, Any]
    similar_incidents: list[dict]       # retrieved from ChromaDB

    # ── Planning ──────────────────────────────────────────────────────────────
    # {"actions": [...], "confidence": float, "risk": str,
    #  "root_cause": str, "summary": str, "reasoning": str}
    plan: dict[str, Any]

    # ── Decision ──────────────────────────────────────────────────────────────
    risk_score:              float
    requires_human_approval: bool
    approval_reason:         str
    approval_deadline:       float      # unix timestamp — 0 means no deadline set

    # ── Execution ─────────────────────────────────────────────────────────────
    executed_actions: list[dict]        # actions that ran (ok, failed, dry_run)
    blocked_actions:  list[dict]        # actions blocked by rbac or policy

    # ── Validation ────────────────────────────────────────────────────────────
    validation_passed: bool
    validation_detail: dict[str, Any]

    # ── Flow control ──────────────────────────────────────────────────────────
    retry_count: int
    errors:      list[str]
    status:      str    # running | completed | escalated | awaiting_approval | failed

    # ── Observability ─────────────────────────────────────────────────────────
    started_at:   str   # ISO-8601
    completed_at: Optional[str]

    # ── Output ────────────────────────────────────────────────────────────────
    summary: str


def initial_state(
    incident_id: str,
    description: str,
    user: str = "system",
    role: str = "viewer",
    tenant_id: str = "default",
    auto_remediate: bool = False,
    dry_run: bool = False,
    metadata: Optional[dict] = None,
) -> PipelineState:
    """
    Factory for a fully initialised PipelineState.
    Always call this instead of constructing the dict manually —
    it guarantees all required fields are present with safe defaults.
    """
    return PipelineState(
        incident_id    = incident_id,
        correlation_id = str(uuid.uuid4()),
        trace_id       = str(uuid.uuid4()),
        user           = user,
        role           = role,
        tenant_id      = tenant_id,
        description    = description,
        auto_remediate = auto_remediate,
        dry_run        = dry_run,
        metadata       = metadata or {},
        # context — filled by collect_context node
        aws_context       = {},
        k8s_context       = {},
        github_context    = {},
        similar_incidents = [],
        # execution bookkeeping
        executed_actions = [],
        blocked_actions  = [],
        errors           = [],
        retry_count      = 0,
        approval_deadline = 0.0,
        # flow
        status     = "running",
        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
