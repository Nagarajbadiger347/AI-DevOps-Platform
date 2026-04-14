"""PipelineState — single shared state object passed through all LangGraph nodes.

Fields are first-class typed properties. Use initial_state() factory — never
construct the dict manually, as it guarantees all fields have safe defaults.
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
    trace_id:       str     # propagated to every log line via context var
    user:           str     # resolved from JWT — NEVER from request body
    role:           str     # resolved from JWT
    tenant_id:      str

    # ── Input ─────────────────────────────────────────────────────────────────
    description:    str
    severity:       str     # critical | high | medium | low
    auto_remediate: bool
    dry_run:        bool
    metadata:       dict[str, Any]   # aws_cfg, k8s_cfg, hours, slack_channel, etc.

    # ── Context collection ────────────────────────────────────────────────────
    aws_context:       dict[str, Any]
    k8s_context:       dict[str, Any]
    github_context:    dict[str, Any]
    similar_incidents: list[dict]    # retrieved from ChromaDB

    # ── Planning ──────────────────────────────────────────────────────────────
    # {"actions": [...], "confidence": float, "risk": str,
    #  "root_cause": str, "summary": str, "reasoning": str, "data_gaps": [...]}
    plan: dict[str, Any]

    # ── Decision ──────────────────────────────────────────────────────────────
    risk_score:              float
    requires_human_approval: bool
    approval_reason:         str
    approval_deadline:       float   # unix timestamp; 0.0 = not set

    # ── Execution ─────────────────────────────────────────────────────────────
    executed_actions: list[dict]     # actions that ran (ok | failed | dry_run)
    blocked_actions:  list[dict]     # actions blocked by RBAC or policy

    # ── Validation ────────────────────────────────────────────────────────────
    validation_passed: bool
    validation_detail: dict[str, Any]

    # ── Flow control ──────────────────────────────────────────────────────────
    retry_count: int
    errors:      list[str]
    status:      str   # running | completed | escalated | awaiting_approval | failed

    # ── Observability ─────────────────────────────────────────────────────────
    started_at:   str          # ISO-8601
    completed_at: Optional[str]

    # ── Output ────────────────────────────────────────────────────────────────
    summary: str


def initial_state(
    incident_id: str,
    description: str,
    user: str = "system",
    role: str = "viewer",
    tenant_id: str = "default",
    severity: str = "medium",
    auto_remediate: bool = False,
    dry_run: bool = False,
    metadata: Optional[dict] = None,
) -> PipelineState:
    """
    Factory for a fully-initialised PipelineState with safe defaults.
    Always use this instead of constructing the dict manually.
    """
    trace_id = str(uuid.uuid4())
    return PipelineState(
        incident_id    = incident_id,
        correlation_id = str(uuid.uuid4()),
        trace_id       = trace_id,
        user           = user,
        role           = role,
        tenant_id      = tenant_id,
        description    = description,
        severity       = severity,
        auto_remediate = auto_remediate,
        dry_run        = dry_run,
        metadata       = metadata or {},
        # context — filled by collect_context node
        aws_context       = {},
        k8s_context       = {},
        github_context    = {},
        similar_incidents = [],
        # execution bookkeeping
        executed_actions  = [],
        blocked_actions   = [],
        errors            = [],
        retry_count       = 0,
        approval_deadline = 0.0,
        # flow
        status     = "running",
        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
