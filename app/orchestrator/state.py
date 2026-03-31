"""PipelineState — the single shared state object passed through LangGraph nodes."""
from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    # ── Input ───────────────────────────────────────────────────────────────
    incident_id:    str
    description:    str
    auto_remediate: bool
    # caller-supplied extras: user, role, aws_cfg, k8s_cfg, hours, slack_channel
    metadata:       dict[str, Any]

    # ── Context collection ──────────────────────────────────────────────────
    aws_context:       dict[str, Any]
    k8s_context:       dict[str, Any]
    github_context:    dict[str, Any]
    similar_incidents: list[dict]       # retrieved from ChromaDB

    # ── Planning ────────────────────────────────────────────────────────────
    # {"actions": [...], "confidence": float, "risk": str,
    #  "root_cause": str, "summary": str, "reasoning": str}
    plan: dict[str, Any]

    # ── Decision ────────────────────────────────────────────────────────────
    risk_score:              float
    requires_human_approval: bool
    approval_reason:         str

    # ── Execution ───────────────────────────────────────────────────────────
    executed_actions: list[dict]   # actions that ran (ok or failed)
    blocked_actions:  list[dict]   # actions blocked by policy

    # ── Validation ──────────────────────────────────────────────────────────
    validation_passed: bool
    validation_detail: dict[str, Any]

    # ── Flow control ────────────────────────────────────────────────────────
    retry_count:    int
    errors:         list[str]
    status:         str   # running | completed | escalated | awaiting_approval

    # ── Observability ───────────────────────────────────────────────────────
    correlation_id: str
    started_at:     str
    completed_at:   str

    # ── Output ──────────────────────────────────────────────────────────────
    summary: str
