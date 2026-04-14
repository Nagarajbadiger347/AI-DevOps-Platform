"""Runner — public entry point for the LangGraph pipeline.

Usage:
    from app.orchestrator.runner import run_pipeline

    result = run_pipeline(
        incident_id    = "INC-001",
        description    = "API pods crash-looping in production",
        user           = "alice",
        role           = "admin",
        auto_remediate = False,
        dry_run        = True,
        metadata       = {"k8s_cfg": {"namespace": "prod"}, "hours": 2},
    )

The runner is the only place that builds the initial state and calls the graph.
It owns: correlation ID generation, trace context injection, error boundary,
Slack notification, and VS Code bridge output.
"""
from __future__ import annotations

import datetime

from app.core.logging import get_logger, new_correlation_id, new_trace_id, set_context
from app.orchestrator.graph import pipeline_graph
from app.orchestrator.state import PipelineState, initial_state

logger = get_logger(__name__)


# ── Slack notification ────────────────────────────────────────────────────────

def _slack_notify_pipeline(
    incident_id: str,
    description: str,
    state: dict,
    triggered_by: str = "monitor",
) -> None:
    """Create a dedicated Slack channel and post a comprehensive incident briefing."""
    try:
        from app.integrations.slack import create_incident_channel, post_message, SLACK_BOT_TOKEN
        if not SLACK_BOT_TOKEN:
            return

        plan        = state.get("plan") or {}
        root_cause  = plan.get("root_cause") or state.get("root_cause") or "Under investigation"
        risk        = (plan.get("risk") or state.get("risk_level") or "unknown").lower()
        confidence  = plan.get("confidence") or 0.0
        actions     = plan.get("actions") or []
        executed    = state.get("executed_actions") or []
        blocked     = state.get("blocked_actions") or []
        errors      = state.get("errors") or []
        status      = state.get("status") or "completed"
        aws_ctx     = state.get("aws_context") or {}
        k8s_ctx     = state.get("k8s_context") or {}
        gh_ctx      = state.get("github_context") or {}
        reasoning   = plan.get("reasoning") or ""
        data_gaps   = plan.get("data_gaps") or []
        severity    = risk if risk in ("critical", "high", "medium", "low") else "unknown"
        trace_id    = state.get("trace_id", "")

        sev_emoji  = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
        st_emoji   = "✅" if status == "completed" else "⏳" if status == "awaiting_approval" else "❌"
        conf_pct   = int(confidence * 100)
        conf_bar   = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
        now        = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        trigger_src = (
            "🤖 Auto-detected alert" if triggered_by == "monitor"
            else f"👤 Manually triggered by *{triggered_by}*"
        )

        ch      = create_incident_channel(incident_id, topic=f"[{severity.upper()}] {description[:70]}")
        channel = ch.get("channel_id") or ch.get("channel_name") or ""
        if not channel:
            from app.integrations.slack import SLACK_CHANNEL
            channel = SLACK_CHANNEL

        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": f"{sev_emoji} Incident {incident_id}"}},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*🕐 When*\n{now}"},
                {"type": "mrkdwn", "text": f"*👤 Triggered by*\n{trigger_src}"},
                {"type": "mrkdwn", "text": f"*⚠️ Severity*\n{sev_emoji} {severity.upper()}"},
                {"type": "mrkdwn", "text": f"*🎯 Confidence*\n`{conf_bar}` {conf_pct}%"},
            ]},
            {"type": "divider"},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*📋 What happened*\n{description}"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*🔍 Root Cause*\n{root_cause}"}},
        ]

        if reasoning:
            blocks.append({"type": "section", "text": {"type": "mrkdwn",
                "text": f"*🧠 AI Reasoning*\n{reasoning[:600]}"}})

        # Infrastructure context
        infra_lines = []
        if aws_ctx.get("_data_available"):
            firing = [a for a in (aws_ctx.get("alarms") or []) if str(a.get("state", "")).upper() == "ALARM"]
            if firing:
                infra_lines.append(f"• *AWS Alarms firing:* {', '.join(a.get('name', '?') for a in firing[:4])}")
            stopped = [i for i in (aws_ctx.get("instances") or []) if i.get("state") != "running"]
            if stopped:
                infra_lines.append(f"• *EC2 stopped:* {', '.join(i.get('id', '?') for i in stopped[:4])}")
        if k8s_ctx.get("_data_available"):
            bad_pods = k8s_ctx.get("unhealthy_pods") or []
            if bad_pods:
                infra_lines.append(f"• *K8s unhealthy:* {len(bad_pods)} pods — {', '.join(p.get('name', '?') for p in bad_pods[:3])}")
        if gh_ctx.get("_data_available"):
            commits = gh_ctx.get("recent_commits") or []
            if commits:
                c = commits[0]
                infra_lines.append(f"• *Last commit:* `{c.get('sha','')[:7]}` {c.get('message','')[:60]} by {c.get('author','?')}")
        if infra_lines:
            blocks += [
                {"type": "divider"},
                {"type": "section", "text": {"type": "mrkdwn",
                    "text": "*🏗️ Infrastructure snapshot*\n" + "\n".join(infra_lines)}},
            ]

        # Actions
        action_lines = []
        executed_types = {x.get("type") for x in executed + blocked}
        for a in executed[:6]:
            action_lines.append(f"✅ *{a.get('type', '?')}* ({a.get('duration_ms', '?')}ms) — {a.get('description', '')[:80]}")
        for a in blocked[:6]:
            action_lines.append(f"🚫 *{a.get('type', '?')}* blocked by {a.get('blocked_by', '?')} — {a.get('reason', '')[:80]}")
        for a in actions[:6]:
            if a.get("type") not in executed_types:
                action_lines.append(f"📋 *{a.get('type', '?')}* — {a.get('description', '')[:80]}")
        if action_lines:
            blocks.append({"type": "section", "text": {"type": "mrkdwn",
                "text": "*⚡ Actions*\n" + "\n".join(action_lines)}})

        if errors:
            blocks.append({"type": "section", "text": {"type": "mrkdwn",
                "text": f"*⚠️ Errors*\n" + "\n".join(f"• {e}" for e in errors[:3])}})

        if data_gaps:
            blocks.append({"type": "section", "text": {"type": "mrkdwn",
                "text": f"*🔎 Data gaps*\n" + "\n".join(f"• {g}" for g in data_gaps[:4])}})

        blocks += [
            {"type": "divider"},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*Pipeline status*\n{st_emoji} {status.upper()}"},
                {"type": "mrkdwn", "text": f"*Actions*\n{len(executed)} executed · {len(blocked)} blocked"},
            ]},
            {"type": "context", "elements": [
                {"type": "mrkdwn",
                 "text": f":robot_face: *NexusOps AI* is watching. Mention `@nsops` or prefix with `ai:` to ask questions. Trace: `{trace_id}`"}
            ]},
        ]

        post_message(
            channel=channel,
            text=f"{sev_emoji} Incident *{incident_id}* — {severity.upper()} | {description[:80]}",
            blocks=blocks,
        )

        logger.info("slack_incident_channel_notified", extra={
            "incident_id": incident_id, "channel": channel, "trace_id": trace_id,
        })

        # Auto-create war room
        try:
            from app.incident.war_room_store import create as _wr_create
            _wr_create(
                incident_id=incident_id,
                description=description,
                pipeline_state={
                    "root_cause":       root_cause,
                    "severity":         severity,
                    "status":           status,
                    "executed_actions": executed,
                    "blocked_actions":  blocked,
                    "actions":          actions,
                    "confidence":       confidence,
                    "reasoning":        reasoning,
                    "aws_context":      aws_ctx,
                    "k8s_context":      k8s_ctx,
                    "github_context":   gh_ctx,
                    "triggered_by":     triggered_by,
                    "triggered_at":     now,
                },
                slack_channel=channel,
            )
        except Exception as exc:
            logger.warning("war_room_auto_create_failed", extra={
                "incident_id": incident_id, "error": str(exc),
            })

    except Exception as exc:
        logger.warning("slack_notify_pipeline_failed", extra={
            "incident_id": incident_id, "error": str(exc),
        })


# ── Public entry point ────────────────────────────────────────────────────────

def run_pipeline(
    incident_id:    str,
    description:    str,
    user:           str = "system",
    role:           str = "viewer",
    tenant_id:      str = "default",
    severity:       str = "medium",
    auto_remediate: bool = False,
    dry_run:        bool = False,
    metadata:       dict | None = None,
) -> dict:
    """Invoke the LangGraph pipeline synchronously and return the final state.

    Args:
        incident_id:    Unique ID for this incident run.
        description:    Human-readable description of the incident.
        user:           Resolved from JWT — never pass from request body.
        role:           Resolved from JWT.
        tenant_id:      Tenant context.
        severity:       critical | high | medium | low.
        auto_remediate: If True and risk is low/medium, execute without approval.
        dry_run:        If True, plan but do not execute any real actions.
        metadata:       Extra context — aws_cfg, k8s_cfg, hours, slack_channel, etc.

    Returns:
        Final PipelineState as a plain dict.
    """
    cid      = new_correlation_id()
    trace_id = new_trace_id()

    set_context(
        correlation_id = cid,
        trace_id       = trace_id,
        incident_id    = incident_id,
        user           = user,
        tenant_id      = tenant_id,
    )

    logger.info("pipeline_started", extra={
        "incident_id":    incident_id,
        "trace_id":       trace_id,
        "correlation_id": cid,
        "user":           user,
        "role":           role,
        "tenant_id":      tenant_id,
        "severity":       severity,
        "auto_remediate": auto_remediate,
        "dry_run":        dry_run,
    })

    state = initial_state(
        incident_id    = incident_id,
        description    = description,
        user           = user,
        role           = role,
        tenant_id      = tenant_id,
        severity       = severity,
        auto_remediate = auto_remediate,
        dry_run        = dry_run,
        metadata       = metadata or {},
    )
    # Inject the generated IDs so graph nodes can access them
    state["correlation_id"] = cid
    state["trace_id"]       = trace_id

    try:
        final_state: dict = pipeline_graph.invoke(state)
    except Exception as exc:
        logger.error("pipeline_unhandled_error", extra={
            "incident_id": incident_id,
            "trace_id":    trace_id,
            "error":       str(exc),
        }, exc_info=True)
        final_state = {
            **state,
            "status":       "failed",
            "errors":       state.get("errors", []) + [str(exc)],
            "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    logger.info("pipeline_finished", extra={
        "incident_id":      incident_id,
        "trace_id":         trace_id,
        "status":           final_state.get("status"),
        "validation_passed": final_state.get("validation_passed"),
        "actions_executed": len(final_state.get("executed_actions", [])),
        "actions_blocked":  len(final_state.get("blocked_actions", [])),
        "errors":           len(final_state.get("errors", [])),
    })

    # Auto-post to Slack: dedicated channel + full briefing
    _slack_notify_pipeline(incident_id, description, final_state, triggered_by=user or "system")

    # VS Code bridge output
    try:
        from app.integrations.vscode import write_output, notify
        status     = final_state.get("status", "unknown")
        root_cause = (final_state.get("plan") or {}).get("root_cause") or ""
        executed   = final_state.get("executed_actions", [])
        risk       = (final_state.get("plan") or {}).get("risk", "")
        icon       = "✅" if status == "completed" else ("⏳" if status == "awaiting_approval" else "❌")
        write_output(
            f"{icon} PIPELINE [{incident_id}] status={status} risk={risk} "
            f"actions={len(executed)} trace={trace_id} "
            f"root_cause={root_cause[:80] if root_cause else 'n/a'}"
        )
        if status == "awaiting_approval":
            notify(f"[{incident_id}] Awaiting approval — {root_cause[:60]}", level="warning")
        elif status == "failed":
            notify(f"[{incident_id}] Pipeline failed — {description[:60]}", level="error")
    except Exception:
        pass

    return final_state
