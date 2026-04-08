"""Runner — public entry point for the LangGraph pipeline.

Usage (from FastAPI or monitoring loop):
    from app.orchestrator.runner import run_pipeline
    result = run_pipeline(
        incident_id="INC-001",
        description="API pods are crash-looping",
        auto_remediate=False,
        metadata={"user": "nagaraj", "role": "admin",
                  "k8s_cfg": {"namespace": "prod"}, "hours": 2},
    )
"""
from __future__ import annotations

import datetime

from app.orchestrator.graph import pipeline_graph
from app.orchestrator.state import PipelineState
from app.core.logging import get_logger, new_correlation_id

logger = get_logger(__name__)


def _slack_notify_pipeline(incident_id: str, description: str, state: dict,
                           triggered_by: str = "monitor") -> None:
    """Create a dedicated Slack channel and post a comprehensive incident briefing."""
    try:
        from app.integrations.slack import create_incident_channel, post_message, SLACK_BOT_TOKEN
        if not SLACK_BOT_TOKEN:
            return

        plan       = state.get("plan") or {}
        root_cause = plan.get("root_cause") or state.get("root_cause") or "Under investigation"
        risk       = (plan.get("risk") or state.get("risk_level") or "unknown").lower()
        confidence = plan.get("confidence") or 0.0
        actions    = plan.get("actions") or []
        executed   = state.get("executed_actions") or []
        blocked    = state.get("blocked_actions") or []
        errors     = state.get("errors") or []
        status     = state.get("status") or "completed"
        aws_ctx    = state.get("aws_context") or {}
        k8s_ctx    = state.get("k8s_context") or {}
        gh_ctx     = state.get("github_context") or {}
        reasoning  = plan.get("reasoning") or ""
        data_gaps  = plan.get("data_gaps") or []
        severity   = risk if risk in ("critical","high","medium","low") else "unknown"

        sev_emoji  = {"critical":"🔴","high":"🟠","medium":"🟡","low":"🟢"}.get(severity,"⚪")
        st_emoji   = "✅" if status=="completed" else "⏳" if status=="awaiting_approval" else "❌"
        conf_pct   = int(confidence * 100)
        conf_bar   = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
        now        = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        trigger_src = "🤖 Auto-detected alert" if triggered_by == "monitor" else f"👤 Manually triggered by *{triggered_by}*"

        # 1. Create dedicated channel
        ch = create_incident_channel(incident_id, topic=f"[{severity.upper()}] {description[:70]}")
        channel = ch.get("channel_id") or ch.get("channel_name") or ""
        if not channel:
            from app.integrations.slack import SLACK_CHANNEL
            channel = SLACK_CHANNEL

        # 2. Post comprehensive briefing using Block Kit
        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": f"{sev_emoji} Incident {incident_id}"}},

            # ── WHAT / WHEN / WHO / HOW ──────────────────────────
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*🕐 When*\n{now}"},
                {"type": "mrkdwn", "text": f"*👤 Triggered by*\n{trigger_src}"},
                {"type": "mrkdwn", "text": f"*⚠️ Severity*\n{sev_emoji} {severity.upper()}"},
                {"type": "mrkdwn", "text": f"*🎯 Confidence*\n`{conf_bar}` {conf_pct}%"},
            ]},
            {"type": "divider"},

            # ── WHAT HAPPENED ─────────────────────────────────────
            {"type": "section", "text": {"type": "mrkdwn",
                "text": f"*📋 What happened*\n{description}"}},

            # ── WHY / ROOT CAUSE ─────────────────────────────────
            {"type": "section", "text": {"type": "mrkdwn",
                "text": f"*🔍 Why (Root Cause)*\n{root_cause}"}},
        ]

        # ── HOW (Reasoning) ──────────────────────────────────────
        if reasoning:
            blocks.append({"type": "section", "text": {"type": "mrkdwn",
                "text": f"*🧠 How (AI Reasoning)*\n{reasoning[:600]}"}})

        # ── INFRASTRUCTURE CONTEXT ───────────────────────────────
        infra_lines = []
        if aws_ctx.get("_data_available"):
            alarms = aws_ctx.get("alarms") or []
            firing = [a for a in alarms if str(a.get("state","")).upper() == "ALARM"]
            if firing:
                infra_lines.append(f"• *AWS Alarms firing:* {', '.join(a.get('name','?') for a in firing[:4])}")
            instances = aws_ctx.get("instances") or []
            stopped = [i for i in instances if i.get("state") != "running"]
            if stopped:
                infra_lines.append(f"• *EC2 stopped:* {', '.join(i.get('id','?') for i in stopped[:4])}")
        if k8s_ctx.get("_data_available"):
            bad_pods = k8s_ctx.get("unhealthy_pods") or []
            if bad_pods:
                infra_lines.append(f"• *K8s unhealthy pods:* {len(bad_pods)} — {', '.join(p.get('name','?') for p in bad_pods[:3])}")
        if gh_ctx.get("_data_available"):
            commits = gh_ctx.get("recent_commits") or []
            if commits:
                last = commits[0]
                infra_lines.append(f"• *Last commit:* `{last.get('sha','')[:7]}` {last.get('message','')[:60]} by {last.get('author','?')}")
        if infra_lines:
            blocks.append({"type": "divider"})
            blocks.append({"type": "section", "text": {"type": "mrkdwn",
                "text": "*🏗️ Infrastructure at time of incident*\n" + "\n".join(infra_lines)}})

        # ── WHAT AI DID / PLANS TO DO ────────────────────────────
        blocks.append({"type": "divider"})
        action_lines = []
        for a in executed[:6]:
            action_lines.append(f"✅ *{a.get('type','?')}* — {a.get('description','')[:100]}")
        for a in blocked[:6]:
            action_lines.append(f"⏳ *{a.get('type','?')}* _(pending approval)_ — {a.get('description','')[:80]}")
        for a in actions[:6]:
            if a.get('type') not in {x.get('type') for x in executed+blocked}:
                action_lines.append(f"📋 *{a.get('type','?')}* — {a.get('description','')[:80]}")
        if action_lines:
            blocks.append({"type": "section", "text": {"type": "mrkdwn",
                "text": "*⚡ Actions*\n" + "\n".join(action_lines)}})

        if errors:
            blocks.append({"type": "section", "text": {"type": "mrkdwn",
                "text": f"*⚠️ Errors*\n" + "\n".join(f"• {e}" for e in errors[:3])}})

        if data_gaps:
            blocks.append({"type": "section", "text": {"type": "mrkdwn",
                "text": f"*🔎 Data gaps*\n" + "\n".join(f"• {g}" for g in data_gaps[:4])}})

        # ── STATUS + CHAT HINT ───────────────────────────────────
        blocks.append({"type": "divider"})
        blocks.append({"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*Pipeline status*\n{st_emoji} {status.upper()}"},
            {"type": "mrkdwn", "text": f"*Actions executed / pending*\n{len(executed)} executed · {len(blocked)} pending approval"},
        ]})
        blocks.append({"type": "context", "elements": [
            {"type": "mrkdwn",
             "text": ":robot_face: *NsOps AI is watching this channel.* "
                     "Mention `@nsops` or start your message with `ai:` to ask the AI anything about this incident."}
        ]})

        post_message(channel=channel,
                     text=f"{sev_emoji} Incident *{incident_id}* — {severity.upper()} | {description[:80]}",
                     blocks=blocks)

        logger.info("slack_incident_channel_notified", extra={
            "incident_id": incident_id, "channel": channel,
        })

        # Auto-create / update war room linked to this channel
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
            logger.warning("war_room_auto_create_failed", extra={"incident_id": incident_id, "error": str(exc)})

    except Exception as exc:
        logger.warning("slack_notify_pipeline_failed", extra={"incident_id": incident_id, "error": str(exc)})



def run_pipeline(
    incident_id: str,
    description: str,
    auto_remediate: bool = False,
    dry_run: bool = False,
    metadata: dict | None = None,
) -> dict:
    """Invoke the LangGraph pipeline synchronously and return the final state.

    Args:
        incident_id:     Unique ID for this incident run.
        description:     Human-readable description of the incident.
        auto_remediate:  If True and risk is low/medium, execute without approval.
        metadata:        Extra context — user, role, aws_cfg, k8s_cfg, hours, etc.

    Returns:
        Final PipelineState as a plain dict.
    """
    cid = new_correlation_id()
    logger.info("pipeline_started", extra={
        "incident_id": incident_id,
        "auto_remediate": auto_remediate,
        "correlation_id": cid,
    })

    initial_state: PipelineState = {
        "incident_id":    incident_id,
        "description":    description,
        "auto_remediate": auto_remediate,
        "dry_run":        dry_run,
        "metadata":       metadata or {},
        "errors":         [],
        "retry_count":    0,
        "status":         "running",
        "correlation_id": cid,
        "started_at":     datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    try:
        final_state: dict = pipeline_graph.invoke(initial_state)
    except Exception as exc:
        logger.error("pipeline_unhandled_error", extra={
            "incident_id": incident_id, "error": str(exc),
        })
        final_state = {**initial_state,
                       "status": "failed",
                       "errors": [str(exc)],
                       "completed_at": datetime.datetime.now(
                           datetime.timezone.utc).isoformat()}

    logger.info("pipeline_finished", extra={
        "incident_id": incident_id,
        "status": final_state.get("status"),
        "validation_passed": final_state.get("validation_passed"),
        "actions_executed": len(final_state.get("executed_actions", [])),
        "actions_blocked": len(final_state.get("blocked_actions", [])),
    })

    # Auto-post to Slack: create dedicated channel + send full analysis
    _slack_notify_pipeline(incident_id, description, final_state)

    try:
        from app.integrations.vscode import write_output, notify
        status = final_state.get("status", "unknown")
        root_cause = final_state.get("root_cause") or ""
        executed = final_state.get("executed_actions", [])
        risk = final_state.get("risk_level", "")
        icon = "✅" if status == "completed" else ("⏳" if status == "awaiting_approval" else "❌")
        write_output(
            f"{icon} PIPELINE DONE  [{incident_id}]  status={status}  risk={risk}  "
            f"actions={len(executed)}  root_cause={root_cause[:80] if root_cause else 'n/a'}"
        )
        if status == "awaiting_approval":
            notify(f"[{incident_id}] Awaiting approval — {root_cause[:60]}", level="warning")
        elif status == "failed":
            notify(f"[{incident_id}] Pipeline failed — {description[:60]}", level="error")
    except Exception:
        pass

    return final_state
