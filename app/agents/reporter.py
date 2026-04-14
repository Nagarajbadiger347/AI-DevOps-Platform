"""
Reporter Agent — formats and delivers incident/analysis reports.
Supports Slack, API response, and structured JSON output.
"""
from __future__ import annotations
import logging

logger = logging.getLogger("nsops.agent.reporter")

_SEVERITY_EMOJI = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}


class ReporterAgent:
    """Formats workflow output and delivers it via configured channels."""

    def report(
        self,
        incident_state: dict,
        channels: list[str] | None = None,
    ) -> dict:
        """
        Generate and deliver a report from a workflow state dict.

        channels: list of ["slack", "api"] — defaults to ["api"]
        """
        channels = channels or ["api"]
        report_text = self._format_report(incident_state)
        deliveries = {}

        if "slack" in channels:
            deliveries["slack"] = self._send_slack(report_text, incident_state)
        if "api" in channels:
            deliveries["api"] = {"success": True, "report": report_text}

        logger.info("[REPORTER] reported to channels=%s severity=%s",
                    channels, incident_state.get("severity", "unknown"))

        return {
            "success": True,
            "report": report_text,
            "deliveries": deliveries,
            "severity": incident_state.get("severity"),
            "failure_type": incident_state.get("failure_type"),
        }

    def _format_report(self, state: dict) -> str:
        severity = state.get("severity", "unknown")
        emoji = _SEVERITY_EMOJI.get(severity, "⚪")
        failure_type = state.get("failure_type", "Unknown")
        pod = state.get("pod_name", "?")
        ns = state.get("namespace", "?")
        root_cause = state.get("root_cause", "N/A")
        fix = state.get("fix_suggestion", "N/A")
        fix_result = state.get("fix_result", {})
        errors = state.get("errors", [])
        timings = state.get("step_timings", {})

        # Fix status
        if fix_result.get("dry_run"):
            fix_line = f"**Fix (DRY-RUN):** {fix_result.get('message', '')}"
        elif fix_result.get("executed"):
            ok = fix_result.get("success", False)
            fix_line = f"**Fix:** {'✅ Executed successfully' if ok else '❌ Execution failed'}"
        else:
            fix_line = "**Fix:** Not executed (pass `auto_fix: true` to enable)"

        warnings = f"\n**⚠️ Warnings:** {'; '.join(errors[:2])}" if errors else ""
        timing_str = " | ".join(f"{k}:{v}s" for k, v in timings.items())

        return f"""{emoji} **K8s Incident Report**

**Pod:** `{pod}` | **Namespace:** `{ns}`
**Failure:** `{failure_type}` | **Severity:** {severity.upper()}

### Root Cause
{root_cause}

### Fix
{fix}

{fix_line}{warnings}

*Timing: {timing_str}*""".strip()

    def _send_slack(self, text: str, state: dict) -> dict:
        try:
            from app.integrations.slack import send_slack_message
            severity = state.get("severity", "medium")
            pod = state.get("pod_name", "?")
            ns = state.get("namespace", "?")
            failure = state.get("failure_type", "?")
            emoji = _SEVERITY_EMOJI.get(severity, "⚪")

            slack_text = (
                f"{emoji} *K8s Alert* | Pod: `{pod}` | NS: `{ns}` | `{failure}`\n"
                f"*Root cause:* {state.get('root_cause','')[:200]}\n"
                f"*Fix:* {state.get('fix_suggestion','')[:200]}"
            )
            send_slack_message(slack_text)
            return {"success": True}
        except Exception as e:
            logger.warning("[REPORTER] Slack send failed: %s", e)
            return {"success": False, "error": str(e)}
