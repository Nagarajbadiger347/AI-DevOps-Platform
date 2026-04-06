"""Incident email notifications.

Sends alerts for:
  - New incident created (awaiting_approval or running)
  - Approval required (awaiting_approval)
  - Incident completed or failed

Requires the same SMTP env vars used by invite.py:
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM
  ALERT_EMAIL_TO   — comma-separated list of recipient addresses
  APP_URL          — base URL for dashboard links
"""
from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _smtp_configured() -> bool:
    return bool(os.getenv("SMTP_HOST", "").strip() and os.getenv("SMTP_USER", "").strip())


def _send(subject: str, html_body: str, to_addrs: list[str]) -> dict:
    """Low-level send via SMTP. Returns {success, error?}."""
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_pass = os.getenv("SMTP_PASSWORD", "").strip()
    smtp_from = os.getenv("SMTP_FROM", smtp_user).strip()

    if not smtp_host or not smtp_user:
        return {"success": False, "error": "SMTP not configured"}
    if not to_addrs:
        return {"success": False, "error": "No recipients configured (set ALERT_EMAIL_TO in .env)"}

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = ", ".join(to_addrs)
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as s:
            s.ehlo()
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(smtp_from, to_addrs, msg.as_string())
        return {"success": True, "recipients": to_addrs}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _get_recipients() -> list[str]:
    raw = os.getenv("ALERT_EMAIL_TO", "").strip()
    if not raw:
        return []
    return [e.strip() for e in raw.split(",") if e.strip()]


def _risk_color(risk: str) -> str:
    return {"critical": "#ef4444", "high": "#f97316", "medium": "#f59e0b", "low": "#22c55e"}.get(
        (risk or "").lower(), "#6b7280"
    )


def _status_color(status: str) -> str:
    return {"completed": "#22c55e", "failed": "#ef4444", "escalated": "#f97316",
            "awaiting_approval": "#f59e0b"}.get((status or "").lower(), "#6b7280")


def send_approval_required(incident_id: str, description: str, risk: str,
                            confidence: float, actions: list, approval_reason: str,
                            app_url: str = "") -> dict:
    """Email alert when an incident requires human approval before execution."""
    if not _smtp_configured():
        return {"success": False, "error": "SMTP not configured"}

    recipients = _get_recipients()
    app_url = (app_url or os.getenv("APP_URL", "http://localhost:8000")).rstrip("/")
    dashboard_link = f"{app_url}/#incidents"
    risk_col = _risk_color(risk)
    conf_pct = int(confidence * 100)

    actions_html = ""
    for a in actions[:10]:
        atype = a.get("type", "unknown")
        adesc = a.get("description", "")
        actions_html += f'<tr><td style="padding:6px 10px;font-family:monospace;font-size:13px;color:#7c3aed">{atype}</td><td style="padding:6px 10px;font-size:13px;color:#374151">{adesc}</td></tr>'

    html = f"""
<div style="font-family:sans-serif;max-width:560px;margin:0 auto;padding:24px;background:#fff">
  <div style="border-left:4px solid {risk_col};padding:16px 20px;background:#fafafa;border-radius:0 8px 8px 0;margin-bottom:20px">
    <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">⚠️ Approval Required</div>
    <div style="font-size:20px;font-weight:700;color:#111;margin-bottom:4px">{incident_id}</div>
    <div style="font-size:14px;color:#374151">{description}</div>
  </div>

  <table style="width:100%;border-collapse:collapse;margin-bottom:20px">
    <tr>
      <td style="padding:8px;background:#f3f4f6;border-radius:6px;text-align:center;width:33%">
        <div style="font-size:11px;color:#6b7280;margin-bottom:2px">RISK</div>
        <div style="font-size:16px;font-weight:700;color:{risk_col}">{(risk or 'unknown').upper()}</div>
      </td>
      <td style="width:8px"></td>
      <td style="padding:8px;background:#f3f4f6;border-radius:6px;text-align:center;width:33%">
        <div style="font-size:11px;color:#6b7280;margin-bottom:2px">CONFIDENCE</div>
        <div style="font-size:16px;font-weight:700;color:#111">{conf_pct}%</div>
      </td>
      <td style="width:8px"></td>
      <td style="padding:8px;background:#f3f4f6;border-radius:6px;text-align:center;width:33%">
        <div style="font-size:11px;color:#6b7280;margin-bottom:2px">ACTIONS</div>
        <div style="font-size:16px;font-weight:700;color:#111">{len(actions)}</div>
      </td>
    </tr>
  </table>

  <div style="font-size:13px;color:#374151;background:#fffbeb;border:1px solid #fde68a;border-radius:6px;padding:10px 14px;margin-bottom:18px">
    <strong>Reason for approval:</strong> {approval_reason}
  </div>

  {'<table style="width:100%;border-collapse:collapse;margin-bottom:18px;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden"><thead><tr style="background:#f9fafb"><th style="padding:8px 10px;text-align:left;font-size:12px;color:#6b7280">ACTION TYPE</th><th style="padding:8px 10px;text-align:left;font-size:12px;color:#6b7280">DESCRIPTION</th></tr></thead><tbody>' + actions_html + '</tbody></table>' if actions else ''}

  <a href="{dashboard_link}" style="display:block;background:#7c3aed;color:#fff;text-decoration:none;padding:13px 24px;border-radius:8px;text-align:center;font-weight:700;font-size:15px;margin-bottom:16px">
    Review &amp; Approve in Dashboard →
  </a>

  <div style="font-size:11px;color:#9ca3af;border-top:1px solid #e5e7eb;padding-top:14px;margin-top:8px">
    NsOps AI · Automated incident response · Incident ID: {incident_id}
  </div>
</div>
"""
    return _send(
        subject=f"[NsOps] Approval Required — {incident_id} ({(risk or 'unknown').upper()} risk)",
        html_body=html,
        to_addrs=recipients,
    )


def send_incident_completed(incident_id: str, description: str, risk: str,
                             status: str, root_cause: str, summary: str,
                             actions_executed: int, validation_passed: bool) -> dict:
    """Email summary when an incident pipeline completes or fails."""
    if not _smtp_configured():
        return {"success": False, "error": "SMTP not configured"}

    recipients = _get_recipients()
    app_url = os.getenv("APP_URL", "http://localhost:8000").rstrip("/")
    dashboard_link = f"{app_url}/#incidents"
    risk_col = _risk_color(risk)
    status_col = _status_color(status)
    val_icon = "✅" if validation_passed else "⚠️"

    html = f"""
<div style="font-family:sans-serif;max-width:560px;margin:0 auto;padding:24px;background:#fff">
  <div style="border-left:4px solid {status_col};padding:16px 20px;background:#fafafa;border-radius:0 8px 8px 0;margin-bottom:20px">
    <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">Incident Resolved</div>
    <div style="font-size:20px;font-weight:700;color:#111;margin-bottom:4px">{incident_id}</div>
    <div style="font-size:14px;color:#374151">{description}</div>
  </div>

  <table style="width:100%;border-collapse:collapse;margin-bottom:20px">
    <tr>
      <td style="padding:8px;background:#f3f4f6;border-radius:6px;text-align:center;width:25%">
        <div style="font-size:11px;color:#6b7280;margin-bottom:2px">STATUS</div>
        <div style="font-size:14px;font-weight:700;color:{status_col}">{(status or '—').upper()}</div>
      </td>
      <td style="width:6px"></td>
      <td style="padding:8px;background:#f3f4f6;border-radius:6px;text-align:center;width:25%">
        <div style="font-size:11px;color:#6b7280;margin-bottom:2px">RISK</div>
        <div style="font-size:14px;font-weight:700;color:{risk_col}">{(risk or '—').upper()}</div>
      </td>
      <td style="width:6px"></td>
      <td style="padding:8px;background:#f3f4f6;border-radius:6px;text-align:center;width:25%">
        <div style="font-size:11px;color:#6b7280;margin-bottom:2px">ACTIONS</div>
        <div style="font-size:14px;font-weight:700;color:#111">{actions_executed}</div>
      </td>
      <td style="width:6px"></td>
      <td style="padding:8px;background:#f3f4f6;border-radius:6px;text-align:center;width:25%">
        <div style="font-size:11px;color:#6b7280;margin-bottom:2px">VALIDATION</div>
        <div style="font-size:16px">{val_icon}</div>
      </td>
    </tr>
  </table>

  {'<div style="margin-bottom:14px"><div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px">Root Cause</div><div style="font-size:13px;color:#374151;background:#f9fafb;border-radius:6px;padding:10px 14px">' + root_cause + '</div></div>' if root_cause else ''}

  {'<div style="margin-bottom:18px"><div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px">Summary</div><div style="font-size:13px;color:#374151;background:#f9fafb;border-radius:6px;padding:10px 14px">' + summary + '</div></div>' if summary else ''}

  <a href="{dashboard_link}" style="display:block;background:#7c3aed;color:#fff;text-decoration:none;padding:13px 24px;border-radius:8px;text-align:center;font-weight:700;font-size:15px;margin-bottom:16px">
    View Full Report →
  </a>

  <div style="font-size:11px;color:#9ca3af;border-top:1px solid #e5e7eb;padding-top:14px">
    NsOps AI · Incident ID: {incident_id}
  </div>
</div>
"""
    status_label = status.replace("_", " ").title()
    return _send(
        subject=f"[NsOps] {status_label} — {incident_id} ({(risk or 'unknown').upper()} risk)",
        html_body=html,
        to_addrs=recipients,
    )
