"""One-time invite / password-setup tokens.

Flow:
  1. Admin creates a user → POST /users/invite {username, email, role}
  2. App generates a 6-digit OTP + secure token, stores in invites.json
  3. Email is sent with OTP and a link to /auth/setup-password?token=<token>
  4. User opens link, enters OTP + new password → POST /auth/setup-password
  5. Token is deleted — can only be used once

Requires env vars for email:
  SMTP_HOST     e.g. smtp.gmail.com
  SMTP_PORT     587
  SMTP_USER     your@email.com
  SMTP_PASSWORD app password / token
  SMTP_FROM     "DevOps AI <your@email.com>"  (optional, defaults to SMTP_USER)
  APP_URL       https://your-domain.com  (for the invite link)
"""

import json
import os
import secrets
import datetime
import smtplib
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_INVITES_PATH = Path(__file__).resolve().parent / "invites.json"

# In-memory: token → {username, email, otp, expires_at}
_invites: dict[str, dict] = {}


# ── Persistence ────────────────────────────────────────────────

def _load() -> None:
    try:
        with open(_INVITES_PATH) as f:
            _invites.update(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        pass

def _save() -> None:
    try:
        tmp = _INVITES_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(_invites, indent=2))
        tmp.replace(_INVITES_PATH)
    except Exception:
        pass

_load()


# ── Token management ───────────────────────────────────────────

def create_invite(username: str, email: str) -> dict:
    """Generate a one-time invite token + 6-digit OTP. Returns token."""
    token = secrets.token_urlsafe(32)
    otp   = str(random.randint(100000, 999999))
    expires_at = (datetime.datetime.utcnow() + datetime.timedelta(hours=24)).isoformat()
    _invites[token] = {
        "username":   username.strip().lower(),
        "email":      email.strip().lower(),
        "otp":        otp,
        "expires_at": expires_at,
    }
    _save()
    return {"token": token, "otp": otp, "expires_at": expires_at}


def validate_invite(token: str, otp: str) -> dict:
    """Validate token + OTP. Returns {valid, username} or {valid: False, error}."""
    invite = _invites.get(token)
    if not invite:
        return {"valid": False, "error": "Invalid or expired invite link"}
    expires = datetime.datetime.fromisoformat(invite["expires_at"])
    if datetime.datetime.utcnow() > expires:
        del _invites[token]
        _save()
        return {"valid": False, "error": "Invite link has expired (24h limit)"}
    if invite["otp"] != otp.strip():
        return {"valid": False, "error": "Incorrect OTP code"}
    return {"valid": True, "username": invite["username"], "token": token}


def consume_invite(token: str) -> None:
    """Delete invite after use (one-time only)."""
    if token in _invites:
        del _invites[token]
        _save()


def get_invite_username(token: str) -> Optional[str]:
    """Return username for a token without consuming it."""
    inv = _invites.get(token)
    if not inv:
        return None
    expires = datetime.datetime.fromisoformat(inv["expires_at"])
    if datetime.datetime.utcnow() > expires:
        return None
    return inv["username"]


# ── Email sender ───────────────────────────────────────────────

def send_invite_email(email: str, username: str, otp: str, token: str) -> dict:
    """Send invite email with OTP and setup link."""
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_pass = os.getenv("SMTP_PASSWORD", "").strip()
    smtp_from = os.getenv("SMTP_FROM", smtp_user).strip()
    app_url   = os.getenv("APP_URL", "http://localhost:8000").rstrip("/")

    if not smtp_host or not smtp_user:
        return {"success": False, "error": "SMTP not configured (set SMTP_HOST, SMTP_USER, SMTP_PASSWORD in .env)"}

    setup_link = f"{app_url}/auth/setup-password?token={token}"

    html_body = f"""
<div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:24px">
  <div style="font-size:24px;font-weight:700;margin-bottom:8px">⚡ DevOps AI Platform</div>
  <div style="font-size:15px;color:#333;margin-bottom:24px">You've been invited to join the platform.</div>

  <div style="background:#f5f5f5;border-radius:8px;padding:20px;margin-bottom:24px">
    <div style="font-size:13px;color:#666;margin-bottom:6px">Your username</div>
    <div style="font-size:18px;font-weight:700;font-family:monospace">{username}</div>
  </div>

  <div style="background:#1a1a2e;color:#fff;border-radius:8px;padding:20px;margin-bottom:24px;text-align:center">
    <div style="font-size:12px;color:#aaa;margin-bottom:8px;letter-spacing:.1em;text-transform:uppercase">Your One-Time Password (OTP)</div>
    <div style="font-size:40px;font-weight:700;letter-spacing:12px;font-family:monospace;color:#7c3aed">{otp}</div>
    <div style="font-size:11px;color:#aaa;margin-top:8px">Valid for 24 hours</div>
  </div>

  <a href="{setup_link}" style="display:block;background:#7c3aed;color:#fff;text-decoration:none;padding:14px 24px;border-radius:8px;text-align:center;font-weight:700;font-size:15px;margin-bottom:16px">
    Set My Password →
  </a>

  <div style="font-size:12px;color:#888;margin-top:16px">
    Or open this link manually:<br>
    <a href="{setup_link}" style="color:#7c3aed;word-break:break-all">{setup_link}</a>
  </div>
  <div style="font-size:11px;color:#aaa;margin-top:20px;border-top:1px solid #eee;padding-top:16px">
    This invite expires in 24 hours. If you didn't expect this email, ignore it.
  </div>
</div>
"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"⚡ DevOps AI — You're invited, {username}"
    msg["From"]    = smtp_from
    msg["To"]      = email
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as s:
            s.ehlo()
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(smtp_from, [email], msg.as_string())
        return {"success": True, "email": email}
    except Exception as e:
        return {"success": False, "error": str(e)}
