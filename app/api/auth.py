"""
Authentication, user management, and SSO routes.
Paths: /auth/*, /users/*, /admin/backup/*
"""
import os
import time as _time
from pathlib import Path
from typing import Optional, Dict

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.api.deps import (
    require_admin, require_viewer, require_super_admin,
    AuthContext, _bearer_scheme, RoleAssignment,
)

router = APIRouter(tags=["auth"])

# ── Brute-force protection ─────────────────────────────────────
_login_failures: dict[str, list[float]] = {}
_LOGIN_MAX_ATTEMPTS = 5
_LOGIN_LOCKOUT_SECONDS = 300  # 5 minutes

# ── .env writer (used by /auth/configure-smtp and /secrets) ───
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def _write_env(updates: Dict[str, str]) -> None:
    """Merge updates into the .env file (create if absent)."""
    lines: list[str] = []
    existing_keys: set[str] = set()
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                key = stripped.split("=", 1)[0].strip()
                if key in updates:
                    lines.append(f'{key}={updates[key]}')
                    existing_keys.add(key)
                    continue
            lines.append(line)
    for key, val in updates.items():
        if key not in existing_keys:
            lines.append(f"{key}={val}")
    _ENV_FILE.write_text("\n".join(lines) + "\n")
    for key, val in updates.items():
        os.environ[key] = val


# ── Pydantic models ────────────────────────────────────────────

class UserCreateRequest(BaseModel):
    username: str
    password: str = "INVITE_PENDING"
    role: str = "viewer"
    email: Optional[str] = None


class PasswordChangeRequest(BaseModel):
    new_password: str


class SelfPasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


class SetupPasswordRequest(BaseModel):
    token: str
    otp: str
    new_password: str


class SmtpConfigRequest(BaseModel):
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    app_url: str = "http://localhost:8000"


# ── /auth/me ──────────────────────────────────────────────────

@router.get("/auth/me")
def auth_me(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    user: str = "nagaraj",
    x_user: Optional[str] = Header(default=None),
):
    """Return role and permissions for a user. Supports JWT Bearer token."""
    from app.security.rbac import ROLE_PERMISSIONS, get_user_role
    username = None
    if credentials and credentials.credentials:
        from app.core.auth import decode_token
        payload = decode_token(credentials.credentials)
        username = payload.get("sub")
    if not username:
        username = (x_user or user or "nagaraj").strip().lower()
    role = get_user_role(username)
    perms = list(ROLE_PERMISSIONS.get(role, set()))
    return {"username": username, "user": username, "role": role, "permissions": perms}


@router.post("/auth/token", tags=["auth"])
def login(request: Request, form: OAuth2PasswordRequestForm = Depends()):
    from app.security.users import authenticate, user_exists
    from app.security.rbac import get_user_role
    from app.core.auth import create_token
    username = form.username.strip().lower()
    now = _time.time()
    attempts = _login_failures.get(username, [])
    recent = [t for t in attempts if now - t < _LOGIN_LOCKOUT_SECONDS]
    if len(recent) >= _LOGIN_MAX_ATTEMPTS:
        retry_after = int(_LOGIN_LOCKOUT_SECONDS - (now - recent[0]))
        raise HTTPException(status_code=429, detail=f"Too many failed login attempts. Try again in {retry_after}s.")
    if not user_exists(username) or not authenticate(username, form.password):
        recent.append(now)
        _login_failures[username] = recent[-_LOGIN_MAX_ATTEMPTS:]
        raise HTTPException(status_code=401, detail="Invalid username or password")
    _login_failures.pop(username, None)
    role = get_user_role(username)
    token = create_token(username, role)
    try:
        from app.core.audit import audit_log as _al
        _al(user=username, action="login", params={"method": "password", "role": role},
            result={"success": True}, source="auth")
    except Exception:
        pass
    return {"access_token": token, "token_type": "bearer", "role": role, "username": username}


@router.get("/users", tags=["users"])
def list_users_endpoint(auth: AuthContext = Depends(require_admin)):
    from app.security.users import list_users as _list
    return {"users": _list()}


@router.post("/users", tags=["users"])
def create_user_endpoint(req: UserCreateRequest, auth: AuthContext = Depends(require_admin)):
    """Direct user creation (admin only). Use /users/invite for email-verified onboarding."""
    from app.security.users import create_user as _create
    from app.security.rbac import assign_role
    result = _create(req.username, req.password, created_by=auth.username)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    assign_role(req.username, req.role, changed_by=auth.username)
    return {"success": True, "username": req.username.lower(), "role": req.role}


@router.delete("/users/{username}", tags=["users"])
def delete_user_endpoint(username: str, auth: AuthContext = Depends(require_admin)):
    if username.strip().lower() == auth.username:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    from app.security.users import delete_user as _delete
    from app.security.rbac import revoke_role, get_user_role, PROTECTED_ROLES
    target_role = get_user_role(username.strip().lower())
    if target_role in PROTECTED_ROLES and auth.role != "super_admin":
        raise HTTPException(status_code=403, detail=f"Only a super_admin can delete a user with role '{target_role}'")
    result = _delete(username)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["error"])
    revoke_role(username, changed_by=auth.username)
    return {"success": True, "username": username}


@router.put("/users/{username}/role", tags=["users"])
def set_user_role_endpoint(username: str, req: RoleAssignment, auth: AuthContext = Depends(require_admin)):
    from app.security.rbac import assign_role
    result = assign_role(username, req.role, changed_by=auth.username, changer_role=auth.role)
    if not result["success"]:
        raise HTTPException(status_code=403, detail=result.get("reason", "Failed"))
    return result


@router.post("/auth/change-password", tags=["auth"])
def change_own_password(req: SelfPasswordChangeRequest, auth: AuthContext = Depends(require_viewer)):
    """Allow any authenticated user to change their own password."""
    from app.security.users import authenticate, change_password
    if not authenticate(auth.username, req.current_password):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")
    result = change_password(auth.username, req.new_password)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to update password"))
    from app.core.audit import audit_log
    audit_log(user=auth.username, action="change_own_password", params={}, result={"success": True}, source="api")
    return {"success": True, "message": "Password updated successfully"}


@router.put("/users/{username}/password", tags=["users"])
def reset_password_endpoint(username: str, req: PasswordChangeRequest, auth: AuthContext = Depends(require_admin)):
    from app.security.users import change_password
    result = change_password(username, req.new_password)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ── SSO / OAuth2 endpoints ─────────────────────────────────────

@router.get("/auth/sso/status", tags=["auth"])
def sso_status():
    from app.core.auth import SSO_PROVIDER
    return {"configured": bool(SSO_PROVIDER), "provider": SSO_PROVIDER or ""}


@router.get("/auth/sso/login", tags=["auth"])
def sso_login():
    from app.core.auth import get_sso_login_url, SSO_PROVIDER
    import secrets as _sec
    if not SSO_PROVIDER:
        raise HTTPException(status_code=400, detail="SSO is not configured. Set SSO_PROVIDER, SSO_CLIENT_ID, SSO_CLIENT_SECRET, SSO_REDIRECT_URI in .env")
    state = _sec.token_urlsafe(16)
    url = get_sso_login_url(state=state)
    return RedirectResponse(url=url)


@router.get("/auth/sso/callback", tags=["auth"])
def sso_callback(code: str = "", error: str = ""):
    """OAuth2 callback — exchange code for a platform JWT."""
    def _error_page(msg: str) -> HTMLResponse:
        return HTMLResponse(status_code=400, content=f"""<!doctype html><html>
<head><title>SSO Error</title>
<style>body{{font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#0f172a}}
.box{{background:#1e293b;color:#f1f5f9;padding:2rem 3rem;border-radius:12px;text-align:center;max-width:420px}}
h2{{color:#f87171;margin-top:0}}a{{color:#60a5fa;text-decoration:none}}</style></head>
<body><div class="box"><h2>SSO Login Failed</h2><p>{msg}</p><br><a href="/">← Back to Login</a></div></body></html>""")

    if error:
        return _error_page(f"Google returned an error: <b>{error}</b>")
    if not code:
        return _error_page("Missing authorization code.")

    from app.core.auth import exchange_sso_code, create_token, SSO_DEFAULT_ROLE
    from app.security.users import user_exists, create_user as _create_user, find_user_by_email, _users
    from app.security.rbac import get_user_role, assign_role
    from app.security.invite import get_invite_by_email

    try:
        user_info = exchange_sso_code(code)
    except Exception as e:
        return _error_page(str(e))

    email    = user_info["email"].strip().lower()
    username = user_info["username"]

    if os.getenv("SSO_REQUIRE_INVITE", "false").lower() == "true":
        import json as _json
        from pathlib import Path as _Path
        _inv_path = _Path(__file__).resolve().parents[1] / "security" / "invites.json"
        try:
            inv_data = _json.loads(_inv_path.read_text())
        except Exception:
            inv_data = {}
        invited_emails = {v.get("email", "").lower() for v in inv_data.values() if isinstance(v, dict)}
        if email not in invited_emails:
            return _error_page(
                f"<b>{email}</b> has not been invited to this platform.<br>"
                "Ask an admin to send you an invite first."
            )

    existing_by_email = find_user_by_email(email)
    if existing_by_email:
        username = existing_by_email
        if not _users[username].get("email"):
            _users[username]["email"] = email
            from app.security.users import _save as _users_save
            _users_save()
    elif user_exists(username):
        existing_info = _users.get(username, {})
        existing_email = existing_info.get("email", "")
        if existing_info.get("sso_only"):
            pass
        else:
            return _error_page(
                f"The username <b>{username}</b> is already taken by a different account "
                f"({'linked to ' + existing_email if existing_email else 'password-based'}).<br>"
                "Contact an admin to link your Google account to your existing account."
            )
    else:
        invite = get_invite_by_email(email)
        role_to_assign = invite["role"] if invite and invite.get("role") else SSO_DEFAULT_ROLE
        _create_user(username, password=None, created_by="sso", email=email)
        assign_role(username, role_to_assign, changed_by="sso")

    role  = get_user_role(username)
    token = create_token(username, role)
    try:
        from app.core.audit import audit_log as _al
        _al(user=username, action="login", params={"method": "sso_google", "role": role},
            result={"success": True}, source="auth")
    except Exception:
        pass
    return HTMLResponse(content=f"""<!doctype html><html><body>
<script>
  localStorage.setItem('nexusops_token', '{token}');
  localStorage.setItem('nexusops_user', '{username}');
  localStorage.setItem('nexusops_role', '{role}');
  window.location.href = '/';
</script></body></html>""")


@router.post("/auth/logout", tags=["auth"])
def logout(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
           auth: AuthContext = Depends(require_viewer)):
    """Revoke the current token so it cannot be reused even before expiry."""
    from app.core.auth import blacklist_token
    if credentials and credentials.credentials:
        blacklist_token(credentials.credentials)
    try:
        from app.core.audit import audit_log as _al
        _al(user=auth.username, action="logout", params={"role": auth.role},
            result={"success": True}, source="auth")
    except Exception:
        pass
    return {"success": True, "message": "Logged out and token revoked"}


@router.post("/auth/revoke-all", tags=["auth"])
def revoke_all_tokens(auth: AuthContext = Depends(require_admin)):
    raise HTTPException(status_code=422, detail="Use POST /auth/revoke-all/{username}")


@router.post("/auth/revoke-all/{username}", tags=["auth"])
def revoke_all_user(username: str, auth: AuthContext = Depends(require_admin)):
    """Admin: invalidate all existing tokens for a user (forces re-login)."""
    from app.core.auth import revoke_all_user_tokens
    revoke_all_user_tokens(username)
    from app.core.audit import audit_log
    audit_log(user=auth.username, action="revoke_all_tokens",
              params={"target_user": username}, result={"success": True}, source="api")
    return {"success": True, "message": f"All tokens for '{username}' have been revoked"}


# ── ChromaDB backup endpoints ──────────────────────────────────

@router.post("/admin/backup/chromadb", tags=["admin"])
def trigger_backup(auth: AuthContext = Depends(require_admin)):
    """Manually trigger a ChromaDB backup."""
    from app.memory.vector_db import backup_chromadb
    result = backup_chromadb()
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Backup failed"))
    return result


@router.get("/admin/backup/list", tags=["admin"])
def list_backups(auth: AuthContext = Depends(require_admin)):
    """List available ChromaDB backups."""
    from app.memory.vector_db import get_backup_list
    return {"backups": get_backup_list()}


# ── User invite endpoints ──────────────────────────────────────

@router.post("/users/invite", tags=["users"])
def invite_user_endpoint(req: UserCreateRequest, auth: AuthContext = Depends(require_admin)):
    from app.security.invite import create_invite, send_invite_email, has_pending_invite, cancel_invite, get_invite_by_email
    from app.security.users import create_user as _create
    from app.security.rbac import assign_role, PROTECTED_ROLES
    if req.role in PROTECTED_ROLES and auth.role != "super_admin":
        raise HTTPException(status_code=403, detail=f"Only a super_admin can invite users with role '{req.role}'")
    email = (req.email or "").strip().lower() or req.username + "@company.com"
    from app.security.users import find_user_by_email, user_exists, _users
    existing_user = find_user_by_email(email)
    if existing_user:
        existing_info = _users.get(existing_user, {})
        pw = existing_info.get("password_hash", "")
        is_active = pw != "INVITE_PENDING" and pw
        if is_active:
            raise HTTPException(
                status_code=409,
                detail=f"Email '{email}' is already linked to the active account '{existing_user}'. No invite needed."
            )
    if user_exists(req.username):
        existing_info = _users.get(req.username.strip().lower(), {})
        pw = existing_info.get("password_hash", "")
        if pw and pw != "INVITE_PENDING":
            raise HTTPException(
                status_code=409,
                detail=f"User '{req.username}' already has an active account."
            )
    existing_inv = get_invite_by_email(email)
    if existing_inv:
        cancel_invite(existing_inv["full_token"])
    result = _create(req.username, "INVITE_PENDING", created_by=auth.username)
    if not result["success"] and "already" not in result.get("error", "").lower():
        raise HTTPException(status_code=400, detail=result["error"])
    assign_role(req.username, req.role, changed_by=auth.username)
    invite = create_invite(req.username, email, role=req.role)
    email_result = send_invite_email(email, req.username, invite["otp"], invite["token"])
    email_sent = isinstance(email_result, dict) and email_result.get("success") is True
    app_url = os.getenv("APP_URL", "http://localhost:8000")
    return {"success": True, "username": req.username,
            "email_sent": email_sent,
            "expires_in_hours": 48,
            "setup_link": f"{app_url}/auth/setup-password?token={invite['token']}"}


@router.get("/users/invites", tags=["users"])
def list_invites_endpoint(auth: AuthContext = Depends(require_admin)):
    from app.security.invite import list_pending_invites
    return {"invites": list_pending_invites()}


@router.delete("/users/invites/{token}", tags=["users"])
def cancel_invite_endpoint(token: str, auth: AuthContext = Depends(require_admin)):
    from app.security.invite import cancel_invite
    ok = cancel_invite(token)
    if not ok:
        raise HTTPException(status_code=404, detail="Invite not found or already expired")
    return {"success": True}


@router.get("/auth/setup-password", response_class=HTMLResponse, include_in_schema=False)
def setup_password_page(token: str = ""):
    """Password setup page for invited users."""
    if not token:
        return HTMLResponse("<h2>Invalid link — no token provided.</h2>", status_code=400)
    from app.security.invite import get_invite_username
    username = get_invite_username(token)
    if not username:
        return HTMLResponse("""<!DOCTYPE html><html><head><meta charset="UTF-8"/>
<title>Expired Link</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:Inter,sans-serif;background:#04060f;color:#e2e8f0;min-height:100vh;display:flex;align-items:center;justify-content:center}</style>
</head><body><div style="text-align:center;padding:40px">
<div style="font-size:3em;margin-bottom:16px">&#x274C;</div>
<h2 style="font-size:1.3em;margin-bottom:8px">Link Expired or Invalid</h2>
<p style="color:#4f6a9a">This invite link has expired or already been used.<br>Ask your admin to send a new invite.</p>
</div></body></html>""", status_code=400)
    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Set Your Password — NsOps</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>
  <style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{font-family:Inter,sans-serif;background:#04060f;color:#e2e8f0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}}
    @keyframes in{{from{{opacity:0;transform:translateY(20px)}}to{{opacity:1;transform:translateY(0)}}}}
    .orb1{{position:fixed;width:400px;height:400px;border-radius:50%;background:radial-gradient(circle,rgba(124,58,237,.2) 0%,transparent 70%);top:-80px;right:-60px;pointer-events:none}}
    .orb2{{position:fixed;width:350px;height:350px;border-radius:50%;background:radial-gradient(circle,rgba(37,99,235,.15) 0%,transparent 70%);bottom:-60px;left:-40px;pointer-events:none}}
    .card{{position:relative;z-index:1;background:rgba(13,20,36,.9);backdrop-filter:blur(20px);border:1px solid rgba(79,142,247,.2);border-radius:16px;padding:36px;width:100%;max-width:420px;animation:in .4s ease both}}
    .logo{{display:flex;align-items:center;gap:10px;margin-bottom:28px}}
    .logo-icon{{width:40px;height:40px;background:linear-gradient(135deg,#7c3aed,#2563eb);border-radius:10px;display:flex;align-items:center;justify-content:center}}
    .logo-text{{font-size:1.1em;font-weight:800;letter-spacing:-.02em}}
    h2{{font-size:1.25em;font-weight:700;margin-bottom:4px}}
    .sub{{font-size:.83em;color:#4f6a9a;margin-bottom:24px}}
    .field{{margin-bottom:16px}}
    label{{display:block;font-size:.76em;font-weight:600;color:#4f8ef7;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px}}
    input{{width:100%;padding:11px 14px;border-radius:8px;border:1px solid rgba(79,142,247,.2);background:rgba(4,6,15,.6);color:#e2e8f0;font-size:.9em;font-family:inherit;transition:border-color .15s,box-shadow .15s;outline:none}}
    input:focus{{border-color:#4f8ef7;box-shadow:0 0 0 3px rgba(79,142,247,.15)}}
    .err{{display:none;color:#fca5a5;font-size:.82em;padding:10px 14px;background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.25);border-radius:8px;margin-bottom:14px}}
    .btn{{width:100%;padding:12px;border-radius:8px;border:none;background:linear-gradient(135deg,#7c3aed,#2563eb);color:#fff;font-weight:700;font-size:.95em;cursor:pointer;font-family:inherit;transition:all .15s;margin-top:8px}}
    .btn:hover:not(:disabled){{filter:brightness(1.1);box-shadow:0 4px 20px rgba(124,58,237,.4);transform:translateY(-1px)}}
    .btn:disabled{{opacity:.6;cursor:not-allowed}}
    .req{{font-size:.75em;color:#3d5080;margin-top:4px}}
    .success{{display:none;text-align:center;padding:20px 0}}
    .success .check{{font-size:3em;margin-bottom:12px}}
    .success h3{{font-size:1.1em;font-weight:700;margin-bottom:6px}}
    .success p{{font-size:.85em;color:#4f6a9a}}
  </style>
</head>
<body>
  <div class="orb1"></div>
  <div class="orb2"></div>
  <div class="card">
    <div class="logo">
      <div class="logo-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
      </div>
      <span class="logo-text">NsOps</span>
    </div>
    <h2>Set Your Password</h2>
    <p class="sub">Welcome, <strong>{username}</strong>. Enter the OTP from your invite email and choose a password.</p>
    <div id="err" class="err"></div>
    <div class="field">
      <label>One-Time Password (OTP)</label>
      <input id="otp" type="text" placeholder="6-digit code from email" maxlength="6" inputmode="numeric" autocomplete="one-time-code"/>
    </div>
    <div class="field">
      <label>New Password</label>
      <input id="pw1" type="password" placeholder="Choose a strong password" autocomplete="new-password"/>
      <div class="req">Min 8 characters</div>
    </div>
    <div class="field">
      <label>Confirm Password</label>
      <input id="pw2" type="password" placeholder="Repeat your password" autocomplete="new-password"/>
    </div>
    <button class="btn" id="btn" onclick="submit()">Activate Account</button>
    <div class="success" id="success">
      <div class="check">&#x2705;</div>
      <h3>Password set!</h3>
      <p>Your account is ready. <a href="/" style="color:#7c3aed;font-weight:600">Sign in now &rarr;</a></p>
    </div>
  </div>
  <script>
    var TOKEN = '{token}';
    function submit() {{
      var otp = document.getElementById('otp').value.trim();
      var pw1 = document.getElementById('pw1').value;
      var pw2 = document.getElementById('pw2').value;
      var err = document.getElementById('err');
      var btn = document.getElementById('btn');
      err.style.display = 'none';
      if (!otp || otp.length < 6) {{ err.textContent = 'Enter the 6-digit OTP from your email'; err.style.display='block'; return; }}
      if (pw1.length < 8) {{ err.textContent = 'Password must be at least 8 characters'; err.style.display='block'; return; }}
      if (pw1 !== pw2) {{ err.textContent = 'Passwords do not match'; err.style.display='block'; return; }}
      btn.disabled = true; btn.textContent = 'Activating...';
      fetch('/auth/setup-password', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{token: TOKEN, otp: otp, new_password: pw1}})
      }}).then(function(r){{ return r.json(); }}).then(function(d) {{
        btn.disabled = false; btn.textContent = 'Activate Account';
        if (d.success) {{
          document.getElementById('success').style.display = 'block';
          btn.style.display = 'none';
          document.querySelectorAll('.field').forEach(function(f){{ f.style.display='none'; }});
          document.getElementById('err').style.display = 'none';
        }} else {{
          err.textContent = d.detail || d.error || 'Invalid OTP or link expired';
          err.style.display = 'block';
        }}
      }}).catch(function(){{ btn.disabled=false; btn.textContent='Activate Account'; err.textContent='Network error'; err.style.display='block'; }});
    }}
    document.addEventListener('keydown', function(e){{ if(e.key==='Enter') submit(); }});
  </script>
</body>
</html>""")


@router.post("/auth/setup-password", tags=["auth"])
def setup_password(req: SetupPasswordRequest):
    """Complete account setup: validate OTP, set password, consume invite token."""
    from app.security.invite import validate_invite, consume_invite
    from app.security.users import change_password
    result = validate_invite(req.token, req.otp)
    if not result["valid"]:
        raise HTTPException(status_code=400, detail=result["error"])
    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    username = result["username"]
    pw_result = change_password(username, req.new_password)
    if not pw_result.get("success"):
        raise HTTPException(status_code=400, detail=pw_result.get("error", "Failed to set password"))
    consume_invite(req.token)
    return {"success": True, "username": username, "message": "Password set. You can now sign in."}


@router.post("/auth/configure-smtp", tags=["auth"])
def configure_smtp(req: SmtpConfigRequest, auth: AuthContext = Depends(require_admin)):
    """Save SMTP settings to .env and test the connection. Admin only."""
    import smtplib
    updates = {}
    if req.smtp_host:     updates["SMTP_HOST"]     = req.smtp_host
    if req.smtp_user:     updates["SMTP_USER"]     = req.smtp_user
    if req.smtp_password: updates["SMTP_PASSWORD"] = req.smtp_password
    if req.smtp_from or req.smtp_user:
        updates["SMTP_FROM"] = req.smtp_from or req.smtp_user
    updates["SMTP_PORT"] = str(req.smtp_port)
    if req.app_url:       updates["APP_URL"]       = req.app_url
    _write_env(updates)
    if req.smtp_host and req.smtp_user and req.smtp_password:
        try:
            with smtplib.SMTP(req.smtp_host, req.smtp_port, timeout=8) as s:
                s.ehlo(); s.starttls(); s.login(req.smtp_user, req.smtp_password)
            return {"success": True, "message": "SMTP configured and connection verified"}
        except Exception as e:
            return {"success": False, "message": f"Settings saved but SMTP test failed: {e}"}
    return {"success": True, "message": "SMTP settings saved (no test — fill all fields to verify)"}


@router.post("/auth/test-email", tags=["auth"])
def test_email(auth: AuthContext = Depends(require_admin)):
    """Send a test email to the configured SMTP_USER address."""
    from app.security.invite import send_invite_email
    to = os.getenv("SMTP_USER", "")
    if not to or "@" not in to:
        raise HTTPException(400, detail="SMTP_USER not configured — set it in .env or via /auth/configure-smtp")
    result = send_invite_email(to, auth.username, "123456", "test-token")
    if result.get("success"):
        return {"success": True, "message": f"Test email sent to {to}"}
    raise HTTPException(400, detail=result.get("error", "Failed to send test email"))
