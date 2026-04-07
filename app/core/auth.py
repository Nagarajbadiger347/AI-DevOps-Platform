"""JWT authentication middleware for production use.

Flow:
  POST /auth/token  { username, password }  →  { access_token, token_type }
  All protected endpoints: Authorization: Bearer <token>

Configuration (.env):
  JWT_SECRET_KEY        — random 32-byte hex string (openssl rand -hex 32)
  JWT_SECRET_KEY_OLD    — previous key kept during rotation (tokens signed with
                          old key still accepted for JWT_EXPIRE_MINS after rotation)
  JWT_ALGORITHM         — HS256 (default)
  JWT_EXPIRE_MINS       — token lifetime in minutes (default 480 = 8 hours)
  AUTH_ENABLED          — set to "false" to disable in dev (default true)
  SSO_PROVIDER          — "google" | "github" | "" (empty = disabled)
  SSO_CLIENT_ID         — OAuth2 client ID
  SSO_CLIENT_SECRET     — OAuth2 client secret
  SSO_REDIRECT_URI      — e.g. https://your-domain.com/auth/sso/callback
  SSO_DEFAULT_ROLE      — role assigned to new SSO users (default: viewer)
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Optional

from fastapi import Depends, HTTPException, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

try:
    from jose import JWTError, jwt as _jwt
    _JOSE_AVAILABLE = True
except ImportError:
    _JOSE_AVAILABLE = False

SECRET_KEY     = os.getenv("JWT_SECRET_KEY", "change-me-in-production-use-openssl-rand-hex-32")
SECRET_KEY_OLD = os.getenv("JWT_SECRET_KEY_OLD", "")   # kept during key rotation
ALGORITHM      = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRE_MINS    = int(os.getenv("JWT_EXPIRE_MINS", "480"))
AUTH_ENABLED   = os.getenv("AUTH_ENABLED", "true").lower() != "false"

if os.getenv("ENVIRONMENT", "development").lower() == "production":
    if SECRET_KEY == "change-me-in-production-use-openssl-rand-hex-32":
        raise RuntimeError("JWT_SECRET_KEY must be set in production. Run: openssl rand -hex 32")

if not AUTH_ENABLED:
    import warnings
    warnings.warn("AUTH_ENABLED=false — authentication is disabled.", stacklevel=1)

_bearer = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Token blacklist — Redis-backed with in-memory fallback
# ---------------------------------------------------------------------------
_blacklist_memory: set[str] = set()

def _token_fingerprint(token: str) -> str:
    """Short hash of a token used as the blacklist key."""
    return hashlib.sha256(token.encode()).hexdigest()[:32]

def blacklist_token(token: str, expires_in: int = EXPIRE_MINS * 60) -> None:
    """Add a token to the blacklist so it is rejected even if not yet expired."""
    fp = _token_fingerprint(token)
    try:
        from app.core.ratelimit import _get_redis
        r = _get_redis()
        if r:
            r.setex(f"bl:{fp}", expires_in, "1")
            return
    except Exception:
        pass
    _blacklist_memory.add(fp)

def is_blacklisted(token: str) -> bool:
    """Return True if the token has been revoked."""
    fp = _token_fingerprint(token)
    try:
        from app.core.ratelimit import _get_redis
        r = _get_redis()
        if r:
            return bool(r.exists(f"bl:{fp}"))
    except Exception:
        pass
    return fp in _blacklist_memory

def revoke_all_user_tokens(username: str) -> None:
    """Store a per-user revocation timestamp. Tokens issued before this time are rejected."""
    ts = int(time.time())
    try:
        from app.core.ratelimit import _get_redis
        r = _get_redis()
        if r:
            r.set(f"revoke_before:{username}", ts, ex=EXPIRE_MINS * 60 + 60)
            return
    except Exception:
        pass
    # In-memory fallback stored in module-level dict
    _user_revoke_before[username] = ts

_user_revoke_before: dict[str, int] = {}

def _is_user_revoked(username: str, issued_at: int) -> bool:
    """Return True if the token was issued before a global revocation for the user."""
    try:
        from app.core.ratelimit import _get_redis
        r = _get_redis()
        if r:
            val = r.get(f"revoke_before:{username}")
            if val and issued_at < int(val):
                return True
    except Exception:
        pass
    revoke_ts = _user_revoke_before.get(username, 0)
    return issued_at < revoke_ts


# ---------------------------------------------------------------------------
# Token creation / decoding
# ---------------------------------------------------------------------------

def create_token(username: str, role: str) -> str:
    """Create a signed JWT for the given user."""
    if not _JOSE_AVAILABLE:
        import base64, hmac
        exp = int(time.time()) + EXPIRE_MINS * 60
        iat = int(time.time())
        payload = f"{username}:{role}:{exp}:{iat}"
        sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()[:16]
        return base64.b64encode(f"{payload}:{sig}".encode()).decode()

    payload = {
        "sub": username,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + EXPIRE_MINS * 60,
    }
    return _jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT. Returns payload dict or raises HTTPException.

    During key rotation, tries SECRET_KEY first, then SECRET_KEY_OLD.
    """
    if is_blacklisted(token):
        raise HTTPException(status_code=401, detail="Token has been revoked")

    if not _JOSE_AVAILABLE:
        import base64, hmac
        try:
            decoded = base64.b64decode(token.encode()).decode()
            parts = decoded.split(":")
            if len(parts) < 4:
                raise HTTPException(status_code=401, detail="Invalid token")
            username, role, exp_str, iat_str, sig = parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else "0", parts[4] if len(parts) > 4 else parts[3]
            if int(exp_str) < time.time():
                raise HTTPException(status_code=401, detail="Token has expired")
            expected_payload = f"{username}:{role}:{exp_str}:{iat_str}"
            expected_sig = hmac.new(SECRET_KEY.encode(), expected_payload.encode(), hashlib.sha256).hexdigest()[:16]
            if not hmac.compare_digest(sig, expected_sig):
                raise HTTPException(status_code=401, detail="Invalid token signature")
            if _is_user_revoked(username, int(iat_str)):
                raise HTTPException(status_code=401, detail="Token has been revoked")
            return {"sub": username, "role": role, "iat": int(iat_str)}
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid token")

    # Try primary key, then old key (rotation grace period)
    keys_to_try = [SECRET_KEY]
    if SECRET_KEY_OLD:
        keys_to_try.append(SECRET_KEY_OLD)

    last_error = None
    for key in keys_to_try:
        try:
            payload = _jwt.decode(token, key, algorithms=[ALGORITHM])
            if payload.get("exp", 0) < time.time():
                raise HTTPException(status_code=401, detail="Token has expired")
            if _is_user_revoked(payload.get("sub", ""), payload.get("iat", 0)):
                raise HTTPException(status_code=401, detail="Token has been revoked")
            return payload
        except HTTPException:
            raise
        except JWTError as e:
            last_error = e
            continue

    raise HTTPException(status_code=401, detail=f"Invalid token: {last_error}")


# ---------------------------------------------------------------------------
# SSO / OAuth2 helpers
# ---------------------------------------------------------------------------

SSO_PROVIDER     = os.getenv("SSO_PROVIDER", "").lower()       # google | github
SSO_CLIENT_ID    = os.getenv("SSO_CLIENT_ID", "")
SSO_CLIENT_SECRET = os.getenv("SSO_CLIENT_SECRET", "")
SSO_REDIRECT_URI = os.getenv("SSO_REDIRECT_URI", "")
SSO_DEFAULT_ROLE = os.getenv("SSO_DEFAULT_ROLE", "viewer")

_SSO_CONFIGS = {
    "google": {
        "auth_url":    "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url":   "https://oauth2.googleapis.com/token",
        "userinfo_url":"https://www.googleapis.com/oauth2/v3/userinfo",
        "scope":       "openid email profile",
    },
    "github": {
        "auth_url":    "https://github.com/login/oauth/authorize",
        "token_url":   "https://github.com/login/oauth/access_token",
        "userinfo_url":"https://api.github.com/user",
        "scope":       "read:user user:email",
    },
}

def get_sso_login_url(state: str = "") -> str | None:
    """Return the OAuth2 authorization URL or None if SSO is not configured."""
    cfg = _SSO_CONFIGS.get(SSO_PROVIDER)
    if not cfg or not SSO_CLIENT_ID:
        return None
    import urllib.parse
    params = {
        "client_id":     SSO_CLIENT_ID,
        "redirect_uri":  SSO_REDIRECT_URI,
        "scope":         cfg["scope"],
        "response_type": "code",
        "state":         state,
    }
    return cfg["auth_url"] + "?" + urllib.parse.urlencode(params)

def exchange_sso_code(code: str) -> dict:
    """Exchange OAuth2 authorization code for user info.

    Returns dict with: username, email, display_name
    Raises HTTPException on failure.
    """
    cfg = _SSO_CONFIGS.get(SSO_PROVIDER)
    if not cfg:
        raise HTTPException(status_code=400, detail="SSO not configured")

    import urllib.request, urllib.parse, json as _json

    # 1. Exchange code for access token
    token_data = urllib.parse.urlencode({
        "client_id":     SSO_CLIENT_ID,
        "client_secret": SSO_CLIENT_SECRET,
        "code":          code,
        "redirect_uri":  SSO_REDIRECT_URI,
        "grant_type":    "authorization_code",
    }).encode()
    headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
    try:
        req = urllib.request.Request(cfg["token_url"], data=token_data, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            token_resp = _json.loads(resp.read())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"SSO token exchange failed: {e}")

    access_token = token_resp.get("access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail=f"SSO auth failed: {token_resp.get('error_description', 'no access token')}")

    # 2. Fetch user info
    try:
        req = urllib.request.Request(
            cfg["userinfo_url"],
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            user_info = _json.loads(resp.read())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"SSO user info failed: {e}")

    # Normalize across providers
    if SSO_PROVIDER == "google":
        email = user_info.get("email", "")
        username = email.split("@")[0].replace(".", "_").lower()
        display_name = user_info.get("name", username)
    elif SSO_PROVIDER == "github":
        username = (user_info.get("login") or "").lower()
        email = user_info.get("email") or f"{username}@github.com"
        display_name = user_info.get("name") or username
    else:
        raise HTTPException(status_code=400, detail="Unknown SSO provider")

    if not username:
        raise HTTPException(status_code=401, detail="Could not determine username from SSO provider")

    return {"username": username, "email": email, "display_name": display_name}


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

class AuthContext:
    def __init__(self, username: str, role: str):
        self.username = username
        self.role     = role

    def __repr__(self):
        return f"AuthContext(user={self.username}, role={self.role})"


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    x_user: str = Header(default=""),
) -> AuthContext:
    if not AUTH_ENABLED:
        if credentials and credentials.credentials:
            try:
                payload  = decode_token(credentials.credentials)
                username = payload.get("sub", "anonymous")
                role     = payload.get("role", "viewer")
                return AuthContext(username=username, role=role)
            except Exception:
                pass
        username = x_user or "anonymous"
        try:
            from app.security.rbac import get_user_role
            role = get_user_role(username)
        except Exception:
            role = "viewer"
        return AuthContext(username=username, role=role)

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Pass Authorization: Bearer <token>.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload  = decode_token(credentials.credentials)
    username = payload.get("sub", "unknown")
    role     = payload.get("role", "viewer")
    return AuthContext(username=username, role=role)


def require_role(*allowed_roles: str):
    def _dep(auth: AuthContext = Depends(get_current_user)) -> AuthContext:
        if auth.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{auth.role}' is not allowed. Required: {list(allowed_roles)}",
            )
        return auth
    return _dep


require_admin     = require_role("admin")
require_developer = require_role("admin", "developer")
require_viewer    = require_role("admin", "developer", "viewer")
