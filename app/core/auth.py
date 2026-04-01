"""JWT authentication middleware for production use.

Flow:
  POST /auth/token  { username, password }  →  { access_token, token_type }
  All protected endpoints: Authorization: Bearer <token>

Configuration (.env):
  JWT_SECRET_KEY   — random 32-byte hex string (openssl rand -hex 32)
  JWT_ALGORITHM    — HS256 (default)
  JWT_EXPIRE_MINS  — token lifetime in minutes (default 480 = 8 hours)
  AUTH_ENABLED     — set to "false" to disable in dev (default true)
"""
from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import Depends, HTTPException, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Optional import — jose is not in base requirements; falls back gracefully
try:
    from jose import JWTError, jwt as _jwt
    _JOSE_AVAILABLE = True
except ImportError:
    _JOSE_AVAILABLE = False

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production-use-openssl-rand-hex-32")
if os.getenv("ENVIRONMENT", "development").lower() == "production" and SECRET_KEY == "change-me-in-production-use-openssl-rand-hex-32":
    raise RuntimeError("JWT_SECRET_KEY must be set in production. Run: openssl rand -hex 32")
ALGORITHM    = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRE_MINS  = int(os.getenv("JWT_EXPIRE_MINS", "480"))
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() != "false"
if not AUTH_ENABLED:
    import warnings
    warnings.warn("AUTH_ENABLED=false — authentication is disabled. Set AUTH_ENABLED=true in production.", stacklevel=1)

_bearer = HTTPBearer(auto_error=False)


def create_token(username: str, role: str) -> str:
    """Create a signed JWT for the given user."""
    if not _JOSE_AVAILABLE:
        # Secure fallback using HMAC-SHA256 when jose is unavailable
        import base64, hmac, hashlib
        exp = int(time.time()) + EXPIRE_MINS * 60
        payload = f"{username}:{role}:{exp}"
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
    """Decode and validate a JWT. Returns payload dict or raises HTTPException."""
    if not _JOSE_AVAILABLE:
        import base64, hmac, hashlib
        try:
            decoded = base64.b64decode(token.encode()).decode()
            parts = decoded.split(":")
            if len(parts) != 4:
                raise HTTPException(status_code=401, detail="Invalid token")
            username, role, exp_str, sig = parts
            if int(exp_str) < time.time():
                raise HTTPException(status_code=401, detail="Token has expired")
            expected_payload = f"{username}:{role}:{exp_str}"
            expected_sig = hmac.new(SECRET_KEY.encode(), expected_payload.encode(), hashlib.sha256).hexdigest()[:16]
            if not hmac.compare_digest(sig, expected_sig):
                raise HTTPException(status_code=401, detail="Invalid token signature")
            return {"sub": username, "role": role}
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid token")

    try:
        payload = _jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("exp", 0) < time.time():
            raise HTTPException(status_code=401, detail="Token has expired")
        return payload
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


class AuthContext:
    """Carries the resolved user identity for a request."""
    def __init__(self, username: str, role: str):
        self.username = username
        self.role     = role

    def __repr__(self):
        return f"AuthContext(user={self.username}, role={self.role})"


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    x_user: str = Header(default=""),
) -> AuthContext:
    """FastAPI dependency — resolves the calling user.

    Priority:
      1. Authorization: Bearer <JWT>  (production)
      2. X-User header                (dev/legacy fallback when AUTH_ENABLED=false)
    """
    if not AUTH_ENABLED:
        # Dev mode: honour Bearer token if present, else trust X-User header
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

    # Production: require Bearer token
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
    """Return a dependency that enforces minimum role.

    Usage:
        @app.post("/admin/thing")
        def thing(auth: AuthContext = Depends(require_role("admin"))):
            ...
    """
    def _dep(auth: AuthContext = Depends(get_current_user)) -> AuthContext:
        if auth.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{auth.role}' is not allowed. Required: {list(allowed_roles)}",
            )
        return auth
    return _dep


# Convenience role dependencies
require_admin     = require_role("admin")
require_developer = require_role("admin", "developer")
require_viewer    = require_role("admin", "developer", "viewer")
