"""User store — hashed passwords + role management.

Users and their hashed passwords are persisted to USERS_CONFIG_PATH (JSON).
Roles are stored separately in rbac.py (roles.json).

First-run bootstrap:
  If no users exist, creates an initial admin from env vars:
    ADMIN_USERNAME  (default: admin)
    ADMIN_PASSWORD  (required — app will refuse to start if not set in production)

Format of users.json:
    {
      "alice": {"password_hash": "<bcrypt hash>", "created_at": "...", "created_by": "system"},
      "bob":   {"password_hash": "<bcrypt hash>", "created_at": "...", "created_by": "alice"}
    }
"""

import json
import os
import hashlib
import hmac
import secrets
import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_USERS_PATH = Path(
    os.getenv("USERS_CONFIG_PATH", "") or
    Path(__file__).resolve().parent / "users.json"
)

# In-memory store: username → {password_hash, created_at, created_by}
_users: dict[str, dict] = {}


# ── Password hashing (bcrypt with HMAC-SHA256 legacy fallback) ────────────────

try:
    import bcrypt as _bcrypt
    _BCRYPT_AVAILABLE = True
except ImportError:
    _bcrypt = None  # type: ignore
    _BCRYPT_AVAILABLE = False


def _app_secret() -> bytes:
    # Use APP_SECRET_KEY (dedicated to password hashing) so that rotating
    # JWT_SECRET_KEY (for token signing) does NOT invalidate password hashes.
    # Falls back to a hardcoded stable value — NOT JWT_SECRET_KEY.
    secret = os.getenv("APP_SECRET_KEY", "nexusops-pw-hmac-stable-fallback-v1")
    return secret.encode()


def _is_legacy_hash(stored_hash: str) -> bool:
    """Return True if the hash is in the old HMAC-SHA256 format (salt:hex)."""
    # bcrypt hashes start with $2b$ or $2a$; legacy format is salt:hexdigest
    return stored_hash.startswith("$2") is False and ":" in stored_hash


def hash_password(password: str) -> str:
    """Hash a password using bcrypt (preferred) or HMAC-SHA256 fallback."""
    if _BCRYPT_AVAILABLE:
        hashed = _bcrypt.hashpw(password.encode(), _bcrypt.gensalt())
        return hashed.decode()
    # Fallback: HMAC-SHA256 with salt
    salt = secrets.token_hex(16)
    h = hmac.new(_app_secret() + salt.encode(), password.encode(), hashlib.sha256).hexdigest()
    return f"{salt}:{h}"


def _verify_legacy(password: str, stored_hash: str) -> bool:
    """Verify a password against the old HMAC-SHA256 hash format."""
    try:
        salt, h = stored_hash.split(":", 1)
        expected = hmac.new(_app_secret() + salt.encode(), password.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(h, expected)
    except Exception:
        return False


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password.

    Handles both bcrypt hashes and legacy HMAC-SHA256 hashes.
    Returns True if the password matches.
    """
    try:
        if _is_legacy_hash(stored_hash):
            return _verify_legacy(password, stored_hash)
        if _BCRYPT_AVAILABLE:
            return _bcrypt.checkpw(password.encode(), stored_hash.encode())
        return False
    except Exception:
        return False


# ── Persistence ────────────────────────────────────────────────────────────────

def _load() -> None:
    try:
        with open(_USERS_PATH) as f:
            data = json.load(f)
        if isinstance(data, dict):
            _users.update(data)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

def _save() -> None:
    try:
        _USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _USERS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(_users, indent=2))
        tmp.replace(_USERS_PATH)
    except Exception:
        pass


# ── User operations ────────────────────────────────────────────────────────────

def find_user_by_email(email: str) -> str | None:
    """Return the username of the user with this email, or None."""
    email = email.strip().lower()
    for uname, info in _users.items():
        if info.get("email", "").lower() == email:
            return uname
    return None


def create_user(username: str, password: str | None, created_by: str = "system", email: str = "") -> dict:
    """Create a new user. Pass password=None for SSO-only accounts (no local login)."""
    username = username.strip().lower()
    if not username or len(username) < 2:
        return {"success": False, "error": "Username must be at least 2 characters"}
    if password is not None and password not in ("INVITE_PENDING",) and len(password) < 8:
        return {"success": False, "error": "Password must be at least 8 characters"}
    if username in _users:
        return {"success": False, "error": f"User '{username}' already exists"}
    _users[username] = {
        "password_hash": hash_password(password) if password else "SSO_ONLY",
        "created_at":    datetime.datetime.utcnow().isoformat(),
        "created_by":    created_by,
        "sso_only":      password is None,
        "email":         email.strip().lower(),
    }
    _save()
    return {"success": True, "username": username}


def change_password(username: str, new_password: str) -> dict:
    """Update a user's password."""
    username = username.strip().lower()
    if username not in _users:
        return {"success": False, "error": f"User '{username}' not found"}
    if len(new_password) < 8:
        return {"success": False, "error": "Password must be at least 8 characters"}
    _users[username]["password_hash"] = hash_password(new_password)
    _save()
    return {"success": True, "username": username}


def delete_user(username: str) -> dict:
    """Delete a user account."""
    username = username.strip().lower()
    if username not in _users:
        return {"success": False, "error": f"User '{username}' not found"}
    del _users[username]
    _save()
    return {"success": True, "username": username}


def authenticate(username: str, password: str) -> bool:
    """Return True if username + password are valid.

    If the stored hash uses the legacy HMAC-SHA256 format and bcrypt is
    available, the password is re-hashed with bcrypt on successful login.
    """
    username = username.strip().lower()
    user = _users.get(username)
    if not user:
        return False
    if user.get("sso_only") or user.get("password_hash") == "SSO_ONLY":
        return False  # SSO-only account — must login via SSO
    stored = user["password_hash"]
    if not verify_password(password, stored):
        return False
    # Upgrade legacy hash to bcrypt on successful login
    if _BCRYPT_AVAILABLE and _is_legacy_hash(stored):
        user["password_hash"] = hash_password(password)
        _save()
    return True


def user_exists(username: str) -> bool:
    return username.strip().lower() in _users


def list_users() -> list[dict]:
    """List all users (no password hashes)."""
    from app.security.rbac import _user_roles
    result = []
    for username, info in _users.items():
        result.append({
            "username":   username,
            "role":       _user_roles.get(username, "no role"),
            "email":      info.get("email", ""),
            "created_at": info.get("created_at", ""),
            "created_by": info.get("created_by", ""),
        })
    return sorted(result, key=lambda u: u["username"])


# ── Bootstrap ──────────────────────────────────────────────────────────────────

def _bootstrap() -> None:
    """On first run (or when ADMIN_PASSWORD is set), ensure the admin account exists."""
    _load()
    admin_username = os.getenv("ADMIN_USERNAME", "admin").strip().lower()
    admin_password = os.getenv("ADMIN_PASSWORD", "").strip()

    if not _users:
        # First run — create initial admin
        if not admin_password:
            admin_password = secrets.token_urlsafe(16)
            print(f"\n{'='*60}")
            print(f"  FIRST RUN — Initial admin account created:")
            print(f"  Username: {admin_username}")
            print(f"  Password: {admin_password}")
            print(f"  Change this password immediately after first login.")
            print(f"  Set ADMIN_PASSWORD in .env to use a fixed password.")
            print(f"{'='*60}\n")
        create_user(admin_username, admin_password, created_by="system")
        from app.security.rbac import assign_role
        assign_role(admin_username, "admin")

    elif admin_password and admin_username in _users:
        # If ADMIN_PASSWORD is set in env, keep the stored hash in sync
        if not verify_password(admin_password, _users[admin_username]["password_hash"]):
            change_password(admin_username, admin_password)


_bootstrap()
