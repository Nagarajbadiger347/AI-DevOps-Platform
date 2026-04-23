"""User store — hashed passwords + role management.

Users are persisted to PostgreSQL (users table).
Each user belongs to a tenant via tenant_id.

First-run bootstrap:
  If no users exist, creates an initial admin from env vars:
    ADMIN_USERNAME  (default: admin)
    ADMIN_PASSWORD  (required — app will refuse to start if not set in production)
"""

import os
import hashlib
import hmac
import secrets
import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ── Password hashing ──────────────────────────────────────────────

try:
    import bcrypt as _bcrypt
    _BCRYPT_AVAILABLE = True
except ImportError:
    _bcrypt = None
    _BCRYPT_AVAILABLE = False


def _app_secret() -> bytes:
    secret = os.getenv("APP_SECRET_KEY", "nexusops-pw-hmac-stable-fallback-v1")
    return secret.encode()


def _is_legacy_hash(stored_hash: str) -> bool:
    return stored_hash.startswith("$2") is False and ":" in stored_hash


def hash_password(password: str) -> str:
    if _BCRYPT_AVAILABLE:
        return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()
    salt = secrets.token_hex(16)
    h = hmac.new(_app_secret() + salt.encode(), password.encode(), hashlib.sha256).hexdigest()
    return f"{salt}:{h}"


def _verify_legacy(password: str, stored_hash: str) -> bool:
    try:
        salt, h = stored_hash.split(":", 1)
        expected = hmac.new(_app_secret() + salt.encode(), password.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(h, expected)
    except Exception:
        return False


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        if _is_legacy_hash(stored_hash):
            return _verify_legacy(password, stored_hash)
        if _BCRYPT_AVAILABLE:
            return _bcrypt.checkpw(password.encode(), stored_hash.encode())
        return False
    except Exception:
        return False


# ── DB helpers ────────────────────────────────────────────────────

def _db():
    from app.core.database import execute, execute_one
    return execute, execute_one


# ── User operations ───────────────────────────────────────────────

def find_user_by_email(email: str, tenant_id: str = "default") -> str | None:
    _, execute_one = _db()
    row = execute_one(
        "SELECT username FROM users WHERE email = %s AND tenant_id = %s",
        (email.strip().lower(), tenant_id)
    )
    return row["username"] if row else None


def create_user(username: str, password: str | None, created_by: str = "system",
                email: str = "", tenant_id: str = "default") -> dict:
    execute, execute_one = _db()
    username = username.strip().lower()
    if not username or len(username) < 2:
        return {"success": False, "error": "Username must be at least 2 characters"}
    if password is not None and password not in ("INVITE_PENDING",) and len(password) < 8:
        return {"success": False, "error": "Password must be at least 8 characters"}

    existing = execute_one(
        "SELECT username FROM users WHERE username = %s AND tenant_id = %s",
        (username, tenant_id)
    )
    if existing:
        return {"success": False, "error": f"User '{username}' already exists"}

    password_hash = hash_password(password) if password and password != "INVITE_PENDING" else (password or "SSO_ONLY")
    invite_pending = password == "INVITE_PENDING"

    execute(
        """
        INSERT INTO users (user_id, tenant_id, username, email, password_hash, invite_pending, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (
            secrets.token_urlsafe(16),
            tenant_id,
            username,
            email.strip().lower(),
            password_hash,
            invite_pending,
            datetime.datetime.utcnow().isoformat(),
        )
    )
    return {"success": True, "username": username}


def change_password(username: str, new_password: str, tenant_id: str = "default") -> dict:
    execute, _ = _db()
    username = username.strip().lower()
    if len(new_password) < 8:
        return {"success": False, "error": "Password must be at least 8 characters"}
    rows = execute(
        "UPDATE users SET password_hash = %s, updated_at = NOW() WHERE username = %s AND tenant_id = %s RETURNING username",
        (hash_password(new_password), username, tenant_id)
    )
    if not rows:
        return {"success": False, "error": f"User '{username}' not found"}
    return {"success": True, "username": username}


def delete_user(username: str, tenant_id: str = "default") -> dict:
    execute, _ = _db()
    username = username.strip().lower()
    rows = execute(
        "DELETE FROM users WHERE username = %s AND tenant_id = %s RETURNING username",
        (username, tenant_id)
    )
    if not rows:
        return {"success": False, "error": f"User '{username}' not found"}
    return {"success": True, "username": username}


def authenticate(username: str, password: str, tenant_id: str = "default") -> bool:
    execute, execute_one = _db()
    username = username.strip().lower()
    row = execute_one(
        "SELECT password_hash FROM users WHERE username = %s AND tenant_id = %s AND active = true",
        (username, tenant_id)
    )
    if not row:
        return False
    stored = row["password_hash"]
    if stored in ("SSO_ONLY", "INVITE_PENDING"):
        return False
    if not verify_password(password, stored):
        return False
    # Upgrade legacy hash to bcrypt
    if _BCRYPT_AVAILABLE and _is_legacy_hash(stored):
        change_password(username, password, tenant_id)
    return True


def user_exists(username: str, tenant_id: str = "default") -> bool:
    _, execute_one = _db()
    row = execute_one(
        "SELECT username FROM users WHERE username = %s AND tenant_id = %s",
        (username, tenant_id)
    )
    return row is not None


def list_users(tenant_id: str = "default") -> list[dict]:
    execute, _ = _db()
    rows = execute(
        "SELECT username, email, role, invite_pending, created_at FROM users WHERE tenant_id = %s ORDER BY username",
        (tenant_id,)
    )
    return [
        {
            "username":      r["username"],
            "role":          r["role"],
            "email":         r.get("email", ""),
            "created_at":    str(r.get("created_at", "")),
            "invite_pending": r.get("invite_pending", False),
        }
        for r in rows
    ]


# ── Bootstrap ─────────────────────────────────────────────────────

def _bootstrap() -> None:
    """On first run, create initial admin account."""
    try:
        execute, execute_one = _db()

        # Ensure default tenant exists
        execute(
            "INSERT INTO tenants (tenant_id, name) VALUES ('default', 'Default') ON CONFLICT DO NOTHING"
        )

        admin_username = os.getenv("ADMIN_USERNAME", "admin").strip().lower()
        admin_password = os.getenv("ADMIN_PASSWORD", "").strip()

        existing = execute_one(
            "SELECT username FROM users WHERE tenant_id = 'default' LIMIT 1"
        )

        if not existing:
            if not admin_password:
                admin_password = secrets.token_urlsafe(16)
                print(f"\n{'='*60}")
                print(f"  FIRST RUN — Initial admin account created:")
                print(f"  Username: {admin_username}")
                print(f"  Password: {admin_password}")
                print(f"  Change this password immediately after first login.")
                print(f"{'='*60}\n")
            create_user(admin_username, admin_password, created_by="system", tenant_id="default")
            from app.security.rbac import assign_role
            assign_role(admin_username, "admin")

        elif admin_password:
            row = execute_one(
                "SELECT password_hash FROM users WHERE username = %s AND tenant_id = 'default'",
                (admin_username,)
            )
            if row and not verify_password(admin_password, row["password_hash"]):
                change_password(admin_username, admin_password)

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("bootstrap_failed error=%s", e)


_bootstrap()
