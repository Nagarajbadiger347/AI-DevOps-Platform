"""Role-Based Access Control.

Role → permissions are defined here.
User → role assignments are persisted to RBAC_CONFIG_PATH (JSON file) and
kept in memory for fast lookups. Changes via assign_role() / revoke_role()
are immediately written to disk so they survive restarts.

Expected JSON format:
    {"alice": "developer", "bob": "viewer"}
"""

import json
import os
from pathlib import Path

try:
    from app.core.audit import audit_log as _audit_log
except Exception:
    def _audit_log(**kwargs): pass  # noqa: E731

# Maps roles to the set of allowed actions
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "super_admin": {"deploy", "rollback", "read", "write", "delete", "manage_users", "manage_secrets", "manage_admins"},
    "admin":       {"deploy", "rollback", "read", "write", "delete", "manage_users", "manage_secrets"},
    "developer":   {"deploy", "read", "write"},
    "viewer":      {"read"},
}

# Roles that require super_admin to assign or remove
PROTECTED_ROLES = {"admin", "super_admin"}



# In-memory user → role registry (populated at startup or via API)
_user_roles: dict[str, str] = {}

# Persistence path — env var or default next to this file
_config_path = Path(
    os.getenv("RBAC_CONFIG_PATH", "") or
    Path(__file__).resolve().parent / "roles.json"
)


def _load_from_file(path: Path) -> None:
    """Load user→role mappings from a JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            _user_roles.update({str(k): str(v) for k, v in data.items()})
    except (FileNotFoundError, json.JSONDecodeError):
        pass


def _save_to_file(path: Path) -> None:
    """Persist current user→role mappings to disk (atomic write)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_user_roles, indent=2))
        tmp.replace(path)
    except Exception:
        pass  # Never crash the caller due to persistence failure


# Auto-load on import
_load_from_file(_config_path)
_save_to_file(_config_path)


def assign_role(user: str, role: str, changed_by: str = "system", changer_role: str = "") -> dict:
    """Assign a role to a user at runtime (persisted to disk).

    Only super_admin may assign admin or super_admin roles.
    Pass changer_role="" to skip the enforcement (internal/bootstrap calls).
    """
    user = user.strip().lower()
    if role not in ROLE_PERMISSIONS:
        return {"success": False, "reason": f"Unknown role '{role}'. Valid roles: {list(ROLE_PERMISSIONS)}"}
    # Enforce: only super_admin can assign protected roles
    if changer_role and role in PROTECTED_ROLES and changer_role != "super_admin":
        return {"success": False, "reason": f"Only a super_admin can assign the '{role}' role"}
    previous = _user_roles.get(user, "none")
    _user_roles[user] = role
    _save_to_file(_config_path)
    try:
        from app.core.database import execute
        execute("UPDATE users SET role = %s, updated_at = NOW() WHERE username = %s", (role, user))
    except Exception:
        pass
    _audit_log(user=changed_by, action="assign_role",
               params={"target_user": user, "new_role": role, "previous_role": previous},
               result={"success": True}, source="rbac")
    return {"success": True, "user": user, "role": role}


def revoke_role(user: str, changed_by: str = "system") -> dict:
    """Remove a user's role assignment (persisted to disk)."""
    if user not in _user_roles:
        return {"success": False, "reason": f"User '{user}' has no role assigned"}
    previous = _user_roles.pop(user)
    _save_to_file(_config_path)
    _audit_log(user=changed_by, action="revoke_role",
               params={"target_user": user, "previous_role": previous},
               result={"success": True}, source="rbac")
    return {"success": True, "user": user}


def get_user_role(user: str) -> str:
    """Return the role for a user — checks DB first, falls back to in-memory cache."""
    user = user.strip().lower()
    try:
        from app.core.database import execute_one
        row = execute_one("SELECT role FROM users WHERE username = %s", (user,))
        if row and row.get("role"):
            _user_roles[user] = row["role"]  # sync cache
            return row["role"]
    except Exception:
        pass
    return _user_roles.get(user, "viewer")


def has_permission(role: str, permission: str) -> bool:
    """
    Check whether a role holds a specific permission.
    Used by the Executor for per-action RBAC enforcement at the execution layer
    (separate from route-level auth which guards the HTTP endpoint).

    Example:
        has_permission("developer", "write")  → True
        has_permission("viewer", "deploy")    → False
    """
    return permission in ROLE_PERMISSIONS.get(role, set())


def check_access(user: str, action: str) -> dict:
    user = user.strip().lower()
    role = _user_roles.get(user)
    if role is None:
        return {"allowed": False, "reason": f"User '{user}' has no role assigned"}
    allowed_actions = ROLE_PERMISSIONS.get(role, set())
    if action in allowed_actions:
        return {"allowed": True, "role": role}
    return {"allowed": False, "reason": f"Role '{role}' is not permitted to perform '{action}'"}
