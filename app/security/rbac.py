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

# Maps roles to the set of allowed actions
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin":     {"deploy", "rollback", "read", "write", "delete", "manage_users", "manage_secrets"},
    "developer": {"deploy", "read", "write"},
    "viewer":    {"read"},
}



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


def assign_role(user: str, role: str) -> dict:
    user = user.strip().lower()   
    """Assign a role to a user at runtime (persisted to disk)."""
    if role not in ROLE_PERMISSIONS:
        return {"success": False, "reason": f"Unknown role '{role}'. Valid roles: {list(ROLE_PERMISSIONS)}"}
    _user_roles[user] = role
    _save_to_file(_config_path)
    return {"success": True, "user": user, "role": role}


def revoke_role(user: str) -> dict:
    """Remove a user's role assignment (persisted to disk)."""
    if user not in _user_roles:
        return {"success": False, "reason": f"User '{user}' has no role assigned"}
    del _user_roles[user]
    _save_to_file(_config_path)
    return {"success": True, "user": user}


def get_user_role(user: str) -> str:
    """Return the role for a user, defaulting to 'viewer' if no role is assigned."""
    user = user.strip().lower()
    return _user_roles.get(user, "viewer")


def check_access(user: str, action: str) -> dict:
    user = user.strip().lower()
    role = _user_roles.get(user)
    if role is None:
        return {"allowed": False, "reason": f"User '{user}' has no role assigned"}
    allowed_actions = ROLE_PERMISSIONS.get(role, set())
    if action in allowed_actions:
        return {"allowed": True, "role": role}
    return {"allowed": False, "reason": f"Role '{role}' is not permitted to perform '{action}'"}
