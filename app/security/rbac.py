"""Role-Based Access Control.

Role → permissions are defined here.
User → role assignments are loaded from RBAC_CONFIG_PATH (JSON file) at startup,
or managed at runtime via assign_role() / revoke_role().

Expected JSON format:
    {"alice": "developer", "bob": "viewer"}
"""

import json
import os

# Maps roles to the set of allowed actions
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin":     {"deploy", "rollback", "read", "write", "delete", "manage_users"},
    "developer": {"deploy", "read", "write"},
    "viewer":    {"read"},
}

# In-memory user → role registry (populated at startup or via API)
_user_roles: dict[str, str] = {}


def _load_from_file(path: str) -> None:
    """Load user→role mappings from a JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            _user_roles.update({str(k): str(v) for k, v in data.items()})
    except (FileNotFoundError, json.JSONDecodeError):
        pass


# Auto-load on import if env var is set
_config_path = os.getenv("RBAC_CONFIG_PATH", "")
if _config_path:
    _load_from_file(_config_path)


def assign_role(user: str, role: str) -> dict:
    """Assign a role to a user at runtime."""
    if role not in ROLE_PERMISSIONS:
        return {"success": False, "reason": f"Unknown role '{role}'. Valid roles: {list(ROLE_PERMISSIONS)}"}
    _user_roles[user] = role
    return {"success": True, "user": user, "role": role}


def revoke_role(user: str) -> dict:
    """Remove a user's role assignment."""
    if user not in _user_roles:
        return {"success": False, "reason": f"User '{user}' has no role assigned"}
    del _user_roles[user]
    return {"success": True, "user": user}


def check_access(user: str, action: str) -> dict:
    role = _user_roles.get(user)
    if role is None:
        return {"allowed": False, "reason": f"User '{user}' has no role assigned"}
    allowed_actions = ROLE_PERMISSIONS.get(role, set())
    if action in allowed_actions:
        return {"allowed": True, "role": role}
    return {"allowed": False, "reason": f"Role '{role}' is not permitted to perform '{action}'"}
