"""
Security, RBAC, and policy routes.
Paths: /security/*, /policies/*
"""
import re
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, Header, HTTPException

from app.routes.deps import (
    require_viewer, require_super_admin, _rbac_guard,
    AuthContext, AccessRequest, RoleAssignment,
)
from app.security.rbac import check_access, assign_role, revoke_role

router = APIRouter(tags=["security"])


@router.get("/policies/rules", tags=["policies"])
def get_policy_rules(auth: AuthContext = Depends(require_viewer)):
    """Return the current policy rules JSON."""
    import json as _json
    rules_path = Path(__file__).resolve().parents[1] / "policies" / "rules.json"
    try:
        return _json.loads(rules_path.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read rules.json: {exc}")


@router.put("/policies/rules", tags=["policies"])
def update_policy_rules(rules: Dict[str, Any], auth: AuthContext = Depends(require_super_admin)):
    """Overwrite policy rules JSON. Super-admin only."""
    import json as _json
    rules_path = Path(__file__).resolve().parents[1] / "policies" / "rules.json"
    required_keys = {"blocked_actions", "action_permissions", "role_permissions", "guardrails"}
    missing = required_keys - set(rules.keys())
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required keys: {missing}")
    try:
        tmp = rules_path.with_suffix(".tmp")
        tmp.write_text(_json.dumps(rules, indent=2))
        tmp.replace(rules_path)
        try:
            from app.policies.policy_engine import PolicyEngine as _PE
            _PE._rules_cache = None  # type: ignore[attr-defined]
        except Exception:
            pass
        return {"success": True, "message": "Policy rules updated"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not write rules.json: {exc}")


@router.post("/security/check")
def security_check(req: AccessRequest):
    try:
        from app.policies.policy_engine import PolicyEngine
        engine = PolicyEngine()
        required_perm = engine.get_required_permission(req.action)
        if required_perm:
            result = check_access(req.user, required_perm)
            result["action"] = req.action
            result["required_permission"] = required_perm
        else:
            result = check_access(req.user, req.action)
    except Exception:
        result = check_access(req.user, req.action)
    return {"access": result}


@router.post("/security/roles")
def security_assign_role(req: RoleAssignment, x_user: Optional[str] = Header(default=None)):
    _rbac_guard(x_user, "manage_users")
    result = assign_role(req.user, req.role, changed_by=x_user or "system")
    return {"result": result}


@router.delete("/security/roles/{user}")
def security_revoke_role(user: str, x_user: Optional[str] = Header(default=None)):
    _rbac_guard(x_user, "manage_users")
    result = revoke_role(user, changed_by=x_user or "system")
    return {"result": result}


@router.get("/security/roles")
def security_get_roles(auth: AuthContext = Depends(require_viewer)):
    from app.security.rbac import _user_roles
    return {"roles": dict(_user_roles)}


@router.post("/security/roles/assign")
def security_assign_role_v2(req: RoleAssignment, auth: AuthContext = Depends(require_viewer)):
    result = assign_role(req.user, req.role, changed_by=auth.username, changer_role=auth.role)
    if not result.get("success"):
        raise HTTPException(status_code=403, detail=result.get("reason", "Failed"))
    return result
