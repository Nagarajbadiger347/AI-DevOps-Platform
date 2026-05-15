"""
Shared dependencies, state, auth helpers, and Pydantic models
used across multiple routers.
"""
import time as _time
from collections import defaultdict as _defaultdict
from typing import Any, Optional, Dict

from fastapi import Depends, Header, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# ── Prometheus-style in-memory metrics ────────────────────────
_METRICS: dict = _defaultdict(int)
_METRICS_HIST: dict = _defaultdict(list)   # path → list of durations

# ── Pending pipeline states (awaiting human approval) ─────────
# keyed by correlation_id; allows /approvals/{id}/resume to
# re-invoke only the execute→validate→memory portion of the graph.
_PENDING_PIPELINE_STATES: dict = {}

# Recent pipeline results cache — keyed by incident_id, capped at 50 entries
_RECENT_RESULTS: dict = {}
_MAX_CACHED_RESULTS = 50

# ── War Room state — delegated to shared store ────────────────
from app.incident.war_room_store import (
    WAR_ROOMS as _WAR_ROOMS,
    save      as _wr_save,
    create    as _create_war_room,
    answer    as _answer_war_room_question,
)


def _wr_timeline(war_room_id: str) -> list:
    wr = _WAR_ROOMS.get(war_room_id)
    if not wr:
        return []
    events = [{
        "timestamp": wr.created_at,
        "event": f"War room created for {wr.incident_id}: {wr.incident_description}",
        "actor": "system",
        "source": "war_room",
    }]
    for a in (wr.pipeline_state.get("actions_taken") or wr.pipeline_state.get("executed_actions") or []):
        if isinstance(a, dict):
            events.append({
                "timestamp": a.get("executed_at", wr.created_at),
                "event": f"Action: {a.get('type','?')} — {a.get('description','')}",
                "actor": "pipeline",
                "source": "pipeline_state",
            })
    try:
        from app.chat.memory import get_history
        for msg in get_history(f"war_room::{war_room_id}", max_messages=100):
            role = getattr(msg, "role", "")
            if role in ("user", "assistant"):
                events.append({
                    "timestamp": getattr(msg, "timestamp", ""),
                    "event": getattr(msg, "content", "")[:200],
                    "actor": role,
                    "source": "conversation",
                })
    except Exception:
        pass
    events.sort(key=lambda e: e.get("timestamp") or "")
    return events


def _inc(key: str, amount: int = 1):
    _METRICS[key] += amount


def _cache_result(incident_id: str, result: dict) -> None:
    """Store a pipeline result in the rolling LRU cache."""
    _RECENT_RESULTS[incident_id] = result
    if len(_RECENT_RESULTS) > _MAX_CACHED_RESULTS:
        oldest = next(iter(_RECENT_RESULTS))
        del _RECENT_RESULTS[oldest]


# ── RBAC guard (legacy header-based) ──────────────────────────
from app.security.rbac import check_access


def _rbac_guard(x_user: Optional[str], required_action: str):
    """Raise 403 if x_user header is missing or lacks the required permission."""
    if not x_user:
        raise HTTPException(
            status_code=403,
            detail="X-User header required for this endpoint. "
                   "Assign a role via POST /security/roles first.",
        )
    result = check_access(x_user, required_action)
    if not result.get("allowed"):
        raise HTTPException(
            status_code=403,
            detail=f"User '{x_user}' lacks '{required_action}' permission. "
                   f"Role: {result.get('role', 'none')}.",
        )


# ── JWT / Bearer auth helpers ──────────────────────────────────
_bearer_scheme = HTTPBearer(auto_error=False)


class AuthContext:
    def __init__(self, username: str, role: str, tenant_id: str = "default",
                 workspace_id: str = "", plan_name: str = "starter"):
        self.username     = username
        self.role         = role
        self.tenant_id    = tenant_id
        self.workspace_id = workspace_id
        self.plan_name    = plan_name

    def can(self, feature: str) -> bool:
        """Check if the current tenant's plan allows a feature."""
        try:
            from app.tenants.store import get_plan
            plan = get_plan(self.plan_name)
            return plan.allows(feature) if plan else True
        except Exception:
            return True

    def check_usage_limit(self, resource: str) -> None:
        """Raise 402 if tenant has exceeded their plan limit for resource."""
        try:
            from app.tenants.store import check_limit
            result = check_limit(self.tenant_id, resource)
            if not result["allowed"]:
                raise HTTPException(
                    status_code=402,
                    detail=(
                        f"Plan limit reached: {result['current']}/{result['limit']} {resource} "
                        f"this month. Upgrade your plan at /settings/billing."
                    )
                )
        except HTTPException:
            raise
        except Exception:
            pass  # never block on metering errors


SESSION_COOKIE = "nsops_session"
CSRF_COOKIE    = "nsops_csrf"
CSRF_HEADER    = "X-CSRF-Token"


def _resolve_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    x_user: Optional[str] = Header(default=None),
    x_tenant_id: Optional[str] = Header(default=None),
) -> "AuthContext":
    """Resolve auth from EITHER `Authorization: Bearer <jwt>` OR the
    httpOnly `nsops_session` cookie. Cookie path additionally requires a
    matching X-CSRF-Token header on mutating requests.
    Bearer header path is unchanged for API/CLI consumers."""
    from app.security.rbac import get_user_role
    username = None
    jwt_role = None
    jwt_tenant = None
    jwt_workspace = None
    jwt_plan = None

    # ── 1. Bearer header (existing — for API clients, CLI, scripts) ──
    token_str = ""
    used_cookie = False
    if credentials and credentials.credentials:
        token_str = credentials.credentials
    else:
        # ── 2. Cookie (browser session) — must also carry CSRF token on
        #     mutating requests. GET/HEAD/OPTIONS are safe; we only enforce
        #     CSRF on writes.
        cookie_token = request.cookies.get(SESSION_COOKIE, "")
        if cookie_token:
            method = (request.method or "").upper()
            if method not in ("GET", "HEAD", "OPTIONS"):
                hdr = request.headers.get(CSRF_HEADER, "")
                cookie_csrf = request.cookies.get(CSRF_COOKIE, "")
                if not hdr or not cookie_csrf:
                    raise HTTPException(status_code=403, detail="CSRF token missing")
                import hmac as _hmac
                if not _hmac.compare_digest(hdr, cookie_csrf):
                    raise HTTPException(status_code=403, detail="CSRF token mismatch")
            token_str = cookie_token
            used_cookie = True

    if token_str:
        try:
            from app.core.auth import decode_token
            payload = decode_token(token_str)
            username      = payload.get("sub")
            jwt_role      = payload.get("role")
            jwt_tenant    = payload.get("tenant_id")
            jwt_workspace = payload.get("workspace_id", "")
            jwt_plan      = payload.get("plan_name", "starter")
        except HTTPException:
            username = None
            jwt_role = None
    if not username and x_user:
        username = x_user.strip().lower()
    if not username:
        username = "anonymous"
    role = get_user_role(username) if username != "anonymous" else (jwt_role or "viewer")
    # Tenant resolution: JWT claim > X-Tenant-ID header > "default"
    tenant_id = jwt_tenant or x_tenant_id or "default"
    # Resolve plan from tenant if not in JWT
    plan_name = jwt_plan or "starter"
    if not jwt_plan:
        try:
            from app.tenants.store import get_tenant
            t = get_tenant(tenant_id)
            if t:
                plan_name = t.plan_name
        except Exception:
            pass
    ctx = AuthContext(username=username, role=role, tenant_id=tenant_id,
                      workspace_id=jwt_workspace or "", plan_name=plan_name)
    ctx._bad_token = bool(token_str) and username == "anonymous" and not jwt_role
    return ctx


def require_super_admin(auth: AuthContext = Depends(_resolve_auth)) -> AuthContext:
    if getattr(auth, "_bad_token", False):
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    if auth.role != "super_admin":
        raise HTTPException(status_code=403, detail="Super-admin access required")
    return auth


def require_admin(auth: AuthContext = Depends(_resolve_auth)) -> AuthContext:
    if getattr(auth, "_bad_token", False):
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    if auth.role not in ("admin", "super_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return auth


def require_operator(auth: AuthContext = Depends(_resolve_auth)) -> AuthContext:
    if getattr(auth, "_bad_token", False):
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    if auth.role not in ("super_admin", "admin", "operator", "developer"):
        raise HTTPException(status_code=403, detail="Operator role or above required")
    return auth


def require_developer(auth: AuthContext = Depends(_resolve_auth)) -> AuthContext:
    if getattr(auth, "_bad_token", False):
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    if auth.role not in ("super_admin", "admin", "developer"):
        raise HTTPException(status_code=403, detail="Role 'developer' or above required")
    return auth


def require_viewer(auth: AuthContext = Depends(_resolve_auth)) -> AuthContext:
    if auth.role not in ("super_admin", "admin", "developer", "viewer"):
        raise HTTPException(status_code=403, detail="Authentication required")
    return auth


def optional_auth(auth: AuthContext = Depends(_resolve_auth)) -> Optional[AuthContext]:
    """Returns auth context if valid token provided, None otherwise (no 401)."""
    if auth.role in ("admin", "developer", "viewer"):
        return auth
    return None


# ── Shared Pydantic models ─────────────────────────────────────

class Event(BaseModel):
    id: str
    type: str
    source: str
    payload: Any


class ContextRequest(BaseModel):
    incident_id: str
    details: Any


class AccessRequest(BaseModel):
    user: str
    action: str


class RoleAssignment(BaseModel):
    user: str = ""
    role: str


class AWSConfig(BaseModel):
    resource_type: str = ""
    resource_id: str = ""
    log_group: str = ""


class K8sConfig(BaseModel):
    namespace: str = "default"


class IncidentRunRequest(BaseModel):
    incident_id:    str
    description:    str
    severity:       str  = "high"
    aws:            Optional[AWSConfig] = None
    k8s:            Optional[K8sConfig] = None
    auto_remediate: bool = False
    hours:          int  = 2
    aws_cfg:        Optional[Dict[str, Any]] = None
    k8s_cfg:        Optional[Dict[str, Any]] = None
    slack_channel:  str  = "#incidents"
    dry_run:             bool = False
    llm_provider:        str  = ""
    create_slack_channel: bool = False
    metadata:            Optional[Dict[str, Any]] = None
    # NOTE: user and role are resolved server-side from JWT — do NOT add them here
