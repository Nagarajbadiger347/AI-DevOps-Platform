import re
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Header, Depends
from fastapi.security import OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Any, Optional, Dict

from app.correlation.engine import correlate_events
from app.plugins.aws_checker import check_aws_infrastructure
from app.plugins.k8s_checker import check_k8s_cluster, check_k8s_nodes, check_k8s_pods, check_k8s_deployments
from app.integrations.github import (
    get_recent_commits as _get_recent_commits,
    get_recent_prs as _get_recent_prs,
    list_repos as _list_repos,
    get_profile_summary as _get_github_profile,
)
from app.integrations.k8s_ops import (
    restart_deployment, scale_deployment, get_pod_logs,
    list_pods, list_deployments, list_namespaces, get_cluster_events,
    get_unhealthy_pods, delete_pod, cordon_node, uncordon_node, get_resource_usage,
)
from app.integrations.aws_ops import (
    list_ec2_instances, get_ec2_status_checks, get_ec2_console_output,
    start_ec2_instance, stop_ec2_instance, reboot_ec2_instance, get_ec2_instance_info,
    scale_ecs_service, get_ecs_service_detail, force_new_ecs_deployment,
    invoke_lambda, set_alarm_state, get_sqs_queue_depth,
    reboot_rds_instance, get_rds_instance_detail,
    list_log_groups, get_recent_logs, search_logs,
    list_cloudwatch_alarms, get_metric,
    list_ecs_services, get_stopped_ecs_tasks,
    list_lambda_functions, get_lambda_errors,
    list_rds_instances, get_rds_events,
    get_target_health, get_cloudtrail_events,
    collect_diagnosis_context, get_scaling_metrics,
    list_s3_buckets, list_sqs_queues, list_dynamodb_tables,
    list_route53_healthchecks, list_sns_topics,
)
from app.llm.claude import (
    analyze_context, diagnose_aws_resource, review_pr, predict_scaling,
    assess_deployment, interpret_jira_for_pr, chat_devops,
)
from app.agents.incident_pipeline import run_incident_pipeline
from app.orchestrator.runner import run_pipeline as run_pipeline_v2
from app.core.config import settings as _settings
from app.integrations.slack import create_war_room, create_incident_channel, post_incident_summary, post_message
from app.integrations.universal_collector import collect_all_context, summarize_health
from app.integrations.jira import create_incident, add_comment as jira_add_comment
from app.integrations.opsgenie import notify_on_call
from app.integrations.github import (
    create_issue, create_pull_request,
    get_pr_for_review, post_pr_review_comment,
    create_incident_pr,
)
from app.memory.vector_db import store_incident, search_similar_incidents
from app.security.rbac import check_access, assign_role, revoke_role


# ── RBAC helper ───────────────────────────────────────────────

def _rbac_guard(x_user: Optional[str], required_action: str):
    """Raise 403 if x_user header is missing or lacks the required permission."""
    if not x_user:
        raise HTTPException(
            status_code=403,
            detail="X-User header required for this endpoint. "
                   "Assign a role via POST /security/roles first."
        )
    result = check_access(x_user, required_action)
    if not result.get("allowed"):
        raise HTTPException(
            status_code=403,
            detail=f"User '{x_user}' lacks '{required_action}' permission. "
                   f"Role: {result.get('role', 'none')}."
        )

async def _approval_cleanup_loop() -> None:
    """Background task: clean up expired approval requests every 5 minutes."""
    import asyncio
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            from app.incident.approval import cleanup_expired
            removed = cleanup_expired()
            if removed:
                import logging
                logging.getLogger(__name__).info(
                    "approval_cleanup_ran", extra={"removed": removed}
                )
        except asyncio.CancelledError:
            break
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "approval_cleanup_error", extra={"error": str(exc)}
            )


@asynccontextmanager
async def _lifespan(_: FastAPI):
    import asyncio
    if _settings.ENABLE_MONITOR_LOOP:
        from app.monitoring.loop import monitoring_loop
        asyncio.create_task(monitoring_loop())
    # Start approval cleanup background task
    cleanup_task = asyncio.create_task(_approval_cleanup_loop())
    # Validate critical configuration
    import warnings as _warnings
    if not os.getenv("JWT_SECRET_KEY"):
        _warnings.warn("JWT_SECRET_KEY not set — using insecure default. Set it with: openssl rand -hex 32")
    llm_keys = [os.getenv(k) for k in ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY")]
    if not any(llm_keys):
        _warnings.warn("No LLM API key configured (ANTHROPIC_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY). AI features will fail.")
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="AI DevOps Intelligence Platform",
    description="Autonomous DevOps management powered by multi-agent AI — built by Nagaraj",
    version="2.0.0",
    lifespan=_lifespan,
)
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    import os
    if os.path.exists("favicon.ico"):
        return FileResponse("favicon.ico")
    from fastapi.responses import Response
    return Response(status_code=204)

@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def chrome_devtools():
    """Silence Chrome DevTools 404 noise."""
    return {}

_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",") if o.strip()]
# Wildcard origins + credentials is a security violation — strip wildcards when credentials enabled
_CORS_ORIGINS = [o for o in _CORS_ORIGINS if o != "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-User"],
)

# ── Simple in-memory rate limiter ─────────────────────────────
import time as _time
from collections import defaultdict as _defaultdict
from fastapi import Request

_rate_store: dict = _defaultdict(list)
_RATE_LIMIT = 60   # requests
_RATE_WINDOW = 60  # seconds

# ── Prometheus-style in-memory metrics ────────────────────────
_METRICS: dict = _defaultdict(int)
_METRICS_HIST: dict = _defaultdict(list)  # path → list of durations

# ── Pending pipeline states (awaiting human approval) ─────────
# keyed by correlation_id; allows /approvals/{id}/resume to
# re-invoke only the execute→validate→memory portion of the graph.
_PENDING_PIPELINE_STATES: dict = {}

def _inc(key: str, amount: int = 1):
    _METRICS[key] += amount

@app.middleware("http")
async def _rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    method = request.method
    # Only rate-limit AI-heavy endpoints
    if path in ("/chat", "/chat/v2", "/incidents/run", "/warroom/create"):
        client_ip = request.client.host if request.client else "unknown"
        now = _time.time()
        window_start = now - _RATE_WINDOW
        _rate_store[client_ip] = [t for t in _rate_store[client_ip] if t > window_start]
        if len(_rate_store[client_ip]) >= _RATE_LIMIT:
            from fastapi.responses import JSONResponse
            _inc(f"nexusops_errors_total{{endpoint=\"{path}\"}}")
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again in a minute."})
        _rate_store[client_ip].append(now)
    # Track metrics for all non-static paths
    if not path.startswith("/static") and path not in ("/favicon.ico",):
        t0 = _time.time()
        response = await call_next(request)
        duration = _time.time() - t0
        _inc(f'nexusops_requests_total{{endpoint="{path}",method="{method}"}}')
        _METRICS_HIST[path].append(duration)
        # keep only last 1000 samples per path
        if len(_METRICS_HIST[path]) > 1000:
            _METRICS_HIST[path] = _METRICS_HIST[path][-1000:]
        if response.status_code >= 400:
            _inc(f'nexusops_errors_total{{endpoint="{path}"}}')
        return response
    return await call_next(request)

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
    user: str
    role: str

class K8sRestartRequest(BaseModel):
    namespace: str
    deployment: str

class K8sScaleRequest(BaseModel):
    namespace: str
    deployment: str
    replicas: int

class AWSMetricRequest(BaseModel):
    namespace: str
    metric_name: str
    dimensions: list
    hours: int = 1
    period: int = 300
    stat: str = "Average"

class AWSDiagnoseRequest(BaseModel):
    resource_type: str   # ec2 | ecs | lambda | rds | alb
    resource_id: str
    log_group: str = ""
    hours: int = 1

class AWSConfig(BaseModel):
    resource_type: str = ""
    resource_id: str = ""
    log_group: str = ""

class K8sConfig(BaseModel):
    namespace: str = "default"

class IncidentRunRequest(BaseModel):
    incident_id:    str
    description:    str
    severity:       str  = "high"     # critical | high | medium | low
    aws:            AWSConfig = None
    k8s:            K8sConfig = None
    auto_remediate: bool = False      # if True, execute actions automatically
    hours:          int  = 2          # lookback window for observability data
    # v2 LangGraph extras (optional — backwards compatible)
    user:           str  = "system"
    role:           str  = "admin"
    aws_cfg:        Optional[Dict[str, Any]] = None
    k8s_cfg:        Optional[Dict[str, Any]] = None
    slack_channel:  str  = "#incidents"
    dry_run:        bool = False          # if True, simulate pipeline without executing actions
    llm_provider:   str  = ""
    metadata:       Optional[Dict[str, Any]] = None

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request = None):
    import os as _os
    _aws_region = _os.getenv("AWS_REGION", "us-east-1")
    _html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>NexusOps — AI DevOps Platform</title>
<style>
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#090d16;--bg2:#0f1623;--bg3:#0c1220;
  --surface:#0f1623;--surface2:#141e2e;--surface3:#18243a;
  --border:rgba(255,255,255,0.08);--border2:rgba(255,255,255,0.14);
  --text:#e2e8f4;--text2:#94a3b8;--muted:#3d5070;
  --purple:#7c3aed;--purple-light:#a78bfa;--purple-glow:rgba(124,58,237,0.18);
  --cyan:#06b6d4;--cyan2:#0e7490;
  --green:#22c55e;--green2:#16a34a;
  --red:#ef4444;--red2:#dc2626;
  --amber:#f59e0b;--amber2:#d97706;
  --blue:#3b82f6;--blue2:#1d4ed8;
  --r:10px;--r-sm:6px;--r-lg:12px;
  --sidebar:248px;--topbar:56px;
  --shadow:0 1px 3px rgba(0,0,0,0.4);
  --shadow-lg:0 8px 32px rgba(0,0,0,0.6);
  --trans:.15s ease;
}
html{scroll-behavior:smooth}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;font-size:13.5px;overflow:hidden}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--surface3);border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:var(--muted)}
code,pre,.mono{font-family:'SF Mono','Fira Code','Cascadia Code',ui-monospace,monospace}

/* ── LOGIN ── */
#login-screen{position:fixed;inset:0;background:var(--bg);display:flex;align-items:stretch;z-index:999;overflow:hidden}
.login-left{flex:1;display:flex;flex-direction:column;align-items:flex-start;justify-content:center;padding:60px 64px;position:relative;overflow:hidden;background:linear-gradient(135deg,#0d0f1e 0%,#110d22 50%,#0a1020 100%)}
.login-left::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse at 30% 40%,rgba(124,58,237,0.28) 0%,transparent 60%),radial-gradient(ellipse at 80% 80%,rgba(6,182,212,0.15) 0%,transparent 55%);pointer-events:none}
.login-left-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,0.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none}
.login-left-content{position:relative;z-index:1;max-width:480px}
.login-product-logo{display:flex;align-items:center;gap:14px;margin-bottom:52px}
.login-product-icon{width:48px;height:48px;border-radius:13px;background:linear-gradient(135deg,#7c3aed,#06b6d4);display:flex;align-items:center;justify-content:center;font-size:22px;box-shadow:0 0 32px rgba(124,58,237,0.5),0 0 0 1px rgba(255,255,255,0.12)}
.login-product-name{font-size:1.4em;font-weight:800;letter-spacing:-.03em;background:linear-gradient(135deg,#c4b5fd 30%,#67e8f9);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.login-hero-heading{font-size:2.4em;font-weight:800;line-height:1.15;letter-spacing:-.04em;margin-bottom:18px;color:#f1f5f9}
.login-hero-heading span{background:linear-gradient(135deg,#a78bfa,#38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.login-hero-sub{font-size:.95em;color:var(--text2);line-height:1.7;margin-bottom:40px;max-width:380px}
.login-features{display:flex;flex-direction:column;gap:14px}
.login-feature{display:flex;align-items:flex-start;gap:12px}
.login-feature-icon{width:32px;height:32px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0;margin-top:1px}
.login-feature-text strong{display:block;font-size:.88em;font-weight:600;color:var(--text);margin-bottom:2px}
.login-feature-text span{font-size:.78em;color:var(--text2);line-height:1.5}
.login-right{width:420px;display:flex;align-items:center;justify-content:center;padding:40px 48px;background:var(--bg2);border-left:1px solid var(--border);flex-shrink:0}
.login-card{width:100%;max-width:340px}
.login-card-heading{font-size:1.35em;font-weight:800;letter-spacing:-.02em;margin-bottom:6px;color:var(--text)}
.login-card-sub{font-size:.82em;color:var(--text2);margin-bottom:32px}
.form-group{margin-bottom:18px}
.form-label{display:block;font-size:.71em;font-weight:700;color:var(--text2);text-transform:uppercase;letter-spacing:.07em;margin-bottom:7px}
.form-input-wrap{position:relative}
.form-input{width:100%;background:var(--surface2);border:1px solid var(--border);border-radius:var(--r-sm);padding:10px 14px;color:var(--text);font-size:.9em;font-family:inherit;outline:none;transition:border-color var(--trans),box-shadow var(--trans),background var(--trans)}
.form-input:focus{border-color:var(--purple);box-shadow:0 0 0 3px rgba(124,58,237,0.15);background:var(--surface3)}
.form-input.error-field{border-color:var(--red)!important;box-shadow:0 0 0 3px rgba(239,68,68,0.12)!important}
.form-input::placeholder{color:var(--muted)}
.form-input-icon{position:absolute;right:12px;top:50%;transform:translateY(-50%);color:var(--text2);cursor:pointer;user-select:none;font-size:14px;padding:3px 5px;border-radius:4px;transition:color var(--trans),background var(--trans)}
.form-input-icon:hover{color:var(--text);background:rgba(255,255,255,0.06)}
.login-btn{width:100%;padding:11px;border-radius:var(--r-sm);background:linear-gradient(135deg,#7c3aed,#6d28d9);color:#fff;font-size:.9em;font-weight:700;border:1px solid rgba(255,255,255,0.12);cursor:pointer;transition:filter var(--trans),transform var(--trans),box-shadow var(--trans);letter-spacing:.01em;box-shadow:0 4px 16px rgba(124,58,237,.35);margin-top:4px;display:flex;align-items:center;justify-content:center;gap:8px}
.login-btn:hover{filter:brightness(1.1);transform:translateY(-1px);box-shadow:0 6px 24px rgba(124,58,237,.5)}
.login-btn:active{transform:translateY(0);filter:brightness(.95)}
.login-btn:disabled{opacity:.6;cursor:not-allowed;transform:none;filter:none}
.login-hint{text-align:center;color:var(--muted);font-size:.72em;margin-top:20px;letter-spacing:.01em}
#login-error{color:var(--red);font-size:.81em;text-align:left;margin-bottom:16px;display:none;padding:10px 12px;background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);border-radius:var(--r-sm)}
@keyframes loginFadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.login-right{animation:loginFadeUp .4s cubic-bezier(.22,.68,0,1.1)}

/* ── APP LAYOUT ── */
#app{display:none;height:100vh;overflow:hidden}

/* ── SIDEBAR ── */
.sidebar{position:fixed;left:0;top:0;bottom:0;width:var(--sidebar);background:linear-gradient(180deg,#0b0f1c 0%,#0d1220 100%);border-right:1px solid rgba(124,58,237,0.18);display:flex;flex-direction:column;z-index:100;overflow-y:auto;overflow-x:hidden;box-shadow:4px 0 24px rgba(0,0,0,0.4)}
.sidebar::before{content:'';position:absolute;top:0;left:0;right:0;height:200px;background:radial-gradient(ellipse at 50% 0%,rgba(124,58,237,0.15) 0%,transparent 70%);pointer-events:none}
.sidebar-logo{display:flex;align-items:center;gap:10px;padding:18px 16px 16px;border-bottom:1px solid rgba(255,255,255,0.06)}
.sidebar-logo-icon{width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,#7c3aed,#06b6d4);display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;box-shadow:0 0 24px rgba(124,58,237,.5),0 0 0 1px rgba(255,255,255,.1)}
.sidebar-logo-text{font-size:.97em;font-weight:800;letter-spacing:-.025em;background:linear-gradient(135deg,#c4b5fd,#67e8f9);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sidebar-logo-sub{font-size:.58em;color:var(--muted);font-weight:600;display:block;margin-top:1px;letter-spacing:.06em;text-transform:uppercase}
.nav-section{padding:20px 16px 6px;font-size:.61em;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.12em}
.nav-item{display:flex;align-items:center;gap:9px;padding:8px 10px 8px 14px;border-radius:var(--r-sm);cursor:pointer;transition:background var(--trans),color var(--trans);color:var(--text2);margin:1px 8px;position:relative;font-size:.855em;font-weight:500;user-select:none}
.nav-item:hover{background:rgba(255,255,255,0.06);color:var(--text);transform:translateX(2px)}
.nav-item.active{background:linear-gradient(90deg,rgba(124,58,237,0.22),rgba(124,58,237,0.08));color:#c4b5fd;font-weight:600;border:1px solid rgba(124,58,237,0.2)}
.nav-item.active::before{content:'';position:absolute;left:-9px;top:50%;transform:translateY(-50%);width:3px;height:20px;background:linear-gradient(180deg,#a78bfa,#7c3aed);border-radius:0 3px 3px 0;box-shadow:0 0 8px rgba(124,58,237,0.6)}
.nav-icon{width:16px;text-align:center;font-size:14px;flex-shrink:0}
.nav-badge{margin-left:auto;background:var(--red);color:#fff;font-size:.61em;font-weight:700;padding:1px 6px;border-radius:10px;min-width:18px;text-align:center;line-height:1.5}
.sidebar-footer{margin-top:auto;padding:12px;border-top:1px solid var(--border)}
.user-tile{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:var(--r-sm);background:rgba(255,255,255,0.02);border:1px solid var(--border)}
.user-avatar{width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,var(--purple),var(--cyan2));display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:800;flex-shrink:0;color:#fff}
.user-info{flex:1;min-width:0}
.user-name{font-size:.82em;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.user-role{font-size:.67em;color:var(--muted);text-transform:capitalize;letter-spacing:.03em}
.logout-btn{width:26px;height:26px;display:flex;align-items:center;justify-content:center;border-radius:6px;color:var(--muted);transition:all var(--trans);flex-shrink:0;cursor:pointer;border:1px solid transparent;background:none;font-size:12px}
.logout-btn:hover{background:rgba(239,68,68,0.1);border-color:rgba(239,68,68,0.2);color:var(--red)}

/* ── MAIN CONTENT ── */
.main{margin-left:var(--sidebar);height:100vh;overflow-y:auto;display:flex;flex-direction:column}
.topbar{height:var(--topbar);background:rgba(9,13,22,0.95);backdrop-filter:blur(20px) saturate(1.4);border-bottom:1px solid rgba(124,58,237,0.12);display:flex;align-items:center;padding:0 24px;gap:12px;position:sticky;top:0;z-index:50;flex-shrink:0;box-shadow:0 1px 0 rgba(124,58,237,0.08)}
.topbar-title{font-size:1em;font-weight:700;flex:1;letter-spacing:-.02em;color:var(--text)}
.topbar-actions{display:flex;gap:8px;align-items:center}
.content{padding:24px;flex:1}

/* ── BUTTONS ── */
.btn{display:inline-flex;align-items:center;gap:6px;padding:7px 14px;border-radius:var(--r-sm);font-size:.84em;font-weight:600;cursor:pointer;border:1px solid transparent;transition:all var(--trans);font-family:inherit;text-decoration:none;white-space:nowrap;line-height:1.2;letter-spacing:.01em}
.btn-primary{background:linear-gradient(135deg,#7c3aed,#6d28d9);color:#fff;border-color:rgba(255,255,255,.1);box-shadow:0 2px 8px rgba(124,58,237,.3)}
.btn-primary:hover:not(:disabled){filter:brightness(1.1);transform:translateY(-1px);box-shadow:0 4px 14px rgba(124,58,237,.45)}
.btn-secondary{background:var(--surface2);color:var(--text);border-color:var(--border)}
.btn-secondary:hover:not(:disabled){background:var(--surface3);border-color:var(--border2)}
.btn-ghost{background:transparent;color:var(--text2);border-color:var(--border)}
.btn-ghost:hover:not(:disabled){background:var(--surface2);color:var(--text)}
.btn-success{background:rgba(34,197,94,0.1);color:var(--green);border-color:rgba(34,197,94,0.25)}
.btn-success:hover:not(:disabled){background:rgba(34,197,94,0.18)}
.btn-danger{background:rgba(239,68,68,0.08);color:var(--red);border-color:rgba(239,68,68,0.2)}
.btn-danger:hover:not(:disabled){background:rgba(239,68,68,0.16)}
.btn-sm{padding:5px 10px;font-size:.78em}
.btn:disabled{opacity:.4;cursor:not-allowed;transform:none!important;filter:none!important}

/* ── CARDS ── */
.card{background:linear-gradient(145deg,#111827 0%,#0f1623 100%);border:1px solid var(--border);border-radius:var(--r);padding:20px;transition:all .2s ease;box-shadow:0 2px 8px rgba(0,0,0,0.3)}
.card:hover{border-color:rgba(124,58,237,0.2);box-shadow:0 4px 20px rgba(0,0,0,.4)}
.card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid rgba(255,255,255,0.05)}
.card-title{font-size:.875em;font-weight:700;display:flex;align-items:center;gap:8px;color:var(--text)}
.card-icon{font-size:15px;opacity:.75}

/* ── GRID LAYOUTS ── */
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.grid-2{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}
.grid-2-1{display:grid;grid-template-columns:1fr 360px;gap:16px}
@media(max-width:1200px){.grid-4{grid-template-columns:repeat(2,1fr)}}
@media(max-width:900px){.grid-2-1{grid-template-columns:1fr}}

/* ── STAT CARDS ── */
.stat-card{background:linear-gradient(135deg,var(--surface2) 0%,var(--surface) 100%);border:1px solid var(--border);border-radius:var(--r);padding:22px 20px;transition:all .2s ease;cursor:default;position:relative;overflow:hidden}
.stat-card::after{content:'';position:absolute;top:-30px;right:-30px;width:100px;height:100px;border-radius:50%;opacity:.07;pointer-events:none}
.stat-card:nth-child(1)::after{background:var(--red)}
.stat-card:nth-child(2)::after{background:var(--amber)}
.stat-card:nth-child(3)::after{background:var(--blue)}
.stat-card:nth-child(4)::after{background:var(--green)}
.stat-card:hover{border-color:var(--border2);box-shadow:0 8px 32px rgba(0,0,0,.4);transform:translateY(-2px)}
.stat-header{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:16px}
.stat-icon-box{width:42px;height:42px;border-radius:11px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.stat-value{font-size:2.3em;font-weight:800;line-height:1;letter-spacing:-.05em;margin-bottom:6px}
.stat-label{font-size:.76em;color:var(--text2);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.stat-sub{font-size:.7em;margin-top:8px;display:flex;align-items:center;gap:5px}

/* ── BADGES ── */
.badge{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:20px;font-size:.72em;font-weight:600;letter-spacing:.02em}
.badge-purple{background:rgba(124,58,237,0.15);color:var(--purple-light)}
.badge-cyan{background:rgba(6,182,212,0.12);color:var(--cyan)}
.badge-green{background:rgba(34,197,94,0.12);color:var(--green)}
.badge-red{background:rgba(239,68,68,0.12);color:var(--red)}
.badge-amber{background:rgba(245,158,11,0.12);color:var(--amber)}
.badge-gray{background:rgba(148,163,184,0.08);color:var(--text2)}
.badge-blue{background:rgba(59,130,246,0.12);color:var(--blue)}

/* ── TABLES ── */
.table-wrap{overflow-x:auto;border-radius:var(--r-sm);border:1px solid var(--border)}
table{width:100%;border-collapse:collapse}
thead{background:var(--surface2)}
th{padding:10px 14px;text-align:left;font-size:.71em;font-weight:700;color:var(--text2);text-transform:uppercase;letter-spacing:.07em;white-space:nowrap}
td{padding:11px 14px;font-size:.855em;border-top:1px solid var(--border);vertical-align:middle}
tr:hover td{background:rgba(255,255,255,0.02)}

/* ── FORMS ── */
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
textarea.form-input{resize:vertical;min-height:90px;line-height:1.55}
select.form-input{cursor:pointer;appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' fill='none'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%2394a3b8' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 12px center;padding-right:32px}
.form-toggle{display:flex;align-items:center;gap:10px;cursor:pointer;padding:10px 12px;background:var(--surface2);border:1px solid var(--border);border-radius:var(--r-sm)}
.toggle-track{width:38px;height:21px;background:var(--surface3);border-radius:12px;position:relative;transition:background .2s;flex-shrink:0;border:1px solid rgba(255,255,255,0.06)}
.toggle-track.on{background:var(--purple);border-color:rgba(124,58,237,0.4)}
.toggle-thumb{width:15px;height:15px;background:#fff;border-radius:50%;position:absolute;top:2px;left:2px;transition:left .2s;box-shadow:0 1px 4px rgba(0,0,0,0.4)}
.toggle-track.on .toggle-thumb{left:19px}
.toggle-label{font-size:.855em;color:var(--text2);flex:1}

/* ── PROGRESS/CONFIDENCE BAR ── */
.conf-bar{height:5px;background:var(--surface3);border-radius:3px;overflow:hidden}
.conf-fill{height:100%;border-radius:3px;transition:width .6s ease}
.conf-fill.high{background:linear-gradient(90deg,var(--green),var(--cyan))}
.conf-fill.med{background:linear-gradient(90deg,var(--amber),var(--amber2))}
.conf-fill.low{background:linear-gradient(90deg,var(--red),var(--red2))}

/* ── STATUS DOT ── */
.status-dot{width:7px;height:7px;border-radius:50%;display:inline-block;flex-shrink:0}
.dot-green{background:var(--green);box-shadow:0 0 5px var(--green)}
.dot-red{background:var(--red);box-shadow:0 0 5px var(--red)}
.dot-amber{background:var(--amber);box-shadow:0 0 5px var(--amber)}
.dot-gray{background:var(--muted)}
.dot-pulse{animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

/* ── CHAT ── */
.chat-messages{flex:1;overflow-y:auto;padding:20px 24px;display:flex;flex-direction:column;gap:0;min-height:0}
.chat-row{display:flex;gap:12px;padding:10px 0;animation:msgIn .2s ease}
.chat-row.user{flex-direction:row-reverse}
@keyframes msgIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.chat-avatar{width:32px;height:32px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:.8em;font-weight:700;margin-top:2px}
.chat-avatar.ai{background:linear-gradient(135deg,#7c3aed,#a855f7);color:#fff}
.chat-avatar.user-av{background:var(--surface3,#334155);color:var(--text2);border:1px solid var(--border)}
.chat-body{display:flex;flex-direction:column;max-width:72%;gap:4px}
.chat-row.user .chat-body{align-items:flex-end}
.chat-bubble{padding:12px 16px;border-radius:18px;line-height:1.65;font-size:.875em;word-break:break-word}
.chat-bubble.user{background:linear-gradient(135deg,#7c3aed,#5b21b6);color:#fff;border-bottom-right-radius:4px;box-shadow:0 2px 14px rgba(124,58,237,.3)}
.chat-bubble.assistant{background:var(--surface2);border:1px solid var(--border);border-bottom-left-radius:4px;color:var(--text)}
/* markdown inside assistant bubble */
.chat-bubble.assistant p{margin:.35em 0}
.chat-bubble.assistant p:first-child{margin-top:0}
.chat-bubble.assistant p:last-child{margin-bottom:0}
.chat-bubble.assistant ul,.chat-bubble.assistant ol{margin:.4em 0;padding-left:1.4em}
.chat-bubble.assistant li{margin:.2em 0}
.chat-bubble.assistant h1,.chat-bubble.assistant h2,.chat-bubble.assistant h3{margin:.6em 0 .3em;font-weight:700;line-height:1.3}
.chat-bubble.assistant h1{font-size:1.1em}.chat-bubble.assistant h2{font-size:1em}.chat-bubble.assistant h3{font-size:.95em}
.chat-bubble.assistant hr{border:none;border-top:1px solid var(--border);margin:.6em 0}
.chat-bubble.assistant blockquote{border-left:3px solid var(--purple);padding-left:10px;color:var(--text2);margin:.4em 0}
.chat-bubble.assistant table{border-collapse:collapse;width:100%;margin:.5em 0;font-size:.9em}
.chat-bubble.assistant th,.chat-bubble.assistant td{border:1px solid var(--border);padding:5px 10px;text-align:left}
.chat-bubble.assistant th{background:var(--surface3,#1e293b);font-weight:600}
.chat-bubble.assistant code:not(.code-block-code){background:rgba(124,58,237,.15);padding:1px 6px;border-radius:4px;font-family:'SF Mono','Cascadia Code',ui-monospace,monospace;font-size:.85em;color:var(--purple-light,#a78bfa)}
.resource-link{display:inline;cursor:pointer;border-radius:4px;padding:0 3px;font-family:'SF Mono','Cascadia Code',ui-monospace,monospace;font-size:.85em;text-decoration:none;transition:background .15s}
a.resource-link{color:#38bdf8}a.resource-link:hover{background:rgba(56,189,248,.15);text-decoration:underline}
.ec2-link{color:#f59e0b}.ec2-link:hover{background:rgba(245,158,11,.15)}
.rds-link{color:#34d399}.rds-link:hover{background:rgba(52,211,153,.15)}
.lambda-link{color:#a78bfa}.lambda-link:hover{background:rgba(167,139,250,.15)}
.sg-link{color:#fb923c}.sg-link:hover{background:rgba(251,146,60,.15)}
.sha-link,.pr-link{color:#94a3b8;background:rgba(148,163,184,.1);padding:1px 5px;border-radius:4px}
.arn-link{color:#64748b;font-size:.8em}
.chat-code-block{position:relative;margin:.5em 0;border-radius:10px;overflow:hidden;border:1px solid var(--border)}
.chat-code-header{display:flex;align-items:center;justify-content:space-between;padding:5px 12px;background:var(--surface3,#1e293b);font-size:.72em;color:var(--muted)}
.chat-code-header button{background:none;border:1px solid var(--border);color:var(--text2);padding:2px 8px;border-radius:4px;cursor:pointer;font-size:.9em;transition:all .15s}
.chat-code-header button:hover{background:var(--purple);color:#fff;border-color:var(--purple)}
.chat-code-block pre{margin:0;padding:12px 14px;background:#0f172a;overflow-x:auto;font-family:'SF Mono','Cascadia Code',ui-monospace,monospace;font-size:.82em;line-height:1.6;color:#e2e8f0}
.chat-meta{font-size:.67em;color:var(--muted);margin-top:3px;display:flex;align-items:center;gap:8px}
.chat-row.user .chat-meta{justify-content:flex-end}
.chat-copy-btn{background:none;border:none;color:var(--muted);cursor:pointer;padding:2px 4px;border-radius:4px;font-size:.9em;opacity:0;transition:opacity .15s}
.chat-body:hover .chat-copy-btn{opacity:1}
.chat-copy-btn:hover{color:var(--text2)}
.chat-input-row{display:flex;gap:0;padding:12px 0 2px;border-top:1px solid var(--border);align-items:flex-end;background:var(--surface);border-radius:0 0 var(--r-md) var(--r-md)}
.chat-input-wrap{flex:1;display:flex;align-items:flex-end;background:var(--surface2);border:1px solid var(--border);border-radius:14px;padding:8px 12px;transition:border-color .2s,box-shadow .2s;gap:8px}
.chat-input-wrap:focus-within{border-color:var(--purple);box-shadow:0 0 0 3px rgba(124,58,237,0.12)}
.chat-input{flex:1;background:none;border:none;color:var(--text);font-family:inherit;font-size:.9em;outline:none;resize:none;min-height:24px;max-height:160px;overflow-y:auto;line-height:1.5;padding:0}
.chat-send-btn{background:var(--purple);border:none;color:#fff;width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;cursor:pointer;flex-shrink:0;transition:background .15s,transform .1s}
.chat-send-btn:hover{background:#6d28d9;transform:scale(1.08)}
.chat-send-btn:active{transform:scale(.95)}
.chat-chips{display:flex;gap:7px;flex-wrap:wrap;padding:0 0 12px}
.chip{padding:6px 14px;border-radius:20px;font-size:.755em;background:var(--surface2);border:1px solid var(--border);color:var(--text2);cursor:pointer;transition:all .15s;white-space:nowrap}
.chip:hover{border-color:var(--purple);color:var(--purple-light,#a78bfa);background:rgba(124,58,237,0.1)}
.typing-row{display:flex;gap:12px;padding:10px 0;align-items:flex-start}
.typing{display:none;align-items:center;gap:5px;padding:12px 16px;background:var(--surface2);border:1px solid var(--border);border-radius:18px;border-bottom-left-radius:4px}
.typing span{width:7px;height:7px;background:var(--purple,#7c3aed);border-radius:50%;animation:bounce .9s infinite;opacity:.7}
.typing span:nth-child(2){animation-delay:.18s}
.typing span:nth-child(3){animation-delay:.36s}
@keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px)}}
.chat-welcome{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:20px;padding:20px}
.chat-welcome-icon{width:56px;height:56px;background:linear-gradient(135deg,#7c3aed,#a855f7);border-radius:50%;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 24px rgba(124,58,237,.35)}
.chat-welcome h2{font-size:1.1em;font-weight:700;color:var(--text);margin:0}
.chat-welcome p{font-size:.82em;color:var(--muted);margin:0;text-align:center}
.chat-suggestion-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;width:100%;max-width:460px}
.chat-suggestion{padding:10px 14px;border-radius:10px;background:var(--surface2);border:1px solid var(--border);color:var(--text2);font-size:.78em;cursor:pointer;transition:all .15s;text-align:left;line-height:1.4}
.chat-suggestion:hover{border-color:var(--purple);color:var(--text);background:rgba(124,58,237,0.07)}

.stat-card{position:relative;overflow:hidden}
.stat-card::after{content:'';position:absolute;top:0;right:0;width:80px;height:80px;border-radius:50%;opacity:.06;transform:translate(20px,-20px)}
.stat-icon-box{display:flex;align-items:center;justify-content:center;width:38px;height:38px;border-radius:10px}

/* ── LOADING ── */
.spinner{width:18px;height:18px;border:2px solid var(--border);border-top-color:var(--purple);border-radius:50%;animation:spin .65s linear infinite;flex-shrink:0}
@keyframes spin{to{transform:rotate(360deg)}}
.skeleton{background:linear-gradient(90deg,var(--surface2) 25%,var(--surface3) 50%,var(--surface2) 75%);background-size:400% 100%;animation:shimmer 1.5s infinite;border-radius:5px;height:13px}
@keyframes shimmer{to{background-position:-400% 0}}
.loading-state{display:flex;align-items:center;justify-content:center;gap:10px;padding:36px;color:var(--text2);font-size:.84em}
.empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:40px 20px;color:var(--text2);gap:10px}
.empty-state .empty-icon{font-size:2.2em;opacity:.35}
.empty-state p{font-size:.84em;opacity:.65}

/* ── ALERTS ── */
.alert-item{display:flex;align-items:flex-start;gap:12px;padding:12px 14px;border-radius:var(--r-sm);border:1px solid var(--border);background:var(--surface2);transition:background .15s}
.alert-item:hover{background:var(--surface3)}
.alert-source{font-size:.64em;font-weight:700;padding:2px 7px;border-radius:4px;text-transform:uppercase;letter-spacing:.06em;flex-shrink:0;margin-top:1px}
.src-grafana{background:rgba(245,158,11,0.12);color:var(--amber)}
.src-cloudwatch{background:rgba(239,68,68,0.12);color:var(--red)}
.src-k8s{background:rgba(6,182,212,0.12);color:var(--cyan)}
.src-opsgenie{background:rgba(124,58,237,0.12);color:var(--purple-light)}

/* ── RESULT CARD ── */
.result-card{background:var(--surface2);border:1px solid var(--border);border-radius:var(--r);padding:20px;margin-top:16px;animation:fadeUp .3s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.result-section{margin-bottom:16px}
.result-section-label{font-size:.69em;font-weight:700;color:var(--text2);text-transform:uppercase;letter-spacing:.09em;margin-bottom:8px;display:flex;align-items:center;gap:6px}
.result-root-cause{font-size:.875em;line-height:1.6;color:var(--text);background:var(--surface);border-left:3px solid var(--purple);padding:12px 14px;border-radius:0 var(--r-sm) var(--r-sm) 0}
.result-reasoning{font-size:.8em;line-height:1.6;color:var(--text2);background:var(--surface);border:1px solid var(--border);padding:10px 14px;border-radius:var(--r-sm);white-space:pre-wrap;display:none}
.result-reasoning.open{display:block}
.result-data-gap{font-size:.8em;color:var(--amber);background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.2);border-radius:6px;padding:8px 12px;margin-top:6px;line-height:1.5}
/* Action steps list */
.action-steps{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:8px}
.action-step{display:flex;align-items:flex-start;gap:12px;background:var(--surface);border:1px solid var(--border);border-radius:var(--r-sm);padding:12px 14px;transition:border-color .15s}
.action-step:hover{border-color:var(--border2)}
.step-num{min-width:24px;height:24px;border-radius:50%;background:rgba(124,58,237,.18);color:var(--purple-light);font-size:.72em;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px}
.step-num.executed{background:rgba(16,185,129,.15);color:var(--green)}
.step-num.blocked{background:rgba(245,158,11,.15);color:var(--amber)}
.step-body{flex:1;min-width:0}
.step-type{font-size:.68em;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text2);margin-bottom:3px}
.step-desc{font-size:.84em;line-height:1.5;color:var(--text)}
.step-target{font-size:.74em;color:var(--purple-light);margin-top:4px;font-family:'SF Mono','Cascadia Code',ui-monospace,monospace}
.step-cost{font-size:.7em;color:var(--text2);margin-top:3px}
.step-reason{font-size:.74em;color:var(--amber);margin-top:4px}
/* Data source badges */
.data-source-row{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}
.data-src-badge{display:inline-flex;align-items:center;gap:5px;padding:3px 9px;border-radius:20px;font-size:.7em;font-weight:600;border:1px solid}
.data-src-ok{background:rgba(16,185,129,.1);color:var(--green);border-color:rgba(16,185,129,.25)}
.data-src-miss{background:rgba(248,113,113,.08);color:#f87171;border-color:rgba(248,113,113,.2)}
.action-chip{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;font-size:.74em;background:var(--surface3);border:1px solid var(--border);margin:3px;color:var(--text2)}

/* ── INTEGRATION CARDS ── */
.integration-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:18px;transition:all .2s;display:flex;flex-direction:column;gap:12px}
.integration-card:hover{border-color:var(--border2);transform:translateY(-2px);box-shadow:0 4px 16px rgba(0,0,0,.25)}
.int-header{display:flex;align-items:center;gap:11px}
.int-icon{width:38px;height:38px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0}
.int-name{font-size:.9em;font-weight:700;color:var(--text)}
.int-status{font-size:.74em;color:var(--text2);display:flex;align-items:center;gap:6px}

/* ── MODAL ── */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,0.75);backdrop-filter:blur(6px);z-index:500;display:flex;align-items:center;justify-content:center;padding:20px;display:none}
.modal-overlay.open{display:flex}
.modal{background:var(--surface);border:1px solid var(--border2);border-radius:var(--r);padding:28px;width:100%;max-width:520px;max-height:80vh;overflow-y:auto;animation:fadeUp .25s ease;box-shadow:var(--shadow-lg)}
.modal-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:22px}
.modal-title{font-size:1em;font-weight:700;letter-spacing:-.01em}
.modal-close{background:none;border:none;color:var(--text2);cursor:pointer;font-size:18px;padding:4px;border-radius:5px;transition:all .15s;line-height:1}
.modal-close:hover{color:var(--text);background:rgba(255,255,255,0.06)}

/* ── TOAST ── */
#toast-container{position:fixed;bottom:24px;right:24px;z-index:999;display:flex;flex-direction:column;gap:8px;pointer-events:none}
.toast{padding:11px 16px;border-radius:var(--r-sm);font-size:.835em;font-weight:500;box-shadow:0 4px 16px rgba(0,0,0,.4);animation:slideIn .25s cubic-bezier(.22,.68,0,1.2);display:flex;align-items:center;gap:9px;max-width:320px;pointer-events:all}
.toast-success{background:#0f2016;border:1px solid rgba(34,197,94,.25);color:var(--green)}
.toast-error{background:#1e0a0a;border:1px solid rgba(239,68,68,.25);color:var(--red)}
.toast-info{background:var(--surface2);border:1px solid var(--border2);color:var(--text)}
@keyframes slideIn{from{opacity:0;transform:translateX(24px)}to{opacity:1;transform:translateX(0)}}

/* ── MISC ── */
.divider{height:1px;background:var(--border);margin:16px 0}
.text-muted{color:var(--text2);font-size:.82em}
.text-sm{font-size:.82em}
.flex{display:flex}
.flex-center{display:flex;align-items:center}
.gap-8{gap:8px}
.gap-12{gap:12px}
.mb-4{margin-bottom:4px}
.mb-8{margin-bottom:8px}
.mb-12{margin-bottom:12px}
.mb-16{margin-bottom:16px}
.section-page{display:none}
.section-page.active{display:block}
.topbar-status{display:flex;align-items:center;gap:6px;font-size:.76em;font-weight:600;color:var(--green);padding:4px 12px;background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);border-radius:20px;letter-spacing:.01em}
.topbar-status.degraded{color:var(--amber);background:rgba(245,158,11,0.08);border-color:rgba(245,158,11,0.2)}
.topbar-status.error{color:var(--red);background:rgba(239,68,68,0.08);border-color:rgba(239,68,68,0.2)}
.section-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}
.section-title{font-size:1.05em;font-weight:700;letter-spacing:-.02em;color:var(--text)}
.section-sub{font-size:.78em;color:var(--text2);margin-top:2px}
.tab-pills{display:flex;gap:2px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:3px}
.tab-pill{padding:5px 14px;border-radius:6px;font-size:.8em;font-weight:600;cursor:pointer;border:none;background:transparent;color:var(--text2);transition:all .15s;font-family:inherit}
.tab-pill.active{background:var(--surface3);color:var(--text);box-shadow:0 1px 3px rgba(0,0,0,.3)}
.tab-pill:hover:not(.active){color:var(--text);background:rgba(255,255,255,0.03)}
.info-row{display:flex;align-items:center;gap:8px;padding:9px 0;border-bottom:1px solid var(--border);font-size:.855em}
.info-row:last-child{border-bottom:none}
.info-label{color:var(--text2);width:120px;flex-shrink:0;font-size:.9em}
.info-value{color:var(--text);flex:1}
</style>
</head>
<body>

<!-- LOGIN -->
<div id="login-screen">
  <div class="login-left">
    <div class="login-left-grid"></div>
    <div class="login-left-content">
      <div class="login-product-logo">
        <div class="login-product-icon">&#9889;</div>
        <div class="login-product-name">NexusOps</div>
      </div>
      <h1 class="login-hero-heading">AI-powered ops for<br><span>modern infrastructure</span></h1>
      <p class="login-hero-sub">Automated incident response, intelligent cost analysis, and real-time infrastructure monitoring — all in one platform.</p>
      <div class="login-features">
        <div class="login-feature">
          <div class="login-feature-icon" style="background:rgba(124,58,237,0.15);color:#a78bfa">&#128680;</div>
          <div class="login-feature-text">
            <strong>Automated Incident Response</strong>
            <span>AI triages alerts, identifies root causes, and executes remediation actions autonomously.</span>
          </div>
        </div>
        <div class="login-feature">
          <div class="login-feature-icon" style="background:rgba(6,182,212,0.12);color:#06b6d4">&#9729;</div>
          <div class="login-feature-text">
            <strong>Multi-cloud Infrastructure</strong>
            <span>Unified visibility across AWS EC2, CloudWatch, Kubernetes, and third-party integrations.</span>
          </div>
        </div>
        <div class="login-feature">
          <div class="login-feature-icon" style="background:rgba(34,197,94,0.12);color:#22c55e">&#129302;</div>
          <div class="login-feature-text">
            <strong>Conversational AI Assistant</strong>
            <span>Ask questions about your infrastructure in plain English and get actionable insights.</span>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div class="login-right">
    <div class="login-card">
      <div class="login-card-heading">Welcome back</div>
      <div class="login-card-sub">Sign in to your NexusOps workspace</div>
      <div id="login-error"></div>
      <div class="form-group">
        <label class="form-label">Username</label>
        <input type="text" id="login-user" class="form-input" value="admin" autocomplete="username" spellcheck="false" placeholder="your-username"/>
      </div>
      <div class="form-group">
        <label class="form-label">Password</label>
        <div class="form-input-wrap">
          <input type="password" id="login-pass" class="form-input" style="padding-right:42px" placeholder="Enter your password" autocomplete="current-password"
            onkeydown="if(event.key==='Enter')App.login()"/>
          <span class="form-input-icon" id="pw-toggle" onclick="App.togglePw()" title="Show password">&#128065;</span>
        </div>
      </div>
      <button class="login-btn" onclick="App.login()" id="login-btn">
        <span id="login-btn-text">Sign In</span>
        <span style="font-size:14px">&#8594;</span>
      </button>
      <p class="login-hint">Contact your administrator for access</p>
    </div>
  </div>
</div>

<!-- APP -->
<div id="app">
  <!-- SIDEBAR -->
  <nav class="sidebar">
    <div class="sidebar-logo">
      <div class="sidebar-logo-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg></div>
      <div>
        <div class="sidebar-logo-text">NexusOps</div>
        <span class="sidebar-logo-sub">AI DevOps Platform</span>
      </div>
    </div>

    <div class="nav-section">Overview</div>
    <div class="nav-item active" onclick="App.navigate('dashboard')" data-s="dashboard">
      <div class="nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg></div> Dashboard
    </div>
    <div class="nav-item" onclick="App.navigate('monitoring')" data-s="monitoring">
      <div class="nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></div> Monitoring
    </div>

    <div class="nav-section">Operations</div>
    <div class="nav-item" onclick="App.navigate('incidents')" data-s="incidents">
      <div class="nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></div> Incidents
      <span class="nav-badge" id="badge-incidents" style="display:none">0</span>
    </div>
    <div class="nav-item" onclick="App.navigate('warroom')" data-s="warroom">
      <div class="nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg></div> War Room
    </div>
    <div class="nav-item" onclick="App.navigate('approvals')" data-s="approvals">
      <div class="nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg></div> Approvals
      <span class="nav-badge" id="badge-approvals" style="display:none">0</span>
    </div>
    <div class="nav-item" onclick="App.navigate('cost')" data-s="cost">
      <div class="nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg></div> Cost Analysis
    </div>

    <div class="nav-section">Intelligence</div>
    <div class="nav-item" onclick="App.navigate('chat')" data-s="chat">
      <div class="nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z"/></svg></div> AI Assistant
    </div>
    <div class="nav-item" onclick="App.navigate('infra')" data-s="infra">
      <div class="nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg></div> Infrastructure
    </div>
    <div class="nav-item" onclick="App.navigate('integrations')" data-s="integrations">
      <div class="nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg></div> Integrations
    </div>

    <div class="nav-section">Admin</div>
    <div class="nav-item" onclick="App.navigate('users')" data-s="users">
      <div class="nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg></div> Users
    </div>
    <div class="nav-item" onclick="App.navigate('security')" data-s="security">
      <div class="nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg></div> Security
    </div>

    <div class="sidebar-footer">
      <div class="user-tile">
        <div class="user-avatar" id="user-avatar-initial">A</div>
        <div class="user-info">
          <div class="user-name" id="sidebar-username">admin</div>
          <div class="user-role" id="sidebar-role">admin</div>
        </div>
        <button class="logout-btn" onclick="App.logout()" title="Sign out">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
        </button>
      </div>
    </div>
  </nav>

  <!-- MAIN -->
  <div class="main">
    <div class="topbar">
      <div class="topbar-title" id="topbar-title">Dashboard</div>
      <div class="topbar-actions">
        <div class="topbar-status" id="topbar-status">
          <span class="status-dot dot-green dot-pulse"></span>
          All Systems Operational
        </div>
        <div style="display:flex;align-items:center;gap:6px;padding:4px 10px;background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.2);border-radius:7px" title="Global AI model — used for chat, incident pipeline, and all AI features">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z"/></svg>
          <select id="global-llm-select" onchange="App.setGlobalLLM(this.value)" style="background:transparent;border:none;color:#c4b5fd;font-size:.78em;font-weight:600;font-family:inherit;cursor:pointer;outline:none;padding-right:4px">
            <option value="" style="background:#0f1623;color:#e2e8f4">Auto (best available)</option>
            <option value="anthropic" style="background:#0f1623;color:#e2e8f4">Claude (Anthropic)</option>
            <option value="openai" style="background:#0f1623;color:#e2e8f4">GPT (OpenAI)</option>
            <option value="groq" style="background:#0f1623;color:#e2e8f4">Groq / Llama</option>
            <option value="ollama" style="background:#0f1623;color:#e2e8f4">Ollama (local)</option>
          </select>
        </div>
        <button class="btn btn-ghost btn-sm" onclick="App.refreshCurrent()" title="Refresh current section">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
          Refresh
        </button>
        <button class="btn btn-primary btn-sm" onclick="App.newIncident()">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
          New Incident
        </button>
      </div>
    </div>
    <div class="content">

      <!-- DASHBOARD -->
      <div id="s-dashboard" class="section-page active">
        <!-- Stat Cards Row -->
        <div class="grid-4 mb-16">
          <div class="stat-card" style="border-top:2px solid var(--red);box-shadow:0 0 0 1px rgba(239,68,68,0.1),0 4px 24px rgba(0,0,0,0.35)">
            <div class="stat-header">
              <div class="stat-icon-box" style="background:linear-gradient(135deg,rgba(239,68,68,.3),rgba(239,68,68,.12));color:#f87171;box-shadow:0 0 20px rgba(239,68,68,.2)"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></div>
              <span class="badge badge-gray" id="ds-incidents-badge" style="display:none">live</span>
            </div>
            <div class="stat-value" style="color:var(--text)" id="ds-incidents"><div class="skeleton" style="width:48px;height:32px"></div></div>
            <div class="stat-label">Active Incidents</div>
            <div class="stat-sub" id="ds-incidents-sub"><span style="color:var(--muted)">Loading...</span></div>
          </div>
          <div class="stat-card" style="border-top:2px solid var(--amber);box-shadow:0 0 0 1px rgba(245,158,11,0.1),0 4px 24px rgba(0,0,0,0.35)">
            <div class="stat-header">
              <div class="stat-icon-box" style="background:linear-gradient(135deg,rgba(245,158,11,.3),rgba(245,158,11,.12));color:#fbbf24;box-shadow:0 0 20px rgba(245,158,11,.2)"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg></div>
            </div>
            <div class="stat-value" style="color:var(--text)" id="ds-approvals"><div class="skeleton" style="width:48px;height:32px"></div></div>
            <div class="stat-label">Pending Approvals</div>
            <div class="stat-sub" id="ds-approvals-sub"><span style="color:var(--muted)">Loading...</span></div>
          </div>
          <div class="stat-card" style="border-top:2px solid var(--blue);box-shadow:0 0 0 1px rgba(59,130,246,0.1),0 4px 24px rgba(0,0,0,0.35)">
            <div class="stat-header">
              <div class="stat-icon-box" style="background:linear-gradient(135deg,rgba(59,130,246,.3),rgba(59,130,246,.12));color:#60a5fa;box-shadow:0 0 20px rgba(59,130,246,.2)"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg></div>
            </div>
            <div class="stat-value" style="color:var(--text)" id="ds-alerts"><div class="skeleton" style="width:48px;height:32px"></div></div>
            <div class="stat-label">AWS Alarms Firing</div>
            <div class="stat-sub" id="ds-alerts-sub"><span style="color:var(--muted)">Loading...</span></div>
          </div>
          <div class="stat-card" style="border-top:2px solid var(--green);box-shadow:0 0 0 1px rgba(34,197,94,0.1),0 4px 24px rgba(0,0,0,0.35)">
            <div class="stat-header">
              <div class="stat-icon-box" style="background:linear-gradient(135deg,rgba(34,197,94,.3),rgba(34,197,94,.12));color:#4ade80;box-shadow:0 0 20px rgba(34,197,94,.2)" id="ds-k8s-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg></div>
            </div>
            <div id="ds-k8s"><div class="skeleton" style="width:72px;height:28px"></div></div>
            <div class="stat-label">K8s Cluster</div>
            <div class="stat-sub" id="ds-k8s-sub"></div>
          </div>
        </div>
        <!-- Second Row: Activity + Health + Quick Actions -->
        <div style="display:grid;grid-template-columns:1fr 1fr 320px;gap:16px">
          <div class="card">
            <div class="card-header">
              <div class="card-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                Recent Activity
              </div>
              <button class="btn btn-ghost btn-sm" onclick="App._sectionLoaded.dashboard=0;App.loadDashboard()">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
              </button>
            </div>
            <div id="dash-activity"><div class="skeleton mb-8"></div><div class="skeleton mb-8" style="width:82%"></div><div class="skeleton" style="width:68%"></div></div>
          </div>
          <div class="card">
            <div class="card-header">
              <div class="card-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Integration Health
              </div>
            </div>
            <div id="dash-health"><div class="skeleton mb-8"></div><div class="skeleton mb-8" style="width:78%"></div><div class="skeleton" style="width:60%"></div></div>
          </div>
          <div class="card">
            <div class="card-header">
              <div class="card-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                Quick Actions
              </div>
            </div>
            <div style="display:flex;flex-direction:column;gap:5px">
  <button onclick="App.navigate('incidents')" style="display:flex;align-items:center;gap:10px;width:100%;padding:9px 12px;background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.18);border-radius:8px;color:var(--text);cursor:pointer;transition:all .15s;text-align:left" onmouseover="this.style.background='rgba(239,68,68,.14)'" onmouseout="this.style.background='rgba(239,68,68,.07)'">
    <span style="width:28px;height:28px;border-radius:7px;background:rgba(239,68,68,.15);color:#f87171;display:flex;align-items:center;justify-content:center;flex-shrink:0">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
    </span>
    <span style="flex:1;font-size:.82em;font-weight:500">Run Incident Pipeline</span>
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color:var(--muted)"><polyline points="9 18 15 12 9 6"/></svg>
  </button>

  <button onclick="App.navigate('warroom')" style="display:flex;align-items:center;gap:10px;width:100%;padding:9px 12px;background:rgba(124,58,237,.07);border:1px solid rgba(124,58,237,.18);border-radius:8px;color:var(--text);cursor:pointer;transition:all .15s;text-align:left" onmouseover="this.style.background='rgba(124,58,237,.14)'" onmouseout="this.style.background='rgba(124,58,237,.07)'">
    <span style="width:28px;height:28px;border-radius:7px;background:rgba(124,58,237,.15);color:#a78bfa;display:flex;align-items:center;justify-content:center;flex-shrink:0">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
    </span>
    <span style="flex:1;font-size:.82em;font-weight:500">Create War Room</span>
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color:var(--muted)"><polyline points="9 18 15 12 9 6"/></svg>
  </button>

  <button onclick="App.navigate('chat')" style="display:flex;align-items:center;gap:10px;width:100%;padding:9px 12px;background:rgba(6,182,212,.07);border:1px solid rgba(6,182,212,.18);border-radius:8px;color:var(--text);cursor:pointer;transition:all .15s;text-align:left" onmouseover="this.style.background='rgba(6,182,212,.14)'" onmouseout="this.style.background='rgba(6,182,212,.07)'">
    <span style="width:28px;height:28px;border-radius:7px;background:rgba(6,182,212,.15);color:#22d3ee;display:flex;align-items:center;justify-content:center;flex-shrink:0">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z"/></svg>
    </span>
    <span style="flex:1;font-size:.82em;font-weight:500">Ask AI Assistant</span>
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color:var(--muted)"><polyline points="9 18 15 12 9 6"/></svg>
  </button>

  <button onclick="App.navigate('approvals')" style="display:flex;align-items:center;gap:10px;width:100%;padding:9px 12px;background:rgba(34,197,94,.07);border:1px solid rgba(34,197,94,.18);border-radius:8px;color:var(--text);cursor:pointer;transition:all .15s;text-align:left" onmouseover="this.style.background='rgba(34,197,94,.14)'" onmouseout="this.style.background='rgba(34,197,94,.07)'">
    <span style="width:28px;height:28px;border-radius:7px;background:rgba(34,197,94,.15);color:#4ade80;display:flex;align-items:center;justify-content:center;flex-shrink:0">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
    </span>
    <span style="flex:1;font-size:.82em;font-weight:500">Review Approvals</span>
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color:var(--muted)"><polyline points="9 18 15 12 9 6"/></svg>
  </button>

  <button onclick="App.navigate('infra')" style="display:flex;align-items:center;gap:10px;width:100%;padding:9px 12px;background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.18);border-radius:8px;color:var(--text);cursor:pointer;transition:all .15s;text-align:left" onmouseover="this.style.background='rgba(245,158,11,.14)'" onmouseout="this.style.background='rgba(245,158,11,.07)'">
    <span style="width:28px;height:28px;border-radius:7px;background:rgba(245,158,11,.15);color:#fbbf24;display:flex;align-items:center;justify-content:center;flex-shrink:0">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>
    </span>
    <span style="flex:1;font-size:.82em;font-weight:500">View Infrastructure</span>
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color:var(--muted)"><polyline points="9 18 15 12 9 6"/></svg>
  </button>
</div>
          </div>
        </div>
      </div>

      <!-- MONITORING -->
      <div id="s-monitoring" class="section-page">
        <div class="section-header">
          <div><div class="section-title">Monitoring</div><div class="section-sub">Real-time alerts from all connected sources</div></div>
          <button class="btn btn-secondary btn-sm" onclick="App.loadMonitoring()">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
            Refresh
          </button>
        </div>
        <div class="card">
          <div class="card-header">
            <div class="card-title">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
              Alert Stream
            </div>
            <span class="badge badge-gray" id="alert-count-badge"></span>
          </div>
          <div id="monitoring-alerts"><div class="loading-state"><div class="spinner"></div> Loading alerts...</div></div>
        </div>
      </div>

      <!-- INCIDENTS -->
      <div id="s-incidents" class="section-page">
        <div class="section-header">
          <div><div class="section-title">Incidents</div><div class="section-sub">Run the AI pipeline to analyze and remediate incidents</div></div>
        </div>
        <div class="grid-2-1">
          <div class="card">
            <div class="card-header">
              <div class="card-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                Run Incident Pipeline
              </div>
            </div>
            <div class="form-group mb-12">
              <label class="form-label">Incident ID</label>
              <input type="text" id="inc-id" class="form-input" placeholder="INC-2024-001 (auto-generated if empty)"/>
            </div>
            <div class="form-group mb-12">
              <label class="form-label">Description <span style="color:var(--red)">*</span></label>
              <textarea id="inc-desc" class="form-input" placeholder="Describe the incident — be specific: which service, what error, what symptoms..." rows="4"></textarea>
            </div>
            <div class="form-row mb-12">
              <div class="form-group">
                <label class="form-label">Severity</label>
                <select id="inc-sev" class="form-input">
                  <option value="critical">Critical</option>
                  <option value="high" selected>High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
              </div>
              <div class="form-group">
                <label class="form-label">Lookback (hours)</label>
                <input type="number" id="inc-hours" class="form-input" value="2" min="1" max="24"/>
              </div>
            </div>
            <div class="form-group mb-12">
              <div style="display:flex;gap:16px;flex-wrap:wrap">
                <div class="form-toggle" onclick="App.toggleAutoRemediate()">
                  <div class="toggle-track" id="auto-rem-toggle"><div class="toggle-thumb"></div></div>
                  <span class="toggle-label">Auto-Remediate</span>
                </div>
                <div class="form-toggle" onclick="App.toggleDryRun()">
                  <div class="toggle-track" id="dry-run-toggle"><div class="toggle-thumb"></div></div>
                  <span class="toggle-label">Dry Run <small style="color:var(--muted)">(preview only)</small></span>
                </div>
              </div>
            </div>
            <div style="display:flex;gap:8px;margin-bottom:16px">
              <button class="btn btn-primary" onclick="App.runIncident(false)" id="run-inc-btn" style="flex:1;justify-content:center;padding:10px">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                Run Pipeline
              </button>
              <button class="btn btn-ghost" onclick="App.runIncident(true)" id="dry-run-btn" style="padding:10px 14px" title="Preview what the pipeline would do without executing">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                Preview
              </button>
            </div>
            <div id="inc-result"></div>
          </div>
          <div class="card">
            <div class="card-header">
              <div class="card-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                Active Incidents
              </div>
              <button class="btn btn-ghost btn-sm" onclick="App.loadIncidents()">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
              </button>
            </div>
            <div id="active-incidents"><div class="empty-state"><div class="empty-icon">&#9989;</div><p>No active incidents</p></div></div>
          </div>
        </div>
      </div>

      <!-- WAR ROOM -->
      <div id="s-warroom" class="section-page">
        <div class="section-header">
          <div><div class="section-title">War Room</div><div class="section-sub">Collaborative incident command center with AI assistance</div></div>
        </div>
        <div class="grid-2-1">
          <div id="warroom-detail" style="display:none;flex-direction:column;gap:16px">
            <div class="card" id="warroom-info"></div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
              <div class="card" style="display:flex;flex-direction:column;height:520px">
                <div class="card-header">
                  <div class="card-title">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                    AI Assistant
                  </div>
                  <button class="btn btn-ghost btn-sm" onclick="App.suggestNextSteps()">&#129302; Next Steps</button>
                </div>
                <div class="chat-messages" id="warroom-messages" style="flex:1;height:0"></div>
                <div class="chat-input-row">
                  <textarea class="chat-input" id="warroom-input" placeholder="Ask the AI anything about this incident..." rows="1"
                    onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();App.askWarRoom()}"></textarea>
                  <button class="btn btn-primary" onclick="App.askWarRoom()">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                    Send
                  </button>
                </div>
              </div>
              <div class="card" style="display:flex;flex-direction:column;height:520px">
                <div class="card-header">
                  <div class="card-title">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                    Slack Channel
                  </div>
                  <button class="btn btn-ghost btn-sm" onclick="App.refreshSlackHistory()" id="slack-refresh-btn">&#8635; Refresh</button>
                </div>
                <div class="chat-messages" id="slack-messages" style="flex:1;height:0">
                  <div class="empty-state"><p>Slack channel messages will appear here</p></div>
                </div>
                <div class="chat-input-row">
                  <textarea class="chat-input" id="slack-input" placeholder="Send a message to Slack channel..." rows="1"
                    onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();App.sendSlackMessage()}"></textarea>
                  <button class="btn btn-primary" onclick="App.sendSlackMessage()">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                    Send
                  </button>
                </div>
              </div>
            </div>
          </div>
          <div id="warroom-list-panel" style="display:flex;flex-direction:column;gap:16px">
            <div class="card">
              <div class="card-header">
                <div class="card-title">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                  Create War Room
                </div>
              </div>
              <div class="form-group mb-12">
                <label class="form-label">Incident ID</label>
                <input type="text" id="wr-inc-id" class="form-input" placeholder="INC-001"/>
              </div>
              <div class="form-group mb-12">
                <label class="form-label">Description</label>
                <input type="text" id="wr-desc" class="form-input" placeholder="Brief incident description"/>
              </div>
              <div class="form-group mb-16">
                <label class="form-label">Severity</label>
                <select id="wr-sev" class="form-input">
                  <option value="critical">Critical</option>
                  <option value="high" selected>High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
              </div>
              <button class="btn btn-primary" onclick="App.createWarRoom()" style="width:100%;justify-content:center">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                Create War Room
              </button>
            </div>
            <div class="card">
              <div class="card-header">
                <div class="card-title">
                  <span class="status-dot dot-red dot-pulse"></span>
                  Active War Rooms
                </div>
                <button class="btn btn-ghost btn-sm" onclick="App.loadWarRooms()">
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
                </button>
              </div>
              <div id="warroom-list"><div class="loading-state"><div class="spinner"></div> Loading...</div></div>
            </div>
          </div>
        </div>
      </div>

      <!-- APPROVALS -->
      <div id="s-approvals" class="section-page">
        <div class="section-header">
          <div><div class="section-title">Approvals</div><div class="section-sub">Review and approve AI-recommended remediation actions</div></div>
          <button class="btn btn-secondary btn-sm" onclick="App.loadApprovals()">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
            Refresh
          </button>
        </div>
        <div class="card">
          <div class="card-header">
            <div class="card-title">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
              Pending Approvals
            </div>
            <span class="badge badge-amber" id="approvals-count-badge" style="display:none"></span>
          </div>
          <div id="approvals-list"><div class="loading-state"><div class="spinner"></div> Loading...</div></div>
        </div>
      </div>

      <!-- COST ANALYSIS -->
      <div id="s-cost" class="section-page">
        <div class="section-header">
          <div><div class="section-title">Cost Analysis</div><div class="section-sub">Live AWS spend data + action impact estimation</div></div>
          <button class="btn btn-ghost btn-sm" onclick="App.loadCostOverview()">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
            Refresh
          </button>
        </div>
        <!-- Live AWS spend summary -->
        <div id="cost-summary-row" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:16px"></div>
        <!-- Service breakdown + trend -->
        <div class="grid-2" style="margin-bottom:16px">
          <div class="card">
            <div class="card-header"><div class="card-title">&#128202; Top Services This Month</div></div>
            <div id="cost-services"><div class="loading-state"><div class="spinner"></div></div></div>
          </div>
          <div class="card">
            <div class="card-header"><div class="card-title">&#128200; 6-Month Spend Trend</div></div>
            <div id="cost-trend"><div class="loading-state"><div class="spinner"></div></div></div>
          </div>
        </div>
        <div class="grid-2">
          <div class="card">
            <div class="card-header">
              <div class="card-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                Analyze Action Costs
              </div>
            </div>
            <div class="form-group mb-12">
              <label class="form-label">Actions JSON</label>
              <textarea id="cost-actions" class="form-input" rows="7" style="font-family:'SF Mono','Cascadia Code',ui-monospace,monospace;font-size:.82em" placeholder='[{"type":"k8s_scale","namespace":"prod","deployment":"api","replicas":5}]'></textarea>
            </div>
            <button class="btn btn-primary" onclick="App.analyzeCost()" style="width:100%;justify-content:center">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
              Analyze Cost Impact
            </button>
            <div id="cost-result"></div>
          </div>
          <div class="card">
            <div class="card-header">
              <div class="card-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-6"/></svg>
                Action Impact Result
              </div>
            </div>
            <div id="cost-aws" style="padding:8px 0"><div class="empty-state"><p class="text-muted">Run an analysis to see impact</p></div></div>
          </div>
        </div>
      </div>

      <!-- AI ASSISTANT -->
      <div id="s-chat" class="section-page">
        <div class="card" style="height:calc(100vh - var(--topbar) - 48px);display:flex;flex-direction:column">
          <div class="card-header" style="flex-shrink:0">
            <div class="card-title">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z"/></svg>
              AI Assistant
              <span class="text-sm" style="color:var(--muted);font-weight:400" id="chat-session-label"></span>
            </div>
            <div style="display:flex;gap:8px;align-items:center">
              <button class="btn btn-ghost btn-sm" onclick="App.newChat()">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
                New Chat
              </button>
            </div>
          </div>
          <div class="chat-chips" id="chat-chips" style="flex-shrink:0">
            <div class="chip" onclick="App.chipChat('What is the current K8s cluster health?')">K8s health?</div>
            <div class="chip" onclick="App.chipChat('Are there any AWS CloudWatch alarms firing?')">AWS alarms?</div>
            <div class="chip" onclick="App.chipChat('Show me the last 5 GitHub commits')">Last commits?</div>
            <div class="chip" onclick="App.chipChat('What is the current AWS monthly cost?')">Current costs?</div>
            <div class="chip" onclick="App.chipChat('List all EC2 instances and their state')">EC2 instances?</div>
            <div class="chip" onclick="App.chipChat('Are there any active incidents right now?')">Active incidents?</div>
          </div>
          <div class="chat-messages" id="chat-messages" style="flex:1;height:0;overflow-y:auto">
            <div class="chat-welcome" id="chat-welcome">
              <div class="chat-welcome-icon">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z"/></svg>
              </div>
              <h2>NexusOps AI</h2>
              <p>Ask me anything about your infrastructure — I can check alerts,<br>EC2 instances, K8s pods, GitHub activity, costs, and more.</p>
            </div>
          </div>
          <div id="typing-indicator" style="display:none;flex-shrink:0;padding:4px 24px 0">
            <div class="typing-row">
              <div class="chat-avatar ai">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z"/></svg>
              </div>
              <div class="typing"><span></span><span></span><span></span></div>
            </div>
          </div>
          <div class="chat-input-row" style="flex-shrink:0;padding:10px 0 2px;border-top:1px solid var(--border)">
            <div class="chat-input-wrap">
              <textarea class="chat-input" id="chat-input" placeholder="Ask anything about your infrastructure..." rows="1"
                onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();App.sendChat()}"
                oninput="this.style.height='';this.style.height=Math.min(this.scrollHeight,160)+'px'"></textarea>
              <button class="chat-send-btn" onclick="App.sendChat()" title="Send (Enter)">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- INFRASTRUCTURE -->
      <div id="s-infra" class="section-page">
        <div class="section-header">
          <div><div class="section-title">Infrastructure</div><div class="section-sub">AWS and Kubernetes resource overview</div></div>
          <div class="tab-pills" id="infra-tabs">
            <button class="tab-pill active" onclick="App.infraTab('aws',this)">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline;margin-right:5px;vertical-align:middle"><path d="M4.5 16.5c-1.5 1.26-2 5-1 5.5s2.5.31 2.5-1.5c0-3.84 2.5-6.5 4.5-6.5 1.14 0 2.5.5 2.5.5S15 16.5 18 14.5c1.28-.86 2-2.5 2-4 0-2.5-1.5-4-3.5-4-1.5 0-3 1-3 1S12.5 6.5 10 6.5C7 6.5 4.5 9.5 4.5 12.5c0 1 .5 2.5 0 4z"/></svg>
              AWS
            </button>
            <button class="tab-pill" onclick="App.infraTab('k8s',this)">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline;margin-right:5px;vertical-align:middle"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
              Kubernetes
            </button>
          </div>
        </div>
        <div id="infra-aws">
          <div class="grid-2">
            <div class="card">
              <div class="card-header">
                <div class="card-title">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
                  EC2 Instances
                </div>
                <button class="btn btn-ghost btn-sm" onclick="App.loadEC2()">
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
                </button>
              </div>
              <div id="ec2-list"><div class="loading-state"><div class="spinner"></div> Loading...</div></div>
            </div>
            <div class="card">
              <div class="card-header">
                <div class="card-title">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/></svg>
                  CloudWatch Alarms
                </div>
                <button class="btn btn-ghost btn-sm" onclick="App.loadAlarms()">
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
                </button>
              </div>
              <div id="alarms-list"><div class="loading-state"><div class="spinner"></div> Loading...</div></div>
            </div>
          </div>
        </div>
        <div id="infra-k8s" style="display:none">
          <div class="grid-2">
            <div class="card">
              <div class="card-header">
                <div class="card-title">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                  Cluster Health
                </div>
                <button class="btn btn-ghost btn-sm" onclick="App.loadK8sHealth()">
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
                </button>
              </div>
              <div id="k8s-health"><div class="loading-state"><div class="spinner"></div> Loading...</div></div>
            </div>
            <div class="card">
              <div class="card-header">
                <div class="card-title">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                  Pods
                </div>
                <button class="btn btn-ghost btn-sm" onclick="App.loadPods()">
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
                </button>
              </div>
              <div id="pods-list"><div class="loading-state"><div class="spinner"></div> Loading...</div></div>
            </div>
          </div>
        </div>
      </div>

      <!-- INTEGRATIONS -->
      <div id="s-integrations" class="section-page">
        <div class="section-header">
          <div><div class="section-title">Integrations</div><div class="section-sub">Connected services and API key status</div></div>
          <button class="btn btn-secondary btn-sm" onclick="App.loadIntegrations()">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
            Refresh
          </button>
        </div>
        <div class="grid-3" id="integrations-grid">
          <div class="loading-state" style="grid-column:1/-1"><div class="spinner"></div> Checking integrations...</div>
        </div>
      </div>

      <!-- USERS -->
      <div id="s-users" class="section-page">
        <div class="section-header">
          <div><div class="section-title">Users</div><div class="section-sub">Manage platform access and roles</div></div>
          <button class="btn btn-primary btn-sm" onclick="App.openInviteModal()">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            Invite User
          </button>
        </div>
        <div class="card">
          <div class="card-header">
            <div class="card-title">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
              Team Members
            </div>
          </div>
          <div id="users-table"><div class="loading-state"><div class="spinner"></div> Loading...</div></div>
        </div>
      </div>

      <!-- SECURITY -->
      <div id="s-security" class="section-page">
        <div class="section-header">
          <div><div class="section-title">Security</div><div class="section-sub">API keys, audit log, and webhook configuration</div></div>
        </div>
        <div class="grid-2 mb-16">
          <div class="card">
            <div class="card-header">
              <div class="card-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
                API Keys Status
              </div>
            </div>
            <div id="secrets-list"><div class="loading-state"><div class="spinner"></div> Loading...</div></div>
          </div>
          <div class="card">
            <div class="card-header">
              <div class="card-title">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                Audit Log
              </div>
            </div>
            <div id="audit-list"><div class="loading-state"><div class="spinner"></div> Loading...</div></div>
          </div>
        </div>
        <div class="card">
          <div class="card-header">
            <div class="card-title">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
              Webhook Endpoints
            </div>
          </div>
          <div id="webhook-urls"></div>
        </div>
      </div>

    </div><!-- /content -->
  </div><!-- /main -->
</div><!-- /app -->

<!-- MODALS -->
<div class="modal-overlay" id="approve-modal">
  <div class="modal">
    <div class="modal-header">
      <div class="modal-title">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline;vertical-align:middle;margin-right:6px"><polyline points="20 6 9 17 4 12"/></svg>
        Review Approval Request
      </div>
      <button class="modal-close" onclick="App.closeModal('approve-modal')">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>
    <div id="approve-modal-body"></div>
    <div style="display:flex;gap:8px;margin-top:22px">
      <button class="btn btn-success" onclick="App.submitApproval()">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
        Approve Selected
      </button>
      <button class="btn btn-danger" onclick="App.submitRejection()">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        Reject
      </button>
      <button class="btn btn-ghost" onclick="App.closeModal('approve-modal')">Cancel</button>
    </div>
  </div>
</div>
<div class="modal-overlay" id="invite-modal">
  <div class="modal">
    <div class="modal-header">
      <div class="modal-title">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline;vertical-align:middle;margin-right:6px"><path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="8.5" cy="7" r="4"/><line x1="20" y1="8" x2="20" y2="14"/><line x1="23" y1="11" x2="17" y2="11"/></svg>
        Invite New User
      </div>
      <button class="modal-close" onclick="App.closeModal('invite-modal')">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>
    <div class="form-group mb-12">
      <label class="form-label">Username</label>
      <input type="text" id="invite-username" class="form-input" placeholder="jane.doe"/>
    </div>
    <div class="form-group mb-12">
      <label class="form-label">Email</label>
      <input type="email" id="invite-email" class="form-input" placeholder="jane@company.com"/>
    </div>
    <div class="form-group mb-20">
      <label class="form-label">Role</label>
      <select id="invite-role" class="form-input">
        <option value="viewer">Viewer — Read-only access</option>
        <option value="developer" selected>Developer — Manage incidents and infra</option>
        <option value="admin">Admin — Full platform access</option>
      </select>
    </div>
    <button class="btn btn-primary" onclick="App.submitInvite()" style="width:100%;justify-content:center;padding:10px">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
      Send Invite
    </button>
  </div>
</div>
<div id="toast-container"></div>

<script>
const App = {
  token: localStorage.getItem('nexusops_token'),
  username: localStorage.getItem('nexusops_user') || '',
  role: localStorage.getItem('nexusops_role') || '',
  globalLLM: localStorage.getItem('nexusops_llm') || '',
  currentSection: 'dashboard',
  chatSessionId: null,
  pendingAction: null,
  pendingParams: null,
  autoRemediate: false,
  dryRun: false,
  currentWarRoomId: null,
  currentSlackChannel: '',
  currentApprovalId: null,
  refreshTimer: null,
  _sectionLoaded: {},   // section -> timestamp of last load
  _CACHE_TTL: 45000,    // 45s cache per section
  _awsRegion: '{{ aws_region }}',

  setGlobalLLM(val){
    this.globalLLM=val;
    localStorage.setItem('nexusops_llm',val);
    const sel=document.getElementById('global-llm-select');
    if(sel) sel.value=val;
  },

  // ── HTTP ──────────────────────────────────────────────────────────
  async api(method, path, body=null) {
    const headers = {};
    if(this.token) headers['Authorization']='Bearer '+this.token;
    if(body!==null) headers['Content-Type']='application/json';
    try{
      const r=await fetch(path,{method,headers,body:body!==null?JSON.stringify(body):undefined});
      if(r.status===401){this.logout();return null;}
      return r;
    }catch(e){this.toast('Connection error: '+e.message,'error');return null;}
  },

  // ── TOASTS ────────────────────────────────────────────────────────
  toast(msg,type='info'){
    const el=document.createElement('div');
    el.className='toast toast-'+type;
    const icons={success:'&#9989;',error:'&#10060;',info:'&#8505;',warn:'&#9888;'};
    el.innerHTML=(icons[type]||'')+' '+msg;
    document.getElementById('toast-container').appendChild(el);
    setTimeout(()=>{el.style.opacity='0';el.style.transform='translateX(20px)';setTimeout(()=>el.remove(),300)},3500);
  },

  // ── ROUTING (hash-based for back/forward support) ─────────────────
  navigate(section, pushState=true, forceReload=false){
    if(!['dashboard','monitoring','incidents','warroom','approvals','cost','chat','infra','integrations','users','security'].includes(section))section='dashboard';
    // If user explicitly clicks the same section they're on, bust cache so it reloads
    const sameSectionClick = pushState && section === this.currentSection;
    if(sameSectionClick || forceReload) this._sectionLoaded[section]=0;
    document.querySelectorAll('.section-page').forEach(s=>s.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
    const pg=document.getElementById('s-'+section);
    if(pg)pg.classList.add('active');
    const ni=document.querySelector('[data-s="'+section+'"]');
    if(ni)ni.classList.add('active');
    const titles={dashboard:'Dashboard',monitoring:'Monitoring',incidents:'Incidents',warroom:'War Room',approvals:'Approvals',cost:'Cost Analysis',chat:'AI Assistant',infra:'Infrastructure',integrations:'Integrations',users:'Users',security:'Security'};
    document.getElementById('topbar-title').textContent=titles[section]||section;
    this.currentSection=section;
    if(pushState && window.location.hash!=='#'+section)history.pushState({section},'','#'+section);
    const now=Date.now();
    const stale=!this._sectionLoaded[section]||(now-this._sectionLoaded[section])>this._CACHE_TTL;
    if(!stale)return;
    this._sectionLoaded[section]=now;
    if(section==='dashboard')this.loadDashboard();
    else if(section==='monitoring')this.loadMonitoring();
    else if(section==='approvals')this.loadApprovals();
    else if(section==='infra'){this.loadEC2();this.loadAlarms();}
    else if(section==='integrations')this.loadIntegrations();
    else if(section==='users')this.loadUsers();
    else if(section==='security'){this.loadSecrets();this.loadAudit();this.loadWebhookUrls();}
    else if(section==='cost')this.loadCostOverview();
    else if(section==='warroom')this.loadWarRooms();
    else if(section==='incidents')this.loadIncidents();
  },

  refreshCurrent(){
    this._sectionLoaded[this.currentSection]=0;
    this.navigate(this.currentSection,false);
  },

  newIncident(){
    // Navigate to incidents section (bust cache) and reset the form
    this._sectionLoaded['incidents']=0;
    this.navigate('incidents');
    // Reset form fields
    setTimeout(()=>{
      const idEl=document.getElementById('inc-id');
      const descEl=document.getElementById('inc-desc');
      const resultEl=document.getElementById('inc-result');
      if(idEl) idEl.value='';
      if(descEl) descEl.value='';
      if(resultEl) resultEl.innerHTML='';
    },50);
  },

  // ── INIT ──────────────────────────────────────────────────────────
  async init(){
    // Back/forward button support
    window.addEventListener('popstate',e=>{
      const s=(e.state&&e.state.section)||window.location.hash.replace('#','');
      if(s)this.navigate(s,false);
    });
    if(!this.token){this.showLogin();return;}
    this.showApp();
    const initialSection=window.location.hash.replace('#','')||'dashboard';
    this.navigate(initialSection,true);
    this.startAutoRefresh();
  },

  showLogin(){
    document.getElementById('login-screen').style.display='flex';
    document.getElementById('app').style.display='none';
    setTimeout(()=>document.getElementById('login-pass').focus(),100);
  },
  showApp(){
    document.getElementById('login-screen').style.display='none';
    document.getElementById('app').style.display='flex';
    document.getElementById('sidebar-username').textContent=this.username;
    document.getElementById('sidebar-role').textContent=this.role;
    document.getElementById('user-avatar-initial').textContent=(this.username||'A')[0].toUpperCase();
    // Restore global LLM selector from localStorage
    const llmSel=document.getElementById('global-llm-select');
    if(llmSel) llmSel.value=this.globalLLM;
    // Hide admin-only nav items for non-admins
    if(this.role==='viewer'){
      document.querySelectorAll('[data-role-min="developer"]').forEach(el=>el.style.display='none');
    }
  },

  togglePw(){
    const inp=document.getElementById('login-pass');
    const icon=document.getElementById('pw-toggle');
    if(inp.type==='password'){inp.type='text';icon.style.opacity='1';icon.title='Hide password';}
    else{inp.type='password';icon.style.opacity='.5';icon.title='Show password';}
  },

  async login(){
    const btn=document.getElementById('login-btn');
    const err=document.getElementById('login-error');
    const u=document.getElementById('login-user').value.trim();
    const p=document.getElementById('login-pass').value;
    if(!u||!p){err.textContent='Username and password are required.';err.style.display='block';return;}
    btn.disabled=true;document.getElementById('login-btn-text').textContent='Signing in...';err.style.display='none';document.getElementById('login-user').classList.remove('error-field');document.getElementById('login-pass').classList.remove('error-field');
    try{
      const r=await fetch('/auth/token',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'username='+encodeURIComponent(u)+'&password='+encodeURIComponent(p)});
      const d=await r.json();
      if(!r.ok){err.textContent=d.detail||'Invalid credentials. Please try again.';err.style.display='block';document.getElementById('login-pass').classList.add('error-field');document.getElementById('login-pass').focus();return;}
      this.token=d.access_token;this.username=d.username||u;this.role=d.role||'viewer';
      localStorage.setItem('nexusops_token',this.token);
      localStorage.setItem('nexusops_user',this.username);
      localStorage.setItem('nexusops_role',this.role);
      this.showApp();this.navigate('dashboard');this.startAutoRefresh();
    }catch(e){err.textContent='Connection failed — is the server running?';err.style.display='block';}
    finally{btn.disabled=false;document.getElementById('login-btn-text').textContent='Sign In';}
  },

  logout(){
    localStorage.removeItem('nexusops_token');localStorage.removeItem('nexusops_user');localStorage.removeItem('nexusops_role');
    this.token=null;this._sectionLoaded={};clearInterval(this.refreshTimer);
    history.replaceState(null,'',window.location.pathname);
    this.showLogin();
  },

  startAutoRefresh(){
    clearInterval(this.refreshTimer);
    this.refreshTimer=setInterval(()=>{
      this.loadBadges();
      // Only silently refresh dashboard stats if user is on dashboard
      if(this.currentSection==='dashboard'){
        this._sectionLoaded['dashboard']=0;
        this.loadDashboard();
      }
    },60000); // 60s — not 30s to reduce server load
  },

  async loadBadges(){
    try{
      const r=await this.api('GET','/approvals/pending');
      if(r&&r.ok){const d=await r.json();const c=(d.approvals||[]).length;const b=document.getElementById('badge-approvals');if(c>0){b.textContent=c;b.style.display='';}else b.style.display='none';}
    }catch(e){}
  },

  // ── DASHBOARD ────────────────────────────────────────────────────
  async loadDashboard(){
    this.loadBadges();
    // Run all dashboard API calls in parallel for speed
    const [healthR, approvalsR, alarmsR, k8sR, auditR] = await Promise.allSettled([
      this.api('GET','/health/full'),
      this.api('GET','/approvals/pending'),
      this.api('GET','/aws/cloudwatch/alarms'),
      this.api('GET','/check/k8s'),
      this.api('GET','/audit/log?limit=8')
    ]);

    // Health / Integrations
    if(healthR.status==='fulfilled'&&healthR.value?.ok){
      const d=await healthR.value.json();
      document.getElementById('ds-incidents').textContent=d.health?.issue_count??d.incident_count??'0';
      const el=document.getElementById('dash-health');
      const sources=(d.health?.sources||[]);
      const issues=(d.health?.issues||[]);
      const statusCls=d.status==='ok'?'badge-green':d.status==='degraded'?'badge-amber':'badge-red';
      let html=`<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px"><span class="badge ${statusCls}" style="font-size:.8em">${(d.status||'unknown').toUpperCase()}</span>${issues.length?`<span style="font-size:.78em;color:var(--red)">${issues.length} issue${issues.length>1?'s':''}</span>`:''}</div>`;
      if(issues.length){html+=issues.map(i=>`<div style="padding:7px 10px;background:rgba(248,113,113,.07);border-left:2px solid var(--red);border-radius:0 6px 6px 0;font-size:.8em;margin-bottom:6px;color:var(--text2)">${i}</div>`).join('');}
      if(sources.length){html+=`<div style="font-size:.72em;color:var(--muted);margin-top:8px">Sources: ${sources.join(', ')}</div>`;}
      el.innerHTML=html||'<div class="empty-state"><p>Health data unavailable</p></div>';
      // Update topbar status
      const ts=document.getElementById('topbar-status');
      if(d.status==='ok'){ts.className='topbar-status';ts.innerHTML='<span class="status-dot dot-green dot-pulse"></span> All Systems Operational';}
      else if(d.status==='degraded'){ts.className='topbar-status degraded';ts.innerHTML='<span class="status-dot dot-amber"></span> Degraded';}
      else{ts.className='topbar-status error';ts.innerHTML='<span class="status-dot dot-red"></span> Issues Detected';}
    }

    // Approvals
    if(approvalsR.status==='fulfilled'&&approvalsR.value?.ok){
      const d=await approvalsR.value.json();
      const cnt=(d.approvals||[]).length;
      document.getElementById('ds-approvals').textContent=cnt;
      if(cnt>0){document.getElementById('ds-approvals-sub').innerHTML=`<span style="color:var(--amber)">&#9888; Needs review</span>`;}
      else{document.getElementById('ds-approvals-sub').innerHTML=`<span style="color:var(--muted)">None pending</span>`;}
    }

    // AWS Alarms
    if(alarmsR.status==='fulfilled'&&alarmsR.value?.ok){
      const d=await alarmsR.value.json();
      const alarms=(d.cloudwatch_alarms?.alarms||d.alarms||d.data||[]);
      const firing=alarms.filter(a=>(a.state_value||a.StateValue||'')==='ALARM').length;
      document.getElementById('ds-alerts').textContent=firing;
      document.getElementById('ds-alerts-sub').innerHTML=firing>0?`<span style="color:var(--red)">&#128308; ${firing} firing</span>`:`<span style="color:var(--muted)">All clear</span>`;
    } else {
      document.getElementById('ds-alerts').textContent='—';
      document.getElementById('ds-alerts-sub').innerHTML=`<span style="color:var(--muted)">Not connected</span>`;
    }

    // K8s — render as badge, never raw "error" text
    if(k8sR.status==='fulfilled'&&k8sR.value?.ok){
      const d=await k8sR.value.json();
      const st=(d.k8s_check?.status||d.status||'unknown').toLowerCase();
      const det=d.k8s_check?.details||d.message||'';
      const k8sEl=document.getElementById('ds-k8s');
      if(st==='healthy'||st==='ok'){
        k8sEl.innerHTML=`<span class="badge badge-green" style="font-size:1em;padding:4px 12px">Healthy</span>`;
      }else if(st==='error'||st==='not configured'||det.includes('not found')||det.includes('KUBECONFIG')){
        k8sEl.innerHTML=`<span class="badge badge-gray" style="font-size:1em;padding:4px 12px">Not configured</span>`;
        document.getElementById('ds-k8s-sub').innerHTML=`<span style="color:var(--muted)">Set KUBECONFIG to connect</span>`;
      }else{
        k8sEl.innerHTML=`<span class="badge badge-amber" style="font-size:1em;padding:4px 12px">${st}</span>`;
      }
    } else {
      document.getElementById('ds-k8s').innerHTML=`<span class="badge badge-gray" style="font-size:1em;padding:4px 12px">Unavailable</span>`;
    }

    // Audit log / Recent Activity
    if(auditR.status==='fulfilled'&&auditR.value?.ok){
      const d=await auditR.value.json();const logs=d.entries||d.logs||[];
      const el=document.getElementById('dash-activity');
      if(!logs.length){
        el.innerHTML='<div class="empty-state"><div class="empty-icon">&#128336;</div><p>No recent activity</p></div>';
      } else {
        el.innerHTML=`<div style="display:flex;flex-direction:column">`+logs.map(l=>{
          const t=(l.timestamp||l.ts||'').substring(11,19)||'--:--';
          const action=l.action||l.event||'event';
          const user=l.user||'system';
          return`<div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--border);font-size:.82em">
            <span style="color:var(--muted);font-family:'SF Mono','Cascadia Code',ui-monospace,monospace;flex-shrink:0;width:52px">${t}</span>
            <span style="flex:1;font-weight:500">${action}</span>
            <span style="color:var(--text2);font-size:.9em">${user}</span>
          </div>`;
        }).join('')+'</div>';
      }
    } else {
      document.getElementById('dash-activity').innerHTML='<div class="empty-state"><div class="empty-icon">&#128336;</div><p>No recent activity</p></div>';
    }
  },

  // ── MONITORING ───────────────────────────────────────────────────
  async loadMonitoring(){
    const el=document.getElementById('monitoring-alerts');
    el.innerHTML='<div class="loading-state"><div class="spinner"></div> Loading alerts...</div>';
    const alerts=[];const unavailable=[];
    try{const r=await this.api('GET','/grafana/alerts');if(r&&r.ok){const d=await r.json();if(d.success===false){unavailable.push('Grafana: '+(d.error||'unavailable'));}else{(d.alerts||d.data||[]).forEach(a=>alerts.push({source:'grafana',name:a.name||a.labels?.alertname||'Alert',state:a.state||a.status||'firing',msg:a.message||a.annotations?.summary||''}));}}else{unavailable.push('Grafana: not reachable');}}catch(e){unavailable.push('Grafana: '+e.message);}
    try{const r=await this.api('GET','/aws/cloudwatch/alarms');if(r&&r.ok){const d=await r.json();if(d.cloudwatch_alarms?.success===false){unavailable.push('CloudWatch: '+(d.cloudwatch_alarms.error||'unavailable'));}else{(d.cloudwatch_alarms?.alarms||d.alarms||d.data||[]).forEach(a=>alerts.push({source:'cloudwatch',name:a.alarm_name||a.AlarmName||'Alarm',state:a.state_value||a.StateValue||'OK',msg:a.alarm_description||a.AlarmDescription||''}));}}else{unavailable.push('CloudWatch: not reachable');}}catch(e){unavailable.push('CloudWatch: '+e.message);}
    try{const r=await this.api('GET','/check/k8s');if(r&&r.ok){const d=await r.json();const st=d.k8s_check?.status||d.status;const det=d.k8s_check?.details||d.message||'';if(st==='error'){unavailable.push('Kubernetes: '+(det||'not configured'));}else if(st&&st!=='healthy'){alerts.push({source:'k8s',name:'K8s Cluster',state:st,msg:det});}}else{unavailable.push('Kubernetes: not reachable');}}catch(e){unavailable.push('Kubernetes: '+e.message);}
    let html='';
    if(unavailable.length){html+='<div style="margin-bottom:12px;padding:10px 12px;border-radius:8px;background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.2);font-size:.8em;color:var(--amber)"><b>&#9888; Some monitoring sources unavailable:</b><ul style="margin:4px 0 0 16px;padding:0">'+unavailable.map(u=>`<li>${u}</li>`).join('')+'</ul></div>';}
    if(!alerts.length){html+='<div class="empty-state"><div class="empty-icon">&#9989;</div><p>No active alerts from available sources</p></div>';}
    else{html+='<div style="display:flex;flex-direction:column;gap:8px">'+alerts.map(a=>{const srcCls={'grafana':'src-grafana','cloudwatch':'src-cloudwatch','k8s':'src-k8s','opsgenie':'src-opsgenie'}[a.source]||'src-opsgenie';const stateCls=a.state.toLowerCase().includes('alarm')||a.state.toLowerCase().includes('firing')?'badge-red':a.state.toLowerCase().includes('ok')?'badge-green':'badge-amber';return`<div class="alert-item"><span class="alert-source ${srcCls}">${a.source}</span><div style="flex:1"><div style="font-size:.88em;font-weight:500">${a.name}</div><div class="text-muted">${a.msg}</div></div><span class="badge ${stateCls}">${a.state}</span></div>`;}).join('')+'</div>';}
    el.innerHTML=html;
  },

  // ── INCIDENTS ────────────────────────────────────────────────────
  toggleDryRun(){
    this.dryRun=!this.dryRun;
    const t=document.getElementById('dry-run-toggle');
    t.classList.toggle('on',this.dryRun);
    if(this.dryRun)this.toast('Dry Run ON — pipeline will preview actions only','info');
  },

  toggleAutoRemediate(){
    if(!this.autoRemediate){
      if(!confirm('Enable Auto-Remediate?\\n\\nThis will allow the pipeline to automatically execute infrastructure actions (restarts, scaling) without waiting for approval.\\n\\nOnly enable this if you trust the AI to act autonomously on your infrastructure.')){return;}
    }
    this.autoRemediate=!this.autoRemediate;
    const t=document.getElementById('auto-rem-toggle');
    t.classList.toggle('on',this.autoRemediate);
  },

  async runIncident(dryRun=false){
    const isDry=dryRun||this.dryRun;
    const btn=document.getElementById(isDry?'dry-run-btn':'run-inc-btn');
    const desc=document.getElementById('inc-desc').value.trim();
    if(!desc){this.toast('Description is required','error');return;}
    const id=document.getElementById('inc-id').value.trim()||'INC-'+Date.now();
    const sev=document.getElementById('inc-sev').value;
    const hours=parseInt(document.getElementById('inc-hours').value)||2;
    const origHtml=btn.innerHTML;
    btn.disabled=true;btn.innerHTML='<div class="spinner" style="width:14px;height:14px;border-width:2px"></div> '+(isDry?'Previewing...':'Running pipeline...');
    const el=document.getElementById('inc-result');
    el.innerHTML='<div class="result-card"><div class="loading-state"><div class="spinner"></div> '+(isDry?'Generating dry-run preview...':'Analyzing incident — collecting AWS, K8s and GitHub context...')+'</div></div>';
    try{
      const r=await this.api('POST','/incidents/run',{incident_id:id,description:desc,severity:sev,auto_remediate:this.autoRemediate,dry_run:isDry,hours,llm_provider:this.globalLLM,metadata:{user:this.username,role:this.role}});
      if(!r){el.innerHTML='<div class="result-card"><div style="color:var(--red)">&#10005; Request failed</div></div>';return;}
      const d=await r.json();
      if(!r.ok){el.innerHTML='<div class="result-card"><div style="color:var(--red)">&#10005; '+(d.detail||'Pipeline error')+'</div></div>';return;}
      this.renderIncidentResult(el,d,id,isDry);
      this.toast((isDry?'Dry-run preview for ':'Pipeline completed — ')+id,'success');
    }catch(e){el.innerHTML='<div class="result-card"><div style="color:var(--red)">&#10005; '+e.message+'</div></div>';}
    finally{btn.disabled=false;btn.innerHTML=origHtml;}
  },

  renderIncidentResult(el,d,id,isDry=false){
    const plan=d.plan||{};
    const risk=(plan.risk||d.risk_level||'unknown').toLowerCase();
    const riskCls=risk==='critical'?'badge-red':risk==='high'?'badge-red':risk==='medium'?'badge-amber':'badge-green';
    const conf=Math.round((plan.confidence||0)*100);
    const confCls=conf>=70?'high':conf>=40?'med':'low';
    const rootCause=plan.root_cause||plan.summary||d.summary||'Analysis complete.';
    const reasoning=plan.reasoning||'';
    const dataGaps=plan.data_gaps||[];
    const executedActions=d.executed_actions||[];
    const blockedActions=d.blocked_actions||[];
    const status=d.status||'completed';
    const statusCls=status==='completed'?'badge-green':status==='awaiting_approval'?'badge-amber':status==='failed'?'badge-red':'badge-cyan';
    const planActions=plan.actions||[];

    // Merge plan actions with execution results
    const executedTypes=new Set(executedActions.map(a=>a.type));
    const blockedMap={};
    blockedActions.forEach(a=>{blockedMap[a.type]=(a.reason||'requires approval');});

    // Data source availability
    const awsCtx=d.aws_context||{};const k8sCtx=d.k8s_context||{};const ghCtx=d.github_context||{};
    const awsOk=awsCtx._data_available===true;
    const k8sOk=k8sCtx._data_available===true;
    const ghOk=ghCtx._data_available===true;

    // Action type icons + colors
    const actionMeta={
      investigate:  {icon:'&#128269;',label:'Investigate',  color:'#60a5fa'},
      k8s_restart:  {icon:'&#8635;',  label:'K8s Restart',  color:'#22d3ee'},
      k8s_scale:    {icon:'&#9650;',  label:'K8s Scale',    color:'#34d399'},
      aws_restart:  {icon:'&#8635;',  label:'AWS Restart',  color:'#fb923c'},
      aws_scale:    {icon:'&#9650;',  label:'AWS Scale',    color:'#fb923c'},
      slack_notify: {icon:'&#128172;',label:'Slack Notify', color:'#818cf8'},
      create_jira:  {icon:'&#128195;',label:'Create Jira',  color:'#60a5fa'},
      create_pr:    {icon:'&#128257;',label:'Create PR',    color:'#a78bfa'},
      opsgenie_alert:{icon:'&#128680;',label:'OpsGenie Alert',color:'#f87171'},
      runbook:      {icon:'&#128196;',label:'Runbook',      color:'#94a3b8'},
    };

    const stepsHtml=planActions.map((a,i)=>{
      const meta=actionMeta[a.type]||{icon:'&#9654;',label:a.type,color:'var(--text2)'};
      const isExec=executedTypes.has(a.type);
      const blockReason=blockedMap[a.type];
      const numCls=isExec?'executed':blockReason?'blocked':'';
      const numContent=isExec?'&#10003;':blockReason?'!':''+( i+1);
      const desc=a.description||a.message||a.body||'';
      const target=a.target||a.deployment||(a.namespace&&a.deployment?a.namespace+'/'+a.deployment:'')||a.channel||a.summary||'';
      const costDelta=a.estimated_cost_delta||0;
      return `<li class="action-step">
        <div class="step-num ${numCls}">${numContent}</div>
        <div class="step-body">
          <div class="step-type" style="color:${meta.color}">${meta.icon} ${meta.label}</div>
          <div class="step-desc">${desc||'No description provided.'}</div>
          ${target?`<div class="step-target">Target: ${target}</div>`:''}
          ${costDelta?`<div class="step-cost">Estimated cost delta: $${costDelta}/mo</div>`:''}
          ${blockReason?`<div class="step-reason">&#9888; Blocked: ${blockReason}</div>`:''}
        </div>
      </li>`;
    }).join('');

    const dataGapsHtml=dataGaps.length?`<div class="result-data-gap"><strong>&#9888; Missing data for higher confidence:</strong><ul style="margin:4px 0 0 16px;padding:0">${dataGaps.map(g=>`<li>${g}</li>`).join('')}</ul></div>`:'';

    const rid='reasoning-'+Date.now();
    el.innerHTML=`<div class="result-card">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;flex-wrap:wrap">
        <span style="font-weight:700;font-size:.92em">${id}</span>
        <span class="badge ${statusCls}">${status.replace(/_/g,' ')}</span>
        <span class="badge ${riskCls}">${risk} risk</span>
        ${isDry?'<span class="badge badge-cyan">&#128065; DRY RUN</span>':''}
        ${d.requires_human_approval?'<span class="badge badge-amber">&#9888; Awaiting Approval</span>':''}
      </div>

      <!-- Data sources -->
      <div class="data-source-row">
        <span class="data-src-badge ${awsOk?'data-src-ok':'data-src-miss'}">${awsOk?'&#10003;':'&#10005;'} AWS</span>
        <span class="data-src-badge ${k8sOk?'data-src-ok':'data-src-miss'}">${k8sOk?'&#10003;':'&#10005;'} Kubernetes</span>
        <span class="data-src-badge ${ghOk?'data-src-ok':'data-src-miss'}">${ghOk?'&#10003;':'&#10005;'} GitHub</span>
      </div>

      <!-- Root cause -->
      <div class="result-section">
        <div class="result-section-label">&#128269; Root Cause Analysis</div>
        <div class="result-root-cause">${rootCause}</div>
        ${dataGapsHtml}
      </div>

      <!-- Confidence -->
      <div class="result-section">
        <div class="result-section-label">Confidence &mdash; ${conf}%</div>
        <div class="conf-bar"><div class="conf-fill ${confCls}" style="width:${conf}%"></div></div>
      </div>

      <!-- Remediation steps -->
      ${planActions.length?`<div class="result-section">
        <div class="result-section-label">&#9875; Remediation Steps (${planActions.length})</div>
        <ol class="action-steps">${stepsHtml}</ol>
      </div>`:'<div class="result-section"><div class="result-section-label">No actions recommended</div></div>'}

      <!-- Errors -->
      ${d.errors&&d.errors.length?`<div class="result-section"><div class="result-section-label" style="color:var(--red)">&#9888; Pipeline Errors</div><div style="font-size:.8em;color:var(--red);background:rgba(248,113,113,0.08);padding:8px;border-radius:6px">${d.errors.join('<br>')}</div></div>`:''}

      <!-- Reasoning toggle -->
      ${reasoning?`<div class="result-section">
        <button class="btn btn-ghost btn-sm" style="font-size:.74em" onclick="document.getElementById('${rid}').classList.toggle('open');this.textContent=document.getElementById('${rid}').classList.contains('open')?'&#9650; Hide Reasoning':'&#9660; Show Reasoning'">&#9660; Show Reasoning</button>
        <div id="${rid}" class="result-reasoning">${reasoning}</div>
      </div>`:''}

      <!-- Actions -->
      <div style="display:flex;gap:8px;margin-top:14px;flex-wrap:wrap">
        <button class="btn btn-secondary btn-sm" onclick="App.createWarRoomFromIncident('${id}','${(plan.summary||'').replace(/'/g,'&apos;')}')">&#9876; Create War Room</button>
        <button class="btn btn-ghost btn-sm" onclick="App.generatePostMortem('${id}')">&#128221; Post-Mortem</button>
      </div>
    </div>`;
  },

  async loadIncidents(){
    const el=document.getElementById('active-incidents');
    el.innerHTML='<div class="loading-state"><div class="spinner"></div></div>';
    try{
      const r=await this.api('GET','/memory/incidents?limit=5');
      if(r&&r.ok){const d=await r.json();const items=d.incidents||d.results||[];
        if(!items.length){el.innerHTML='<div class="empty-state"><div class="empty-icon">&#9989;</div><p>No incidents in memory</p></div>';return;}
        el.innerHTML=items.map(i=>`<div style="padding:10px;border-bottom:1px solid var(--border);font-size:.83em"><div style="font-weight:600">${i.id||'unknown'}</div><div class="text-muted">${i.description||''}</div></div>`).join('');
      }else el.innerHTML='<div class="empty-state"><div class="empty-icon">&#9989;</div><p>No active incidents</p></div>';
    }catch(e){el.innerHTML='<div class="empty-state"><p>Could not load incidents</p></div>';}
  },

  createWarRoomFromIncident(id,desc){document.getElementById('wr-inc-id').value=id;document.getElementById('wr-desc').value=desc;this.navigate('warroom');},

  async generatePostMortem(id){
    if(!id){this.toast('No incident selected','error');return;}
    this.toast('Generating post-mortem...','info');
    const resp=await this.api('POST',`/incidents/${id}/post-mortem`,{incident_id:id,description:'',root_cause:''});
    const r=resp&&resp.ok?await resp.json():null;
    if(r&&r.markdown){
      this.toast('Post-mortem generated!','success');
      const el=document.getElementById('inc-result');
      if(el)el.innerHTML=`<div class="result-card"><pre style="white-space:pre-wrap;font-size:.82em">${r.markdown.replace(/</g,'&lt;')}</pre></div>`;
    }else{
      this.toast('Post-mortem generation failed — check LLM configuration','error');
    }
  },

  // ── WAR ROOM ─────────────────────────────────────────────────────
  async createWarRoom(){
    const id=document.getElementById('wr-inc-id').value.trim()||'INC-'+Date.now();
    const desc=document.getElementById('wr-desc').value.trim();
    const sev=document.getElementById('wr-sev').value;
    if(!desc){this.toast('Description is required','error');return;}
    try{
      const r=await this.api('POST','/warroom/create',{incident_id:id,description:desc,severity:sev,post_to_slack:false});
      if(r&&r.ok){const d=await r.json();this.toast('War room created!','success');this.loadWarRooms();
        if(d.war_room_id||d.id) this.openWarRoom(d.war_room_id||d.id,id,desc,d.slack_channel||'');
      }else this.toast('Failed to create war room','error');
    }catch(e){this.toast('Error: '+e.message,'error');}
  },

  async loadWarRooms(){
    const el=document.getElementById('warroom-list');
    el.innerHTML='<div class="loading-state"><div class="spinner"></div></div>';
    try{
      const r=await this.api('GET','/warroom/active');
      const d=r&&r.ok?await r.json():{war_rooms:[]};
      const rooms=d.war_rooms||[];
      if(!rooms.length){el.innerHTML='<div class="empty-state"><div class="empty-icon">&#9876;</div><p>No active war rooms</p></div>';return;}
      el.innerHTML=rooms.map(wr=>`<div style="padding:12px;border-bottom:1px solid var(--border);cursor:pointer" onclick="App.openWarRoom('${wr.war_room_id||wr.id}','${wr.incident_id||''}','${(wr.incident_description||'').replace(/'/g,'&apos;')}','${wr.slack_channel||''}')">
        <div style="display:flex;align-items:center;gap:8px;font-size:.88em"><span style="font-weight:600">${wr.incident_id||wr.war_room_id}</span><span class="status-dot dot-green dot-pulse"></span></div>
        <div class="text-muted">${wr.incident_description||''}</div>
      </div>`).join('');
    }catch(e){el.innerHTML='<div class="empty-state"><p>Could not load war rooms</p></div>';}
  },

  openWarRoom(warRoomId,incidentId,desc,slackChannel){
    this.currentWarRoomId=warRoomId;
    this.currentSlackChannel=slackChannel||'';
    document.getElementById('warroom-list-panel').style.display='none';
    document.getElementById('warroom-detail').style.display='flex';
    const slackBadge=slackChannel?`<span class="badge badge-green" style="margin-left:8px">&#35;${slackChannel.replace('#','')}</span>`:'';
    document.getElementById('warroom-info').innerHTML=`<div class="card-header"><div class="card-title"><span class="status-dot dot-green dot-pulse" style="margin-right:4px"></span> War Room: ${incidentId}${slackBadge}</div><button class="btn btn-ghost btn-sm" onclick="App.closeWarRoom()">&#8592; Back</button></div><p class="text-muted">${desc}</p>`;
    document.getElementById('warroom-messages').innerHTML='<div class="empty-state"><p>Ask the AI about this incident</p></div>';
    document.getElementById('slack-messages').innerHTML='<div class="empty-state"><p>Slack channel messages will appear here</p></div>';
    if(slackChannel) this.refreshSlackHistory();
  },

  closeWarRoom(){
    this.currentWarRoomId=null;
    this.currentSlackChannel='';
    if(this._slackPollTimer){clearInterval(this._slackPollTimer);this._slackPollTimer=null;}
    document.getElementById('warroom-list-panel').style.display='flex';
    document.getElementById('warroom-detail').style.display='none';
  },

  async refreshSlackHistory(){
    const ch=this.currentSlackChannel;
    if(!ch){document.getElementById('slack-messages').innerHTML='<div class="empty-state"><p>No Slack channel linked to this war room</p></div>';return;}
    try{
      const r=await this.api('GET',`/warroom/${this.currentWarRoomId}/slack-history?limit=30`);
      if(!r||!r.ok){document.getElementById('slack-messages').innerHTML='<div class="empty-state"><p>Could not load Slack messages</p></div>';return;}
      const d=await r.json();
      const msgs=document.getElementById('slack-messages');
      const messages=d.messages||[];
      if(!messages.length){msgs.innerHTML='<div class="empty-state"><p>No messages yet in '+ch+'</p></div>';return;}
      msgs.innerHTML='';
      messages.forEach(m=>{
        const el=document.createElement('div');
        el.className='chat-bubble '+(m.username===this.username?'user':'assistant');
        const safe=(m.text||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        el.innerHTML='<div>'+safe+'</div><div class="chat-meta">'+m.username+' &bull; '+m.time+'</div>';
        msgs.appendChild(el);
      });
      msgs.scrollTop=msgs.scrollHeight;
    }catch(e){console.error('Slack history error',e);}
  },

  async sendSlackMessage(){
    const input=document.getElementById('slack-input');
    const text=input.value.trim();
    if(!text||!this.currentSlackChannel) return;
    input.value='';
    try{
      const r=await this.api('POST',`/warroom/${this.currentWarRoomId}/slack-send`,{message:text,sent_by:this.username});
      if(r&&r.ok){
        const el=document.createElement('div');
        el.className='chat-bubble user';
        const safe=text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        el.innerHTML='<div>'+safe+'</div><div class="chat-meta">'+this.username+' &bull; '+new Date().toLocaleTimeString()+'</div>';
        const msgs=document.getElementById('slack-messages');
        if(msgs.querySelector('.empty-state')) msgs.innerHTML='';
        msgs.appendChild(el);
        msgs.scrollTop=msgs.scrollHeight;
        setTimeout(()=>this.refreshSlackHistory(),2000);
      } else this.toast('Failed to send Slack message','error');
    }catch(e){this.toast('Slack error: '+e.message,'error');}
  },

  async askWarRoom(){
    const input=document.getElementById('warroom-input');
    const q=input.value.trim();if(!q||!this.currentWarRoomId) return;
    input.value='';
    this.appendWarRoomMsg('user',q);
    this.appendWarRoomMsg('typing','...');
    try{
      const r=await this.api('POST','/warroom/'+this.currentWarRoomId+'/ask',{question:q,asked_by:this.username});
      const msgs=document.getElementById('warroom-messages');
      msgs.querySelector('.typing-bubble')&&msgs.querySelector('.typing-bubble').remove();
      if(r&&r.ok){const d=await r.json();this.appendWarRoomMsg('assistant',d.answer||d.response||'No response');}
      else this.appendWarRoomMsg('assistant','Could not get response from war room AI.');
    }catch(e){this.appendWarRoomMsg('assistant','Error: '+e.message);}
  },

  appendWarRoomMsg(role,text){
    const msgs=document.getElementById('warroom-messages');
    if(role==='typing'){const el=document.createElement('div');el.className='typing-bubble';el.style.cssText='font-size:.8em;color:var(--muted);padding:6px 0';el.textContent='AI is thinking...';msgs.appendChild(el);msgs.scrollTop=msgs.scrollHeight;return;}
    if(msgs.querySelector('.empty-state')) msgs.innerHTML='';
    const el=document.createElement('div');
    el.className='chat-bubble '+role;
    const safe=text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    const fmt=safe.replace(/\\n/g,'<br>').replace(/[*][*](.*?)[*][*]/g,'<strong>$1</strong>').replace(/`(.*?)`/g,'<code>$1</code>');
    el.innerHTML='<div>'+fmt+'</div><div class="chat-meta">'+(role==='user'?this.username:'NexusOps AI')+' &bull; '+new Date().toLocaleTimeString()+'</div>';
    msgs.appendChild(el);msgs.scrollTop=msgs.scrollHeight;
  },

  async suggestNextSteps(){
    if(!this.currentWarRoomId){this.toast('No active war room','error');return;}
    this.appendWarRoomMsg('user','What should we do next?');
    this.appendWarRoomMsg('typing','...');
    const r=await this.api('POST','/warroom/'+this.currentWarRoomId+'/ask',{question:'What should we do next? Give me 3-5 concrete next steps.',asked_by:this.username});
    const msgs=document.getElementById('warroom-messages');
    msgs.querySelector('.typing-bubble')&&msgs.querySelector('.typing-bubble').remove();
    if(r&&r.ok){const d=await r.json();this.appendWarRoomMsg('assistant',d.answer||'No suggestions available.');}
  },

  // ── APPROVALS ────────────────────────────────────────────────────
  async loadApprovals(){
    const el=document.getElementById('approvals-list');
    el.innerHTML='<div class="loading-state"><div class="spinner"></div></div>';
    try{
      const r=await this.api('GET','/approvals/pending');
      if(!r||!r.ok){el.innerHTML='<div class="empty-state"><p>Could not load approvals</p></div>';return;}
      const d=await r.json();const approvals=d.approvals||[];
      if(!approvals.length){el.innerHTML='<div class="empty-state"><div class="empty-icon">&#9989;</div><p>No pending approvals</p></div>';return;}
      el.innerHTML=`<div class="table-wrap"><table><thead><tr><th>Incident</th><th>Risk</th><th>Cost Impact</th><th>Requested By</th><th>Expires</th><th>Actions</th></tr></thead><tbody>`+
        approvals.map(a=>{
          const risk=(a.risk_score||0);const riskCls=risk>=0.8?'badge-red':risk>=0.5?'badge-amber':'badge-green';
          const riskLabel=risk>=0.8?'Critical':risk>=0.5?'High':'Low';
          const cost=a.cost_report?'$'+(a.cost_report.total_estimated_monthly_delta||0).toFixed(0)+'/mo':'N/A';
          const exp=a.expires_at?new Date(a.expires_at).toLocaleTimeString():'--';
          return`<tr><td><span style="font-weight:600">${a.incident_id||a.correlation_id}</span></td><td><span class="badge ${riskCls}">${riskLabel}</span></td><td>${cost}</td><td>${a.requested_by||'system'}</td><td style="color:var(--amber)">${exp}</td><td><button class="btn btn-secondary btn-sm" onclick="App.openApproval('${a.correlation_id}')">Review</button></td></tr>`;
        }).join('')+`</tbody></table></div>`;
    }catch(e){el.innerHTML='<div class="empty-state"><p>Error: '+e.message+'</p></div>';}
  },

  async openApproval(correlationId){
    this.currentApprovalId=correlationId;
    try{
      const r=await this.api('GET','/approvals/pending');
      if(!r||!r.ok) return;
      const d=await r.json();
      const ap=(d.approvals||[]).find(a=>a.correlation_id===correlationId);
      if(!ap){this.toast('Approval not found','error');return;}
      const actions=ap.actions||[];
      document.getElementById('approve-modal-body').innerHTML=`
        <p class="text-muted mb-12">Incident: <strong>${ap.incident_id}</strong> &bull; Risk: ${(ap.risk_score||0).toFixed(2)}</p>
        <p class="text-muted mb-12">Plan: ${ap.plan_summary||'No summary'}</p>
        <div class="mb-16"><div class="result-section-label mb-8">Select actions to approve:</div>
        ${actions.map((a,i)=>`<label style="display:flex;align-items:center;gap:8px;padding:8px;border-radius:6px;cursor:pointer;border:1px solid var(--border);margin-bottom:6px;font-size:.85em"><input type="checkbox" value="${i}" checked style="accent-color:var(--purple)"/> <span class="action-chip">${a.type||'action'}</span> ${a.deployment||a.summary||''}</label>`).join('')}
        </div>
        <div><label class="form-label">Rejection reason (if rejecting)</label><input type="text" id="reject-reason" class="form-input" placeholder="Optional reason..."/></div>`;
      document.getElementById('approve-modal').classList.add('open');
    }catch(e){this.toast('Error: '+e.message,'error');}
  },

  async submitApproval(){
    if(!this.currentApprovalId) return;
    const checks=[...document.querySelectorAll('#approve-modal-body input[type=checkbox]')];
    const indices=checks.filter(c=>c.checked).map(c=>parseInt(c.value));
    try{
      const r=await this.api('POST','/approvals/'+this.currentApprovalId+'/approve',{approved_action_indices:indices});
      if(r&&r.ok){this.toast('Actions approved!','success');this.closeModal('approve-modal');this.loadApprovals();}
      else this.toast('Approval failed','error');
    }catch(e){this.toast('Error: '+e.message,'error');}
  },

  async submitRejection(){
    if(!this.currentApprovalId) return;
    const reason=document.getElementById('reject-reason').value||'Rejected by user';
    try{
      const r=await this.api('POST','/approvals/'+this.currentApprovalId+'/reject',{reason});
      if(r&&r.ok){this.toast('Request rejected','info');this.closeModal('approve-modal');this.loadApprovals();}
    }catch(e){this.toast('Error: '+e.message,'error');}
  },

  closeModal(id){document.getElementById(id).classList.remove('open');},

  // ── COST ─────────────────────────────────────────────────────────
  async loadCostOverview(){
    const sumRow=document.getElementById('cost-summary-row');
    const svcEl=document.getElementById('cost-services');
    const trendEl=document.getElementById('cost-trend');
    sumRow.innerHTML='<div class="loading-state" style="grid-column:1/-1"><div class="spinner"></div> Fetching live AWS cost data...</div>';
    svcEl.innerHTML='<div class="loading-state"><div class="spinner"></div></div>';
    trendEl.innerHTML='<div class="loading-state"><div class="spinner"></div></div>';
    try{
      const r=await this.api('GET','/cost/dashboard');
      if(!r||!r.ok){
        sumRow.innerHTML='<div style="grid-column:1/-1;padding:12px;color:var(--muted)">&#9888; AWS Cost Explorer not available — check credentials and that Cost Explorer is enabled in your AWS account.</div>';
        svcEl.innerHTML='<div class="empty-state"><p>Unavailable</p></div>';
        trendEl.innerHTML='<div class="empty-state"><p>Unavailable</p></div>';
        return;
      }
      const d=await r.json();
      if(!d.available){
        sumRow.innerHTML=`<div style="grid-column:1/-1;padding:12px;background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.2);border-radius:8px;font-size:.83em;color:var(--amber)">&#9888; Cost Explorer unavailable: ${d.error||'AWS credentials missing or Cost Explorer not enabled'}.<br><span style="color:var(--muted)">Enable at: AWS Console → Billing → Cost Explorer → Enable</span></div>`;
        svcEl.innerHTML='<div class="empty-state"><p>Unavailable</p></div>';
        trendEl.innerHTML='<div class="empty-state"><p>Unavailable</p></div>';
        return;
      }
      // Summary cards
      const mtd=d.current_monthly_spend||0;const last=d.last_month_spend||0;const forecast=d.forecast_month_end||0;
      const mtdVsLast=last>0?((mtd/last*30/new Date().getDate()-1)*100):0;
      const trend=mtdVsLast>5?'&#8599;':mtdVsLast<-5?'&#8600;':'&#8594;';
      const trendColor=mtdVsLast>5?'var(--red)':mtdVsLast<-5?'var(--green)':'var(--text2)';
      sumRow.innerHTML=[
        {label:'Month-to-Date',val:'$'+mtd.toFixed(2),sub:'Current billing period',color:'var(--purple)'},
        {label:'Last Month Total',val:'$'+last.toFixed(2),sub:'Previous full month',color:'var(--cyan)'},
        {label:'Forecast (Month-End)',val:forecast?'$'+forecast.toFixed(2):'—',sub:'Projected total',color:'var(--amber)'},
        {label:'MoM Trend',val:`<span style="color:${trendColor}">${trend} ${Math.abs(mtdVsLast).toFixed(1)}%</span>`,sub:'vs last month pace',color:'var(--text2)'},
      ].map(c=>`<div class="card" style="padding:14px">
        <div class="text-muted" style="font-size:.75em;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">${c.label}</div>
        <div style="font-size:1.7em;font-weight:800;color:${c.color};line-height:1.1">${c.val}</div>
        <div class="text-muted" style="font-size:.76em;margin-top:4px">${c.sub}</div>
      </div>`).join('');
      // Service breakdown
      const svcs=d.service_breakdown||[];
      if(!svcs.length){svcEl.innerHTML='<div class="empty-state"><p>No service data</p></div>';}
      else{
        const max=svcs[0].amount_usd||1;
        svcEl.innerHTML='<div style="display:flex;flex-direction:column;gap:8px;padding:4px 0">'+
          svcs.map(s=>{const pct=Math.round(s.amount_usd/mtd*100)||0;const barW=Math.round(s.amount_usd/max*100);
            return`<div style="display:flex;align-items:center;gap:10px;font-size:.83em">
              <div style="width:110px;flex-shrink:0;color:var(--text2)">${s.service}</div>
              <div style="flex:1;background:var(--surface3);border-radius:4px;height:8px;overflow:hidden"><div style="width:${barW}%;height:100%;background:var(--purple);border-radius:4px"></div></div>
              <div style="width:70px;text-align:right;font-weight:600">$${s.amount_usd.toFixed(2)}</div>
              <div style="width:36px;text-align:right;color:var(--muted);font-size:.9em">${pct}%</div>
            </div>`;
          }).join('')+'</div>';
      }
      // Monthly trend
      const trend6=d.monthly_trend||[];
      if(!trend6.length){trendEl.innerHTML='<div class="empty-state"><p>No trend data</p></div>';}
      else{
        const tMax=Math.max(...trend6.map(m=>m.amount_usd),0.01);
        trendEl.innerHTML='<div style="display:flex;align-items:flex-end;gap:6px;padding:12px 0 4px;height:120px">'+
          trend6.map(m=>{const h=Math.max(Math.round(m.amount_usd/tMax*80),4);
            return`<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:4px">
              <div style="font-size:.68em;color:var(--muted)">$${m.amount_usd>=1000?(m.amount_usd/1000).toFixed(1)+'k':m.amount_usd.toFixed(0)}</div>
              <div style="width:100%;background:var(--cyan);border-radius:3px 3px 0 0;height:${h}px;opacity:0.8"></div>
              <div style="font-size:.67em;color:var(--muted)">${m.month.slice(5)}</div>
            </div>`;
          }).join('')+'</div>';
      }
    }catch(e){
      sumRow.innerHTML=`<div style="grid-column:1/-1;color:var(--red)">Error: ${e.message}</div>`;
      svcEl.innerHTML='';trendEl.innerHTML='';
    }
  },

  async analyzeCost(){
    const raw=document.getElementById('cost-actions').value.trim();
    let actions;try{actions=JSON.parse(raw);}catch(e){this.toast('Invalid JSON in actions field','error');return;}
    const el=document.getElementById('cost-result');
    const impactEl=document.getElementById('cost-aws');
    el.innerHTML='<div class="loading-state"><div class="spinner"></div></div>';
    impactEl.innerHTML='<div class="loading-state"><div class="spinner"></div></div>';
    try{
      const r=await this.api('POST','/cost/analyze',{actions});
      if(r&&r.ok){
        const d=await r.json();const rep=d.report||{};
        const delta=rep.total_estimated_monthly_delta||0;const approved=rep.approved!==false;
        el.innerHTML=`<div class="result-card" style="margin-top:12px">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap">
            <span class="badge ${approved?'badge-green':'badge-red'}">${approved?'&#9989; Within budget':'&#10060; Exceeds budget'}</span>
            <span style="font-size:1.15em;font-weight:800;color:${delta>0?'var(--red)':delta<0?'var(--green)':'var(--text2)'}">
              ${delta===0?'No cost change':'$'+Math.abs(delta).toFixed(2)+'/mo '+(delta>0?'increase':'savings')}
            </span>
          </div>
          ${(rep.per_action_costs||[]).length?`<div style="display:flex;flex-direction:column;gap:6px;font-size:.83em">
            ${(rep.per_action_costs||[]).map(a=>`<div style="padding:8px 10px;background:var(--surface2);border-radius:6px;border:1px solid var(--border)">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-weight:600;color:var(--text)">${a.action_type}</span>
                <span style="font-weight:700;color:${a.monthly_delta_usd>0?'var(--red)':a.monthly_delta_usd<0?'var(--green)':'var(--muted)'}">$${(a.monthly_delta_usd>=0?'+':'')}${a.monthly_delta_usd.toFixed(2)}/mo</span>
              </div>
              <div class="text-muted" style="margin-top:3px">${a.description}</div>
              ${a.notes?`<div style="color:var(--muted);margin-top:2px;font-size:.9em">${a.notes}</div>`:''}
            </div>`).join('')}
          </div>`:''}
          ${rep.warnings&&rep.warnings.length?`<div style="margin-top:10px;padding:8px 10px;background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.2);border-radius:6px;font-size:.8em;color:var(--amber)">
            ${rep.warnings.map(w=>'&#9888; '+w).join('<br>')}
          </div>`:''}
        </div>`;
        // Show MTD spend from real AWS in impact panel
        const mtd=rep.current_monthly_spend||0;
        impactEl.innerHTML=mtd?`<div style="text-align:center;padding:16px">
          <div style="font-size:2em;font-weight:800;background:linear-gradient(135deg,var(--purple),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent">$${mtd.toFixed(2)}</div>
          <div class="text-muted" style="font-size:.82em">Month-to-Date AWS Spend</div>
          ${rep.forecast_month_end?`<div style="margin-top:8px;font-size:.85em;color:var(--amber)">&#127362; Forecast month-end: <b>$${rep.forecast_month_end.toFixed(2)}</b></div>`:''}
          <div style="margin-top:10px;font-size:.8em;color:var(--green)">&#9989; Live from AWS Cost Explorer</div>
        </div>`:`<div class="empty-state"><p class="text-muted">AWS Cost Explorer not available</p></div>`;
      }else{
        el.innerHTML='<div class="result-card"><p class="text-muted">Cost analysis unavailable — check AWS credentials</p></div>';
        impactEl.innerHTML='';
      }
    }catch(e){el.innerHTML=`<div class="result-card"><p style="color:var(--red)">Error: ${e.message}</p></div>`;}
  },

  // ── CHAT ─────────────────────────────────────────────────────────
  newChat(){
    this.chatSessionId='sess-'+Date.now();
    const msgs=document.getElementById('chat-messages');
    msgs.innerHTML='';
    const w=document.createElement('div');w.className='chat-welcome';w.id='chat-welcome';
    w.innerHTML='<div class="chat-welcome-icon"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z"/></svg></div><h2>NexusOps AI</h2><p>Ask me anything about your infrastructure.</p>';
    msgs.appendChild(w);
    document.getElementById('chat-session-label').textContent='';
    document.getElementById('chat-chips').style.display='flex';
    this.pendingAction=null;this.pendingParams=null;
  },

  chipChat(msg){document.getElementById('chat-chips').style.display='none';this.sendChatMsg(msg);},

  async sendChat(){
    const input=document.getElementById('chat-input');
    const msg=input.value.trim();if(!msg) return;
    input.value='';input.style.height='';
    this.sendChatMsg(msg);
  },

  _awsRegion:'us-west-2',

  _linkifyResources(s){
    const region=this._awsRegion||'us-east-1';
    // EC2 instance IDs → AWS console
    s=s.replace(/\b(i-[0-9a-f]{8,17})\b/g,function(_,id){
      const url='https://console.aws.amazon.com/ec2/v2/home?region='+region+'#Instances:instanceId='+id;
      return '<a href="'+url+'" target="_blank" rel="noopener" class="resource-link ec2-link" title="Open in AWS Console">'+id+' &#x2197;</a>';
    });
    // Security groups
    s=s.replace(/\b(sg-[0-9a-f]{8,17})\b/g,function(_,id){
      const url='https://console.aws.amazon.com/ec2/v2/home?region='+region+'#SecurityGroup:groupId='+id;
      return '<a href="'+url+'" target="_blank" rel="noopener" class="resource-link sg-link" title="Open in AWS Console">'+id+' &#x2197;</a>';
    });
    // CloudWatch alarms
    s=s.replace(/\b(arn:aws:[a-z0-9:/.+-]+)\b/g,function(_,arn){
      return '<span class="resource-link arn-link" title="ARN: '+arn+'">'+arn.split(':').pop()+' <small title="'+arn+'">&#x24B2;</small></span>';
    });
    // RDS instance IDs (db-xxx)
    s=s.replace(/\b(db-[A-Z0-9]{26})\b/g,function(_,id){
      const url='https://console.aws.amazon.com/rds/home?region='+region+'#database:id='+id;
      return '<a href="'+url+'" target="_blank" rel="noopener" class="resource-link rds-link" title="Open in AWS Console">'+id+' &#x2197;</a>';
    });
    // GitHub commit SHAs (7-40 hex chars that look like commits in context)
    s=s.replace(/\b([0-9a-f]{7,40})\b(?=.*(?:commit|sha|merge|push))/gi,function(_,sha){
      return '<span class="resource-link sha-link" title="Git SHA: '+sha+'">'+sha.slice(0,7)+'</span>';
    });
    // GitHub PR links (e.g. #123 or PR #123)
    s=s.replace(/\\b(?:PR[ ]*)?#(\\d+)\\b/g,function(_,num){
      return '<span class="resource-link pr-link" title="PR/Issue #'+num+'">#'+num+' &#x2197;</span>';
    });
    // Lambda function names after "function" keyword
    s=s.replace(/\\bfunction[s]?[ ]+[*][*]([^*]+)[*][*]/g,function(_,name){
      const url='https://console.aws.amazon.com/lambda/home?region='+region+'#/functions/'+name;
      return 'function <a href="'+url+'" target="_blank" rel="noopener" class="resource-link lambda-link" title="Open in AWS Console"><strong>'+name+'</strong> &#x2197;</a>';
    });
    return s;
  },

  _md(text){
    var s=text,_i,_j;while((_i=s.indexOf('[TOOL_CALL:'))>=0){_j=s.indexOf(']',_i);if(_j<0)break;s=s.slice(0,_i)+s.slice(_j+1);}s=s.trim();
    s=s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    s=s.replace(/```([a-z]*)[ \t]*([^]*?)```/g,function(_,lang,code){
      var id='cb'+Math.random().toString(36).slice(2,7);
      return '<div class="chat-code-block"><div class="chat-code-header"><span>'+(lang||'code')+'</span>'
        +'<button data-t="'+id+'" onclick="navigator.clipboard.writeText(document.getElementById(this.dataset.t).textContent)">Copy</button></div>'
        +'<pre><code class="code-block-code" id="'+id+'">'+code.trim()+'</code></pre></div>';
    });
    s=s.replace(/`([^`]+)`/g,'<code>$1</code>');
    s=s.replace(/[*][*][*]([^]*?)[*][*][*]/g,'<strong><em>$1</em></strong>');
    s=s.replace(/[*][*]([^]*?)[*][*]/g,'<strong>$1</strong>');
    s=s.replace(/[*]([^]+?)[*]/g,'<em>$1</em>');
    s=s.replace(/^[#]{3} (.+)$/gm,'<h3>$1</h3>');
    s=s.replace(/^[#]{2} (.+)$/gm,'<h2>$1</h2>');
    s=s.replace(/^[#] (.+)$/gm,'<h1>$1</h1>');
    s=s.replace(/^[-]{3,}$/gm,'<hr>');
    s=s.replace(/^&gt; (.+)$/gm,'<blockquote>$1</blockquote>');
    s=s.replace(/^[-*] (.+)$/gm,'<li>$1</li>');
    s=s.replace(/(<li>[^]*?<[/]li>)/g,'<ul>$1</ul>');
    s=s.replace(/^[0-9]+[.] (.+)$/gm,'<oli>$1</oli>');
    s=s.replace(/(<oli>[^]*?<[/]oli>)/g,function(m){return '<ol>'+m.replace(/oli>/g,'li>')+'</ol>';});
    s=s.split('\\n\\n').map(function(p){return p.startsWith('<')?p:'<p>'+p.replace(/\\n/g,'<br>')+'</p>';}).join('');
    s=this._linkifyResources(s);
    return s;
  },

  appendChatMsg(role,text,provider){
    document.getElementById('chat-chips').style.display='none';
    const wEl=document.getElementById('chat-welcome');if(wEl)wEl.remove();
    const msgs=document.getElementById('chat-messages');
    const row=document.createElement('div');
    row.className='chat-row '+(role==='user'?'user':'');
    const providerNames={'anthropic':'Claude','openai':'GPT-4','groq':'Groq/Llama','ollama':'Ollama'};
    const providerLabel=role==='assistant'&&provider?(providerNames[provider]||provider):'';
    const initials=role==='user'?(this.username||'U').slice(0,2).toUpperCase():'AI';
    const avatarCls=role==='user'?'user-av':'ai';
    const avatarInner=role==='user'?initials
      :'<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z"/></svg>';
    const metaName=role==='user'?(this.username||'You'):'NexusOps AI'+(providerLabel?' &bull; '+providerLabel:'');
    const bubbleContent=role==='user'
      ?text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      :this._md(text);
    const msgId='m'+Math.random().toString(36).slice(2,8);
    row.innerHTML=
      '<div class="chat-avatar '+avatarCls+'">'+avatarInner+'</div>'
      +'<div class="chat-body">'
        +'<div class="chat-bubble '+(role==='user'?'user':'assistant')+'" id="'+msgId+'">'+bubbleContent+'</div>'
        +'<div class="chat-meta">'
          +metaName+' &bull; '+new Date().toLocaleTimeString()
          +(role==='assistant'?'<button class="chat-copy-btn" data-t="'+msgId+'" onclick="navigator.clipboard.writeText(document.getElementById(this.dataset.t).innerText)" title="Copy">&#128203;</button>':'')
        +'</div>'
      +'</div>';
    msgs.appendChild(row);msgs.scrollTop=msgs.scrollHeight;
  },

  async sendChatMsg(msg){
    if(!this.chatSessionId) this.chatSessionId='sess-'+Date.now();
    this.appendChatMsg('user',msg);
    const ti=document.getElementById('typing-indicator');ti.style.display='flex';
    document.getElementById('chat-session-label').textContent='Session '+this.chatSessionId.slice(-8);
    const body={message:msg,session_id:this.chatSessionId,provider:this.globalLLM};
    if(this.pendingAction){body.pending_action=this.pendingAction;body.pending_params=this.pendingParams;}
    if(['yes','yeah','yep','sure','ok','proceed','go ahead','confirm','do it'].includes(msg.toLowerCase().trim())&&this.pendingAction) body.confirmed=true;
    try{
      const r=await this.api('POST','/chat',body);
      ti.style.display='none';
      if(!r){this.appendChatMsg('assistant','Connection failed');return;}
      const d=await r.json();
      if(!r.ok){this.appendChatMsg('assistant','Error: '+(d.detail||'Unknown error'));return;}
      const reply=d.reply||d.answer||d.response||'No response.';
      const providerUsed=d.llm_provider||'';
      this.appendChatMsg('assistant',reply,providerUsed);
      this.chatSessionId=d.session_id||this.chatSessionId;
      this.pendingAction=d.needs_confirm?d.pending_action:null;
      this.pendingParams=d.needs_confirm?d.pending_params:null;
    }catch(e){ti.style.display='none';this.appendChatMsg('assistant','Error: '+e.message);}
  },

  // ── INFRASTRUCTURE ───────────────────────────────────────────────
  infraTab(tab,btn){
    document.getElementById('infra-aws').style.display=tab==='aws'?'block':'none';
    document.getElementById('infra-k8s').style.display=tab==='k8s'?'block':'none';
    document.querySelectorAll('#infra-tabs .tab-pill').forEach(b=>{b.classList.remove('active');});
    btn.classList.add('active');
    if(tab==='k8s'){this.loadK8sHealth();this.loadPods();}
  },

  async loadEC2(){
    const el=document.getElementById('ec2-list');
    el.innerHTML='<div class="loading-state"><div class="spinner"></div></div>';
    try{
      const r=await this.api('GET','/aws/ec2/instances');
      if(!r||!r.ok){el.innerHTML='<div class="empty-state"><p>AWS not configured</p></div>';return;}
      const d=await r.json();const ec2d=d.ec2_instances||d;const instances=ec2d.instances||d.instances||d.data||[];
      if(!instances.length){el.innerHTML='<div class="empty-state"><div class="empty-icon">&#9729;</div><p>No EC2 instances</p></div>';return;}
      el.innerHTML=`<div class="table-wrap"><table><thead><tr><th>ID</th><th>Name</th><th>State</th><th>Type</th><th>IP</th><th>Actions</th></tr></thead><tbody>`+
        instances.map(i=>{const stateStr=i.state||i.State?.Name||'unknown';const stateCls=stateStr==='running'?'badge-green':stateStr==='stopped'?'badge-gray':'badge-amber';
          const name=i.name||i.Name||i.instance_id||i.id||'--';
          const iid=i.id||i.instance_id||'';
          const running=stateStr==='running';const stopped=stateStr==='stopped';
          return`<tr><td><code style="font-size:.8em">${iid||'--'}</code></td><td>${name}</td><td><span class="badge ${stateCls}">${stateStr}</span></td><td>${i.type||i.instance_type||'--'}</td><td>${i.public_ip||i.private_ip||'--'}</td><td style="white-space:nowrap">
            ${stopped?`<button class="btn btn-ghost btn-sm" style="color:var(--green);font-size:.75em;padding:3px 7px" onclick="App.ec2Action('${iid}','start')">&#9654; Start</button>`:''}
            ${running?`<button class="btn btn-ghost btn-sm" style="color:var(--amber);font-size:.75em;padding:3px 7px" onclick="App.ec2Action('${iid}','stop')">&#9646;&#9646; Stop</button>`:''}
            ${running?`<button class="btn btn-ghost btn-sm" style="color:var(--cyan);font-size:.75em;padding:3px 7px" onclick="App.ec2Action('${iid}','reboot')">&#8635; Reboot</button>`:''}
          </td></tr>`;
        }).join('')+`</tbody></table></div>`;
    }catch(e){el.innerHTML='<div class="empty-state"><p>Error: '+e.message+'</p></div>';}
  },

  async ec2Action(instanceId,action){
    if(!confirm(`Are you sure you want to ${action} instance ${instanceId}?`))return;
    this.toast(`Sending ${action} to ${instanceId}...`,'info');
    try{
      const r=await this.api('POST',`/aws/ec2/${instanceId}/${action}`);
      if(r&&r.ok){this.toast(`Instance ${instanceId} ${action} initiated`,'success');setTimeout(()=>this.loadEC2(),3000);}
      else{const d=r?await r.json():{};this.toast(d.detail||`Failed to ${action} instance`,'error');}
    }catch(e){this.toast('Error: '+e.message,'error');}
  },

  async loadAlarms(){
    const el=document.getElementById('alarms-list');
    el.innerHTML='<div class="loading-state"><div class="spinner"></div></div>';
    try{
      const r=await this.api('GET','/aws/cloudwatch/alarms');
      if(!r||!r.ok){el.innerHTML='<div class="empty-state"><p>CloudWatch not configured</p></div>';return;}
      const d=await r.json();const cwd=d.cloudwatch_alarms||d;const alarms=cwd.alarms||d.alarms||d.data||[];
      if(!alarms.length){el.innerHTML='<div class="empty-state"><div class="empty-icon">&#9989;</div><p>No alarms</p></div>';return;}
      el.innerHTML=alarms.slice(0,10).map(a=>{const state=a.state_value||a.StateValue||'OK';const cls=state==='ALARM'?'badge-red':state==='OK'?'badge-green':'badge-amber';
        return`<div style="display:flex;align-items:center;gap:8px;padding:8px 0;border-bottom:1px solid var(--border);font-size:.83em"><span class="badge ${cls}">${state}</span><span style="flex:1">${a.alarm_name||a.AlarmName||'Alarm'}</span></div>`;
      }).join('');
    }catch(e){el.innerHTML='<div class="empty-state"><p>Error: '+e.message+'</p></div>';}
  },

  async loadK8sHealth(){
    const el=document.getElementById('k8s-health');
    el.innerHTML='<div class="loading-state"><div class="spinner"></div></div>';
    try{
      const r=await this.api('GET','/check/k8s');
      if(!r||!r.ok){el.innerHTML='<div class="empty-state"><p>K8s not configured</p></div>';return;}
      const d=await r.json();
      const status=d.status||'unknown';const cls=status==='healthy'?'dot-green':status==='degraded'?'dot-amber':'dot-red';
      el.innerHTML=`<div style="padding:12px 0">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px"><span class="status-dot ${cls}"></span><span style="font-size:1.1em;font-weight:700;text-transform:capitalize">${status}</span></div>
        ${d.nodes?`<div class="flex-center gap-8 mb-8 text-sm"><span class="text-muted">Nodes:</span><span class="badge badge-green">${d.nodes.ready||0} ready</span><span class="badge badge-red">${d.nodes.not_ready||0} not ready</span></div>`:''}
        ${d.pods?`<div class="flex-center gap-8 text-sm"><span class="text-muted">Pods:</span><span class="badge badge-green">${d.pods.running||0} running</span><span class="badge badge-red">${d.pods.failed||0} failed</span></div>`:''}
        ${d.message?`<div class="text-muted" style="margin-top:10px;font-size:.8em">${d.message}</div>`:''}
      </div>`;
    }catch(e){el.innerHTML='<div class="empty-state"><p>K8s: '+e.message+'</p></div>';}
  },

  async loadPods(){
    const el=document.getElementById('pods-list');
    el.innerHTML='<div class="loading-state"><div class="spinner"></div></div>';
    try{
      const r=await this.api('GET','/k8s/pods');
      if(!r||!r.ok){el.innerHTML='<div class="empty-state"><p>K8s not configured</p></div>';return;}
      const d=await r.json();const k8sd=d.k8s_pods||d;if(k8sd.status==='error'){el.innerHTML=`<div class="empty-state"><p>K8s: ${k8sd.details||'not configured'}</p></div>`;return;}const pods=k8sd.pods||d.pods||d.data||[];
      if(!pods.length){el.innerHTML='<div class="empty-state"><p>No pods found</p></div>';return;}
      el.innerHTML=`<div class="table-wrap"><table><thead><tr><th>Name</th><th>Namespace</th><th>Status</th><th>Actions</th></tr></thead><tbody>`+
        pods.slice(0,15).map(p=>{const status=p.status||p.phase||'unknown';const cls=status==='Running'?'badge-green':status==='Pending'?'badge-amber':status==='Succeeded'?'badge-cyan':'badge-red';
          const dep=p.deployment||p.name||'';const ns=p.namespace||'default';
          return`<tr><td style="font-size:.78em;font-family:'SF Mono','Cascadia Code',ui-monospace,monospace">${p.name||'--'}</td><td>${ns}</td><td><span class="badge ${cls}">${status}</span></td><td><button class="btn btn-ghost btn-sm" style="font-size:.75em;padding:3px 7px;color:var(--cyan)" onclick="App.k8sRestart('${ns}','${dep}')">&#8635; Restart</button></td></tr>`;
        }).join('')+`</tbody></table></div>`;
    }catch(e){el.innerHTML='<div class="empty-state"><p>Error: '+e.message+'</p></div>';}
  },

  async k8sRestart(namespace,deployment){
    if(!deployment){this.toast('Cannot determine deployment name','error');return;}
    if(!confirm(`Restart deployment "${deployment}" in namespace "${namespace}"?`))return;
    this.toast(`Restarting ${deployment}...`,'info');
    try{
      const r=await this.api('POST','/k8s/restart',{namespace,deployment});
      if(r&&r.ok){this.toast(`Deployment ${deployment} restart triggered`,'success');setTimeout(()=>this.loadPods(),3000);}
      else{const d=r?await r.json():{};this.toast(d.detail||'Restart failed','error');}
    }catch(e){this.toast('Error: '+e.message,'error');}
  },

  // ── INTEGRATIONS ─────────────────────────────────────────────────
  async loadIntegrations(){
    const el=document.getElementById('integrations-grid');
    el.innerHTML='<div class="loading-state" style="grid-column:1/-1"><div class="spinner"></div></div>';
    const intDefs=[
      {key:'aws',name:'AWS',icon:'&#9729;',color:'#f59e0b'},
      {key:'github',name:'GitHub',icon:'&#128049;',color:'#94a3b8'},
      {key:'slack',name:'Slack',icon:'&#128172;',color:'#34d399'},
      {key:'jira',name:'Jira',icon:'&#128202;',color:'#60a5fa'},
      {key:'opsgenie',name:'OpsGenie',icon:'&#128680;',color:'#f87171'},
      {key:'grafana',name:'Grafana',icon:'&#128200;',color:'#f59e0b'},
      {key:'k8s',name:'Kubernetes',icon:'&#128736;',color:'#22d3ee'},
      {key:'gitlab',name:'GitLab',icon:'&#128049;',color:'#fb923c'},
    ];
    try{
      const r=await this.api('GET','/health/integrations');
      const d=r&&r.ok?await r.json():{integrations:{}};
      const integs=d.integrations||d||{};
      el.innerHTML=intDefs.map(i=>{
        const ok=integs[i.key]===true||integs[i.key+'_configured']===true||integs[i.name.toLowerCase()]===true;
        return`<div class="integration-card">
          <div class="int-header"><div class="int-icon" style="background:${i.color}22;color:${i.color}">${i.icon}</div><div><div class="int-name">${i.name}</div></div></div>
          <div class="int-status"><span class="status-dot ${ok?'dot-green':'dot-gray'}"></span>${ok?'Configured':'Not configured'}</div>
        </div>`;
      }).join('');
    }catch(e){el.innerHTML='<div class="empty-state" style="grid-column:1/-1"><p>Could not load integrations</p></div>';}
  },

  // ── USERS ────────────────────────────────────────────────────────
  async loadUsers(){
    const el=document.getElementById('users-table');
    el.innerHTML='<div class="loading-state"><div class="spinner"></div></div>';
    try{
      const r=await this.api('GET','/users');
      if(!r||!r.ok){el.innerHTML='<div class="empty-state"><p>Access denied</p></div>';return;}
      const d=await r.json();const users=d.users||[];
      if(!users.length){el.innerHTML='<div class="empty-state"><p>No users</p></div>';return;}
      el.innerHTML=`<div class="table-wrap"><table><thead><tr><th>Username</th><th>Role</th><th>Created</th><th>Actions</th></tr></thead><tbody>`+
        users.map(u=>{const roleCls=u.role==='admin'?'badge-purple':u.role==='developer'?'badge-cyan':'badge-gray';
          return`<tr><td><div style="display:flex;align-items:center;gap:8px"><div style="width:26px;height:26px;border-radius:50%;background:linear-gradient(135deg,var(--purple),var(--cyan2));display:flex;align-items:center;justify-content:center;font-size:.75em;font-weight:700">${(u.username||'?')[0].toUpperCase()}</div>${u.username}</div></td>
          <td><span class="badge ${roleCls}">${u.role}</span></td>
          <td class="text-muted">${(u.created_at||'').substring(0,10)||'--'}</td>
          <td><button class="btn btn-danger btn-sm" onclick="App.deleteUser('${u.username}')" ${u.username===this.username?'disabled':''}>Delete</button></td></tr>`;
        }).join('')+`</tbody></table></div>`;
    }catch(e){el.innerHTML='<div class="empty-state"><p>Error: '+e.message+'</p></div>';}
  },

  async deleteUser(username){
    if(!confirm('Delete user '+username+'?')) return;
    const r=await this.api('DELETE','/users/'+username);
    if(r&&r.ok){this.toast('User deleted','success');this.loadUsers();}
    else this.toast('Failed to delete user','error');
  },

  openInviteModal(){document.getElementById('invite-modal').classList.add('open');},

  async submitInvite(){
    const username=document.getElementById('invite-username').value.trim();
    const email=document.getElementById('invite-email').value.trim();
    const role=document.getElementById('invite-role').value;
    if(!username){this.toast('Username required','error');return;}
    try{
      const r=await this.api('POST','/users/invite',{username,email,role});
      if(r&&r.ok){const d=await r.json();this.toast('Invite sent! OTP: '+d.otp,'success');this.closeModal('invite-modal');this.loadUsers();}
      else this.toast('Failed to create invite','error');
    }catch(e){this.toast('Error: '+e.message,'error');}
  },

  // ── SECURITY ─────────────────────────────────────────────────────
  async loadSecrets(){
    const el=document.getElementById('secrets-list');
    el.innerHTML='<div class="loading-state"><div class="spinner"></div></div>';
    try{
      const r=await this.api('GET','/secrets/status');
      if(!r||!r.ok){el.innerHTML='<div class="empty-state"><p>Could not load secrets</p></div>';return;}
      const d=await r.json();const secrets=d.secrets||d||{};
      el.innerHTML=Object.entries(secrets).map(([k,v])=>`<div style="display:flex;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid var(--border);font-size:.85em"><span class="status-dot ${v?'dot-green':'dot-red'}"></span><span style="flex:1;font-family:'SF Mono','Cascadia Code',ui-monospace,monospace;font-size:.88em">${k}</span><span class="badge ${v?'badge-green':'badge-red'}">${v?'Configured':'Missing'}</span></div>`).join('');
    }catch(e){el.innerHTML='<div class="empty-state"><p>Error loading secrets</p></div>';}
  },

  async loadAudit(){
    const el=document.getElementById('audit-list');
    el.innerHTML='<div class="loading-state"><div class="spinner"></div></div>';
    try{
      const r=await this.api('GET','/audit/log?limit=10');
      if(!r||!r.ok){el.innerHTML='<div class="empty-state"><p>Could not load audit log</p></div>';return;}
      const d=await r.json();const logs=d.entries||d.logs||[];
      if(!logs.length){el.innerHTML='<div class="empty-state"><p>No audit entries</p></div>';return;}
      el.innerHTML=`<div class="table-wrap"><table><thead><tr><th>Time</th><th>User</th><th>Action</th><th>Result</th></tr></thead><tbody>`+
        logs.map(l=>`<tr><td class="text-muted" style="font-size:.78em;font-family:'SF Mono','Cascadia Code',ui-monospace,monospace">${(l.timestamp||l.ts||'').substring(11,19)||'--'}</td><td>${l.user||'system'}</td><td>${l.action||l.event||'--'}</td><td><span class="badge ${(l.result&&(l.result.success||l.result.ok))?'badge-green':'badge-gray'}">${(l.result&&(l.result.success||l.result.ok))?'ok':'--'}</span></td></tr>`).join('')+`</tbody></table></div>`;
    }catch(e){el.innerHTML='<div class="empty-state"><p>Error loading audit log</p></div>';}
  },

  loadWebhookUrls(){
    const base=window.location.origin;
    document.getElementById('webhook-urls').innerHTML=`
      <div style="display:flex;flex-direction:column;gap:8px">
        ${[['Grafana','/webhooks/grafana'],['CloudWatch (SNS)','/webhooks/cloudwatch'],['OpsGenie','/webhooks/opsgenie'],['PagerDuty','/webhooks/pagerduty']].map(([name,path])=>`
        <div style="display:flex;align-items:center;gap:10px;padding:10px;background:var(--surface2);border-radius:8px;border:1px solid var(--border)">
          <span style="font-size:.85em;font-weight:600;width:130px;flex-shrink:0">${name}</span>
          <code style="flex:1;font-size:.78em;color:var(--cyan);font-family:'SF Mono','Cascadia Code',ui-monospace,monospace">${base+path}</code>
          <button class="btn btn-ghost btn-sm" onclick="navigator.clipboard.writeText('${base+path}').then(()=>App.toast('Copied!','success'))">Copy</button>
        </div>`).join('')}
      </div>`;
  },

};

document.addEventListener('DOMContentLoaded', () => App.init());
</script>
</body>
</html>
"""
    _html = _html.replace("'{{ aws_region }}'", f"'{_aws_region}'")
    return HTMLResponse(_html, headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


# ── Auth helpers ──────────────────────────────────────────────────────────────

_bearer_scheme = HTTPBearer(auto_error=False)

class AuthContext:
    def __init__(self, username: str, role: str):
        self.username = username
        self.role = role

def _resolve_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    x_user: Optional[str] = Header(default=None),
) -> AuthContext:
    from app.security.rbac import get_user_role
    username = None
    jwt_role = None
    token_provided = bool(credentials and credentials.credentials)
    if token_provided:
        try:
            from app.core.auth import decode_token
            payload = decode_token(credentials.credentials)
            username = payload.get("sub")
            jwt_role = payload.get("role")
        except HTTPException:
            # Token was sent but is invalid/expired — mark as unauthenticated
            # require_developer / require_admin will raise 401 to trigger browser re-login
            username = None
            jwt_role = None
    if not username and x_user:
        username = x_user.strip().lower()
    if not username:
        username = "anonymous"
    role = jwt_role if jwt_role else get_user_role(username)
    # Attach whether a (bad) token was attempted — used by stricter guards
    ctx = AuthContext(username=username, role=role)
    ctx._bad_token = token_provided and username == "anonymous" and not jwt_role
    return ctx

def require_admin(auth: AuthContext = Depends(_resolve_auth)) -> AuthContext:
    if getattr(auth, "_bad_token", False):
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    if auth.role not in ("admin",):
        raise HTTPException(status_code=403, detail="Admin access required")
    return auth

def require_developer(auth: AuthContext = Depends(_resolve_auth)) -> AuthContext:
    if getattr(auth, "_bad_token", False):
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    if auth.role not in ("admin", "developer"):
        raise HTTPException(status_code=403, detail="Role 'developer' or 'admin' required")
    return auth

def require_viewer(auth: AuthContext = Depends(_resolve_auth)) -> AuthContext:
    # Viewer endpoints allow anonymous access — even a bad token degrades gracefully
    if auth.role not in ("admin", "developer", "viewer"):
        raise HTTPException(status_code=403, detail="Authentication required")
    return auth

def optional_auth(auth: AuthContext = Depends(_resolve_auth)) -> Optional[AuthContext]:
    """Returns auth context if valid token provided, None otherwise (no 401)."""
    if auth.role in ("admin", "developer", "viewer"):
        return auth
    return None

# ── /auth/me ──────────────────────────────────────────────────────────────────

@app.get("/auth/me")
def auth_me(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    user: str = "nagaraj",
    x_user: Optional[str] = Header(default=None),
):
    """Return role and permissions for a user. Supports JWT Bearer token."""
    from app.security.rbac import ROLE_PERMISSIONS, get_user_role
    # Try JWT first
    username = None
    if credentials and credentials.credentials:
        try:
            from app.core.auth import decode_token
            payload = decode_token(credentials.credentials)
            username = payload.get("sub")
        except Exception:
            pass
    if not username:
        username = (x_user or user or "nagaraj").strip().lower()
    role = get_user_role(username)
    perms = list(ROLE_PERMISSIONS.get(role, set()))
    return {"username": username, "user": username, "role": role, "permissions": perms}


@app.post("/auth/token", tags=["auth"])
def login(form: OAuth2PasswordRequestForm = Depends()):
    from app.security.users import authenticate, user_exists
    from app.security.rbac import get_user_role
    from app.core.auth import create_token
    username = form.username.strip().lower()
    if not user_exists(username) or not authenticate(username, form.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    role = get_user_role(username)
    token = create_token(username, role)
    return {"access_token": token, "token_type": "bearer", "role": role, "username": username}


class UserCreateRequest(BaseModel):
    username: str
    password: str = "INVITE_PENDING"
    role: str = "viewer"
    email: Optional[str] = None

class PasswordChangeRequest(BaseModel):
    new_password: str

@app.get("/users", tags=["users"])
def list_users_endpoint(auth: AuthContext = Depends(require_admin)):
    from app.security.users import list_users as _list
    return {"users": _list()}

@app.post("/users", tags=["users"])
def create_user_endpoint(req: UserCreateRequest, auth: AuthContext = Depends(require_admin)):
    from app.security.users import create_user as _create
    from app.security.rbac import assign_role
    result = _create(req.username, req.password, created_by=auth.username)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    assign_role(req.username, req.role)
    return {"success": True, "username": req.username.lower(), "role": req.role}

@app.delete("/users/{username}", tags=["users"])
def delete_user_endpoint(username: str, auth: AuthContext = Depends(require_admin)):
    if username.strip().lower() == auth.username:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    from app.security.users import delete_user as _delete
    from app.security.rbac import revoke_role
    result = _delete(username)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["error"])
    revoke_role(username)
    return {"success": True, "username": username}

@app.put("/users/{username}/role", tags=["users"])
def set_user_role_endpoint(username: str, req: RoleAssignment, auth: AuthContext = Depends(require_admin)):
    from app.security.rbac import assign_role
    result = assign_role(username, req.role)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("reason", "Failed"))
    return result

@app.put("/users/{username}/password", tags=["users"])
def reset_password_endpoint(username: str, req: PasswordChangeRequest, auth: AuthContext = Depends(require_admin)):
    from app.security.users import change_password
    result = change_password(username, req.new_password)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/users/invite", tags=["users"])
def invite_user_endpoint(req: UserCreateRequest, auth: AuthContext = Depends(require_admin)):
    from app.security.invite import create_invite, send_invite_email
    from app.security.users import create_user as _create
    from app.security.rbac import assign_role
    result = _create(req.username, "INVITE_PENDING", created_by=auth.username)
    if not result["success"] and "already" not in result.get("error", "").lower():
        raise HTTPException(status_code=400, detail=result["error"])
    assign_role(req.username, req.role)
    email = req.email or req.username + "@company.com"
    invite = create_invite(req.username, email)
    email_result = send_invite_email(email, req.username, invite["otp"], invite["token"])
    email_sent = isinstance(email_result, dict) and email_result.get("success") is True
    import os as _os
    app_url = _os.getenv("APP_URL", "http://localhost:8000")
    return {"success": True, "username": req.username, "otp": invite["otp"],
            "email_sent": email_sent,
            "setup_link": f"{app_url}/auth/setup-password?token={invite['token']}"}


@app.get("/auth/setup-password", response_class=HTMLResponse, include_in_schema=False)
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
  <title>Set Your Password — NexusOps</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>
  <style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{font-family:Inter,sans-serif;background:#04060f;color:#e2e8f0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}}
    @keyframes bg{{0%{{background-position:0% 50%}}50%{{background-position:100% 50%}}100%{{background-position:0% 50%}}}}
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
      <span class="logo-text">NexusOps</span>
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


class SetupPasswordRequest(BaseModel):
    token: str
    otp: str
    new_password: str

@app.post("/auth/setup-password", tags=["auth"])
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


class SmtpConfigRequest(BaseModel):
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    app_url: str = "http://localhost:8000"

@app.post("/auth/configure-smtp", tags=["auth"])
def configure_smtp(req: SmtpConfigRequest, auth: AuthContext = Depends(require_admin)):
    """Save SMTP settings to .env and test the connection. Admin only."""
    import smtplib
    updates = {}
    if req.smtp_host:    updates["SMTP_HOST"]     = req.smtp_host
    if req.smtp_user:    updates["SMTP_USER"]     = req.smtp_user
    if req.smtp_password: updates["SMTP_PASSWORD"] = req.smtp_password
    if req.smtp_from or req.smtp_user:
        updates["SMTP_FROM"] = req.smtp_from or req.smtp_user
    updates["SMTP_PORT"] = str(req.smtp_port)
    if req.app_url:      updates["APP_URL"]       = req.app_url
    _write_env(updates)
    # reload env vars in this process
    import os
    for k, v in updates.items():
        os.environ[k] = v
    # test connection
    if req.smtp_host and req.smtp_user and req.smtp_password:
        try:
            with smtplib.SMTP(req.smtp_host, req.smtp_port, timeout=8) as s:
                s.ehlo(); s.starttls(); s.login(req.smtp_user, req.smtp_password)
            return {"success": True, "message": "SMTP configured and connection verified"}
        except Exception as e:
            return {"success": False, "message": f"Settings saved but SMTP test failed: {e}"}
    return {"success": True, "message": "SMTP settings saved (no test — fill all fields to verify)"}

@app.post("/auth/test-email", tags=["auth"])
def test_email(auth: AuthContext = Depends(require_admin)):
    """Send a test email to the configured SMTP_USER address."""
    from app.security.invite import send_invite_email
    import os
    to = os.getenv("SMTP_USER", "")
    if not to or "@" not in to:
        raise HTTPException(400, detail="SMTP_USER not configured — set it in .env or via /auth/configure-smtp")
    result = send_invite_email(to, auth.username, "123456", "test-token")
    if result.get("success"):
        return {"success": True, "message": f"Test email sent to {to}"}
    raise HTTPException(400, detail=result.get("error", "Failed to send test email"))


@app.get("/health")
def health():
    incident_count = 0
    try:
        from app.memory.vector_db import search_similar_incidents
        results = search_similar_incidents("incident", n_results=100)
        incident_count = len(results) if isinstance(results, list) else 0
    except Exception:
        pass
    return {"status": "ok", "incident_count": incident_count, "version": "2.0.0"}


@app.get("/health/live", tags=["health"])
def health_live():
    """Liveness probe — returns 200 if the process is alive."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["health"])
def health_ready():
    """Readiness probe — checks ChromaDB, required env vars, and module imports."""
    from fastapi.responses import JSONResponse
    checks: dict = {}
    all_ok = True

    # 1. Core module imports
    try:
        from app.llm import claude as _claude_mod  # noqa: F401
        from app.memory import vector_db as _vdb_mod  # noqa: F401
        checks["modules"] = "ok"
    except Exception as exc:
        checks["modules"] = f"error: {exc}"
        all_ok = False

    # 2. ChromaDB accessible
    try:
        from app.memory.vector_db import search_similar_incidents
        search_similar_incidents("probe", n_results=1)
        checks["chroma"] = "ok"
    except Exception as exc:
        checks["chroma"] = f"error: {exc}"
        all_ok = False

    # 3. At least one LLM key present
    import os as _os
    llm_keys = ["ANTHROPIC_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"]
    checks["llm_key"] = "ok" if any(_os.getenv(k) for k in llm_keys) else "missing"
    if checks["llm_key"] != "ok":
        all_ok = False

    status = "ready" if all_ok else "not_ready"
    code = 200 if all_ok else 503
    return JSONResponse(status_code=code, content={"status": status, "checks": checks})


@app.get("/metrics", tags=["observability"])
def prometheus_metrics(auth: AuthContext = Depends(require_viewer)):
    """Prometheus text-format metrics endpoint."""
    from fastapi.responses import PlainTextResponse

    lines: list[str] = []

    # ── nexusops_requests_total ───────────────────────────────
    lines.append("# HELP nexusops_requests_total Total HTTP requests")
    lines.append("# TYPE nexusops_requests_total counter")
    for key, val in _METRICS.items():
        if key.startswith("nexusops_requests_total"):
            lines.append(f"{key} {val}")

    # ── nexusops_errors_total ─────────────────────────────────
    lines.append("# HELP nexusops_errors_total Total HTTP errors (4xx/5xx)")
    lines.append("# TYPE nexusops_errors_total counter")
    for key, val in _METRICS.items():
        if key.startswith("nexusops_errors_total"):
            lines.append(f"{key} {val}")

    # ── nexusops_incidents_total ──────────────────────────────
    lines.append("# HELP nexusops_incidents_total Total incidents processed")
    lines.append("# TYPE nexusops_incidents_total counter")
    lines.append(f"nexusops_incidents_total {_METRICS.get('nexusops_incidents_total', 0)}")

    # ── nexusops_active_warrooms ──────────────────────────────
    lines.append("# HELP nexusops_active_warrooms Currently active war rooms")
    lines.append("# TYPE nexusops_active_warrooms gauge")
    lines.append(f"nexusops_active_warrooms {_METRICS.get('nexusops_active_warrooms', 0)}")

    # ── nexusops_llm_calls_total ──────────────────────────────
    lines.append("# HELP nexusops_llm_calls_total Total LLM API calls")
    lines.append("# TYPE nexusops_llm_calls_total counter")
    lines.append(f"nexusops_llm_calls_total {_METRICS.get('nexusops_llm_calls_total', 0)}")

    # ── nexusops_request_duration_seconds ────────────────────
    lines.append("# HELP nexusops_request_duration_seconds Request duration histogram")
    lines.append("# TYPE nexusops_request_duration_seconds histogram")
    buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    for path, durations in _METRICS_HIST.items():
        if not durations:
            continue
        safe_path = path.replace('"', '\\"')
        total_count = len(durations)
        total_sum = sum(durations)
        for b in buckets:
            count_le = sum(1 for d in durations if d <= b)
            lines.append(f'nexusops_request_duration_seconds_bucket{{endpoint="{safe_path}",le="{b}"}} {count_le}')
        lines.append(f'nexusops_request_duration_seconds_bucket{{endpoint="{safe_path}",le="+Inf"}} {total_count}')
        lines.append(f'nexusops_request_duration_seconds_sum{{endpoint="{safe_path}"}} {total_sum:.6f}')
        lines.append(f'nexusops_request_duration_seconds_count{{endpoint="{safe_path}"}} {total_count}')

    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.post("/correlate")
def correlate(event_list: List[Event]):
    if not event_list:
        raise HTTPException(status_code=400, detail="No events provided")
    events = [e.model_dump() for e in event_list]
    result = correlate_events(events)
    return {"correlation": result}

@app.post("/llm/analyze")
def llm_analyze(req: ContextRequest):
    result = analyze_context(req.model_dump())
    return {"analysis": result}

@app.get("/check/aws")
def aws_check():
    result = check_aws_infrastructure()
    return {"aws_check": result}

# ── AWS Observability ─────────────────────────────────────────

# EC2
@app.get("/aws/ec2/instances")
def aws_ec2_list(state: str = ""):
    result = list_ec2_instances(state)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"ec2_instances": result}

@app.get("/aws/ec2/status")
def aws_ec2_status(instance_id: str = ""):
    """Status checks for all EC2 instances, or a specific one via ?instance_id=i-xxx"""
    result = get_ec2_status_checks(instance_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"status_checks": result}

@app.get("/aws/ec2/console")
def aws_ec2_console(instance_id: str):
    result = get_ec2_console_output(instance_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"console_output": result}

@app.post("/aws/ec2/{instance_id}/start")
def aws_ec2_start(instance_id: str, auth: AuthContext = Depends(require_developer)):
    result = start_ec2_instance(instance_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to start instance"))
    return result

@app.post("/aws/ec2/{instance_id}/stop")
def aws_ec2_stop(instance_id: str, auth: AuthContext = Depends(require_developer)):
    result = stop_ec2_instance(instance_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to stop instance"))
    return result

@app.post("/aws/ec2/{instance_id}/reboot")
def aws_ec2_reboot(instance_id: str, auth: AuthContext = Depends(require_developer)):
    result = reboot_ec2_instance(instance_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to reboot instance"))
    return result

# CloudWatch Logs
@app.get("/aws/logs/groups")
def aws_log_groups(prefix: str = "", limit: int = 50):
    result = list_log_groups(prefix, limit)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"log_groups": result}

@app.get("/aws/logs/recent")
def aws_logs_recent(log_group: str, minutes: int = 30, limit: int = 100):
    result = get_recent_logs(log_group, minutes, limit)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"logs": result}

@app.get("/aws/logs/search")
def aws_logs_search(log_group: str, pattern: str, hours: int = 1, limit: int = 100):
    result = search_logs(log_group, pattern, hours, limit)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"logs": result}

# CloudWatch Metrics & Alarms
@app.get("/aws/cloudwatch/alarms")
def aws_cw_alarms(state: str = ""):
    valid = {"", "OK", "ALARM", "INSUFFICIENT_DATA"}
    if state.upper() not in valid:
        raise HTTPException(status_code=400, detail=f"state must be one of {valid - {''}}")
    result = list_cloudwatch_alarms(state)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"cloudwatch_alarms": result}

@app.post("/aws/cloudwatch/metrics")
def aws_cw_metrics(req: AWSMetricRequest):
    result = get_metric(req.namespace, req.metric_name, req.dimensions,
                        req.hours, req.period, req.stat)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"metric": result}

# ECS
@app.get("/aws/ecs/services")
def aws_ecs_services(cluster: str = "default"):
    result = list_ecs_services(cluster)
    # ClusterNotFoundException → return empty rather than 400
    if not result.get("success"):
        return {"ecs_services": {"success": True, "cluster": cluster, "services": [], "count": 0, "note": result.get("error")}}
    return {"ecs_services": result}

@app.get("/aws/ecs/stopped-tasks")
def aws_ecs_stopped(cluster: str = "default", limit: int = 20):
    result = get_stopped_ecs_tasks(cluster, limit)
    if not result.get("success"):
        return {"stopped_tasks": {"success": True, "cluster": cluster, "stopped_tasks": [], "count": 0, "note": result.get("error")}}
    return {"stopped_tasks": result}

# Lambda
@app.get("/aws/lambda/functions")
def aws_lambda_list():
    result = list_lambda_functions()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"lambda_functions": result}

@app.get("/aws/lambda/errors")
def aws_lambda_errors(function_name: str = "", hours: int = 1):
    """Error metrics for all Lambda functions, or a specific one via ?function_name=xxx"""
    if function_name:
        result = get_lambda_errors(function_name, hours)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return {"lambda_metrics": [result]}
    # auto-discover all functions
    all_fns = list_lambda_functions()
    if not all_fns.get("success"):
        raise HTTPException(status_code=400, detail=all_fns.get("error"))
    metrics = []
    for fn in all_fns.get("functions", []):
        r = get_lambda_errors(fn["name"], hours)
        if r.get("success"):
            metrics.append(r)
    return {"lambda_metrics": metrics, "count": len(metrics)}

# RDS
@app.get("/aws/rds/instances")
def aws_rds_list():
    result = list_rds_instances()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"rds_instances": result}

@app.get("/aws/rds/events")
def aws_rds_events(db_instance_id: str = "", hours: int = 24):
    """Events for all RDS instances, or a specific one via ?db_instance_id=xxx"""
    if db_instance_id:
        result = get_rds_events(db_instance_id, hours)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return {"rds_events": [result]}
    # auto-discover all DB instances
    all_dbs = list_rds_instances()
    if not all_dbs.get("success"):
        raise HTTPException(status_code=400, detail=all_dbs.get("error"))
    all_events = []
    for db in all_dbs.get("instances", []):
        r = get_rds_events(db["id"], hours)
        if r.get("success"):
            all_events.append(r)
    return {"rds_events": all_events, "count": len(all_events)}

# ELB / ALB
@app.get("/aws/elb/target-health")
def aws_elb_health(target_group_arn: str):
    result = get_target_health(target_group_arn)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"target_health": result}

# CloudTrail
@app.get("/aws/cloudtrail/events")
def aws_cloudtrail(hours: int = 1, resource_name: str = ""):
    result = get_cloudtrail_events(hours, resource_name)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"cloudtrail_events": result}

# S3 / SQS / DynamoDB / Route53 / SNS
@app.get("/aws/s3/buckets")
def aws_s3_buckets():
    result = list_s3_buckets()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"s3_buckets": result}

@app.get("/aws/sqs/queues")
def aws_sqs_queues():
    result = list_sqs_queues()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"sqs_queues": result}

@app.get("/aws/dynamodb/tables")
def aws_dynamodb_tables():
    result = list_dynamodb_tables()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"dynamodb_tables": result}

@app.get("/aws/route53/healthchecks")
def aws_route53_healthchecks():
    result = list_route53_healthchecks()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"route53_healthchecks": result}

@app.get("/aws/sns/topics")
def aws_sns_topics():
    result = list_sns_topics()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"sns_topics": result}

# AI Diagnosis
@app.post("/aws/diagnose")
def aws_diagnose(req: AWSDiagnoseRequest):
    valid_types = {"ec2", "ecs", "lambda", "rds", "alb"}
    if req.resource_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"resource_type must be one of {valid_types}")
    obs = collect_diagnosis_context(req.resource_type, req.resource_id, req.log_group, req.hours)
    diagnosis = diagnose_aws_resource(obs)
    return {"resource": req.resource_id, "type": req.resource_type, "diagnosis": diagnosis, "raw_context": obs}

# ── Kubernetes ──────────────────────────────────────────────
@app.get("/check/k8s")
def k8s_check():
    return {"k8s_check": check_k8s_cluster()}

@app.get("/check/k8s/nodes")
def k8s_nodes():
    return {"k8s_nodes": check_k8s_nodes()}

@app.get("/check/k8s/pods")
def k8s_pods(namespace: str = "default"):
    return {"k8s_pods": check_k8s_pods(namespace)}

@app.get("/check/k8s/deployments")
def k8s_deployments(namespace: str = "default"):
    return {"k8s_deployments": check_k8s_deployments(namespace)}

@app.post("/k8s/restart")
def k8s_restart(req: K8sRestartRequest, x_user: Optional[str] = Header(default=None)):
    _rbac_guard(x_user, "deploy")
    result = restart_deployment(req.namespace, req.deployment)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"result": result}

@app.post("/k8s/scale")
def k8s_scale(req: K8sScaleRequest, x_user: Optional[str] = Header(default=None)):
    _rbac_guard(x_user, "deploy")
    if req.replicas < 0:
        raise HTTPException(status_code=400, detail="replicas must be >= 0")
    result = scale_deployment(req.namespace, req.deployment, req.replicas)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"result": result}

@app.get("/k8s/logs")
def k8s_logs(namespace: str, pod: str, container: str = "", tail_lines: int = 100):
    if tail_lines < 1 or tail_lines > 5000:
        raise HTTPException(status_code=400, detail="tail_lines must be between 1 and 5000")
    result = get_pod_logs(namespace, pod, container, tail_lines)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"result": result}

@app.post("/incident/war-room")
def incident_war_room():
    result = create_war_room()
    return {"war_room": result}

@app.post("/incident/jira")
def incident_jira():
    result = create_incident()
    return {"jira_incident": result}

@app.post("/incident/opsgenie")
def incident_opsgenie():
    result = notify_on_call()
    return {"opsgenie_notify": result}

@app.post("/incident/github/issue")
def incident_github_issue():
    result = create_issue()
    return {"github_issue": result}

_SAFE_BRANCH = re.compile(r'^[\w\-./]+$')

@app.post("/incident/github/pr")
def incident_github_pr(head: str, base: str = "main"):
    if not _SAFE_BRANCH.match(head) or not _SAFE_BRANCH.match(base):
        raise HTTPException(status_code=400, detail="Invalid branch name")
    result = create_pull_request(head, base)
    return {"github_pr": result}

@app.get("/memory/incidents")
def memory_incidents_list(limit: int = 10):
    try:
        from app.memory.vector_db import search_similar_incidents
        results = search_similar_incidents("", n_results=limit)
        if results and isinstance(results[0], list):
            results = results[0]
        return {"incidents": results or []}
    except Exception as exc:
        return {"incidents": [], "error": str(exc)}

@app.post("/memory/incidents")
def memory_incident(incident: Event):
    record = store_incident(incident.model_dump())
    return {"stored": record}

@app.post("/security/check")
def security_check(req: AccessRequest):
    # Map action type → required permission via policy engine, then check role
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

@app.post("/security/roles")
def security_assign_role(req: RoleAssignment, x_user: Optional[str] = Header(default=None)):
    _rbac_guard(x_user, "manage_users")
    result = assign_role(req.user, req.role)
    return {"result": result}

@app.delete("/security/roles/{user}")
def security_revoke_role(user: str, x_user: Optional[str] = Header(default=None)):
    _rbac_guard(x_user, "manage_users")
    result = revoke_role(user)
    return {"result": result}

@app.post("/incident/run")
def incident_run(req: IncidentRunRequest, x_user: Optional[str] = Header(default=None),
                 auth: Optional[AuthContext] = Depends(optional_auth)):
    """End-to-end autonomous incident response pipeline (LangGraph v2).

    Collects AWS + K8s + GitHub context, runs AI root cause analysis via
    PlannerAgent, gates risky actions through DecisionAgent approval workflow,
    executes policy-validated actions, validates results, and stores to memory.
    """
    # Resolve user/role from JWT auth or X-User header
    resolved_user = (auth.username if auth else None) or x_user or req.user or "system"
    resolved_role = (auth.role if auth else None) or req.role or "admin"

    if req.auto_remediate:
        if auth and auth.role not in ("admin", "developer"):
            raise HTTPException(status_code=403, detail="deploy permission required for auto_remediate")
        elif not auth:
            _rbac_guard(x_user, "deploy")

    import os as _os
    if req.llm_provider:
        _os.environ["LLM_PROVIDER"] = req.llm_provider

    # Merge aws_cfg: prefer explicit aws_cfg field, fall back to AWSConfig object
    aws_cfg = req.aws_cfg or (req.aws.model_dump() if req.aws else {})
    k8s_cfg = req.k8s_cfg or (req.k8s.model_dump() if req.k8s else {})

    result = run_pipeline_v2(
        incident_id    = req.incident_id,
        description    = req.description,
        auto_remediate = req.auto_remediate,
        dry_run        = req.dry_run,
        metadata       = {
            "user":          resolved_user,
            "role":          resolved_role,
            "aws_cfg":       aws_cfg,
            "k8s_cfg":       k8s_cfg,
            "hours":         req.hours,
            "slack_channel": req.slack_channel,
        },
    )

    # Store state for resume-after-approval
    if result.get("status") == "awaiting_approval":
        cid = result.get("correlation_id")
        if cid:
            _PENDING_PIPELINE_STATES[cid] = result
    return result


# ── v2: LangGraph multi-agent pipeline ───────────────────────

class IncidentRunV2Request(BaseModel):
    incident_id:    str
    description:    str
    auto_remediate: bool = False
    user:           str  = "system"
    role:           str  = "viewer"
    aws_cfg:        Optional[Dict[str, Any]] = None
    k8s_cfg:        Optional[Dict[str, Any]] = None
    hours:          int  = 2
    slack_channel:  str  = "#incidents"
    llm_provider:   str  = ""
    metadata:       Optional[Dict[str, Any]] = None

@app.post("/v2/incident/run")
def incident_run_v2(req: IncidentRunV2Request,
                    auth: AuthContext = Depends(require_developer)):
    """Alias for /incidents/run — kept for backwards compatibility."""
    unified = IncidentRunRequest(
        incident_id    = req.incident_id,
        description    = req.description,
        auto_remediate = req.auto_remediate,
        hours          = req.hours,
        user           = req.user,
        role           = req.role,
        aws_cfg        = req.aws_cfg,
        k8s_cfg        = req.k8s_cfg,
        slack_channel  = req.slack_channel,
        llm_provider   = req.llm_provider,
        metadata       = req.metadata,
    )
    return incident_run(unified, x_user=None, auth=auth)


# ── GitHub PR Review ──────────────────────────────────────────

class PRReviewRequest(BaseModel):
    pr_number: int
    post_comment: bool = False   # if True, post review as comment on the PR

@app.post("/github/review-pr")
def github_review_pr(req: PRReviewRequest):
    """AI-powered PR review — security, infra, and code quality analysis via Claude.

    Fetches the PR diff from GitHub, runs Claude analysis, and optionally posts
    the review as a comment directly on the PR.
    """
    pr_data = get_pr_for_review(req.pr_number)
    if not pr_data.get("success"):
        raise HTTPException(status_code=400, detail=pr_data.get("error"))

    review = review_pr(pr_data)

    result = {
        "pr_number": req.pr_number,
        "pr_url":    pr_data.get("url"),
        "review":    review,
    }

    if req.post_comment and review.get("comment"):
        comment_result = post_pr_review_comment(req.pr_number, review["comment"])
        result["comment_posted"] = comment_result

    return result


# ── Predictive Scaling ────────────────────────────────────────

class PredictScalingRequest(BaseModel):
    resource_type: str   # ecs | ec2 | alb | lambda
    resource_id:   str
    hours:         int = 6   # lookback window for trend analysis

@app.post("/aws/predict-scaling")
def aws_predict_scaling(req: PredictScalingRequest):
    """Analyse CloudWatch metric trends and predict if scaling is needed.

    Collects CPU, memory, request-count, and error metrics then feeds them
    to Claude AI which returns a scaling recommendation with confidence score.
    """
    metrics = get_scaling_metrics(req.resource_type, req.resource_id, req.hours)
    prediction = predict_scaling(metrics)
    return {
        "resource_type": req.resource_type,
        "resource_id":   req.resource_id,
        "hours_analysed": req.hours,
        "prediction":    prediction,
    }


# ── Pre-deployment Assessment ─────────────────────────────────

class DeployAssessRequest(BaseModel):
    deployment:  str
    namespace:   str = "default"
    new_image:   str = ""
    description: str = ""
    hours:       int = 2   # lookback for recent incidents + commits

@app.post("/deploy/assess")
def deploy_assess(req: DeployAssessRequest, x_user: Optional[str] = Header(default=None)):
    """Pre-deployment risk assessment — go / no-go decision before any deploy.

    Collects current K8s state, past similar incidents from ChromaDB memory,
    active AWS alarms, and recent GitHub commits, then feeds everything to
    Claude AI for a structured risk assessment with a checklist.

    Requires X-User header with 'deploy' permission.
    """
    _rbac_guard(x_user, "deploy")

    k8s_state = {
        "cluster":     check_k8s_cluster(),
        "pods":        check_k8s_pods(req.namespace),
        "deployments": check_k8s_deployments(req.namespace),
    }
    aws_alarms     = list_cloudwatch_alarms(state="ALARM")
    recent_commits = _get_recent_commits(hours=req.hours)
    past_incidents = search_similar_incidents(
        f"{req.deployment} {req.description}", n_results=3
    )

    assessment = assess_deployment({
        "deployment":       req.deployment,
        "namespace":        req.namespace,
        "new_image":        req.new_image,
        "description":      req.description,
        "k8s_state":        k8s_state,
        "recent_incidents": past_incidents,
        "aws_alarms":       aws_alarms,
        "recent_commits":   recent_commits,
    })

    return {
        "deployment":  req.deployment,
        "namespace":   req.namespace,
        "new_image":   req.new_image,
        "assessment":  assessment,
    }


# ── Jira Webhook → Auto-create PR ────────────────────────────

class JiraWebhookPayload(BaseModel):
    """Jira sends this shape on issue_created / issue_updated events."""
    webhookEvent: str = ""
    issue: dict = {}

_AUTO_PR_ISSUE_TYPES = {"change request", "change-request", "task", "story"}
_AUTO_PR_LABELS      = {"auto-pr", "auto_pr", "create-pr"}

@app.post("/jira/webhook")
def jira_webhook(payload: JiraWebhookPayload):
    """Jira webhook receiver — auto-creates a GitHub PR for change-request tickets.

    Configure in Jira: Project Settings → Webhooks → point to this URL.
    Triggers on: issue_created events where issue type is Change Request
                 OR the issue has an 'auto-pr' label.

    Flow:
      1. Parse Jira issue fields
      2. Claude interprets the ticket → generates PR plan + optional file patches
      3. GitHub PR is created on a branch named jira/<ticket-key>-<slug>
      4. Jira issue gets a comment with the PR link
    """
    issue_fields = payload.issue.get("fields", {})
    issue_key    = payload.issue.get("key", "")

    if not issue_key:
        return {"skipped": True, "reason": "No issue key in payload"}

    issue_type = (issue_fields.get("issuetype", {}).get("name", "") or "").lower()
    labels     = [str(l).lower() for l in (issue_fields.get("labels") or [])]

    # Only act on change-request types or explicit auto-pr label
    if issue_type not in _AUTO_PR_ISSUE_TYPES and not any(l in _AUTO_PR_LABELS for l in labels):
        return {
            "skipped":     True,
            "issue_key":   issue_key,
            "reason":      f"Issue type '{issue_type}' with labels {labels} does not trigger auto-PR",
        }

    jira_data = {
        "key":         issue_key,
        "summary":     issue_fields.get("summary", ""),
        "description": issue_fields.get("description", "") or "",
        "issue_type":  issue_fields.get("issuetype", {}).get("name", ""),
        "reporter":    (issue_fields.get("reporter") or {}).get("displayName", ""),
        "labels":      labels,
    }

    # Step 1: Claude interprets the ticket
    pr_plan = interpret_jira_for_pr(jira_data)

    if pr_plan.get("error"):
        return {"error": pr_plan["error"], "issue_key": issue_key}

    # Step 2: Create GitHub PR
    pr_result = create_incident_pr(
        incident_id  = issue_key,
        title        = pr_plan.get("pr_title", jira_data["summary"]),
        body         = pr_plan.get("pr_body", ""),
        file_changes = pr_plan.get("file_patches") or None,
    )

    # Step 3: Comment on Jira with the PR link
    comment_result = {"skipped": True}
    if pr_result.get("success"):
        pr_url = pr_result.get("url", "")
        comment_body = (
            f"🤖 *AI DevOps Platform* automatically created a GitHub PR for this ticket.\n\n"
            f"*PR:* [{pr_plan.get('pr_title')}|{pr_url}]\n"
            f"*Branch:* `{pr_result.get('branch')}`\n\n"
            f"_Confidence: {pr_plan.get('confidence', 0):.0%}_\n"
            + (f"\n*Target files:* {', '.join(pr_plan.get('target_files', []))}"
               if pr_plan.get("target_files") else "")
        )
        comment_result = jira_add_comment(issue_key, comment_body)

    return {
        "issue_key":      issue_key,
        "pr_plan":        pr_plan,
        "pr_created":     pr_result,
        "jira_commented": comment_result,
    }


# ── SECRETS MANAGEMENT ────────────────────────────────────────

_SECRET_SCHEMA: Dict[str, List[str]] = {
    "Claude AI":   ["ANTHROPIC_API_KEY", "GROQ_API_KEY"],
    "AWS":         ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
    "GitHub":      ["GITHUB_TOKEN", "GITHUB_REPO"],
    "GitLab":      ["GITLAB_URL", "GITLAB_TOKEN", "GITLAB_PROJECT"],
    "Kubernetes":  ["KUBECONFIG"],
    "Slack":       ["SLACK_BOT_TOKEN", "SLACK_CHANNEL"],
    "Jira":        ["JIRA_URL", "JIRA_USER", "JIRA_TOKEN"],
    "OpsGenie":    ["OPSGENIE_API_KEY"],
    "Grafana":     ["GRAFANA_URL", "GRAFANA_TOKEN"],
}

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
    # Reload into current process
    for key, val in updates.items():
        os.environ[key] = val


class SecretsPayload(BaseModel):
    secrets: Dict[str, str]


@app.get("/secrets/status")
def secrets_status(auth: AuthContext = Depends(require_admin)):
    """Return which env vars are configured (boolean only — never exposes values)."""
    status: Dict[str, Dict[str, bool]] = {}
    for group, keys in _SECRET_SCHEMA.items():
        status[group] = {k: bool(os.environ.get(k)) for k in keys}
    return status


@app.post("/secrets")
def secrets_update(payload: SecretsPayload, auth: AuthContext = Depends(require_admin)):
    """Write secrets to .env file. Requires admin role."""
    if not payload.secrets:
        raise HTTPException(status_code=400, detail="No secrets provided.")
    _write_env(payload.secrets)
    updated = list(payload.secrets.keys())
    return {"updated": updated, "env_file": str(_ENV_FILE)}


# ── CHAT ──────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    message: str
    history: List[ChatMessage] = []
    provider: str = ""                     # "anthropic" | "groq" | "ollama" | "" (auto)
    confirmed: bool = False                # True when user confirms a pending action
    pending_action: Optional[str] = None  # action name carried from previous turn
    pending_params: Optional[Dict] = None # params carried from previous turn
    dry_run: bool = False                  # if True: describe what would happen, don't execute
    session_id: Optional[str] = None      # conversation memory session ID
    incident_context: Optional[Dict] = None  # optional war-room context

class WarRoomRequest(BaseModel):
    incident_id:  str
    description:  str
    severity:     str = "high"
    post_to_slack: bool = True

class GitHubWebhookPayload(BaseModel):
    action: str = ""
    ref: str = ""
    commits: list = []
    pull_request: dict = {}
    repository: dict = {}

class PagerDutyWebhookPayload(BaseModel):
    messages: list = []

@app.post("/webhooks/github", tags=["Webhooks"])
async def webhook_github(
    request: Request,
    payload: GitHubWebhookPayload,
    x_github_event: str = Header("", alias="X-GitHub-Event"),
    x_hub_signature_256: str = Header("", alias="X-Hub-Signature-256"),
):
    """Receive GitHub push/PR events and trigger pipeline automatically."""
    # Verify webhook signature when GITHUB_WEBHOOK_SECRET is set
    webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET", "").strip()
    if webhook_secret and x_hub_signature_256:
        import hmac, hashlib
        body = await request.body()
        expected = "sha256=" + hmac.new(
            webhook_secret.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, x_hub_signature_256):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "Invalid webhook signature"})
    event = x_github_event
    if event == "push":
        commits = payload.commits or []
        if not commits:
            return {"status": "skipped", "reason": "no commits"}
        desc = f"GitHub push to {payload.ref}: {commits[0].get('message','')[:120]}"
        incident_id = f"gh-push-{payload.ref.split('/')[-1]}-{len(commits)}c"
        from app.orchestrator.runner import run_pipeline
        result = run_pipeline(
            incident_id=incident_id,
            description=desc,
            severity="medium",
            auto_remediate=False,
        )
        return {"status": "triggered", "incident_id": incident_id, "pipeline": result.get("status")}
    elif event == "pull_request":
        pr = payload.pull_request
        if payload.action not in ("opened", "synchronize"):
            return {"status": "skipped", "reason": f"action={payload.action}"}
        pr_num = pr.get("number")
        if pr_num:
            from app.llm.claude import review_pr
            from app.integrations.github import get_pr_for_review, post_pr_review_comment
            pr_data = get_pr_for_review(pr_num)
            if pr_data.get("success"):
                review = review_pr(pr_data)
                post_pr_review_comment(pr_num, review)
                return {"status": "reviewed", "pr": pr_num}
        return {"status": "skipped", "reason": "no pr number"}
    return {"status": "ignored", "event": event}

@app.post("/webhooks/pagerduty", tags=["Webhooks"])
async def webhook_pagerduty(payload: PagerDutyWebhookPayload):
    """Receive PagerDuty incident trigger webhooks."""
    for msg in payload.messages:
        inc = msg.get("incident", {})
        inc_id = inc.get("id", "pd-unknown")
        title = inc.get("title", "PagerDuty incident")
        urgency = inc.get("urgency", "high")
        from app.orchestrator.runner import run_pipeline
        result = run_pipeline(
            incident_id=f"pd-{inc_id}",
            description=title,
            severity=urgency,
            auto_remediate=False,
        )
        return {"status": "triggered", "incident_id": f"pd-{inc_id}", "pipeline": result.get("status")}
    return {"status": "no_messages"}

_chat_action_count = 0  # session counter shown in UI metric

# ── agentic action catalogue ──────────────────────────────────
_ACTION_CATALOGUE = {

    # ════════════════════════════════════════════════════════════
    # KUBERNETES
    # ════════════════════════════════════════════════════════════
    "restart_deployment": {
        "desc": "Rolling restart a Kubernetes deployment",
        "params": ["namespace", "deployment"],
        "handler": lambda p: restart_deployment(p["namespace"], p["deployment"]),
    },
    "scale_deployment": {
        "desc": "Scale a Kubernetes deployment to N replicas",
        "params": ["namespace", "deployment", "replicas"],
        "handler": lambda p: scale_deployment(p["namespace"], p["deployment"], int(p["replicas"])),
    },
    "delete_pod": {
        "desc": "Delete a pod so Kubernetes reschedules it (force restart)",
        "params": ["namespace", "pod"],
        "handler": lambda p: delete_pod(p["namespace"], p["pod"]),
    },
    "get_pod_logs": {
        "desc": "Fetch recent logs from a specific pod",
        "params": ["namespace", "pod"],
        "handler": lambda p: get_pod_logs(p["namespace"], p["pod"], tail_lines=int(p.get("lines", 100))),
    },
    "list_pods": {
        "desc": "List all pods and their status in a namespace",
        "params": ["namespace"],
        "handler": lambda p: list_pods(p.get("namespace", "")),
    },
    "list_deployments": {
        "desc": "List all deployments and their replica health",
        "params": ["namespace"],
        "handler": lambda p: list_deployments(p.get("namespace", "")),
    },
    "list_namespaces": {
        "desc": "List all Kubernetes namespaces",
        "params": [],
        "handler": lambda _: list_namespaces(),
    },
    "get_cluster_events": {
        "desc": "Get warning events from the cluster (OOMKilled, BackOff, Failed etc.)",
        "params": ["namespace"],
        "handler": lambda p: get_cluster_events(p.get("namespace", "")),
    },
    "get_unhealthy_pods": {
        "desc": "Get all pods that are not healthy or crash-looping",
        "params": ["namespace"],
        "handler": lambda p: get_unhealthy_pods(p.get("namespace", "")),
    },
    "get_resource_usage": {
        "desc": "Get CPU and memory requests/limits for pods in a namespace",
        "params": ["namespace"],
        "handler": lambda p: get_resource_usage(p.get("namespace", "default")),
    },
    "cordon_node": {
        "desc": "Cordon a node to stop new pods scheduling on it (pre-maintenance)",
        "params": ["node"],
        "handler": lambda p: cordon_node(p["node"]),
    },
    "uncordon_node": {
        "desc": "Uncordon a node to allow pods to schedule on it again",
        "params": ["node"],
        "handler": lambda p: uncordon_node(p["node"]),
    },

    # ════════════════════════════════════════════════════════════
    # EC2
    # ════════════════════════════════════════════════════════════
    "list_ec2": {
        "desc": "List all EC2 instances with their current state, optionally in a specific region",
        "params": ["region"],
        "handler": lambda p: list_ec2_instances(region=p.get("region", "")),
    },
    "get_ec2_info": {
        "desc": "Get full details for a specific EC2 instance",
        "params": ["instance_id", "region"],
        "handler": lambda p: get_ec2_instance_info(p["instance_id"]),
    },
    "get_ec2_status": {
        "desc": "Get system and instance status checks for EC2",
        "params": ["instance_id", "region"],
        "handler": lambda p: get_ec2_status_checks(p.get("instance_id", "")),
    },
    "start_ec2": {
        "desc": "Start a stopped EC2 instance",
        "params": ["instance_id", "region"],
        "handler": lambda p: start_ec2_instance(p["instance_id"], region=p.get("region", "")),
    },
    "stop_ec2": {
        "desc": "Stop a running EC2 instance",
        "params": ["instance_id", "region"],
        "handler": lambda p: stop_ec2_instance(p["instance_id"], region=p.get("region", "")),
    },
    "reboot_ec2": {
        "desc": "Reboot an EC2 instance",
        "params": ["instance_id", "region"],
        "handler": lambda p: reboot_ec2_instance(p["instance_id"], region=p.get("region", "")),
    },

    # ════════════════════════════════════════════════════════════
    # ECS
    # ════════════════════════════════════════════════════════════
    "list_ecs_services": {
        "desc": "List ECS services and their task counts",
        "params": ["cluster"],
        "handler": lambda p: list_ecs_services(p.get("cluster", "default")),
    },
    "get_ecs_service": {
        "desc": "Get detailed status for a specific ECS service",
        "params": ["cluster", "service"],
        "handler": lambda p: get_ecs_service_detail(p.get("cluster", "default"), p["service"]),
    },
    "scale_ecs_service": {
        "desc": "Scale an ECS service to a desired task count",
        "params": ["cluster", "service", "count"],
        "handler": lambda p: scale_ecs_service(p.get("cluster", "default"), p["service"], int(p["count"])),
    },
    "redeploy_ecs_service": {
        "desc": "Force a new ECS deployment (restarts all running tasks)",
        "params": ["cluster", "service"],
        "handler": lambda p: force_new_ecs_deployment(p.get("cluster", "default"), p["service"]),
    },
    "get_stopped_ecs_tasks": {
        "desc": "List recently stopped ECS tasks and their stop reasons",
        "params": ["cluster"],
        "handler": lambda p: get_stopped_ecs_tasks(p.get("cluster", "default")),
    },

    # ════════════════════════════════════════════════════════════
    # LAMBDA
    # ════════════════════════════════════════════════════════════
    "list_lambda": {
        "desc": "List all Lambda functions, optionally in a specific region",
        "params": ["region"],
        "handler": lambda p: list_lambda_functions(region=p.get("region", "")),
    },
    "get_lambda_errors": {
        "desc": "Get error and throttle metrics for a Lambda function",
        "params": ["function_name"],
        "handler": lambda p: get_lambda_errors(p["function_name"]),
    },
    "invoke_lambda": {
        "desc": "Invoke a Lambda function and return its response",
        "params": ["function_name", "payload"],
        "handler": lambda p: invoke_lambda(p["function_name"], p.get("payload", {})),
    },

    # ════════════════════════════════════════════════════════════
    # RDS
    # ════════════════════════════════════════════════════════════
    "list_rds": {
        "desc": "List all RDS database instances, optionally in a specific region",
        "params": ["region"],
        "handler": lambda p: list_rds_instances(region=p.get("region", "")),
    },
    "get_rds_detail": {
        "desc": "Get detailed status for a specific RDS instance",
        "params": ["db_instance_id"],
        "handler": lambda p: get_rds_instance_detail(p["db_instance_id"]),
    },
    "get_rds_events": {
        "desc": "Get recent RDS events (failovers, reboots, errors)",
        "params": ["db_instance_id"],
        "handler": lambda p: get_rds_events(p["db_instance_id"]),
    },
    "reboot_rds": {
        "desc": "Reboot an RDS database instance",
        "params": ["db_instance_id"],
        "handler": lambda p: reboot_rds_instance(p["db_instance_id"]),
    },

    # ════════════════════════════════════════════════════════════
    # CLOUDWATCH / LOGS / ALARMS
    # ════════════════════════════════════════════════════════════
    "get_alarms": {
        "desc": "List CloudWatch alarms, optionally filtered by state and region",
        "params": ["state", "region"],
        "handler": lambda p: list_cloudwatch_alarms(p.get("state", ""), region=p.get("region", "")),
    },
    "get_firing_alarms": {
        "desc": "Get only alarms currently in ALARM state",
        "params": ["region"],
        "handler": lambda p: list_cloudwatch_alarms("ALARM", region=p.get("region", "")),
    },
    "set_alarm_state": {
        "desc": "Manually set a CloudWatch alarm state (OK/ALARM/INSUFFICIENT_DATA)",
        "params": ["alarm_name", "state"],
        "handler": lambda p: set_alarm_state(p["alarm_name"], p["state"]),
    },
    "search_logs": {
        "desc": "Search CloudWatch logs for a pattern",
        "params": ["log_group", "pattern", "hours"],
        "handler": lambda p: search_logs(p["log_group"], p["pattern"], int(p.get("hours", 1))),
    },
    "get_recent_logs": {
        "desc": "Get recent log events from a CloudWatch log group",
        "params": ["log_group", "minutes"],
        "handler": lambda p: get_recent_logs(p["log_group"], int(p.get("minutes", 30))),
    },
    "get_cloudtrail": {
        "desc": "Get recent CloudTrail API events",
        "params": ["hours"],
        "handler": lambda p: get_cloudtrail_events(int(p.get("hours", 1))),
    },

    # ════════════════════════════════════════════════════════════
    # SQS / S3 / DYNAMODB
    # ════════════════════════════════════════════════════════════
    "list_sqs": {
        "desc": "List SQS queues and their message counts",
        "params": [],
        "handler": lambda _: list_sqs_queues(),
    },
    "get_sqs_depth": {
        "desc": "Get message depth for a specific SQS queue",
        "params": ["queue_url"],
        "handler": lambda p: get_sqs_queue_depth(p["queue_url"]),
    },
    "list_s3": {
        "desc": "List S3 buckets",
        "params": [],
        "handler": lambda _: list_s3_buckets(),
    },
    "list_dynamodb": {
        "desc": "List DynamoDB tables",
        "params": [],
        "handler": lambda _: list_dynamodb_tables(),
    },

    # ════════════════════════════════════════════════════════════
    # GITHUB
    # ════════════════════════════════════════════════════════════
    "list_repos": {
        "desc": "List GitHub repositories",
        "params": [],
        "handler": lambda _: _list_repos(),
    },
    "get_recent_commits": {
        "desc": "Get recent commits from GitHub",
        "params": ["hours"],
        "handler": lambda p: _get_recent_commits(hours=int(p.get("hours", 24))),
    },
    "get_recent_prs": {
        "desc": "Get recently merged pull requests",
        "params": ["hours"],
        "handler": lambda p: _get_recent_prs(hours=int(p.get("hours", 48))),
    },
    "create_github_issue": {
        "desc": "Create a GitHub issue",
        "params": ["title", "body"],
        "handler": lambda p: create_issue(p.get("title", "AI-generated issue"), p.get("body", "")),
    },

    # ════════════════════════════════════════════════════════════
    # SLACK
    # ════════════════════════════════════════════════════════════
    "post_slack": {
        "desc": "Post a message to a Slack channel",
        "params": ["channel", "message"],
        "handler": lambda p: post_message(p.get("channel", "#general"), p["message"]),
    },

    # ════════════════════════════════════════════════════════════
    # INCIDENTS / PIPELINE
    # ════════════════════════════════════════════════════════════
    "run_pipeline": {
        "desc": "Run the full autonomous incident response pipeline",
        "params": ["description", "severity"],
        "handler": lambda p: run_incident_pipeline(
            incident_id=p.get("incident_id", f"chat-{__import__('time').strftime('%H%M%S')}"),
            description=p["description"],
            severity=p.get("severity", "high"),
            aws_config={}, k8s_config={}, auto_remediate=False,
        ),
    },
    "create_jira_ticket": {
        "desc": "Create a Jira incident ticket",
        "params": ["summary", "description"],
        "handler": lambda p: create_incident(
            summary=p.get("summary", "AI DevOps Incident"),
            description=p.get("description", p.get("summary", "")),
        ),
    },
    "notify_oncall": {
        "desc": "Page the on-call team via OpsGenie",
        "params": ["message", "priority"],
        "handler": lambda p: notify_on_call(p.get("message", ""), p.get("priority", "P2")),
    },
    "debug_and_fix": {
        "desc": "Collect full context from all integrations and run AI root cause analysis + automated fix",
        "params": ["description", "severity"],
        "handler": lambda p: run_incident_pipeline(
            incident_id=f"debug-{__import__('time').strftime('%H%M%S')}",
            description=p.get("description", "infrastructure issue"),
            severity=p.get("severity", "high"),
            aws_config={}, k8s_config={}, auto_remediate=True,
        ),
    },
}

_INTENT_SYSTEM = """You are an intent classifier for a DevOps automation platform.

Decide: is the user asking to DO something, or asking a question?

If DO → output:
{"intent": "action", "action": "<name>", "params": {<key>: <value>}}

If QUESTION → output:
{"intent": "question"}

Available actions (ONLY these — never invent others):

KUBERNETES:
- restart_deployment: namespace, deployment
- scale_deployment: namespace, deployment, replicas (int)
- delete_pod: namespace, pod
- get_pod_logs: namespace, pod, container (optional), tail_lines (optional int)
- list_pods: namespace (optional)
- list_deployments: namespace (optional)
- list_namespaces: (no params)
- get_cluster_events: namespace (optional), limit (optional int)
- get_unhealthy_pods: namespace (optional)
- get_resource_usage: namespace (optional)
- cordon_node: node
- uncordon_node: node

EC2:
- list_ec2: (no params)
- get_ec2_info: instance_id
- get_ec2_status: instance_id
- start_ec2: instance_id
- stop_ec2: instance_id
- reboot_ec2: instance_id

ECS:
- list_ecs_services: cluster (optional)
- get_ecs_service: cluster, service
- scale_ecs_service: cluster, service, desired_count (int)
- redeploy_ecs_service: cluster, service
- get_stopped_ecs_tasks: cluster (optional)

LAMBDA:
- list_lambda: (no params)
- get_lambda_errors: function_name, hours (optional int)
- invoke_lambda: function_name, payload (optional JSON string)

RDS:
- list_rds: (no params)
- get_rds_detail: db_instance_id
- get_rds_events: db_instance_id, hours (optional int)
- reboot_rds: db_instance_id

CLOUDWATCH / LOGS:
- get_alarms: (no params)
- get_firing_alarms: (no params)
- set_alarm_state: alarm_name, state (OK/ALARM/INSUFFICIENT_DATA), reason
- search_logs: log_group, query, hours (optional int)
- get_recent_logs: log_group, tail_lines (optional int)
- get_cloudtrail: hours (optional int)

SQS / S3 / DYNAMODB:
- list_sqs: (no params)
- get_sqs_depth: queue_url
- list_s3: (no params)
- list_dynamodb: (no params)

GITHUB:
- list_repos: (no params)
- get_recent_commits: hours (optional int), branch (optional)
- get_recent_prs: hours (optional int)
- create_github_issue: title, body

INCIDENTS / PIPELINE:
- run_pipeline: description, severity (critical/high/medium/low), incident_id (optional)
- create_jira_ticket: summary, description
- notify_oncall: message, priority (P1/P2/P3)
- debug_and_fix: service, error_description

Rules:
- Extract params literally from the message
- REGION: if the user mentions a region (e.g. "us-east-1", "us-east-2", "eu-west-1", "ap-southeast-1", "us-west-2"), always include "region": "<value>" in params. If no region is mentioned, omit the region param (do not default it).
- If a required param (like instance_id) is missing and cannot be inferred, output {"intent": "question"} instead
- Output ONLY valid JSON, no markdown, nothing else"""


def _detect_intent(message: str, force_provider: str = "", conv_context: str = "") -> dict:
    """Use LLM to classify whether message is an action or a question.

    conv_context: last few turns of conversation so pronouns like 'that'/'it' can be resolved.
    """
    import json as _json
    from app.llm.claude import _llm, _extract_json
    content = message
    if conv_context:
        content = (
            f"=== RECENT CONVERSATION (for pronoun/context resolution) ===\n{conv_context}\n"
            f"=== NEW MESSAGE ===\n{message}"
        )
    try:
        raw = _llm(_INTENT_SYSTEM, [{"role": "user", "content": content}],
                   max_tokens=400, force_provider=force_provider)
        return _json.loads(_extract_json(raw))
    except Exception:
        return {"intent": "question"}


# Actions that mutate state and require confirmation before executing.
# Read-only actions (list_*, get_*, check_*) run immediately without asking.
_CONFIRM_REQUIRED = {
    "restart_deployment", "scale_deployment", "delete_pod",
    "cordon_node", "uncordon_node",
    "start_ec2", "stop_ec2", "reboot_ec2",
    "scale_ecs_service", "redeploy_ecs_service",
    "invoke_lambda",
    "reboot_rds",
    "set_alarm_state",
    "create_github_issue", "create_jira_ticket",
    "run_pipeline", "notify_oncall", "debug_and_fix",
}


def _confirmation_message(action_name: str, params: dict) -> str:
    """Build a human-readable confirmation prompt for a mutating action."""
    p = params or {}
    descriptions = {
        "restart_deployment":   f"rolling restart deployment **{p.get('deployment','?')}** in namespace **{p.get('namespace','?')}**",
        "scale_deployment":     f"scale deployment **{p.get('deployment','?')}** in **{p.get('namespace','?')}** to **{p.get('replicas','?')}** replicas",
        "delete_pod":           f"delete pod **{p.get('pod','?')}** in namespace **{p.get('namespace','?')}** (will be rescheduled)",
        "cordon_node":          f"cordon node **{p.get('node','?')}** — no new pods will be scheduled on it",
        "uncordon_node":        f"uncordon node **{p.get('node','?')}** — allow scheduling again",
        "start_ec2":            f"start EC2 instance **{p.get('instance_id','?')}**",
        "stop_ec2":             f"stop EC2 instance **{p.get('instance_id','?')}**",
        "reboot_ec2":           f"reboot EC2 instance **{p.get('instance_id','?')}**",
        "scale_ecs_service":    f"scale ECS service **{p.get('service','?')}** in cluster **{p.get('cluster','?')}** to **{p.get('desired_count','?')}** tasks",
        "redeploy_ecs_service": f"force a new deployment of ECS service **{p.get('service','?')}** in cluster **{p.get('cluster','?')}**",
        "invoke_lambda":        f"invoke Lambda function **{p.get('function_name','?')}**",
        "reboot_rds":           f"reboot RDS instance **{p.get('db_instance_id','?')}** (brief downtime expected)",
        "set_alarm_state":      f"set CloudWatch alarm **{p.get('alarm_name','?')}** to state **{p.get('state','?')}**",
        "create_github_issue":  f"create GitHub issue: **{p.get('title','?')}**",
        "create_jira_ticket":   f"create Jira ticket: **{p.get('summary','?')}**",
        "run_pipeline":         f"run the full incident pipeline — *{p.get('description','?')}* (severity: {p.get('severity','?')})",
        "notify_oncall":        f"page on-call with priority **{p.get('priority','?')}**: _{p.get('message','?')}_",
        "debug_and_fix":        f"run automated debug & fix for **{p.get('service','?')}**: _{p.get('error_description','?')}_",
    }
    desc = descriptions.get(action_name, f"**{action_name.replace('_',' ')}** with params: {p}")
    return f"\u26a0\ufe0f I\u2019m about to {desc}.\n\n**Confirm?** Reply **yes** to proceed or **no** to cancel."


def _build_action_reply(action_name: str, user_msg: str, action_result: dict,
                        force_prov: str, _llm, _j) -> str:
    """Turn a real action result into a concise, honest reply."""
    succeeded = action_result.get("success", False) if isinstance(action_result, dict) else True
    result_json = _j.dumps(action_result, default=str, indent=2)
    ACTION_REPLY_SYSTEM = (
        "You are a DevOps assistant. The user asked you to perform an operation. "
        "The ACTUAL result from executing that operation is shown below as JSON — use ONLY the values in it. "
        "RESPONSE LENGTH: "
        "If success=true and the result is simple (restart, scale, delete): 1-2 sentences confirming what happened using real values (IDs, counts, names). "
        "If success=true and the result contains a list (pods, instances, logs, events, alarms): present it in a clean readable format — use a markdown table or bullet list with natural language labels. "
        "If success=false or there is an error key: state it failed, quote the exact error message, and briefly explain what it likely means. "
        "FORMATTING: "
        "NEVER show raw JSON, dict keys, or field names like 'success', 'count', 'alarms_firing' in your reply. "
        "Translate everything to natural English. Use **bold** for resource names and states. "
        "Use ✅ for success, ❌ for failure, ⚠️ for warnings. "
        "NEVER claim success if success=false. NEVER fabricate values. NEVER add padding."
    )
    try:
        return _llm(
            ACTION_REPLY_SYSTEM,
            [{"role": "user", "content":
                f"User asked: {user_msg}\n\nOperation: {action_name}\nResult:\n{result_json}"}],
            max_tokens=600,
            force_provider=force_prov,
        )
    except Exception:
        if succeeded:
            return "✅ `{}` completed. {}".format(action_name, _j.dumps(
                {k: v for k, v in action_result.items() if k != "success"}, default=str))
        else:
            return "❌ `{}` failed: {}".format(action_name, action_result.get("error", "unknown error"))


@app.post("/chat")
def chat(payload: ChatPayload, auth: AuthContext = Depends(require_viewer)):
    """Conversational DevOps AI with confirmation flow, rate limiting, audit log, and dry-run."""
    try:
        return _chat_inner(payload, auth.username)
    except HTTPException:
        raise
    except Exception as exc:
        import traceback, logging
        logging.getLogger("chat").error("Unhandled chat error: %s\n%s", exc, traceback.format_exc())
        return {
            "reply": f"Something went wrong on the server: `{type(exc).__name__}: {exc}`. Check server logs for details.",
            "sources": [], "llm_provider": "none", "action_taken": None,
            "action_result": None, "action_count": _chat_action_count,
            "pending_action": None, "pending_params": None, "needs_confirm": False,
        }


def _chat_inner(payload: ChatPayload, x_user: str):
    global _chat_action_count
    import json as _j
    from app.llm.claude import _provider, _llm
    from app.core.ratelimit import check_chat, check_action
    from app.core.audit import audit_log

    force_prov = payload.provider or ""
    history = [{"role": m.role, "content": m.content} for m in payload.history]

    # ── Rate limit: general chat ──────────────────────────────
    allowed, remaining = check_chat(x_user)
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded — max 20 messages per minute. Please slow down.")

    action_result = None
    action_taken  = None
    reply         = ""

    # ── Path A: user confirmed a pending action ───────────────
    if payload.confirmed and payload.pending_action:
        action_name = payload.pending_action
        params      = payload.pending_params or {}
        action_def  = _ACTION_CATALOGUE.get(action_name)
        if action_def:
            # Rate-limit mutating actions separately
            act_ok, _ = check_action(x_user)
            if not act_ok:
                raise HTTPException(status_code=429, detail="Action rate limit exceeded — max 10 operations per minute.")
            if payload.dry_run:
                reply = f"**Dry-run:** Would execute `{action_name}` with params `{_j.dumps(params)}`.\nNo changes made."
            else:
                try:
                    action_result = action_def["handler"](params)
                    action_taken  = action_name
                    _chat_action_count += 1
                except Exception as exc:
                    action_result = {"success": False, "error": str(exc)}
                audit_log(user=x_user, action=action_name, params=params,
                          result=action_result or {}, source="chat")
                reply = _build_action_reply(action_name, payload.message, action_result, force_prov, _llm, _j)
        else:
            reply = "Sorry, I could not find that operation. Please try again."

    # ── Path B: fresh message — classify intent ───────────────
    if not reply:
        _cancel = {"no", "cancel", "nope", "stop", "abort", "never mind", "nevermind"}
        if payload.message.lower().strip().rstrip(".,!") in _cancel:
            reply = "Got it — operation cancelled."

    if not reply:
        # Fetch recent conversation history to pass as context to intent detection
        # so pronouns like "that", "it", "the instance" can be resolved
        _conv_context = ""
        try:
            from app.chat.memory import get_history as _get_hist
            _sid_for_ctx = payload.session_id or f"chat-{x_user}"
            _hist = _get_hist(_sid_for_ctx, max_messages=6)
            if _hist:
                _conv_context = "\n".join(
                    f"{getattr(m,'role','?').upper()}: {getattr(m,'content','')[:300]}"
                    for m in _hist
                )
        except Exception:
            pass
        intent_data = _detect_intent(payload.message, force_prov, conv_context=_conv_context)

        if intent_data.get("intent") == "action":
            action_name = intent_data.get("action", "")
            params      = intent_data.get("params", {})
            action_def  = _ACTION_CATALOGUE.get(action_name)

            # ── EC2 instance ID validation ──────────────────────────────────
            # If action needs an instance_id but it's vague/invalid, resolve from AWS
            if action_name in ("start_ec2", "stop_ec2", "reboot_ec2"):
                raw_iid = params.get("instance_id", "")
                if not raw_iid or not str(raw_iid).startswith("i-"):
                    # Try to resolve from session EC2 cache or live AWS
                    from app.chat.intelligence import _ec2_session_cache
                    cached = _ec2_session_cache.get(sid, [])
                    if not cached:
                        try:
                            from app.integrations.aws_ops import list_ec2_instances
                            result = list_ec2_instances()
                            instances = result.get("instances", [])
                            if instances:
                                _ec2_session_cache[sid] = [
                                    {"id": i["id"], "name": i.get("name",""), "state": i.get("state","")}
                                    for i in instances
                                ]
                                cached = _ec2_session_cache[sid]
                        except Exception:
                            pass
                    if len(cached) == 1:
                        params = dict(params)
                        params["instance_id"] = cached[0]["id"]
                    elif len(cached) > 1:
                        names = ", ".join(f'{i["id"]} ({i.get("name","?")}, {i.get("state","?")})' for i in cached[:5])
                        reply = f"Multiple EC2 instances found: {names}\n\nWhich one should I {action_name.replace('_ec2','')}? Please specify the instance ID (e.g. `i-0abc1234`)."
                        used_provider = force_prov or _provider or "none"
                        return {"reply": reply, "sources": [], "llm_provider": used_provider,
                                "action_taken": None, "action_result": None, "action_count": _chat_action_count,
                                "pending_action": None, "pending_params": None, "needs_confirm": False}
                    else:
                        reply = "I couldn't find any EC2 instances in your AWS account."
                        used_provider = force_prov or _provider or "none"
                        return {"reply": reply, "sources": [], "llm_provider": used_provider,
                                "action_taken": None, "action_result": None, "action_count": _chat_action_count,
                                "pending_action": None, "pending_params": None, "needs_confirm": False}

            if not action_def:
                reply = (
                    f"I understood you want to **{action_name.replace('_', ' ')}**, "
                    f"but that operation is not available yet. "
                    f"I can restart/scale K8s deployments, start/stop/reboot EC2, manage ECS/Lambda/RDS, "
                    f"query CloudWatch logs and alarms, create GitHub issues or Jira tickets, "
                    f"and run the full incident pipeline."
                )
            elif action_name in _CONFIRM_REQUIRED:
                # Mutating action — ask for confirmation first (dry-run: skip confirmation)
                if payload.dry_run:
                    reply = f"**Dry-run:** Would execute `{action_name}` with params `{_j.dumps(params)}`.\nNo changes made."
                else:
                    reply = _confirmation_message(action_name, params)
                    used_provider = force_prov or _provider or "none"
                    return {
                        "reply":          reply,
                        "sources":        [],
                        "llm_provider":   used_provider,
                        "action_taken":   None,
                        "action_result":  None,
                        "action_count":   _chat_action_count,
                        "pending_action": action_name,
                        "pending_params": params,
                        "needs_confirm":  True,
                    }
            else:
                # Read-only — run immediately, no confirmation needed
                try:
                    action_result = action_def["handler"](params)
                    action_taken  = action_name
                    _chat_action_count += 1
                except Exception as exc:
                    action_result = {"success": False, "error": str(exc)}
                audit_log(user=x_user, action=action_name, params=params,
                          result=action_result or {}, source="chat")
                reply = _build_action_reply(action_name, payload.message, action_result, force_prov, _llm, _j)

    # ── Path C: general question / conversation ───────────────
    if not reply:
        import uuid as _uuid, logging as _log_m
        sid = payload.session_id or f"chat-{x_user}-default"
        try:
            from app.chat.intelligence import chat_with_intelligence
            reply = chat_with_intelligence(
                message=payload.message,
                session_id=sid,
                incident_context=payload.incident_context,
                preferred_provider=force_prov or None,
            )
        except Exception as exc:
            _log_m.getLogger("chat").error("chat_with_intelligence failed: %s", exc, exc_info=True)
            # Fallback: build history from memory if available, then use basic LLM
            fallback_history = history  # payload.history (may be empty)
            try:
                from app.chat.memory import get_history as _gh
                _mh = _gh(sid, max_messages=10)
                if _mh:
                    fallback_history = [
                        {"role": getattr(m, "role", "user"), "content": getattr(m, "content", "")}
                        for m in _mh
                    ]
            except Exception:
                pass
            context: dict = {}
            try:
                context = collect_all_context(hours=2)
            except Exception:
                pass
            reply = chat_devops(payload.message, fallback_history, context, force_provider=force_prov)

    used_provider = force_prov or _provider or "none"
    import uuid as _uuid2
    sid_out = payload.session_id or f"chat-{x_user}-{_uuid2.uuid4().hex[:8]}"
    return {
        "reply":          reply,
        "answer":         reply,   # alias so UI can use either field
        "session_id":     sid_out,
        "sources":        [],
        "llm_provider":   used_provider,
        "action_taken":   action_taken,
        "action_result":  action_result,
        "action_count":   _chat_action_count,
        "pending_action": None,
        "pending_params": None,
        "needs_confirm":  False,
    }

@app.get("/audit/log")
def get_audit_log_endpoint(limit: int = 50, user: str = "", action: str = ""):
    """Return recent audit log entries (newest first). Filter by user or action."""
    from app.core.audit import get_audit_log
    return {"entries": get_audit_log(limit=limit, user=user, action=action)}


@app.get("/rate-limit/status")
def rate_limit_status(x_user: str = Header(default="anonymous")):
    """Return current rate-limit usage for the calling user."""
    from app.core.ratelimit import get_usage
    return get_usage(x_user)


@app.get("/chat/action_count")
def chat_action_count():
    return {"count": _chat_action_count}

@app.post("/warroom/create")
def warroom_create(req: WarRoomRequest, auth: AuthContext = Depends(require_developer)):
    """Create a war room: collect universal context, run AI analysis, create Slack channel, post findings."""
    # 1. Collect context from all integrations
    context: dict = {}
    try:
        context = collect_all_context(hours=2)
    except Exception:
        pass

    # 2. Run AI analysis
    from app.llm.claude import synthesize_incident
    synthesis = synthesize_incident({
        "incident_id":   req.incident_id,
        "description":   req.description,
        "severity":      req.severity,
        "aws_context":   context.get("aws", {}),
        "k8s_context":   context.get("k8s", {}),
        "github_context": context.get("github", {}),
    })

    result = {
        "incident_id": req.incident_id,
        "analysis":    synthesis,
        "sources":     context.get("configured", []),
        "slack":       None,
    }

    # 3. Create Slack war room channel + post findings
    if req.post_to_slack:
        channel_result = create_incident_channel(req.incident_id, topic=f"{req.severity.upper()} — {req.description[:80]}")
        if channel_result.get("success"):
            channel_id = channel_result["channel_id"]
            post_incident_summary(
                channel    = channel_id,
                incident_id = req.incident_id,
                summary    = synthesis.get("summary", req.description),
                findings   = synthesis.get("findings", []),
                severity   = synthesis.get("severity", req.severity),
                actions    = synthesis.get("actions_to_take", []),
            )
            result["slack"] = {
                "channel_name": channel_result.get("channel_name"),
                "channel_url":  channel_result.get("channel_url"),
            }
        else:
            result["slack"] = {"error": channel_result.get("error")}

    return result

@app.get("/health/full")
def health_full():
    """Full health check — collects universal context and returns per-integration status."""
    context: dict = {}
    try:
        context = collect_all_context(hours=1)
    except Exception:
        pass
    health = summarize_health(context)
    return {"status": "healthy" if health["healthy"] else "degraded", "health": health}


@app.get("/github/repos")
def github_repos():
    """List all repositories for the configured GitHub account."""
    return _list_repos()

@app.get("/github/profile")
def github_profile():
    """GitHub account summary — repos, stars, top languages."""
    return _get_github_profile()

@app.get("/github/commits")
def github_commits(hours: int = 24, repo: str = ""):
    """Recent commits across all repos (or a specific one)."""
    return _get_recent_commits(hours=hours, repo_name=repo)

@app.get("/github/prs")
def github_prs(hours: int = 48, repo: str = ""):
    """Recent merged PRs across all repos (or a specific one)."""
    return _get_recent_prs(hours=hours, repo_name=repo)

@app.get("/health/integrations")
def health_integrations():
    """Diagnostic endpoint — shows exactly which integrations and LLM providers are configured.

    Use this to debug 'not configured' errors before running the pipeline.
    """
    from app.llm.claude import _provider, _provider_warning, ANTHROPIC_API_KEY, GROQ_API_KEY, OLLAMA_HOST
    import os

    # ── LLM status ────────────────────────────────────────────────────────
    llm = {
        "active_provider": _provider or "none — no LLM configured",
        "warning":         _provider_warning,
        "anthropic": {
            "key_set":   bool(ANTHROPIC_API_KEY),
            "key_valid": ANTHROPIC_API_KEY.startswith("sk-ant-") if ANTHROPIC_API_KEY else False,
            "note":      "Key must start with 'sk-ant-'. Wrap in quotes in .env if it contains special chars." if ANTHROPIC_API_KEY and not ANTHROPIC_API_KEY.startswith("sk-ant-") else None,
        },
        "groq": {
            "key_set":   bool(GROQ_API_KEY),
            "key_valid": GROQ_API_KEY.startswith("gsk_") if GROQ_API_KEY else False,
        },
        "ollama": {
            "host": OLLAMA_HOST,
        },
        "openai": {
            "key_set": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        },
    }

    # ── GitHub status ─────────────────────────────────────────────────────
    gh_token = os.getenv("GITHUB_TOKEN", "").strip()
    gh_repo  = os.getenv("GITHUB_REPO", "").strip()
    gh_slug  = None
    gh_error = None
    if gh_repo:
        try:
            from app.integrations.github import _parse_github_url
            owner, repo_name = _parse_github_url(gh_repo)
            gh_slug = f"{owner}/{repo_name}" if repo_name else f"{owner} (profile-level)"
        except Exception as e:
            gh_error = str(e)

    github = {
        "token_set":   bool(gh_token),
        "repo_raw":    gh_repo or "(not set)",
        "repo_parsed": gh_slug,
        "repo_valid":  bool(gh_token and gh_slug),
        "error":       gh_error,
        "note":        "Profile-level access — works across all repos" if gh_slug and "profile-level" in (gh_slug or "") else None,
    }

    # ── Other integrations ────────────────────────────────────────────────
    def _env_set(key: str) -> bool:
        v = os.getenv(key, "").strip()
        return bool(v) and not v.startswith("your_")

    integrations = {
        "slack":    {"configured": _env_set("SLACK_BOT_TOKEN")},
        "jira":     {"configured": _env_set("JIRA_URL") and _env_set("JIRA_TOKEN")},
        "opsgenie": {"configured": _env_set("OPSGENIE_API_KEY")},
        "aws":      {"configured": _env_set("AWS_ACCESS_KEY_ID") or bool(os.getenv("AWS_PROFILE"))},
        "k8s":      {"configured": bool(os.getenv("KUBECONFIG") or os.getenv("K8S_IN_CLUSTER"))},
        "grafana":  {"configured": _env_set("GRAFANA_URL") and _env_set("GRAFANA_TOKEN")},
        "gitlab":   {"configured": _env_set("GITLAB_TOKEN")},
    }

    return {
        "llm":          llm,
        "github":       github,
        "integrations": integrations,
    }


# ── Clean public-facing aliases for all UI-shown paths ───────

# K8s — canonical clean paths
@app.get("/k8s/health")
def k8s_health():
    return {"k8s_check": check_k8s_cluster()}

@app.get("/k8s/pods")
def k8s_pods_clean(namespace: str = "default"):
    return {"k8s_pods": check_k8s_pods(namespace)}

@app.get("/k8s/deployments")
def k8s_deployments_clean(namespace: str = "default"):
    return {"k8s_deployments": check_k8s_deployments(namespace)}

@app.get("/k8s/nodes")
def k8s_nodes_clean():
    return {"k8s_nodes": check_k8s_nodes()}

@app.post("/k8s/diagnose")
def k8s_diagnose(req: AWSDiagnoseRequest):
    """AI-powered K8s namespace diagnosis using live pod/deployment data."""
    from app.llm.claude import analyze_context as _analyze
    pods = check_k8s_pods(req.resource_id or "default")
    deps = check_k8s_deployments(req.resource_id or "default")
    return _analyze({"incident_id": f"k8s-{req.resource_id}", "details": {"pods": pods, "deployments": deps}})

# Incidents — canonical paths matching UI
@app.post("/incidents/run")
def incidents_run_alias(req: IncidentRunRequest, x_user: Optional[str] = Header(default=None),
                        auth: Optional[AuthContext] = Depends(optional_auth)):
    """Primary incident pipeline endpoint — LangGraph multi-agent (same as /incident/run)."""
    return incident_run(req, x_user, auth)

@app.post("/incidents/run/async")
async def incidents_run_async(req: IncidentRunRequest, x_user: Optional[str] = Header(default=None)):
    """Fire-and-forget async pipeline — returns job ID immediately."""
    import asyncio, uuid
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, lambda: run_incident_pipeline(
        incident_id=req.incident_id, description=req.description,
        severity=req.severity, aws_config={}, k8s_config={}, auto_remediate=req.auto_remediate,
    ))
    return {"job_id": job_id, "status": "accepted", "incident_id": req.incident_id}

# GitHub — additional aliases
@app.get("/github/pr/{pr_number}/review")
def github_pr_review_clean(pr_number: int):
    """Get AI review for a PR by number."""
    from app.integrations.github import get_pr_for_review
    from app.llm.claude import review_pr
    data = get_pr_for_review(pr_number)
    if not data.get("success"):
        raise HTTPException(status_code=404, detail=data.get("error", "PR not found"))
    review = review_pr(data)
    return {"pr": pr_number, "review": review, "pr_data": data}

@app.post("/github/issue")
def github_issue_clean(title: str = "AI DevOps Issue", body: str = "", repo: str = ""):
    return create_issue(title=title, body=body, repo_name=repo)

# Jira — clean path
@app.post("/jira/incident")
def jira_incident_clean(summary: str = "AI DevOps Incident", description: str = ""):
    return create_incident(summary=summary, description=description or summary)

# Deploy — alias
@app.post("/aws/assess-deployment")
def aws_assess_deployment_alias(req: ContextRequest):
    from app.llm.claude import assess_deployment
    return assess_deployment(req.model_dump())

@app.post("/deploy/jira-to-pr")
def deploy_jira_to_pr_clean(issue_key: str):
    """Same as /jira/webhook but via direct call."""
    from app.integrations.jira import get_issue
    from app.llm.claude import interpret_jira_for_pr
    try:
        issue = get_issue(issue_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return interpret_jira_for_pr(issue)

# Security — GET to list roles
@app.get("/security/roles")
def security_roles_list():
    """List all configured RBAC roles and their permissions."""
    from app.security.rbac import ROLE_PERMISSIONS, _user_roles
    return {
        "roles": {r: list(p) for r, p in ROLE_PERMISSIONS.items()},
        "assignments": dict(_user_roles),
    }

@app.post("/security/roles/assign")
def security_roles_assign(req: RoleAssignment, x_user: Optional[str] = Header(default=None)):
    _rbac_guard(x_user, "manage_users")
    return assign_role(req.user, req.role)

# AWS — missing endpoints
@app.get("/aws/cloudwatch/logs")
def aws_cw_logs(log_group: str = "", minutes: int = 30):
    """Recent CloudWatch logs. Pass ?log_group=name to filter."""
    if log_group:
        from app.integrations.aws_ops import get_recent_logs
        result = get_recent_logs(log_group, minutes)
        return {"logs": result}
    from app.integrations.aws_ops import list_log_groups
    result = list_log_groups(limit=50)
    return {"log_groups": result}

@app.get("/aws/context")
def aws_context_snapshot(resource_type: str = "ec2", resource_id: str = ""):
    """Full AWS observability snapshot for a given resource."""
    result = collect_diagnosis_context(resource_type=resource_type, resource_id=resource_id)
    return {"aws_context": result}

@app.get("/aws/synthesize")
def aws_synthesize(incident_id: str = "snapshot", description: str = "infrastructure status review",
                   resource_type: str = "ec2", resource_id: str = ""):
    """AI synthesis of current AWS infrastructure state."""
    from app.llm.claude import synthesize_incident
    context = collect_diagnosis_context(resource_type=resource_type, resource_id=resource_id)
    result = synthesize_incident({"incident_id": incident_id, "description": description, "aws_context": context})
    return {"synthesis": result}

@app.get("/aws/route53/health")
def aws_route53_health_alias():
    """Route53 health checks (alias)."""
    result = list_route53_healthchecks()
    return {"route53_healthchecks": result}

@app.get("/aws/cost/summary")
def aws_cost_summary():
    """Approximate cost summary from resource counts (AWS Cost Explorer requires separate permission)."""
    ec2 = list_ec2_instances()
    rds = list_rds_instances()
    lam = list_lambda_functions()
    ecs = list_ecs_services()
    return {
        "note": "Resource inventory — configure AWS Cost Explorer for billing data",
        "ec2_count": len(ec2.get("instances", [])) if ec2.get("success") else "unavailable",
        "rds_count": len(rds.get("instances", [])) if rds.get("success") else "unavailable",
        "lambda_count": len(lam.get("functions", [])) if lam.get("success") else "unavailable",
        "ecs_services": len(ecs.get("services", [])) if ecs.get("success") else "unavailable",
    }

# Grafana — full implementation
@app.get("/grafana/alerts")
def grafana_alerts():
    """Firing Grafana alerts."""
    from app.integrations.grafana import get_firing_alerts
    return get_firing_alerts()

@app.get("/grafana/dashboards")
def grafana_dashboards():
    """Grafana datasources (dashboards API requires Grafana admin token)."""
    from app.integrations.grafana import get_datasources
    return get_datasources()

# WebSocket — clean /ws alias
@app.websocket("/ws")
async def websocket_ws(websocket: WebSocket):
    """WebSocket alias for /realtime/events."""
    # Check auth token from query param
    token = websocket.query_params.get("token", "")
    if token:
        try:
            from app.core.auth import decode_token
            payload = decode_token(token)
            ws_user = payload.get("sub", "anonymous")
        except Exception:
            await websocket.close(code=4401, reason="Invalid token")
            return
    else:
        ws_user = "anonymous"
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            events = [payload] if isinstance(payload, dict) else payload if isinstance(payload, list) else None
            if events is None:
                await websocket.send_json({"error": "invalid payload"})
                continue
            correlation = correlate_events(events)
            analysis = analyze_context({"incident_id": "ws-realtime", "details": events})
            await websocket.send_json({"correlation": correlation, "analysis": analysis})
    except WebSocketDisconnect:
        pass


@app.websocket("/realtime/events")
async def websocket_events(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()

            # Better to validate as events list in production.
            if isinstance(payload, dict):
                events = [payload]
            elif isinstance(payload, list):
                events = payload
            else:
                await websocket.send_json({"error": "invalid payload format, expected event or list"})
                continue

            correlation = correlate_events(events)
            analysis = analyze_context({"incident_id": "realtime", "details": events})

            await websocket.send_json({
                "correlation": correlation,
                "analysis": analysis,
            })
    except WebSocketDisconnect:
        pass

# ── Approvals ────────────────────────────────────────────────────────────────

@app.get("/approvals/pending", tags=["approvals"])
def list_pending_approvals_endpoint(auth: AuthContext = Depends(require_viewer)):
    try:
        from app.incident.approval import list_pending_approvals
        approvals = list_pending_approvals()
        return {"approvals": [vars(a) for a in approvals]}
    except Exception as e:
        return {"approvals": [], "error": str(e)}

class ApprovalDecision(BaseModel):
    approved_action_indices: List[int] = []
    reason: Optional[str] = None

@app.post("/approvals/{correlation_id}/approve", tags=["approvals"])
def approve_actions_endpoint(correlation_id: str, req: ApprovalDecision, auth: AuthContext = Depends(require_developer)):
    try:
        from app.incident.approval import approve_actions
        result = approve_actions(correlation_id, req.approved_action_indices, auth.username)
        return {"success": True, "approval": vars(result)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/approvals/{correlation_id}/reject", tags=["approvals"])
def reject_approval_endpoint(correlation_id: str, req: ApprovalDecision, auth: AuthContext = Depends(require_developer)):
    try:
        from app.incident.approval import reject_approval
        result = reject_approval(correlation_id, req.reason or "Rejected by user", auth.username)
        # Remove from pending states
        _PENDING_PIPELINE_STATES.pop(correlation_id, None)
        return {"success": True, "approval": vars(result)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/approvals/{correlation_id}/resume", tags=["approvals"])
def resume_approved_pipeline(
    correlation_id: str,
    auth: AuthContext = Depends(require_developer),
):
    """Resume an awaiting-approval pipeline after human approval.

    Retrieves the paused pipeline state, filters to only approved actions,
    and runs execute → validate → store_memory using the existing context
    so no re-collection or re-planning is needed.
    """
    from app.incident.approval import get_approval_request, STATUS_APPROVED
    from app.execution.executor import Executor
    from app.execution.validator import Validator
    from app.agents.memory.agent import MemoryAgent
    import datetime as _dt

    # Check approval record exists and is approved
    approval = get_approval_request(correlation_id)
    if not approval:
        raise HTTPException(status_code=404, detail=f"Approval {correlation_id} not found")
    if approval.status != STATUS_APPROVED:
        raise HTTPException(
            status_code=400,
            detail=f"Approval is in state '{approval.status}', not approved",
        )

    # Retrieve the saved pipeline state
    saved_state = _PENDING_PIPELINE_STATES.get(correlation_id)
    if not saved_state:
        raise HTTPException(
            status_code=404,
            detail="Pipeline state not found — it may have expired. Re-run the incident.",
        )

    # Build a resumable state: restore plan but allow execution
    resume_state = dict(saved_state)
    resume_state["auto_remediate"]         = True
    resume_state["requires_human_approval"] = False
    resume_state["approval_reason"]         = f"Manually approved by {auth.username}"
    resume_state["status"]                  = "running"

    # Filter plan actions to only those approved
    plan = resume_state.get("plan") or {}
    all_actions = plan.get("actions", [])
    if approval.approved_action_indices:
        approved_actions = [
            all_actions[i] for i in approval.approved_action_indices
            if i < len(all_actions)
        ]
        resume_state["plan"] = {**plan, "actions": approved_actions}

    try:
        # Execute approved actions
        executor = Executor()
        resume_state = executor.run(resume_state)

        # Validate
        validator = Validator()
        resume_state = validator.run(resume_state)

        # Store memory
        memory_agent = MemoryAgent()
        resume_state = memory_agent.run(resume_state)

        resume_state["status"]       = "completed"
        resume_state["completed_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        resume_state["resumed_by"]   = auth.username

    except Exception as exc:
        resume_state["status"] = "failed"
        resume_state["errors"] = resume_state.get("errors", []) + [str(exc)]

    # Clean up pending state
    _PENDING_PIPELINE_STATES.pop(correlation_id, None)
    return resume_state

# ── War Room AI ───────────────────────────────────────────────────────────────

class WarRoomQuestion(BaseModel):
    question: str
    asked_by: Optional[str] = None

@app.post("/warroom/{war_room_id}/ask", tags=["warroom"])
async def ask_war_room_ai(war_room_id: str, req: WarRoomQuestion, auth: AuthContext = Depends(require_viewer)):
    try:
        from app.incident.war_room_intelligence import answer_war_room_question
        answer = answer_war_room_question(war_room_id, req.question, req.asked_by or auth.username)
        return {"answer": answer, "war_room_id": war_room_id}
    except Exception as e:
        return {"answer": f"War room intelligence unavailable: {e}", "war_room_id": war_room_id}

@app.get("/warroom/active", tags=["warroom"])
def list_active_war_rooms(auth: AuthContext = Depends(require_viewer)):
    try:
        from app.incident.war_room_intelligence import list_active_war_rooms as _list
        return {"war_rooms": _list()}
    except Exception as e:
        return {"war_rooms": [], "error": str(e)}


class SlackSendRequest(BaseModel):
    message:  str
    sent_by:  str = "user"


@app.get("/warroom/{war_room_id}/slack-history", tags=["warroom"])
def get_war_room_slack_history(war_room_id: str, limit: int = 30, auth: AuthContext = Depends(require_viewer)):
    """Fetch recent messages from the Slack channel linked to this war room."""
    try:
        from app.incident.war_room_intelligence import _war_rooms
        import datetime as _dt
        wr = _war_rooms.get(war_room_id)
        channel = wr.slack_channel if wr else ""
        if not channel:
            return {"messages": [], "channel": "", "note": "No Slack channel linked to this war room"}
        from app.integrations.slack import _client as _slack_client
        sc = _slack_client()
        # resolve channel name to ID
        ch_id = channel
        if not channel.startswith("C"):  # not already an ID
            name = channel.lstrip("#")
            result = sc.conversations_list(types="public_channel,private_channel", limit=500)
            for ch in result.get("channels", []):
                if ch["name"] == name:
                    ch_id = ch["id"]
                    break
        resp = sc.conversations_history(channel=ch_id, limit=limit)
        messages = []
        for m in reversed(resp.get("messages", [])):
            if m.get("subtype"):
                continue  # skip join/leave system messages
            ts = float(m.get("ts", 0))
            time_str = _dt.datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else ""
            # try to resolve user display name
            username = m.get("username") or m.get("user", "unknown")
            messages.append({"username": username, "text": m.get("text", ""), "time": time_str, "ts": m.get("ts","")})
        return {"messages": messages, "channel": channel, "count": len(messages)}
    except Exception as e:
        return {"messages": [], "channel": "", "error": str(e)}


@app.post("/warroom/{war_room_id}/slack-send", tags=["warroom"])
def send_war_room_slack_message(war_room_id: str, req: SlackSendRequest, auth: AuthContext = Depends(require_viewer)):
    """Send a message to the Slack channel linked to this war room."""
    try:
        from app.incident.war_room_intelligence import _war_rooms
        wr = _war_rooms.get(war_room_id)
        channel = wr.slack_channel if wr else ""
        if not channel:
            raise HTTPException(status_code=400, detail="No Slack channel linked to this war room")
        from app.integrations.slack import post_message
        text = f"*{req.sent_by}* (via NexusOps): {req.message}"
        result = post_message(channel=channel, text=text)
        return {"success": result.get("ok", False), "channel": channel}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Post-Mortem ────────────────────────────────────────────────────────────────

class PostMortemRequest(BaseModel):
    incident_id:       str
    description:       Optional[str] = None
    root_cause:        Optional[str] = None
    severity:          Optional[str] = "SEV2"
    started_at:        Optional[str] = None
    resolved_at:       Optional[str] = None
    actions_taken:     Optional[list] = []
    validation:        Optional[dict] = {}
    errors:            Optional[list] = []
    save_to_disk:      bool = False

@app.post("/incidents/{incident_id}/post-mortem", tags=["incidents"])
def generate_post_mortem_endpoint(
    incident_id: str,
    req: PostMortemRequest,
    auth: AuthContext = Depends(require_viewer),
):
    """Generate an AI-written blameless post-mortem for a resolved incident."""
    try:
        from app.incident.post_mortem import generate_post_mortem, format_as_markdown, save_post_mortem
        state = {
            "incident_id":    incident_id,
            "description":    req.description or "",
            "root_cause":     req.root_cause or "",
            "severity":       req.severity,
            "started_at":     req.started_at or "",
            "resolved_at":    req.resolved_at or "",
            "actions_taken":  req.actions_taken or [],
            "validation":     req.validation or {},
            "errors":         req.errors or [],
        }
        pm       = generate_post_mortem(state)
        markdown = format_as_markdown(pm)
        result   = {
            "incident_id":          pm.incident_id,
            "title":                pm.title,
            "severity":             pm.severity,
            "duration_minutes":     pm.duration_minutes,
            "root_cause":           pm.root_cause,
            "contributing_factors": pm.contributing_factors,
            "impact":               pm.impact,
            "resolution":           pm.resolution,
            "action_items":         [vars(a) for a in pm.action_items],
            "lessons_learned":      pm.lessons_learned,
            "prevention_steps":     pm.prevention_steps,
            "generated_at":         pm.generated_at,
            "markdown":             markdown,
        }
        if req.save_to_disk:
            saved_path = save_post_mortem(pm)
            result["saved_to"] = saved_path
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Post-mortem generation failed: {e}")

# /chat/v2 is retired — /chat now uses the intelligent engine directly

@app.get("/chat/sessions", tags=["chat"])
def list_chat_sessions(auth: AuthContext = Depends(require_viewer)):
    try:
        from app.chat.memory import list_sessions
        return {"sessions": list_sessions()}
    except Exception as e:
        return {"sessions": [], "error": str(e)}

# ── Webhook receivers ─────────────────────────────────────────────────────────

@app.post("/webhooks/grafana", tags=["webhooks"])
async def grafana_webhook(payload: Dict[str, Any]):
    try:
        from app.integrations.webhooks import process_grafana_webhook
        return process_grafana_webhook(payload)
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/webhooks/cloudwatch", tags=["webhooks"])
async def cloudwatch_webhook(payload: Dict[str, Any]):
    try:
        from app.integrations.webhooks import process_cloudwatch_webhook
        return process_cloudwatch_webhook(payload)
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/webhooks/opsgenie", tags=["webhooks"])
async def opsgenie_webhook(payload: Dict[str, Any]):
    try:
        from app.integrations.webhooks import process_opsgenie_webhook
        return process_opsgenie_webhook(payload)
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# ── Post-mortem ───────────────────────────────────────────────────────────────

class PostMortemRequest(BaseModel):
    incident_id: str
    incident_state: Dict[str, Any]

# ── Cost Analysis ─────────────────────────────────────────────────────────────

class CostAnalysisRequest(BaseModel):
    actions: List[Dict[str, Any]]
    aws_cfg: Optional[Dict[str, Any]] = None

@app.post("/cost/analyze", tags=["cost"])
async def analyze_costs_endpoint(req: CostAnalysisRequest, auth: AuthContext = Depends(require_viewer)):
    try:
        from app.cost.analyzer import analyze_action_costs, format_cost_report
        import dataclasses as _dc
        report = analyze_action_costs(req.actions, req.aws_cfg)
        report_dict = _dc.asdict(report)
        return {"report": report_dict, "formatted": format_cost_report(report)}
    except Exception as e:
        return {"report": None, "formatted": f"Cost analysis unavailable: {e}", "error": str(e)}

@app.get("/cost/dashboard", tags=["cost"])
async def cost_dashboard_endpoint(auth: AuthContext = Depends(require_viewer)):
    """Full cost dashboard: MTD spend, forecast, service breakdown, 6-month trend."""
    try:
        from app.cost.analyzer import fetch_cost_dashboard
        return fetch_cost_dashboard()
    except Exception as e:
        return {"available": False, "error": str(e), "current_monthly_spend": 0.0,
                "last_month_spend": 0.0, "forecast_month_end": 0.0,
                "service_breakdown": [], "monthly_trend": []}
