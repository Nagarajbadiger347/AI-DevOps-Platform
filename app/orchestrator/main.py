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

@asynccontextmanager
async def _lifespan(_: FastAPI):
    if _settings.ENABLE_MONITOR_LOOP:
        import asyncio
        from app.monitoring.loop import monitoring_loop
        asyncio.create_task(monitoring_loop())
    # Validate critical configuration
    import warnings as _warnings
    if not os.getenv("JWT_SECRET_KEY"):
        _warnings.warn("JWT_SECRET_KEY not set — using insecure default. Set it with: openssl rand -hex 32")
    llm_keys = [os.getenv(k) for k in ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY")]
    if not any(llm_keys):
        _warnings.warn("No LLM API key configured (ANTHROPIC_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY). AI features will fail.")
    yield

app = FastAPI(
    title="AI DevOps Intelligence Platform",
    description="Autonomous DevOps management powered by multi-agent AI — built by Nagaraj",
    version="2.0.0",
    lifespan=_lifespan,
)
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("favicon.ico")

@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def chrome_devtools():
    """Silence Chrome DevTools 404 noise."""
    return {}

_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

def _inc(key: str, amount: int = 1):
    _METRICS[key] += amount

@app.middleware("http")
async def _rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    method = request.method
    # Only rate-limit AI-heavy endpoints
    if path in ("/chat", "/incidents/run", "/warroom/create"):
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
    incident_id: str
    description: str
    severity: str = "high"          # critical | high | medium | low
    aws: AWSConfig = None
    k8s: K8sConfig = None
    auto_remediate: bool = False    # if True, execute K8s control-plane actions automatically
    hours: int = 2                  # lookback window for observability data

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>DevOps AI — Autonomous Operations Platform</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet"/>
  <style>
    *,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
    :root{
      --bg:#04060f;--bg2:#070b18;--bg3:#0a0f1e;
      --surface:#0d1424;--surface2:#111a2e;--surface3:#162038;
      --border:#1a2540;--border2:#223060;
      --text:#e2e8f0;--text2:#8b9ec7;--muted:#3d5080;
      --blue:#4f8ef7;--blue2:#2563eb;--blue3:#1d4ed8;
      --cyan:#22d3ee;--cyan2:#0891b2;
      --purple:#a78bfa;--purple2:#7c3aed;
      --green:#34d399;--green2:#059669;
      --red:#f87171;--amber:#fbbf24;
      --glow-blue:rgba(79,142,247,0.15);
      --glow-purple:rgba(167,139,250,0.12);
      --glow-cyan:rgba(34,211,238,0.1);
      --radius:10px;--radius-sm:6px;--radius-lg:16px;
    }
    html{scroll-behavior:smooth}
    body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex;font-size:13px;overflow:hidden}
    ::-webkit-scrollbar{width:3px;height:3px}
    ::-webkit-scrollbar-track{background:transparent}
    ::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px}

    /* ── ANIMATED BG ── */
    body::before{content:'';position:fixed;inset:0;background:radial-gradient(ellipse 80% 60% at 20% 0%,rgba(37,99,235,0.08) 0%,transparent 60%),radial-gradient(ellipse 60% 50% at 80% 100%,rgba(124,58,237,0.07) 0%,transparent 60%);pointer-events:none;z-index:0}

    /* ── SIDEBAR ── */
    .sidebar{width:224px;flex-shrink:0;background:var(--bg2);border-right:1px solid var(--border);display:flex;flex-direction:column;height:100vh;position:sticky;top:0;z-index:10}
    .sb-logo{padding:18px 16px 14px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px}
    .logo-mark{width:34px;height:34px;background:linear-gradient(135deg,#2563eb,#7c3aed);border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:17px;flex-shrink:0;box-shadow:0 0 20px rgba(79,142,247,0.4)}
    .logo-text .name{font-size:14px;font-weight:800;letter-spacing:-.02em;background:linear-gradient(90deg,#4f8ef7,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .logo-text .tag{font-size:9.5px;color:var(--muted);font-weight:400;letter-spacing:.04em;margin-top:1px}
    .sb-status{padding:10px 14px;border-bottom:1px solid var(--border)}
    .status-row{display:flex;align-items:center;gap:6px;font-size:11px}
    .pulse{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 0 0 rgba(52,211,153,.6);animation:pulse 2s ease infinite}
    @keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(52,211,153,.5)}50%{box-shadow:0 0 0 5px rgba(52,211,153,0)}}
    .sb-section{padding:10px 8px 4px}
    .sb-label{font-size:9px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);padding:0 8px 7px;display:block}
    .nav-link{display:flex;align-items:center;gap:9px;padding:7px 10px;border-radius:var(--radius-sm);font-size:12.5px;font-weight:500;color:var(--text2);cursor:pointer;border:none;background:transparent;width:100%;text-align:left;text-decoration:none;transition:all .15s;position:relative;outline:none}
    .nav-link:hover{background:var(--surface2);color:var(--text)}
    .nav-link.active{background:linear-gradient(90deg,rgba(37,99,235,0.18),rgba(79,142,247,0.06));color:var(--blue);border-right:2px solid var(--blue)}
    .nav-link .ico{font-size:13px;width:18px;text-align:center;flex-shrink:0}
    .nav-link .cnt{margin-left:auto;font-size:10px;background:var(--surface3);color:var(--text2);padding:1px 6px;border-radius:10px;font-weight:600}
    .nav-link.active .cnt{background:rgba(79,142,247,0.2);color:var(--blue)}
    .sb-divider{height:1px;background:var(--border);margin:6px 14px}
    .sb-footer{margin-top:auto;padding:12px 14px;border-top:1px solid var(--border)}
    .sb-user{display:flex;align-items:center;gap:9px}
    .avatar{width:30px;height:30px;border-radius:8px;background:linear-gradient(135deg,#2563eb,#7c3aed);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;flex-shrink:0}
    .user-info .uname{font-size:12px;font-weight:600;color:var(--text)}
    .user-info .urole{font-size:10px;color:var(--muted)}
    .live-dot{margin-left:auto;display:flex;align-items:center;gap:4px;font-size:9.5px;color:var(--green)}

    /* ── MAIN ── */
    .main{flex:1;min-width:0;display:flex;flex-direction:column;height:100vh;overflow:hidden;position:relative;z-index:1}

    /* ── TOPBAR ── */
    .topbar{display:flex;align-items:center;gap:12px;padding:0 20px;height:52px;border-bottom:1px solid var(--border);background:rgba(4,6,15,0.8);backdrop-filter:blur(12px);flex-shrink:0}
    .topbar-title{font-size:14px;font-weight:700;letter-spacing:-.01em}
    .topbar-ver{font-size:10px;color:var(--muted);background:var(--surface2);padding:2px 7px;border-radius:10px;border:1px solid var(--border)}
    .topbar-spacer{flex:1}
    .tb-chip{display:flex;align-items:center;gap:5px;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid var(--border);background:var(--surface);cursor:pointer;transition:all .15s;color:var(--text2)}
    .tb-chip:hover{border-color:var(--blue);color:var(--blue)}
    .tb-chip.on{border-color:rgba(52,211,153,.4);color:var(--green);background:rgba(52,211,153,.06)}
    .tb-chip .dot{width:5px;height:5px;border-radius:50%;background:currentColor}

    /* ── METRIC STRIP ── */
    .metric-strip{display:flex;gap:12px;padding:14px 20px;border-bottom:1px solid var(--border);flex-shrink:0;overflow-x:auto;background:var(--bg2)}
    .metric-strip::-webkit-scrollbar{height:0}
    .mc{flex:1;min-width:120px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:10px 14px;position:relative;overflow:hidden;cursor:default;transition:border-color .2s}
    .mc:hover{border-color:var(--border2)}
    .mc::before{content:'';position:absolute;inset:0;opacity:0;background:radial-gradient(ellipse at 50% 0%,var(--glow-blue),transparent 70%);transition:opacity .3s}
    .mc:hover::before{opacity:1}
    .mc-label{font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:6px;display:flex;align-items:center;gap:5px}
    .mc-val{font-size:22px;font-weight:800;letter-spacing:-.02em;line-height:1}
    .mc-sub{font-size:10px;color:var(--text2);margin-top:4px}
    .mc-blue .mc-val{color:var(--blue)}
    .mc-green .mc-val{color:var(--green)}
    .mc-purple .mc-val{color:var(--purple)}
    .mc-cyan .mc-val{color:var(--cyan)}
    .mc-amber .mc-val{color:var(--amber)}
    .mc-bar{position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--blue2),var(--purple2))}
    .mc-green .mc-bar{background:linear-gradient(90deg,var(--green2),var(--cyan2))}
    .mc-purple .mc-bar{background:linear-gradient(90deg,var(--purple2),var(--blue2))}
    .mc-cyan .mc-bar{background:linear-gradient(90deg,var(--cyan2),var(--blue2))}
    .mc-amber .mc-bar{background:linear-gradient(90deg,#d97706,var(--amber))}

    /* ── CONTENT WRAP ── */
    #content-wrap{flex:1;display:flex;flex-direction:column;overflow:hidden;min-height:0}

    /* ── VIEW: ENDPOINTS ── */
    #view-endpoints{flex:1;overflow-y:auto;padding:20px}
    .view-header{display:flex;align-items:center;gap:10px;margin-bottom:16px}
    .view-header h2{font-size:15px;font-weight:700}
    .view-header p{font-size:12px;color:var(--text2)}
    .search-bar{display:flex;align-items:center;gap:8px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:7px 12px;flex:1;max-width:340px}
    .search-bar input{background:transparent;border:none;outline:none;font-size:12.5px;color:var(--text);flex:1;font-family:inherit}
    .search-bar input::placeholder{color:var(--muted)}
    .ep-group{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;margin-bottom:10px}
    .ep-group-hdr{display:flex;align-items:center;gap:9px;padding:10px 14px;background:var(--surface2);border-bottom:1px solid var(--border)}
    .ep-group-hdr .ico{font-size:14px}
    .g-name{font-size:12.5px;font-weight:700;color:var(--text);flex:1}
    .g-cnt{font-size:10px;color:var(--muted);background:var(--surface3);padding:1px 7px;border-radius:10px;border:1px solid var(--border)}
    .ep-row{display:flex;align-items:center;gap:12px;padding:8px 14px;border-top:1px solid var(--border);cursor:pointer;transition:background .12s}
    .ep-row:hover{background:var(--surface2)}
    .ep-row.hidden{display:none}
    .badge{font-size:9px;font-weight:700;padding:2px 7px;border-radius:4px;letter-spacing:.04em;text-transform:uppercase;flex-shrink:0;min-width:42px;text-align:center}
    .GET{background:rgba(52,211,153,.12);color:var(--green);border:1px solid rgba(52,211,153,.25)}
    .POST{background:rgba(79,142,247,.12);color:var(--blue);border:1px solid rgba(79,142,247,.25)}
    .DELETE{background:rgba(248,113,113,.12);color:var(--red);border:1px solid rgba(248,113,113,.25)}
    .PATCH,.PUT{background:rgba(251,191,36,.12);color:var(--amber);border:1px solid rgba(251,191,36,.25)}
    .ep-path{font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--text2);flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;transition:color .12s}
    .ep-row:hover .ep-path{color:var(--text)}
    .ep-desc{font-size:11px;color:var(--muted);flex:1.5;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .ep-lock{font-size:10px;color:var(--amber);opacity:.7}

    /* ── VIEW: SECRETS ── */
    .secrets-panel{display:none;flex:1;overflow-y:auto;padding:20px}
    .secrets-panel.active{display:block}
    .sec-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;margin-bottom:10px}
    .sec-card-hdr{display:flex;align-items:center;gap:9px;padding:10px 14px;background:var(--surface2);border-bottom:1px solid var(--border)}
    .sec-row{display:flex;align-items:center;gap:12px;padding:8px 14px;border-top:1px solid var(--border)}
    .sec-key{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text2);width:240px;flex-shrink:0}
    .sec-input{flex:1;background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:6px 10px;font-size:11.5px;font-family:'JetBrains Mono',monospace;color:var(--text);outline:none;transition:border-color .15s,box-shadow .15s}
    .sec-input:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(79,142,247,.1)}
    .sec-input::placeholder{color:var(--muted)}
    .sec-status{width:16px;flex-shrink:0;font-size:12px;text-align:center}
    .sec-actions-bar{display:flex;align-items:center;gap:10px;padding:14px 20px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);margin-bottom:16px}
    .sec-user-wrap{display:flex;flex-direction:column;gap:3px}
    .sec-user-lbl{font-size:9.5px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.1em}
    .sec-user-input{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:6px 10px;font-size:12px;font-family:'JetBrains Mono',monospace;color:var(--text);outline:none;width:180px;transition:border-color .15s}
    .sec-user-input:focus{border-color:var(--blue)}
    .save-btn{margin-left:auto;padding:7px 18px;background:linear-gradient(135deg,#2563eb,#1d4ed8);border:1px solid rgba(79,142,247,.4);border-radius:var(--radius-sm);font-size:12px;font-weight:700;color:#fff;cursor:pointer;transition:all .15s;font-family:inherit}
    .save-btn:hover{background:linear-gradient(135deg,#3b82f6,#2563eb);box-shadow:0 0 16px rgba(79,142,247,.3)}
    .save-btn:disabled{background:var(--surface2);border-color:var(--border);color:var(--muted);cursor:not-allowed;box-shadow:none}
    .sec-msg{font-size:11.5px;font-weight:500;padding:4px 10px;border-radius:var(--radius-sm);display:none}
    .sec-msg.ok{background:rgba(52,211,153,.12);color:var(--green);border:1px solid rgba(52,211,153,.3)}
    .sec-msg.err{background:rgba(248,113,113,.12);color:var(--red);border:1px solid rgba(248,113,113,.3)}
    .int-chip{display:inline-flex;align-items:center;gap:4px;font-size:10px;font-weight:600;padding:2px 8px;border-radius:20px;border:1px solid var(--border);background:var(--surface2);color:var(--muted);transition:all .2s}
    .int-chip.on{border-color:rgba(52,211,153,.4);color:var(--green);background:rgba(52,211,153,.08)}
    .int-chip .dot{width:5px;height:5px;border-radius:50%;background:currentColor}

    /* ── VIEW: PIPELINE ── */
    .pipeline-card{background:var(--surface);border:1px solid rgba(79,142,247,.2);border-radius:var(--radius);overflow:hidden;margin-bottom:12px;box-shadow:0 0 30px rgba(79,142,247,.04)}
    .pipeline-hdr{display:flex;align-items:center;gap:8px;padding:10px 14px;background:linear-gradient(90deg,rgba(37,99,235,.1),rgba(124,58,237,.06));border-bottom:1px solid rgba(79,142,247,.15)}
    .pipeline-badge{font-size:9px;font-weight:700;background:rgba(79,142,247,.15);color:var(--blue);border:1px solid rgba(79,142,247,.3);padding:2px 8px;border-radius:20px;letter-spacing:.06em;text-transform:uppercase}
    .flow-row{display:flex;align-items:center;padding:14px 16px;gap:0;overflow-x:auto;border-bottom:1px solid var(--border);background:var(--surface2)}
    .fstep{flex:1;min-width:90px;background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius-sm);padding:10px 8px;text-align:center;transition:all .2s}
    .fstep:hover{border-color:var(--blue);box-shadow:0 0 12px rgba(79,142,247,.1)}
    .fstep-ic{font-size:18px;display:block;margin-bottom:5px}
    .fstep-lb{font-size:10.5px;font-weight:700;color:var(--text)}
    .fstep-sb{font-size:9px;color:var(--muted);margin-top:2px}
    .farr{padding:0 8px;color:var(--border2);font-size:16px;flex-shrink:0;user-select:none}
    pre.sample{margin:0;padding:12px 14px;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text2);line-height:1.7;overflow-x:auto;background:var(--bg2);border-top:1px solid var(--border);white-space:pre-wrap}

    /* ── CHAT PANEL ── */
    #view-chat{display:none;flex-direction:column;height:100vh;min-height:0}
    .chat-topbar{padding:12px 20px;border-bottom:1px solid var(--border);background:var(--bg2);display:flex;align-items:center;gap:10px;flex-shrink:0}
    .chat-topbar-title{font-size:14px;font-weight:700;background:linear-gradient(90deg,var(--blue),var(--purple));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .chat-topbar-sub{font-size:10.5px;color:var(--muted)}
    .tb-spacer{flex:1}
    .llm-select{background:var(--surface2);border:1px solid var(--border2);border-radius:var(--radius-sm);color:var(--text2);font-size:11.5px;padding:4px 8px;outline:none;cursor:pointer;font-family:inherit;transition:border-color .15s}
    .llm-select:focus{border-color:var(--blue)}
    .chat-warroom-btn{padding:6px 12px;background:linear-gradient(135deg,rgba(248,113,113,.15),rgba(220,38,38,.1));border:1px solid rgba(248,113,113,.3);border-radius:var(--radius-sm);font-size:11.5px;font-weight:700;color:var(--red);cursor:pointer;transition:all .15s;font-family:inherit}
    .chat-warroom-btn:hover{background:linear-gradient(135deg,rgba(248,113,113,.25),rgba(220,38,38,.2));box-shadow:0 0 12px rgba(248,113,113,.15)}
    .chat-messages{flex:1;overflow-y:auto;padding:24px 20px;display:flex;flex-direction:column;gap:16px}
    .chat-empty{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;text-align:center;padding:24px}
    .chat-empty-icon{font-size:44px;filter:drop-shadow(0 0 20px rgba(79,142,247,.4))}
    .chat-empty-title{font-size:18px;font-weight:800;background:linear-gradient(90deg,var(--blue),var(--purple));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .chat-empty-sub{font-size:12.5px;color:var(--text2);max-width:420px;line-height:1.7}
    .chat-suggestions{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:4px}
    .chat-suggestion{background:var(--surface);border:1px solid var(--border2);border-radius:20px;padding:6px 14px;font-size:11.5px;color:var(--text2);cursor:pointer;transition:all .15s;font-family:inherit}
    .chat-suggestion:hover{border-color:var(--blue);color:var(--blue);background:rgba(79,142,247,.06)}
    .chat-row{display:flex;flex-direction:column;gap:4px}
    .chat-row.user{align-items:flex-end}
    .chat-row.ai{align-items:flex-start}
    .chat-meta{font-size:10px;color:var(--muted);padding:0 4px}
    .chat-bubble{max-width:78%;padding:11px 14px;border-radius:12px;line-height:1.65;font-size:13px;word-break:break-word}
    .chat-bubble.user{background:linear-gradient(135deg,#1d4ed8,#2563eb);color:#fff;border-radius:12px 12px 3px 12px;box-shadow:0 4px 16px rgba(37,99,235,.25)}
    .chat-bubble.ai{background:var(--surface2);border:1px solid var(--border2);border-radius:12px 12px 12px 3px;color:var(--text)}
    .chat-bubble pre{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:10px 12px;overflow-x:auto;margin:8px 0;font-family:'JetBrains Mono',monospace;font-size:11px}
    .chat-bubble code{background:rgba(255,255,255,.08);padding:1px 5px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:11.5px}
    .chat-bubble ul{margin:6px 0 6px 16px}
    .chat-bubble li{margin-bottom:3px}
    .chat-bubble h2,.chat-bubble h3,.chat-bubble h4{margin:10px 0 4px;font-weight:700;color:var(--text)}
    .chat-bubble h2{font-size:14px}
    .chat-bubble h3{font-size:13px}
    .chat-bubble h4{font-size:12px;color:var(--text2)}
    .chat-bubble strong{color:var(--text)}
    .chat-bubble hr{border:none;border-top:1px solid var(--border);margin:8px 0}
    .chat-bubble.typing{min-width:60px}
    .typing-dots{display:inline-flex;gap:4px;align-items:center;height:18px}
    .typing-dots span{width:7px;height:7px;border-radius:50%;background:var(--blue);animation:td .9s ease infinite}
    .typing-dots span:nth-child(2){animation-delay:.2s}
    .typing-dots span:nth-child(3){animation-delay:.4s}
    @keyframes td{0%,80%,100%{transform:scale(.7);opacity:.4}40%{transform:scale(1);opacity:1}}
    .chat-input-bar{display:flex;align-items:flex-end;gap:10px;padding:14px 20px;border-top:1px solid var(--border);background:var(--bg2);flex-shrink:0}
    #chat-input{flex:1;background:var(--surface);border:1px solid var(--border2);border-radius:var(--radius);padding:10px 14px;font-size:13px;font-family:'Inter',sans-serif;color:var(--text);outline:none;resize:none;max-height:130px;min-height:42px;line-height:1.55;transition:border-color .15s,box-shadow .15s}
    #chat-input:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(79,142,247,.1)}
    #chat-input::placeholder{color:var(--muted)}
    .chat-send-btn{padding:10px 18px;background:linear-gradient(135deg,#2563eb,#1d4ed8);border:none;border-radius:var(--radius-sm);font-size:12.5px;font-weight:700;color:#fff;cursor:pointer;transition:all .15s;flex-shrink:0;font-family:inherit;box-shadow:0 0 16px rgba(37,99,235,.3)}
    .chat-send-btn:hover{background:linear-gradient(135deg,#3b82f6,#2563eb);box-shadow:0 0 24px rgba(79,142,247,.4)}
    .chat-send-btn:disabled{background:var(--surface2);box-shadow:none;cursor:not-allowed;color:var(--muted)}

    /* ── FOOTER ── */
    .footer{padding:9px 20px;border-top:1px solid var(--border);display:flex;align-items:center;justify-content:center;gap:6px;font-size:10.5px;color:var(--muted);flex-shrink:0;background:var(--bg2)}
    .footer a{color:var(--text2);text-decoration:none;font-weight:500}
    .footer a:hover{color:var(--blue)}

    /* ── ANIMATIONS ── */
    @keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
    .fade-in{animation:fadeIn .25s ease}
    @keyframes slideUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:none}}
    @keyframes slideDown{from{opacity:0;transform:translateY(-10px)}to{opacity:1;transform:none}}

    /* ── TOAST ── */
    #toast-wrap{position:fixed;bottom:20px;right:20px;display:flex;flex-direction:column;gap:8px;z-index:9999;pointer-events:none}
    .toast{display:flex;align-items:center;gap:9px;padding:10px 14px;background:var(--surface2);border:1px solid var(--border2);border-radius:var(--radius);font-size:12px;color:var(--text);box-shadow:0 8px 24px rgba(0,0,0,.4);animation:slideUp .25s ease;min-width:200px;pointer-events:auto}
    .toast.ok{border-color:rgba(52,211,153,.4);background:rgba(52,211,153,.08)}
    .toast.err{border-color:rgba(248,113,113,.4);background:rgba(248,113,113,.08)}
    .toast.info{border-color:rgba(79,142,247,.4);background:rgba(79,142,247,.08)}

    /* ── RUN PIPELINE MODAL ── */
    .modal-overlay{position:fixed;inset:0;background:rgba(4,6,15,.7);backdrop-filter:blur(6px);z-index:100;display:none;align-items:center;justify-content:center}
    .modal-overlay.open{display:flex}
    .modal{background:var(--surface);border:1px solid var(--border2);border-radius:var(--radius-lg);padding:24px;width:480px;max-width:95vw;animation:slideUp .25s ease;box-shadow:0 24px 60px rgba(0,0,0,.5)}
    .modal-title{font-size:16px;font-weight:800;margin-bottom:4px;background:linear-gradient(90deg,var(--blue),var(--purple));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .modal-sub{font-size:12px;color:var(--muted);margin-bottom:20px}
    .modal-field{display:flex;flex-direction:column;gap:5px;margin-bottom:14px}
    .modal-label{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
    .modal-input{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:8px 11px;font-size:12.5px;color:var(--text);outline:none;font-family:inherit;transition:border-color .15s}
    .modal-input:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(79,142,247,.1)}
    .modal-input::placeholder{color:var(--muted)}
    textarea.modal-input{resize:vertical;min-height:72px;line-height:1.5}
    .modal-row{display:flex;gap:10px}
    .modal-row .modal-field{flex:1}
    .modal-actions{display:flex;gap:8px;margin-top:20px;justify-content:flex-end}
    .btn-cancel{padding:7px 16px;background:transparent;border:1px solid var(--border2);border-radius:var(--radius-sm);font-size:12px;font-weight:600;color:var(--text2);cursor:pointer;font-family:inherit;transition:all .15s}
    .btn-cancel:hover{border-color:var(--muted);color:var(--text)}
    .btn-run{padding:7px 20px;background:linear-gradient(135deg,#2563eb,#7c3aed);border:none;border-radius:var(--radius-sm);font-size:12px;font-weight:700;color:#fff;cursor:pointer;font-family:inherit;transition:all .15s;box-shadow:0 0 16px rgba(79,142,247,.25)}
    .btn-run:hover{box-shadow:0 0 24px rgba(79,142,247,.45)}
    .btn-run:disabled{background:var(--surface2);box-shadow:none;color:var(--muted);cursor:not-allowed}
    .sev-pills{display:flex;gap:6px;flex-wrap:wrap}
    .sev-pill{padding:4px 12px;border-radius:20px;border:1px solid var(--border2);font-size:11px;font-weight:600;cursor:pointer;transition:all .15s;background:transparent;font-family:inherit;color:var(--text2)}
    .sev-pill:hover{border-color:var(--blue);color:var(--blue)}
    .sev-pill.active{background:rgba(79,142,247,.15);border-color:var(--blue);color:var(--blue)}
    .sev-pill.critical.active{background:rgba(248,113,113,.15);border-color:var(--red);color:var(--red)}
    .sev-pill.high.active{background:rgba(251,191,36,.15);border-color:var(--amber);color:var(--amber)}
    .sev-pill.medium.active{background:rgba(79,142,247,.15);border-color:var(--blue);color:var(--blue)}
    .sev-pill.low.active{background:rgba(52,211,153,.15);border-color:var(--green);color:var(--green)}
    .modal-result{margin-top:14px;padding:12px;background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius-sm);font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text2);max-height:160px;overflow-y:auto;display:none;white-space:pre-wrap}

    /* ── GITHUB REPOS DRAWER ── */
    .gh-drawer{position:fixed;top:0;right:-420px;width:400px;height:100vh;background:var(--bg2);border-left:1px solid var(--border2);z-index:50;display:flex;flex-direction:column;transition:right .3s cubic-bezier(.4,0,.2,1);box-shadow:-8px 0 32px rgba(0,0,0,.4)}
    .gh-drawer.open{right:0}
    .gh-drawer-hdr{padding:16px 18px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px;flex-shrink:0}
    .gh-drawer-title{font-size:14px;font-weight:700;flex:1}
    .gh-drawer-close{background:transparent;border:none;color:var(--muted);font-size:18px;cursor:pointer;padding:2px 6px;border-radius:4px;transition:color .15s}
    .gh-drawer-close:hover{color:var(--text)}
    .gh-drawer-body{flex:1;overflow-y:auto;padding:12px}
    .repo-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);padding:11px 13px;margin-bottom:8px;transition:border-color .15s;cursor:pointer}
    .repo-card:hover{border-color:var(--blue)}
    .repo-name{font-size:12.5px;font-weight:700;color:var(--blue);margin-bottom:3px}
    .repo-desc{font-size:11px;color:var(--muted);margin-bottom:7px;line-height:1.4}
    .repo-meta{display:flex;gap:10px;font-size:10.5px;color:var(--text2)}
    .repo-meta span{display:flex;align-items:center;gap:3px}
    .lang-dot{width:8px;height:8px;border-radius:50%;background:var(--blue);flex-shrink:0}

    /* ── KEYBOARD SHORTCUT HINT ── */
    .kbd{display:inline-flex;align-items:center;gap:3px;font-size:9.5px;color:var(--muted);background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:1px 5px;font-family:'JetBrains Mono',monospace}

    /* ── API EXPLORER DRAWER ── */
    .ep-drawer-backdrop{position:fixed;inset:0;background:rgba(4,6,15,.6);backdrop-filter:blur(4px);z-index:200;display:none;opacity:0;transition:opacity .25s}
    .ep-drawer-backdrop.open{display:block;opacity:1}
    .ep-drawer{position:fixed;top:0;right:-520px;width:480px;max-width:100vw;height:100vh;background:var(--bg2);border-left:1px solid var(--border2);z-index:201;display:flex;flex-direction:column;transition:right .3s cubic-bezier(.4,0,.2,1);box-shadow:-12px 0 48px rgba(0,0,0,.5)}
    .ep-drawer.open{right:0}
    .ep-drawer-hdr{display:flex;align-items:center;gap:10px;padding:16px 18px;border-bottom:1px solid var(--border);flex-shrink:0;background:var(--surface)}
    .ep-drawer-method{font-size:10px;font-weight:800;padding:3px 9px;border-radius:5px;letter-spacing:.06em;flex-shrink:0}
    .ep-drawer-path{font-family:'JetBrains Mono',monospace;font-size:13px;color:var(--text);font-weight:600;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .ep-drawer-close{background:transparent;border:none;color:var(--muted);font-size:20px;cursor:pointer;padding:2px 6px;border-radius:4px;line-height:1;transition:color .15s;flex-shrink:0}
    .ep-drawer-close:hover{color:var(--text)}
    .ep-drawer-body{flex:1;overflow-y:auto;display:flex;flex-direction:column;min-height:0}
    .ep-section{padding:16px 18px;border-bottom:1px solid var(--border)}
    .ep-section-title{font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:.14em;color:var(--muted);margin-bottom:12px;display:flex;align-items:center;gap:6px}
    .ep-section-title::after{content:'';flex:1;height:1px;background:var(--border)}
    .ep-desc-text{font-size:12.5px;color:var(--text2);line-height:1.6;margin-bottom:10px}
    .ep-auth-badge{display:inline-flex;align-items:center;gap:5px;font-size:10px;font-weight:700;padding:3px 9px;border-radius:20px;text-transform:uppercase;letter-spacing:.06em}
    .ep-auth-none{background:rgba(52,211,153,.1);color:var(--green);border:1px solid rgba(52,211,153,.3)}
    .ep-auth-viewer{background:rgba(34,211,238,.1);color:var(--cyan);border:1px solid rgba(34,211,238,.3)}
    .ep-auth-developer{background:rgba(79,142,247,.1);color:var(--blue);border:1px solid rgba(79,142,247,.3)}
    .ep-auth-admin{background:rgba(167,139,250,.1);color:var(--purple);border:1px solid rgba(167,139,250,.3)}
    .ep-field{display:flex;flex-direction:column;gap:5px;margin-bottom:12px}
    .ep-field-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);display:flex;align-items:center;gap:5px}
    .ep-field-label .req{color:var(--red);font-size:11px}
    .ep-field-type{font-size:9px;color:var(--muted);background:var(--surface3);padding:1px 6px;border-radius:10px;font-weight:600;margin-left:auto}
    .ep-input{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:7px 11px;font-size:12.5px;color:var(--text);outline:none;font-family:inherit;transition:border-color .15s,box-shadow .15s;width:100%;box-sizing:border-box}
    .ep-input:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(79,142,247,.1)}
    .ep-input::placeholder{color:var(--muted)}
    .ep-select{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:7px 11px;font-size:12.5px;color:var(--text);outline:none;font-family:inherit;transition:border-color .15s;width:100%;box-sizing:border-box;cursor:pointer}
    .ep-select:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(79,142,247,.1)}
    .ep-textarea{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:7px 11px;font-size:12px;font-family:'JetBrains Mono',monospace;color:var(--text);outline:none;resize:vertical;min-height:72px;line-height:1.55;transition:border-color .15s;width:100%;box-sizing:border-box}
    .ep-textarea:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(79,142,247,.1)}
    .ep-toggle-wrap{display:flex;align-items:center;gap:8px}
    .ep-toggle{width:36px;height:20px;background:var(--border2);border-radius:20px;cursor:pointer;position:relative;transition:background .2s;flex-shrink:0;border:none}
    .ep-toggle.on{background:var(--blue2)}
    .ep-toggle::after{content:'';position:absolute;top:3px;left:3px;width:14px;height:14px;border-radius:50%;background:#fff;transition:left .2s}
    .ep-toggle.on::after{left:19px}
    .ep-toggle-label{font-size:12px;color:var(--text2)}
    .ep-resp-wrap{display:none;flex-direction:column;gap:8px}
    .ep-resp-wrap.visible{display:flex}
    .ep-status-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
    .ep-status-badge{font-weight:700;font-family:'JetBrains Mono',monospace;padding:2px 9px;border-radius:4px;font-size:11px}
    .ep-status-2xx{background:rgba(52,211,153,.12);color:var(--green);border:1px solid rgba(52,211,153,.3)}
    .ep-status-3xx{background:rgba(251,191,36,.12);color:var(--amber);border:1px solid rgba(251,191,36,.3)}
    .ep-status-4xx,.ep-status-5xx{background:rgba(248,113,113,.12);color:var(--red);border:1px solid rgba(248,113,113,.3)}
    .ep-timing{font-size:10px;color:var(--muted);font-family:'JetBrains Mono',monospace}
    .ep-resp-pre{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:12px 14px;font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--text2);max-height:340px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;line-height:1.6}
    .ep-resp-pre.ok{border-color:rgba(52,211,153,.25);color:var(--text)}
    .ep-resp-pre.err{border-color:rgba(248,113,113,.25)}
    .ep-drawer-ftr{display:flex;align-items:center;gap:8px;padding:12px 18px;border-top:1px solid var(--border);flex-shrink:0;background:var(--surface)}
    .ep-exec-btn{padding:8px 22px;background:linear-gradient(135deg,#2563eb,#1d4ed8);border:none;border-radius:var(--radius-sm);font-size:12px;font-weight:700;color:#fff;cursor:pointer;font-family:inherit;transition:all .15s;box-shadow:0 0 14px rgba(37,99,235,.25)}
    .ep-exec-btn:hover{box-shadow:0 0 22px rgba(79,142,247,.45)}
    .ep-exec-btn:disabled{background:var(--surface2);box-shadow:none;color:var(--muted);cursor:not-allowed}
    .ep-outline-btn{padding:7px 14px;background:transparent;border:1px solid var(--border2);border-radius:var(--radius-sm);font-size:12px;font-weight:600;color:var(--text2);cursor:pointer;font-family:inherit;transition:all .15s}
    .ep-outline-btn:hover{border-color:var(--blue);color:var(--blue)}
    /* legacy — keep for WS terminal */
    .api-field-label{font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:4px}
    .api-body-editor{width:100%;background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:10px 12px;font-size:12px;font-family:'JetBrains Mono',monospace;color:var(--text);outline:none;resize:vertical;min-height:110px;line-height:1.6;transition:border-color .15s;box-sizing:border-box}
    .api-body-editor:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(79,142,247,.1)}
    .api-resp{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:12px;font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--text2);max-height:280px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;line-height:1.6}
    .api-resp.ok{border-color:rgba(52,211,153,.3);color:var(--text)}
    .api-resp.err{border-color:rgba(248,113,113,.3);color:var(--red)}
    .api-status-bar{display:flex;align-items:center;gap:8px;font-size:11px}
    .api-status-code{font-weight:700;font-family:'JetBrains Mono',monospace;padding:2px 8px;border-radius:4px}
    .api-status-code.ok{background:rgba(52,211,153,.12);color:var(--green)}
    .api-status-code.err{background:rgba(248,113,113,.12);color:var(--red)}

    /* ── WS TERMINAL ── */
    .ws-terminal{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:10px 12px;font-family:'JetBrains Mono',monospace;font-size:11.5px;height:200px;overflow-y:auto;line-height:1.7}
    .ws-line{margin:0;padding:0}
    .ws-line.sent{color:var(--blue)}
    .ws-line.recv{color:var(--green)}
    .ws-line.sys{color:var(--muted)}
    .ws-line.err{color:var(--red)}
    .ws-input-row{display:flex;gap:8px;margin-top:8px}
    .ws-input{flex:1;background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:6px 10px;font-size:12px;font-family:'JetBrains Mono',monospace;color:var(--text);outline:none;transition:border-color .15s}
    .ws-input:focus{border-color:var(--blue)}
    .ws-send-btn{padding:6px 14px;background:var(--blue2);border:none;border-radius:var(--radius-sm);font-size:11.5px;font-weight:700;color:#fff;cursor:pointer;font-family:inherit}

    /* ── RUN PIPELINE BUTTON ── */
    .run-btn{display:flex;align-items:center;gap:6px;padding:5px 14px;background:linear-gradient(135deg,rgba(37,99,235,.2),rgba(124,58,237,.15));border:1px solid rgba(79,142,247,.35);border-radius:20px;font-size:11.5px;font-weight:700;color:var(--blue);cursor:pointer;transition:all .15s;font-family:inherit}
    .run-btn:hover{background:linear-gradient(135deg,rgba(37,99,235,.35),rgba(124,58,237,.25));border-color:var(--blue);box-shadow:0 0 14px rgba(79,142,247,.2)}

    /* ── ROLE BADGE ── */
    .role-badge{display:inline-flex;align-items:center;gap:4px;font-size:9.5px;font-weight:700;padding:2px 8px;border-radius:20px;text-transform:uppercase;letter-spacing:.06em}
    .role-admin{background:rgba(167,139,250,.15);color:var(--purple);border:1px solid rgba(167,139,250,.35)}
    .role-developer{background:rgba(79,142,247,.15);color:var(--blue);border:1px solid rgba(79,142,247,.35)}
    .role-viewer{background:rgba(52,211,153,.12);color:var(--green);border:1px solid rgba(52,211,153,.3)}

    /* ── RBAC VISIBILITY ── */
    /* viewer: hide write ops and admin panels */
    body[data-role="viewer"] .rbac-dev,body[data-role="viewer"] .rbac-admin{display:none!important}
    /* developer: hide admin-only */
    body[data-role="developer"] .rbac-admin{display:none!important}
    /* before login: hide role-gated items */
    body:not([data-role]) .rbac-dev,body:not([data-role]) .rbac-admin{display:none!important}
    /* generic hidden util */
    .rbac-dev,.rbac-admin{transition:opacity .2s}
  </style>
</head>
<body>

<nav class="sidebar">
  <div class="sb-logo">
    <div class="logo-mark">&#x26A1;</div>
    <div class="logo-text">
      <div class="name">DevOps AI</div>
      <div class="tag">Autonomous Platform</div>
    </div>
  </div>
  <div class="sb-status">
    <div class="status-row"><div class="pulse"></div><span style="color:var(--green);font-weight:600">System Online</span></div>
  </div>

  <div class="sb-section">
    <span class="sb-label">Navigation</span>
    <button type="button" class="nav-link active" onclick="showView('endpoints','all',this)"><span class="ico">&#x26A1;</span>All Endpoints<span class="cnt" id="cnt-all">0</span></button>
    <button type="button" class="nav-link" onclick="showView('endpoints','general',this)"><span class="ico">&#x1F527;</span>General &amp; AI<span class="cnt">13</span></button>
    <button type="button" class="nav-link" onclick="showView('endpoints','k8s',this)"><span class="ico">&#x2638;</span>Kubernetes<span class="cnt">7</span></button>
    <button type="button" class="nav-link" onclick="showView('endpoints','aws',this)"><span class="ico">&#x2601;</span>AWS<span class="cnt">23</span></button>
    <button type="button" class="nav-link" onclick="showView('endpoints','pipeline',this)"><span class="ico">&#x1F916;</span>Pipeline<span class="cnt">1</span></button>
    <button type="button" class="nav-link" onclick="showView('endpoints','deploy',this)"><span class="ico">&#x1F680;</span>Deploy &amp; Jira<span class="cnt">3</span></button>
    <button type="button" class="nav-link rbac-admin" onclick="showView('endpoints','webhooks',this)"><span class="ico">&#x1F517;</span>Webhooks<span class="cnt">2</span></button>
  </div>

  <div class="sb-divider"></div>

  <div class="sb-section">
    <span class="sb-label">Tools</span>
    <button type="button" class="nav-link rbac-admin" onclick="showView('secrets','',this)"><span class="ico">&#x1F511;</span>Secrets &amp; Config</button>
    <button type="button" class="nav-link rbac-admin" onclick="showView('users','',this)"><span class="ico">&#x1F465;</span>Team &amp; Access</button>
    <button type="button" class="nav-link" onclick="showView('chat','',this)"><span class="ico">&#x1F4AC;</span>AI Chat</button>
  </div>

  <div class="sb-section">
    <span class="sb-label">Docs</span>
    <a class="nav-link" href="/docs" target="_blank"><span class="ico">&#x1F4D6;</span>Swagger UI</a>
    <a class="nav-link" href="/redoc" target="_blank"><span class="ico">&#x1F4C4;</span>ReDoc</a>
    <a class="nav-link" href="/health/full" target="_blank"><span class="ico">&#x1F49A;</span>Health Check</a>
  </div>

  <div class="sb-footer">
    <div class="sb-user">
      <div class="avatar" id="sb-avatar">N</div>
      <div class="user-info">
        <div class="uname" id="sb-uname">Nagaraj</div>
        <div class="urole"><span id="sb-role-badge" class="role-badge role-admin">admin</span></div>
      </div>
      <div class="live-dot" style="cursor:pointer;position:relative" title="Switch user" onclick="toggleUserSwitch()">
        <span style="font-size:11px;color:var(--muted)">&#x21C5;</span>
      </div>
    </div>
    <div id="user-switch-panel" style="display:none;margin-top:10px;padding-top:10px;border-top:1px solid var(--border)">
      <div style="font-size:9.5px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px">Switch user</div>
      <div style="display:flex;gap:6px">
        <input id="sb-user-input" style="flex:1;background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:5px 8px;font-size:11.5px;color:var(--text);outline:none;font-family:inherit" placeholder="username" value="nagaraj"/>
        <button onclick="applyUser()" style="padding:5px 10px;background:var(--blue2);border:none;border-radius:var(--radius-sm);font-size:11px;font-weight:700;color:#fff;cursor:pointer;font-family:inherit">Go</button>
      </div>
    </div>
  </div>
</nav>

<div class="main">

  <!-- ── CHAT PANEL ── -->
  <div id="view-chat" style="display:none;flex-direction:column;height:100vh">
    <div class="chat-topbar">
      <div style="width:32px;height:32px;background:linear-gradient(135deg,#2563eb,#7c3aed);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0">&#x1F916;</div>
      <div>
        <div class="chat-topbar-title">DevOps AI Assistant</div>
        <div class="chat-topbar-sub">Live context &#x2022; AWS &#x2022; K8s &#x2022; GitHub &#x2022; Grafana</div>
      </div>
      <div class="tb-spacer"></div>
      <div style="display:flex;align-items:center;gap:6px;margin-right:8px">
        <span style="font-size:10px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.08em">Model</span>
        <select id="llm-selector" class="llm-select" title="Select LLM provider">
          <option value="">&#x1F504; Auto</option>
          <option value="anthropic">&#x2728; Claude</option>
          <option value="groq">&#x26A1; Groq / Llama</option>
          <option value="ollama">&#x1F3E0; Ollama (Local)</option>
        </select>
      </div>
      <button class="chat-warroom-btn" id="chat-warroom-btn" onclick="createWarRoom()">&#x1F6A8; War Room</button>
    </div>
    <div id="chat-messages" class="chat-messages">
      <div class="chat-empty" id="chat-empty">
        <div class="chat-empty-icon">&#x1F916;</div>
        <div class="chat-empty-title">DevOps AI Assistant</div>
        <div class="chat-empty-sub">Ask questions or give commands. I can restart deployments, scale pods, create issues, page on-call, run pipelines &#x2014; and always use live data, never guesses.</div>
        <div class="chat-suggestions">
          <button class="chat-suggestion" onclick="sendSuggestion(this)">&#x1F4CA; Full infrastructure overview</button>
          <button class="chat-suggestion" onclick="sendSuggestion(this)">&#x1F6A8; Any alerts firing right now?</button>
          <button class="chat-suggestion" onclick="sendSuggestion(this)">&#x2638; Restart the payment-service deployment in production</button>
          <button class="chat-suggestion" onclick="sendSuggestion(this)">&#x1F4C8; Scale the api-gateway to 5 replicas in default namespace</button>
          <button class="chat-suggestion" onclick="sendSuggestion(this)">&#x1F50D; My service is down &#x2014; find root cause</button>
          <button class="chat-suggestion" onclick="sendSuggestion(this)">&#x1F41B; Create a GitHub issue: high CPU usage on production pods</button>
        </div>
      </div>
    </div>
    <div class="chat-input-bar">
      <textarea id="chat-input" placeholder="Describe your issue or ask anything about your infrastructure&#x2026;" rows="1" onkeydown="chatKeydown(event)" oninput="autoResize(this)"></textarea>
      <button class="chat-send-btn" id="chat-send-btn" onclick="sendChat()">&#x27A4; Send</button>
    </div>
  </div>

  <!-- ── MAIN CONTENT ── -->
  <div id="content-wrap">
    <div class="topbar">
      <div class="topbar-title">AI Operations Platform</div>
      <span class="topbar-ver">v2.0</span>
      <div class="topbar-spacer"></div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center" id="int-chips">
        <span class="tb-chip int-chip" id="int-claude" title="LLM Provider"><span class="dot"></span>Claude AI</span>
        <span class="tb-chip int-chip" id="int-aws" title="AWS"><span class="dot"></span>AWS</span>
        <span class="tb-chip int-chip" id="int-k8s" title="Kubernetes"><span class="dot"></span>K8s</span>
        <span class="tb-chip int-chip" id="int-github" title="Click to view repos" onclick="openGhDrawer()" style="cursor:pointer"><span class="dot"></span>GitHub</span>
        <span class="tb-chip int-chip" id="int-grafana" title="Grafana"><span class="dot"></span>Grafana</span>
        <span class="tb-chip int-chip" id="int-slack" title="Slack"><span class="dot"></span>Slack</span>
        <span class="tb-chip int-chip" id="int-jira" title="Jira"><span class="dot"></span>Jira</span>
        <span class="tb-chip int-chip" id="int-opsgenie" title="OpsGenie"><span class="dot"></span>OpsGenie</span>
        <div style="width:1px;height:20px;background:var(--border);margin:0 4px"></div>
        <button class="run-btn rbac-dev" onclick="document.getElementById('run-modal').classList.add('open')" title="Run AI Pipeline (R)">&#x25B6; Run Pipeline</button>
      </div>
    </div>

    <div class="metric-strip" id="metric-strip">
      <div class="mc mc-blue fade-in" onclick="showView('endpoints','all',document.querySelector('.nav-link'))" style="cursor:pointer" title="View all endpoints">
        <div class="mc-label">&#x26A1; Total Endpoints</div>
        <div class="mc-val" id="m-eps">&#x2014;</div>
        <div class="mc-sub">REST API surface</div>
        <div class="mc-bar"></div>
      </div>
      <div class="mc mc-green fade-in" title="Active LLM provider">
        <div class="mc-label">&#x1F9E0; LLM Provider</div>
        <div class="mc-val" id="m-llm" style="font-size:13px;padding-top:4px">&#x2014;</div>
        <div class="mc-sub" id="m-llm-sub">Active model</div>
        <div class="mc-bar"></div>
      </div>
      <div class="mc mc-purple fade-in" title="Incidents stored in vector memory">
        <div class="mc-label">&#x1F4BE; Incident Memory</div>
        <div class="mc-val" id="m-mem">&#x2014;</div>
        <div class="mc-sub">Stored in ChromaDB</div>
        <div class="mc-bar"></div>
      </div>
      <div class="mc mc-cyan fade-in" title="Configured integrations">
        <div class="mc-label">&#x1F517; Integrations Live</div>
        <div class="mc-val" id="m-ints">&#x2014;</div>
        <div class="mc-sub" id="m-ints-sub">of 8 services</div>
        <div class="mc-bar"></div>
      </div>
      <div class="mc mc-amber fade-in" title="Pipeline policy guardrails">
        <div class="mc-label">&#x1F6E1; Policy Guardrails</div>
        <div class="mc-val">4</div>
        <div class="mc-sub">Blocked actions</div>
        <div class="mc-bar"></div>
      </div>
      <div class="mc mc-amber fade-in" title="Your current access role">
        <div class="mc-label">&#x1F512; Access Role</div>
        <div class="mc-val" id="m-role" style="font-size:15px;padding-top:4px">&#x2014;</div>
        <div class="mc-sub" id="m-role-perms">Checking...</div>
        <div class="mc-bar"></div>
      </div>
      <div class="mc mc-green fade-in" title="AI actions executed this session">
        <div class="mc-label">&#x1F916; AI Actions</div>
        <div class="mc-val" id="m-actions" style="font-size:22px">0</div>
        <div class="mc-sub">Executed this session</div>
        <div class="mc-bar"></div>
      </div>
    </div>

    <div id="view-endpoints">
      <div class="view-header">
        <div>
          <h2>API Endpoints</h2>
          <p>Complete REST API surface — click any endpoint to copy path</p>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          <div class="search-bar"><span style="color:var(--muted);font-size:13px">&#x1F50D;</span><input type="text" id="ep-search" placeholder="Search endpoints&#x2026;" oninput="filterEps(this.value)"/></div>
          <span class="kbd" title="Press / to focus search">/</span>
        </div>
      </div>

      <div data-section="general">
        <div class="ep-group">
          <div class="ep-group-hdr"><span class="ico">&#x1F527;</span><span class="g-name">General &amp; AI</span><span class="g-cnt">13</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/</span><span class="ep-desc">Dashboard UI</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/health</span><span class="ep-desc">Basic health check</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/health/full</span><span class="ep-desc">Full integration health</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/health/integrations</span><span class="ep-desc">Integration diagnostics</span></div>
          <div class="ep-row" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/chat</span><span class="ep-desc">Conversational AI (LLM + live context)</span></div>
          <div class="ep-row" data-ep data-method="WS" onclick="epClick(this)"><span class="badge" style="background:rgba(167,139,250,.12);color:var(--purple);border:1px solid rgba(167,139,250,.25)">WS</span><span class="ep-path">/ws</span><span class="ep-desc">WebSocket real-time events</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/secrets/status</span><span class="ep-desc">Integration key status</span></div>
          <div class="ep-row rbac-admin" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/secrets</span><span class="ep-desc">Save/update credentials</span><span class="ep-lock">&#x1F512;</span></div>
          <div class="ep-row rbac-admin" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/security/roles</span><span class="ep-desc">List all RBAC roles</span><span class="ep-lock">&#x1F512;</span></div>
          <div class="ep-row rbac-admin" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/security/roles/assign</span><span class="ep-desc">Assign role to user</span><span class="ep-lock">&#x1F512;</span></div>
          <div class="ep-row" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/warroom/create</span><span class="ep-desc">AI war room + Slack channel</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/grafana/alerts</span><span class="ep-desc">Firing Grafana alerts</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/grafana/dashboards</span><span class="ep-desc">Grafana datasources</span></div>
        </div>
      </div>

      <div data-section="pipeline" class="rbac-dev">
        <div class="ep-group">
          <div class="ep-group-hdr"><span class="ico">&#x1F916;</span><span class="g-name">AI Response Engine</span><span class="g-cnt">4</span></div>
          <div class="ep-row" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/incidents/run</span><span class="ep-desc">Run full autonomous pipeline</span></div>
          <div class="ep-row" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/incidents/run/async</span><span class="ep-desc">Async pipeline — returns job ID immediately</span></div>
          <div class="ep-row" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/v2/incident/run</span><span class="ep-desc">Multi-agent pipeline (extended)</span></div>
        </div>
      </div>

      <div data-section="webhooks" class="rbac-admin">
        <div class="ep-group">
          <div class="ep-group-hdr"><span class="ico">&#x1F517;</span><span class="g-name">Webhooks (Event-Driven)</span><span class="g-cnt">2</span></div>
          <div class="ep-row" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/webhooks/github</span><span class="ep-desc">GitHub push / PR events</span></div>
          <div class="ep-row" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/webhooks/pagerduty</span><span class="ep-desc">PagerDuty incident trigger</span></div>
        </div>
      </div>

      <div data-section="k8s">
        <div class="ep-group">
          <div class="ep-group-hdr"><span class="ico">&#x2638;</span><span class="g-name">Kubernetes</span><span class="g-cnt">7</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/k8s/health</span><span class="ep-desc">Cluster health overview</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/k8s/pods</span><span class="ep-desc">List pods with status</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/k8s/deployments</span><span class="ep-desc">Deployment readiness</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/k8s/logs</span><span class="ep-desc">Pod logs (query: namespace, pod)</span></div>
          <div class="ep-row rbac-dev" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/k8s/restart</span><span class="ep-desc">Rolling restart deployment</span><span class="ep-lock">&#x1F512;</span></div>
          <div class="ep-row rbac-dev" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/k8s/scale</span><span class="ep-desc">Scale deployment replicas</span><span class="ep-lock">&#x1F512;</span></div>
          <div class="ep-row" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/k8s/diagnose</span><span class="ep-desc">AI K8s diagnosis</span></div>
        </div>
      </div>

      <div data-section="aws">
        <div class="ep-group">
          <div class="ep-group-hdr"><span class="ico">&#x2601;</span><span class="g-name">AWS</span><span class="g-cnt">23</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/ec2/instances</span><span class="ep-desc">EC2 instances</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/ecs/services</span><span class="ep-desc">ECS services</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/lambda/functions</span><span class="ep-desc">Lambda functions</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/cloudwatch/alarms</span><span class="ep-desc">CloudWatch alarms</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/cloudwatch/logs</span><span class="ep-desc">Log groups &amp; streams</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/rds/instances</span><span class="ep-desc">RDS instances</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/s3/buckets</span><span class="ep-desc">S3 buckets</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/sqs/queues</span><span class="ep-desc">SQS queues</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/dynamodb/tables</span><span class="ep-desc">DynamoDB tables</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/cloudtrail/events</span><span class="ep-desc">CloudTrail events</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/route53/health</span><span class="ep-desc">Route53 health checks</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/sns/topics</span><span class="ep-desc">SNS topics</span></div>
          <div class="ep-row" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/aws/diagnose</span><span class="ep-desc">AI AWS root cause analysis</span></div>
          <div class="ep-row" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/aws/predict-scaling</span><span class="ep-desc">AI scaling prediction</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/ec2/console</span><span class="ep-desc">EC2 console output (query: instance_id)</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/ecs/stopped-tasks</span><span class="ep-desc">Stopped ECS tasks</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/lambda/errors</span><span class="ep-desc">Lambda error stats (query: function_name)</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/rds/events</span><span class="ep-desc">RDS events (query: db_instance_id)</span></div>
          <div class="ep-row" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/aws/cloudwatch/metrics</span><span class="ep-desc">CloudWatch metrics query</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/cost/summary</span><span class="ep-desc">Resource inventory &amp; cost overview</span></div>
          <div class="ep-row" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/aws/assess-deployment</span><span class="ep-desc">Pre-deploy risk gate</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/context</span><span class="ep-desc">Full AWS context snapshot</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/synthesize</span><span class="ep-desc">AI incident synthesis</span></div>
        </div>
      </div>

      <div data-section="deploy">
        <div class="ep-group">
          <div class="ep-group-hdr"><span class="ico">&#x1F680;</span><span class="g-name">Deploy, GitHub &amp; Jira</span><span class="g-cnt">7</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/github/commits</span><span class="ep-desc">Recent commits</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/github/prs</span><span class="ep-desc">Recent pull requests</span></div>
          <div class="ep-row" data-ep data-method="GET" onclick="epClick(this)"><span class="badge GET">GET</span><span class="ep-path">/github/profile</span><span class="ep-desc">GitHub account summary</span></div>
          <div class="ep-row" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/github/pr/{n}/review</span><span class="ep-desc">AI PR code review</span></div>
          <div class="ep-row" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/github/issue</span><span class="ep-desc">Create GitHub issue</span></div>
          <div class="ep-row" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/jira/incident</span><span class="ep-desc">Create Jira ticket</span></div>
          <div class="ep-row" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/deploy/assess</span><span class="ep-desc">Pre-deploy AI risk gate</span></div>
          <div class="ep-row" data-ep data-method="POST" onclick="epClick(this)"><span class="badge POST">POST</span><span class="ep-path">/deploy/jira-to-pr</span><span class="ep-desc">Jira ticket &#x2192; GitHub PR plan</span></div>
        </div>
      </div>

      <div data-section="pipeline">
        <div class="ep-group" style="border-color:rgba(79,142,247,.25)">
          <div class="pipeline-hdr"><span class="ico" style="font-size:16px">&#x1F916;</span><span class="g-name" style="color:#93c5fd">Autonomous Response Flow</span><span class="pipeline-badge">7-Stage</span></div>
          <div class="flow-row">
            <div class="fstep"><span class="fstep-ic">&#x1F4E1;</span><span class="fstep-lb">Collect</span><span class="fstep-sb">AWS/K8s/Git</span></div>
            <span class="farr">&#x279C;</span>
            <div class="fstep"><span class="fstep-ic">&#x1F9E0;</span><span class="fstep-lb">Plan</span><span class="fstep-sb">LLM actions</span></div>
            <span class="farr">&#x279C;</span>
            <div class="fstep"><span class="fstep-ic">&#x2696;</span><span class="fstep-lb">Decide</span><span class="fstep-sb">Risk gate</span></div>
            <span class="farr">&#x279C;</span>
            <div class="fstep"><span class="fstep-ic">&#x1F6E1;</span><span class="fstep-lb">Policy</span><span class="fstep-sb">RBAC check</span></div>
            <span class="farr">&#x279C;</span>
            <div class="fstep"><span class="fstep-ic">&#x26A1;</span><span class="fstep-lb">Execute</span><span class="fstep-sb">6 handlers</span></div>
            <span class="farr">&#x279C;</span>
            <div class="fstep"><span class="fstep-ic">&#x1F4CA;</span><span class="fstep-lb">Validate</span><span class="fstep-sb">Health check</span></div>
            <span class="farr">&#x279C;</span>
            <div class="fstep"><span class="fstep-ic">&#x1F4BE;</span><span class="fstep-lb">Memory</span><span class="fstep-sb">ChromaDB</span></div>
          </div>
          <pre class="sample">POST /incidents/run
{
  "incident_id": "INC-2024-001",
  "description": "Payment service pods crash-looping in production",
  "severity": "critical",
  "auto_remediate": false,
  "k8s": {"namespace": "production"}
}</pre>
        </div>
      </div>

    </div><!-- /view-endpoints -->

    <div id="view-secrets" class="secrets-panel">
      <div class="sec-actions-bar">
        <div class="sec-user-wrap">
          <span class="sec-user-lbl">Authenticated as</span>
          <input class="sec-user-input" id="sec-user" type="text" placeholder="username" readonly/>
        </div>
        <button id="save-btn" class="save-btn" onclick="saveSecrets()">&#x1F4BE; Save to .env</button>
        <span id="sec-msg" class="sec-msg"></span>
      </div>

      <div class="sec-card">
        <div class="sec-card-hdr"><span class="ico">&#x1F916;</span><span class="g-name">LLM Providers</span><span class="int-chip" id="int-claude-s"><span class="dot"></span>Claude AI</span></div>
        <div class="sec-row"><span class="sec-key">ANTHROPIC_API_KEY</span><input class="sec-input" id="ANTHROPIC_API_KEY" type="password" placeholder="sk-ant-api03-&#x2026;"/><span class="sec-status" id="st-ANTHROPIC_API_KEY"></span></div>
        <div class="sec-row"><span class="sec-key">GROQ_API_KEY</span><input class="sec-input" id="GROQ_API_KEY" type="password" placeholder="gsk_&#x2026;"/><span class="sec-status" id="st-GROQ_API_KEY"></span></div>
      </div>

      <div class="sec-card">
        <div class="sec-card-hdr"><span class="ico">&#x2601;</span><span class="g-name">AWS</span><span class="int-chip" id="int-aws-s"><span class="dot"></span>AWS</span></div>
        <div class="sec-row"><span class="sec-key">AWS_ACCESS_KEY_ID</span><input class="sec-input" id="AWS_ACCESS_KEY_ID" type="password" placeholder="AKIA&#x2026;"/><span class="sec-status" id="st-AWS_ACCESS_KEY_ID"></span></div>
        <div class="sec-row"><span class="sec-key">AWS_SECRET_ACCESS_KEY</span><input class="sec-input" id="AWS_SECRET_ACCESS_KEY" type="password" placeholder="Secret key&#x2026;"/><span class="sec-status" id="st-AWS_SECRET_ACCESS_KEY"></span></div>
        <div class="sec-row"><span class="sec-key">AWS_REGION</span><input class="sec-input" id="AWS_REGION" type="text" placeholder="us-east-1"/><span class="sec-status" id="st-AWS_REGION"></span></div>
      </div>

      <div class="sec-card">
        <div class="sec-card-hdr"><span class="ico">&#x1F419;</span><span class="g-name">GitHub</span><span class="int-chip" id="int-github-s"><span class="dot"></span>GitHub</span></div>
        <div class="sec-row"><span class="sec-key">GITHUB_TOKEN</span><input class="sec-input" id="GITHUB_TOKEN" type="password" placeholder="ghp_&#x2026;"/><span class="sec-status" id="st-GITHUB_TOKEN"></span></div>
        <div class="sec-row"><span class="sec-key">GITHUB_REPO</span><input class="sec-input" id="GITHUB_REPO" type="text" placeholder="owner/repo-name"/><span class="sec-status" id="st-GITHUB_REPO"></span></div>
      </div>

      <div class="sec-card">
        <div class="sec-card-hdr"><span class="ico">&#x1F4CA;</span><span class="g-name">Grafana</span><span class="int-chip" id="int-grafana-s"><span class="dot"></span>Grafana</span></div>
        <div class="sec-row"><span class="sec-key">GRAFANA_URL</span><input class="sec-input" id="GRAFANA_URL" type="text" placeholder="http://localhost:3000"/><span class="sec-status" id="st-GRAFANA_URL"></span></div>
        <div class="sec-row"><span class="sec-key">GRAFANA_TOKEN</span><input class="sec-input" id="GRAFANA_TOKEN" type="password" placeholder="Service account token"/><span class="sec-status" id="st-GRAFANA_TOKEN"></span></div>
      </div>

      <div class="sec-card">
        <div class="sec-card-hdr"><span class="ico">&#x1F4AC;</span><span class="g-name">Slack</span><span class="int-chip" id="int-slack-s"><span class="dot"></span>Slack</span></div>
        <div class="sec-row"><span class="sec-key">SLACK_BOT_TOKEN</span><input class="sec-input" id="SLACK_BOT_TOKEN" type="password" placeholder="xoxb-&#x2026;"/><span class="sec-status" id="st-SLACK_BOT_TOKEN"></span></div>
        <div class="sec-row"><span class="sec-key">SLACK_CHANNEL</span><input class="sec-input" id="SLACK_CHANNEL" type="text" placeholder="#incidents"/><span class="sec-status" id="st-SLACK_CHANNEL"></span></div>
      </div>

      <div class="sec-card">
        <div class="sec-card-hdr"><span class="ico">&#x1F4CB;</span><span class="g-name">Jira &amp; OpsGenie</span><span class="int-chip" id="int-jira-s"><span class="dot"></span>Jira</span></div>
        <div class="sec-row"><span class="sec-key">JIRA_URL</span><input class="sec-input" id="JIRA_URL" type="text" placeholder="https://company.atlassian.net"/><span class="sec-status" id="st-JIRA_URL"></span></div>
        <div class="sec-row"><span class="sec-key">JIRA_USER</span><input class="sec-input" id="JIRA_USER" type="text" placeholder="you@company.com"/><span class="sec-status" id="st-JIRA_USER"></span></div>
        <div class="sec-row"><span class="sec-key">JIRA_TOKEN</span><input class="sec-input" id="JIRA_TOKEN" type="password" placeholder="Jira API token"/><span class="sec-status" id="st-JIRA_TOKEN"></span></div>
        <div class="sec-row"><span class="sec-key">OPSGENIE_API_KEY</span><input class="sec-input" id="OPSGENIE_API_KEY" type="password" placeholder="OpsGenie key"/><span class="sec-status" id="st-OPSGENIE_API_KEY"></span></div>
      </div>

      <div class="sec-card">
        <div class="sec-card-hdr"><span class="ico">&#x2638;</span><span class="g-name">Kubernetes &amp; GitLab</span></div>
        <div class="sec-row"><span class="sec-key">GITLAB_URL</span><input class="sec-input" id="GITLAB_URL" type="text" placeholder="https://gitlab.com"/><span class="sec-status" id="st-GITLAB_URL"></span></div>
        <div class="sec-row"><span class="sec-key">GITLAB_TOKEN</span><input class="sec-input" id="GITLAB_TOKEN" type="password" placeholder="GitLab token"/><span class="sec-status" id="st-GITLAB_TOKEN"></span></div>
        <div class="sec-row"><span class="sec-key">GITLAB_PROJECT</span><input class="sec-input" id="GITLAB_PROJECT" type="text" placeholder="namespace/project"/><span class="sec-status" id="st-GITLAB_PROJECT"></span></div>
      </div>

      <!-- ── EMAIL / SMTP CONFIG ── -->
      <div class="sec-card" style="border:1px solid rgba(79,142,247,.25);background:rgba(79,142,247,.04)">
        <div class="sec-card-hdr">
          <span class="ico">&#x2709;</span>
          <span class="g-name">Email / SMTP</span>
          <span class="int-chip" id="int-smtp-s"><span class="dot"></span>SMTP</span>
        </div>
        <div class="sec-row"><span class="sec-key">SMTP_USER</span><input class="sec-input" id="smtp_user_inp" type="email" placeholder="you@gmail.com"/></div>
        <div class="sec-row"><span class="sec-key">SMTP_PASSWORD</span><input class="sec-input" id="smtp_pass_inp" type="password" placeholder="16-char Gmail App Password"/></div>
        <div class="sec-row"><span class="sec-key">SMTP_FROM</span><input class="sec-input" id="smtp_from_inp" type="text" placeholder="DevOps AI &lt;you@gmail.com&gt;"/></div>
        <div class="sec-row"><span class="sec-key">APP_URL</span><input class="sec-input" id="app_url_inp" type="text" placeholder="http://localhost:8000"/></div>
        <div style="padding:10px 16px 14px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
          <button onclick="saveSMTP()" style="padding:6px 16px;background:var(--blue2);border:none;border-radius:5px;color:#fff;font-size:.82em;font-weight:700;cursor:pointer">Save &amp; Test Connection</button>
          <button onclick="testEmail()" style="padding:6px 16px;background:var(--surface2);border:1px solid var(--border);border-radius:5px;color:var(--text);font-size:.82em;cursor:pointer">Send Test Email to Myself</button>
          <span id="smtp-msg" style="font-size:.8em;display:none"></span>
        </div>
        <div style="padding:0 16px 12px;font-size:.76em;color:var(--muted);line-height:1.6">
          Gmail: enable 2FA &#x2192; <a href="https://myaccount.google.com/apppasswords" target="_blank" style="color:var(--blue)">myaccount.google.com/apppasswords</a> &#x2192; create App Password for "Mail".
        </div>
      </div>

    </div><!-- /view-secrets -->

    <!-- ── TEAM & ACCESS PANEL ── -->
    <div id="view-users" style="display:none;padding:24px;max-width:900px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">
        <div>
          <h2 style="font-size:1.2em;font-weight:700;margin-bottom:4px">&#x1F465; Team &amp; Access</h2>
          <p style="font-size:.82em;color:var(--muted)">Manage users, roles, and send invites. Admin only.</p>
        </div>
        <button onclick="showInviteModal()" style="padding:8px 18px;background:var(--blue2);border:none;border-radius:var(--radius-sm);color:#fff;font-weight:700;font-size:13px;cursor:pointer">+ Invite User</button>
      </div>

      <!-- Role legend -->
      <div style="display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap">
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;background:var(--surface2);padding:6px 12px;border-radius:20px"><span style="background:rgba(239,68,68,.2);color:#f87171;padding:2px 8px;border-radius:10px;font-size:.78em;font-weight:700">admin</span> Full access</div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;background:var(--surface2);padding:6px 12px;border-radius:20px"><span style="background:rgba(34,211,238,.15);color:#22d3ee;padding:2px 8px;border-radius:10px;font-size:.78em;font-weight:700">developer</span> Read &amp; write</div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;background:var(--surface2);padding:6px 12px;border-radius:20px"><span style="background:rgba(167,139,250,.15);color:#a78bfa;padding:2px 8px;border-radius:10px;font-size:.78em;font-weight:700">viewer</span> Read-only</div>
      </div>

      <!-- User table -->
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius)">
        <div style="padding:14px 18px;border-bottom:1px solid var(--border);font-size:11px;font-weight:700;opacity:.5;display:grid;grid-template-columns:1fr 160px 140px 100px;gap:10px;text-transform:uppercase;letter-spacing:.06em">
          <span>User</span><span>Role</span><span>Created</span><span>Actions</span>
        </div>
        <div id="users-list">
          <div style="padding:24px;text-align:center;opacity:.5;font-size:13px">Loading...</div>
        </div>
      </div>

      <!-- Invite result -->
      <div id="invite-result" style="display:none;margin-top:16px;padding:16px;border-radius:var(--radius);border:1px solid var(--border);background:var(--surface2);font-size:13px"></div>
    </div>

    <!-- Invite Modal -->
    <div id="invite-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:999;align-items:center;justify-content:center">
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:32px;width:100%;max-width:400px">
        <h3 style="margin-bottom:6px;font-size:1.1em;font-weight:700">Invite New User</h3>
        <p style="font-size:.82em;color:var(--muted);margin-bottom:20px">They will receive a one-time OTP by email to set their password.</p>
        <div id="invite-err" style="color:#f87171;font-size:.82em;margin-bottom:10px;display:none"></div>
        <label style="display:block;font-size:.82em;color:var(--muted);margin-bottom:4px">Username</label>
        <input id="inv-username" type="text" placeholder="alice" style="width:100%;box-sizing:border-box;padding:8px 12px;border-radius:6px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.9em;margin-bottom:12px;outline:none"/>
        <label style="display:block;font-size:.82em;color:var(--muted);margin-bottom:4px">Email</label>
        <input id="inv-email" type="email" placeholder="alice@company.com" style="width:100%;box-sizing:border-box;padding:8px 12px;border-radius:6px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.9em;margin-bottom:12px;outline:none"/>
        <label style="display:block;font-size:.82em;color:var(--muted);margin-bottom:4px">Role</label>
        <select id="inv-role" style="width:100%;box-sizing:border-box;padding:8px 12px;border-radius:6px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:.9em;margin-bottom:20px;outline:none">
          <option value="viewer">viewer</option>
          <option value="developer" selected>developer</option>
          <option value="admin">admin</option>
        </select>
        <div style="display:flex;gap:10px">
          <button onclick="closeInviteModal()" style="flex:1;padding:10px;border-radius:6px;border:1px solid var(--border);background:transparent;color:var(--text);font-size:.9em;cursor:pointer">Cancel</button>
          <button id="inv-send-btn" onclick="sendInvite()" style="flex:2;padding:10px;border-radius:6px;border:none;background:var(--blue2);color:#fff;font-weight:700;font-size:.9em;cursor:pointer">Send Invite</button>
        </div>
      </div>
    </div>

    <div class="footer">
      <span>&#x26A1; NexusOps</span>
      <span style="opacity:.3">&#x2022;</span>
      <a href="/docs" target="_blank">Swagger</a>
      <span style="opacity:.3">&#x2022;</span>
      <a href="/redoc" target="_blank">ReDoc</a>
      <span style="opacity:.3">&#x2022;</span>
      <a href="/health/full" target="_blank">Health</a>
      <span style="opacity:.3">&#x2022;</span>
      <span>v2.0.0 &#x2022; Multi-Agent AI &#x2022; Autonomous</span>
    </div>
  </div><!-- /content-wrap -->

</div><!-- /main -->

<!-- ── API EXPLORER DRAWER ── -->
<div class="ep-drawer-backdrop" id="ep-backdrop" onclick="closeApiModal()"></div>
<div class="ep-drawer" id="ep-drawer">
  <div class="ep-drawer-hdr">
    <span class="ep-drawer-method GET" id="ep-method-badge">GET</span>
    <span class="ep-drawer-path" id="ep-path-label">/health</span>
    <button class="ep-drawer-close" onclick="closeApiModal()">&times;</button>
  </div>
  <div class="ep-drawer-body" id="ep-drawer-body">
    <!-- populated by openApiModal() -->
  </div>
  <div class="ep-drawer-ftr">
    <button class="ep-exec-btn" id="ep-exec-btn" onclick="sendApiRequest()">&#x25B6; Execute</button>
    <button class="ep-outline-btn" id="ep-curl-btn" onclick="copyAsCurl()">Copy cURL</button>
    <button class="ep-outline-btn" id="ep-copy-resp-btn" onclick="copyApiResp()" style="display:none">Copy Response</button>
    <span id="api-status-wrap" style="margin-left:auto"></span>
  </div>
</div>

<!-- ── TOAST CONTAINER ── -->
<div id="toast-wrap"></div>

<!-- ── RUN PIPELINE MODAL ── -->
<div class="modal-overlay" id="run-modal" onclick="if(event.target===this)this.classList.remove('open')">
  <div class="modal">
    <div class="modal-title">&#x1F916; Run AI Pipeline</div>
    <div class="modal-sub">Trigger the full autonomous AI incident response engine</div>
    <div class="modal-field">
      <label class="modal-label">Incident ID</label>
      <input class="modal-input" id="m-inc-id" placeholder="INC-2024-001" type="text"/>
    </div>
    <div class="modal-field">
      <label class="modal-label">Description</label>
      <textarea class="modal-input" id="m-inc-desc" placeholder="e.g. Payment service pods crash-looping in production namespace&#x2026;"></textarea>
    </div>
    <div class="modal-row">
      <div class="modal-field">
        <label class="modal-label">Severity</label>
        <div class="sev-pills">
          <button class="sev-pill critical" onclick="setSev(this,'critical')">&#x1F534; Critical</button>
          <button class="sev-pill high active" onclick="setSev(this,'high')">&#x1F7E0; High</button>
          <button class="sev-pill medium" onclick="setSev(this,'medium')">&#x1F7E1; Medium</button>
          <button class="sev-pill low" onclick="setSev(this,'low')">&#x1F7E2; Low</button>
        </div>
      </div>
    </div>
    <div class="modal-field" style="margin-top:8px">
      <label style="display:flex;align-items:center;gap:8px;cursor:pointer;font-size:12px;color:var(--text2)">
        <input type="checkbox" id="m-auto-rem" style="accent-color:var(--blue)"/>
        Auto-remediate (execute K8s actions without approval)
      </label>
    </div>
    <div id="m-result" class="modal-result"></div>
    <div class="modal-actions">
      <button class="btn-cancel" onclick="document.getElementById('run-modal').classList.remove('open')">Cancel</button>
      <button class="btn-run" id="m-run-btn" onclick="runPipeline()">&#x25B6; Run Pipeline</button>
    </div>
  </div>
</div>

<!-- ── GITHUB REPOS DRAWER ── -->
<div class="gh-drawer" id="gh-drawer">
  <div class="gh-drawer-hdr">
    <span style="font-size:16px">&#x1F419;</span>
    <div class="gh-drawer-title">GitHub Repositories</div>
    <button class="gh-drawer-close" onclick="document.getElementById('gh-drawer').classList.remove('open')">&times;</button>
  </div>
  <div class="gh-drawer-body" id="gh-drawer-body">
    <div style="text-align:center;padding:30px;color:var(--muted)">Loading&#x2026;</div>
  </div>
</div>

<script>
var ALL_KEYS=['ANTHROPIC_API_KEY','GROQ_API_KEY','AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','AWS_REGION','GITHUB_TOKEN','GITHUB_REPO','GITLAB_URL','GITLAB_TOKEN','GITLAB_PROJECT','SLACK_BOT_TOKEN','SLACK_CHANNEL','JIRA_URL','JIRA_USER','JIRA_TOKEN','OPSGENIE_API_KEY','GRAFANA_URL','GRAFANA_TOKEN'];
var INT_MAP={'Claude AI':'int-claude','AWS':'int-aws','Grafana':'int-grafana','GitHub':'int-github','GitLab':'int-gitlab','Kubernetes':'int-k8s','Slack':'int-slack','Jira':'int-jira','OpsGenie':'int-opsgenie'};
var SEC_MAP={'Claude AI':'int-claude-s','AWS':'int-aws-s','Grafana':'int-grafana-s','GitHub':'int-github-s','Slack':'int-slack-s','Jira':'int-jira-s'};

function showView(view, section, btn) {
  try { localStorage.setItem('devops_view', view); } catch(e){}
  try {
    document.querySelectorAll('.nav-link').forEach(function(l){ l.classList.remove('active'); });
    if (btn) btn.classList.add('active');
    var cw = document.getElementById('content-wrap');
    var cp = document.getElementById('view-chat');
    if (cp) cp.style.display = view === 'chat' ? 'flex' : 'none';
    if (cw) cw.style.display = view === 'chat' ? 'none' : 'flex';
    if (view !== 'chat') {
      cw.style.flexDirection = 'column';
    }
    var ep = document.getElementById('view-endpoints');
    if (ep) ep.style.display = view === 'endpoints' ? '' : 'none';
    var sp = document.getElementById('view-secrets');
    if (sp) { if (view === 'secrets') { sp.classList.add('active'); } else { sp.classList.remove('active'); } }
    var up = document.getElementById('view-users');
    if (up) { up.style.display = view === 'users' ? 'block' : 'none'; if (view === 'users') loadUsers(); }
    if (view === 'endpoints') filterSection(section || 'all');
  } catch(e) { console.error('showView error:', e); }
}

function filterSection(name) {
  document.querySelectorAll('[data-section]').forEach(function(g) {
    g.style.display = (name === 'all' || g.dataset.section === name) ? '' : 'none';
  });
}

function filterEps(q) {
  q = q.toLowerCase().trim();
  document.querySelectorAll('[data-ep]').forEach(function(ep) {
    ep.classList.toggle('hidden', q !== '' && !ep.textContent.toLowerCase().includes(q));
  });
  document.querySelectorAll('[data-section]').forEach(function(grp) {
    var vis = Array.from(grp.querySelectorAll('[data-ep]')).some(function(e){ return !e.classList.contains('hidden'); });
    grp.style.display = (!q || vis) ? '' : 'none';
  });
}

/* ── API EXPLORER ── */
var _apiWs = null;
var _apiLastResp = '';
var _apiCurrentMethod = 'GET';
var _apiCurrentPath = '';

var _EP_DEFAULTS = {
  '/incidents/run':       '{\\n  "incident_id": "INC-001",\\n  "description": "Payment service pods crash-looping",\\n  "severity": "high",\\n  "auto_remediate": false\\n}',
  '/incidents/run/async': '{\\n  "incident_id": "INC-001",\\n  "description": "Payment service pods crash-looping",\\n  "severity": "high"\\n}',
  '/warroom/create':      '{\\n  "incident_id": "INC-001",\\n  "description": "Database connection failures",\\n  "severity": "critical",\\n  "post_to_slack": false\\n}',
  '/chat':                '{\\n  "message": "What is the current status of my infrastructure?",\\n  "history": [],\\n  "provider": ""\\n}',
  '/k8s/restart':         '{\\n  "namespace": "default",\\n  "deployment": "my-deployment"\\n}',
  '/k8s/scale':           '{\\n  "namespace": "default",\\n  "deployment": "my-deployment",\\n  "replicas": 3\\n}',
  '/k8s/diagnose':        '{\\n  "resource_type": "k8s",\\n  "resource_id": "default"\\n}',
  '/aws/diagnose':        '{\\n  "resource_type": "ec2",\\n  "resource_id": "i-0123456789abcdef0"\\n}',
  '/github/issue':        '?title=Bug+found&body=Description+here',
  '/jira/incident':       '?summary=Service+Down&description=Payment+service+unavailable',
  '/security/roles/assign': '{\\n  "user": "alice",\\n  "role": "developer"\\n}',
  '/webhooks/github':     '{\\n  "action": "push",\\n  "ref": "refs/heads/main",\\n  "commits": [{"message": "fix: payment timeout"}],\\n  "repository": {},\\n  "pull_request": {}\\n}',
  '/ws':                  '{"id":"evt-1","type":"pod_crash","source":"k8s","payload":{"pod":"payment-api-xyz","namespace":"production"}}',
};

var _EP_META = {
  '/health':          { desc:'Basic liveness check — returns status and incident memory count.', auth:'none', params:[] },
  '/health/full':     { desc:'Full integration health with per-service latency probing.', auth:'none', params:[] },
  '/health/live':     { desc:'Kubernetes liveness probe — always returns 200 if process is alive.', auth:'none', params:[] },
  '/health/ready':    { desc:'Kubernetes readiness probe — verifies ChromaDB, modules, and LLM key.', auth:'none', params:[] },
  '/metrics':         { desc:'Prometheus text-format metrics — hook this into your scrape config.', auth:'none', params:[] },
  '/chat':            { desc:'Conversational AI with live infrastructure context (AWS, K8s, GitHub).', auth:'viewer', params:[
    {name:'message',type:'string',required:true,placeholder:'What is the current status of api-gateway?'},
    {name:'history',type:'array',required:false,default:'[]'},
    {name:'provider',type:'select',options:['','anthropic','groq','ollama'],required:false}
  ]},
  '/warroom/create':  { desc:'Create an AI war room: runs multi-source analysis and posts to Slack.', auth:'developer', params:[
    {name:'incident_id',type:'string',required:true,placeholder:'INC-2024-001'},
    {name:'severity',type:'select',options:['critical','high','medium','low'],required:true},
    {name:'description',type:'string',required:true,placeholder:'Payment service pods crash-looping in prod'},
    {name:'post_to_slack',type:'boolean',default:false}
  ]},
  '/incidents/run':   { desc:'Run the full 7-stage autonomous incident response pipeline.', auth:'developer', params:[
    {name:'incident_id',type:'string',required:true,placeholder:'INC-2024-001'},
    {name:'description',type:'string',required:true,placeholder:'Describe the incident'},
    {name:'severity',type:'select',options:['critical','high','medium','low'],required:true},
    {name:'auto_remediate',type:'boolean',default:false}
  ]},
  '/k8s/pods':        { desc:'List all pods across namespaces with phase and readiness status.', auth:'viewer', params:[] },
  '/k8s/deployments': { desc:'Deployment readiness — replica counts and rollout status.', auth:'viewer', params:[] },
  '/k8s/logs':        { desc:'Fetch recent log lines from a pod container.', auth:'viewer', params:[
    {name:'pod',type:'string',required:true,placeholder:'api-gateway-7d9f-xxx'},
    {name:'namespace',type:'string',required:false,default:'default'},
    {name:'lines',type:'number',required:false,default:100}
  ]},
  '/aws/cloudwatch/alarms': { desc:'CloudWatch alarms currently in ALARM state.', auth:'viewer', params:[] },
  '/aws/logs/recent': { desc:'Recent CloudWatch log events from a log group.', auth:'viewer', params:[
    {name:'group',type:'string',required:false,placeholder:'/aws/lambda/my-function'},
    {name:'hours',type:'number',required:false,default:1}
  ]},
  '/grafana/alerts':  { desc:'Active Grafana alert rules that are firing.', auth:'viewer', params:[] },
  '/github/prs':      { desc:'Open pull requests across the configured repository.', auth:'viewer', params:[] },
  '/kb/query':        { desc:'Query the RAG knowledge base for runbook and incident history.', auth:'viewer', params:[
    {name:'query',type:'string',required:true,placeholder:'How to handle OOMKilled pods?'},
    {name:'top_k',type:'number',required:false,default:3}
  ]},
  '/correlate':       { desc:'AI correlation of multiple events to find causal patterns.', auth:'viewer', params:[
    {name:'events',type:'array',required:true,placeholder:'["CPU spike at 14:00","deployment at 13:55"]'}
  ]},
  '/secrets':         { desc:'Save integration credentials to the server .env file.', auth:'admin', params:[] },
  '/users':           { desc:'List all platform users and their roles.', auth:'admin', params:[] },
};

function _getFieldId(name) { return 'ep-field-' + name.replace(/[^a-z0-9]/gi,'-'); }

function _buildFormFields(params) {
  if (!params || !params.length) return '<div style="font-size:12px;color:var(--muted);padding:4px 0">No parameters required for this endpoint.</div>';
  return params.map(function(p) {
    var fid = _getFieldId(p.name);
    var typeTag = '<span class="ep-field-type">' + p.type + '</span>';
    var reqTag  = p.required ? '<span class="req">*</span>' : '';
    var label   = '<div class="ep-field-label">' + p.name + reqTag + typeTag + '</div>';
    var input   = '';
    if (p.type === 'select') {
      input = '<select class="ep-select" id="' + fid + '">' +
        p.options.map(function(o){ return '<option value="' + o + '">' + (o || '(default)') + '</option>'; }).join('') +
        '</select>';
    } else if (p.type === 'boolean') {
      var checked = p.default ? 'on' : '';
      input = '<div class="ep-toggle-wrap">' +
        '<button type="button" class="ep-toggle ' + checked + '" id="' + fid + '" onclick="this.classList.toggle(&apos;on&apos;)" data-val="' + (p.default||false) + '"></button>' +
        '<span class="ep-toggle-label" id="' + fid + '-lbl">' + (p.default ? 'true' : 'false') + '</span></div>';
    } else if (p.type === 'array') {
      input = '<textarea class="ep-textarea" id="' + fid + '" placeholder="' + (p.placeholder||'[]') + '" rows="3">' + (p.default||'') + '</textarea>';
    } else {
      var defVal = (p.default !== undefined && p.default !== false) ? String(p.default) : '';
      input = '<input class="ep-input" id="' + fid + '" type="' + (p.type==='number'?'number':'text') + '" placeholder="' + (p.placeholder||defVal||'') + '" value="' + defVal + '"/>';
    }
    return '<div class="ep-field">' + label + input + '</div>';
  }).join('');
}

function _collectParams(params) {
  var obj = {};
  (params||[]).forEach(function(p) {
    var fid = _getFieldId(p.name);
    var el  = document.getElementById(fid);
    if (!el) return;
    var val;
    if (p.type === 'boolean') {
      val = el.classList.contains('on');
    } else if (p.type === 'number') {
      val = el.value.trim() !== '' ? Number(el.value) : (p.default !== undefined ? p.default : null);
    } else if (p.type === 'array') {
      try { val = JSON.parse(el.value.trim() || '[]'); } catch(_){ val = el.value.trim(); }
    } else {
      val = el.value.trim();
    }
    if (val !== '' && val !== null) obj[p.name] = val;
  });
  return obj;
}

function epClick(row) {
  var pathEl = row.querySelector('.ep-path');
  if (!pathEl) return;
  var method  = row.dataset.method || 'GET';
  var pathStr = pathEl.textContent.trim();
  openApiModal(method, pathStr);
}

function openApiModal(method, pathStr) {
  _apiCurrentMethod = method;
  _apiCurrentPath   = pathStr;
  _apiLastResp      = '';

  // Header
  var badge = document.getElementById('ep-method-badge');
  badge.textContent = method;
  badge.className   = 'ep-drawer-method ' + method;
  document.getElementById('ep-path-label').textContent = pathStr;

  // Reset footer state
  var execBtn = document.getElementById('ep-exec-btn');
  execBtn.disabled  = false;
  execBtn.textContent = '\\u25B6 Execute';
  document.getElementById('ep-copy-resp-btn').style.display = 'none';
  document.getElementById('api-status-wrap').innerHTML = '';

  var meta   = _EP_META[pathStr] || {};
  var desc   = meta.desc || '';
  var auth   = meta.auth || 'viewer';
  var params = meta.params || [];

  var authClass = 'ep-auth-' + auth;
  var authLabel = auth === 'none' ? '&#x1F513; Public' : '&#x1F512; ' + auth;

  // Build drawer body
  var html = '';

  // Overview section
  html += '<div class="ep-section">';
  html += '<div class="ep-section-title">Overview</div>';
  if (desc) html += '<div class="ep-desc-text">' + desc + '</div>';
  html += '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">';
  html += '<span class="badge ' + method + '" style="font-size:10px">' + method + '</span>';
  html += '<span style="font-family:JetBrains Mono,monospace;font-size:12px;color:var(--text2)">' + pathStr + '</span>';
  html += '<span class="ep-auth-badge ' + authClass + '">' + authLabel + '</span>';
  html += '</div>';
  html += '</div>';

  // Request section
  if (method === 'WS') {
    html += '<div class="ep-section" style="flex:1">';
    html += '<div class="ep-section-title">WebSocket Terminal</div>';
    html += '<div class="ws-terminal" id="ws-term"></div>';
    html += '<div class="ws-input-row"><input class="ws-input" id="ws-msg" placeholder="JSON message to send..."/><button class="ws-send-btn" onclick="wsSend()">Send</button></div>';
    html += '</div>';
    execBtn.textContent = '\\u26A1 Connect';
    execBtn.onclick = function(){ wsConnect(pathStr); };
  } else {
    html += '<div class="ep-section">';
    html += '<div class="ep-section-title">Request</div>';
    if (params.length > 0) {
      html += _buildFormFields(params);
    } else {
      // Fallback: show path-override input for GET with path params, or raw textarea for POST
      var hasPathParam = pathStr.includes('{');
      if (hasPathParam) {
        html += '<div class="ep-field"><div class="ep-field-label">Path</div>';
        html += '<input class="ep-input" id="api-path-override" value="' + pathStr + '"/></div>';
      } else if (method !== 'GET') {
        var defaultBody = _EP_DEFAULTS[pathStr] || '{}';
        if (defaultBody.startsWith('?')) {
          html += '<div class="ep-field"><div class="ep-field-label">Path + Query</div>';
          html += '<input class="ep-input" id="api-path-override" value="' + pathStr + defaultBody + '"/></div>';
        } else {
          html += '<div class="ep-field"><div class="ep-field-label">Request Body (JSON)</div>';
          html += '<textarea class="ep-textarea" id="api-req-body" style="min-height:110px">' + defaultBody + '</textarea></div>';
        }
      } else {
        html += '<div style="font-size:12px;color:var(--muted)">No parameters required — click Execute to call this endpoint.</div>';
      }
    }
    html += '</div>';
    // Response section (hidden until executed)
    html += '<div class="ep-section ep-resp-wrap" id="ep-resp-wrap">';
    html += '<div class="ep-section-title">Response</div>';
    html += '<div class="ep-status-row" id="ep-status-row"></div>';
    html += '<pre class="ep-resp-pre" id="ep-resp-pre" style="margin-top:8px"></pre>';
    html += '</div>';
    execBtn.onclick = sendApiRequest;
  }

  document.getElementById('ep-drawer-body').innerHTML = html;

  // Wire up boolean toggle labels
  (params||[]).forEach(function(p) {
    if (p.type !== 'boolean') return;
    var btn = document.getElementById(_getFieldId(p.name));
    if (!btn) return;
    var lbl = document.getElementById(_getFieldId(p.name) + '-lbl');
    btn.addEventListener('click', function(){ if(lbl) lbl.textContent = btn.classList.contains('on') ? 'true' : 'false'; });
  });

  // Open drawer + backdrop
  document.getElementById('ep-backdrop').classList.add('open');
  document.getElementById('ep-drawer').classList.add('open');
}

function closeApiModal() {
  document.getElementById('ep-backdrop').classList.remove('open');
  document.getElementById('ep-drawer').classList.remove('open');
  if (_apiWs) { try { _apiWs.close(); } catch(e){} _apiWs = null; }
}

function sendApiRequest() {
  var method  = _apiCurrentMethod;
  var path    = _apiCurrentPath;

  // Path override
  var pathOverride = document.getElementById('api-path-override');
  if (pathOverride) path = pathOverride.value.trim();

  var execBtn = document.getElementById('ep-exec-btn');
  execBtn.disabled = true;
  execBtn.textContent = '\\u23F3 Running...';

  var opts = { method: method, headers: Object.assign({}, authHeaders()) };
  delete opts.headers['Content-Type']; // set below only if needed

  var meta   = _EP_META[path] || _EP_META[_apiCurrentPath] || {};
  var params = meta.params || [];

  if (method !== 'GET' && method !== 'DELETE') {
    var bodyObj = null;
    var rawBody = document.getElementById('api-req-body');
    if (rawBody) {
      try { bodyObj = JSON.parse(rawBody.value); }
      catch(e) { _showDrawerResp('Invalid JSON: ' + e.message, 'err', null, 0); execBtn.disabled=false; execBtn.textContent='\\u25B6 Execute'; return; }
    } else if (params.length > 0) {
      bodyObj = _collectParams(params);
    } else {
      bodyObj = {};
    }
    opts.body = JSON.stringify(bodyObj);
    opts.headers['Content-Type'] = 'application/json';
  }

  var t0 = Date.now();
  fetch(path, opts)
    .then(function(r) {
      var status = r.status;
      var ms     = Date.now() - t0;
      return r.text().then(function(t){ return {status: status, body: t, ms: ms}; });
    })
    .then(function(res) {
      var pretty = res.body;
      try { pretty = JSON.stringify(JSON.parse(res.body), null, 2); } catch(e){}
      _apiLastResp = pretty;
      var cls = res.status >= 200 && res.status < 300 ? 'ok' : 'err';
      _showDrawerResp(pretty, cls, res.status, res.ms);
    })
    .catch(function(e) { _showDrawerResp('Network error: ' + e, 'err', null, 0); })
    .finally(function() { execBtn.disabled=false; execBtn.textContent='\\u25B6 Execute'; });
}

function _showDrawerResp(text, cls, status, ms) {
  var wrap   = document.getElementById('ep-resp-wrap');
  var pre    = document.getElementById('ep-resp-pre');
  var srow   = document.getElementById('ep-status-row');
  var sw     = document.getElementById('api-status-wrap');
  var copyBtn = document.getElementById('ep-copy-resp-btn');
  if (!wrap || !pre) return;
  wrap.classList.add('visible');
  pre.textContent = text;
  pre.className   = 'ep-resp-pre ' + cls;
  if (srow && status !== null) {
    var sc = status >= 200 && status < 300 ? '2xx' : status >= 300 && status < 400 ? '3xx' : status >= 500 ? '5xx' : '4xx';
    srow.innerHTML = '<span class="ep-status-badge ep-status-' + sc + '">' + status + '</span>' +
      (ms > 0 ? '<span class="ep-timing">' + ms + ' ms</span>' : '');
    if (sw) sw.innerHTML = '<span class="ep-status-badge ep-status-' + sc + '">' + status + '</span>';
  }
  if (copyBtn) copyBtn.style.display = '';
  // Scroll response into view
  wrap.scrollIntoView({behavior:'smooth', block:'nearest'});
}

function copyAsCurl() {
  var method  = _apiCurrentMethod;
  var path    = _apiCurrentPath;
  var pathOv  = document.getElementById('api-path-override');
  if (pathOv) path = pathOv.value.trim();
  var base    = location.origin + path;
  var headers = authHeaders();
  var hParts  = Object.keys(headers).filter(function(k){ return k !== 'Content-Type'; }).map(function(k){
    return "-H '" + k + ': ' + headers[k] + "'";
  });

  var meta   = _EP_META[path] || _EP_META[_apiCurrentPath] || {};
  var params = meta.params || [];
  var body   = '';
  if (method !== 'GET' && method !== 'DELETE') {
    var rawBody = document.getElementById('api-req-body');
    if (rawBody) {
      body = rawBody.value.trim();
    } else if (params.length > 0) {
      body = JSON.stringify(_collectParams(params));
    }
  }

  var cmd = "curl -X " + method + " '" + base + "'";
  hParts.forEach(function(h){ cmd += ' ' + h; });
  if (body) {
    cmd += " -H 'Content-Type: application/json'";
    cmd += " -d " + JSON.stringify(body);
  }
  navigator.clipboard.writeText(cmd);
  toast('cURL command copied', 'ok', 1800);
}

function copyApiResp() {
  if (_apiLastResp) {
    navigator.clipboard.writeText(_apiLastResp);
    toast('Response copied', 'ok', 1600);
  }
}

/* ── WS TERMINAL ── */
function wsLog(msg, cls) {
  var term = document.getElementById('ws-term');
  if (!term) return;
  var p = document.createElement('p');
  p.className = 'ws-line ' + (cls||'sys');
  p.textContent = msg;
  term.appendChild(p);
  term.scrollTop = term.scrollHeight;
}

function wsConnect(pathStr) {
  if (_apiWs) { _apiWs.close(); _apiWs = null; }
  var btn = document.getElementById('api-send-btn');
  var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  var url = proto + '//' + location.host + pathStr;
  wsLog('Connecting to ' + url + '...', 'sys');
  try {
    _apiWs = new WebSocket(url);
    _apiWs.onopen = function() {
      wsLog('\\u2705 Connected', 'sys');
      btn.textContent = '\\u274C Disconnect';
      btn.onclick = function(){ _apiWs.close(); };
    };
    _apiWs.onmessage = function(e) {
      var txt = e.data;
      try { txt = JSON.stringify(JSON.parse(e.data), null, 2); } catch(_){}
      wsLog('\\u2190 ' + txt, 'recv');
    };
    _apiWs.onerror = function() { wsLog('\\u274C Error', 'err'); };
    _apiWs.onclose = function() {
      wsLog('Disconnected', 'sys');
      btn.textContent = '\\u26A1 Connect';
      btn.onclick = function(){ wsConnect(pathStr); };
      _apiWs = null;
    };
  } catch(e) { wsLog('Failed: ' + e, 'err'); }
}

function wsSend() {
  var inp = document.getElementById('ws-msg');
  if (!inp || !inp.value.trim()) return;
  if (!_apiWs || _apiWs.readyState !== 1) { wsLog('Not connected — click Connect first', 'err'); return; }
  _apiWs.send(inp.value.trim());
  wsLog('\\u2192 ' + inp.value.trim(), 'sent');
  inp.value = '';
}

function loadMetrics() {
  fetch('/health/integrations').then(function(r){ return r.json(); }).then(function(d) {
    var el = document.getElementById('m-llm');
    if (el) {
      var provLabels = {anthropic:'Claude', groq:'Groq / Llama', ollama:'Ollama (Local)', none:'None'};
      el.textContent = provLabels[d.llm_provider] || d.llm_provider || '—';
      var sub = document.getElementById('m-llm-sub');
      if (sub) sub.textContent = d.llm_provider === 'groq' ? 'llama-3.3-70b' : d.llm_provider === 'anthropic' ? 'claude-sonnet-4-6' : 'Active model';
    }
    var ints = document.getElementById('m-ints');
    if (ints) {
      var count = Object.values(d.integrations || {}).filter(function(v){ return v; }).length;
      animateNum(ints, count);
      var sub2 = document.getElementById('m-ints-sub');
      if (sub2) sub2.textContent = 'of 8 services';
    }
    // Update integration chips
    var imap = d.integrations || {};
    var chipKeys = {
      'ANTHROPIC_API_KEY': 'int-claude', 'GROQ_API_KEY': 'int-claude',
      'AWS_ACCESS_KEY_ID': 'int-aws',
      'GITHUB_TOKEN': 'int-github',
      'GRAFANA_TOKEN': 'int-grafana',
      'SLACK_BOT_TOKEN': 'int-slack',
      'JIRA_TOKEN': 'int-jira',
      'OPSGENIE_API_KEY': 'int-opsgenie',
      'KUBECONFIG': 'int-k8s',
    };
    Object.keys(imap).forEach(function(k) {
      var chipId = chipKeys[k];
      if (chipId) {
        var chip = document.getElementById(chipId);
        if (chip && imap[k]) chip.classList.add('on');
      }
    });
  }).catch(function(){});

  fetch('/secrets/status').then(function(r){ return r.json(); }).then(function(data) {
    Object.keys(data).forEach(function(group) {
      var keys = data[group];
      Object.keys(keys).forEach(function(k) {
        var st = document.getElementById('st-'+k);
        if (st) {
          st.textContent = keys[k] ? '\\u2713' : '';
          st.style.color = keys[k] ? 'var(--green)' : '';
          st.title = keys[k] ? 'Configured' : 'Not set';
        }
      });
      var allSet = Object.values(keys).every(Boolean);
      var someSet = Object.values(keys).some(Boolean);
      var chipId = INT_MAP[group];
      if (chipId) {
        var chip = document.getElementById(chipId);
        if (chip) { chip.className = 'tb-chip int-chip' + (someSet ? ' on' : ''); }
      }
      var schipId = SEC_MAP[group];
      if (schipId) {
        var schip = document.getElementById(schipId);
        if (schip) { schip.className = 'int-chip' + (someSet ? ' on' : ''); }
      }
    });
    // Count eps
    var epEls = document.querySelectorAll('[data-ep]');
    var cntEl = document.getElementById('cnt-all');
    if (cntEl) cntEl.textContent = epEls.length;
    var mEps = document.getElementById('m-eps');
    if (mEps) animateNum(mEps, epEls.length);
  }).catch(function(){});

  // Memory count from health
  fetch('/health').then(function(r){ return r.json(); }).then(function(d) {
    var mMem = document.getElementById('m-mem');
    if (mMem) {
      if (d.incident_count !== undefined) animateNum(mMem, d.incident_count);
      else mMem.textContent = '0';
    }
  }).catch(function(){});
}

function saveSecrets() {
  var secrets = {};
  ALL_KEYS.forEach(function(k) {
    var el = document.getElementById(k);
    if (el && el.value.trim()) secrets[k] = el.value.trim();
  });
  if (Object.keys(secrets).length === 0) { showMsg('No values entered', 'err'); return; }
  var btn = document.getElementById('save-btn');
  btn.disabled = true; btn.textContent = 'Saving...';
  fetch('/secrets', {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({secrets: secrets})
  }).then(function(r){ return r.json().then(function(d){ return {ok: r.ok, data: d}; }); })
  .then(function(res) {
    if (res.ok) {
      showMsg('Saved ' + res.data.updated.length + ' secret(s)', 'ok');
      ALL_KEYS.forEach(function(k){ var el=document.getElementById(k); if(el) el.value=''; });
      loadMetrics();
    } else { showMsg(res.data.detail || 'Error', 'err'); }
  }).catch(function(){ showMsg('Network error', 'err'); })
  .finally(function(){ btn.disabled=false; btn.textContent='&#x1F4BE; Save to .env'; });
}
function showMsg(text, type) {
  var m = document.getElementById('sec-msg');
  m.textContent = text; m.className = 'sec-msg ' + type; m.style.display = '';
  setTimeout(function(){ m.style.display='none'; }, 4000);
}

function smtpMsg(text, ok) {
  var m = document.getElementById('smtp-msg');
  if (!m) return;
  m.textContent = text;
  m.style.color = ok ? 'var(--green)' : '#f87171';
  m.style.display = 'inline';
  setTimeout(function(){ m.style.display='none'; }, 6000);
}

function saveSMTP() {
  var user = (document.getElementById('smtp_user_inp')||{}).value||'';
  var pass = (document.getElementById('smtp_pass_inp')||{}).value||'';
  var from = (document.getElementById('smtp_from_inp')||{}).value||'';
  var url  = (document.getElementById('app_url_inp')||{}).value||'';
  if (!user || !pass) { smtpMsg('Enter SMTP_USER and SMTP_PASSWORD', false); return; }
  smtpMsg('Saving & testing...', true);
  fetch('/auth/configure-smtp', {
    method: 'POST', headers: authHeaders(),
    body: JSON.stringify({smtp_user: user, smtp_password: pass, smtp_from: from||user,
                          app_url: url||'http://localhost:8000', smtp_host:'smtp.gmail.com', smtp_port:587})
  }).then(function(r){ return r.json(); }).then(function(d){
    smtpMsg(d.message || (d.success ? 'Saved' : 'Error'), d.success);
  }).catch(function(){ smtpMsg('Network error', false); });
}

function testEmail() {
  smtpMsg('Sending test email...', true);
  fetch('/auth/test-email', {method:'POST', headers: authHeaders()})
    .then(function(r){ return r.json(); })
    .then(function(d){
      if (d.success) smtpMsg(d.message, true);
      else smtpMsg(d.detail || 'Failed', false);
    }).catch(function(){ smtpMsg('Network error', false); });
}

/* ── MARKDOWN RENDERER ── */
function _renderMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/```[\\w]*(\\r?\\n)([\\s\\S]*?)```/g, function(_,nl,c){ return '<pre><code>'+c.trim()+'</code></pre>'; })
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
    .replace(/__(.+?)__/g, '<strong>$1</strong>')
    .replace(/\\*([^*]+)\\*/g, '<em>$1</em>')
    .replace(/^---+$/gm, '<hr>')
    .replace(/^[ \\t]*[-*] (.+)$/gm, '<li>$1</li>')
    .replace(/^[ \\t]*\\d+\\. (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>[\\s\\S]*?<\\/li>)+/g, function(m){ return '<ul>'+m+'</ul>'; })
    .replace(/^### (.+)$/gm,'<h4>$1</h4>')
    .replace(/^## (.+)$/gm,'<h3>$1</h3>')
    .replace(/^# (.+)$/gm,'<h2>$1</h2>')
    .replace(/\\n/g, '<br>');
}

var _chatHistory = [];

function appendMsg(role, text, isHtml) {
  var empty = document.getElementById('chat-empty');
  if (empty) empty.remove();
  var container = document.getElementById('chat-messages');
  var row = document.createElement('div');
  row.className = 'chat-row ' + role + ' fade-in';
  var meta = document.createElement('div');
  meta.className = 'chat-meta' + (role === 'user' ? ' right' : '');
  meta.textContent = role === 'user' ? 'You' : 'AI DevOps';
  var bubble = document.createElement('div');
  bubble.className = 'chat-bubble ' + role;
  if (isHtml) { bubble.innerHTML = text; } else { bubble.textContent = text; }
  row.appendChild(meta);
  row.appendChild(bubble);
  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
  return bubble;
}

function sendSuggestion(btn) {
  document.getElementById('chat-input').value = btn.textContent.replace(/^\\S+\\s*/, '');
  sendChat();
}

function chatKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
}
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 130) + 'px';
}

var _pendingAction = null;
var _pendingParams = null;

function sendChat(overrideMsg, confirmed, pendingAction, pendingParams) {
  var input = document.getElementById('chat-input');
  var msg = overrideMsg !== undefined ? overrideMsg : input.value.trim();
  if (!msg) return;
  var btn = document.getElementById('chat-send-btn');
  if (!overrideMsg) { input.value = ''; input.style.height = 'auto'; }
  btn.disabled = true;
  appendMsg('user', msg);
  _chatHistory.push({role: 'user', content: msg});
  var typingBubble = appendMsg('ai', '<span class="typing-dots"><span>.</span><span>.</span><span>.</span></span>', true);
  typingBubble.classList.add('typing');
  var selProvider = (document.getElementById('llm-selector') || {}).value || '';
  var body = {
    message: msg,
    history: _chatHistory.slice(0, -1),
    provider: selProvider,
    confirmed: confirmed || false,
    pending_action: pendingAction || null,
    pending_params: pendingParams || null
  };
  fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  })
  .then(function(r) {
    var status = r.status;
    return r.text().then(function(t) { return {status: status, text: t}; });
  })
  .then(function(res) {
    var d;
    try { d = JSON.parse(res.text); }
    catch(e) {
      typingBubble.classList.remove('typing');
      typingBubble.innerHTML = '<span style="color:var(--red)">Server error (' + res.status + '). Please try again or check server logs.</span>';
      _chatHistory.push({role: 'assistant', content: 'Server error. Please try again.'});
      document.getElementById('chat-send-btn').disabled = false;
      document.getElementById('chat-input').focus();
      return;
    }
    typingBubble.classList.remove('typing');
    var reply = d.reply || d.detail || 'No response';
    typingBubble.innerHTML = _renderMarkdown(reply);
    _chatHistory.push({role: 'assistant', content: reply});

    // ── Confirmation card ──────────────────────────────────
    if (d.needs_confirm && d.pending_action) {
      _pendingAction = d.pending_action;
      _pendingParams = d.pending_params;
      var confirmCard = document.createElement('div');
      confirmCard.style.cssText = 'margin-top:10px;display:flex;gap:8px;flex-wrap:wrap';
      var yesBtn = document.createElement('button');
      yesBtn.innerHTML = '\\u2705 Yes, proceed';
      yesBtn.style.cssText = 'padding:7px 16px;background:rgba(52,211,153,.15);border:1px solid rgba(52,211,153,.4);color:var(--green);border-radius:6px;cursor:pointer;font-size:12px;font-weight:600;font-family:inherit';
      yesBtn.onclick = function() {
        confirmCard.remove();
        sendChat('Yes, confirmed', true, _pendingAction, _pendingParams);
        _pendingAction = null; _pendingParams = null;
      };
      var noBtn = document.createElement('button');
      noBtn.innerHTML = '\\u274c No, cancel';
      noBtn.style.cssText = 'padding:7px 16px;background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);color:var(--red);border-radius:6px;cursor:pointer;font-size:12px;font-weight:600;font-family:inherit';
      noBtn.onclick = function() {
        confirmCard.remove();
        appendMsg('ai', 'Operation cancelled.');
        _chatHistory.push({role: 'assistant', content: 'Operation cancelled.'});
        _pendingAction = null; _pendingParams = null;
        document.getElementById('chat-messages').scrollTop = 999999;
      };
      confirmCard.appendChild(yesBtn);
      confirmCard.appendChild(noBtn);
      typingBubble.parentElement.appendChild(confirmCard);
    }

    var footer = document.createElement('div');
    footer.style.cssText = 'font-size:10px;color:var(--muted);margin-top:6px;padding:0 4px;display:flex;gap:10px;flex-wrap:wrap';
    if (d.sources && d.sources.length) {
      var src = document.createElement('span');
      src.textContent = '\\u1F4E1 ' + d.sources.join(', ');
      footer.appendChild(src);
    }
    if (d.llm_provider && d.llm_provider !== 'none') {
      var prov = document.createElement('span');
      var pLabel = {anthropic:'\\u2728 Claude',groq:'\\u26A1 Groq/Llama',openai:'GPT-4o',ollama:'\\u1F3E0 Ollama'}[d.llm_provider] || d.llm_provider;
      prov.textContent = pLabel;
      footer.appendChild(prov);
    }
    // Only show badge + toast for mutating actions, not read-only ones
    var _mutatingActions = ['restart_deployment','scale_deployment','delete_pod','cordon_node','uncordon_node',
      'start_ec2','stop_ec2','reboot_ec2','scale_ecs_service','redeploy_ecs_service','invoke_lambda',
      'reboot_rds','set_alarm_state','create_github_issue','create_jira_ticket','run_pipeline','notify_oncall','debug_and_fix'];
    if (d.action_taken && _mutatingActions.indexOf(d.action_taken) !== -1) {
      var actionBadge = document.createElement('div');
      actionBadge.style.cssText = 'margin-top:8px;padding:6px 10px;background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.3);border-radius:6px;font-size:11px;color:var(--green);display:flex;align-items:center;gap:6px';
      actionBadge.innerHTML = '<span>&#x2705;</span><span><strong>' + d.action_taken.replace(/_/g,' ') + '</strong> executed successfully</span>';
      typingBubble.parentElement.appendChild(actionBadge);
      toast(d.action_taken.replace(/_/g,' ') + ' completed', 'ok', 3000);
    }
    if (d.action_taken) {
      var mA = document.getElementById('m-actions');
      if (mA && d.action_count !== undefined) animateNum(mA, d.action_count);
    }
    if (footer.children.length) typingBubble.parentElement.appendChild(footer);
    document.getElementById('chat-messages').scrollTop = 999999;
    btn.disabled = false;
    document.getElementById('chat-input').focus();
  })
  .catch(function(e) {
    typingBubble.classList.remove('typing');
    typingBubble.textContent = 'Error: ' + e;
    typingBubble.style.color = 'var(--red)';
    btn.disabled = false;
  });
}

function createWarRoom() {
  var lastMsg = '';
  for (var i = _chatHistory.length-1; i >= 0; i--) {
    if (_chatHistory[i].role === 'user') { lastMsg = _chatHistory[i].content; break; }
  }
  var desc = lastMsg || document.getElementById('chat-input').value.trim() || 'Infrastructure incident';
  var incId = 'INC-' + new Date().toISOString().replace(/[^0-9]/g,'').slice(0,12);
  var btn = document.getElementById('chat-warroom-btn');
  btn.disabled = true; btn.textContent = '\\u23F3 Creating...';
  appendMsg('ai', '\\u1F6A8 Creating war room and running analysis across all integrations...');
  fetch('/warroom/create', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({incident_id: incId, description: desc, severity: 'high', post_to_slack: true})
  })
  .then(function(r){ return r.json(); })
  .then(function(d) {
    btn.disabled = false; btn.textContent = '\\u1F6A8 War Room';
    var msg = '\\u2705 War room created: ' + incId;
    var a = d.analysis || {};
    if (a.summary) msg += '\\n\\n\\u1F4CB ' + a.summary;
    if (a.root_cause) msg += '\\n\\n\\u1F50D Root cause: ' + a.root_cause;
    if (d.slack && d.slack.channel_url) msg += '\\n\\n\\u1F517 Slack: ' + d.slack.channel_url;
    appendMsg('ai', msg);
    _chatHistory.push({role: 'assistant', content: msg});
    document.getElementById('chat-messages').scrollTop = 999999;
  })
  .catch(function(e) {
    btn.disabled = false; btn.textContent = '\\u1F6A8 War Room';
    appendMsg('ai', 'Error: ' + e);
  });
}

loadMetrics();

// Restore last active view so page reloads/restarts keep you where you were
(function() {
  try {
    var saved = localStorage.getItem('devops_view');
    if (saved && saved !== 'dashboard') {
      // Find the matching nav link and activate it
      var navLink = document.querySelector('.nav-link[onclick*="' + saved + '"]');
      showView(saved, null, navLink || null);
      if (navLink) {
        document.querySelectorAll('.nav-link').forEach(function(l){ l.classList.remove('active'); });
        navLink.classList.add('active');
      }
    }
  } catch(e) {}
})();

// On load: check JWT and show login or restore session
(function() {
  var token = localStorage.getItem('devops_jwt');
  if (!token) {
    showLoginPage();
    return;
  }
  fetch('/auth/me', {headers: {'Authorization': 'Bearer ' + token}})
    .then(function(r){ return r.json().then(function(d){ return {ok: r.ok, data: d}; }); })
    .then(function(res) {
      if (!res.ok) { showLoginPage(); return; }
      _currentUser = res.data.username || res.data.user || _currentUser;
      _currentRole = res.data.role || 'viewer';
      applyRoleUI(res.data);
    })
    .catch(function(){ showLoginPage(); });
})();

/* ── RBAC ROLE LOADER ── */
var _currentUser = localStorage.getItem('devops_user') || '';
var _currentRole = localStorage.getItem('devops_role') || 'viewer';

function authHeaders(extra) {
  var h = Object.assign({'Content-Type': 'application/json'}, extra || {});
  var token = localStorage.getItem('devops_jwt');
  if (token) {
    h['Authorization'] = 'Bearer ' + token;
  } else if (_currentUser) {
    h['X-User'] = _currentUser;
  }
  return h;
}

function doLogin(username, password, onSuccess, onError) {
  var body = 'username=' + encodeURIComponent(username) + '&password=' + encodeURIComponent(password);
  fetch('/auth/token', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body: body})
    .then(function(r){ return r.json().then(function(d){ return {ok:r.ok, data:d}; }); })
    .then(function(res) {
      if (!res.ok) { onError(res.data.detail || 'Invalid credentials'); return; }
      _currentUser = res.data.username;
      _currentRole = res.data.role;
      localStorage.setItem('devops_jwt',  res.data.access_token);
      localStorage.setItem('devops_user', res.data.username);
      localStorage.setItem('devops_role', res.data.role);
      onSuccess(res.data);
    })
    .catch(function(e){ onError('Network error: ' + e); });
}

function doLogout() {
  localStorage.removeItem('devops_jwt');
  localStorage.removeItem('devops_user');
  localStorage.removeItem('devops_role');
  _currentUser = ''; _currentRole = 'viewer';
  document.body.removeAttribute('data-role');
  showLoginPage();
}

function loadRole(user) {
  // Legacy no-op - identity comes from JWT
}

function applyRoleUI(d) {
  var role = d.role || 'viewer';
  var perms = d.permissions || [];
  var resolvedUser = d.username || d.user || _currentUser;
  // Set body data-role for CSS visibility rules
  document.body.dataset.role = role;
  // Update sidebar footer
  var badge = document.getElementById('sb-role-badge');
  if (badge) {
    badge.textContent = role;
    badge.className = 'role-badge role-' + role;
  }
  var uname = document.getElementById('sb-uname');
  if (uname) uname.textContent = resolvedUser;
  var avatar = document.getElementById('sb-avatar');
  if (avatar) avatar.textContent = resolvedUser[0].toUpperCase();
  // Update role metric card
  var mRole = document.getElementById('m-role');
  if (mRole) mRole.textContent = role.charAt(0).toUpperCase() + role.slice(1);
  var mRolePerms = document.getElementById('m-role-perms');
  var roleDesc = {admin: 'Full platform access', developer: 'Read, write & deploy', viewer: 'View-only access'};
  if (mRolePerms) mRolePerms.textContent = roleDesc[role] || 'Limited access';
  // Sync user field in secrets panel
  var secUser = document.getElementById('sec-user');
  if (secUser) secUser.value = resolvedUser;
  toast('Signed in as ' + resolvedUser + ' (' + role + ')', 'info', 2000);
}

function showLoginPage() {
  var existing = document.getElementById('login-page');
  if (existing) { existing.style.display = 'flex'; setTimeout(function(){ var u=document.getElementById('login-username'); if(u)u.focus(); },80); return; }
  var page = document.createElement('div');
  page.id = 'login-page';
  page.style.cssText = 'position:fixed;inset:0;z-index:9999;display:flex;align-items:center;justify-content:center;overflow:hidden;font-family:Inter,sans-serif';
  page.innerHTML =
    /* ── animated gradient background ── */
    '<style>' +
    '@keyframes lp-bg{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}' +
    '@keyframes lp-float{0%,100%{transform:translateY(0) scale(1)}50%{transform:translateY(-18px) scale(1.04)}}' +
    '@keyframes lp-in{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}' +
    '#login-page{background:#04060f}' +
    '#lp-bg{position:absolute;inset:0;background:linear-gradient(135deg,#04060f 0%,#0d1424 40%,#0a0f1e 70%,#04060f 100%);background-size:400% 400%;animation:lp-bg 12s ease infinite}' +
    '#lp-orb1{position:absolute;width:500px;height:500px;border-radius:50%;background:radial-gradient(circle,rgba(124,58,237,.18) 0%,transparent 70%);top:-120px;right:-100px;animation:lp-float 8s ease-in-out infinite}' +
    '#lp-orb2{position:absolute;width:400px;height:400px;border-radius:50%;background:radial-gradient(circle,rgba(37,99,235,.15) 0%,transparent 70%);bottom:-100px;left:-80px;animation:lp-float 10s ease-in-out infinite reverse}' +
    '#lp-card{position:relative;z-index:2;width:100%;max-width:400px;margin:0 20px;animation:lp-in .5s ease both}' +
    '#lp-card input{transition:border-color .15s,box-shadow .15s}' +
    '#lp-card input:focus{border-color:#4f8ef7!important;box-shadow:0 0 0 3px rgba(79,142,247,.15)!important;outline:none}' +
    '#login-btn{transition:all .15s;letter-spacing:.02em}' +
    '#login-btn:hover:not(:disabled){background:linear-gradient(135deg,#3b82f6,#2563eb)!important;box-shadow:0 4px 20px rgba(79,142,247,.4)!important;transform:translateY(-1px)}' +
    '#login-btn:active:not(:disabled){transform:translateY(0)}' +
    '#login-btn:disabled{opacity:.6;cursor:not-allowed}' +
    '</style>' +
    '<div id="lp-bg"></div>' +
    '<div id="lp-orb1"></div>' +
    '<div id="lp-orb2"></div>' +
    '<div id="lp-card">' +
      /* logo + title */
      '<div style="text-align:center;margin-bottom:28px">' +
        '<div style="display:inline-flex;align-items:center;justify-content:center;width:56px;height:56px;background:linear-gradient(135deg,#7c3aed,#2563eb);border-radius:14px;margin-bottom:14px;box-shadow:0 8px 32px rgba(124,58,237,.4)">' +
          '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>' +
        '</div>' +
        '<h1 style="font-size:1.5em;font-weight:800;color:#e2e8f0;letter-spacing:-.02em;margin:0 0 4px">NexusOps</h1>' +
        '<p style="font-size:.83em;color:#4f6a9a;margin:0">AI-Powered DevOps Platform</p>' +
      '</div>' +
      /* card */
      '<div style="background:rgba(13,20,36,.85);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border:1px solid rgba(79,142,247,.18);border-radius:16px;padding:32px;box-shadow:0 24px 64px rgba(0,0,0,.6)">' +
        '<div id="login-err" style="display:none;color:#fca5a5;font-size:.82em;margin-bottom:16px;padding:10px 14px;background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.25);border-radius:8px;line-height:1.4"></div>' +
        '<div style="margin-bottom:16px">' +
          '<label style="display:block;font-size:.78em;font-weight:600;color:#4f8ef7;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Username</label>' +
          '<input id="login-username" type="text" placeholder="Enter your username" autocomplete="username" style="width:100%;box-sizing:border-box;padding:11px 14px;border-radius:8px;border:1px solid rgba(79,142,247,.2);background:rgba(4,6,15,.6);color:#e2e8f0;font-size:.9em;font-family:inherit"/>' +
        '</div>' +
        '<div style="margin-bottom:24px">' +
          '<label style="display:block;font-size:.78em;font-weight:600;color:#4f8ef7;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Password</label>' +
          '<input id="login-password" type="password" placeholder="Enter your password" autocomplete="current-password" style="width:100%;box-sizing:border-box;padding:11px 14px;border-radius:8px;border:1px solid rgba(79,142,247,.2);background:rgba(4,6,15,.6);color:#e2e8f0;font-size:.9em;font-family:inherit"/>' +
        '</div>' +
        '<button id="login-btn" onclick="submitLoginPage()" style="width:100%;padding:12px;border-radius:8px;border:none;background:linear-gradient(135deg,#2563eb,#1d4ed8);color:#fff;font-weight:700;font-size:.95em;cursor:pointer;font-family:inherit">Sign In</button>' +
        '<p style="text-align:center;font-size:.75em;color:#3d5080;margin-top:16px;margin-bottom:0">Press <kbd style="background:rgba(79,142,247,.12);border:1px solid rgba(79,142,247,.2);padding:1px 5px;border-radius:3px;font-family:monospace">Ctrl+K</kbd> to open AI Chat anytime</p>' +
      '</div>' +
    '</div>';
  document.body.appendChild(page);
  var uInput = document.getElementById('login-username');
  var pInput = document.getElementById('login-password');
  if (uInput) setTimeout(function(){ uInput.focus(); }, 100);
  if (pInput) pInput.addEventListener('keydown', function(e){ if (e.key === 'Enter') submitLoginPage(); });
  if (uInput) uInput.addEventListener('keydown', function(e){ if (e.key === 'Enter') { var p=document.getElementById('login-password'); if(p)p.focus(); } });
}

function submitLoginPage() {
  var username = (document.getElementById('login-username') || {}).value || '';
  var password = (document.getElementById('login-password') || {}).value || '';
  var errEl = document.getElementById('login-err');
  var btn = document.getElementById('login-btn');
  if (!username.trim()) { if(errEl){errEl.textContent='Enter username';errEl.style.display='block';} return; }
  if (btn) { btn.disabled = true; btn.textContent = 'Signing in...'; }
  doLogin(username.trim(), password, function(data) {
    if (btn) { btn.disabled = false; btn.textContent = 'Sign In'; }
    var page = document.getElementById('login-page');
    if (page) page.style.display = 'none';
    applyRoleUI(data);
    loadMetrics();
  }, function(err) {
    if (btn) { btn.disabled = false; btn.textContent = 'Sign In'; }
    if (errEl) { errEl.textContent = err; errEl.style.display = 'block'; }
  });
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function toggleUserSwitch() {
  var p = document.getElementById('user-switch-panel');
  if (p) p.style.display = p.style.display === 'none' ? '' : 'none';
}

function applyUser() {
  var inp = document.getElementById('sb-user-input');
  var u = inp ? inp.value.trim() : '';
  if (!u) return;
  var panel = document.getElementById('user-switch-panel');
  if (panel) panel.style.display = 'none';
  loadRole(u);
}

/* ── TOAST ── */
function toast(msg, type, dur) {
  type = type || 'info'; dur = dur || 2800;
  var w = document.getElementById('toast-wrap');
  var t = document.createElement('div');
  t.className = 'toast ' + type;
  var icons = {ok:'\\u2713', err:'\\u26A0', info:'\\u2139'};
  t.innerHTML = '<span style="font-size:14px">' + (icons[type]||'\\u2139') + '</span><span>' + msg + '</span>';
  w.appendChild(t);
  setTimeout(function(){ t.style.opacity='0'; t.style.transform='translateY(10px)'; t.style.transition='all .3s'; setTimeout(function(){ t.remove(); }, 300); }, dur);
}

/* ── ANIMATED COUNTER ── */
function animateNum(el, target) {
  var start = 0, dur = 600, startTime = null;
  function step(ts) {
    if (!startTime) startTime = ts;
    var p = Math.min((ts - startTime) / dur, 1);
    el.textContent = Math.floor(p * target);
    if (p < 1) requestAnimationFrame(step);
    else el.textContent = target;
  }
  requestAnimationFrame(step);
}

/* ── SEVERITY PICKER ── */
var _selSev = 'high';
function setSev(btn, val) {
  _selSev = val;
  document.querySelectorAll('.sev-pill').forEach(function(b){ b.classList.remove('active'); });
  btn.classList.add('active');
}

/* ── RUN PIPELINE ── */
function runPipeline() {
  var incId = document.getElementById('m-inc-id').value.trim() ||
              'INC-' + new Date().toISOString().replace(/[^0-9]/g,'').slice(0,12);
  var desc  = document.getElementById('m-inc-desc').value.trim();
  if (!desc) { toast('Please enter a description', 'err'); return; }
  var auto  = document.getElementById('m-auto-rem').checked;
  var btn   = document.getElementById('m-run-btn');
  var res   = document.getElementById('m-result');
  btn.disabled = true; btn.textContent = '\\u23F3 Running...';
  res.style.display = 'block';
  res.textContent = 'Sending to pipeline...';
  fetch('/incidents/run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({incident_id: incId, description: desc, severity: _selSev, auto_remediate: auto})
  })
  .then(function(r){ return r.json(); })
  .then(function(d) {
    btn.disabled = false; btn.textContent = '\\u25B6 Run Pipeline';
    var status = d.status || d.detail || 'unknown';
    res.textContent = JSON.stringify(d, null, 2);
    toast('Pipeline ' + status + ' — ' + incId, status === 'completed' ? 'ok' : 'info', 4000);
    loadMetrics();
  })
  .catch(function(e) {
    btn.disabled = false; btn.textContent = '\\u25B6 Run Pipeline';
    res.textContent = 'Error: ' + e;
    toast('Pipeline error: ' + e, 'err');
  });
}

/* ── GITHUB REPOS DRAWER ── */
function openGhDrawer() {
  var drawer = document.getElementById('gh-drawer');
  drawer.classList.add('open');
  var body = document.getElementById('gh-drawer-body');
  body.innerHTML = '<div style="text-align:center;padding:30px;color:var(--muted)">Loading repos&#x2026;</div>';
  fetch('/github/repos')
    .then(function(r){ return r.json(); })
    .then(function(d) {
      if (!d.success) {
        body.innerHTML = '<div style="padding:20px;color:var(--muted);font-size:12px">&#x26A0; ' + (d.error || 'Not configured') + '</div>';
        return;
      }
      var langColors = {JavaScript:'#f1e05a',TypeScript:'#2b7489',Python:'#3572A5',Go:'#00ADD8',Rust:'#dea584',Java:'#b07219',Ruby:'#701516',CSS:'#563d7c',HTML:'#e34c26',Shell:'#89e051'};
      var html = '<div style="font-size:11px;color:var(--muted);padding:4px 2px 10px;font-weight:600">' + d.owner + ' &#x2022; ' + d.count + ' repos</div>';
      (d.repos || []).forEach(function(r) {
        var lc = langColors[r.language] || '#8b9ec7';
        html += '<div class="repo-card" onclick="window.open(\\'' + r.url + '\\',\\'_blank\\')">';
        html += '<div class="repo-name">' + r.name + (r.private ? ' &#x1F512;' : '') + '</div>';
        if (r.description) html += '<div class="repo-desc">' + r.description + '</div>';
        html += '<div class="repo-meta">';
        if (r.language) html += '<span><span class="lang-dot" style="background:' + lc + '"></span>' + r.language + '</span>';
        html += '<span>&#x2B50; ' + r.stars + '</span>';
        html += '<span>&#x1F374; ' + r.forks + '</span>';
        if (r.open_issues > 0) html += '<span style="color:var(--amber)">&#x26A0; ' + r.open_issues + '</span>';
        html += '</div></div>';
      });
      body.innerHTML = html;
    })
    .catch(function(){ body.innerHTML = '<div style="padding:20px;color:var(--muted);font-size:12px">Failed to load repos</div>'; });
}

/* ── TEAM & ACCESS ── */
var _ROLES = ['viewer','developer','admin'];

function loadUsers() {
  var list = document.getElementById('users-list');
  if (!list) return;
  list.innerHTML = '<div style="padding:20px;text-align:center;opacity:.5;font-size:13px">Loading...</div>';
  fetch('/users', {headers: authHeaders()})
    .then(function(r){ return r.json(); })
    .then(function(d) {
      var users = d.users || [];
      if (!users.length) { list.innerHTML = '<div style="padding:20px;text-align:center;opacity:.5;font-size:13px">No users</div>'; return; }
      list.innerHTML = users.map(function(u) {
        var isMe = u.username === _currentUser;
        var roleOpts = _ROLES.map(function(r){ return '<option value="'+r+'"'+(u.role===r?' selected':'')+'>'+r+'</option>'; }).join('');
        var created = u.created_at ? u.created_at.slice(0,10) : '\\u2014';
        return '<div style="padding:13px 18px;border-bottom:1px solid var(--border);display:grid;grid-template-columns:1fr 160px 140px 100px;gap:10px;align-items:center;font-size:13px">' +
          '<div style="display:flex;align-items:center;gap:10px">' +
            '<div style="width:32px;height:32px;background:linear-gradient(135deg,#7c3aed,#2563eb);border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;flex-shrink:0">'+u.username[0].toUpperCase()+'</div>' +
            '<div><div style="font-weight:600">'+escHtml(u.username)+(isMe?' <span style="font-size:.75em;opacity:.5">(you)</span>':'')+'</div>' +
            '<div style="font-size:.75em;color:var(--muted)">by '+escHtml(u.created_by||'system')+'</div></div>' +
          '</div>' +
          '<div><select data-un="'+escHtml(u.username)+'" onchange="changeRole(this.dataset.un,this.value)" style="background:var(--bg);border:1px solid var(--border);border-radius:4px;color:var(--text);font-size:.82em;padding:4px 6px;width:100%;cursor:pointer">'+roleOpts+'</select></div>' +
          '<div style="color:var(--muted);font-size:.82em">'+created+'</div>' +
          '<div style="display:flex;gap:6px">' +
            '<button data-un="'+escHtml(u.username)+'" onclick="resetPw(this.dataset.un)" style="padding:4px 8px;background:var(--surface2);border:1px solid var(--border);border-radius:4px;font-size:.78em;cursor:pointer;color:var(--text)" title="Reset password">&#x1F511;</button>' +
            (isMe ? '' : '<button data-un="'+escHtml(u.username)+'" onclick="removeUser(this.dataset.un)" style="padding:4px 8px;background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.3);border-radius:4px;font-size:.78em;cursor:pointer;color:#f87171" title="Remove">&#x1F5D1;</button>') +
          '</div>' +
        '</div>';
      }).join('');
    })
    .catch(function(e){ list.innerHTML = '<div style="padding:20px;color:#f87171;font-size:13px">Error: '+e+'</div>'; });
}

function changeRole(username, role) {
  fetch('/users/'+encodeURIComponent(username)+'/role', {
    method: 'PUT', headers: authHeaders(),
    body: JSON.stringify({user: username, role: role})
  }).then(function(r){ return r.json(); }).then(function(d){
    if (d.success) toast('Role updated for '+username, 'ok');
    else toast(d.detail || d.reason || 'Failed', 'err');
    loadUsers();
  });
}

function removeUser(username) {
  if (!confirm('Delete user "'+username+'"? This cannot be undone.')) return;
  fetch('/users/'+encodeURIComponent(username), {method:'DELETE', headers: authHeaders()})
    .then(function(r){ return r.json(); }).then(function(d){
      if (d.success) { toast('User removed', 'ok'); loadUsers(); }
      else toast(d.detail || 'Failed', 'err');
    });
}

function resetPw(username) {
  var pw = prompt('Set new password for "'+username+'" (min 8 chars):');
  if (!pw) return;
  if (pw.length < 8) { toast('Password too short', 'err'); return; }
  fetch('/users/'+encodeURIComponent(username)+'/password', {
    method:'PUT', headers: authHeaders(),
    body: JSON.stringify({new_password: pw})
  }).then(function(r){ return r.json(); }).then(function(d){
    if (d.success) toast('Password reset', 'ok');
    else toast(d.detail || d.error || 'Failed', 'err');
  });
}

function showInviteModal() {
  var m = document.getElementById('invite-modal');
  if (m) { m.style.display = 'flex'; var u=document.getElementById('inv-username'); if(u) u.focus(); }
}
function closeInviteModal() {
  var m = document.getElementById('invite-modal');
  if (m) m.style.display = 'none';
  var e = document.getElementById('invite-err'); if(e) e.style.display='none';
}

function sendInvite() {
  var username = document.getElementById('inv-username').value.trim();
  var email    = document.getElementById('inv-email').value.trim();
  var role     = document.getElementById('inv-role').value;
  var errEl    = document.getElementById('invite-err');
  errEl.style.display = 'none';
  if (!username) { errEl.textContent='Enter a username'; errEl.style.display='block'; return; }
  if (!email || !email.includes('@')) { errEl.textContent='Enter a valid email'; errEl.style.display='block'; return; }
  var btn = document.getElementById('inv-send-btn');
  btn.disabled = true; btn.textContent = 'Sending...';
  fetch('/users/invite', {method:'POST', headers: authHeaders(), body: JSON.stringify({username:username,email:email,role:role})})
    .then(function(r){ return r.json(); })
    .then(function(d) {
      btn.disabled=false; btn.textContent='Send Invite';
      closeInviteModal();
      var res = document.getElementById('invite-result');
      if (d.success) {
        var html = '<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">' +
          '<span style="color:var(--green);font-size:1.1em">&#x2705;</span>' +
          '<div><div style="font-weight:600;color:var(--green)">User created successfully</div>' +
          '<div style="font-size:.8em;color:var(--muted)">Share the OTP and setup link with '+escHtml(username)+'</div></div>' +
          '</div>';
        if (d.otp) {
          html += '<div style="margin-bottom:10px">' +
            '<div style="font-size:.75em;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px">One-Time Password (OTP)</div>' +
            '<div style="background:var(--bg);border:1px solid var(--border2);padding:12px 16px;border-radius:8px;display:flex;align-items:center;justify-content:space-between">' +
              '<span style="font-family:JetBrains Mono,monospace;font-size:1.4em;font-weight:700;letter-spacing:.25em;color:var(--blue)">'+escHtml(d.otp)+'</span>' +
              '<button onclick="navigator.clipboard.writeText(this.dataset.v)" data-v="'+escHtml(d.otp)+'" style="padding:4px 10px;font-size:.75em;background:var(--surface2);border:1px solid var(--border);border-radius:4px;color:var(--text);cursor:pointer">Copy</button>' +
            '</div></div>';
        }
        if (d.setup_link) {
          html += '<div>' +
            '<div style="font-size:.75em;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px">Setup Link</div>' +
            '<div style="background:var(--bg);border:1px solid var(--border);padding:10px 12px;border-radius:8px;display:flex;align-items:center;gap:8px;flex-wrap:wrap">' +
              '<a href="'+escHtml(d.setup_link)+'" target="_blank" style="color:var(--blue);font-size:.8em;word-break:break-all;flex:1">'+escHtml(d.setup_link)+'</a>' +
              '<button onclick="navigator.clipboard.writeText(this.dataset.v)" data-v="'+escHtml(d.setup_link)+'" style="padding:4px 10px;font-size:.75em;background:var(--surface2);border:1px solid var(--border);border-radius:4px;color:var(--text);cursor:pointer;flex-shrink:0">Copy</button>' +
            '</div></div>';
        }
        if (d.email_sent === false) {
          html += '<div style="margin-top:10px;padding:8px 12px;background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.2);border-radius:6px;font-size:.78em;color:var(--amber)">&#x26A0; Email not sent — SMTP not configured. Share OTP manually.</div>';
        }
        res.innerHTML = html;
        loadUsers();
      } else {
        res.innerHTML = '<div style="color:#f87171;display:flex;gap:8px;align-items:flex-start"><span>&#x274C;</span><span>'+escHtml(d.detail||'Failed to create invite')+'</span></div>';
      }
      res.style.display = 'block';
    })
    .catch(function(e){ btn.disabled=false; btn.textContent='Send Invite'; toast('Network error','err'); });
}

/* ── KEYBOARD SHORTCUTS ── */
document.addEventListener('keydown', function(e) {
  // Ignore if typing in an input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'r' || e.key === 'R') {
    document.getElementById('run-modal').classList.add('open');
  }
  if (e.key === '/' ) {
    e.preventDefault();
    var s = document.getElementById('ep-search');
    if (s) { s.focus(); showView('endpoints','all',document.querySelector('.nav-link')); }
  }
  if (e.key === 'Escape') {
    document.getElementById('run-modal').classList.remove('open');
    document.getElementById('gh-drawer').classList.remove('open');
    closeApiModal();
    var backdrop = document.getElementById('ep-backdrop');
    if (backdrop && backdrop.classList.contains('open')) closeApiModal();
  }
  if (e.key === 'c' || e.key === 'C') {
    showView('chat','',document.querySelector('.nav-link[onclick*=chat]'));
  }
});

// Keyboard shortcut: Ctrl+K or Cmd+K opens AI Chat panel
document.addEventListener('keydown', function(e) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault();
    var chatLink = document.querySelector('.nav-link[onclick*="chat"]');
    if (chatLink) chatLink.click();
    var chatInput = document.getElementById('chat-input');
    if (chatInput) { setTimeout(function(){ chatInput.focus(); }, 100); }
  }
});
</script>
</body>
</html>
""", headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


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
    if credentials and credentials.credentials:
        try:
            from app.core.auth import decode_token
            payload = decode_token(credentials.credentials)
            username = payload.get("sub")
            jwt_role = payload.get("role")  # trust role embedded in JWT
        except Exception:
            pass
    if not username and x_user:
        username = x_user.strip().lower()
    if not username:
        username = "anonymous"
    # JWT role takes precedence; fall back to RBAC lookup for X-User / anonymous
    role = jwt_role if jwt_role else get_user_role(username)
    return AuthContext(username=username, role=role)

def require_admin(auth: AuthContext = Depends(_resolve_auth)) -> AuthContext:
    if auth.role not in ("admin",):
        raise HTTPException(status_code=403, detail="Admin access required")
    return auth

def require_developer(auth: AuthContext = Depends(_resolve_auth)) -> AuthContext:
    if auth.role not in ("admin", "developer"):
        raise HTTPException(status_code=403, detail="Role 'developer' or 'admin' required")
    return auth

def require_viewer(auth: AuthContext = Depends(_resolve_auth)) -> AuthContext:
    if auth.role not in ("admin", "developer", "viewer"):
        raise HTTPException(status_code=403, detail="Authentication required")
    return auth

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

@app.post("/memory/incidents")
def memory_incident(incident: Event):
    record = store_incident(incident.model_dump())
    return {"stored": record}

@app.post("/security/check")
def security_check(req: AccessRequest):
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
def incident_run(req: IncidentRunRequest, x_user: Optional[str] = Header(default=None)):
    """End-to-end autonomous incident response pipeline.

    Collects AWS + K8s + GitHub observability data, runs AI root cause analysis,
    executes recommended actions (GitHub PR, Jira, Slack, OpsGenie, K8s ops),
    stores the incident in memory, and returns a full incident report.

    Requires X-User header with 'deploy' permission for auto_remediate=true.
    """
    if req.auto_remediate:
        _rbac_guard(x_user, "deploy")
    report = run_incident_pipeline(
        incident_id    = req.incident_id,
        description    = req.description,
        severity       = req.severity,
        aws_cfg        = req.aws.model_dump() if req.aws else {},
        k8s_cfg        = req.k8s.model_dump() if req.k8s else {},
        auto_remediate = req.auto_remediate,
        hours          = req.hours,
    )
    return report


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

@app.post("/v2/incident/run")
def incident_run_v2(req: IncidentRunV2Request,
                    x_user: Optional[str] = Header(default=None)):
    """LangGraph multi-agent incident pipeline (v2).

    Runs: Context Collection → PlannerAgent → DecisionAgent →
          Executor (policy-gated) → Validator → MemoryAgent.

    Requires X-User header with 'deploy' permission when auto_remediate=true.
    """
    if req.auto_remediate:
        _rbac_guard(x_user, "deploy")
    result = run_pipeline_v2(
        incident_id    = req.incident_id,
        description    = req.description,
        auto_remediate = req.auto_remediate,
        metadata       = {
            "user":          req.user,
            "role":          req.role,
            "aws_cfg":       req.aws_cfg or {},
            "k8s_cfg":       req.k8s_cfg or {},
            "hours":         req.hours,
            "slack_channel": req.slack_channel,
        },
    )
    return result


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
def secrets_status(auth: AuthContext = Depends(require_viewer)):
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


def _detect_intent(message: str, force_provider: str = "") -> dict:
    """Use LLM to classify whether message is an action or a question."""
    import json as _json
    from app.llm.claude import _llm, _extract_json
    try:
        raw = _llm(_INTENT_SYSTEM, [{"role": "user", "content": message}],
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
        intent_data = _detect_intent(payload.message, force_prov)

        if intent_data.get("intent") == "action":
            action_name = intent_data.get("action", "")
            params      = intent_data.get("params", {})
            action_def  = _ACTION_CATALOGUE.get(action_name)

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
        context: dict = {}
        try:
            context = collect_all_context(hours=2)
        except Exception:
            pass
        reply = chat_devops(payload.message, history, context, force_provider=force_prov)

    used_provider = force_prov or _provider or "none"
    return {
        "reply":          reply,
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
def incidents_run_alias(req: IncidentRunRequest, x_user: Optional[str] = Header(default=None)):
    """Alias for /incident/run — matches the documented path."""
    return incident_run(req, x_user)

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
def aws_context_snapshot():
    """Full AWS observability snapshot."""
    result = collect_diagnosis_context()
    return {"aws_context": result}

@app.get("/aws/synthesize")
def aws_synthesize(incident_id: str = "snapshot", description: str = "infrastructure status review"):
    """AI synthesis of current AWS infrastructure state."""
    from app.llm.claude import synthesize_incident
    context = collect_diagnosis_context()
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
