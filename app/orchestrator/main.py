import re
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Header
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
from app.integrations.k8s_ops import restart_deployment, scale_deployment, get_pod_logs
from app.integrations.aws_ops import (
    list_ec2_instances, get_ec2_status_checks, get_ec2_console_output,
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

@app.middleware("http")
async def _rate_limit_middleware(request: Request, call_next):
    # Only rate-limit AI-heavy endpoints
    if request.url.path in ("/chat", "/incidents/run", "/warroom/create"):
        client_ip = request.client.host if request.client else "unknown"
        now = _time.time()
        window_start = now - _RATE_WINDOW
        _rate_store[client_ip] = [t for t in _rate_store[client_ip] if t > window_start]
        if len(_rate_store[client_ip]) >= _RATE_LIMIT:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again in a minute."})
        _rate_store[client_ip].append(now)
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
  <title>NexusOps — AI DevOps Intelligence</title>
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
    body[data-role="viewer"] .rbac-dev,.body[data-role="viewer"] .rbac-admin{display:none!important}
    /* developer: hide admin-only */
    body[data-role="developer"] .rbac-admin{display:none!important}
    /* generic hidden util */
    .rbac-dev,.rbac-admin{transition:opacity .2s}
  </style>
</head>
<body>

<nav class="sidebar">
  <div class="sb-logo">
    <div class="logo-mark">&#x26A1;</div>
    <div class="logo-text">
      <div class="name">NexusOps</div>
      <div class="tag">AI DevOps Intelligence</div>
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
        <div class="chat-empty-sub">Ask anything about your infrastructure. I have live access to AWS, Kubernetes, GitHub, Grafana and more &#x2014; I will never fabricate data.</div>
        <div class="chat-suggestions">
          <button class="chat-suggestion" onclick="sendSuggestion(this)">&#x1F4CA; Full infrastructure overview</button>
          <button class="chat-suggestion" onclick="sendSuggestion(this)">&#x1F6A8; Any alerts firing right now?</button>
          <button class="chat-suggestion" onclick="sendSuggestion(this)">&#x2638; Unhealthy pods or failed deployments?</button>
          <button class="chat-suggestion" onclick="sendSuggestion(this)">&#x1F4C8; Did a recent deploy cause this issue?</button>
          <button class="chat-suggestion" onclick="sendSuggestion(this)">&#x1F50D; My service is down &#x2014; find root cause</button>
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
      <div class="topbar-title">AI DevOps Intelligence Platform</div>
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
        <div class="mc-sub" id="m-role-perms">Loading...</div>
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
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/</span><span class="ep-desc">Dashboard UI</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/health</span><span class="ep-desc">Basic health check</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/health/full</span><span class="ep-desc">Full integration health</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/health/integrations</span><span class="ep-desc">Integration diagnostics</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/chat</span><span class="ep-desc">Conversational AI (LLM + live context)</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/ws</span><span class="ep-desc">WebSocket streaming chat</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/secrets/status</span><span class="ep-desc">Integration key status</span></div>
          <div class="ep-row rbac-admin" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/secrets</span><span class="ep-desc">Save/update credentials</span><span class="ep-lock">&#x1F512;</span></div>
          <div class="ep-row rbac-admin" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/security/roles</span><span class="ep-desc">List all RBAC roles</span><span class="ep-lock">&#x1F512;</span></div>
          <div class="ep-row rbac-admin" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/security/roles/assign</span><span class="ep-desc">Assign role to user</span><span class="ep-lock">&#x1F512;</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/warroom/create</span><span class="ep-desc">AI war room + Slack channel</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/grafana/alerts</span><span class="ep-desc">Firing Grafana alerts</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/grafana/dashboards</span><span class="ep-desc">List dashboards</span></div>
        </div>
      </div>

      <div data-section="pipeline" class="rbac-dev">
        <div class="ep-group">
          <div class="ep-group-hdr"><span class="ico">&#x1F916;</span><span class="g-name">AI Pipeline (LangGraph)</span><span class="g-cnt">4</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/incidents/run</span><span class="ep-desc">Run full autonomous pipeline</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/incidents/run/async</span><span class="ep-desc">Async pipeline with job ID</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/incidents/{id}</span><span class="ep-desc">Get incident status</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/v2/incident/run</span><span class="ep-desc">V2 pipeline endpoint</span></div>
        </div>
      </div>

      <div data-section="webhooks" class="rbac-admin">
        <div class="ep-group">
          <div class="ep-group-hdr"><span class="ico">&#x1F517;</span><span class="g-name">Webhooks (Event-Driven)</span><span class="g-cnt">2</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/webhooks/github</span><span class="ep-desc">GitHub push / PR events</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/webhooks/pagerduty</span><span class="ep-desc">PagerDuty incident trigger</span></div>
        </div>
      </div>

      <div data-section="k8s">
        <div class="ep-group">
          <div class="ep-group-hdr"><span class="ico">&#x2638;</span><span class="g-name">Kubernetes</span><span class="g-cnt">7</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/k8s/health</span><span class="ep-desc">Cluster health overview</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/k8s/pods</span><span class="ep-desc">List pods with status</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/k8s/deployments</span><span class="ep-desc">Deployment readiness</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/k8s/logs/{ns}/{pod}</span><span class="ep-desc">Pod logs</span></div>
          <div class="ep-row rbac-dev" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/k8s/restart</span><span class="ep-desc">Rolling restart deployment</span><span class="ep-lock">&#x1F512;</span></div>
          <div class="ep-row rbac-dev" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/k8s/scale</span><span class="ep-desc">Scale deployment replicas</span><span class="ep-lock">&#x1F512;</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/k8s/diagnose</span><span class="ep-desc">AI K8s diagnosis</span></div>
        </div>
      </div>

      <div data-section="aws">
        <div class="ep-group">
          <div class="ep-group-hdr"><span class="ico">&#x2601;</span><span class="g-name">AWS</span><span class="g-cnt">23</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/ec2/instances</span><span class="ep-desc">EC2 instances</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/ecs/services</span><span class="ep-desc">ECS services</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/lambda/functions</span><span class="ep-desc">Lambda functions</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/cloudwatch/alarms</span><span class="ep-desc">CloudWatch alarms</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/cloudwatch/logs</span><span class="ep-desc">Log streams</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/rds/instances</span><span class="ep-desc">RDS instances</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/s3/buckets</span><span class="ep-desc">S3 buckets</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/sqs/queues</span><span class="ep-desc">SQS queues</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/dynamodb/tables</span><span class="ep-desc">DynamoDB tables</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/cloudtrail/events</span><span class="ep-desc">CloudTrail events</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/route53/health</span><span class="ep-desc">Route53 health checks</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/sns/topics</span><span class="ep-desc">SNS topics</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/aws/diagnose</span><span class="ep-desc">AI AWS root cause analysis</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/aws/predict-scaling</span><span class="ep-desc">AI scaling prediction</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/ec2/console/{id}</span><span class="ep-desc">EC2 console output</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/ecs/stopped-tasks</span><span class="ep-desc">Stopped ECS tasks</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/lambda/errors/{fn}</span><span class="ep-desc">Lambda error stats</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/rds/events/{id}</span><span class="ep-desc">RDS events</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/cloudwatch/metrics</span><span class="ep-desc">CloudWatch metrics query</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/cost/summary</span><span class="ep-desc">Cost explorer summary</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/aws/assess-deployment</span><span class="ep-desc">Pre-deploy risk gate</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/context</span><span class="ep-desc">Full AWS context snapshot</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/aws/synthesize</span><span class="ep-desc">AI incident synthesis</span></div>
        </div>
      </div>

      <div data-section="deploy">
        <div class="ep-group">
          <div class="ep-group-hdr"><span class="ico">&#x1F680;</span><span class="g-name">Deploy, GitHub &amp; Jira</span><span class="g-cnt">7</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/github/commits</span><span class="ep-desc">Recent commits</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/github/prs</span><span class="ep-desc">Recent pull requests</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge GET">GET</span><span class="ep-path">/github/pr/{n}/review</span><span class="ep-desc">AI PR code review</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/github/issue</span><span class="ep-desc">Create GitHub issue</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/jira/incident</span><span class="ep-desc">Create Jira ticket</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/deploy/assess</span><span class="ep-desc">Pre-deploy AI risk gate</span></div>
          <div class="ep-row" data-ep onclick="copyPath(this)"><span class="badge POST">POST</span><span class="ep-path">/deploy/jira-to-pr</span><span class="ep-desc">Jira ticket &#x2192; GitHub PR plan</span></div>
        </div>
      </div>

      <div data-section="pipeline">
        <div class="ep-group" style="border-color:rgba(79,142,247,.25)">
          <div class="pipeline-hdr"><span class="ico" style="font-size:16px">&#x1F916;</span><span class="g-name" style="color:#93c5fd">LangGraph Pipeline Flow</span><span class="pipeline-badge">7-Stage</span></div>
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
          <input class="sec-user-input" id="sec-user" type="text" value="nagaraj" placeholder="username"/>
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

    </div><!-- /view-secrets -->

    <div class="footer">
      <span>&#x26A1; NexusOps</span>
      <span style="opacity:.3">&#x2022;</span>
      <a href="/docs" target="_blank">Swagger</a>
      <span style="opacity:.3">&#x2022;</span>
      <a href="/redoc" target="_blank">ReDoc</a>
      <span style="opacity:.3">&#x2022;</span>
      <a href="/health/full" target="_blank">Health</a>
      <span style="opacity:.3">&#x2022;</span>
      <span>v2.0.0 &#x2022; LangGraph &#x2022; Multi-LLM</span>
    </div>
  </div><!-- /content-wrap -->

</div><!-- /main -->

<!-- ── TOAST CONTAINER ── -->
<div id="toast-wrap"></div>

<!-- ── RUN PIPELINE MODAL ── -->
<div class="modal-overlay" id="run-modal" onclick="if(event.target===this)this.classList.remove('open')">
  <div class="modal">
    <div class="modal-title">&#x1F916; Run AI Pipeline</div>
    <div class="modal-sub">Trigger the full LangGraph autonomous incident response pipeline</div>
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
  console.log('[showView] view='+view);
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

function copyPath(row) {
  var path = row.querySelector('.ep-path');
  if (!path) return;
  navigator.clipboard.writeText(path.textContent).then(function() {
    var orig = path.style.color;
    path.style.color = 'var(--green)';
    setTimeout(function(){ path.style.color = orig; }, 900);
    toast('Copied: ' + path.textContent, 'ok', 1800);
  });
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
  var user = document.getElementById('sec-user').value.trim() || 'nagaraj';
  var btn = document.getElementById('save-btn');
  btn.disabled = true; btn.textContent = 'Saving...';
  fetch('/secrets', {
    method: 'POST',
    headers: {'Content-Type':'application/json','X-User': user},
    body: JSON.stringify({secrets: secrets})
  }).then(function(r){ return r.json().then(function(d){ return {ok: r.ok, data: d}; }); })
  .then(function(res) {
    if (res.ok) {
      showMsg('Saved ' + res.data.updated.length + ' secret(s)', 'ok');
      ALL_KEYS.forEach(function(k){ var el=document.getElementById(k); if(el) el.value=''; });
      loadMetrics();
    } else { showMsg(res.data.detail || 'Error', 'err'); }
  }).catch(function(){ showMsg('Network error', 'err'); })
  .finally(function(){ btn.disabled=false; btn.textContent='\\u1F4BE Save to .env'; });
}
function showMsg(text, type) {
  var m = document.getElementById('sec-msg');
  m.textContent = text; m.className = 'sec-msg ' + type; m.style.display = '';
  setTimeout(function(){ m.style.display='none'; }, 4000);
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
  document.getElementById('chat-input').value = btn.textContent.replace(/^[\\u{1F300}-\\u{1FFFF}]\\s*/u, '');
  sendChat();
}

function chatKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
}
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 130) + 'px';
}

function sendChat() {
  var input = document.getElementById('chat-input');
  var msg = input.value.trim();
  if (!msg) return;
  var btn = document.getElementById('chat-send-btn');
  input.value = ''; input.style.height = 'auto';
  btn.disabled = true;
  appendMsg('user', msg);
  _chatHistory.push({role: 'user', content: msg});
  var typingBubble = appendMsg('ai', '<span class="typing-dots"><span>.</span><span>.</span><span>.</span></span>', true);
  typingBubble.classList.add('typing');
  var selProvider = (document.getElementById('llm-selector') || {}).value || '';
  fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: msg, history: _chatHistory.slice(0,-1), provider: selProvider})
  })
  .then(function(r){ return r.json(); })
  .then(function(d) {
    typingBubble.classList.remove('typing');
    var reply = d.reply || d.detail || 'No response';
    typingBubble.innerHTML = _renderMarkdown(reply);
    _chatHistory.push({role: 'assistant', content: reply});
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
loadRole('nagaraj');

/* ── RBAC ROLE LOADER ── */
var _currentUser = 'nagaraj';
var _currentRole = 'admin';

function loadRole(user) {
  _currentUser = user || 'nagaraj';
  fetch('/auth/me?user=' + encodeURIComponent(_currentUser))
    .then(function(r){ return r.json(); })
    .then(function(d) {
      _currentRole = d.role || 'viewer';
      applyRoleUI(d);
    }).catch(function(){});
}

function applyRoleUI(d) {
  var role = d.role || 'viewer';
  var perms = d.permissions || [];
  // Set body data-role for CSS visibility rules
  document.body.dataset.role = role;
  // Update sidebar footer
  var badge = document.getElementById('sb-role-badge');
  if (badge) {
    badge.textContent = role;
    badge.className = 'role-badge role-' + role;
  }
  var uname = document.getElementById('sb-uname');
  if (uname) uname.textContent = d.user || _currentUser;
  var avatar = document.getElementById('sb-avatar');
  if (avatar) avatar.textContent = (d.user || _currentUser)[0].toUpperCase();
  // Update role metric card
  var mRole = document.getElementById('m-role');
  if (mRole) mRole.textContent = role.charAt(0).toUpperCase() + role.slice(1);
  var mRolePerms = document.getElementById('m-role-perms');
  if (mRolePerms) mRolePerms.textContent = perms.length ? perms.join(', ') : 'read only';
  // Sync user field in secrets panel
  var secUser = document.getElementById('sec-user');
  if (secUser) secUser.value = d.user || _currentUser;
  toast('Signed in as ' + (d.user || _currentUser) + ' (' + role + ')', 'info', 2000);
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
  }
  if (e.key === 'c' || e.key === 'C') {
    showView('chat','',document.querySelector('.nav-link[onclick*=chat]'));
  }
});
</script>
</body>
</html>
""", headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@app.get("/auth/me")
def auth_me(user: str = "nagaraj"):
    """Return role and permissions for a user. Defaults to 'nagaraj' (admin)."""
    from app.security.rbac import ROLE_PERMISSIONS, _user_roles
    u = user.strip().lower()
    role = _user_roles.get(u, "viewer")
    perms = list(ROLE_PERMISSIONS.get(role, set()))
    return {"user": u, "role": role, "permissions": perms}


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
def secrets_status():
    """Return which env vars are configured (boolean only — never exposes values)."""
    status: Dict[str, Dict[str, bool]] = {}
    for group, keys in _SECRET_SCHEMA.items():
        status[group] = {k: bool(os.environ.get(k)) for k in keys}
    return status


@app.post("/secrets")
def secrets_update(payload: SecretsPayload, x_user: str = Header(...)):
    """Write secrets to .env file. Requires admin role."""
    _rbac_guard(x_user, "manage_secrets")
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
    provider: str = ""   # "anthropic" | "groq" | "ollama" | "" (auto)

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

@app.post("/chat")
def chat(payload: ChatPayload):
    """Conversational DevOps AI — collects live context from ALL integrations."""
    from app.llm.claude import _provider

    context: dict = {}
    context_error: str = ""
    try:
        context = collect_all_context(hours=2)
    except Exception as e:
        context_error = str(e)

    history = [{"role": m.role, "content": m.content} for m in payload.history]
    reply = chat_devops(payload.message, history, context,
                        force_provider=payload.provider or "")

    used_provider = payload.provider or _provider or "none"
    return {
        "reply":          reply,
        "sources":        context.get("configured", []),
        "llm_provider":   used_provider,
        "context_error":  context_error or None,
    }

@app.post("/warroom/create")
def warroom_create(req: WarRoomRequest):
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
