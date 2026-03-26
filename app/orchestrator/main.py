import re
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Any, Optional, Dict

from app.correlation.engine import correlate_events
from app.plugins.aws_checker import check_aws_infrastructure
from app.plugins.linux_checker import check_linux_node
from app.plugins.k8s_checker import check_k8s_cluster, check_k8s_nodes, check_k8s_pods, check_k8s_deployments
from app.integrations.github import get_recent_commits as _get_recent_commits
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
from app.integrations.slack import create_war_room, create_incident_channel, post_incident_summary, post_message
from app.integrations.universal_collector import collect_all_context, summarize_health
from app.integrations.jira import create_incident, add_comment as jira_add_comment
from app.integrations.opsgenie import notify_on_call
from app.integrations.github import (
    create_issue, create_pull_request,
    get_pr_for_review, post_pr_review_comment,
    create_incident_pr,
)
from app.integrations.vscode import trigger_code_action, open_file_in_vscode
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

app = FastAPI(
    title="AI DevOps Intelligence Platform",
    description="Autonomous DevOps management powered by multi-agent AI — built by Nagaraj",
    version="1.0.0",
)

_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>AI DevOps Platform</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <style>
    *,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
    :root{
      --bg:#0d1117;--bg2:#010409;--surface:#161b22;--surface2:#21262d;
      --border:#30363d;--border2:#484f58;
      --text:#e6edf3;--text2:#8b949e;--muted:#484f58;
      --blue:#58a6ff;--green:#3fb950;--purple:#bc8cff;--red:#ff7b72;--amber:#d29922;
    }
    html{scroll-behavior:smooth}
    body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex}
    ::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
    .sidebar{width:215px;flex-shrink:0;background:var(--bg2);border-right:1px solid var(--border);display:flex;flex-direction:column;height:100vh;position:sticky;top:0;overflow-y:auto}
    .sb-header{padding:14px 14px 12px;border-bottom:1px solid var(--border)}
    .logo{display:flex;align-items:center;gap:9px}
    .logo-mark{width:28px;height:28px;background:#1f6feb;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0}
    .logo-name{font-size:13px;font-weight:700}
    .logo-tag{font-size:10px;color:var(--text2);margin-top:1px}
    .sb-section{padding:10px 8px 4px}
    .sb-label{font-size:10px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);padding:0 6px 5px;display:block}
    .nav-link{display:flex;align-items:center;gap:7px;padding:5px 6px;border-radius:5px;font-size:12.5px;font-weight:500;color:var(--text2);cursor:pointer;border:none;background:transparent;width:100%;text-align:left;text-decoration:none;transition:background .1s,color .1s}
    .nav-link:hover{background:var(--surface2);color:var(--text)}
    .nav-link.active{background:rgba(88,166,255,.1);color:var(--blue)}
    .nav-link .ico{width:15px;font-size:12px;text-align:center;flex-shrink:0}
    .nav-link .cnt{margin-left:auto;font-size:10px;background:var(--surface2);padding:1px 5px;border-radius:999px;color:var(--muted)}
    .sb-footer{margin-top:auto;padding:10px 14px;border-top:1px solid var(--border)}
    .user-row{display:flex;align-items:center;gap:8px}
    .avatar{width:24px;height:24px;border-radius:50%;background:#1f6feb;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;color:#fff;flex-shrink:0}
    .user-name{font-size:12px;font-weight:600}
    .user-role{font-size:10px;color:var(--text2)}
    .main{flex:1;min-width:0;display:flex;flex-direction:column;height:100vh;overflow-y:auto}
    .topbar{background:var(--bg2);border-bottom:1px solid var(--border);padding:8px 20px;display:flex;align-items:center;gap:10px;flex-shrink:0;position:sticky;top:0;z-index:10}
    .status-dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2.5s ease-in-out infinite;flex-shrink:0}
    @keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(63,185,80,.4)}50%{box-shadow:0 0 0 5px rgba(63,185,80,0)}}
    .status-lbl{font-size:11px;color:var(--green);font-weight:600}
    .topbar-title{font-size:12.5px;font-weight:600;color:var(--text2);margin-left:6px}
    .tb-spacer{flex:1}
    .tb-btn{display:flex;align-items:center;gap:4px;padding:4px 9px;border-radius:5px;background:var(--surface2);border:1px solid var(--border);font-size:11px;font-weight:500;color:var(--text2);text-decoration:none;transition:all .12s;cursor:pointer}
    .tb-btn:hover{background:var(--surface);border-color:var(--border2);color:var(--text)}
    .tb-btn.pri{background:#1f6feb;border-color:#388bfd;color:#fff}
    .tb-btn.pri:hover{background:#388bfd}
    .content{padding:18px 20px;flex:1}
    .stats-row{display:flex;gap:8px;margin-bottom:16px}
    .stat-chip{flex:1;background:var(--surface);border:1px solid var(--border);border-radius:7px;padding:10px 12px;display:flex;align-items:center;gap:9px}
    .stat-ico{font-size:15px;flex-shrink:0}
    .stat-val{font-size:17px;font-weight:700;line-height:1}
    .stat-lbl{font-size:10px;color:var(--text2);margin-top:2px;font-weight:500}
    .int-row{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:16px}
    .int-chip{display:flex;align-items:center;gap:5px;background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:3px 8px;font-size:11px;font-weight:500;color:var(--text2)}
    .int-chip.on{border-color:rgba(63,185,80,.3);color:#7ee787}
    .int-dot{width:5px;height:5px;border-radius:50%;background:var(--muted)}
    .int-chip.on .int-dot{background:var(--green);box-shadow:0 0 4px var(--green)}
    .search-wrap{position:relative;margin-bottom:14px}
    .search-wrap svg{position:absolute;left:9px;top:50%;transform:translateY(-50%);color:var(--muted);pointer-events:none}
    #search{width:100%;background:var(--surface);border:1px solid var(--border);border-radius:5px;padding:6px 9px 6px 30px;color:var(--text);font-size:12px;font-family:'Inter',sans-serif;outline:none;transition:border-color .15s}
    #search:focus{border-color:#1f6feb}
    #search::placeholder{color:var(--muted)}
    .ep-group{background:var(--surface);border:1px solid var(--border);border-radius:7px;overflow:hidden;margin-bottom:8px}
    .ep-group-hdr{display:flex;align-items:center;gap:7px;padding:8px 12px;background:var(--surface2);border-bottom:1px solid var(--border)}
    .g-ico{font-size:13px}
    .g-name{font-size:12px;font-weight:600;color:var(--text);flex:1}
    .g-cnt{font-size:10px;color:var(--muted)}
    .ep-row{display:flex;align-items:center;gap:9px;padding:7px 12px;text-decoration:none;color:inherit;border-top:1px solid var(--border);transition:background .1s}
    .ep-row:hover{background:rgba(255,255,255,.025)}
    .ep-row:hover .ep-path{color:var(--blue)}
    .method{font-size:9px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;padding:2px 5px;border-radius:3px;min-width:34px;text-align:center;flex-shrink:0}
    .get{background:rgba(63,185,80,.1);color:#7ee787;border:1px solid rgba(63,185,80,.2)}
    .post{background:rgba(188,140,255,.1);color:#d2a8ff;border:1px solid rgba(188,140,255,.2)}
    .del{background:rgba(255,123,114,.1);color:#ffa198;border:1px solid rgba(255,123,114,.2)}
    .ws{background:rgba(210,153,34,.1);color:#e3b341;border:1px solid rgba(210,153,34,.2)}
    .ep-path{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--text2);flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;transition:color .1s}
    .ep-desc{font-size:11px;color:var(--muted);flex:1.5;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .ep-lock{font-size:9px;opacity:.45;flex-shrink:0}
    .ep-arrow{font-size:11px;color:var(--muted);opacity:0;transition:opacity .1s;flex-shrink:0}
    .ep-row:hover .ep-arrow{opacity:1}
    .pipeline-card{background:var(--surface);border:1px solid rgba(31,111,235,.4);border-radius:7px;overflow:hidden;margin-bottom:8px}
    .pipeline-hdr{display:flex;align-items:center;gap:7px;padding:8px 12px;background:rgba(31,111,235,.08);border-bottom:1px solid rgba(31,111,235,.25)}
    .pipeline-hdr .g-name{color:#79c0ff}
    .flow-row{display:flex;align-items:center;padding:12px;gap:0;overflow-x:auto;border-bottom:1px solid var(--border)}
    .fstep{flex:1;min-width:80px;background:var(--surface2);border:1px solid var(--border);border-radius:5px;padding:7px 8px;text-align:center}
    .fstep-ic{font-size:15px;display:block;margin-bottom:3px}
    .fstep-lb{font-size:10px;font-weight:600;color:var(--text)}
    .fstep-sb{font-size:9px;color:var(--muted);margin-top:1px}
    .farr{padding:0 6px;color:var(--muted);font-size:13px;flex-shrink:0}
    pre.sample{margin:0;padding:9px 12px;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text2);line-height:1.65;overflow-x:auto;background:var(--bg2);white-space:pre-wrap}
    /* SECRETS PANEL */
    .secrets-panel{display:none}
    .secrets-panel.active{display:block}
    .sec-group{background:var(--surface);border:1px solid var(--border);border-radius:7px;overflow:hidden;margin-bottom:8px}
    .sec-group-hdr{display:flex;align-items:center;gap:7px;padding:8px 12px;background:var(--surface2);border-bottom:1px solid var(--border)}
    .sec-row{display:flex;align-items:center;gap:10px;padding:7px 12px;border-top:1px solid var(--border)}
    .sec-key{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text2);width:220px;flex-shrink:0}
    .sec-input{flex:1;background:var(--bg2);border:1px solid var(--border);border-radius:4px;padding:4px 8px;font-size:11px;font-family:'JetBrains Mono',monospace;color:var(--text);outline:none;transition:border-color .15s}
    .sec-input:focus{border-color:var(--blue)}
    .sec-input::placeholder{color:var(--muted)}
    .sec-status{width:14px;flex-shrink:0;font-size:11px}
    .sec-actions{display:flex;align-items:center;gap:8px;margin-bottom:14px;padding:10px 12px;background:var(--surface);border:1px solid var(--border);border-radius:7px}
    .sec-user-wrap{display:flex;flex-direction:column;gap:3px;flex:1}
    .sec-user-lbl{font-size:10px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.07em}
    .sec-user-input{background:var(--bg2);border:1px solid var(--border);border-radius:4px;padding:4px 8px;font-size:11px;font-family:'JetBrains Mono',monospace;color:var(--text);outline:none;transition:border-color .15s;width:160px}
    .sec-user-input:focus{border-color:var(--blue)}
    .save-btn{padding:5px 14px;background:#1f6feb;border:1px solid #388bfd;border-radius:5px;font-size:11px;font-weight:600;color:#fff;cursor:pointer;transition:background .12s;flex-shrink:0}
    .save-btn:hover{background:#388bfd}
    .save-btn:disabled{background:var(--surface2);border-color:var(--border);color:var(--muted);cursor:not-allowed}
    .sec-msg{font-size:11px;font-weight:500;padding:3px 8px;border-radius:4px;flex-shrink:0}
    .sec-msg.ok{background:rgba(63,185,80,.1);color:#7ee787;border:1px solid rgba(63,185,80,.2)}
    .sec-msg.err{background:rgba(255,123,114,.1);color:#ffa198;border:1px solid rgba(255,123,114,.2)}
    .footer{padding:10px 20px;border-top:1px solid var(--border);text-align:center;font-size:11px;color:var(--muted);flex-shrink:0}
    .footer span{color:var(--text2)}
    .inc-result-hdr{display:flex;align-items:center;gap:7px;padding:8px 12px;background:var(--surface2);border-bottom:1px solid var(--border)}
    .inc-result-hdr .g-name{font-size:12px;font-weight:600;flex:1}
    .inc-result-hdr.ok .g-name{color:#7ee787}
    .inc-result-hdr.err .g-name{color:#ffa198}
    .inc-result-body{padding:12px;font-size:12px;line-height:1.7;color:var(--text2)}
    .inc-result-body h4{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin:12px 0 4px}
    .inc-result-body h4:first-child{margin-top:0}
    .inc-result-body p{color:var(--text);margin-bottom:4px}
    .inc-action-item{display:flex;align-items:flex-start;gap:7px;padding:4px 0;border-top:1px solid var(--border);margin-top:4px}
    .inc-action-type{font-size:9px;font-weight:700;text-transform:uppercase;padding:2px 5px;border-radius:3px;background:rgba(88,166,255,.12);color:var(--blue);border:1px solid rgba(88,166,255,.2);flex-shrink:0;margin-top:2px}
    .inc-action-desc{font-size:11px;color:var(--text2)}
    pre.inc-pre{background:var(--bg2);border:1px solid var(--border);border-radius:4px;padding:8px 10px;font-size:11px;font-family:'JetBrains Mono',monospace;color:var(--text2);overflow-x:auto;white-space:pre-wrap;margin-top:4px}
    /* CHAT PANEL */
    .chat-panel{display:none;flex-direction:column;flex:1;min-height:0}
    .chat-panel.active{display:flex}
    .chat-messages{flex:1;overflow-y:auto;padding:16px 20px;display:flex;flex-direction:column;gap:12px}
    .chat-messages::-webkit-scrollbar{width:4px}
    .chat-messages::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
    .chat-bubble{max-width:78%;padding:9px 13px;border-radius:10px;font-size:12.5px;line-height:1.65;word-break:break-word;white-space:pre-wrap}
    .chat-bubble.user{background:#1f6feb;color:#fff;align-self:flex-end;border-bottom-right-radius:3px}
    .chat-bubble.ai{background:var(--surface);border:1px solid var(--border);color:var(--text);align-self:flex-start;border-bottom-left-radius:3px}
    .chat-bubble.typing{color:var(--muted);font-style:italic}
    .chat-meta{font-size:10px;color:var(--muted);margin-bottom:2px}
    .chat-meta.right{text-align:right}
    .chat-row{display:flex;flex-direction:column}
    .chat-row.user{align-items:flex-end}
    .chat-row.ai{align-items:flex-start}
    .chat-input-bar{padding:12px 16px;border-top:1px solid var(--border);background:var(--bg2);display:flex;gap:8px;align-items:flex-end;flex-shrink:0}
    #chat-input{flex:1;background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:8px 11px;font-size:12.5px;font-family:'Inter',sans-serif;color:var(--text);outline:none;resize:none;max-height:120px;min-height:36px;line-height:1.5;transition:border-color .15s}
    #chat-input:focus{border-color:#1f6feb}
    #chat-input::placeholder{color:var(--muted)}
    .chat-send-btn{padding:7px 14px;background:#1f6feb;border:1px solid #388bfd;border-radius:6px;font-size:12px;font-weight:600;color:#fff;cursor:pointer;transition:background .12s;flex-shrink:0}
    .chat-send-btn:hover{background:#388bfd}
    .chat-send-btn:disabled{background:var(--surface2);border-color:var(--border);color:var(--muted);cursor:not-allowed}
    .chat-warroom-btn{padding:7px 11px;background:var(--surface2);border:1px solid var(--border);border-radius:6px;font-size:11px;font-weight:600;color:var(--amber);cursor:pointer;transition:all .12s;flex-shrink:0}
    .chat-warroom-btn:hover{background:rgba(210,153,34,.15);border-color:var(--amber)}
    .chat-warroom-btn:disabled{opacity:.4;cursor:not-allowed}
    .chat-empty{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;color:var(--muted);text-align:center;padding:20px}
    .chat-empty-icon{font-size:36px;opacity:.4}
    .chat-empty-title{font-size:13px;font-weight:600;color:var(--text2)}
    .chat-empty-sub{font-size:11px;line-height:1.6;max-width:280px}
    .chat-suggestion{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:6px 12px;font-size:11px;color:var(--text2);cursor:pointer;transition:all .12s;margin:2px}
    .chat-suggestion:hover{background:var(--surface2);border-color:var(--border2);color:var(--text)}
    .chat-suggestions{display:flex;flex-wrap:wrap;justify-content:center;gap:4px;margin-top:8px}
    .hidden{display:none!important}
    @media(max-width:860px){.sidebar{display:none}.stats-row{flex-wrap:wrap}.stat-chip{min-width:calc(50% - 4px)}.ep-desc{display:none}}
  </style>
</head>
<body>

<nav class="sidebar">
  <div class="sb-header">
    <div class="logo">
      <div class="logo-mark">&#x1F916;</div>
      <div><div class="logo-name">AI DevOps</div><div class="logo-tag">Intelligence Platform</div></div>
    </div>
  </div>
  <div class="sb-section">
    <span class="sb-label">Navigation</span>
    <button type="button" class="nav-link active" onclick="showView('endpoints','all',this)" data-v="endpoints" data-s="all"><span class="ico">&#x26A1;</span>All Endpoints<span class="cnt">40+</span></button>
    <button type="button" class="nav-link" onclick="showView('endpoints','general',this)" data-v="endpoints" data-s="general"><span class="ico">&#x1F527;</span>General &amp; AI<span class="cnt">12</span></button>
    <button type="button" class="nav-link" onclick="showView('endpoints','k8s',this)" data-v="endpoints" data-s="k8s"><span class="ico">&#x2638;</span>Kubernetes<span class="cnt">7</span></button>
    <button type="button" class="nav-link" onclick="showView('endpoints','aws',this)" data-v="endpoints" data-s="aws"><span class="ico">&#x2601;</span>AWS<span class="cnt">16</span></button>
    <button type="button" class="nav-link" onclick="showView('endpoints','pipeline',this)" data-v="endpoints" data-s="pipeline"><span class="ico">&#x1F916;</span>Pipeline<span class="cnt">1</span></button>
    <button type="button" class="nav-link" onclick="showView('endpoints','deploy',this)" data-v="endpoints" data-s="deploy"><span class="ico">&#x1F680;</span>Deploy &amp; Jira<span class="cnt">4</span></button>
  </div>
  <div class="sb-section">
    <span class="sb-label">Config</span>
    <button type="button" class="nav-link" onclick="showView('secrets','',this)" data-v="secrets"><span class="ico">&#x1F511;</span>Secrets / Env Vars</button>
    <button type="button" class="nav-link" onclick="showView('chat','',this)" data-v="chat"><span class="ico">&#x1F4AC;</span>AI Chat</button>
  </div>
  <div class="sb-section">
    <span class="sb-label">Links</span>
    <a class="nav-link" href="/docs" target="_blank"><span class="ico">&#x1F4D6;</span>Swagger UI</a>
    <a class="nav-link" href="/redoc" target="_blank"><span class="ico">&#x1F4C4;</span>ReDoc</a>
    <a class="nav-link" href="/health" target="_blank"><span class="ico">&#x1F49A;</span>Health Check</a>
  </div>
  <div class="sb-footer">
    <div class="user-row">
      <div class="avatar">N</div>
      <div><div class="user-name">Nagaraj</div><div class="user-role">Platform Owner</div></div>
    </div>
  </div>
</nav>

<div class="main">
  <div class="topbar">
    <div class="status-dot"></div>
    <span class="status-lbl">System Online</span>
    <span class="topbar-title">AI DevOps Intelligence Platform</span>
    <div class="tb-spacer"></div>
    <a class="tb-btn pri" href="/docs" target="_blank">&#x1F4D6; Swagger</a>
    <a class="tb-btn" href="/health" target="_blank">Health</a>
    <a class="tb-btn" href="/redoc" target="_blank">ReDoc</a>
  </div>

  <div class="content" id="content-wrap">

    <div id="view-endpoints">
      <div class="stats-row">
        <div class="stat-chip"><span class="stat-ico">&#x1F50C;</span><div><div class="stat-val">40+</div><div class="stat-lbl">API Endpoints</div></div></div>
        <div class="stat-chip"><span class="stat-ico">&#x1F517;</span><div><div class="stat-val">7</div><div class="stat-lbl">Integrations</div></div></div>
        <div class="stat-chip"><span class="stat-ico">&#x1F6E1;</span><div><div class="stat-val">3</div><div class="stat-lbl">RBAC Roles</div></div></div>
        <div class="stat-chip"><span class="stat-ico">&#x2705;</span><div><div class="stat-val">68</div><div class="stat-lbl">Tests Passing</div></div></div>
      </div>
      <div class="int-row" id="int-row">
        <div class="int-chip" id="int-claude"><span class="int-dot"></span>Claude AI</div>
        <div class="int-chip" id="int-aws"><span class="int-dot"></span>AWS</div>
        <div class="int-chip" id="int-grafana"><span class="int-dot"></span>Grafana</div>
        <div class="int-chip" id="int-github"><span class="int-dot"></span>GitHub</div>
        <div class="int-chip" id="int-gitlab"><span class="int-dot"></span>GitLab</div>
        <div class="int-chip" id="int-k8s"><span class="int-dot"></span>Kubernetes</div>
        <div class="int-chip" id="int-slack"><span class="int-dot"></span>Slack</div>
        <div class="int-chip" id="int-jira"><span class="int-dot"></span>Jira</div>
        <div class="int-chip" id="int-opsgenie"><span class="int-dot"></span>OpsGenie</div>
      </div>
      <div class="search-wrap">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>
        <input id="search" type="text" placeholder="Search endpoints&#x2026;" oninput="filterEps(this.value)"/>
      </div>
      <div id="ep-content">
        <div class="ep-group" data-section="general">
          <div class="ep-group-hdr"><span class="g-ico">&#x1F527;</span><span class="g-name">General &amp; AI</span><span class="g-cnt">12 endpoints</span></div>
          <a class="ep-row" href="/health" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/health</span><span class="ep-desc">System health status</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/docs</span><span class="ep-desc">Interactive Swagger UI</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/redoc" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/redoc</span><span class="ep-desc">ReDoc API reference</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/correlate_correlate_post" target="_blank" data-ep><span class="method post">POST</span><span class="ep-path">/correlate</span><span class="ep-desc">Correlate events &amp; find patterns</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/llm_analyze_llm_analyze_post" target="_blank" data-ep><span class="method post">POST</span><span class="ep-path">/llm/analyze</span><span class="ep-desc">Claude AI root cause analysis</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/check/aws" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/check/aws</span><span class="ep-desc">AWS infrastructure health</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/check/linux" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/check/linux</span><span class="ep-desc">Linux node health</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/incident_war_room_incident_war_room_post" target="_blank" data-ep><span class="method post">POST</span><span class="ep-path">/incident/war-room</span><span class="ep-desc">Create Slack war room</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/incident_jira_incident_jira_post" target="_blank" data-ep><span class="method post">POST</span><span class="ep-path">/incident/jira</span><span class="ep-desc">Create Jira incident ticket</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/memory_incident_memory_incidents_post" target="_blank" data-ep><span class="method post">POST</span><span class="ep-path">/memory/incidents</span><span class="ep-desc">Store incident in ChromaDB</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/security_check_security_check_post" target="_blank" data-ep><span class="method post">POST</span><span class="ep-path">/security/check</span><span class="ep-desc">RBAC access check</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/security_assign_role_security_roles_post" target="_blank" data-ep><span class="method post">POST</span><span class="ep-path">/security/roles</span><span class="ep-desc">Assign role to user</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/websocket_events_realtime_events_ws" target="_blank" data-ep><span class="method ws">WS</span><span class="ep-path">/realtime/events</span><span class="ep-desc">Live event stream</span><span class="ep-arrow">&#x2197;</span></a>
        </div>
        <div class="ep-group" data-section="k8s">
          <div class="ep-group-hdr"><span class="g-ico">&#x2638;</span><span class="g-name">Kubernetes</span><span class="g-cnt">7 endpoints</span></div>
          <a class="ep-row" href="/check/k8s" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/check/k8s</span><span class="ep-desc">Cluster summary</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/check/k8s/nodes" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/check/k8s/nodes</span><span class="ep-desc">Per-node ready status</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/check/k8s/pods" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/check/k8s/pods</span><span class="ep-desc">Pod health by namespace</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/check/k8s/deployments" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/check/k8s/deployments</span><span class="ep-desc">Deployment rollout status</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/k8s_restart_k8s_restart_post" target="_blank" data-ep><span class="method post">POST</span><span class="ep-path">/k8s/restart</span><span class="ep-desc">Rolling restart &#x2014; requires X-User</span><span class="ep-lock">&#x1F512;</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/k8s_scale_k8s_scale_post" target="_blank" data-ep><span class="method post">POST</span><span class="ep-path">/k8s/scale</span><span class="ep-desc">Scale replicas &#x2014; requires X-User</span><span class="ep-lock">&#x1F512;</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/k8s/logs?namespace=default&pod=my-pod&tail_lines=50" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/k8s/logs</span><span class="ep-desc">Fetch pod logs</span><span class="ep-arrow">&#x2197;</span></a>
        </div>
        <div class="ep-group" data-section="aws">
          <div class="ep-group-hdr"><span class="g-ico">&#x2601;</span><span class="g-name">AWS Observability &amp; AI Diagnosis</span><span class="g-cnt">21 endpoints</span></div>
          <a class="ep-row" href="/aws/ec2/instances" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/aws/ec2/instances</span><span class="ep-desc">List EC2 instances</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/aws_ec2_status_aws_ec2_status_get" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/aws/ec2/status</span><span class="ep-desc">Status checks</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/aws_ec2_console_aws_ec2_console_get" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/aws/ec2/console</span><span class="ep-desc">Serial console output</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/aws/logs/groups" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/aws/logs/groups</span><span class="ep-desc">CloudWatch log groups</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/aws_logs_recent_aws_logs_recent_get" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/aws/logs/recent</span><span class="ep-desc">Recent log events</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/aws_logs_search_aws_logs_search_get" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/aws/logs/search</span><span class="ep-desc">Search logs by pattern</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/aws/cloudwatch/alarms" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/aws/cloudwatch/alarms</span><span class="ep-desc">Firing alarms</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/aws_cw_metrics_aws_cloudwatch_metrics_post" target="_blank" data-ep><span class="method post">POST</span><span class="ep-path">/aws/cloudwatch/metrics</span><span class="ep-desc">Fetch metric series</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/aws/ecs/services" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/aws/ecs/services</span><span class="ep-desc">ECS running vs desired</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/aws/ecs/stopped-tasks" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/aws/ecs/stopped-tasks</span><span class="ep-desc">Stop reasons &amp; exit codes</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/aws_lambda_errors_aws_lambda_errors_get" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/aws/lambda/errors</span><span class="ep-desc">Lambda error metrics</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/aws_rds_events_aws_rds_events_get" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/aws/rds/events</span><span class="ep-desc">RDS events</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/aws_elb_health_aws_elb_target_health_get" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/aws/elb/target-health</span><span class="ep-desc">ALB target health</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/aws/cloudtrail/events" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/aws/cloudtrail/events</span><span class="ep-desc">Recent API changes (CloudTrail)</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/aws_diagnose_aws_diagnose_post" target="_blank" data-ep><span class="method post">POST</span><span class="ep-path">/aws/diagnose</span><span class="ep-desc">AI root cause analysis from live AWS data</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/aws_predict_scaling_aws_predict_scaling_post" target="_blank" data-ep><span class="method post">POST</span><span class="ep-path">/aws/predict-scaling</span><span class="ep-desc">Predict scaling from CloudWatch trends</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/aws/s3/buckets" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/aws/s3/buckets</span><span class="ep-desc">List S3 buckets</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/aws/sqs/queues" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/aws/sqs/queues</span><span class="ep-desc">SQS queues &amp; message depths</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/aws/dynamodb/tables" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/aws/dynamodb/tables</span><span class="ep-desc">DynamoDB tables &amp; status</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/aws/route53/healthchecks" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/aws/route53/healthchecks</span><span class="ep-desc">Route53 health check statuses</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/aws/sns/topics" target="_blank" data-ep><span class="method get">GET</span><span class="ep-path">/aws/sns/topics</span><span class="ep-desc">SNS topics</span><span class="ep-arrow">&#x2197;</span></a>
        </div>
        <div class="pipeline-card" data-section="pipeline">
          <div class="pipeline-hdr"><span class="g-ico">&#x1F916;</span><span class="g-name">Autonomous Incident Pipeline</span><span class="g-cnt" style="color:var(--text2)">Flagship feature</span></div>
          <div class="flow-row">
            <div class="fstep"><span class="fstep-ic">&#x1F4E1;</span><div class="fstep-lb">Collect</div><div class="fstep-sb">AWS &#xB7; K8s &#xB7; GitHub</div></div>
            <div class="farr">&#x2192;</div>
            <div class="fstep"><span class="fstep-ic">&#x1F9E0;</span><div class="fstep-lb">Synthesise</div><div class="fstep-sb">Claude AI RCA</div></div>
            <div class="farr">&#x2192;</div>
            <div class="fstep"><span class="fstep-ic">&#x26A1;</span><div class="fstep-lb">Remediate</div><div class="fstep-sb">K8s &#xB7; PR &#xB7; Jira &#xB7; Slack</div></div>
            <div class="farr">&#x2192;</div>
            <div class="fstep"><span class="fstep-ic">&#x1F4CB;</span><div class="fstep-lb">Report</div><div class="fstep-sb">ChromaDB memory</div></div>
          </div>
          <a class="ep-row" href="/docs#/default/incident_run_incident_run_post" target="_blank" data-ep><span class="method post">POST</span><span class="ep-path">/incident/run</span><span class="ep-desc">Full autonomous pipeline &#x2014; collect &#x2192; analyse &#x2192; remediate &#x2192; report</span><span class="ep-lock">&#x1F512;</span><span class="ep-arrow">&#x2197;</span></a>
          <pre class="sample">POST /incident/run
{ "incident_id": "INC-001", "description": "High 5xx rate on API", "severity": "critical",
  "aws": { "resource_type": "ecs", "resource_id": "my-cluster" },
  "k8s": { "namespace": "production" }, "auto_remediate": true, "hours": 2 }</pre>
        </div>
        <div class="ep-group" data-section="deploy">
          <div class="ep-group-hdr"><span class="g-ico">&#x1F680;</span><span class="g-name">Deploy &amp; Jira Automation</span><span class="g-cnt">4 endpoints</span></div>
          <a class="ep-row" href="/docs#/default/deploy_assess_deploy_assess_post" target="_blank" data-ep><span class="method post">POST</span><span class="ep-path">/deploy/assess</span><span class="ep-desc">Pre-deploy risk assessment &#x2014; go / no-go</span><span class="ep-lock">&#x1F512;</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/jira_webhook_jira_webhook_post" target="_blank" data-ep><span class="method post">POST</span><span class="ep-path">/jira/webhook</span><span class="ep-desc">Jira change-request &#x2192; auto GitHub PR</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/github_review_pr_github_review_pr_post" target="_blank" data-ep><span class="method post">POST</span><span class="ep-path">/github/review-pr</span><span class="ep-desc">AI PR review &#x2014; security &amp; infra analysis</span><span class="ep-arrow">&#x2197;</span></a>
          <a class="ep-row" href="/docs#/default/aws_predict_scaling_aws_predict_scaling_post" target="_blank" data-ep><span class="method post">POST</span><span class="ep-path">/aws/predict-scaling</span><span class="ep-desc">Scale prediction from CloudWatch trends</span><span class="ep-arrow">&#x2197;</span></a>
        </div>
      </div>
    </div>

    <div id="view-secrets" class="secrets-panel">
      <div class="sec-actions">
        <div class="sec-user-wrap">
          <div class="sec-user-lbl">Admin User (X-User header)</div>
          <input id="sec-user" class="sec-user-input" type="text" placeholder="e.g. nagaraj" value="nagaraj"/>
        </div>
        <button class="save-btn" onclick="saveSecrets()" id="save-btn">Save to .env</button>
        <span id="sec-msg" class="sec-msg" style="display:none"></span>
      </div>
      <div class="sec-group">
        <div class="sec-group-hdr"><span class="g-ico">&#x1F916;</span><span class="g-name">Claude AI</span></div>
        <div class="sec-row"><span class="sec-key">ANTHROPIC_API_KEY</span><input class="sec-input" type="password" id="ANTHROPIC_API_KEY" placeholder="sk-ant-&#x2026;"/><span class="sec-status" id="st-ANTHROPIC_API_KEY"></span></div>
      </div>
      <div class="sec-group">
        <div class="sec-group-hdr"><span class="g-ico">&#x2601;</span><span class="g-name">AWS</span></div>
        <div class="sec-row"><span class="sec-key">AWS_ACCESS_KEY_ID</span><input class="sec-input" type="password" id="AWS_ACCESS_KEY_ID" placeholder="AKIA&#x2026;"/><span class="sec-status" id="st-AWS_ACCESS_KEY_ID"></span></div>
        <div class="sec-row"><span class="sec-key">AWS_SECRET_ACCESS_KEY</span><input class="sec-input" type="password" id="AWS_SECRET_ACCESS_KEY" placeholder="&#x2026;"/><span class="sec-status" id="st-AWS_SECRET_ACCESS_KEY"></span></div>
        <div class="sec-row"><span class="sec-key">AWS_DEFAULT_REGION</span><input class="sec-input" type="text" id="AWS_DEFAULT_REGION" placeholder="us-east-1"/><span class="sec-status" id="st-AWS_DEFAULT_REGION"></span></div>
      </div>
      <div class="sec-group">
        <div class="sec-group-hdr"><span class="g-ico">&#x1F4BB;</span><span class="g-name">GitHub</span></div>
        <div class="sec-row"><span class="sec-key">GITHUB_TOKEN</span><input class="sec-input" type="password" id="GITHUB_TOKEN" placeholder="ghp_&#x2026;"/><span class="sec-status" id="st-GITHUB_TOKEN"></span></div>
        <div class="sec-row"><span class="sec-key">GITHUB_REPO</span><input class="sec-input" type="text" id="GITHUB_REPO" placeholder="owner/repo"/><span class="sec-status" id="st-GITHUB_REPO"></span></div>
      </div>
      <div class="sec-group">
        <div class="sec-group-hdr"><span class="g-ico">&#x2638;</span><span class="g-name">Kubernetes</span></div>
        <div class="sec-row"><span class="sec-key">KUBECONFIG</span><input class="sec-input" type="text" id="KUBECONFIG" placeholder="/home/user/.kube/config"/><span class="sec-status" id="st-KUBECONFIG"></span></div>
      </div>
      <div class="sec-group">
        <div class="sec-group-hdr"><span class="g-ico">&#x1F4AC;</span><span class="g-name">Slack</span></div>
        <div class="sec-row"><span class="sec-key">SLACK_BOT_TOKEN</span><input class="sec-input" type="password" id="SLACK_BOT_TOKEN" placeholder="xoxb-&#x2026;"/><span class="sec-status" id="st-SLACK_BOT_TOKEN"></span></div>
        <div class="sec-row"><span class="sec-key">SLACK_CHANNEL</span><input class="sec-input" type="text" id="SLACK_CHANNEL" placeholder="#incidents"/><span class="sec-status" id="st-SLACK_CHANNEL"></span></div>
      </div>
      <div class="sec-group">
        <div class="sec-group-hdr"><span class="g-ico">&#x1F3AB;</span><span class="g-name">Jira</span></div>
        <div class="sec-row"><span class="sec-key">JIRA_URL</span><input class="sec-input" type="text" id="JIRA_URL" placeholder="https://your-org.atlassian.net"/><span class="sec-status" id="st-JIRA_URL"></span></div>
        <div class="sec-row"><span class="sec-key">JIRA_USER</span><input class="sec-input" type="text" id="JIRA_USER" placeholder="user@org.com"/><span class="sec-status" id="st-JIRA_USER"></span></div>
        <div class="sec-row"><span class="sec-key">JIRA_TOKEN</span><input class="sec-input" type="password" id="JIRA_TOKEN" placeholder="&#x2026;"/><span class="sec-status" id="st-JIRA_TOKEN"></span></div>
      </div>
      <div class="sec-group">
        <div class="sec-group-hdr"><span class="g-ico">&#x1F6A8;</span><span class="g-name">OpsGenie</span></div>
        <div class="sec-row"><span class="sec-key">OPSGENIE_API_KEY</span><input class="sec-input" type="password" id="OPSGENIE_API_KEY" placeholder="&#x2026;"/><span class="sec-status" id="st-OPSGENIE_API_KEY"></span></div>
      </div>
      <div class="sec-group">
        <div class="sec-group-hdr"><span class="g-ico">&#x1F4CA;</span><span class="g-name">Grafana</span></div>
        <div class="sec-row"><span class="sec-key">GRAFANA_URL</span><input class="sec-input" type="text" id="GRAFANA_URL" placeholder="http://localhost:3000"/><span class="sec-status" id="st-GRAFANA_URL"></span></div>
        <div class="sec-row"><span class="sec-key">GRAFANA_TOKEN</span><input class="sec-input" type="password" id="GRAFANA_TOKEN" placeholder="service account token"/><span class="sec-status" id="st-GRAFANA_TOKEN"></span></div>
      </div>
      <div class="sec-group">
        <div class="sec-group-hdr"><span class="g-ico">&#x1F98A;</span><span class="g-name">GitLab</span></div>
        <div class="sec-row"><span class="sec-key">GITLAB_URL</span><input class="sec-input" type="text" id="GITLAB_URL" placeholder="https://gitlab.com"/><span class="sec-status" id="st-GITLAB_URL"></span></div>
        <div class="sec-row"><span class="sec-key">GITLAB_TOKEN</span><input class="sec-input" type="password" id="GITLAB_TOKEN" placeholder="personal access token"/><span class="sec-status" id="st-GITLAB_TOKEN"></span></div>
        <div class="sec-row"><span class="sec-key">GITLAB_PROJECT</span><input class="sec-input" type="text" id="GITLAB_PROJECT" placeholder="namespace/project or project-id"/><span class="sec-status" id="st-GITLAB_PROJECT"></span></div>
      </div>
    </div>

  </div>

    <div id="view-chat" class="chat-panel">
      <div id="chat-messages" class="chat-messages">
        <div class="chat-empty" id="chat-empty">
          <div class="chat-empty-icon">&#x1F916;</div>
          <div class="chat-empty-title">DevOps AI Assistant</div>
          <div class="chat-empty-sub">Ask me anything about your infrastructure. I have live access to your AWS account.</div>
          <div class="chat-suggestions">
            <button class="chat-suggestion" onclick="sendSuggestion(this)">Give me a full infrastructure overview across all integrations</button>
            <button class="chat-suggestion" onclick="sendSuggestion(this)">Are there any alerts or alarms firing right now?</button>
            <button class="chat-suggestion" onclick="sendSuggestion(this)">Any unhealthy pods or failed deployments in Kubernetes?</button>
            <button class="chat-suggestion" onclick="sendSuggestion(this)">Did any recent GitLab pipeline or GitHub deploy cause this issue?</button>
            <button class="chat-suggestion" onclick="sendSuggestion(this)">My service is down — find the root cause across AWS, K8s, and CI/CD</button>
          </div>
        </div>
      </div>
      <div class="chat-input-bar">
        <textarea id="chat-input" placeholder="Describe your issue or ask a question&#x2026;" rows="1" onkeydown="chatKeydown(event)" oninput="autoResize(this)"></textarea>
        <button class="chat-warroom-btn" id="chat-warroom-btn" onclick="createWarRoom()" title="Create Slack war room from this conversation">&#x1F6A8; War Room</button>
        <button class="chat-send-btn" id="chat-send-btn" onclick="sendChat()">Send</button>
      </div>
    </div>
  <div class="footer">AI DevOps Intelligence Platform &nbsp;&#xB7;&nbsp; <span>v1.0.0</span> &nbsp;&#xB7;&nbsp; Built by <span>Nagaraj</span></div>
</div>

<script>
var ALL_KEYS=['ANTHROPIC_API_KEY','GROQ_API_KEY','AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','AWS_REGION','GITHUB_TOKEN','GITHUB_REPO','GITLAB_URL','GITLAB_TOKEN','GITLAB_PROJECT','KUBECONFIG','SLACK_BOT_TOKEN','SLACK_CHANNEL','JIRA_URL','JIRA_USER','JIRA_TOKEN','OPSGENIE_API_KEY','GRAFANA_URL','GRAFANA_TOKEN'];
var INT_MAP={'Claude AI':'int-claude','AWS':'int-aws','Grafana':'int-grafana','GitHub':'int-github','GitLab':'int-gitlab','Kubernetes':'int-k8s','Slack':'int-slack','Jira':'int-jira','OpsGenie':'int-opsgenie'};

function showView(view, section, btn) {
  try {
    document.querySelectorAll('.nav-link').forEach(function(l){ l.classList.remove('active'); });
    if(btn) btn.classList.add('active');
    // show/hide content wrapper vs chat panel
    var cw = document.getElementById('content-wrap');
    if(cw) cw.style.display = view==='chat' ? 'none' : '';
    // endpoints panel
    var ep = document.getElementById('view-endpoints');
    if(ep) ep.style.display = view==='endpoints' ? '' : 'none';
    // secrets panel
    var sp = document.getElementById('view-secrets');
    if(sp){ if(view==='secrets'){ sp.classList.add('active'); }else{ sp.classList.remove('active'); } }
    // chat panel
    var cp = document.getElementById('view-chat');
    if(cp) cp.style.display = view==='chat' ? 'flex' : 'none';
    // filter endpoint groups
    if(view==='endpoints') filterSection(section || 'all');
  } catch(e) { console.error('showView error:', e); }
}
function filterSection(name) {
  document.querySelectorAll('[data-section]').forEach(function(g){
    g.style.display = (name==='all' || g.dataset.section===name) ? '' : 'none';
  });
}
function filterEps(q) {
  q = q.toLowerCase().trim();
  document.querySelectorAll('[data-ep]').forEach(function(ep){
    ep.classList.toggle('hidden', q!=='' && !ep.textContent.toLowerCase().includes(q));
  });
  document.querySelectorAll('[data-section]').forEach(function(grp){
    var vis = Array.from(grp.querySelectorAll('[data-ep]')).some(function(e){ return !e.classList.contains('hidden'); });
    grp.style.display = (!q || vis) ? '' : 'none';
  });
}
function loadSecretStatus() {
  fetch('/secrets/status').then(function(r){ return r.json(); }).then(function(data){
    Object.keys(data).forEach(function(group){
      var keys = data[group];
      Object.keys(keys).forEach(function(k){
        var st = document.getElementById('st-'+k);
        if(st){ st.textContent = keys[k] ? '&#x2713;' : ''; st.style.color = keys[k] ? '#7ee787' : ''; st.title = keys[k] ? 'Configured' : 'Not set'; }
      });
      var allSet = Object.values(keys).every(Boolean);
      var someSet = Object.values(keys).some(Boolean);
      var chipId = INT_MAP[group];
      if(chipId){
        var chip = document.getElementById(chipId);
        if(chip){ chip.className = 'int-chip' + (someSet ? ' on' : ''); }
      }
    });
  }).catch(function(){});
}
function saveSecrets() {
  var secrets = {};
  ALL_KEYS.forEach(function(k){
    var el = document.getElementById(k);
    if(el && el.value.trim()) secrets[k] = el.value.trim();
  });
  if(Object.keys(secrets).length === 0){ showMsg('No values entered', 'err'); return; }
  var user = document.getElementById('sec-user').value.trim() || 'nagaraj';
  var btn = document.getElementById('save-btn');
  btn.disabled = true; btn.textContent = 'Saving...';
  fetch('/secrets', {
    method: 'POST',
    headers: {'Content-Type':'application/json','X-User': user},
    body: JSON.stringify({secrets: secrets})
  }).then(function(r){
    return r.json().then(function(d){ return {ok: r.ok, data: d}; });
  }).then(function(res){
    if(res.ok){
      showMsg('Saved ' + res.data.updated.length + ' secret(s)', 'ok');
      ALL_KEYS.forEach(function(k){ var el=document.getElementById(k); if(el) el.value=''; });
      loadSecretStatus();
    } else {
      showMsg(res.data.detail || 'Error', 'err');
    }
  }).catch(function(e){ showMsg('Network error', 'err'); })
  .finally(function(){ btn.disabled=false; btn.textContent='Save to .env'; });
}
function showMsg(text, type) {
  var m = document.getElementById('sec-msg');
  m.textContent = text; m.className = 'sec-msg ' + type; m.style.display = '';
  setTimeout(function(){ m.style.display='none'; }, 4000);
}
loadSecretStatus();

var _chatHistory = [];

function sendSuggestion(btn) {
  document.getElementById('chat-input').value = btn.textContent;
  sendChat();
}

function chatKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function appendMsg(role, text) {
  var empty = document.getElementById('chat-empty');
  if (empty) empty.remove();
  var container = document.getElementById('chat-messages');
  var row = document.createElement('div');
  row.className = 'chat-row ' + role;
  var meta = document.createElement('div');
  meta.className = 'chat-meta' + (role === 'user' ? ' right' : '');
  meta.textContent = role === 'user' ? 'You' : 'AI DevOps';
  var bubble = document.createElement('div');
  bubble.className = 'chat-bubble ' + role;
  bubble.textContent = text;
  row.appendChild(meta);
  row.appendChild(bubble);
  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
  return bubble;
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

  var typingBubble = appendMsg('ai', '...');
  typingBubble.classList.add('typing');

  fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: msg, history: _chatHistory.slice(0,-1)})
  })
  .then(function(r){ return r.json(); })
  .then(function(d) {
    typingBubble.classList.remove('typing');
    var reply = d.reply || d.detail || 'No response';
    typingBubble.textContent = reply;
    _chatHistory.push({role: 'assistant', content: reply});
    if (d.sources && d.sources.length) {
      var src = document.createElement('div');
      src.style.cssText = 'font-size:10px;color:var(--muted);margin-top:4px;padding:0 4px';
      src.textContent = '📡 Data from: ' + d.sources.join(', ');
      typingBubble.parentElement.appendChild(src);
    }
    document.getElementById('chat-messages').scrollTop = 999999;
    btn.disabled = false;
    document.getElementById('chat-input').focus();
  })
  .catch(function(e) {
    typingBubble.classList.remove('typing');
    typingBubble.textContent = 'Error: ' + e;
    typingBubble.style.color = '#ffa198';
    btn.disabled = false;
  });
}

function createWarRoom() {
  var lastUserMsg = '';
  for (var i = _chatHistory.length - 1; i >= 0; i--) {
    if (_chatHistory[i].role === 'user') { lastUserMsg = _chatHistory[i].content; break; }
  }
  var desc = lastUserMsg || document.getElementById('chat-input').value.trim() || 'Infrastructure incident';
  var incId = 'INC-' + new Date().toISOString().replace(/[^0-9]/g,'').slice(0,12);
  var btn = document.getElementById('chat-warroom-btn');
  btn.disabled = true; btn.textContent = '⏳ Creating...';
  appendMsg('ai', '🚨 Creating war room and running analysis across all integrations...');
  fetch('/warroom/create', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({incident_id: incId, description: desc, severity: 'high', post_to_slack: true})
  })
  .then(function(r){ return r.json(); })
  .then(function(d) {
    btn.disabled = false; btn.textContent = '🚨 War Room';
    var msg = '✅ War room created — ' + incId + '\\n';
    var a = d.analysis || {};
    if (a.summary) msg += '\\n📋 ' + a.summary;
    if (a.root_cause) msg += '\\n🔍 Root cause: ' + a.root_cause;
    if (d.slack && d.slack.channel_url) msg += '\\n\\n🔗 Slack channel: ' + d.slack.channel_url;
    else if (d.slack && d.slack.error) msg += '\\n⚠️ Slack: ' + d.slack.error;
    if (d.sources && d.sources.length) msg += '\\n\\nData from: ' + d.sources.join(', ');
    appendMsg('ai', msg);
    _chatHistory.push({role: 'assistant', content: msg});
    document.getElementById('chat-messages').scrollTop = 999999;
  })
  .catch(function(e) {
    btn.disabled = false; btn.textContent = '🚨 War Room';
    appendMsg('ai', 'Error creating war room: ' + e);
  });
}
</script>
</body>
</html>
""", headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@app.get("/health")
def health():
    return {"status": "ok"}

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

@app.get("/check/linux")
def linux_check():
    result = check_linux_node()
    return {"linux_check": result}

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
_SAFE_ACTION = {"analyze", "lint", "format", "test"}

def _validate_file_path(file_path: str) -> None:
    if ".." in file_path or file_path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid file path")

@app.post("/incident/github/pr")
def incident_github_pr(head: str, base: str = "main"):
    if not _SAFE_BRANCH.match(head) or not _SAFE_BRANCH.match(base):
        raise HTTPException(status_code=400, detail="Invalid branch name")
    result = create_pull_request(head, base)
    return {"github_pr": result}

@app.post("/vscode/action")
def vscode_action(action: str = "analyze", file_path: str = ""):
    if action not in _SAFE_ACTION:
        raise HTTPException(status_code=400, detail=f"Invalid action. Allowed: {sorted(_SAFE_ACTION)}")
    if file_path:
        _validate_file_path(file_path)
    result = trigger_code_action(action, file_path)
    return {"vscode_action": result}

@app.post("/vscode/open")
def vscode_open(file_path: str):
    _validate_file_path(file_path)
    result = open_file_in_vscode(file_path)
    return {"vscode_open": result}

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
    "Claude AI":   ["ANTHROPIC_API_KEY"],
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
def secrets_update(payload: SecretsPayload, x_user: Optional[str] = Header(None)):
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

class WarRoomRequest(BaseModel):
    incident_id:  str
    description:  str
    severity:     str = "high"
    post_to_slack: bool = True

@app.post("/chat")
def chat(payload: ChatPayload):
    """Conversational DevOps AI — collects live context from ALL integrations."""
    context: dict = {}
    try:
        context = collect_all_context(hours=2)
    except Exception:
        pass
    history = [{"role": m.role, "content": m.content} for m in payload.history]
    reply = chat_devops(payload.message, history, context)
    return {"reply": reply, "sources": context.get("configured", [])}

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
