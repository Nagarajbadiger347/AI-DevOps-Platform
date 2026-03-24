import re
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Any

from app.correlation.engine import correlate_events
from app.plugins.aws_checker import check_aws_infrastructure
from app.plugins.linux_checker import check_linux_node
from app.plugins.k8s_checker import check_k8s_cluster, check_k8s_nodes, check_k8s_pods, check_k8s_deployments
from app.integrations.k8s_ops import restart_deployment, scale_deployment, get_pod_logs
from app.integrations.aws_ops import (
    list_ec2_instances, get_ec2_status_checks, get_ec2_console_output,
    list_log_groups, get_recent_logs, search_logs,
    list_cloudwatch_alarms, get_metric,
    list_ecs_services, get_stopped_ecs_tasks,
    list_lambda_functions, get_lambda_errors,
    list_rds_instances, get_rds_events,
    get_target_health, get_cloudtrail_events,
    collect_diagnosis_context,
)
from app.llm.claude import analyze_context, diagnose_aws_resource
from app.agents.incident_pipeline import run_incident_pipeline
from app.integrations.slack import create_war_room
from app.integrations.jira import create_incident
from app.integrations.opsgenie import notify_on_call
from app.integrations.github import create_issue, create_pull_request
from app.integrations.vscode import trigger_code_action, open_file_in_vscode
from app.memory.vector_db import store_incident
from app.security.rbac import check_access, assign_role, revoke_role

app = FastAPI(
    title="AI DevOps Intelligence Platform",
    description="Autonomous DevOps management powered by multi-agent AI — built by Nagaraj",
    version="1.0.0",
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
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>AI DevOps Platform — Nagaraj</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
  <style>
    *,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
    :root{
      --bg:#060810;--surface:#0c0f1d;--border:#1a1f38;
      --purple:#7c3aed;--indigo:#4f46e5;--cyan:#06b6d4;--green:#10b981;
      --amber:#f59e0b;--pink:#ec4899;--text:#f1f5f9;--muted:#64748b;
    }
    body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;padding:2rem 1rem;overflow-x:hidden}
    body::before,body::after{content:'';position:fixed;border-radius:50%;filter:blur(130px);opacity:.14;pointer-events:none;animation:drift 14s ease-in-out infinite alternate}
    body::before{width:700px;height:700px;background:radial-gradient(circle,#7c3aed,#4f46e5);top:-250px;left:-200px}
    body::after{width:500px;height:500px;background:radial-gradient(circle,#06b6d4,#10b981);bottom:-150px;right:-150px;animation-delay:-7s}
    @keyframes drift{from{transform:translate(0,0) scale(1)}to{transform:translate(50px,40px) scale(1.1)}}
    .wrap{position:relative;z-index:1;max-width:1000px;margin:0 auto}
    /* HERO */
    .hero{text-align:center;padding:2rem 0 2.5rem}
    .pill{display:inline-flex;align-items:center;gap:7px;background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);color:#34d399;border-radius:999px;font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;padding:5px 14px;margin-bottom:1.4rem}
    .dot{width:7px;height:7px;background:#34d399;border-radius:50%;animation:blink 2s ease-in-out infinite}
    @keyframes blink{0%,100%{opacity:1;box-shadow:0 0 6px #34d399}50%{opacity:.2}}
    h1{font-size:clamp(2rem,4.5vw,3rem);font-weight:900;line-height:1.1;letter-spacing:-.03em;margin-bottom:.9rem}
    .grad{background:linear-gradient(135deg,#a78bfa,#60a5fa,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
    .sub{color:var(--muted);font-size:.95rem;max-width:500px;margin:0 auto .9rem;line-height:1.65}
    .author{display:inline-flex;align-items:center;gap:8px;background:linear-gradient(135deg,rgba(124,58,237,.15),rgba(6,182,212,.15));border:1px solid rgba(124,58,237,.3);border-radius:999px;padding:5px 16px;font-size:.82rem;color:#c4b5fd;font-weight:500}
    .av{width:24px;height:24px;background:linear-gradient(135deg,#7c3aed,#06b6d4);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.65rem;font-weight:700;color:#fff}
    /* STATS */
    .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:.9rem;margin-bottom:1.8rem}
    .stat{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1.1rem 1rem;text-align:center;position:relative;overflow:hidden;transition:transform .2s,box-shadow .2s}
    .stat:hover{transform:translateY(-3px)}
    .stat::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:3px 3px 0 0}
    .stat:nth-child(1)::before{background:linear-gradient(90deg,#7c3aed,#a78bfa)}
    .stat:nth-child(2)::before{background:linear-gradient(90deg,#06b6d4,#67e8f9)}
    .stat:nth-child(3)::before{background:linear-gradient(90deg,#10b981,#6ee7b7)}
    .stat:nth-child(4)::before{background:linear-gradient(90deg,#f59e0b,#fcd34d)}
    .stat:hover:nth-child(1){box-shadow:0 8px 30px rgba(124,58,237,.2)}
    .stat:hover:nth-child(2){box-shadow:0 8px 30px rgba(6,182,212,.2)}
    .stat:hover:nth-child(3){box-shadow:0 8px 30px rgba(16,185,129,.2)}
    .stat:hover:nth-child(4){box-shadow:0 8px 30px rgba(245,158,11,.2)}
    .sn{font-size:1.8rem;font-weight:800;line-height:1;margin-bottom:.2rem}
    .stat:nth-child(1) .sn{color:#a78bfa}.stat:nth-child(2) .sn{color:#67e8f9}
    .stat:nth-child(3) .sn{color:#6ee7b7}.stat:nth-child(4) .sn{color:#fcd34d}
    .sl{font-size:.7rem;color:var(--muted);font-weight:500;text-transform:uppercase;letter-spacing:.07em}
    /* SEARCH */
    .search-wrap{position:relative;margin-bottom:1rem}
    .search-wrap svg{position:absolute;left:.85rem;top:50%;transform:translateY(-50%);opacity:.4;pointer-events:none}
    #search{width:100%;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:.65rem .9rem .65rem 2.5rem;color:var(--text);font-size:.83rem;font-family:'Inter',sans-serif;outline:none;transition:border-color .18s}
    #search:focus{border-color:rgba(124,58,237,.5);box-shadow:0 0 0 3px rgba(124,58,237,.1)}
    #search::placeholder{color:var(--muted)}
    /* TABS */
    .tabs{display:flex;gap:.4rem;margin-bottom:1.2rem;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:.4rem}
    .tab{flex:1;padding:.5rem .8rem;border-radius:8px;border:none;background:transparent;color:var(--muted);font-size:.78rem;font-weight:600;cursor:pointer;transition:all .18s;font-family:'Inter',sans-serif}
    .tab:hover{color:var(--text);background:rgba(255,255,255,.05)}
    .tab.active{background:linear-gradient(135deg,rgba(124,58,237,.3),rgba(79,70,229,.3));color:#c4b5fd;border:1px solid rgba(124,58,237,.35)}
    .tab-panel{display:none}.tab-panel.active{display:block}
    /* CARD */
    .card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.6rem;margin-bottom:1rem}
    .sh{display:flex;align-items:center;gap:8px;margin-bottom:1.1rem}
    .si{width:28px;height:28px;border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:.9rem}
    .st{font-size:.72rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
    /* ENDPOINTS */
    .epg{display:grid;grid-template-columns:1fr 1fr;gap:.5rem}
    .ep{display:flex;align-items:center;gap:10px;background:rgba(255,255,255,.02);border:1px solid var(--border);border-radius:9px;padding:.65rem .9rem;text-decoration:none;color:inherit;transition:all .18s;position:relative;overflow:hidden}
    .ep:hover{transform:translateY(-2px)}
    .get-ep:hover{border-color:rgba(16,185,129,.4);box-shadow:0 4px 20px rgba(16,185,129,.12);background:rgba(16,185,129,.06)}
    .get-ep:hover .ep-path{color:#6ee7b7}
    .post-ep:hover{border-color:rgba(124,58,237,.45);box-shadow:0 4px 20px rgba(124,58,237,.15);background:rgba(124,58,237,.07)}
    .post-ep:hover .ep-path{color:#c4b5fd}
    .ws-ep:hover{border-color:rgba(245,158,11,.4);box-shadow:0 4px 20px rgba(245,158,11,.12);background:rgba(245,158,11,.06)}
    .ws-ep:hover .ep-path{color:#fcd34d}
    .ep::after{content:'→';position:absolute;right:.8rem;font-size:.72rem;opacity:0;transform:translateX(-4px);transition:opacity .15s,transform .15s;color:#64748b}
    .ep:hover::after{opacity:1;transform:translateX(0)}
    .badge{font-size:.58rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;padding:3px 8px;border-radius:5px;flex-shrink:0;min-width:40px;text-align:center}
    .get{background:rgba(16,185,129,.15);color:#34d399;border:1px solid rgba(16,185,129,.2)}
    .post{background:rgba(124,58,237,.15);color:#a78bfa;border:1px solid rgba(124,58,237,.2)}
    .ws{background:rgba(245,158,11,.15);color:#fcd34d;border:1px solid rgba(245,158,11,.2)}
    .ep-text{flex:1;min-width:0;padding-right:1.2rem}
    .ep-path{font-family:'JetBrains Mono',monospace;font-size:.77rem;color:#94a3b8;display:block;transition:color .15s;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .ep-desc{font-size:.65rem;color:var(--muted);margin-top:1px}
    /* PIPELINE */
    .pipe-card{background:linear-gradient(135deg,rgba(236,72,153,.07),rgba(124,58,237,.07));border:1px solid rgba(236,72,153,.3);border-radius:16px;padding:1.6rem;margin-bottom:1rem}
    .flow{display:flex;align-items:stretch;gap:0;margin:1.2rem 0;overflow-x:auto}
    .fstep{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:.9rem .8rem;min-width:130px;flex:1;text-align:center}
    .fstep .ic{font-size:1.4rem;display:block;margin-bottom:.35rem}
    .fstep .lb{font-size:.72rem;font-weight:700;margin-bottom:.15rem}
    .fstep .sb{font-size:.62rem;color:var(--muted);line-height:1.4}
    .farr{display:flex;align-items:center;padding:0 .3rem;color:#334155;font-size:1.1rem;flex-shrink:0}
    .fstep.s1{border-color:rgba(6,182,212,.3)}.fstep.s1 .lb{color:#67e8f9}
    .fstep.s2{border-color:rgba(124,58,237,.3)}.fstep.s2 .lb{color:#c4b5fd}
    .fstep.s3{border-color:rgba(236,72,153,.3)}.fstep.s3 .lb{color:#f9a8d4}
    .fstep.s4{border-color:rgba(16,185,129,.3)}.fstep.s4 .lb{color:#6ee7b7}
    .pipe-ep{display:flex;align-items:center;gap:10px;background:rgba(236,72,153,.08);border:1px solid rgba(236,72,153,.4);border-radius:10px;padding:.8rem 1rem;text-decoration:none;color:inherit;transition:all .18s;position:relative}
    .pipe-ep:hover{background:rgba(236,72,153,.15);border-color:rgba(236,72,153,.6);transform:translateY(-2px);box-shadow:0 6px 24px rgba(236,72,153,.2)}
    .pipe-ep:hover .ep-path{color:#fda4af}
    .pipe-ep::after{content:'→';position:absolute;right:.8rem;font-size:.72rem;opacity:0;transform:translateX(-4px);transition:opacity .15s,transform .15s;color:#64748b}
    .pipe-ep:hover::after{opacity:1;transform:translateX(0)}
    .tags{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.9rem}
    .tag{font-size:.7rem;padding:.25rem .7rem;border-radius:999px;font-weight:600;text-decoration:none;transition:all .15s;cursor:pointer}
    .tag:hover{transform:translateY(-1px);opacity:.85}
    pre.sample{background:#080b16;border:1px solid var(--border);border-radius:10px;padding:.9rem 1rem;font-family:'JetBrains Mono',monospace;font-size:.71rem;color:#e2e8f0;line-height:1.75;overflow-x:auto;margin-top:.7rem;white-space:pre-wrap}
    /* LINKS */
    .links{display:flex;gap:.7rem;margin-bottom:.9rem}
    .lbtn{flex:1;display:flex;align-items:center;justify-content:center;gap:8px;padding:.75rem;border-radius:10px;font-size:.82rem;font-weight:600;text-decoration:none;transition:all .18s;border:1px solid transparent}
    .lbtn.pri{background:linear-gradient(135deg,#7c3aed,#4f46e5);color:#fff;box-shadow:0 4px 20px rgba(124,58,237,.35)}
    .lbtn.pri:hover{box-shadow:0 6px 30px rgba(124,58,237,.55);transform:translateY(-2px)}
    .lbtn.sec{background:rgba(6,182,212,.08);color:#67e8f9;border-color:rgba(6,182,212,.25)}
    .lbtn.sec:hover{background:rgba(6,182,212,.15);border-color:rgba(6,182,212,.5);transform:translateY(-2px)}
    .lbtn.ter{background:rgba(16,185,129,.08);color:#6ee7b7;border-color:rgba(16,185,129,.25)}
    .lbtn.ter:hover{background:rgba(16,185,129,.15);border-color:rgba(16,185,129,.5);transform:translateY(-2px)}
    .footer{text-align:center;font-size:.71rem;color:#2d3748;padding-top:.3rem}
    .footer span{color:#475569}
    .hidden{display:none!important}
    @media(max-width:620px){
      .stats{grid-template-columns:repeat(2,1fr)}
      .epg{grid-template-columns:1fr}
      .links{flex-direction:column}
      .tabs{flex-wrap:wrap}
      .flow{flex-direction:column}
    }
  </style>
</head>
<body>
<div class="wrap">

  <div class="hero">
    <div class="pill"><span class="dot"></span> System Online</div>
    <h1>AI <span class="grad">DevOps Intelligence</span><br>Platform</h1>
    <p class="sub">Autonomous incident management, root cause analysis, and multi-agent AI orchestration — built for production.</p>
    <div class="author"><div class="av">N</div>&nbsp;Built by <strong>Nagaraj</strong></div>
  </div>

  <div class="stats">
    <div class="stat"><div class="sn">35+</div><div class="sl">API Endpoints</div></div>
    <div class="stat"><div class="sn">6</div><div class="sl">Integrations</div></div>
    <div class="stat"><div class="sn">3</div><div class="sl">RBAC Roles</div></div>
    <div class="stat"><div class="sn">v1.0</div><div class="sl">Version</div></div>
  </div>

  <div class="search-wrap">
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>
    <input id="search" type="text" placeholder="Search endpoints  (e.g. k8s, aws, lambda, incident…)" oninput="filterEps(this.value)"/>
  </div>

  <div class="tabs">
    <button class="tab active" onclick="switchTab('all',this)">⚡ All</button>
    <button class="tab" onclick="switchTab('general',this)">General &amp; AI</button>
    <button class="tab" onclick="switchTab('k8s',this)">☸ Kubernetes</button>
    <button class="tab" onclick="switchTab('aws',this)">☁ AWS</button>
    <button class="tab" onclick="switchTab('pipeline',this)">🤖 Pipeline</button>
  </div>

  <div id="tab-all" class="tab-panel active">

    <div class="card" data-section="general">
      <div class="sh"><div class="si" style="background:rgba(124,58,237,.15)">⚡</div><span class="st">General &amp; AI</span></div>
      <div class="epg">
        <a class="ep get-ep" href="/health" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/health</span><span class="ep-desc">System health status</span></div></a>
        <a class="ep get-ep" href="/docs" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/docs</span><span class="ep-desc">Interactive Swagger UI</span></div></a>
        <a class="ep post-ep" href="/docs#/default/correlate_correlate_post" target="_blank" data-ep><span class="badge post">POST</span><div class="ep-text"><span class="ep-path">/correlate</span><span class="ep-desc">Correlate events &amp; find patterns</span></div></a>
        <a class="ep post-ep" href="/docs#/default/llm_analyze_llm_analyze_post" target="_blank" data-ep><span class="badge post">POST</span><div class="ep-text"><span class="ep-path">/llm/analyze</span><span class="ep-desc">Claude AI root cause analysis</span></div></a>
        <a class="ep get-ep" href="/check/aws" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/check/aws</span><span class="ep-desc">AWS infra health check</span></div></a>
        <a class="ep get-ep" href="/check/linux" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/check/linux</span><span class="ep-desc">Linux node health</span></div></a>
        <a class="ep post-ep" href="/docs#/default/incident_war_room_incident_war_room_post" target="_blank" data-ep><span class="badge post">POST</span><div class="ep-text"><span class="ep-path">/incident/war-room</span><span class="ep-desc">Open Slack war room</span></div></a>
        <a class="ep post-ep" href="/docs#/default/incident_jira_incident_jira_post" target="_blank" data-ep><span class="badge post">POST</span><div class="ep-text"><span class="ep-path">/incident/jira</span><span class="ep-desc">Create Jira incident ticket</span></div></a>
        <a class="ep post-ep" href="/docs#/default/memory_incident_memory_incidents_post" target="_blank" data-ep><span class="badge post">POST</span><div class="ep-text"><span class="ep-path">/memory/incidents</span><span class="ep-desc">Store incident in ChromaDB</span></div></a>
        <a class="ep post-ep" href="/docs#/default/security_check_security_check_post" target="_blank" data-ep><span class="badge post">POST</span><div class="ep-text"><span class="ep-path">/security/check</span><span class="ep-desc">RBAC access check</span></div></a>
        <a class="ep post-ep" href="/docs#/default/security_assign_role_security_roles_post" target="_blank" data-ep><span class="badge post">POST</span><div class="ep-text"><span class="ep-path">/security/roles</span><span class="ep-desc">Assign role to user</span></div></a>
        <a class="ep ws-ep" href="/docs#/default/websocket_events_realtime_events_ws" target="_blank" data-ep><span class="badge ws">WS</span><div class="ep-text"><span class="ep-path">/realtime/events</span><span class="ep-desc">Live event stream via WebSocket</span></div></a>
      </div>
    </div>

    <div class="card" data-section="k8s">
      <div class="sh"><div class="si" style="background:rgba(6,182,212,.15)">☸</div><span class="st">Kubernetes</span></div>
      <div class="epg">
        <a class="ep get-ep" href="/check/k8s" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/check/k8s</span><span class="ep-desc">Cluster summary</span></div></a>
        <a class="ep get-ep" href="/check/k8s/nodes" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/check/k8s/nodes</span><span class="ep-desc">Per-node ready status</span></div></a>
        <a class="ep get-ep" href="/check/k8s/pods" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/check/k8s/pods</span><span class="ep-desc">Pod health by namespace</span></div></a>
        <a class="ep get-ep" href="/check/k8s/deployments" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/check/k8s/deployments</span><span class="ep-desc">Deployment rollout status</span></div></a>
        <a class="ep post-ep" href="/docs#/default/k8s_restart_k8s_restart_post" target="_blank" data-ep><span class="badge post">POST</span><div class="ep-text"><span class="ep-path">/k8s/restart</span><span class="ep-desc">Rolling restart a deployment</span></div></a>
        <a class="ep post-ep" href="/docs#/default/k8s_scale_k8s_scale_post" target="_blank" data-ep><span class="badge post">POST</span><div class="ep-text"><span class="ep-path">/k8s/scale</span><span class="ep-desc">Scale deployment replicas</span></div></a>
        <a class="ep get-ep" href="/k8s/logs?namespace=default&pod=my-pod&tail_lines=50" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/k8s/logs</span><span class="ep-desc">Fetch pod logs (tail N lines)</span></div></a>
      </div>
    </div>

    <div class="card" data-section="aws">
      <div class="sh"><div class="si" style="background:rgba(245,158,11,.15)">☁</div><span class="st">AWS Observability &amp; AI Diagnosis</span></div>
      <div class="epg">
        <a class="ep get-ep" href="/aws/ec2/instances" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/aws/ec2/instances</span><span class="ep-desc">List EC2 instances &amp; states</span></div></a>
        <a class="ep get-ep" href="/docs#/default/aws_ec2_status_aws_ec2_status_get" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/aws/ec2/status</span><span class="ep-desc">EC2 system status checks</span></div></a>
        <a class="ep get-ep" href="/docs#/default/aws_ec2_console_aws_ec2_console_get" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/aws/ec2/console</span><span class="ep-desc">Serial console output (crash/boot)</span></div></a>
        <a class="ep get-ep" href="/aws/logs/groups" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/aws/logs/groups</span><span class="ep-desc">List CloudWatch Log Groups</span></div></a>
        <a class="ep get-ep" href="/docs#/default/aws_logs_recent_aws_logs_recent_get" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/aws/logs/recent</span><span class="ep-desc">Fetch recent log events</span></div></a>
        <a class="ep get-ep" href="/docs#/default/aws_logs_search_aws_logs_search_get" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/aws/logs/search</span><span class="ep-desc">Search logs by pattern</span></div></a>
        <a class="ep get-ep" href="/aws/cloudwatch/alarms" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/aws/cloudwatch/alarms</span><span class="ep-desc">Firing CloudWatch alarms</span></div></a>
        <a class="ep post-ep" href="/docs#/default/aws_cw_metrics_aws_cloudwatch_metrics_post" target="_blank" data-ep><span class="badge post">POST</span><div class="ep-text"><span class="ep-path">/aws/cloudwatch/metrics</span><span class="ep-desc">Fetch any CloudWatch metric series</span></div></a>
        <a class="ep get-ep" href="/aws/ecs/services" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/aws/ecs/services</span><span class="ep-desc">ECS running vs desired counts</span></div></a>
        <a class="ep get-ep" href="/aws/ecs/stopped-tasks" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/aws/ecs/stopped-tasks</span><span class="ep-desc">Stopped task stop-reasons &amp; exit codes</span></div></a>
        <a class="ep get-ep" href="/docs#/default/aws_lambda_errors_aws_lambda_errors_get" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/aws/lambda/errors</span><span class="ep-desc">Lambda error &amp; throttle metrics</span></div></a>
        <a class="ep get-ep" href="/docs#/default/aws_rds_events_aws_rds_events_get" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/aws/rds/events</span><span class="ep-desc">RDS events — failovers, restarts</span></div></a>
        <a class="ep get-ep" href="/docs#/default/aws_elb_health_aws_elb_target_health_get" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/aws/elb/target-health</span><span class="ep-desc">ALB target group health</span></div></a>
        <a class="ep get-ep" href="/aws/cloudtrail/events" target="_blank" data-ep><span class="badge get">GET</span><div class="ep-text"><span class="ep-path">/aws/cloudtrail/events</span><span class="ep-desc">Recent API changes (who did what)</span></div></a>
        <a class="ep post-ep" href="/docs#/default/aws_diagnose_aws_diagnose_post" target="_blank" data-ep style="border-color:rgba(124,58,237,.4);background:rgba(124,58,237,.07)"><span class="badge post" style="background:rgba(124,58,237,.25);color:#c4b5fd;border-color:rgba(124,58,237,.4)">POST</span><div class="ep-text"><span class="ep-path" style="color:#c4b5fd">/aws/diagnose</span><span class="ep-desc">AI root cause analysis from live AWS data</span></div></a>
      </div>
    </div>

    <div class="pipe-card" data-section="pipeline">
      <div class="sh">
        <div class="si" style="background:linear-gradient(135deg,rgba(236,72,153,.3),rgba(124,58,237,.3));font-size:1rem">🤖</div>
        <span class="st" style="background:linear-gradient(90deg,#ec4899,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent">Autonomous Incident Pipeline</span>
      </div>
      <p style="font-size:.82rem;color:#cbd5e1;line-height:1.65;margin-bottom:1rem">
        One API call triggers the full response loop — collects data from
        <strong style="color:#67e8f9">AWS</strong>, <strong style="color:#c4b5fd">Kubernetes</strong>, and
        <strong style="color:#6ee7b7">GitHub</strong>, runs Claude AI root cause analysis, then executes remediation automatically.
      </p>
      <div class="flow">
        <div class="fstep s1"><span class="ic">📡</span><div class="lb">Collect</div><div class="sb">AWS · K8s · GitHub<br>in parallel</div></div>
        <div class="farr">›</div>
        <div class="fstep s2"><span class="ic">🧠</span><div class="lb">Synthesize</div><div class="sb">Claude AI<br>root cause analysis</div></div>
        <div class="farr">›</div>
        <div class="fstep s3"><span class="ic">⚡</span><div class="lb">Remediate</div><div class="sb">K8s restart · GitHub PR<br>Jira · Slack · OpsGenie</div></div>
        <div class="farr">›</div>
        <div class="fstep s4"><span class="ic">📋</span><div class="lb">Report</div><div class="sb">Full incident report<br>stored in ChromaDB</div></div>
      </div>
      <a class="pipe-ep" href="/docs#/default/incident_run_incident_run_post" target="_blank" data-ep>
        <span class="badge post" style="background:rgba(236,72,153,.25);color:#f9a8d4;border-color:rgba(236,72,153,.5)">POST</span>
        <div class="ep-text"><span class="ep-path" style="color:#f9a8d4">/incident/run</span><span class="ep-desc">Full autonomous pipeline — collect → analyse → act → report</span></div>
      </a>
      <pre class="sample">{ "incident_id": "INC-001",  "description": "High 5xx rate on API",  "severity": "critical",
  "aws": { "resource_type": "ecs", "resource_id": "my-cluster" },
  "k8s": { "namespace": "production" },
  "auto_remediate": true,  "hours": 2 }</pre>
      <div class="tags">
        <a class="tag" href="/docs#/default/correlate_correlate_post" target="_blank" style="background:rgba(6,182,212,.12);border:1px solid rgba(6,182,212,.3);color:#67e8f9">✦ Parallel data collection</a>
        <a class="tag" href="/docs#/default/llm_analyze_llm_analyze_post" target="_blank" style="background:rgba(124,58,237,.12);border:1px solid rgba(124,58,237,.3);color:#c4b5fd">✦ Claude AI synthesis</a>
        <a class="tag" href="/docs#/default/incident_run_incident_run_post" target="_blank" style="background:rgba(236,72,153,.12);border:1px solid rgba(236,72,153,.3);color:#f9a8d4">✦ Auto remediation</a>
        <a class="tag" href="/docs#/default/memory_incident_memory_incidents_post" target="_blank" style="background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.3);color:#fcd34d">✦ ChromaDB memory</a>
      </div>
    </div>

  </div>

  <div id="tab-general" class="tab-panel"></div>
  <div id="tab-k8s" class="tab-panel"></div>
  <div id="tab-aws" class="tab-panel"></div>
  <div id="tab-pipeline" class="tab-panel"></div>

  <div class="links">
    <a class="lbtn pri" href="/docs">📖&nbsp; Swagger UI</a>
    <a class="lbtn sec" href="/redoc">📄&nbsp; ReDoc</a>
    <a class="lbtn ter" href="/health">💚&nbsp; Health</a>
  </div>

  <div class="footer">AI DevOps Intelligence Platform &nbsp;·&nbsp; <span>v1.0.0</span> &nbsp;·&nbsp; Built by <span>Nagaraj</span></div>

</div>
<script>
function switchTab(name,btn){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  var panel=document.getElementById('tab-'+name);
  if(name==='all'){panel.classList.add('active');return;}
  if(!panel.hasChildNodes()){
    var src=document.querySelector('[data-section="'+name+'"]');
    if(src)panel.appendChild(src.cloneNode(true));
  }
  panel.classList.add('active');
}
function filterEps(q){
  q=q.toLowerCase().trim();
  document.querySelectorAll('#tab-all [data-ep]').forEach(function(ep){
    var txt=ep.textContent.toLowerCase();
    ep.classList.toggle('hidden',q!==''&&!txt.includes(q));
  });
  document.querySelectorAll('#tab-all .card,#tab-all .pipe-card').forEach(function(card){
    var vis=[].slice.call(card.querySelectorAll('[data-ep]')).some(function(e){return!e.classList.contains('hidden')});
    card.style.display=(!q||vis)?'':'none';
  });
}
</script>
</body>
</html>
""")


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
def aws_ec2_status(instance_id: str):
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
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"ecs_services": result}

@app.get("/aws/ecs/stopped-tasks")
def aws_ecs_stopped(cluster: str = "default", limit: int = 20):
    result = get_stopped_ecs_tasks(cluster, limit)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"stopped_tasks": result}

# Lambda
@app.get("/aws/lambda/functions")
def aws_lambda_list():
    result = list_lambda_functions()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"lambda_functions": result}

@app.get("/aws/lambda/errors")
def aws_lambda_errors(function_name: str, hours: int = 1):
    result = get_lambda_errors(function_name, hours)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"lambda_metrics": result}

# RDS
@app.get("/aws/rds/instances")
def aws_rds_list():
    result = list_rds_instances()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"rds_instances": result}

@app.get("/aws/rds/events")
def aws_rds_events(db_instance_id: str, hours: int = 24):
    result = get_rds_events(db_instance_id, hours)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"rds_events": result}

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
def k8s_restart(req: K8sRestartRequest):
    result = restart_deployment(req.namespace, req.deployment)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"result": result}

@app.post("/k8s/scale")
def k8s_scale(req: K8sScaleRequest):
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
def security_assign_role(req: RoleAssignment):
    result = assign_role(req.user, req.role)
    return {"result": result}

@app.delete("/security/roles/{user}")
def security_revoke_role(user: str):
    result = revoke_role(user)
    return {"result": result}

@app.post("/incident/run")
def incident_run(req: IncidentRunRequest):
    """End-to-end autonomous incident response pipeline.

    Collects AWS + K8s + GitHub observability data, runs AI root cause analysis,
    executes recommended actions (GitHub PR, Jira, Slack, OpsGenie, K8s ops),
    stores the incident in memory, and returns a full incident report.
    """
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
