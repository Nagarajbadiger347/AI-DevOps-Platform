#!/usr/bin/env python3
"""NexusOps CLI — DevOps debugging and incident response from the terminal.

Usage:
    python cli.py [COMMAND] [OPTIONS]

    # Or install as an entry point:
    pip install -e .  (if setup.py/pyproject.toml exposes nexusops=cli:cli)

Environment:
    NEXUSOPS_URL    Base URL of the NexusOps API  (default: http://localhost:8000)
    NEXUSOPS_TOKEN  JWT bearer token for auth      (obtain via: nexusops login)
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

try:
    import click
    import requests
except ImportError:
    print("Missing CLI dependencies. Run: pip install click requests", file=sys.stderr)
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

_CONFIG_PATH = os.path.expanduser("~/.nexusops/config.json")

def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _save_config(data: dict) -> None:
    os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
    with open(_CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)

def _base_url() -> str:
    cfg = _load_config()
    return os.getenv("NEXUSOPS_URL") or cfg.get("url") or "http://localhost:8000"

def _token() -> str:
    cfg = _load_config()
    return os.getenv("NEXUSOPS_TOKEN") or cfg.get("token") or ""

def _headers() -> dict:
    t = _token()
    h = {"Content-Type": "application/json"}
    if t:
        h["Authorization"] = f"Bearer {t}"
    return h

def _get(path: str, params: dict | None = None) -> dict:
    r = requests.get(f"{_base_url()}{path}", headers=_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def _post(path: str, body: dict) -> dict:
    r = requests.post(f"{_base_url()}{path}", headers=_headers(), json=body, timeout=60)
    r.raise_for_status()
    return r.json()

def _print_json(data: dict) -> None:
    click.echo(json.dumps(data, indent=2, default=str))

def _ok(msg: str) -> None:
    click.secho(f"  {msg}", fg="green")

def _warn(msg: str) -> None:
    click.secho(f"  {msg}", fg="yellow")

def _err(msg: str) -> None:
    click.secho(f"  {msg}", fg="red", err=True)


# ── Root group ────────────────────────────────────────────────────────────────

@click.group()
@click.version_option("2.0.0", prog_name="nexusops")
def cli():
    """NexusOps — AI-powered DevOps debugging and incident response."""
    pass


# ── Auth ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--url", default="http://localhost:8000", help="NexusOps API base URL")
@click.option("--username", prompt=True)
@click.password_option("--password", confirmation_prompt=False)
def login(url: str, username: str, password: str):
    """Authenticate and store a token in ~/.nexusops/config.json."""
    try:
        r = requests.post(
            f"{url}/v1/auth/login",
            json={"username": username, "password": password},
            timeout=15,
        )
        r.raise_for_status()
        token = r.json().get("access_token") or r.json().get("token")
        if not token:
            _err("Login succeeded but no token returned. Check server.")
            sys.exit(1)
        _save_config({"url": url, "token": token})
        _ok(f"Logged in. Token saved to {_CONFIG_PATH}")
    except requests.HTTPError as e:
        _err(f"Login failed: {e.response.status_code} {e.response.text}")
        sys.exit(1)


# ── Health ────────────────────────────────────────────────────────────────────

@cli.group()
def health():
    """Platform health and degraded-mode status."""
    pass

@health.command("status")
def health_status():
    """Quick health check — shows platform mode and component status."""
    try:
        data = _get("/health/degraded")
        mode = data.get("mode", "unknown")
        color = "green" if mode == "full" else ("yellow" if "no-ai" in mode else "red")
        click.secho(f"\n  Mode: {mode.upper()}", fg=color, bold=True)
        for comp, info in data.get("components", {}).items():
            ok = info.get("ok", False)
            err = f" — {info['error']}" if info.get("error") else ""
            status_str = "OK" if ok else "DOWN"
            click.secho(f"  {comp:<12} {status_str}{err}", fg="green" if ok else "red")
        click.echo()
    except Exception as e:
        _err(f"Cannot reach NexusOps at {_base_url()}: {e}")
        sys.exit(1)

@health.command("integrations")
def health_integrations():
    """Show which integrations (AWS, Slack, K8s, GitHub…) are configured."""
    data = _get("/health/integrations")
    _print_json(data)

@health.command("full")
def health_full():
    """Deep health check across AWS, K8s, and Grafana."""
    click.echo("  Running full health scan (may take up to 30s)…")
    data = _get("/health/full")
    status = data.get("status", "unknown")
    color = "green" if status == "healthy" else "red"
    click.secho(f"\n  Overall: {status.upper()}", fg=color, bold=True)
    h = data.get("health", {})
    click.echo(f"  Issues:  {h.get('issue_count', 0)}")
    for issue in h.get("issues", []):
        _warn(f"  • {issue}")
    click.echo()


# ── Incidents ─────────────────────────────────────────────────────────────────

@cli.group()
def incident():
    """Run, inspect, and list incidents."""
    pass

@incident.command("run")
@click.argument("description")
@click.option("--severity", "-s", default="medium",
              type=click.Choice(["critical", "high", "medium", "low"]), show_default=True)
@click.option("--auto-remediate", is_flag=True, default=False,
              help="Auto-execute low-risk actions without approval")
@click.option("--dry-run", is_flag=True, default=False,
              help="Plan actions but do not execute them")
@click.option("--id", "incident_id", default=None,
              help="Custom incident ID (auto-generated if omitted)")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output raw JSON")
def incident_run(description: str, severity: str, auto_remediate: bool,
                  dry_run: bool, incident_id: Optional[str], as_json: bool):
    """Run the AI incident pipeline against DESCRIPTION.

    Example:
        nexusops incident run "payments-api pods crash-looping in prod" --severity high
    """
    iid = incident_id or f"cli-{int(time.time())}"
    click.echo(f"  Starting pipeline for: {description!r}")
    click.echo(f"  ID={iid}  severity={severity}  dry_run={dry_run}  auto_remediate={auto_remediate}\n")
    try:
        result = _post("/v1/incidents/run", {
            "incident_id":    iid,
            "description":    description,
            "severity":       severity,
            "auto_remediate": auto_remediate,
            "dry_run":        dry_run,
        })
        if as_json:
            _print_json(result)
            return
        status = result.get("status", "unknown")
        color = "green" if status == "completed" else ("yellow" if status == "approval_required" else "red")
        click.secho(f"  Status: {status.upper()}", fg=color, bold=True)
        plan = result.get("plan") or {}
        if plan.get("root_cause"):
            click.echo(f"  Root cause: {plan['root_cause']}")
        if plan.get("risk"):
            click.echo(f"  Risk: {plan['risk']}  confidence: {plan.get('confidence', 0):.0%}")
        actions = plan.get("actions") or []
        if actions:
            click.echo(f"\n  Planned actions ({len(actions)}):")
            for a in actions[:10]:
                click.echo(f"    • [{a.get('type','')}] {a.get('description','')}")
        executed = result.get("executed_actions") or []
        if executed:
            click.echo(f"\n  Executed ({len(executed)}):")
            for a in executed[:10]:
                _ok(f"    ✓ {a.get('description', a.get('type',''))}")
        errors = result.get("errors") or []
        if errors:
            click.echo(f"\n  Errors ({len(errors)}):")
            for e in errors[:5]:
                _err(f"    ✗ {e}")
        click.echo()
    except requests.HTTPError as e:
        _err(f"Pipeline failed: {e.response.status_code} {e.response.text}")
        sys.exit(1)

@incident.command("list")
@click.option("--limit", "-n", default=20, show_default=True)
@click.option("--json", "as_json", is_flag=True, default=False)
def incident_list(limit: int, as_json: bool):
    """List recent incidents from memory."""
    data = _get("/v1/memory/incidents", params={"limit": limit})
    if as_json:
        _print_json(data)
        return
    incidents = data if isinstance(data, list) else data.get("incidents", [])
    if not incidents:
        click.echo("  No incidents found.")
        return
    click.echo(f"\n  {'ID':<30} {'Severity':<10} {'Status':<15} {'Root cause'}")
    click.echo("  " + "─" * 80)
    for inc in incidents[:limit]:
        click.echo(
            f"  {inc.get('incident_id',''):<30} "
            f"{inc.get('severity',''):<10} "
            f"{inc.get('status',''):<15} "
            f"{(inc.get('root_cause') or '')[:50]}"
        )
    click.echo()

@incident.command("get")
@click.argument("incident_id")
@click.option("--json", "as_json", is_flag=True, default=False)
def incident_get(incident_id: str, as_json: bool):
    """Show full details for a specific incident."""
    data = _get(f"/v1/incidents/{incident_id}")
    if as_json:
        _print_json(data)
        return
    _print_json(data)


# ── Kubernetes ────────────────────────────────────────────────────────────────

@cli.group()
def k8s():
    """Kubernetes operations — pods, deployments, scaling."""
    pass

@k8s.command("pods")
@click.option("--namespace", "-n", default="default", show_default=True)
@click.option("--json", "as_json", is_flag=True, default=False)
def k8s_pods(namespace: str, as_json: bool):
    """List pods in a namespace."""
    data = _get("/v1/k8s/pods", params={"namespace": namespace})
    if as_json:
        _print_json(data)
        return
    pods = data.get("pods", []) if isinstance(data, dict) else data
    click.echo(f"\n  {'Name':<45} {'Namespace':<15} {'Status':<12} {'Restarts'}")
    click.echo("  " + "─" * 85)
    for p in pods:
        restarts = p.get("restart_count", 0) or 0
        status   = p.get("status", "")
        color    = "green" if status == "Running" else ("yellow" if status == "Pending" else "red")
        click.secho(
            f"  {p.get('name',''):<45} {p.get('namespace',''):<15} {status:<12} {restarts}",
            fg=color if restarts > 5 or status != "Running" else None,
        )
    click.echo()

@k8s.command("restart")
@click.argument("deployment")
@click.option("--namespace", "-n", default="default", show_default=True)
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def k8s_restart(deployment: str, namespace: str, yes: bool):
    """Restart a deployment (rolling restart)."""
    if not yes:
        click.confirm(f"  Restart deployment '{deployment}' in namespace '{namespace}'?", abort=True)
    result = _post("/v1/k8s/restart", {"deployment": deployment, "namespace": namespace})
    if result.get("success"):
        _ok(f"Restart triggered for {deployment}/{namespace}")
    else:
        _err(f"Restart failed: {result.get('error', result)}")

@k8s.command("scale")
@click.argument("deployment")
@click.argument("replicas", type=int)
@click.option("--namespace", "-n", default="default", show_default=True)
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def k8s_scale(deployment: str, replicas: int, namespace: str, yes: bool):
    """Scale a deployment to REPLICAS."""
    if not yes:
        click.confirm(f"  Scale '{deployment}' to {replicas} replicas in '{namespace}'?", abort=True)
    result = _post("/v1/k8s/scale", {
        "deployment": deployment, "namespace": namespace, "replicas": replicas,
    })
    if result.get("success"):
        _ok(f"Scaled {deployment} to {replicas} replicas")
    else:
        _err(f"Scale failed: {result.get('error', result)}")


# ── AWS ───────────────────────────────────────────────────────────────────────

@cli.group()
def aws():
    """AWS operations — EC2, ECS, Lambda, CloudWatch."""
    pass

@aws.command("alarms")
@click.option("--state", default="ALARM", type=click.Choice(["ALARM", "OK", "INSUFFICIENT_DATA"]))
@click.option("--json", "as_json", is_flag=True, default=False)
def aws_alarms(state: str, as_json: bool):
    """List CloudWatch alarms in a given state."""
    data = _get("/v1/aws/cloudwatch/alarms", params={"state": state})
    if as_json:
        _print_json(data)
        return
    alarms = data.get("alarms", []) if isinstance(data, dict) else data
    if not alarms:
        _ok(f"No alarms in state {state}")
        return
    click.echo(f"\n  {'Alarm':<50} {'State':<8} {'Namespace'}")
    click.echo("  " + "─" * 80)
    for a in alarms:
        s = a.get("state", "")
        click.secho(
            f"  {a.get('name',''):<50} {s:<8} {a.get('namespace','')}",
            fg="red" if s == "ALARM" else None,
        )
    click.echo()

@aws.command("ec2")
@click.option("--json", "as_json", is_flag=True, default=False)
def aws_ec2(as_json: bool):
    """List EC2 instances and their states."""
    data = _get("/v1/aws/ec2/instances")
    if as_json:
        _print_json(data)
        return
    instances = data.get("instances", []) if isinstance(data, dict) else data
    click.echo(f"\n  {'ID':<20} {'Name':<30} {'State':<12} {'Type'}")
    click.echo("  " + "─" * 75)
    for i in instances:
        state = i.get("state", "")
        click.secho(
            f"  {i.get('id',''):<20} {i.get('name',''):<30} {state:<12} {i.get('type','')}",
            fg="green" if state == "running" else ("red" if state == "stopped" else None),
        )
    click.echo()

@aws.command("cost")
@click.option("--days", default=30, show_default=True)
@click.option("--json", "as_json", is_flag=True, default=False)
def aws_cost(days: int, as_json: bool):
    """Show AWS cost summary for the last N days."""
    data = _get("/v1/cost/summary", params={"days": days})
    if as_json:
        _print_json(data)
        return
    _print_json(data)


# ── War room ──────────────────────────────────────────────────────────────────

@cli.group()
def warroom():
    """War room management."""
    pass

@warroom.command("list")
@click.option("--json", "as_json", is_flag=True, default=False)
def warroom_list(as_json: bool):
    """List active war rooms."""
    data = _get("/v1/war-rooms")
    if as_json:
        _print_json(data)
        return
    rooms = data if isinstance(data, list) else data.get("war_rooms", [])
    if not rooms:
        click.echo("  No active war rooms.")
        return
    click.echo(f"\n  {'ID':<30} {'Incident':<30} {'Slack channel'}")
    click.echo("  " + "─" * 75)
    for r in rooms:
        click.echo(
            f"  {r.get('id',''):<30} {r.get('incident_id',''):<30} {r.get('slack_channel','')}"
        )
    click.echo()

@warroom.command("open")
@click.argument("incident_id")
def warroom_open(incident_id: str):
    """Open (or get) the war room for an incident."""
    data = _post("/v1/war-rooms", {"incident_id": incident_id})
    _print_json(data)


# ── Chat ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("message")
@click.option("--json", "as_json", is_flag=True, default=False)
def chat(message: str, as_json: bool):
    """Send a one-shot message to the AI DevOps assistant.

    Example:
        nexusops chat "why is the payments service slow?"
    """
    data = _post("/v1/chat", {"message": message})
    if as_json:
        _print_json(data)
        return
    reply = data.get("response") or data.get("message") or json.dumps(data)
    click.echo(f"\n  {reply}\n")


# ── Approvals ─────────────────────────────────────────────────────────────────

@cli.group()
def approvals():
    """Manage pending AI action approvals."""
    pass

@approvals.command("list")
@click.option("--json", "as_json", is_flag=True, default=False)
def approvals_list(as_json: bool):
    """List pending approvals."""
    data = _get("/v1/approvals/pending")
    if as_json:
        _print_json(data)
        return
    items = data if isinstance(data, list) else data.get("approvals", [])
    if not items:
        _ok("No pending approvals.")
        return
    for item in items:
        click.echo(f"\n  Approval ID: {item.get('approval_id')}")
        click.echo(f"  Incident:    {item.get('incident_id')}")
        click.echo(f"  Reason:      {item.get('reason')}")
        actions = item.get("actions") or []
        for a in actions:
            click.echo(f"    • {a.get('description', a.get('type', ''))}")

@approvals.command("approve")
@click.argument("approval_id")
def approvals_approve(approval_id: str):
    """Approve a pending AI action plan."""
    result = _post(f"/v1/approvals/{approval_id}/approve", {})
    _ok(f"Approved: {approval_id}")
    _print_json(result)

@approvals.command("reject")
@click.argument("approval_id")
@click.option("--reason", default="Rejected via CLI")
def approvals_reject(approval_id: str, reason: str):
    """Reject a pending AI action plan."""
    result = _post(f"/v1/approvals/{approval_id}/reject", {"reason": reason})
    _warn(f"Rejected: {approval_id}")
    _print_json(result)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
