"""
Miscellaneous routes: LLM settings, correlate, metrics, grafana, secrets, audit, rate-limit,
                      integration config.
Paths: /audit/log, /rate-limit/status, /settings/llm, /llm/analyze, /correlate, /metrics,
       /grafana/*, /secrets/status, /integrations/configure, /integrations/test/{key}
"""
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from typing import Dict

from app.api.deps import require_viewer, require_developer, AuthContext

router = APIRouter(tags=["misc"])


class LLMSettingPayload(BaseModel):
    provider: str  # "anthropic" | "openai" | "groq" | "ollama" | ""


class IntegrationConfigPayload(BaseModel):
    integration: str          # e.g. "slack", "github", "aws"
    config: Dict[str, str]    # env-var-name -> value


@router.get("/audit/log")
def get_audit_log_endpoint(limit: int = 50, user: str = "", action: str = "",
                            _: AuthContext = Depends(require_viewer)):
    """Return recent audit log entries (newest first). Filter by user or action."""
    from app.core.audit import get_audit_log
    return {"entries": get_audit_log(limit=limit, user=user, action=action)}


@router.get("/rate-limit/status")
def rate_limit_status(x_user: str = Header(default="anonymous")):
    """Return current rate-limit usage for the calling user."""
    from app.core.ratelimit import get_usage
    return get_usage(x_user)


@router.post("/settings/llm")
def set_llm_provider(payload: LLMSettingPayload, _: AuthContext = Depends(require_viewer)):
    """Set the global LLM provider used by chat, pipeline, and all agents."""
    from app.llm.factory import set_global_provider, get_global_provider
    provider = payload.provider.lower().strip()
    valid = {"anthropic", "claude", "openai", "groq", "ollama", ""}
    if provider not in valid:
        raise HTTPException(status_code=400, detail=f"Invalid provider. Choose from: {valid}")
    set_global_provider(provider)
    return {"ok": True, "provider": get_global_provider()}


@router.get("/settings/llm")
def get_llm_provider(_: AuthContext = Depends(require_viewer)):
    """Get the current global LLM provider and its status."""
    from app.llm.factory import get_global_provider
    from app.llm.claude import _provider, _provider_warning
    return {
        "provider":         get_global_provider(),
        "active_provider":  _provider,
        "warning":          _provider_warning,
    }


@router.post("/llm/analyze")
def llm_analyze(payload: dict, _: AuthContext = Depends(require_viewer)):
    """Run LLM analysis on arbitrary context dict."""
    from app.llm.claude import analyze_context
    return analyze_context(payload)


@router.post("/correlate")
def correlate_events_endpoint(events: list, _: AuthContext = Depends(require_viewer)):
    """Correlate a list of events and return grouped clusters."""
    from app.agents.correlator import correlate_events
    return {"correlation": correlate_events(events)}


@router.get("/metrics")
def prometheus_metrics():
    """Expose Prometheus-format metrics."""
    from app.api.deps import _METRICS
    lines = [f"nsops_{k} {v}" for k, v in _METRICS.items()]
    return "\n".join(lines)


@router.get("/grafana/alerts")
def grafana_alerts(_: AuthContext = Depends(require_viewer)):
    """Firing Grafana alerts."""
    from app.integrations.grafana import get_firing_alerts
    return get_firing_alerts()


@router.get("/grafana/dashboards")
def grafana_dashboards(_: AuthContext = Depends(require_viewer)):
    """Grafana datasources."""
    from app.integrations.grafana import get_datasources
    return get_datasources()


@router.get("/secrets/status")
def secrets_status(_: AuthContext = Depends(require_developer)):
    """Return which secrets/env vars are configured (values never returned)."""
    import os
    keys = [
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY",
        "GITHUB_TOKEN", "SLACK_BOT_TOKEN", "JIRA_URL", "JIRA_TOKEN",
        "OPSGENIE_API_KEY", "AWS_ACCESS_KEY_ID", "KUBECONFIG",
        "GRAFANA_URL", "GRAFANA_TOKEN", "GITLAB_TOKEN",
    ]
    return {k: bool(os.getenv(k, "").strip()) for k in keys}


# Backward-compatible alias — kept for any clients using the old path
@router.get("/secrets", include_in_schema=False)
def list_secrets(_: AuthContext = Depends(require_developer)):
    return secrets_status(_)


# ── Integration configuration ──────────────────────────────────

_INTEGRATION_ENV_KEYS: dict = {
    "slack":      ["SLACK_BOT_TOKEN", "SLACK_SIGNING_SECRET", "SLACK_BOT_USER_ID"],
    "github":     ["GITHUB_TOKEN", "GITHUB_REPO"],
    "gitlab":     ["GITLAB_TOKEN", "GITLAB_URL"],
    "aws":        ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"],
    "kubernetes": ["KUBECONFIG", "K8S_NAMESPACE", "K8S_CONTEXT"],
    "grafana":    ["GRAFANA_URL", "GRAFANA_TOKEN"],
    "pagerduty":  ["PAGERDUTY_TOKEN"],
    "opsgenie":   ["OPSGENIE_API_KEY"],
    "jira":       ["JIRA_URL", "JIRA_EMAIL", "JIRA_TOKEN"],
    "anthropic":  ["ANTHROPIC_API_KEY"],
    "openai":     ["OPENAI_API_KEY"],
    "groq":       ["GROQ_API_KEY"],
    "ollama":     ["OLLAMA_BASE_URL"],
}


@router.post("/integrations/configure")
def configure_integration(payload: IntegrationConfigPayload, auth: AuthContext = Depends(require_developer)):
    """
    Persist integration credentials as environment variables and write to .env file.
    Only admin / superadmin (developer+) can call this.
    Values are never logged.
    """
    import os
    from pathlib import Path

    key = payload.integration.lower()
    allowed_keys = set(_INTEGRATION_ENV_KEYS.get(key, []))
    if not allowed_keys:
        raise HTTPException(status_code=400, detail=f"Unknown integration: {key}")

    # Only accept known env-var names for this integration
    rejected = [k for k in payload.config if k not in allowed_keys]
    if rejected:
        raise HTTPException(status_code=400, detail=f"Unknown config keys: {rejected}")

    # Apply to current process environment
    for env_key, value in payload.config.items():
        if value:
            os.environ[env_key] = value

    # Persist to .env file so changes survive restart
    env_path = Path(__file__).resolve().parents[2] / ".env"
    existing: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()

    for env_key, value in payload.config.items():
        if value:
            existing[env_key] = value

    lines = [f"{k}={v}" for k, v in existing.items()]
    env_path.write_text("\n".join(lines) + "\n")

    from app.core.audit import log_action
    log_action(auth.username, "configure_integration", {"integration": key, "keys": list(payload.config.keys())})

    return {"ok": True, "integration": key, "updated_keys": list(payload.config.keys())}


@router.get("/integrations/fields/{integration_key}")
def get_integration_fields(integration_key: str, _: AuthContext = Depends(require_viewer)):
    """Return the list of env-var names required for a given integration."""
    key = integration_key.lower()
    fields = _INTEGRATION_ENV_KEYS.get(key)
    if fields is None:
        raise HTTPException(status_code=404, detail=f"Unknown integration: {key}")
    import os
    return {
        "integration": key,
        "fields": [{"name": f, "configured": bool(os.getenv(f, "").strip())} for f in fields],
    }


@router.post("/integrations/test/{integration_key}")
def test_integration_endpoint(integration_key: str, _: AuthContext = Depends(require_viewer)):
    """Run a lightweight connectivity test for the given integration."""
    import os
    key = integration_key.lower()
    try:
        if key == "slack":
            from app.integrations.slack import _client
            resp = _client().auth_test()
            return {"ok": resp.get("ok", False), "detail": resp.get("team", "Connected")}
        if key == "github":
            from app.integrations.github import get_recent_commits
            r = get_recent_commits(hours=1)
            return {"ok": True, "detail": f"GitHub OK — {len(r.get('commits',[]))} recent commits"}
        if key == "aws":
            from app.integrations.aws_ops import list_ec2_instances
            r = list_ec2_instances()
            return {"ok": True, "detail": f"AWS OK — {len(r.get('instances',[]))} EC2 instances visible"}
        if key == "kubernetes":
            from app.integrations.k8s_ops import list_deployments
            r = list_deployments()
            return {"ok": True, "detail": f"K8s OK — {len(r.get('deployments',[]))} deployments"}
        if key == "grafana":
            from app.integrations.grafana import get_firing_alerts
            r = get_firing_alerts()
            return {"ok": True, "detail": f"Grafana OK — {len(r.get('firing_alerts',[]))} alerts"}
        if key == "anthropic":
            from app.llm.claude import _llm
            r = _llm("Say 'pong'", [{"role": "user", "content": "ping"}], max_tokens=5)
            return {"ok": bool(r), "detail": "Anthropic API reachable"}
        if key == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            return {"ok": bool(api_key), "detail": "OpenAI key present" if api_key else "OPENAI_API_KEY not set"}
        if key == "groq":
            api_key = os.getenv("GROQ_API_KEY", "")
            return {"ok": bool(api_key), "detail": "Groq key present" if api_key else "GROQ_API_KEY not set"}
        if key in ("pagerduty", "opsgenie", "jira", "gitlab", "ollama"):
            env_keys = _INTEGRATION_ENV_KEYS.get(key, [])
            configured = all(os.getenv(k, "").strip() for k in env_keys)
            return {"ok": configured, "detail": "Keys configured" if configured else "One or more keys missing"}
        return {"ok": False, "detail": f"No test implemented for {key}"}
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}
