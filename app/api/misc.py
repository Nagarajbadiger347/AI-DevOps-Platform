"""
Miscellaneous routes: LLM, correlate, metrics, grafana, secrets, audit, settings.
Paths: /correlate, /llm/*, /metrics, /audit/log, /rate-limit/*, /settings/*, /grafana/*, /secrets/*
"""
from typing import Optional
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from app.api.deps import require_viewer, require_developer, AuthContext

router = APIRouter(tags=["misc"])


class LLMSettingPayload(BaseModel):
    provider: str  # "anthropic" | "openai" | "groq" | "ollama" | ""


@router.get("/audit/log")
def get_audit_log_endpoint(limit: int = 50, user: str = "", action: str = "",
                            auth: AuthContext = Depends(require_viewer)):
    """Return recent audit log entries (newest first). Filter by user or action."""
    from app.core.audit import get_audit_log
    return {"entries": get_audit_log(limit=limit, user=user, action=action)}


@router.get("/rate-limit/status")
def rate_limit_status(x_user: str = Header(default="anonymous")):
    """Return current rate-limit usage for the calling user."""
    from app.core.ratelimit import get_usage
    return get_usage(x_user)


@router.post("/settings/llm")
def set_llm_provider(payload: LLMSettingPayload, auth: AuthContext = Depends(require_viewer)):
    """Set the global LLM provider used by chat, pipeline, and all agents."""
    from app.llm.factory import set_global_provider, get_global_provider
    provider = payload.provider.lower().strip()
    valid = {"anthropic", "claude", "openai", "groq", "ollama", ""}
    if provider not in valid:
        raise HTTPException(status_code=400, detail=f"Invalid provider. Choose from: {valid}")
    set_global_provider(provider)
    return {"ok": True, "provider": get_global_provider()}


@router.get("/settings/llm")
def get_llm_provider(auth: AuthContext = Depends(require_viewer)):
    """Get the current global LLM provider."""
    from app.llm.factory import get_global_provider
    return {"provider": get_global_provider()}


@router.get("/llm/status")
def llm_status(auth: AuthContext = Depends(require_viewer)):
    """Return current LLM provider status and available providers."""
    from app.llm.claude import _provider, _provider_warning
    return {"provider": _provider, "warning": _provider_warning}


@router.post("/llm/analyze")
def llm_analyze(payload: dict, auth: AuthContext = Depends(require_viewer)):
    """Run LLM analysis on arbitrary context dict."""
    from app.llm.claude import analyze_context
    return analyze_context(payload)


@router.post("/correlate")
def correlate_events_endpoint(events: list, auth: AuthContext = Depends(require_viewer)):
    """Correlate a list of events and return grouped clusters."""
    from app.agents.correlator import correlate_events
    return {"correlation": correlate_events(events)}


@router.get("/metrics")
def prometheus_metrics():
    """Expose Prometheus-format metrics."""
    from app.api.deps import _METRICS, _METRICS_HIST
    lines = []
    for k, v in _METRICS.items():
        lines.append(f"nsops_{k} {v}")
    return "\n".join(lines)


@router.get("/grafana/alerts")
def grafana_alerts(auth: AuthContext = Depends(require_viewer)):
    """Firing Grafana alerts."""
    from app.integrations.grafana import get_firing_alerts
    return get_firing_alerts()


@router.get("/grafana/dashboards")
def grafana_dashboards(auth: AuthContext = Depends(require_viewer)):
    """Grafana datasources."""
    from app.integrations.grafana import get_datasources
    return get_datasources()


@router.get("/secrets/status")
def secrets_status(auth: AuthContext = Depends(require_developer)):
    """Return which secrets/env vars are configured (no values)."""
    import os
    keys = [
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY",
        "GITHUB_TOKEN", "SLACK_BOT_TOKEN", "JIRA_URL", "JIRA_TOKEN",
        "OPSGENIE_API_KEY", "AWS_ACCESS_KEY_ID", "KUBECONFIG",
        "GRAFANA_URL", "GRAFANA_TOKEN", "GITLAB_TOKEN",
    ]
    return {k: bool(os.getenv(k, "").strip()) for k in keys}


@router.get("/secrets")
def list_secrets(auth: AuthContext = Depends(require_developer)):
    """Alias for /secrets/status."""
    import os
    keys = [
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY",
        "GITHUB_TOKEN", "SLACK_BOT_TOKEN", "JIRA_URL", "JIRA_TOKEN",
        "OPSGENIE_API_KEY", "AWS_ACCESS_KEY_ID", "KUBECONFIG",
        "GRAFANA_URL", "GRAFANA_TOKEN", "GITLAB_TOKEN",
    ]
    return {"secrets": {k: bool(os.getenv(k, "").strip()) for k in keys}}
