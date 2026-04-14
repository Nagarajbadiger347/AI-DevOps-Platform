"""
Miscellaneous routes: LLM settings, correlate, metrics, grafana, secrets, audit, rate-limit.
Paths: /audit/log, /rate-limit/status, /settings/llm, /llm/analyze, /correlate, /metrics,
       /grafana/*, /secrets/status
"""
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from app.api.deps import require_viewer, require_developer, AuthContext

router = APIRouter(tags=["misc"])


class LLMSettingPayload(BaseModel):
    provider: str  # "anthropic" | "openai" | "groq" | "ollama" | ""


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
