"""
Health check and integration status routes.
Paths: /health/*, /health/integrations
"""
from fastapi import APIRouter, Depends
from app.routes.deps import require_viewer, AuthContext, _METRICS, _METRICS_HIST, _WAR_ROOMS
from app.core.config import settings as _settings
from app.integrations.universal_collector import collect_all_context, summarize_health

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    incident_count = 0
    try:
        from app.memory.vector_db import search_similar_incidents
        results = search_similar_incidents("incident", n_results=100)
        incident_count = len(results) if isinstance(results, list) else 0
    except Exception:
        pass
    return {
        "status":           "ok",
        "incident_count":   incident_count,
        "version":          "2.0.0",
        "monitor_loop":     _settings.ENABLE_MONITOR_LOOP,
        "auto_remediate":   _settings.AUTO_REMEDIATE_ON_MONITOR,
    }


@router.get("/health/live", tags=["health"])
def health_live():
    """Liveness probe — returns 200 if the process is alive."""
    return {"status": "alive"}


@router.get("/health/ready", tags=["health"])
def health_ready():
    """Readiness probe — checks ChromaDB, required env vars, and module imports."""
    from fastapi.responses import JSONResponse
    checks: dict = {}
    all_ok = True

    try:
        from app.llm import claude as _claude_mod  # noqa: F401
        from app.memory import vector_db as _vdb_mod  # noqa: F401
        checks["modules"] = "ok"
    except Exception as exc:
        checks["modules"] = f"error: {exc}"
        all_ok = False

    try:
        from app.memory.vector_db import search_similar_incidents
        search_similar_incidents("probe", n_results=1)
        checks["chroma"] = "ok"
    except Exception as exc:
        checks["chroma"] = f"error: {exc}"
        all_ok = False

    import os as _os
    llm_keys = ["ANTHROPIC_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"]
    checks["llm_key"] = "ok" if any(_os.getenv(k) for k in llm_keys) else "missing"
    if checks["llm_key"] != "ok":
        all_ok = False

    status = "ready" if all_ok else "not_ready"
    code = 200 if all_ok else 503
    return JSONResponse(status_code=code, content={"status": status, "checks": checks})


@router.get("/health/full")
def health_full():
    """Full health check — AWS, K8s, Grafana, Linux node, and all integrations."""
    import concurrent.futures as _cf
    from app.plugins.linux_checker import check_linux_node
    from app.plugins.grafana_checker import check_grafana

    # Run all slow checks in parallel with tight timeouts
    context: dict = {}
    linux: dict = {}
    grafana: dict = {}

    def _get_context():
        try:
            return collect_all_context(hours=1)
        except Exception:
            return {}

    def _get_linux():
        try:
            return check_linux_node()
        except Exception:
            return {"status": "unavailable", "success": False}

    def _get_grafana():
        try:
            return check_grafana()
        except Exception:
            return {"status": "unavailable", "success": False}

    pool = _cf.ThreadPoolExecutor(max_workers=3)
    f_ctx     = pool.submit(_get_context)
    f_linux   = pool.submit(_get_linux)
    f_grafana = pool.submit(_get_grafana)
    try:
        context = f_ctx.result(timeout=5)
    except Exception:
        context = {}
    try:
        linux = f_linux.result(timeout=3)
    except Exception:
        linux = {"status": "unavailable", "success": False}
    try:
        grafana = f_grafana.result(timeout=3)
    except Exception:
        grafana = {"status": "unavailable", "success": False}
    pool.shutdown(wait=False)  # don't block on slow threads

    health  = summarize_health(context)
    overall = "healthy" if health["healthy"] else "degraded"
    if grafana.get("firing_alerts", 0) > 0:
        overall = "degraded"

    return {
        "status":     overall,
        "health":     health,
        "linux_node": linux,
        "grafana":    grafana,
    }


@router.get("/health/integrations")
def health_integrations():
    """Diagnostic endpoint — shows exactly which integrations and LLM providers are configured."""
    from app.llm.claude import _provider, _provider_warning, ANTHROPIC_API_KEY, GROQ_API_KEY, OLLAMA_HOST
    import os

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
        "ollama": {"host": OLLAMA_HOST},
        "openai": {"key_set": bool(os.getenv("OPENAI_API_KEY", "").strip())},
    }

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
    }

    def _env_set(key: str) -> bool:
        v = os.getenv(key, "").strip()
        if not v:
            return False
        _placeholders = ("your_", "your-", "example", "placeholder", "changeme", "xxx", "<", "TODO")
        return not any(p in v.lower() for p in _placeholders)

    integrations = {
        "slack":    {"configured": _env_set("SLACK_BOT_TOKEN")},
        "jira":     {"configured": _env_set("JIRA_URL") and _env_set("JIRA_TOKEN")},
        "opsgenie": {"configured": _env_set("OPSGENIE_API_KEY")},
        "aws":      {"configured": _env_set("AWS_ACCESS_KEY_ID") or bool(os.getenv("AWS_PROFILE"))},
        "k8s":      {"configured": bool(os.getenv("KUBECONFIG")) or os.getenv("K8S_IN_CLUSTER", "").lower() == "true"},
        "grafana":  {"configured": _env_set("GRAFANA_URL") and _env_set("GRAFANA_TOKEN")},
        "gitlab":   {"configured": _env_set("GITLAB_TOKEN")},
    }

    return {"llm": llm, "github": github, "integrations": integrations}
