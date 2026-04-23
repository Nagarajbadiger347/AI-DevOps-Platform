"""
Health check and integration status routes.
Paths: /health/*, /health/integrations
"""
from fastapi import APIRouter, Depends
from app.api.deps import require_viewer, AuthContext, _METRICS, _METRICS_HIST, _WAR_ROOMS
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
    """Readiness probe — checks PostgreSQL, required env vars, and module imports."""
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
        from app.core.database import health_check
        checks["database"] = "ok" if health_check() else "error: unreachable"
        if checks["database"] != "ok":
            all_ok = False
    except Exception as exc:
        checks["database"] = f"error: {exc}"
        all_ok = False

    import os as _os
    llm_keys = ["ANTHROPIC_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"]
    checks["llm_key"] = "ok" if any(_os.getenv(k) for k in llm_keys) else "missing"
    if checks["llm_key"] != "ok":
        all_ok = False

    status = "ready" if all_ok else "not_ready"
    code = 200 if all_ok else 503
    return JSONResponse(status_code=code, content={"status": status, "checks": checks})


_health_full_cache: dict = {"data": None, "ts": 0.0}
_HEALTH_FULL_TTL = 30  # seconds

@router.get("/health/full")
def health_full():
    """Full health check — AWS, K8s, Grafana, Linux node, and all integrations."""
    import time as _time
    import concurrent.futures as _cf
    from app.integrations.linux_checker import check_linux_node
    from app.integrations.grafana_checker import check_grafana

    now = _time.time()
    if _health_full_cache["data"] and (now - _health_full_cache["ts"]) < _HEALTH_FULL_TTL:
        return _health_full_cache["data"]

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

    result = {
        "status":     overall,
        "health":     health,
        "linux_node": linux,
        "grafana":    grafana,
    }
    _health_full_cache["data"] = result
    _health_full_cache["ts"]   = _time.time()
    return result


@router.get("/health/detectors")
def health_detectors():
    """Expose monitoring detector health — which detectors are active vs stale."""
    import time as _time
    try:
        from app.monitoring.loop import _detector_health, _DETECTOR_WARN_AFTER
    except ImportError:
        return {"detectors": {}, "any_stale": False}
    now = _time.time()
    detectors = {}
    any_stale = False
    for name, last_ok in _detector_health.items():
        seconds_ago = int(now - last_ok)
        stale = seconds_ago > _DETECTOR_WARN_AFTER
        if stale:
            any_stale = True
        detectors[name] = {
            "last_success_seconds_ago": seconds_ago,
            "stale": stale,
            "status": "stale" if stale else "ok",
        }
    return {"detectors": detectors, "any_stale": any_stale}


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


@router.get("/health/cache")
def health_cache(_: AuthContext = Depends(require_viewer)):
    """LLM response cache stats — hit rate, size, TTL."""
    try:
        from app.core.llm_cache import llm_cache
        return {"cache": llm_cache.stats()}
    except Exception as exc:
        return {"cache": {}, "error": str(exc)}


@router.post("/health/cache/clear")
def health_cache_clear(_: AuthContext = Depends(require_viewer)):
    """Clear the LLM response cache."""
    try:
        from app.core.llm_cache import llm_cache
        llm_cache.clear()
        return {"cleared": True}
    except Exception as exc:
        return {"cleared": False, "error": str(exc)}
