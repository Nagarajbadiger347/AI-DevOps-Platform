"""Centralised configuration via pydantic-settings.

All os.getenv() calls across the app should import from here.
Values are loaded from .env automatically.
"""
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ────────────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    OLLAMA_HOST: str = "http://localhost:11434"

    # Preferred provider: claude | openai | groq | ollama
    LLM_PROVIDER: str = "claude"
    CLAUDE_MODEL: str = "claude-sonnet-4-6"
    OPENAI_MODEL: str = "gpt-4o"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # ── Safety / decision thresholds ───────────────────────────────────────
    # Plans with confidence below this always require human approval
    MIN_CONFIDENCE_THRESHOLD: float = 0.6
    # Risk levels that auto-execute without human approval.
    # Only "low" by default — medium and above require human approval.
    AUTO_EXECUTE_RISK_LEVELS: list[str] = ["low"]

    # ── Continuous monitoring loop ─────────────────────────────────────────
    ENABLE_MONITOR_LOOP: bool = True           # on by default — set False to disable
    MONITOR_INTERVAL_SECONDS: int = 60
    AUTO_REMEDIATE_ON_MONITOR: bool = False    # alert-only by default; set True for auto-fix

    # ── LLM safety ─────────────────────────────────────────────────────────
    LLM_TIMEOUT_SECONDS: int = 10              # per-attempt hard ceiling for LLM calls

    # ── Inbound webhook secrets ─────────────────────────────────────────────
    PAGERDUTY_WEBHOOK_SECRET: str = ""         # HMAC-SHA256 secret from PagerDuty console
    OPSGENIE_WEBHOOK_TOKEN:   str = ""         # bearer token set in OpsGenie integration
    GITHUB_WEBHOOK_SECRET:    str = ""         # HMAC-SHA256 secret on the GitHub side
    GRAFANA_WEBHOOK_TOKEN:    str = ""         # X-Webhook-Token sent by Grafana
    CLOUDWATCH_WEBHOOK_TOKEN: str = ""         # X-Webhook-Token sent by your SNS→HTTP shim

    # ── Deployment mode ─────────────────────────────────────────────────────
    # When ENVIRONMENT=production the platform refuses to start unless every
    # webhook secret above is set. Anything else (dev/staging/test) is open.
    ENVIRONMENT: str = "development"

    # ── Integrations ───────────────────────────────────────────────────────
    GITHUB_TOKEN: str = ""
    GITHUB_REPO: str = ""
    SLACK_BOT_TOKEN: str = ""
    SLACK_CHANNEL: str = "#incidents"
    JIRA_URL: str = ""
    JIRA_USER: str = ""
    JIRA_TOKEN: str = ""
    JIRA_PROJECT: str = ""
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-west-2"
    KUBECONFIG: str = ""
    K8S_IN_CLUSTER: bool = False
    OPSGENIE_API_KEY: str = ""

    # ── SSO / OAuth2 ────────────────────────────────────────────────────────
    SSO_PROVIDER:     str = ""   # google | github | "" (disabled)
    SSO_CLIENT_ID:    str = ""
    SSO_CLIENT_SECRET: str = ""
    SSO_REDIRECT_URI: str = ""
    SSO_DEFAULT_ROLE: str = "viewer"
    SSO_REQUIRE_INVITE: str = "false"   # set "true" to only allow pre-invited emails

    # ── JWT rotation ────────────────────────────────────────────────────────
    JWT_SECRET_KEY_OLD: str = ""   # previous key — kept during rotation grace period

    # ── PostgreSQL ──────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://nexusops:nexusops123@localhost:5432/nexusops"

    # ── Redis ───────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Redis HA ────────────────────────────────────────────────────────────
    REDIS_SENTINEL_HOSTS:  str = ""   # host1:26379,host2:26379
    REDIS_SENTINEL_MASTER: str = "mymaster"

    # ── Server ─────────────────────────────────────────────────────────────
    APP_URL:          str = "http://localhost:8000"
    CORS_ORIGINS:     str = "http://localhost:3000,http://localhost:8000"
    RBAC_CONFIG_PATH: str = ""

    # ── AWS extended ──────────────────────────────────────────────────
    AWS_SESSION_TOKEN: str = ""   # optional — for assumed roles / temporary credentials


    # ── Stripe ───────────────────────────────────────────────────
    STRIPE_SECRET_KEY:     str = ""
    STRIPE_WEBHOOK_SECRET: str = ""
    model_config = {"env_file": str(Path(__file__).resolve().parents[2] / ".env"),
                    "extra": "ignore"}


settings = Settings()


def validate_security() -> None:
    """In production mode, refuse to import (and therefore boot) unless every
    inbound-webhook secret is set. In dev mode, log warnings only.

    Called once at module import time from app.orchestrator.main."""
    import logging as _logging
    _log = _logging.getLogger("nsops.security")
    required = {
        "PAGERDUTY_WEBHOOK_SECRET": settings.PAGERDUTY_WEBHOOK_SECRET,
        "OPSGENIE_WEBHOOK_TOKEN":   settings.OPSGENIE_WEBHOOK_TOKEN,
        "GITHUB_WEBHOOK_SECRET":    settings.GITHUB_WEBHOOK_SECRET,
        "GRAFANA_WEBHOOK_TOKEN":    settings.GRAFANA_WEBHOOK_TOKEN,
        "CLOUDWATCH_WEBHOOK_TOKEN": settings.CLOUDWATCH_WEBHOOK_TOKEN,
        "JWT_SECRET_KEY":           __import__("os").getenv("JWT_SECRET_KEY", ""),
    }
    missing = [k for k, v in required.items() if not v]
    if not missing:
        return
    if settings.ENVIRONMENT.lower() == "production":
        raise RuntimeError(
            "refusing to start in production: missing required secrets: "
            + ", ".join(missing)
            + ". Set them in .env or unset ENVIRONMENT to disable this check."
        )
    _log.warning(
        "security_secrets_missing env=%s missing=%s — these endpoints are open",
        settings.ENVIRONMENT, ",".join(missing),
    )
