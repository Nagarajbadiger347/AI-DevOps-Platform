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
    ENABLE_MONITOR_LOOP: bool = False          # disabled by default — enable in prod
    MONITOR_INTERVAL_SECONDS: int = 60
    AUTO_REMEDIATE_ON_MONITOR: bool = False    # alert-only by default

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
    DATABASE_URL: str = "postgresql://nexusops:nexusops@localhost:5432/nexusops"

    # ── Redis HA ────────────────────────────────────────────────────────────
    REDIS_SENTINEL_HOSTS:  str = ""   # host1:26379,host2:26379
    REDIS_SENTINEL_MASTER: str = "mymaster"

    # ── Server ─────────────────────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"
    RBAC_CONFIG_PATH: str = ""

    model_config = {"env_file": str(Path(__file__).resolve().parents[2] / ".env"),
                    "extra": "ignore"}


settings = Settings()
