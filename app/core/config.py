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
    # Risk levels that auto-execute without human approval: low, medium
    AUTO_EXECUTE_RISK_LEVELS: list[str] = ["low", "medium"]

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

    # ── Server ─────────────────────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"
    RBAC_CONFIG_PATH: str = ""

    model_config = {"env_file": str(Path(__file__).resolve().parents[2] / ".env"),
                    "extra": "ignore"}


settings = Settings()
