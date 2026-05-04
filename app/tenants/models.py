"""Tenant and Workspace data models."""
from __future__ import annotations

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, field_validator


class Plan(BaseModel):
    name: str                           # starter | pro | enterprise
    display_name: str
    price_monthly_cents: int = 0
    price_yearly_cents: int = 0
    max_seats: Optional[int] = None
    max_actions_per_mo: Optional[int] = None
    max_agent_runs_per_mo: Optional[int] = None
    max_llm_tokens_per_mo: Optional[int] = None
    max_training_examples: Optional[int] = None
    features: Dict[str, Any] = {}

    def allows(self, feature: str) -> bool:
        return bool(self.features.get(feature, False))

    def within_limit(self, resource: str, current: int) -> bool:
        limit = getattr(self, f"max_{resource}", None)
        return limit is None or current < limit


class Workspace(BaseModel):
    id: str                             # UUID
    slug: str                           # nexusops subdomain slug
    name: str
    owner_user_id: str
    plan_name: str = "starter"
    # Stripe
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    subscription_status: str = "trialing"   # trialing|active|past_due|cancelled
    trial_ends_at: Optional[str] = None
    current_period_start: Optional[str] = None
    current_period_end: Optional[str] = None
    # Meta
    logo_url: Optional[str] = None
    metadata: Dict[str, Any] = {}
    active: bool = True
    created_at: Optional[str] = None

    @property
    def is_active_subscription(self) -> bool:
        return self.subscription_status in ("active", "trialing")


class WorkspaceMember(BaseModel):
    id: Optional[int] = None
    workspace_id: str
    username: str
    email: Optional[str] = None
    role: str = "developer"
    invited_by: Optional[str] = None
    joined_at: Optional[str] = None
    active: bool = True


class Tenant(BaseModel):
    tenant_id: str
    name: str
    workspace_id: Optional[str] = None
    plan_name: str = "starter"
    aws_role_arn: Optional[str] = None
    aws_region: Optional[str] = None
    slack_channel: Optional[str] = "#incidents"
    llm_provider: Optional[str] = ""
    metadata: Optional[Dict[str, Any]] = {}
    created_at: Optional[str] = None
    active: bool = True


class UsageSummary(BaseModel):
    tenant_id: str
    workspace_id: Optional[str] = None
    billed_period: str              # YYYY-MM
    llm_tokens: int = 0
    agent_runs: int = 0
    actions: int = 0
    api_calls: int = 0
    # plan limits for comparison
    plan_name: str = "starter"
    limits: Dict[str, Any] = {}


class TrainingExample(BaseModel):
    id: Optional[int] = None
    tenant_id: str
    workspace_id: Optional[str] = None
    trigger_text: str
    correct_plan: Dict[str, Any] = {}
    correct_action: Optional[str] = None
    correct_params: Dict[str, Any] = {}
    outcome: str = "resolved"
    tags: List[str] = []
    source: str = "manual"
    quality_score: float = 1.0
    approved: bool = True
    created_by: Optional[str] = None
    created_at: Optional[str] = None
