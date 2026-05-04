"""SaaS onboarding, workspace management, billing, and training API.

Routes (all under /v1 prefix set in main.py):

  Auth / Onboarding
  POST   /v1/auth/register                      — self-serve signup

  Org (workspace)
  GET    /v1/org                                — get current org
  PUT    /v1/org                                — update org settings
  GET    /v1/org/members                        — list members
  POST   /v1/org/members                        — invite a member
  DELETE /v1/org/members/{username}             — remove member

  Billing & Plans
  GET    /v1/billing/plans                      — list Starter/Pro/Enterprise
  POST   /v1/billing/checkout                   — Stripe checkout session
  POST   /v1/billing/portal                     — Stripe billing portal

  Usage
  GET    /v1/usage                              — current period usage + limits
  GET    /v1/usage/history                      — past N months

  Webhooks (external callers)
  POST   /v1/webhooks/stripe                    — Stripe event receiver

  Training
  GET    /v1/training                           — list training examples
  POST   /v1/training                           — add example
  PUT    /v1/training/{id}/approve              — approve example
  DELETE /v1/training/{id}                      — delete example
  POST   /v1/training/export                    — export as Alpaca JSONL
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional, List

from fastapi import APIRouter, Depends, Header, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from app.api.deps import require_viewer, require_developer, AuthContext
from app.tenants.store import (
    get_workspace, get_workspace_by_slug, create_workspace, update_workspace,
    list_workspace_members, add_workspace_member, get_member_role,
    get_tenant, create_tenant, update_tenant, list_plans, get_plan,
    get_usage_summary, record_usage, check_limit,
    add_training_example, list_training_examples,
    approve_training_example, delete_training_example,
)
from app.tenants.models import Tenant, TrainingExample
from app.core.audit import audit_log

router = APIRouter(tags=["saas"])

STRIPE_SECRET_KEY    = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
APP_URL              = os.getenv("APP_URL", "http://localhost:8000")


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    username:   str
    email:      str
    password:   str
    workspace_name: str
    workspace_slug: str  # will become subdomain slug
    plan_name:  str = "starter"

    @field_validator("workspace_slug")
    @classmethod
    def validate_slug(cls, v: str) -> str:
        v = v.lower().strip()
        if not re.match(r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$", v):
            raise ValueError("Slug must be 3–63 chars, lowercase letters, numbers, hyphens only")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class InviteMemberRequest(BaseModel):
    username: str
    email:    str = ""
    role:     str = "developer"


class UpdateWorkspaceRequest(BaseModel):
    name:     Optional[str] = None
    logo_url: Optional[str] = None


class TrainingExampleRequest(BaseModel):
    trigger_text:   str
    correct_action: Optional[str] = None
    correct_params: dict = {}
    correct_plan:   dict = {}
    outcome:        str = "resolved"
    tags:           List[str] = []


# ─────────────────────────────────────────────────────────────────────────────
# Signup + Workspace
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/auth/register", summary="Self-serve signup — creates user + workspace")
def signup(req: SignupRequest):
    from app.security.users import create_user, user_exists
    from app.security.rbac import assign_role

    # Validate slug not taken
    existing = get_workspace_by_slug(req.workspace_slug)
    if existing:
        raise HTTPException(status_code=409, detail=f"Workspace slug '{req.workspace_slug}' already taken")

    # Validate plan exists
    plan = get_plan(req.plan_name)
    if not plan:
        raise HTTPException(status_code=400, detail=f"Plan '{req.plan_name}' does not exist")

    # Create user if not exists
    if not user_exists(req.username):
        result = create_user(req.username, req.password, email=req.email, role="admin")
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to create user"))
    else:
        raise HTTPException(status_code=409, detail=f"Username '{req.username}' already taken")

    assign_role(req.username, "admin", changed_by="signup")

    # Create workspace
    ws = create_workspace(
        slug=req.workspace_slug,
        name=req.workspace_name,
        owner_user_id=req.username,
        plan_name=req.plan_name,
    )

    # Create tenant record (1:1 with workspace for now)
    tenant = create_tenant(Tenant(
        tenant_id=req.workspace_slug,
        name=req.workspace_name,
        workspace_id=ws.id,
        plan_name=req.plan_name,
    ))

    audit_log(user=req.username, action="signup", params={"workspace": req.workspace_slug, "plan": req.plan_name},
              result={"success": True}, source="api")

    return {
        "success": True,
        "workspace_id": ws.id,
        "workspace_slug": ws.slug,
        "tenant_id": tenant.tenant_id,
        "plan": req.plan_name,
        "trial_ends_at": ws.trial_ends_at,
        "message": f"Welcome! Your workspace '{ws.name}' is ready.",
    }


@router.get("/org", summary="Get current workspace")
def get_current_workspace(auth: AuthContext = Depends(require_viewer)):
    tenant = get_tenant(auth.tenant_id)
    if not tenant or not tenant.workspace_id:
        raise HTTPException(status_code=404, detail="Workspace not found")
    ws = get_workspace(tenant.workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    plan = get_plan(ws.plan_name)
    return {
        "workspace": ws.model_dump(),
        "plan": plan.model_dump() if plan else None,
    }


@router.put("/org", summary="Update workspace settings")
def update_current_workspace(req: UpdateWorkspaceRequest, auth: AuthContext = Depends(require_developer)):
    tenant = get_tenant(auth.tenant_id)
    if not tenant or not tenant.workspace_id:
        raise HTTPException(status_code=404, detail="Workspace not found")
    role = get_member_role(tenant.workspace_id, auth.username)
    if role != "admin":
        raise HTTPException(status_code=403, detail="Only workspace admins can update settings")
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    ws = update_workspace(tenant.workspace_id, updates)
    return {"workspace": ws.model_dump() if ws else None}


# ─────────────────────────────────────────────────────────────────────────────
# Members
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/org/members")
def list_members(auth: AuthContext = Depends(require_viewer)):
    tenant = get_tenant(auth.tenant_id)
    if not tenant or not tenant.workspace_id:
        raise HTTPException(status_code=404, detail="Workspace not found")
    members = list_workspace_members(tenant.workspace_id)
    return {"members": [m.model_dump() for m in members]}


@router.post("/org/members")
def invite_member(req: InviteMemberRequest, auth: AuthContext = Depends(require_developer)):
    tenant = get_tenant(auth.tenant_id)
    if not tenant or not tenant.workspace_id:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Enforce seat limit
    plan = get_plan(tenant.plan_name)
    if plan and plan.max_seats:
        current_members = list_workspace_members(tenant.workspace_id)
        if len(current_members) >= plan.max_seats:
            raise HTTPException(
                status_code=402,
                detail=f"Seat limit reached ({plan.max_seats}). Upgrade your plan to add more members."
            )

    member = add_workspace_member(
        workspace_id=tenant.workspace_id,
        username=req.username,
        email=req.email,
        role=req.role,
        invited_by=auth.username,
    )
    audit_log(user=auth.username, action="invite_member",
              params={"username": req.username, "role": req.role},
              result={"success": True}, source="api")
    return {"member": member.model_dump()}


@router.delete("/org/members/{username}")
def remove_member(username: str, auth: AuthContext = Depends(require_developer)):
    tenant = get_tenant(auth.tenant_id)
    if not tenant or not tenant.workspace_id:
        raise HTTPException(status_code=404, detail="Workspace not found")
    role = get_member_role(tenant.workspace_id, auth.username)
    if role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can remove members")
    if username == auth.username:
        raise HTTPException(status_code=400, detail="Cannot remove yourself from the workspace")
    from app.core.database import execute
    execute(
        "UPDATE workspace_members SET active = false WHERE workspace_id = %s AND username = %s",
        (tenant.workspace_id, username)
    )
    return {"success": True, "removed": username}


# ─────────────────────────────────────────────────────────────────────────────
# Plans + Usage
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/billing/plans", summary="List available plans")
def get_plans():
    plans = list_plans()
    return {"plans": [p.model_dump() for p in plans]}


@router.get("/usage", summary="Current period usage and limits")
def get_usage(auth: AuthContext = Depends(require_viewer)):
    summary = get_usage_summary(auth.tenant_id)
    return summary.model_dump()


@router.get("/usage/history", summary="Usage history for past N months")
def get_usage_history(months: int = 3, auth: AuthContext = Depends(require_viewer)):
    import datetime
    result = []
    for i in range(months):
        dt = datetime.datetime.utcnow() - datetime.timedelta(days=30 * i)
        period = dt.strftime("%Y-%m")
        summary = get_usage_summary(auth.tenant_id, period=period)
        result.append(summary.model_dump())
    return {"history": result}


# ─────────────────────────────────────────────────────────────────────────────
# Stripe Billing
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/billing/checkout", summary="Create Stripe checkout session")
def create_checkout(plan_name: str, billing_cycle: str = "monthly",
                    auth: AuthContext = Depends(require_viewer)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=501, detail="Stripe not configured — set STRIPE_SECRET_KEY")

    plan = get_plan(plan_name)
    if not plan:
        raise HTTPException(status_code=400, detail=f"Plan '{plan_name}' not found")

    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY

        tenant = get_tenant(auth.tenant_id)
        ws = get_workspace(tenant.workspace_id) if tenant and tenant.workspace_id else None

        # Get or create Stripe customer
        customer_id = ws.stripe_customer_id if ws else None
        if not customer_id:
            customer = stripe.Customer.create(
                email=auth.username + "@" + (auth.tenant_id or "nexusops"),
                metadata={"tenant_id": auth.tenant_id, "workspace_slug": auth.tenant_id},
            )
            customer_id = customer.id
            if ws:
                update_workspace(ws.id, {"stripe_customer_id": customer_id})

        price_cents = plan.price_yearly_cents if billing_cycle == "yearly" else plan.price_monthly_cents
        interval = "year" if billing_cycle == "yearly" else "month"

        # Create a price on the fly (or use pre-created price IDs from env)
        price_env_key = f"STRIPE_PRICE_{plan_name.upper()}_{interval.upper()}"
        price_id = os.getenv(price_env_key)
        if not price_id:
            price_obj = stripe.Price.create(
                unit_amount=price_cents,
                currency="usd",
                recurring={"interval": interval},
                product_data={"name": f"NexusOps {plan.display_name}"},
            )
            price_id = price_obj.id

        session = stripe.checkout.Session.create(
            customer=customer_id,
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{APP_URL}/settings/billing?success=true&plan={plan_name}",
            cancel_url=f"{APP_URL}/settings/billing?cancelled=true",
            metadata={"tenant_id": auth.tenant_id, "plan_name": plan_name},
            subscription_data={"trial_period_days": 0},
        )
        return {"checkout_url": session.url, "session_id": session.id}

    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Stripe error: {exc}")


@router.post("/billing/portal", summary="Open Stripe billing portal")
def billing_portal(auth: AuthContext = Depends(require_viewer)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=501, detail="Stripe not configured")

    tenant = get_tenant(auth.tenant_id)
    ws = get_workspace(tenant.workspace_id) if tenant and tenant.workspace_id else None
    if not ws or not ws.stripe_customer_id:
        raise HTTPException(status_code=400, detail="No billing account found. Complete checkout first.")

    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY
        session = stripe.billing_portal.Session.create(
            customer=ws.stripe_customer_id,
            return_url=f"{APP_URL}/settings/billing",
        )
        return {"portal_url": session.url}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Stripe error: {exc}")


@router.post("/webhooks/stripe", summary="Stripe webhook receiver", include_in_schema=False)
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=501, detail="Webhook secret not configured")

    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")

    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid webhook: {exc}")

    background_tasks.add_task(_handle_stripe_event, event)
    return {"received": True}


def _handle_stripe_event(event: dict) -> None:
    from app.core.database import execute
    event_type = event.get("type", "")
    event_id   = event.get("id", "")

    try:
        # Idempotency — skip if already processed
        existing = execute(
            "SELECT id FROM billing_events WHERE stripe_event_id = %s", (event_id,)
        )
        if existing:
            return

        execute(
            "INSERT INTO billing_events (stripe_event_id, event_type, payload) VALUES (%s, %s, %s)",
            (event_id, event_type, json.dumps(event))
        )

        data = event.get("data", {}).get("object", {})
        tenant_id = (data.get("metadata") or {}).get("tenant_id")

        if event_type == "checkout.session.completed":
            sub_id = data.get("subscription")
            customer_id = data.get("customer")
            plan_name = (data.get("metadata") or {}).get("plan_name", "starter")
            if tenant_id:
                tenant = get_tenant(tenant_id)
                if tenant and tenant.workspace_id:
                    update_workspace(tenant.workspace_id, {
                        "stripe_subscription_id": sub_id,
                        "stripe_customer_id":     customer_id,
                        "subscription_status":    "active",
                        "plan_name":              plan_name,
                    })
                    update_tenant(tenant_id, {"plan_name": plan_name})

        elif event_type in ("invoice.paid",):
            sub = data.get("subscription")
            if tenant_id:
                tenant = get_tenant(tenant_id)
                if tenant and tenant.workspace_id:
                    update_workspace(tenant.workspace_id, {"subscription_status": "active"})

        elif event_type in ("invoice.payment_failed", "customer.subscription.deleted"):
            if tenant_id:
                tenant = get_tenant(tenant_id)
                if tenant and tenant.workspace_id:
                    status = "cancelled" if "deleted" in event_type else "past_due"
                    update_workspace(tenant.workspace_id, {"subscription_status": status})

        execute(
            "UPDATE billing_events SET processed = true WHERE stripe_event_id = %s", (event_id,)
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("stripe_webhook_handler_error event=%s error=%s", event_type, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Training Examples
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/training", summary="List training examples for current tenant")
def get_training_examples(approved_only: bool = True, auth: AuthContext = Depends(require_viewer)):
    examples = list_training_examples(auth.tenant_id, approved_only=approved_only)
    return {"examples": [e.model_dump() for e in examples], "count": len(examples)}


@router.post("/training", summary="Add a manual training example")
def create_training_example(req: TrainingExampleRequest, auth: AuthContext = Depends(require_developer)):
    # Enforce per-plan limit
    plan = get_plan(get_tenant(auth.tenant_id).plan_name if get_tenant(auth.tenant_id) else "starter")
    if plan and plan.max_training_examples:
        existing = list_training_examples(auth.tenant_id)
        if len(existing) >= plan.max_training_examples:
            raise HTTPException(
                status_code=402,
                detail=f"Training example limit ({plan.max_training_examples}) reached. Upgrade to Pro for more."
            )

    tenant = get_tenant(auth.tenant_id)
    example = add_training_example(TrainingExample(
        tenant_id=auth.tenant_id,
        workspace_id=tenant.workspace_id if tenant else None,
        trigger_text=req.trigger_text,
        correct_action=req.correct_action,
        correct_params=req.correct_params,
        correct_plan=req.correct_plan,
        outcome=req.outcome,
        tags=req.tags,
        source="manual",
        created_by=auth.username,
    ))
    record_usage(auth.tenant_id, "training_example_added")
    return {"example": example.model_dump()}


@router.put("/training/{example_id}/approve", summary="Approve a training example")
def approve_example(example_id: int, auth: AuthContext = Depends(require_developer)):
    ok = approve_training_example(example_id, auth.tenant_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Example not found")
    return {"success": True, "id": example_id}


@router.delete("/training/{example_id}", summary="Delete a training example")
def delete_example(example_id: int, auth: AuthContext = Depends(require_developer)):
    ok = delete_training_example(example_id, auth.tenant_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Example not found")
    return {"success": True, "id": example_id}


@router.post("/training/export", summary="Export approved examples as JSONL for fine-tuning")
def export_training_data(auth: AuthContext = Depends(require_developer)):
    examples = list_training_examples(auth.tenant_id, approved_only=True, limit=10000)
    if not examples:
        return {"count": 0, "jsonl": ""}

    lines = []
    for ex in examples:
        record = {
            "instruction": ex.trigger_text,
            "input": "",
            "output": json.dumps(ex.correct_plan) if ex.correct_plan else (ex.correct_action or ""),
            "system": (
                "You are NexusOps, an expert AI DevOps assistant. "
                "Provide concise, actionable incident response plans."
            ),
            "metadata": {"action": ex.correct_action, "params": ex.correct_params, "outcome": ex.outcome},
        }
        lines.append(json.dumps(record))

    return {
        "count": len(lines),
        "jsonl": "\n".join(lines),
        "format": "alpaca",
    }
