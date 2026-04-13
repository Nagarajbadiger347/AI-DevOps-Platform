"""Tenant data model."""
from typing import Optional, Dict, Any
from pydantic import BaseModel


class Tenant(BaseModel):
    tenant_id: str
    name: str
    aws_role_arn: Optional[str] = None       # STS role to assume for this tenant
    aws_region: Optional[str] = None
    slack_channel: Optional[str] = "#incidents"
    llm_provider: Optional[str] = ""
    metadata: Optional[Dict[str, Any]] = {}
    created_at: Optional[str] = None
    active: bool = True
