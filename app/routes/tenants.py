"""
Tenant management routes (super_admin only).
Paths: /admin/tenants/*
"""
from fastapi import APIRouter, Depends, HTTPException
from app.routes.deps import require_super_admin, AuthContext
from app.tenants.models import Tenant

router = APIRouter(prefix="/admin/tenants", tags=["tenants"])


@router.get("")
def list_tenants_endpoint(auth: AuthContext = Depends(require_super_admin)):
    from app.tenants.store import list_tenants
    return {"tenants": [t.model_dump() for t in list_tenants()]}


@router.get("/{tenant_id}")
def get_tenant_endpoint(tenant_id: str, auth: AuthContext = Depends(require_super_admin)):
    from app.tenants.store import get_tenant
    t = get_tenant(tenant_id)
    if not t:
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found")
    return t.model_dump()


@router.post("")
def create_tenant_endpoint(tenant: Tenant, auth: AuthContext = Depends(require_super_admin)):
    from app.tenants.store import get_tenant, create_tenant
    if get_tenant(tenant.tenant_id):
        raise HTTPException(status_code=409, detail=f"Tenant '{tenant.tenant_id}' already exists")
    return create_tenant(tenant).model_dump()


@router.patch("/{tenant_id}")
def update_tenant_endpoint(tenant_id: str, updates: dict, auth: AuthContext = Depends(require_super_admin)):
    from app.tenants.store import update_tenant
    t = update_tenant(tenant_id, updates)
    if not t:
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found")
    return t.model_dump()


@router.delete("/{tenant_id}")
def delete_tenant_endpoint(tenant_id: str, auth: AuthContext = Depends(require_super_admin)):
    from app.tenants.store import delete_tenant
    if not delete_tenant(tenant_id):
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found")
    return {"deleted": True, "tenant_id": tenant_id}
