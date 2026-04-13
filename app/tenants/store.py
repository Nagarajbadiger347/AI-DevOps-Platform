"""Tenant persistence — JSON file backed store."""
import json
import datetime
import os
from typing import Dict, Optional, List

from app.tenants.models import Tenant

_TENANTS_FILE = os.path.join(os.path.dirname(__file__), "../../data/tenants.json")
_tenants: Dict[str, Tenant] = {}


def _load():
    global _tenants
    try:
        with open(_TENANTS_FILE) as f:
            data = json.load(f)
        _tenants = {k: Tenant(**v) for k, v in data.items()}
    except FileNotFoundError:
        _tenants = {}
    except Exception:
        _tenants = {}


def _save():
    try:
        os.makedirs(os.path.dirname(_TENANTS_FILE), exist_ok=True)
        with open(_TENANTS_FILE, "w") as f:
            json.dump({k: v.model_dump() for k, v in _tenants.items()}, f, indent=2)
    except Exception:
        pass


_load()


def get_tenant(tenant_id: str) -> Optional[Tenant]:
    return _tenants.get(tenant_id)


def list_tenants() -> List[Tenant]:
    return list(_tenants.values())


def create_tenant(tenant: Tenant) -> Tenant:
    if not tenant.created_at:
        tenant.created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _tenants[tenant.tenant_id] = tenant
    _save()
    return tenant


def update_tenant(tenant_id: str, updates: dict) -> Optional[Tenant]:
    t = _tenants.get(tenant_id)
    if not t:
        return None
    updated = t.model_copy(update=updates)
    _tenants[tenant_id] = updated
    _save()
    return updated


def delete_tenant(tenant_id: str) -> bool:
    if tenant_id not in _tenants:
        return False
    del _tenants[tenant_id]
    _save()
    return True
