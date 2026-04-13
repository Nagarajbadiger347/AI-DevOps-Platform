"""
Kubernetes routes.
Paths: /check/k8s, /k8s/*
"""
from typing import Optional
from pydantic import BaseModel

from fastapi import APIRouter, Depends, Header, HTTPException

from app.routes.deps import require_developer, require_viewer, AuthContext, _rbac_guard
from app.plugins.k8s_checker import check_k8s_cluster, check_k8s_nodes, check_k8s_pods, check_k8s_deployments
from app.integrations.k8s_ops import (
    restart_deployment, scale_deployment, get_pod_logs,
    list_pods, list_deployments, list_namespaces, get_cluster_events,
    get_unhealthy_pods, delete_pod, cordon_node, uncordon_node, get_resource_usage,
)

router = APIRouter(tags=["kubernetes"])


class K8sRestartRequest(BaseModel):
    namespace: str
    deployment: str


class K8sScaleRequest(BaseModel):
    namespace: str
    deployment: str
    replicas: int


@router.get("/check/k8s")
def k8s_check():
    return {"k8s_check": check_k8s_cluster()}


@router.post("/k8s/restart")
def k8s_restart(req: K8sRestartRequest, x_user: Optional[str] = Header(default=None)):
    _rbac_guard(x_user, "deploy")
    result = restart_deployment(req.namespace, req.deployment)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"result": result}


@router.post("/k8s/scale")
def k8s_scale(req: K8sScaleRequest, x_user: Optional[str] = Header(default=None)):
    _rbac_guard(x_user, "deploy")
    if req.replicas < 0:
        raise HTTPException(status_code=400, detail="replicas must be >= 0")
    result = scale_deployment(req.namespace, req.deployment, req.replicas)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"result": result}


@router.get("/k8s/logs")
def k8s_logs(namespace: str, pod: str, container: str = "", tail_lines: int = 100):
    if tail_lines < 1 or tail_lines > 5000:
        raise HTTPException(status_code=400, detail="tail_lines must be between 1 and 5000")
    result = get_pod_logs(namespace, pod, container, tail_lines)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"result": result}


@router.get("/k8s/pods")
def k8s_pods(namespace: str = "default"):
    result = list_pods(namespace)
    return {"pods": result}


@router.get("/k8s/deployments")
def k8s_deployments(namespace: str = "default"):
    result = list_deployments(namespace)
    return {"deployments": result}


@router.get("/k8s/nodes")
def k8s_nodes():
    result = check_k8s_nodes()
    return {"nodes": result}


@router.post("/k8s/diagnose")
def k8s_diagnose(namespace: str = "default"):
    """Diagnose unhealthy pods and cluster events in a namespace."""
    unhealthy = get_unhealthy_pods(namespace)
    events = get_cluster_events(namespace)
    cluster = check_k8s_cluster()
    return {
        "cluster":       cluster,
        "unhealthy_pods": unhealthy,
        "events":        events,
        "namespace":     namespace,
    }
