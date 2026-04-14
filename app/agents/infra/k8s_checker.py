"""Kubernetes cluster health checker.

Connects via:
  - In-cluster config  if K8S_IN_CLUSTER=true  (running inside a pod)
  - KUBECONFIG file    otherwise                (local dev / CI)

Never silently falls back to ~/.kube/config to avoid accidentally operating
on the wrong cluster.

Return schema (all functions):
    {
        "status":  "healthy" | "degraded" | "error" | "unavailable",
        "success": bool,
        "details": dict | str,   # dict on success, error string on failure
        ...resource-specific fields...
    }
"""
from __future__ import annotations

import os
from kubernetes import client, config
from kubernetes.client.rest import ApiException


def _load_config() -> bool:
    """Load K8s config.  Returns True on success.

    Only connects if explicitly configured — never falls back to
    ~/.kube/config automatically to avoid silently using an unintended cluster.
    """
    if os.getenv("K8S_IN_CLUSTER", "").lower() == "true":
        try:
            config.load_incluster_config()
            return True
        except config.ConfigException:
            return False

    kubeconfig = os.getenv("KUBECONFIG", "").strip()
    if not kubeconfig:
        return False
    try:
        config.load_kube_config(config_file=kubeconfig)
        return True
    except config.ConfigException:
        return False


def _not_configured() -> dict:
    return {
        "status":  "unavailable",
        "success": False,
        "details": "K8s not configured — set KUBECONFIG or K8S_IN_CLUSTER=true",
    }


def check_k8s_cluster() -> dict:
    """Overall cluster summary — nodes, pods, deployments.

    Returns:
        status  "healthy"     all nodes ready, no failed pods
                "degraded"    some nodes not ready or failed pods present
                "unavailable" K8s not configured
                "error"       K8s API returned an error
    """
    if not _load_config():
        return _not_configured()

    try:
        v1   = client.CoreV1Api()
        apps = client.AppsV1Api()

        nodes       = v1.list_node()
        pods        = v1.list_pod_for_all_namespaces()
        deployments = apps.list_deployment_for_all_namespaces()

        total_nodes   = len(nodes.items)
        ready_nodes   = sum(
            1 for n in nodes.items
            if any(c.type == "Ready" and c.status == "True" for c in n.status.conditions)
        )
        total_pods    = len(pods.items)
        running_pods  = sum(1 for p in pods.items if p.status.phase == "Running")
        failed_pods   = sum(1 for p in pods.items if p.status.phase == "Failed")
        pending_pods  = sum(1 for p in pods.items if p.status.phase == "Pending")

        total_deps   = len(deployments.items)
        ready_deps   = sum(
            1 for d in deployments.items
            if (d.status.ready_replicas or 0) == (d.spec.replicas or 0)
        )

        healthy = (failed_pods == 0 and ready_nodes == total_nodes)
        overall = "healthy" if healthy else "degraded"

        return {
            "status":  overall,
            "success": True,
            "details": {
                "nodes": {
                    "total": total_nodes,
                    "ready": ready_nodes,
                    "not_ready": total_nodes - ready_nodes,
                },
                "pods": {
                    "total":   total_pods,
                    "running": running_pods,
                    "failed":  failed_pods,
                    "pending": pending_pods,
                },
                "deployments": {
                    "total": total_deps,
                    "ready": ready_deps,
                    "degraded": total_deps - ready_deps,
                },
            },
        }

    except ApiException as exc:
        return {"status": "error", "success": False, "details": f"K8s API error {exc.status}: {exc.reason}"}
    except Exception as exc:
        return {"status": "error", "success": False, "details": str(exc)}


def check_k8s_nodes() -> dict:
    """Per-node status and metadata.

    Returns:
        status  "healthy"     all nodes ready
                "degraded"    one or more nodes not ready
                "unavailable" K8s not configured
                "error"       K8s API error
        nodes   list of node dicts
    """
    if not _load_config():
        return _not_configured()

    try:
        v1    = client.CoreV1Api()
        nodes = v1.list_node()

        result = []
        for n in nodes.items:
            ready = next(
                (c.status for c in n.status.conditions if c.type == "Ready"), "Unknown"
            )
            result.append({
                "name":    n.metadata.name,
                "ready":   ready == "True",
                "roles":   [
                    k.replace("node-role.kubernetes.io/", "")
                    for k in (n.metadata.labels or {})
                    if k.startswith("node-role.kubernetes.io/")
                ] or ["worker"],
                "version": n.status.node_info.kubelet_version,
                "os":      n.status.node_info.os_image,
            })

        not_ready = sum(1 for n in result if not n["ready"])
        overall   = "healthy" if not_ready == 0 else "degraded"

        return {
            "status":    overall,
            "success":   True,
            "total":     len(result),
            "not_ready": not_ready,
            "nodes":     result,
        }

    except ApiException as exc:
        return {"status": "error", "success": False, "details": f"K8s API error {exc.status}: {exc.reason}"}
    except Exception as exc:
        return {"status": "error", "success": False, "details": str(exc)}


def check_k8s_pods(namespace: str = "default") -> dict:
    """Pod status for a given namespace, or all namespaces when namespace="all".

    Args:
        namespace: Kubernetes namespace to inspect. Pass "all" for cluster-wide.

    Returns:
        status  "healthy"     all pods running, no restarts > 10
                "degraded"    failed/pending pods or high restart counts
                "unavailable" K8s not configured
                "error"       K8s API error
        pods    list of pod dicts
    """
    if not _load_config():
        return _not_configured()

    try:
        v1 = client.CoreV1Api()
        if namespace == "all":
            pods = v1.list_pod_for_all_namespaces()
        else:
            pods = v1.list_namespaced_pod(namespace=namespace)

        result = []
        for p in pods.items:
            container_statuses = p.status.container_statuses or []
            restarts = sum(c.restart_count for c in container_statuses)
            # Capture waiting reason (e.g. CrashLoopBackOff, OOMKilled)
            waiting_reason = ""
            for cs in container_statuses:
                if cs.state and cs.state.waiting and cs.state.waiting.reason:
                    waiting_reason = cs.state.waiting.reason
                    break
            result.append({
                "name":           p.metadata.name,
                "namespace":      p.metadata.namespace,
                "phase":          p.status.phase,
                "ready":          all(c.ready for c in container_statuses),
                "restarts":       restarts,
                "node":           p.spec.node_name,
                "waiting_reason": waiting_reason,
            })

        failed  = sum(1 for p in result if p["phase"] == "Failed")
        pending = sum(1 for p in result if p["phase"] == "Pending")
        high_restarts = sum(1 for p in result if p["restarts"] > 10)
        overall = "healthy" if (failed == 0 and high_restarts == 0) else "degraded"

        return {
            "status":        overall,
            "success":       True,
            "namespace":     namespace,
            "total":         len(result),
            "running":       sum(1 for p in result if p["phase"] == "Running"),
            "failed":        failed,
            "pending":       pending,
            "high_restarts": high_restarts,
            "pods":          result,
        }

    except ApiException as exc:
        return {"status": "error", "success": False, "details": f"K8s API error {exc.status}: {exc.reason}"}
    except Exception as exc:
        return {"status": "error", "success": False, "details": str(exc)}


def check_k8s_deployments(namespace: str = "default") -> dict:
    """Deployment rollout status for a given namespace.

    Args:
        namespace: Kubernetes namespace to inspect (default: "default").

    Returns:
        status       "healthy"     all deployments fully rolled out
                     "degraded"    one or more deployments not fully ready
                     "unavailable" K8s not configured
                     "error"       K8s API error
        deployments  list of deployment dicts
    """
    if not _load_config():
        return _not_configured()

    try:
        apps        = client.AppsV1Api()
        deployments = apps.list_namespaced_deployment(namespace=namespace)

        result = []
        for d in deployments.items:
            desired   = d.spec.replicas or 0
            ready     = d.status.ready_replicas or 0
            available = d.status.available_replicas or 0
            result.append({
                "name":               d.metadata.name,
                "namespace":          d.metadata.namespace,
                "desired_replicas":   desired,
                "ready_replicas":     ready,
                "available_replicas": available,
                "available":          ready == desired,
                "image":              (
                    d.spec.template.spec.containers[0].image
                    if d.spec.template.spec.containers else ""
                ),
            })

        degraded = sum(1 for d in result if not d["available"])
        overall  = "healthy" if degraded == 0 else "degraded"

        return {
            "status":      overall,
            "success":     True,
            "namespace":   namespace,
            "total":       len(result),
            "ready":       len(result) - degraded,
            "degraded":    degraded,
            "deployments": result,
        }

    except ApiException as exc:
        return {"status": "error", "success": False, "details": f"K8s API error {exc.status}: {exc.reason}"}
    except Exception as exc:
        return {"status": "error", "success": False, "details": str(exc)}


def check_k8s_statefulsets(namespace: str = "default") -> dict:
    """StatefulSet rollout status. Pass namespace="all" for cluster-wide."""
    if not _load_config():
        return _not_configured()
    try:
        apps = client.AppsV1Api()
        sts_list = (apps.list_stateful_set_for_all_namespaces()
                    if namespace == "all"
                    else apps.list_namespaced_stateful_set(namespace=namespace))
        result = []
        for s in sts_list.items:
            desired = s.spec.replicas or 0
            ready   = s.status.ready_replicas or 0
            result.append({
                "name":             s.metadata.name,
                "namespace":        s.metadata.namespace,
                "desired_replicas": desired,
                "ready_replicas":   ready,
                "available":        ready == desired,
            })
        degraded = sum(1 for s in result if not s["available"])
        return {
            "status":       "healthy" if degraded == 0 else "degraded",
            "success":      True,
            "namespace":    namespace,
            "total":        len(result),
            "ready":        len(result) - degraded,
            "degraded":     degraded,
            "statefulsets": result,
        }
    except ApiException as exc:
        return {"status": "error", "success": False, "details": f"K8s API error {exc.status}: {exc.reason}"}
    except Exception as exc:
        return {"status": "error", "success": False, "details": str(exc)}


def check_k8s_daemonsets(namespace: str = "default") -> dict:
    """DaemonSet health — desired == ready on all nodes. Pass namespace="all" for cluster-wide."""
    if not _load_config():
        return _not_configured()
    try:
        apps = client.AppsV1Api()
        ds_list = (apps.list_daemon_set_for_all_namespaces()
                   if namespace == "all"
                   else apps.list_namespaced_daemon_set(namespace=namespace))
        result = []
        for d in ds_list.items:
            desired     = d.status.desired_number_scheduled or 0
            ready       = d.status.number_ready or 0
            unavailable = d.status.number_unavailable or 0
            result.append({
                "name":        d.metadata.name,
                "namespace":   d.metadata.namespace,
                "desired":     desired,
                "ready":       ready,
                "unavailable": unavailable,
                "healthy":     ready == desired and unavailable == 0,
            })
        degraded = sum(1 for d in result if not d["healthy"])
        return {
            "status":     "healthy" if degraded == 0 else "degraded",
            "success":    True,
            "namespace":  namespace,
            "total":      len(result),
            "healthy":    len(result) - degraded,
            "degraded":   degraded,
            "daemonsets": result,
        }
    except ApiException as exc:
        return {"status": "error", "success": False, "details": f"K8s API error {exc.status}: {exc.reason}"}
    except Exception as exc:
        return {"status": "error", "success": False, "details": str(exc)}
