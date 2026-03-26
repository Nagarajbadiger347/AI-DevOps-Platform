"""Kubernetes cluster health checker.

Connects via:
  - In-cluster config  if K8S_IN_CLUSTER=true (running inside a pod)
  - KUBECONFIG file    otherwise (local dev / CI)
"""

import os
from kubernetes import client, config
from kubernetes.client.rest import ApiException


def _load_config() -> bool:
    """Load K8s config. Returns True on success.

    Only connects if explicitly configured — never falls back to ~/.kube/config
    automatically, to avoid silently using a local cluster the user didn't intend.
    """
    if os.getenv("K8S_IN_CLUSTER", "").lower() == "true":
        try:
            config.load_incluster_config()
            return True
        except config.ConfigException:
            return False
    kubeconfig = os.getenv("KUBECONFIG", "").strip()
    if not kubeconfig:
        return False  # Not configured — don't silently use ~/.kube/config
    try:
        config.load_kube_config(config_file=kubeconfig)
        return True
    except config.ConfigException:
        return False


def check_k8s_cluster() -> dict:
    """Overall cluster summary — nodes, pods, deployments."""
    if not _load_config():
        return {"status": "error", "details": "K8s config not found. Set KUBECONFIG or K8S_IN_CLUSTER=true"}

    try:
        v1   = client.CoreV1Api()
        apps = client.AppsV1Api()

        nodes       = v1.list_node()
        pods        = v1.list_pod_for_all_namespaces()
        deployments = apps.list_deployment_for_all_namespaces()

        ready_nodes = sum(
            1 for n in nodes.items
            if any(c.type == "Ready" and c.status == "True" for c in n.status.conditions)
        )
        running_pods = sum(1 for p in pods.items if p.status.phase == "Running")
        failed_pods  = sum(1 for p in pods.items if p.status.phase == "Failed")

        ready_deployments = sum(
            1 for d in deployments.items
            if d.status.ready_replicas == d.status.replicas
        )

        overall = "healthy" if failed_pods == 0 and ready_nodes == len(nodes.items) else "degraded"

        return {
            "status": overall,
            "details": {
                "nodes":       {"total": len(nodes.items),       "ready": ready_nodes},
                "pods":        {"total": len(pods.items),        "running": running_pods, "failed": failed_pods},
                "deployments": {"total": len(deployments.items), "ready": ready_deployments},
            },
        }
    except ApiException as e:
        return {"status": "error", "details": f"API error {e.status}: {e.reason}"}
    except Exception as e:
        return {"status": "error", "details": str(e)}


def check_k8s_nodes() -> dict:
    """Per-node status."""
    if not _load_config():
        return {"status": "error", "details": "K8s config not found"}
    try:
        v1    = client.CoreV1Api()
        nodes = v1.list_node()
        result = []
        for n in nodes.items:
            ready = next(
                (c.status for c in n.status.conditions if c.type == "Ready"), "Unknown"
            )
            result.append({
                "name":   n.metadata.name,
                "ready":  ready == "True",
                "roles":  [
                    k.replace("node-role.kubernetes.io/", "")
                    for k in (n.metadata.labels or {})
                    if k.startswith("node-role.kubernetes.io/")
                ] or ["worker"],
                "version": n.status.node_info.kubelet_version,
            })
        return {"status": "ok", "nodes": result}
    except ApiException as e:
        return {"status": "error", "details": f"API error {e.status}: {e.reason}"}
    except Exception as e:
        return {"status": "error", "details": str(e)}


def check_k8s_pods(namespace: str = "default") -> dict:
    """Pod status for a given namespace."""
    if not _load_config():
        return {"status": "error", "details": "K8s config not found"}
    try:
        v1   = client.CoreV1Api()
        pods = v1.list_namespaced_pod(namespace=namespace)
        result = [
            {
                "name":      p.metadata.name,
                "namespace": p.metadata.namespace,
                "phase":     p.status.phase,
                "ready":     all(
                    c.ready for c in (p.status.container_statuses or [])
                ),
                "restarts":  sum(
                    c.restart_count for c in (p.status.container_statuses or [])
                ),
            }
            for p in pods.items
        ]
        return {"status": "ok", "namespace": namespace, "pods": result}
    except ApiException as e:
        return {"status": "error", "details": f"API error {e.status}: {e.reason}"}
    except Exception as e:
        return {"status": "error", "details": str(e)}


def check_k8s_deployments(namespace: str = "default") -> dict:
    """Deployment rollout status for a given namespace."""
    if not _load_config():
        return {"status": "error", "details": "K8s config not found"}
    try:
        apps        = client.AppsV1Api()
        deployments = apps.list_namespaced_deployment(namespace=namespace)
        result = [
            {
                "name":             d.metadata.name,
                "namespace":        d.metadata.namespace,
                "desired_replicas": d.spec.replicas,
                "ready_replicas":   d.status.ready_replicas or 0,
                "available":        (d.status.ready_replicas or 0) == d.spec.replicas,
            }
            for d in deployments.items
        ]
        return {"status": "ok", "namespace": namespace, "deployments": result}
    except ApiException as e:
        return {"status": "error", "details": f"API error {e.status}: {e.reason}"}
    except Exception as e:
        return {"status": "error", "details": str(e)}
