"""Kubernetes operations — restart, scale, fetch logs."""

import os
import time
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


def restart_deployment(namespace: str, deployment: str) -> dict:
    """Trigger a rolling restart by patching the pod template annotation."""
    if not _load_config():
        return {"success": False, "error": "K8s config not found"}
    try:
        apps = client.AppsV1Api()
        patch = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        }
                    }
                }
            }
        }
        apps.patch_namespaced_deployment(name=deployment, namespace=namespace, body=patch)
        return {"success": True, "message": f"Rolling restart triggered for {deployment} in {namespace}"}
    except ApiException as e:
        return {"success": False, "error": f"API error {e.status}: {e.reason}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def scale_deployment(namespace: str, deployment: str, replicas: int) -> dict:
    """Scale a deployment to the given replica count."""
    if not _load_config():
        return {"success": False, "error": "K8s config not found"}
    if replicas < 0:
        return {"success": False, "error": "replicas must be >= 0"}
    try:
        apps = client.AppsV1Api()
        patch = {"spec": {"replicas": replicas}}
        apps.patch_namespaced_deployment_scale(name=deployment, namespace=namespace, body=patch)
        return {
            "success": True,
            "message": f"Deployment {deployment} in {namespace} scaled to {replicas} replica(s)",
        }
    except ApiException as e:
        return {"success": False, "error": f"API error {e.status}: {e.reason}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_pod_logs(namespace: str, pod: str, container: str = "", tail_lines: int = 100) -> dict:
    """Fetch the last N lines of logs from a pod/container."""
    if not _load_config():
        return {"success": False, "error": "K8s config not found"}
    try:
        v1 = client.CoreV1Api()
        kwargs = dict(namespace=namespace, name=pod, tail_lines=tail_lines, timestamps=True)
        if container:
            kwargs["container"] = container
        logs = v1.read_namespaced_pod_log(**kwargs)
        return {"success": True, "pod": pod, "namespace": namespace, "logs": logs}
    except ApiException as e:
        return {"success": False, "error": f"API error {e.status}: {e.reason}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Observability (read-only) ─────────────────────────────────

def list_namespaces() -> dict:
    """List all namespaces in the cluster."""
    if not _load_config():
        return {"success": False, "error": "K8s config not found"}
    try:
        v1 = client.CoreV1Api()
        ns_list = v1.list_namespace()
        namespaces = [
            {"name": ns.metadata.name, "status": ns.status.phase}
            for ns in ns_list.items
        ]
        return {"success": True, "namespaces": namespaces, "count": len(namespaces)}
    except ApiException as e:
        return {"success": False, "error": f"API error {e.status}: {e.reason}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_pods(namespace: str = "") -> dict:
    """List pods across all namespaces (or a specific one) with status and restarts."""
    if not _load_config():
        return {"success": False, "error": "K8s config not found"}
    try:
        v1 = client.CoreV1Api()
        if namespace:
            pod_list = v1.list_namespaced_pod(namespace)
        else:
            pod_list = v1.list_pod_for_all_namespaces()
        pods = []
        for p in pod_list.items:
            restarts = sum(
                (cs.restart_count or 0)
                for cs in (p.status.container_statuses or [])
            )
            pods.append({
                "name":       p.metadata.name,
                "namespace":  p.metadata.namespace,
                "phase":      p.status.phase or "Unknown",
                "ready":      all(
                    (cs.ready or False) for cs in (p.status.container_statuses or [])
                ),
                "restarts":   restarts,
                "node":       p.spec.node_name or "",
                "age_sec":    int((time.time() - p.metadata.creation_timestamp.timestamp())) if p.metadata.creation_timestamp else 0,
            })
        return {"success": True, "pods": pods, "count": len(pods)}
    except ApiException as e:
        return {"success": False, "error": f"API error {e.status}: {e.reason}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_deployments(namespace: str = "") -> dict:
    """List deployments with ready/desired replica counts."""
    if not _load_config():
        return {"success": False, "error": "K8s config not found"}
    try:
        apps = client.AppsV1Api()
        if namespace:
            dep_list = apps.list_namespaced_deployment(namespace)
        else:
            dep_list = apps.list_deployment_for_all_namespaces()
        deployments = [
            {
                "name":       d.metadata.name,
                "namespace":  d.metadata.namespace,
                "desired":    d.spec.replicas or 0,
                "ready":      d.status.ready_replicas or 0,
                "available":  d.status.available_replicas or 0,
                "healthy":    (d.status.ready_replicas or 0) == (d.spec.replicas or 0),
            }
            for d in dep_list.items
        ]
        return {"success": True, "deployments": deployments, "count": len(deployments)}
    except ApiException as e:
        return {"success": False, "error": f"API error {e.status}: {e.reason}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_cluster_events(namespace: str = "", limit: int = 50) -> dict:
    """Fetch Warning-level events from the cluster (OOMKilled, BackOff, Failed, etc.)."""
    if not _load_config():
        return {"success": False, "error": "K8s config not found"}
    try:
        v1 = client.CoreV1Api()
        if namespace:
            evt_list = v1.list_namespaced_event(namespace, field_selector="type=Warning")
        else:
            evt_list = v1.list_event_for_all_namespaces(field_selector="type=Warning")
        events = []
        for e in evt_list.items[:limit]:
            events.append({
                "namespace":  e.metadata.namespace,
                "name":       e.involved_object.name,
                "kind":       e.involved_object.kind,
                "reason":     e.reason or "",
                "message":    e.message or "",
                "count":      e.count or 1,
                "first_time": e.first_timestamp.isoformat() if e.first_timestamp else "",
                "last_time":  e.last_timestamp.isoformat() if e.last_timestamp else "",
            })
        return {"success": True, "events": events, "count": len(events)}
    except ApiException as e:
        return {"success": False, "error": f"API error {e.status}: {e.reason}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_unhealthy_pods(namespace: str = "") -> dict:
    """Return pods that are not Running/Succeeded or have high restart counts."""
    result = list_pods(namespace)
    if not result["success"]:
        return result
    unhealthy = [
        p for p in result["pods"]
        if p["phase"] not in ("Running", "Succeeded") or not p["ready"] or p["restarts"] >= 3
    ]
    return {"success": True, "unhealthy_pods": unhealthy, "count": len(unhealthy)}


def delete_pod(namespace: str, pod: str) -> dict:
    """Delete (force-restart) a pod — K8s will reschedule it automatically."""
    if not _load_config():
        return {"success": False, "error": "K8s config not found"}
    try:
        v1 = client.CoreV1Api()
        v1.delete_namespaced_pod(name=pod, namespace=namespace)
        return {"success": True, "pod": pod, "namespace": namespace,
                "message": f"Pod {pod} deleted — will be rescheduled automatically"}
    except ApiException as e:
        return {"success": False, "error": f"API error {e.status}: {e.reason}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def cordon_node(node: str) -> dict:
    """Cordon a node — prevents new pods from being scheduled on it."""
    if not _load_config():
        return {"success": False, "error": "K8s config not found"}
    try:
        v1 = client.CoreV1Api()
        v1.patch_node(name=node, body={"spec": {"unschedulable": True}})
        return {"success": True, "node": node, "message": f"Node {node} cordoned — no new pods will be scheduled"}
    except ApiException as e:
        return {"success": False, "error": f"API error {e.status}: {e.reason}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def uncordon_node(node: str) -> dict:
    """Uncordon a node — allows pods to be scheduled on it again."""
    if not _load_config():
        return {"success": False, "error": "K8s config not found"}
    try:
        v1 = client.CoreV1Api()
        v1.patch_node(name=node, body={"spec": {"unschedulable": False}})
        return {"success": True, "node": node, "message": f"Node {node} uncordoned — accepting pods again"}
    except ApiException as e:
        return {"success": False, "error": f"API error {e.status}: {e.reason}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_resource_usage(namespace: str = "default") -> dict:
    """Get CPU/memory requests vs limits for all pods in a namespace."""
    if not _load_config():
        return {"success": False, "error": "K8s config not found"}
    try:
        v1 = client.CoreV1Api()
        pod_list = v1.list_namespaced_pod(namespace) if namespace else v1.list_pod_for_all_namespaces()
        usage = []
        for p in pod_list.items:
            for c in (p.spec.containers or []):
                req  = c.resources.requests or {} if c.resources else {}
                lim  = c.resources.limits  or {} if c.resources else {}
                usage.append({
                    "pod":        p.metadata.name,
                    "namespace":  p.metadata.namespace,
                    "container":  c.name,
                    "cpu_req":    req.get("cpu", "—"),
                    "cpu_lim":    lim.get("cpu", "—"),
                    "mem_req":    req.get("memory", "—"),
                    "mem_lim":    lim.get("memory", "—"),
                })
        return {"success": True, "resource_usage": usage, "count": len(usage)}
    except ApiException as e:
        return {"success": False, "error": f"API error {e.status}: {e.reason}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
