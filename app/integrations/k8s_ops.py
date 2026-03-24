"""Kubernetes operations — restart, scale, fetch logs."""

import os
import time
from kubernetes import client, config
from kubernetes.client.rest import ApiException


def _load_config() -> bool:
    if os.getenv("K8S_IN_CLUSTER", "").lower() == "true":
        try:
            config.load_incluster_config()
            return True
        except config.ConfigException:
            return False
    kubeconfig = os.getenv("KUBECONFIG", "")
    try:
        config.load_kube_config(config_file=kubeconfig or None)
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
