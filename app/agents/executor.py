"""
Executor Agent — performs real actions on K8s, AWS, and CI/CD systems.
Always respects dry_run mode and RBAC approval requirements.
"""
from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger("nsops.agent.executor")

# Actions that require explicit approval before execution
APPROVAL_REQUIRED = {
    "restart_deployment", "scale_deployment", "delete_pod",
    "start_ec2", "stop_ec2", "reboot_ec2",
    "scale_ecs_service", "redeploy_ecs_service",
    "reboot_rds", "retry_pipeline",
}


class ExecutorAgent:
    """
    Executes DevOps actions.
    - dry_run=True: logs what would happen, no real changes
    - requires approval: blocks and returns pending_approval state
    - role_check: validates user role before executing
    """

    def execute(
        self,
        action: str,
        params: dict,
        dry_run: bool = True,
        user_role: str = "viewer",
        approved: bool = False,
    ) -> dict:
        """
        Execute an action with safety checks.

        Returns:
            {
              "success": bool,
              "dry_run": bool,
              "pending_approval": bool,
              "action": str,
              "result": any,
              "error": str|None,
              "message": str,
            }
        """
        logger.info("[EXECUTOR] action=%s dry_run=%s role=%s approved=%s",
                    action, dry_run, user_role, approved)

        # RBAC check
        if user_role == "viewer" and action in APPROVAL_REQUIRED:
            return {
                "success": False, "dry_run": dry_run, "pending_approval": False,
                "action": action,
                "error": f"Role '{user_role}' is not authorized for action '{action}'",
                "message": f"❌ Unauthorized: '{action}' requires operator or admin role",
                "result": None,
            }

        # Approval check
        if action in APPROVAL_REQUIRED and not approved and not dry_run:
            return {
                "success": False, "dry_run": dry_run, "pending_approval": True,
                "action": action, "params": params,
                "message": f"⏳ Action `{action}` requires approval. Use approved=true to proceed.",
                "result": None, "error": None,
            }

        if dry_run:
            return {
                "success": True, "dry_run": True, "pending_approval": False,
                "action": action,
                "message": f"[DRY-RUN] Would execute: {action} with params {params}",
                "result": {"simulated": True, "params": params}, "error": None,
            }

        # Real execution
        handler = self._get_handler(action)
        if not handler:
            return {
                "success": False, "dry_run": False, "pending_approval": False,
                "action": action, "error": f"Unknown action: {action}",
                "message": f"❌ Unknown action: {action}", "result": None,
            }

        try:
            result = handler(**params)
            success = result.get("success", True) if isinstance(result, dict) else True
            logger.info("[EXECUTOR] action=%s success=%s", action, success)
            return {
                "success": success, "dry_run": False, "pending_approval": False,
                "action": action, "result": result,
                "message": f"{'✅' if success else '❌'} {action} completed",
                "error": result.get("error") if isinstance(result, dict) else None,
            }
        except Exception as e:
            logger.error("[EXECUTOR] action=%s error=%s", action, e)
            return {
                "success": False, "dry_run": False, "pending_approval": False,
                "action": action, "error": str(e),
                "message": f"❌ {action} failed: {e}", "result": None,
            }

    # ── Action handlers ───────────────────────────────────────────────────────

    def _get_handler(self, action: str):
        handlers = {
            # K8s
            "restart_pod":          self._restart_pod,
            "restart_deployment":   self._restart_deployment,
            "scale_deployment":     self._scale_deployment,
            "delete_pod":           self._delete_pod,
            # AWS
            "start_ec2":            self._start_ec2,
            "stop_ec2":             self._stop_ec2,
            "reboot_ec2":           self._reboot_ec2,
            "scale_ecs_service":    self._scale_ecs,
            "redeploy_ecs_service": self._redeploy_ecs,
            "reboot_rds":           self._reboot_rds,
            # GitLab
            "retry_pipeline":       self._retry_pipeline,
        }
        return handlers.get(action)

    # K8s handlers
    def _restart_pod(self, namespace: str, pod: str, **_) -> dict:
        from app.tools.kubernetes import KubernetesTool
        return KubernetesTool().restart_pod(namespace=namespace, pod=pod)

    def _restart_deployment(self, namespace: str, deployment: str, **_) -> dict:
        from app.tools.kubernetes import KubernetesTool
        return KubernetesTool().restart_deployment(namespace=namespace, deployment=deployment)

    def _scale_deployment(self, namespace: str, deployment: str, replicas: int, **_) -> dict:
        from app.tools.kubernetes import KubernetesTool
        return KubernetesTool().scale_deployment(namespace=namespace, deployment=deployment, replicas=replicas)

    def _delete_pod(self, namespace: str, pod: str, **_) -> dict:
        from app.tools.kubernetes import KubernetesTool
        return KubernetesTool().restart_pod(namespace=namespace, pod=pod)

    # AWS handlers
    def _start_ec2(self, instance_id: str, **_) -> dict:
        from app.tools.aws import AWSTool
        return AWSTool().start_ec2(instance_id=instance_id)

    def _stop_ec2(self, instance_id: str, **_) -> dict:
        from app.tools.aws import AWSTool
        return AWSTool().stop_ec2(instance_id=instance_id)

    def _reboot_ec2(self, instance_id: str, **_) -> dict:
        from app.integrations.aws_ops import reboot_ec2_instance
        return reboot_ec2_instance(instance_id=instance_id)

    def _scale_ecs(self, cluster: str, service: str, desired_count: int, **_) -> dict:
        from app.integrations.aws_ops import scale_ecs_service
        return scale_ecs_service(cluster=cluster, service=service, desired_count=desired_count)

    def _redeploy_ecs(self, cluster: str, service: str, **_) -> dict:
        from app.integrations.aws_ops import force_new_ecs_deployment
        return force_new_ecs_deployment(cluster=cluster, service=service)

    def _reboot_rds(self, db_instance_id: str, **_) -> dict:
        from app.integrations.aws_ops import reboot_rds_instance
        return reboot_rds_instance(db_instance_id=db_instance_id)

    # GitLab handler
    def _retry_pipeline(self, project_id: str, pipeline_id: str, **_) -> dict:
        from app.tools.gitlab import GitLabTool
        return GitLabTool().retry_pipeline(project_id=project_id, pipeline_id=pipeline_id)
