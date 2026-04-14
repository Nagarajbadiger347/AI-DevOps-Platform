"""
AWS Tool — wraps app/integrations/aws_ops.py for agent use.
"""
from __future__ import annotations
import logging

logger = logging.getLogger("nsops.tools.aws")


class AWSTool:
    """AWS operations tool used by LangGraph agents."""

    def list_ec2(self, state: str = "", region: str = "") -> dict:
        try:
            from app.integrations.aws_ops import list_ec2_instances
            result = list_ec2_instances(state=state, region=region)
            return {"success": True, "data": result.get("instances", []), "error": None}
        except Exception as e:
            return {"success": False, "data": [], "error": str(e)}

    def get_ec2_info(self, instance_id: str) -> dict:
        try:
            from app.integrations.aws_ops import get_ec2_instance_info
            result = get_ec2_instance_info(instance_id)
            return {"success": True, "data": result, "error": None}
        except Exception as e:
            return {"success": False, "data": {}, "error": str(e)}

    def get_cloudwatch_logs(self, log_group: str, minutes: int = 30, limit: int = 100) -> dict:
        try:
            from app.integrations.aws_ops import get_recent_logs
            result = get_recent_logs(log_group=log_group, minutes=minutes, limit=limit)
            return {"success": True, "data": result.get("logs", []), "error": None}
        except Exception as e:
            return {"success": False, "data": [], "error": str(e)}

    def search_logs(self, log_group: str, pattern: str, hours: int = 1) -> dict:
        try:
            from app.integrations.aws_ops import search_logs
            result = search_logs(log_group=log_group, pattern=pattern, hours=hours)
            return {"success": True, "data": result.get("events", []), "error": None}
        except Exception as e:
            return {"success": False, "data": [], "error": str(e)}

    def get_alarms(self, state: str = "ALARM") -> dict:
        try:
            from app.integrations.aws_ops import list_cloudwatch_alarms
            result = list_cloudwatch_alarms(state=state)
            return {"success": True, "data": result.get("alarms", []), "error": None}
        except Exception as e:
            return {"success": False, "data": [], "error": str(e)}

    def secrets_manager_get(self, secret_name: str) -> dict:
        """Retrieve a secret from AWS Secrets Manager."""
        try:
            import boto3, os
            region = os.getenv("AWS_REGION", "us-east-1")
            client = boto3.client("secretsmanager", region_name=region)
            resp = client.get_secret_value(SecretId=secret_name)
            return {"success": True, "data": resp.get("SecretString", ""), "error": None}
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}

    def get_cost_summary(self, days: int = 7) -> dict:
        try:
            from app.integrations.aws_ops import get_cost_by_service
            result = get_cost_by_service(days=days)
            return {"success": True, "data": result, "error": None}
        except Exception as e:
            return {"success": False, "data": {}, "error": str(e)}

    def start_ec2(self, instance_id: str) -> dict:
        try:
            from app.integrations.aws_ops import start_ec2_instance
            result = start_ec2_instance(instance_id)
            return {"success": result.get("success", True), "data": result, "error": None}
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}

    def stop_ec2(self, instance_id: str) -> dict:
        try:
            from app.integrations.aws_ops import stop_ec2_instance
            result = stop_ec2_instance(instance_id)
            return {"success": result.get("success", True), "data": result, "error": None}
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}
