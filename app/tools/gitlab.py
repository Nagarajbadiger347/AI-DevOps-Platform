"""
GitLab Tool — wraps app/integrations/gitlab_ops.py for agent use.
"""
from __future__ import annotations
import logging

logger = logging.getLogger("nsops.tools.gitlab")


class GitLabTool:
    """GitLab CI/CD operations tool used by LangGraph agents."""

    def get_pipeline_logs(self, project_id: str = "", pipeline_id: str = "") -> dict:
        try:
            from app.integrations.gitlab_ops import get_pipeline_jobs
            result = get_pipeline_jobs(project_id=project_id, pipeline_id=pipeline_id)
            return {"success": True, "data": result, "error": None}
        except Exception as e:
            logger.error("get_pipeline_logs failed: %s", e)
            return {"success": False, "data": [], "error": str(e)}

    def retry_pipeline(self, project_id: str, pipeline_id: str) -> dict:
        try:
            from app.integrations.gitlab_ops import retry_pipeline
            result = retry_pipeline(project_id=project_id, pipeline_id=pipeline_id)
            return {"success": True, "data": result, "error": None}
        except Exception as e:
            logger.error("retry_pipeline failed: %s", e)
            return {"success": False, "data": None, "error": str(e)}

    def get_failed_pipelines(self, project_id: str = "", limit: int = 5) -> dict:
        try:
            from app.integrations.gitlab_ops import get_recent_pipelines
            result = get_recent_pipelines(project_id=project_id, status="failed", limit=limit)
            return {"success": True, "data": result, "error": None}
        except Exception as e:
            return {"success": False, "data": [], "error": str(e)}

    def get_job_log(self, project_id: str, job_id: str) -> dict:
        try:
            from app.integrations.gitlab_ops import get_job_log
            result = get_job_log(project_id=project_id, job_id=job_id)
            return {"success": True, "data": result, "error": None}
        except Exception as e:
            return {"success": False, "data": "", "error": str(e)}
