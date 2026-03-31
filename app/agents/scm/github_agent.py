"""GitHubAgent — collects recent SCM activity into pipeline state.

Read-only: no mutations. PR creation goes through the Executor.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.core.logging import get_logger

logger = get_logger(__name__)


class GitHubAgent(BaseAgent):
    """Wraps app.integrations.github for the multi-agent pipeline."""

    def run(self, state: dict) -> dict:
        """Return a dict of GitHub context (merged into state by the graph node)."""
        hours = state.get("metadata", {}).get("hours", 2)

        try:
            from app.integrations.github import get_recent_commits, get_recent_prs
            commits = get_recent_commits(hours=hours)
            prs     = get_recent_prs(hours=hours * 12)

            if not commits.get("success") and not prs.get("success"):
                return {
                    "_data_available": False,
                    "_reason": commits.get("error", "GitHub unavailable"),
                }
            self._log("github_context_collected",
                      incident_id=state.get("incident_id", ""))
            return {
                "_data_available": True,
                "recent_commits":  commits,
                "recent_prs":      prs,
            }
        except Exception as exc:
            self._warn("github_agent_error", error=str(exc))
            return {"_data_available": False, "_reason": str(exc)}
