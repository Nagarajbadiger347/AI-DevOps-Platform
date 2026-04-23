"""
Planner Agent — decomposes a high-level DevOps task into executable steps.
Used standalone or as the entry node in a LangGraph workflow.
"""
from __future__ import annotations
import json
import logging

logger = logging.getLogger("nsops.agent.planner")


class PlannerAgent:
    """
    Given a natural-language task description, produces a structured plan:
    - list of steps with agent assignments
    - required tools per step
    - estimated severity
    """

    SYSTEM = """You are a Senior DevOps Architect and Incident Commander.
Your job is to decompose a task into a concrete execution plan for a multi-agent system.

Agents available:
- debugger: analyzes logs, errors, metrics, K8s events
- executor: runs kubectl, AWS CLI, or API actions
- observer: monitors real-time events and alerts
- reporter: formats and sends outputs (Slack, API response)

Safety Rules:
- Never include destructive actions like 'delete', 'stop', 'terminate' without approval.
- Validate all parameters for security (no arbitrary code execution).
- For critical severity, always require approval.
- Provide reasoning for each step.

Respond ONLY with valid JSON:
{
  "task_type": "<k8s_debug|aws_ops|pipeline_debug|general>",
  "severity": "<critical|high|medium|low>",
  "steps": [
    {"step": 1, "agent": "<agent>", "action": "<description>", "tool": "<tool_name>", "params": {}, "reasoning": "<why this step>"}
  ],
  "requires_approval": <true|false>,
  "estimated_duration_s": <int>,
  "overall_reasoning": "<summary of plan>"
}"""

    def plan(self, task: str, context: dict | None = None) -> dict:
        """Decompose a task into a structured plan."""
        from app.chat.intelligence import _llm_call

        ctx_text = ""
        if context:
            ctx_text = f"\n\nContext:\n{json.dumps(context, indent=2, default=str)[:800]}"

        user = f"Task: {task}{ctx_text}"

        logger.info("[PLANNER] planning task: %s", task[:100])
        raw = _llm_call(user, system=self.SYSTEM, max_tokens=600, temperature=0.2)

        try:
            clean = raw.strip().strip("```json").strip("```").strip()
            plan = json.loads(clean)
            logger.info("[PLANNER] plan: type=%s steps=%d severity=%s",
                        plan.get("task_type"), len(plan.get("steps", [])), plan.get("severity"))
            # Validate the plan
            validation = self._validate_plan(plan)
            if not validation["valid"]:
                logger.warning("[PLANNER] plan validation failed: %s", validation["issues"])
                return {
                    "success": False,
                    "plan": plan,
                    "error": f"Validation failed: {validation['issues']}",
                }
            return {"success": True, "plan": plan}
        except Exception as e:
            logger.warning("[PLANNER] JSON parse failed: %s — raw: %s", e, raw[:200])
            return {
                "success": False,
                "plan": {"task_type": "general", "severity": "medium", "steps": [], "raw": raw},
                "error": str(e),
            }

    def _validate_plan(self, plan: dict) -> dict:
        """Validate the generated plan for safety and completeness."""
        issues = []
        if not plan.get("steps"):
            issues.append("No steps provided")
        for step in plan.get("steps", []):
            if "action" not in step:
                issues.append(f"Step {step.get('step', '?')} missing action")
            if step.get("action", "").lower() in ["delete", "stop", "terminate", "destroy"]:
                if not plan.get("requires_approval", False):
                    issues.append(f"Destructive action '{step['action']}' requires approval")
            if "reasoning" not in step:
                issues.append(f"Step {step.get('step', '?')} missing reasoning")
        if plan.get("severity") == "critical" and not plan.get("requires_approval"):
            issues.append("Critical severity requires approval")
        return {"valid": len(issues) == 0, "issues": issues}
