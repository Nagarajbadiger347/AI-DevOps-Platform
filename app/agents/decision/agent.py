"""DecisionAgent — applies risk scoring and determines if human approval is needed."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.core.config import settings

_RISK_SCORES: dict[str, float] = {
    "low":      0.2,
    "medium":   0.5,
    "high":     0.8,
    "critical": 1.0,
    "unknown":  0.9,   # treat unknown as high risk
}

# These action types are safe to execute automatically without human approval
_SAFE_ACTION_TYPES = {"investigate", "slack_notify", "create_jira", "opsgenie_alert"}


class DecisionAgent(BaseAgent):
    """Scores risk, enforces approval thresholds — no LLM calls."""

    def run(self, state: dict) -> dict:
        plan       = state.get("plan", {})
        risk       = plan.get("risk", "unknown").lower()
        confidence = float(plan.get("confidence", 0.0))
        actions    = plan.get("actions", [])

        risk_score = _RISK_SCORES.get(risk, 0.9)
        state["risk_score"] = risk_score

        auto_execute_risks = [r.lower() for r in settings.AUTO_EXECUTE_RISK_LEVELS]

        # Split actions into safe (auto-execute always) and destructive (need approval)
        destructive_actions = [a for a in actions if a.get("type") not in _SAFE_ACTION_TYPES]
        safe_only = len(destructive_actions) == 0

        # Require human approval when destructive actions exist AND:
        #   1. risk level not in auto-execute list, OR
        #   2. confidence is below threshold, OR
        #   3. caller explicitly set auto_remediate=False
        needs_approval = not safe_only and (
            risk not in auto_execute_risks
            or confidence < settings.MIN_CONFIDENCE_THRESHOLD
            or not state.get("auto_remediate", False)
        )

        # Optional cost analysis — imported lazily so the module remains usable
        # even when app.cost is not installed.
        try:
            from app.cost.analyzer import analyze_action_costs
            cost_report = analyze_action_costs(actions)
            state["cost_report"] = cost_report
            # Default to False (require approval) if the attribute is missing —
            # safer than defaulting to True (auto-approve)
            if not getattr(cost_report, "approved", False):
                needs_approval = True  # force approval regardless of risk level
        except ImportError:
            cost_report = None
        except Exception as exc:
            self._warn("cost_analysis_failed", error=str(exc))
            # Only force approval on cost failure when there are destructive actions
            if not safe_only:
                needs_approval = True
            cost_report = None

        # Edge case: if there are no actions, skip approval
        if not actions:
            state["requires_human_approval"] = False
            state["approval_reason"] = "no actions in plan"
        else:
            state["requires_human_approval"] = needs_approval
            if needs_approval:
                reasons = []
                if risk not in auto_execute_risks:
                    reasons.append(f"risk={risk} (auto-execute only for {auto_execute_risks})")
                if confidence < settings.MIN_CONFIDENCE_THRESHOLD:
                    reasons.append(f"confidence={confidence:.2f} < threshold={settings.MIN_CONFIDENCE_THRESHOLD}")
                if not state.get("auto_remediate", False):
                    reasons.append("auto_remediate=false")
                cost_report_obj = state.get("cost_report")
                if cost_report_obj is not None and not getattr(cost_report_obj, "approved", False):
                    reasons.append("Cost impact exceeds threshold")
                state["approval_reason"] = "; ".join(reasons)

        self._log(
            "decision_made",
            incident_id=state.get("incident_id"),
            risk=risk,
            risk_score=risk_score,
            confidence=confidence,
            requires_approval=state["requires_human_approval"],
        )
        return state
