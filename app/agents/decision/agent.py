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


class DecisionAgent(BaseAgent):
    """Scores risk, enforces approval thresholds — no LLM calls."""

    def run(self, state: dict) -> dict:
        plan       = state.get("plan", {})
        risk       = plan.get("risk", "unknown").lower()
        confidence = float(plan.get("confidence", 0.0))
        actions    = plan.get("actions", [])

        risk_score = _RISK_SCORES.get(risk, 0.9)
        state["risk_score"] = risk_score

        # Require human approval when:
        #   1. risk level not in auto-execute list, OR
        #   2. confidence is below threshold, OR
        #   3. caller explicitly set auto_remediate=False
        auto_execute_risks = [r.lower() for r in settings.AUTO_EXECUTE_RISK_LEVELS]
        needs_approval = (
            risk not in auto_execute_risks
            or confidence < settings.MIN_CONFIDENCE_THRESHOLD
            or not state.get("auto_remediate", False)
            or not actions  # nothing to execute → no approval needed either
        )

        # Edge case: if there are no actions, skip approval but mark as done
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
