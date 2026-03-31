"""PolicyEngine — evaluates every planned action against rules.json before execution.

Guards:
  1. Globally blocked action types
  2. Role-based permission check
  3. Parameter-level guardrails (replica limits, restricted namespaces, etc.)
  4. Max actions per run
"""
from __future__ import annotations

import json
from pathlib import Path
from app.core.logging import get_logger

logger = get_logger(__name__)

_RULES_PATH = Path(__file__).parent / "rules.json"


class PolicyEngine:
    def __init__(self, rules_path: Path = _RULES_PATH) -> None:
        self._rules: dict = json.loads(rules_path.read_text())

    def evaluate(
        self,
        action: dict,
        user: str = "system",
        role: str = "viewer",
    ) -> tuple[bool, str]:
        """Return (allowed, reason).

        allowed=True  → action may proceed
        allowed=False → action must be blocked; reason explains why
        """
        action_type = action.get("type", "")

        # 1. Globally blocked
        if action_type in self._rules.get("blocked_actions", []):
            return False, f"action '{action_type}' is globally blocked by policy"

        # 2. Role permission
        required_perm = self._rules.get("action_permissions", {}).get(action_type)
        if required_perm:
            allowed_roles = self._rules.get("role_permissions", {}).get(required_perm, [])
            if role not in allowed_roles:
                return False, (
                    f"role '{role}' lacks '{required_perm}' permission "
                    f"required for '{action_type}'"
                )

        # 3. Parameter guardrails
        guardrails = self._rules.get("guardrails", {})

        if action_type == "k8s_scale":
            replicas = action.get("replicas", 0)
            max_r = guardrails.get("max_replicas", 20)
            min_r = guardrails.get("min_replicas", 1)
            if replicas > max_r:
                return False, f"replicas {replicas} exceeds max_replicas={max_r}"
            if replicas < min_r:
                return False, f"replicas {replicas} below min_replicas={min_r}"

        if action_type in ("k8s_restart", "k8s_scale"):
            ns = action.get("namespace", "")
            restricted = guardrails.get("restricted_namespaces", [])
            if ns in restricted:
                return False, f"namespace '{ns}' is restricted by policy"

        return True, "allowed"

    def evaluate_batch(
        self,
        actions: list[dict],
        user: str = "system",
        role: str = "viewer",
    ) -> list[tuple[dict, bool, str]]:
        """Evaluate a list of actions; also enforces max_actions_per_run."""
        max_actions = self._rules.get("guardrails", {}).get("max_actions_per_run", 10)
        results: list[tuple[dict, bool, str]] = []

        for i, action in enumerate(actions):
            if i >= max_actions:
                results.append((action, False,
                                 f"max_actions_per_run={max_actions} exceeded"))
                continue
            allowed, reason = self.evaluate(action, user, role)
            results.append((action, allowed, reason))

        return results
