from __future__ import annotations

import textwrap
from typing import Any

from .openai_client import OpenAICompatClient
from .schema import BlockingUnknown, GraphState, PlannerResult
from .tool_selector import OpenAIToolSelector


class OpenAIPlanner:
    """Plan the missing repair fact, then select one executable tool call."""

    def __init__(self, client: OpenAICompatClient, max_queries: int = 1) -> None:
        self.client = client
        self.max_queries = max(1, min(1, int(max_queries)))

    def run(self, state: GraphState) -> PlannerResult:
        payload = self.client.generate_json(
            self._system_prompt(),
            self._user_prompt(state),
            temperature=0.0,
            request_label=f"planner:task_{state.get('task_id', '?')}",
            max_tokens=700,
        )
        result = self._coerce_plan(payload)
        if result.context_needed:
            try:
                query, reason = OpenAIToolSelector(self.client).select(state, result)
            except Exception as exc:
                query, reason = None, f"tool_selector_exception:{type(exc).__name__}"
            if query is None:
                result.context_needed = False
                result.no_context_reason = reason or "No queryable code evidence need."
                result.raw_queries_rejected = [{"reason": result.no_context_reason}]
            else:
                result.need_type = query.need_type
                result.blocking_unknowns = [
                    BlockingUnknown(
                        unknown=result.evidence_need,
                        why_blocking=result.patch_decision,
                        queries=[query],
                    )
                ]
        return result

    def _system_prompt(self) -> str:
        return (
            "You plan SATD repair evidence. Decide the smallest patch decision and "
            "whether one external code fact is needed. Do not choose tools or targets. Return JSON only."
        )

    def _user_prompt(self, state: GraphState) -> str:
        return textwrap.dedent(
            f"""
            SATD comment:
            {state["satd_comment"]}

            Code:
            {state["original_code"]}

            Return exactly:
            {{
              "context_needed": true or false,
              "repair_intent": "short repair intent",
              "patch_decision": "one concrete code decision",
              "evidence_need": "one missing code fact, empty if local code is enough",
              "no_context_reason": "required when context_needed=false"
            }}
            """
        ).strip()

    def _coerce_plan(self, payload: dict[str, Any]) -> PlannerResult:
        repair_intent = self._one_line(payload.get("repair_intent") or payload.get("satd_intent"))
        patch_decision = self._one_line(payload.get("patch_decision") or payload.get("local_repair_plan"))
        evidence_need = self._one_line(payload.get("evidence_need"))
        context_needed = bool(payload.get("context_needed")) and bool(patch_decision and evidence_need)
        return PlannerResult(
            context_needed=context_needed,
            satd_intent=repair_intent,
            local_repair_plan=patch_decision,
            repair_intent=repair_intent,
            patch_decision=patch_decision,
            evidence_need=evidence_need,
            no_context_reason=self._one_line(payload.get("no_context_reason"))
            or ("" if context_needed else "Local code is sufficient for the repair decision."),
        )

    def _one_line(self, value: Any) -> str:
        return " ".join(str(value or "").split())[:260]
