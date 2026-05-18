from __future__ import annotations

from dataclasses import replace
from typing import Any

from .openai_client import OpenAICompatClient
from .schema import ContextPolicyDecision, GraphState, RepositoryEvidenceContext


EVIDENCE_TYPES = {
    "symbol_or_api_definition",
    "related_tests_or_expected_behavior",
    "documentation_or_docstring",
    "project_usage_examples",
    "caller_impact",
    "config_or_project_convention",
    "sibling_implementation",
}


class OpenAIContextPolicyAgent:
    """Decide whether repository evidence should enter the fixer prompt."""

    def __init__(
        self,
        client: OpenAICompatClient,
        logger: Any | None = None,
    ) -> None:
        self.client = client
        self.logger = logger

    def decide_retrieval(self, state: GraphState) -> ContextPolicyDecision:
        system_prompt = (
            "You are a context policy agent for SATD repair. Decide whether repository context is likely to improve "
            "strict exact-match repair. Do not repair the code. Do not use any manual repair answer. Return JSON only."
        )
        policy_rules = self._retrieval_policy_rules()
        user_prompt = (
            "Choose the repair context mode before fixing this SATD.\n\n"
            f"### File path:\n{state['file_path']}\n\n"
            f"### SATD comment:\n{state['satd_comment']}\n\n"
            f"### Code:\n```python\n{state['original_code']}\n```\n\n"
            f"{policy_rules}\n\n"
            "Allowed evidence types:\n"
            "- symbol_or_api_definition\n"
            "- related_tests_or_expected_behavior\n"
            "- documentation_or_docstring\n"
            "- project_usage_examples\n"
            "- caller_impact\n"
            "- config_or_project_convention\n"
            "- sibling_implementation\n\n"
            "Return JSON with keys:\n"
            "{\n"
            '  "use_repository_context": true/false,\n'
            '  "needed_evidence_types": ["..."],\n'
            '  "context_risk": "low"|"medium"|"high",\n'
            '  "decision_reason": "one short reason"\n'
            "}\n"
        )
        try:
            payload = self.client.generate_json(
                system_prompt,
                user_prompt,
                temperature=0.0,
                request_label=f"context_policy_retrieval:task_{state['task_id']}",
                max_tokens=500,
            )
            decision = self._coerce_retrieval_decision(payload)
        except Exception as exc:
            decision = ContextPolicyDecision(
                use_repository_context=False,
                context_risk="high",
                decision_reason=f"context_policy_retrieval_exception:{type(exc).__name__}",
                retrieval_decision_reason=f"context_policy_retrieval_exception:{type(exc).__name__}",
            )
        self._log(
            state,
            "context policy retrieval "
            f"use={decision.use_repository_context} risk={decision.context_risk} "
            f"types={','.join(decision.needed_evidence_types) or 'none'}",
        )
        return decision

    def decide_injection(
        self,
        state: GraphState,
        retrieval_decision: ContextPolicyDecision,
        candidates: list[RepositoryEvidenceContext],
        *,
        max_items: int,
    ) -> ContextPolicyDecision:
        if not retrieval_decision.use_repository_context or not candidates:
            return replace(
                retrieval_decision,
                inject_evidence=False,
                selected_evidence_indices=[],
                candidate_evidence_count=len(candidates),
                selected_evidence_count=0,
                injection_decision_reason="no candidate evidence selected for injection",
            )

        summaries = []
        for index, item in enumerate(candidates, start=1):
            content = " ".join(str(item.content or "").split())
            summaries.append(
                f"[{index}] type={item.evidence_type}/{item.evidence_subtype} "
                f"support={item.support_level} score={item.score:.2f} path={item.source_path}:{item.span}\n"
                f"why={item.why_relevant}\n"
                f"excerpt={content[:700]}"
            )
        system_prompt = (
            "You are a context policy agent for SATD repair. Select only repository evidence that is likely to improve "
            "strict exact-match repair. Do not repair the code. Return JSON only."
        )
        injection_rules = self._injection_policy_rules()
        user_prompt = (
            "Decide whether to inject any of these repository evidence snippets into the fixer prompt.\n\n"
            f"### SATD comment:\n{state['satd_comment']}\n\n"
            f"### Code:\n```python\n{state['original_code']}\n```\n\n"
            f"### Retrieval decision:\nuse_repository_context={retrieval_decision.use_repository_context}; "
            f"needed_evidence_types={retrieval_decision.needed_evidence_types}; "
            f"context_risk={retrieval_decision.context_risk}; reason={retrieval_decision.decision_reason}\n\n"
            "### Candidate evidence:\n"
            f"{chr(10).join(summaries)}\n\n"
            f"{injection_rules}\n"
            "Select at most "
            f"{max_items} snippets.\n\n"
            "Return JSON with keys:\n"
            '{ "inject_evidence": true/false, '
            '"selected_evidence_indices": [1, 2], '
            '"injection_decision_reason": "one short reason" }\n'
        )
        try:
            payload = self.client.generate_json(
                system_prompt,
                user_prompt,
                temperature=0.0,
                request_label=f"context_policy_injection:task_{state['task_id']}",
                max_tokens=500,
            )
            decision = self._coerce_injection_decision(
                payload,
                retrieval_decision,
                candidate_count=len(candidates),
                max_items=max_items,
            )
        except Exception as exc:
            decision = replace(
                retrieval_decision,
                inject_evidence=False,
                selected_evidence_indices=[],
                candidate_evidence_count=len(candidates),
                selected_evidence_count=0,
                injection_decision_reason=f"context_policy_injection_exception:{type(exc).__name__}",
            )
        self._log(
            state,
            "context policy injection "
            f"inject={decision.inject_evidence} selected={decision.selected_evidence_indices}",
        )
        return decision

    def _coerce_retrieval_decision(self, payload: dict[str, Any]) -> ContextPolicyDecision:
        needed = self._coerce_evidence_types(payload.get("needed_evidence_types"))
        reason = " ".join(str(payload.get("decision_reason") or payload.get("reason") or "").split())
        risk = str(payload.get("context_risk") or "unknown").strip().lower()
        if risk not in {"low", "medium", "high", "unknown"}:
            risk = "unknown"
        use_context = bool(payload.get("use_repository_context")) and risk != "high"
        return ContextPolicyDecision(
            use_repository_context=use_context,
            needed_evidence_types=needed,
            context_risk=risk,
            inject_evidence=False,
            decision_reason=reason or "context policy retrieval decision",
            retrieval_decision_reason=reason or "context policy retrieval decision",
        )

    def _coerce_injection_decision(
        self,
        payload: dict[str, Any],
        retrieval_decision: ContextPolicyDecision,
        *,
        candidate_count: int,
        max_items: int,
    ) -> ContextPolicyDecision:
        selected: list[int] = []
        for value in payload.get("selected_evidence_indices") or []:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if 1 <= index <= candidate_count and index not in selected:
                selected.append(index)
            if len(selected) >= max_items:
                break
        inject = bool(payload.get("inject_evidence")) and bool(selected)
        reason = " ".join(str(payload.get("injection_decision_reason") or payload.get("reason") or "").split())
        return replace(
            retrieval_decision,
            inject_evidence=inject,
            selected_evidence_indices=selected if inject else [],
            candidate_evidence_count=candidate_count,
            selected_evidence_count=len(selected) if inject else 0,
            injection_decision_reason=reason or "context policy injection decision",
        )

    def _coerce_evidence_types(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            name = str(item or "").strip()
            if name in EVIDENCE_TYPES and name not in result:
                result.append(name)
        return result

    def _retrieval_policy_rules(self) -> str:
        return (
            "Decide whether repository context is actually needed for exact-match repair.\n\n"
            "Prefer snippet-only when the repair is locally determined by the snippet, especially for cleanup SATD:\n"
            "- remove obsolete, duplicate, debug, hack, workaround, or legacy code\n"
            "- drop an old compatibility branch\n"
            "- uncomment or enable code already visible in the snippet\n"
            "- make a tiny local literal, default, or type change\n\n"
            "Use repository context only when the snippet is missing a concrete project fact needed to finish the repair:\n"
            "- which API or symbol should replace the old one\n"
            "- which exception type should be used\n"
            "- which setting, config key, path, or project-specific value is intended\n"
            "- what exact test expectation or behavior is required\n"
            "- how a sibling implementation or caller establishes the intended behavior\n\n"
            "If the SATD explicitly names a replacement symbol, setting, exception, expected output, or project-wide "
            "choice that is not defined in the snippet, use repository context.\n\n"
            "If you cannot name the missing concrete fact, choose snippet-only. "
            "Do not use repository context merely because the SATD is vague or mentions future cleanup."
        )

    def _injection_policy_rules(self) -> str:
        return (
            "Inject evidence only when it directly supplies the missing fact named above. "
            "Keep snippets that provide an exact replacement API, exception type, setting/config value, expected behavior, "
            "caller contract, or sibling implementation. Reject generic definitions, broad examples, duplicate context, "
            "and anything that does not make the edit more deterministic than the original snippet. "
            "If no snippet provides a concrete missing fact, set inject_evidence=false."
        )

    def _log(self, state: GraphState, message: str) -> None:
        if callable(self.logger):
            self.logger(f"{self._task_prefix(state)} {message}")

    def _task_prefix(self, state: GraphState) -> str:
        task_id = state.get("task_id", "?")
        task_index = state.get("task_index") or 0
        task_total = state.get("task_total") or 0
        if task_index and task_total:
            return f"[task {task_id} {task_index}/{task_total}]"
        return f"[task {task_id}]"
