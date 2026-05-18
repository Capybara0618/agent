from __future__ import annotations

from typing import Any

from .openai_client import OpenAICompatClient
from .schema import AnalysisResult, GraphState, MethodInquiryResult


class OpenAIAnalyzer:
    def __init__(self, client: OpenAICompatClient) -> None:
        self.client = client

    def run(self, state: GraphState, method_context_block: str = "[none]") -> AnalysisResult:
        method_inquiry = state.get("method_inquiry")
        required_methods = method_inquiry.required_methods if isinstance(method_inquiry, MethodInquiryResult) else []
        retrieved = list(state.get("retrieved_method_contexts") or [])
        found_count = sum(1 for item in retrieved if bool(getattr(item, "found", False)))
        missing = [str(item) for item in (state.get("missing_method_names") or []) if str(item).strip()]
        context_summary = self._context_summary(required_methods, found_count, missing)
        system_prompt = (
            "You are the analyzer agent in a SATD repair workflow.\n"
            "Do not filter or drop the SATD. Do not predict success. Do not write replacement code.\n"
            "Extract compact intent labels that help a later reviewer judge whether a candidate repair matches the SATD intent.\n"
            "Return JSON only."
        )
        user_prompt = (
            "Analyze this function-level SATD for intent labeling:\n\n"
            f"SATD comment:\n{state['satd_comment']}\n\n"
            f"Original code:\n```python\n{state['original_code']}\n```\n\n"
            f"Identified methods:\n{self._format_list(required_methods)}\n\n"
            f"Retrieved method context:\n{method_context_block or '[none]'}\n\n"
            f"Missing methods:\n{self._format_list(missing)}\n\n"
            "Classify the developer intent, the likely target, and the expected edit shape.\n"
            "Use the labels to describe what a candidate repair should be judged against, not whether this SATD should be attempted.\n\n"
            "Intent labels:\n"
            "- delete_or_remove: remove obsolete, temporary, deprecated, workaround, debug, compatibility, or no-longer-needed code.\n"
            "- replace_or_switch: use another API, value, option, format, constant, call, or implementation path.\n"
            "- add_or_support: add missing behavior, support a case, implement a feature, handle input, or enable an option.\n"
            "- bug_fix_or_check: fix incorrect behavior, add validation/checking, raise/return correctly, or handle an error.\n"
            "- documentation: update comments, docs, type annotations, lint/type-checker debt, or explanatory text.\n"
            "- refactor_or_rewrite: cleanup, simplify, restructure, improve performance, or make a broader implementation change.\n"
            "- unclear: the repair target or operation cannot be inferred.\n\n"
            "Expected edit shape labels:\n"
            "- delete_line: remove one line or a very small local statement.\n"
            "- delete_block: remove a local block, branch, temporary guard, skipped case, or compatibility block.\n"
            "- replace_call_or_value: replace a call, attribute, value, constant, string, option, format, or condition.\n"
            "- add_parameter_or_option: add or pass a parameter, flag, option, field, config, or keyword.\n"
            "- add_check_or_raise: add validation, guard, exception, error handling, assertion, or conditional handling.\n"
            "- adjust_return: change the returned value or return branch.\n"
            "- documentation_only: edit only comments, docs, annotations, or lint/type-checker related text.\n"
            "- broad_rewrite: restructure or rewrite substantial logic.\n"
            "- unclear: the edit shape is not inferable.\n\n"
            "Evidence requirement labels:\n"
            "- local_cleanup_ok: the debt may be repaid by removing or locally replacing obsolete, temporary, deprecated, workaround, disabled, skipped, compatibility, unnecessary, or no-longer-needed code.\n"
            "- visible_behavior_required: success requires visible added behavior, support, validation, handling, or implementation.\n"
            "- local_replacement_ok: success may be a grounded local switch of API, value, option, format, or call.\n"
            "- documentation_only: repayment is mainly documentation, comments, annotations, or lint/type-checker debt.\n"
            "- unclear: the review standard cannot be inferred.\n\n"
            "If the comment describes existing local code as temporary, disabled, skipped, wrong, unnecessary, obsolete, or no longer needed, prefer local_cleanup_ok when deleting or locally replacing that code could itself repay the debt.\n\n"
            "Return exactly:\n"
            "{\n"
            '  "reason": "one sentence describing the SATD repayment intent",\n'
            '  "target_summary": "short description of the code target",\n'
            '  "intent_type": "delete_or_remove" | "replace_or_switch" | "add_or_support" | "bug_fix_or_check" | "documentation" | "refactor_or_rewrite" | "unclear",\n'
            '  "target_clarity": "high" | "partial" | "low",\n'
            '  "expected_edit_shape": "delete_line" | "delete_block" | "replace_call_or_value" | "add_parameter_or_option" | "add_check_or_raise" | "adjust_return" | "documentation_only" | "broad_rewrite" | "unclear",\n'
            '  "evidence_requirement": "local_cleanup_ok" | "visible_behavior_required" | "local_replacement_ok" | "documentation_only" | "unclear",\n'
            '  "risk_note": "one short note about external conditions, missing context, broadness, or empty string"\n'
            "}\n"
        )
        payload = self.client.generate_json(system_prompt, user_prompt, request_label=f"analyze_intent:task_{state['task_id']}")
        return self._coerce_analysis(payload, context_summary)

    def _coerce_analysis(self, payload: dict[str, Any], context_summary: str) -> AnalysisResult:
        return AnalysisResult(
            decision="pass",
            repairable=True,
            reason=self._one_line(payload.get("reason")) or "intent_labeled",
            repair_plan="",
            target_summary=self._one_line(payload.get("target_summary")),
            context_summary=context_summary,
            intent_type=self._normalize_choice(
                payload.get("intent_type"),
                {
                    "delete_or_remove",
                    "replace_or_switch",
                    "add_or_support",
                    "bug_fix_or_check",
                    "documentation",
                    "refactor_or_rewrite",
                    "unclear",
                },
                "unclear",
            ),
            target_clarity=self._normalize_choice(payload.get("target_clarity"), {"high", "partial", "low"}, "partial"),
            expected_edit_shape=self._normalize_choice(
                payload.get("expected_edit_shape"),
                {
                    "delete_line",
                    "delete_block",
                    "replace_call_or_value",
                    "add_parameter_or_option",
                    "add_check_or_raise",
                    "adjust_return",
                    "documentation_only",
                    "broad_rewrite",
                    "unclear",
                },
                "unclear",
            ),
            evidence_requirement=self._normalize_choice(
                payload.get("evidence_requirement"),
                {
                    "local_cleanup_ok",
                    "visible_behavior_required",
                    "local_replacement_ok",
                    "documentation_only",
                    "unclear",
                },
                "unclear",
            ),
            risk_note=self._one_line(payload.get("risk_note")),
        )

    def fallback_analysis(self, state: GraphState, reason: str) -> AnalysisResult:
        return AnalysisResult(
            decision="pass",
            repairable=True,
            reason=reason,
            repair_plan="",
            target_summary="",
            context_summary="[none]",
            intent_type="unclear",
            target_clarity="partial",
            expected_edit_shape="unclear",
            evidence_requirement="unclear",
            risk_note=reason,
        )

    def _context_summary(self, required_methods: list[str], found_count: int, missing: list[str]) -> str:
        parts = [
            f"identified_methods={len(required_methods)}",
            f"retrieved_methods={found_count}",
        ]
        if missing:
            parts.append("missing_methods=" + ",".join(missing[:5]))
        return "; ".join(parts)

    def _format_list(self, items: list[str]) -> str:
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        return ", ".join(cleaned) if cleaned else "[none]"

    def _normalize_choice(self, value: Any, allowed: set[str], default: str) -> str:
        text = str(value or "").strip().lower()
        return text if text in allowed else default

    def _one_line(self, value: Any) -> str:
        return " ".join(str(value or "").split())[:240]
