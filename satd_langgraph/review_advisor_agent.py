from __future__ import annotations

from typing import Any

from .openai_client import OpenAICompatClient
from .schema import GraphState, MethodInquiryResult, RepairAttempt, RepositoryEvidenceContext, RetrievedMethodContext, ReviewResult


class OpenAIReviewAdvisor:
    def __init__(self, client: OpenAICompatClient) -> None:
        self.client = client

    def run(self, state: GraphState, gate_result: ReviewResult) -> ReviewResult:
        repair = state["latest_repair"]
        assert repair is not None

        system_prompt, user_prompt = self._build_prompts(state, repair, gate_result)
        payload = self.client.generate_json(
            system_prompt,
            user_prompt,
            request_label=f"review_advisor:task_{state['task_id']}:round_{repair.round_id}",
        )
        repair_constraints = self._normalize_string_list(
            payload.get("repair_constraints"),
            default=["make_smallest_evidence_grounded_edit"],
        )
        retry_hint = self._one_line(payload.get("retry_hint")) or "Retry with a smaller repair that directly addresses the SATD evidence."
        failure_anchor = self._one_line(payload.get("failure_anchor")) or (gate_result.failure_modes[0] if gate_result.failure_modes else "retry")
        return self._copy_with_advice(
            gate_result,
            repair_constraints=repair_constraints,
            retry_hint=retry_hint,
            failure_anchor=failure_anchor,
        )

    def _build_prompts(self, state: GraphState, repair: RepairAttempt, gate_result: ReviewResult) -> tuple[str, str]:
        system_prompt = (
            "You are the ReviewAdvisor agent in a SATD repair workflow.\n"
            "Give concise retry guidance for the fixer based on the gate failure.\n"
            "Do not decide accept/reject. Do not write replacement code. Do not add new requirements.\n"
            "Return JSON only."
        )
        user_prompt = (
            "Prepare retry advice for this SATD repair:\n\n"
            f"SATD comment:\n{state['satd_comment']}\n\n"
            f"Analyzer/context summary:\n{self._analysis_plan_summary(state)}\n\n"
            f"Original code:\n```python\n{state['original_code']}\n```\n\n"
            f"Rejected repaired code:\n```python\n{repair.repaired_code}\n```\n\n"
            f"Retrieved context:\n{self._repair_context_summary(state)}\n\n"
            f"Gate decision: {gate_result.gate_decision}\n"
            f"Failure modes: {self._format_list(gate_result.failure_modes)}\n"
            f"Issues: {self._format_list(gate_result.issues)}\n"
            f"Rationale: {gate_result.rationale}\n\n"
            "Give only actionable constraints for the next fixer attempt. Keep them grounded in the SATD, original code, and retrieved context.\n\n"
            "Return exactly:\n"
            "{\n"
            '  "repair_constraints": ["short constraint"],\n'
            '  "retry_hint": "one short sentence",\n'
            '  "failure_anchor": "short anchor for the failed pattern"\n'
            "}\n"
        )
        return system_prompt, user_prompt

    def _copy_with_advice(
        self,
        result: ReviewResult,
        *,
        repair_constraints: list[str],
        retry_hint: str,
        failure_anchor: str,
    ) -> ReviewResult:
        return ReviewResult(
            round_id=result.round_id,
            approved=result.approved,
            issues=list(result.issues),
            candidate_mode=result.candidate_mode,
            gate_decision=result.gate_decision,
            failure_modes=list(result.failure_modes),
            review_score=result.review_score,
            problem_alignment=result.problem_alignment,
            minimality=result.minimality,
            semantic_preservation=result.semantic_preservation,
            internal_consistency=result.internal_consistency,
            revision_advice=self._revision_advice(repair_constraints, retry_hint),
            reject_type=result.reject_type,
            rationale=result.rationale,
            softened_gate_used=result.softened_gate_used,
            failed_checks=list(result.failed_checks),
            repair_constraints=repair_constraints,
            failure_anchor=failure_anchor,
            retry_hint=retry_hint,
        )

    def _analysis_plan_summary(self, state: GraphState) -> str:
        analysis = state.get("analysis")
        if analysis is None:
            return "[none]"
        parts = []
        if getattr(analysis, "context_summary", ""):
            parts.append(f"context: {analysis.context_summary}")
        return "\n".join(parts) if parts else "[none]"

    def _repair_context_summary(self, state: GraphState) -> str:
        method_inquiry = state.get("method_inquiry")
        required_methods = []
        if isinstance(method_inquiry, MethodInquiryResult):
            required_methods = method_inquiry.required_methods
        elif isinstance(method_inquiry, dict):
            required_methods = list(method_inquiry.get("required_methods") or [])
        missing = list(state.get("missing_method_names") or [])
        retrieved = list(state.get("retrieved_method_contexts") or [])
        repository_evidence = list(state.get("retrieved_repository_evidence") or [])
        repository_guidance = str(state.get("repository_evidence_guidance") or "").strip()
        lines = [
            f"required_methods: {self._format_list(required_methods)}",
            f"missing_method_names: {self._format_list(missing)}",
            f"retrieved_method_count: {len(retrieved)}",
            f"retrieved_repository_evidence_count: {len(repository_evidence)}",
        ]
        for item in retrieved[:2]:
            if isinstance(item, RetrievedMethodContext):
                method_name = item.method_name
                path = item.path
                source = item.evidence_slice or item.source
            elif isinstance(item, dict):
                method_name = str(item.get("method_name") or "")
                path = str(item.get("path") or "")
                source = str(item.get("evidence_slice") or item.get("source") or "")
            else:
                continue
            lines.append(f"- method: {method_name} path: {path}")
            if source:
                lines.append(self._truncate(source, 1200))
        for item in repository_evidence[:3]:
            if isinstance(item, RepositoryEvidenceContext):
                evidence_type = item.evidence_type
                subtype = item.evidence_subtype
                path = item.source_path
                source = item.content
            elif isinstance(item, dict):
                evidence_type = str(item.get("evidence_type") or "")
                subtype = str(item.get("evidence_subtype") or "")
                path = str(item.get("source_path") or "")
                source = str(item.get("content") or "")
            else:
                continue
            lines.append(f"- repository evidence: {evidence_type}/{subtype} path: {path}")
            if source:
                lines.append(self._truncate(source, 1200))
        if repository_guidance:
            lines.append("repository_evidence_guidance:")
            lines.append(self._truncate(repository_guidance, 1200))
        return "\n".join(lines)

    def _normalize_string_list(self, raw: Any, default: list[str]) -> list[str]:
        if not isinstance(raw, list):
            raw = [raw] if raw else []
        result: list[str] = []
        for item in raw:
            cleaned = self._one_line(item)
            if cleaned and cleaned not in result:
                result.append(cleaned)
            if len(result) >= 4:
                break
        return result or default

    def _revision_advice(self, repair_constraints: list[str], retry_hint: str) -> str:
        pieces = []
        if repair_constraints:
            pieces.append("constraints:" + ",".join(repair_constraints))
        if retry_hint:
            pieces.append("retry_hint:" + retry_hint)
        return "; ".join(pieces)

    def _format_list(self, items: list[Any]) -> str:
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        return ", ".join(cleaned[:5]) if cleaned else "[none]"

    def _one_line(self, value: Any) -> str:
        return " ".join(str(value or "").split())[:240]

    def _truncate(self, text: str, limit: int) -> str:
        compact = str(text or "").strip()
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."
