from __future__ import annotations

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
        return AnalysisResult(
            decision="pass",
            repairable=True,
            reason="method_context_prepared",
            repair_plan="",
            target_summary="",
            context_summary=self._context_summary(required_methods, found_count, missing),
        )

    def fallback_analysis(self, state: GraphState, reason: str) -> AnalysisResult:
        return AnalysisResult(
            decision="pass",
            repairable=True,
            reason=reason,
            repair_plan="",
            target_summary="",
            context_summary="[none]",
        )

    def _context_summary(self, required_methods: list[str], found_count: int, missing: list[str]) -> str:
        parts = [
            f"identified_methods={len(required_methods)}",
            f"retrieved_methods={found_count}",
        ]
        if missing:
            parts.append("missing_methods=" + ",".join(missing[:5]))
        return "; ".join(parts)
