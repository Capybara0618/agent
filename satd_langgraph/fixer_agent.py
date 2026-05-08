from __future__ import annotations  

import os  
from typing import Any

from .openai_client import OpenAICompatClient
from .schema import (
    EditConstraint,  
    EvidenceCard,
    GraphState,
    MethodInquiryResult,
    RepairAttempt,
    RetrievedMethodContext,
    UncertaintyItem,
    preprocess_python_code,
)


class OpenAIFixer:  
    """Generate a SATD repair from either snippet-only or analyzer-provided context."""

    def __init__(
        self,
        client: OpenAICompatClient,
        repair_context_mode: str = "clone_treesitter",
        max_method_contexts: int = 2,
        logger: Any | None = None,
        checkpoint_callback: Any | None = None,
    ) -> None:
        self.client = client
        self.repair_context_mode = repair_context_mode
        self.max_method_contexts = max(1, int(max_method_contexts))
        self.logger = logger
        self.checkpoint_callback = checkpoint_callback

    def run(
        self,
        state: GraphState,
        candidate_mode: str = "baseline_context",
    ) -> tuple[
        RepairAttempt,
        MethodInquiryResult,
        list[RetrievedMethodContext],
        list[str],
        list[UncertaintyItem],
        list[EditConstraint],
    ]:
        round_id = int(state["round_id"]) + 1
        method_inquiry = state.get("method_inquiry") or MethodInquiryResult()
        contexts = self._contexts_for_mode(state, candidate_mode)
        missing = list(state.get("missing_method_names") or [])
        evidence_cards = self._evidence_cards(state)
        effective_mode = self._effective_mode(candidate_mode, contexts, evidence_cards)

        self._checkpoint(
            state,
            stage="generation_start",
            payload={
                "round_id": round_id,
                "candidate_mode": effective_mode,
                "required_methods": list(method_inquiry.required_methods),
                "retrieved_method_count": len(contexts),
                "evidence_card_count": len(evidence_cards),
                "missing_method_names": missing,
            },
        )

        payload = self.client.generate_json(
            *self._build_prompts(state, method_inquiry, contexts, missing, effective_mode),
            request_label=f"repair:task_{state['task_id']}:round_{round_id}:candidate_{effective_mode}",
            max_tokens=self._repair_max_tokens(),
        )
        repaired_code = str(payload.get("repaired_code") or state["original_code"])

        repair = RepairAttempt(
            round_id=round_id,
            repair_plan=self._repair_plan(state),
            repaired_code=repaired_code,
            changed_scope=self._changed_scope(state["original_code"], repaired_code),
            confidence=self._confidence(state, repaired_code, contexts, evidence_cards),
            notes=self._notes(method_inquiry, contexts, missing, evidence_cards),
            candidate_mode=effective_mode,
        )
        self._log(
            state,
            f"repair output scope={repair.changed_scope} confidence={repair.confidence:.2f} mode={effective_mode}",
        )
        self._checkpoint(
            state,
            stage="generation_done",
            payload={
                "round_id": round_id,
                "candidate_mode": effective_mode,
                "repair_plan": repair.repair_plan,
                "repaired_code": repair.repaired_code,
                "changed_scope": repair.changed_scope,
                "confidence": repair.confidence,
                "notes": repair.notes,
            },
        )
        return (
            repair,
            method_inquiry,
            contexts,
            missing,
            list(state.get("uncertainty_items") or []),
            list(state.get("edit_constraints") or []),
        )

    def _build_prompts(
        self,
        state: GraphState,
        method_inquiry: MethodInquiryResult,
        contexts: list[RetrievedMethodContext],
        missing: list[str],
        candidate_mode: str,
    ) -> tuple[str, str]:
        feedback_block = self._reviewer_feedback_block(state.get("repair_feedback") or {})
        if not self._uses_context(candidate_mode):
            return self._build_no_context_prompts(state, feedback_block)

        system_prompt = (
            "You are the fixer agent in a SATD repair workflow. "
            "Return valid JSON only with key: repaired_code. "
        )
        evidence_block = self._evidence_block(state, contexts)
        hard_rules = self._short_constraints(state)
        user_prompt = (
            "Repair the code according to the SATD comment. Respond in JSON with key repaired_code.\n\n"
            f"### SATD comment:\n{state['satd_comment']}\n\n"
            f"### Code:\n{self._code_block(state['original_code'])}\n\n"
            f"### Supporting evidence:\n{evidence_block}\n\n"
            f"### Hard rules:\n{hard_rules}\n"
            "- Use supporting evidence only to validate the smallest local repair.\n"
            "- Use normal indentation in repaired_code; do not add tab characters, column-alignment padding, or excessive whitespace.\n"
            "- When the SATD requests deleting or removing something, delete the corresponding executable code or statement, not only the SATD comment.\n"
            "- Edit only the smallest local block nearest to the SATD comment.\n"
            f"{feedback_block}"
        )
        return system_prompt, user_prompt

    def _build_no_context_prompts(self, state: GraphState, feedback_block: str = "") -> tuple[str, str]:
        system_prompt = (
            "You are the fixer agent in a SATD repair workflow. "
        )
        user_prompt = (
            "Repair the code according to the SATD comment. Respond in JSON with key repaired_code.\n\n"
            f"### SATD comment:\n{state['satd_comment']}\n\n"
            f"Code:\n{self._code_block(state['original_code'])}\n\n"
            "- Use normal indentation in repaired_code; do not add tab characters, column-alignment padding, or excessive whitespace.\n"
            "- When the SATD requests deleting or removing something, delete the corresponding executable code or statement, not only the SATD comment.\n"
            f"{feedback_block}"
        )
        return system_prompt, user_prompt

    def _evidence_cards(self, state: GraphState) -> list[EvidenceCard]:
        return [item for item in (state.get("evidence_cards") or []) if isinstance(item, EvidenceCard)]

    def _evidence_block(self, state: GraphState, contexts: list[RetrievedMethodContext]) -> str:
        cards = [
            card
            for card in self._evidence_cards(state)
            if getattr(card, "answer", "") and card.relevance in {"high", "medium"} and card.polarity != "weak"
        ][:1]
        if cards:
            blocks: list[str] = []
            for card in cards:
                lines = [
                    f"- [{card.tool}] {card.target}",
                    f"  polarity: {card.polarity}",
                    f"  decision: {card.decision}",
                    f"  answer: {card.answer}",
                    f"  edit_hint: {card.edit_hint}",
                ]
                snippet = self._short_snippet(card.snippet)
                if snippet:
                    lines.append("  snippet:")
                    lines.extend(f"    {line}" for line in snippet.splitlines())
                blocks.append("\n".join(lines))
            return "\n".join(blocks)
        if contexts:
            cards = []
            for context in contexts[:3]:
                snippet = self._short_snippet(context.evidence_slice or context.source)
                lines = [
                    f"- [symbol_definition] {context.method_name}",
                    f"  polarity: support",
                    f"  fact: Definition found in {context.path}:{context.start_line or ''}.",
                    "  patch_use: Use the signature or declaration only if it directly constrains the local repair.",
                ]
                if snippet:
                    lines.append("  snippet:")
                    lines.extend(f"    {line}" for line in snippet.splitlines())
                cards.append("\n".join(lines))
            return "\n".join(cards)
        return "[none]"

    def _short_constraints(self, state: GraphState) -> str:
        lines = [
            "- Make the smallest plausible local edit.",
            "- Preserve the existing function/class signature unless the SATD explicitly asks for a signature-local fix.",
            "- Do not add new helpers, new control flow, or unrelated rewrites.",
            "- Keep unchanged lines unchanged whenever possible.",
            "- If evidence polarity is anchor_to_delete, do not preserve or reintroduce that behavior.",
        ]
        analysis = state.get("analysis")
        for item in getattr(analysis, "repair_constraints", [])[:3] if analysis else []:
            text = " ".join(str(item or "").split())
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines)

    def _short_snippet(self, snippet: str, max_lines: int = 8) -> str:
        lines = [line.rstrip() for line in str(snippet or "").splitlines() if line.strip()]
        return "\n".join(lines[:max_lines])[:1200]

    def _format_generic_constraint_block(self, edit_constraints: list[EditConstraint]) -> str:
        lines = [
            "- Make the smallest plausible local edit.",
            "- Preserve the existing function/class signature unless the SATD explicitly asks for a signature-local fix.",
            "- Do not add new helpers, new control flow, or unrelated rewrites.",
            "- Keep unchanged lines unchanged whenever possible.",
        ]
        for item in edit_constraints[:3]:
            if not isinstance(item, EditConstraint):
                continue
            focus = str(item.focus_point or "").strip()
            if focus:
                lines.append(f"- Focus on: {focus}")
            must_do = str(item.must_do or "").strip()
            if must_do:
                lines.append(f"- Must do: {must_do}")
            must_not_do = str(item.must_not_do or "").strip()
            if must_not_do:
                lines.append(f"- Must not do: {must_not_do}")
        return "\n".join(lines)

    def _contexts_for_mode(self, state: GraphState, candidate_mode: str) -> list[RetrievedMethodContext]:
        if not self._uses_context(candidate_mode):
            return []
        return [
            item
            for item in (state.get("retrieved_method_contexts") or [])
            if isinstance(item, RetrievedMethodContext) and item.found
        ][: self.max_method_contexts]

    def _context_block(
        self,
        method_inquiry: MethodInquiryResult,
        contexts: list[RetrievedMethodContext],
        missing: list[str],
    ) -> str:
        if not contexts and not missing and not method_inquiry.required_methods:
            return "[none]"
        lines: list[str] = []
        if method_inquiry.required_methods:
            lines.append("required methods: " + ", ".join(method_inquiry.required_methods))
        for context in contexts:
            lines.append(f"- method: {context.method_name}")
            if context.path:
                span = f":{context.start_line}" if context.start_line else ""
                if span and context.end_line and context.end_line != context.start_line:
                    span += f"-{context.end_line}"
                lines.append(f"  location: {context.path}{span}")
            source = (context.evidence_slice or context.source or "").strip()
            if source:
                lines.append("  source:")
                lines.extend(f"    {line}" for line in source.splitlines()[:80])
        if missing:
            lines.append("missing methods: " + ", ".join(missing))
        return "\n".join(lines)

    def _reviewer_feedback_block(self, repair_feedback: dict[str, Any]) -> str:
        if not repair_feedback:
            return ""
        constraints = [
            str(item).strip()
            for item in repair_feedback.get("repair_constraints", [])
            if str(item).strip()
        ]
        retry_hint = " ".join(str(repair_feedback.get("retry_hint") or "").split())
        lines = [
            "\n### Reviewer feedback from previous attempt:",
            "Use this only to avoid the previous failed pattern; do not treat it as a new requirement.",
        ]
        if constraints:
            lines.append("Repair constraints: " + ", ".join(constraints[:4]))
        if retry_hint:
            lines.append("Retry hint: " + retry_hint)
        return "\n".join(lines) + "\n"

    def _uses_context(self, candidate_mode: str) -> bool:
        mode = str(candidate_mode or "").strip().lower()
        return not mode.endswith("no_context") and self.repair_context_mode in {"method_query", "clone_treesitter"}

    def _effective_mode(
        self,
        candidate_mode: str,
        contexts: list[RetrievedMethodContext],
        evidence_cards: list[EvidenceCard],
    ) -> str:
        mode = str(candidate_mode or "").strip() or "single"
        if not self._uses_context(mode) or contexts or evidence_cards:
            return mode
        if mode == "baseline_context":
            return "baseline_no_context"
        return f"{mode}_no_context"

    def _repair_max_tokens(self) -> int:
        try:
            return max(512, int(os.environ.get("OPENAI_REPAIR_MAX_TOKENS") or 4096))
        except ValueError:
            return 4096

    def _repair_plan(self, state: GraphState) -> str:
        comment = " ".join(str(state.get("satd_comment") or "").split())
        if len(comment) > 160:
            comment = comment[:157] + "..."
        return f"Apply the smallest local repair required by the SATD comment: {comment}"

    def _confidence(
        self,
        state: GraphState,
        repaired_code: str,
        contexts: list[RetrievedMethodContext],
        evidence_cards: list[EvidenceCard],
    ) -> float:
        if preprocess_python_code(state["original_code"]) == preprocess_python_code(repaired_code):
            return 0.35
        return 0.60 if contexts or evidence_cards else 0.50

    def _notes(
        self,
        method_inquiry: MethodInquiryResult,
        contexts: list[RetrievedMethodContext],
        missing: list[str],
        evidence_cards: list[EvidenceCard],
    ) -> str:
        if evidence_cards:
            return f"evidence_cards={len(evidence_cards)}"
        if contexts:
            return f"method_contexts={len(contexts)}"
        if method_inquiry.required_methods:
            return "no_method_context_found" + (f"; missing={','.join(missing)}" if missing else "")
        return "no_required_methods_identified"

    def _changed_scope(self, original_code: str, repaired_code: str) -> str:
        if (original_code or "") == (repaired_code or ""):
            return "line"
        original_lines = (original_code or "").splitlines()
        repaired_lines = (repaired_code or "").splitlines()
        changed = abs(len(original_lines) - len(repaired_lines))
        changed += sum(1 for before, after in zip(original_lines, repaired_lines) if before != after)
        if changed <= 2:
            return "line"
        if changed <= 12:
            return "function"
        if changed <= 40:
            return "class"
        return "file"

    def _code_block(self, code: str) -> str:
        return "\n".join(line.rstrip() for line in (code or "").expandtabs(4).splitlines())

    def _log(self, state: GraphState, message: str) -> None:
        if callable(self.logger):
            self.logger(f"{self._task_prefix(state)} {message}")

    def _checkpoint(self, state: GraphState, stage: str, payload: dict[str, Any]) -> None:
        if callable(self.checkpoint_callback):
            self.checkpoint_callback(state, stage, payload)

    def _task_prefix(self, state: GraphState) -> str:
        task_id = state.get("task_id", "?")
        task_index = state.get("task_index") or 0
        task_total = state.get("task_total") or 0
        if task_index and task_total:
            return f"[task {task_id} {task_index}/{task_total}]"
        return f"[task {task_id}]"
