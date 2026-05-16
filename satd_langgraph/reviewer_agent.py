from __future__ import annotations

import ast
import builtins
import difflib
import json
import textwrap
from typing import Any

from .openai_client import OpenAICompatClient
from .schema import GraphState, MethodInquiryResult, RepairAttempt, RetrievedMethodContext, ReviewResult, preprocess_python_code


class OpenAIReviewer:
    def __init__(self, client: OpenAICompatClient) -> None:
        self.client = client

    def run(self, state: GraphState) -> ReviewResult:
        repair = state["latest_repair"]
        assert repair is not None

        validity_issue = self._validity_issue(state, repair)
        if validity_issue:
            return self._invalid_result(state, repair, validity_issue)

        return self._run_gate(state, repair)

    def _run_gate(self, state: GraphState, repair: RepairAttempt) -> ReviewResult:
        system_prompt, user_prompt = self._build_gate_prompts(state, repair)
        payload = self.client.generate_json(
            system_prompt,
            user_prompt,
            request_label=f"review_gate:task_{state['task_id']}:round_{repair.round_id}",
        )
        return self._coerce_gate(payload, state, repair)

    def _build_gate_prompts(self, state: GraphState, repair: RepairAttempt) -> tuple[str, str]:
        system_prompt = (
            "You are the ReviewGate agent in a SATD repair workflow.\n"
            "Decide whether the candidate repair is trustworthy enough to submit as the final answer.\n"
            "Do not provide retry advice. Do not repair the code. Do not compare against any hidden human patch.\n"
            "Use only the SATD comment, analyzer/context summary, original code, repaired code, review evidence, and retrieved context.\n"
            "Return JSON only."
        )
        user_prompt = (
            "Gate this SATD repair:\n\n"
            f"SATD comment:\n{state['satd_comment']}\n\n"
            f"Analyzer/context summary:\n{self._analysis_plan_summary(state)}\n\n"
            f"Original code:\n```python\n{state['original_code']}\n```\n\n"
            f"Repaired code:\n```python\n{repair.repaired_code}\n```\n\n"
            f"Retrieved context:\n{self._repair_context_summary(state)}\n\n"
            f"Diff summary:\n{self._diff_summary(state, repair)}\n\n"
            f"Review evidence profile:\n{json.dumps(self._diff_evidence_profile(state, repair), ensure_ascii=False, indent=2)}\n\n"
            "Choose gate_decision:\n"
            '- "accept": plausible final SATD repayment.\n'
            '- "reject": clear failed repair with code evidence.\n\n'
            "Intent-aware gate policy:\n"
            "1. Judge the candidate repair, not whether the SATD itself was worth attempting.\n"
            "2. Protected cleanup intent: remove/delete/drop, temporary/hack/workaround/debug, legacy/compatibility/deprecated, once-fixed/once-upgraded cleanup. Accept focused local deletion or replacement when it matches the SATD target; reject only clear wrong-target, no-op, syntax, opposite-intent, or much-too-broad changes.\n"
            "3. Proof-required intent: implement/support/add/handle, broad refactor/rewrite, unclear/question intent. Accept only when the candidate visibly adds or grounds the requested behavior; reject comment-only fixes, bypasses, unsupported inventions, broad unguided rewrites, or fixes that only delete the unsupported case.\n"
            "4. Local correction intent: replace/switch, bug fix/check, return/value/call adjustment. Accept small grounded edits aligned with the SATD target; reject off-target edits, unsupported new calls/names, unrelated control-flow changes, or semantic drift.\n"
            "5. Treat new names/calls as unsupported only when absent from the SATD, original code, retrieved context, local definitions/imports, and common Python behavior.\n"
            "Uncertainty in protected cleanup should lean accept; uncertainty in proof-required changes should lean reject.\n"
            "Do not require exact-match style edits or hidden human-patch knowledge.\n\n"
            "Failure modes: invalid_or_noop, wrong_target, unsupported_invention, overbroad_change, semantic_drift.\n\n"
            "Return exactly:\n"
            "{\n"
            '  "gate_decision": "accept" | "reject",\n'
            '  "failure_modes": ["invalid_or_noop" | "wrong_target" | "unsupported_invention" | "overbroad_change" | "semantic_drift"],\n'
            '  "issues": ["short issue"],\n'
            '  "rationale": "one short sentence"\n'
            "}\n"
        )
        return system_prompt, user_prompt

    def _coerce_gate(self, payload: dict[str, Any], state: GraphState, repair: RepairAttempt) -> ReviewResult:
        gate_decision = self._normalize_gate_decision(payload.get("gate_decision") or payload.get("decision"), payload.get("approved"))
        failure_modes = [] if gate_decision == "accept" else self._normalize_failure_modes(payload.get("failure_modes"))
        issues = [] if gate_decision == "accept" else self._normalize_string_list(payload.get("issues"), default=["review_gate_uncertain"])
        rationale = self._one_line(payload.get("rationale")) or ("accepted by review gate" if gate_decision == "accept" else "Review gate found a repair issue.")
        score = {"accept": 0.75, "retry": 0.45, "reject": 0.10}[gate_decision]
        return self._build_result(
            repair=repair,
            gate_decision=gate_decision,
            issues=issues,
            failure_modes=failure_modes,
            review_score=score,
            rationale=rationale,
            failure_anchor=gate_decision if gate_decision != "accept" else "",
        )

    def _build_result(
        self,
        *,
        repair: RepairAttempt,
        gate_decision: str,
        issues: list[str],
        failure_modes: list[str],
        review_score: float,
        rationale: str,
        failure_anchor: str,
    ) -> ReviewResult:
        approved = gate_decision == "accept"
        return ReviewResult(
            round_id=repair.round_id,
            approved=approved,
            issues=[] if approved else issues,
            candidate_mode=getattr(repair, "candidate_mode", "single"),
            gate_decision=gate_decision,
            failure_modes=[] if approved else failure_modes,
            review_score=review_score,
            problem_alignment=review_score,
            minimality=review_score,
            semantic_preservation=review_score,
            internal_consistency=review_score,
            revision_advice="",
            reject_type=None if approved else gate_decision,
            rationale=rationale,
            failed_checks=[] if approved else issues,
            repair_constraints=[],
            failure_anchor="" if approved else failure_anchor,
            retry_hint="",
        )

    def _validity_issue(self, state: GraphState, repair: RepairAttempt) -> str:
        repaired_code = repair.repaired_code or ""
        if not repaired_code.strip():
            return "empty_repair"
        if not self._syntax_valid(repaired_code):
            return "syntax_error"
        if self._fallback_or_noop_repair(repair) or self._same_code(state["original_code"], repaired_code):
            return "no_effective_change"
        return ""

    def _invalid_result(self, state: GraphState, repair: RepairAttempt, issue: str) -> ReviewResult:
        return self._build_result(
            repair=repair,
            gate_decision="reject",
            issues=[issue],
            failure_modes=["invalid_or_noop"],
            review_score=0.0,
            rationale=f"Invalid repair output: {issue}.",
            failure_anchor=issue,
        )

    def _analysis_plan_summary(self, state: GraphState) -> str:
        analysis = state.get("analysis")
        if analysis is None:
            return "[none]"
        parts = []
        if getattr(analysis, "context_summary", ""):
            parts.append(f"context: {analysis.context_summary}")
        if getattr(analysis, "intent_type", ""):
            parts.append(f"intent_type: {analysis.intent_type}")
        if getattr(analysis, "target_clarity", ""):
            parts.append(f"target_clarity: {analysis.target_clarity}")
        if getattr(analysis, "expected_edit_shape", ""):
            parts.append(f"expected_edit_shape: {analysis.expected_edit_shape}")
        if getattr(analysis, "target_summary", ""):
            parts.append(f"target_summary: {analysis.target_summary}")
        if getattr(analysis, "risk_note", ""):
            parts.append(f"risk_note: {analysis.risk_note}")
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
        lines = [
            f"required_methods: {self._format_list(required_methods)}",
            f"missing_method_names: {self._format_list(missing)}",
            f"retrieved_method_count: {len(retrieved)}",
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
        return "\n".join(lines)

    def _syntax_valid(self, code: str) -> bool:
        try:
            ast.parse(code or "")
            return True
        except SyntaxError:
            try:
                ast.parse(textwrap.dedent(code or ""))
                return True
            except SyntaxError:
                return False

    def _fallback_or_noop_repair(self, repair: RepairAttempt) -> bool:
        haystack = " ".join(
            [
                str(getattr(repair, "repair_plan", "") or ""),
                str(getattr(repair, "notes", "") or ""),
                str(getattr(repair, "changed_scope", "") or ""),
            ]
        ).lower()
        return "fallback no-op" in haystack or "fixer_exception" in haystack or "scope=none" in haystack

    def _same_code(self, original_code: str, repaired_code: str) -> bool:
        return (original_code or "").strip() == (repaired_code or "").strip()

    def _diff_summary(self, state: GraphState, repair: RepairAttempt) -> str:
        original_code = state["original_code"] or ""
        repaired_code = repair.repaired_code or ""
        added, deleted = self._line_delta_counts(original_code, repaired_code)
        changed = self._changed_line_count(original_code, repaired_code)
        original_lines = max(1, len(original_code.splitlines()))
        repaired_lines = len(repaired_code.splitlines())
        semantic_same = preprocess_python_code(original_code) == preprocess_python_code(repaired_code)
        return "\n".join(
            [
                f"original_line_count: {original_lines}",
                f"repaired_line_count: {repaired_lines}",
                f"changed_line_estimate: {changed}",
                f"changed_line_ratio: {round(changed / original_lines, 3)}",
                f"added_lines: {added}",
                f"deleted_lines: {deleted}",
                f"candidate_changed_scope: {getattr(repair, 'changed_scope', '')}",
                f"canonical_code_unchanged: {semantic_same}",
            ]
        )

    def _diff_evidence_profile(self, state: GraphState, repair: RepairAttempt) -> dict[str, Any]:
        original_code = state["original_code"] or ""
        repaired_code = repair.repaired_code or ""
        original_lines = max(1, len(original_code.splitlines()))
        repaired_lines = len(repaired_code.splitlines())
        changed = self._changed_line_count(original_code, repaired_code)
        added, deleted = self._line_delta_counts(original_code, repaired_code)
        original_profile = self._structure_profile(original_code)
        repaired_profile = self._structure_profile(repaired_code)
        new_calls = self._new_call_targets(original_code, repaired_code)
        unsupported_new_calls = [
            target
            for target in new_calls
            if not self._call_supported_by_input(target, state, original_code, repaired_code)
        ][:8]
        return {
            "syntax_valid": self._syntax_valid(repaired_code),
            "canonical_code_unchanged": preprocess_python_code(original_code) == preprocess_python_code(repaired_code),
            "changed_line_count": changed,
            "changed_line_ratio": round(changed / original_lines, 3),
            "added_line_count": added,
            "deleted_line_count": deleted,
            "line_count_delta": repaired_lines - original_lines,
            "candidate_changed_scope": getattr(repair, "changed_scope", ""),
            "signature_changed": self._signature_changed(original_code, repaired_code),
            "definitions_delta": repaired_profile["definitions"] - original_profile["definitions"],
            "imports_delta": repaired_profile["imports"] - original_profile["imports"],
            "branches_delta": repaired_profile["branches"] - original_profile["branches"],
            "returns_delta": repaired_profile["returns"] - original_profile["returns"],
            "raises_delta": repaired_profile["raises"] - original_profile["raises"],
            "new_call_targets": new_calls[:12],
            "possibly_unsupported_new_calls": unsupported_new_calls,
        }

    def _changed_line_count(self, original_code: str, repaired_code: str) -> int:
        original_lines = (original_code or "").splitlines()
        repaired_lines = (repaired_code or "").splitlines()
        changed = abs(len(original_lines) - len(repaired_lines))
        changed += sum(1 for before, after in zip(original_lines, repaired_lines) if before != after)
        return changed

    def _line_delta_counts(self, original_code: str, repaired_code: str) -> tuple[int, int]:
        original_lines = (original_code or "").splitlines()
        repaired_lines = (repaired_code or "").splitlines()
        added = 0
        deleted = 0
        matcher = difflib.SequenceMatcher(a=original_lines, b=repaired_lines)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag in {"insert", "replace"}:
                added += max(0, j2 - j1)
            if tag in {"delete", "replace"}:
                deleted += max(0, i2 - i1)
        return added, deleted

    def _structure_profile(self, code: str) -> dict[str, int]:
        try:
            tree = ast.parse(textwrap.dedent(code or ""))
        except SyntaxError:
            return {
                "definitions": 0,
                "imports": 0,
                "branches": 0,
                "returns": 0,
                "raises": 0,
            }
        return {
            "definitions": sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree)),
            "imports": sum(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)),
            "branches": sum(isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)) for node in ast.walk(tree)),
            "returns": sum(isinstance(node, ast.Return) for node in ast.walk(tree)),
            "raises": sum(isinstance(node, ast.Raise) for node in ast.walk(tree)),
        }

    def _signature_changed(self, original_code: str, repaired_code: str) -> bool:
        original_signature = self._first_signature_line(original_code)
        repaired_signature = self._first_signature_line(repaired_code)
        return bool(original_signature and repaired_signature and original_signature != repaired_signature)

    def _first_signature_line(self, code: str) -> str:
        for line in (code or "").splitlines():
            stripped = line.strip()
            if stripped.startswith(("def ", "async def ", "class ")):
                return stripped
        return ""

    def _new_call_targets(self, original_code: str, repaired_code: str) -> list[str]:
        original_tree = self._parse_tree(original_code)
        repaired_tree = self._parse_tree(repaired_code)
        if original_tree is None or repaired_tree is None:
            return []
        original_calls = self._call_targets(original_tree)
        repaired_calls = self._call_targets(repaired_tree)
        return sorted(repaired_calls - original_calls)

    def _parse_tree(self, code: str) -> ast.AST | None:
        try:
            return ast.parse(code or "")
        except SyntaxError:
            try:
                return ast.parse(textwrap.dedent(code or ""))
            except SyntaxError:
                return None

    def _call_targets(self, tree: ast.AST) -> set[str]:
        targets = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = self._call_target_name(node.func)
                if target:
                    targets.add(target)
        return targets

    def _call_target_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts = [node.attr]
            current = node.value
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return ""

    def _call_supported_by_input(
        self,
        target: str,
        state: GraphState,
        original_code: str,
        repaired_code: str,
    ) -> bool:
        leaf = target.rsplit(".", 1)[-1]
        if leaf in set(dir(builtins)) or leaf in self._common_method_names():
            return True
        local_defs = self._local_definition_names(repaired_code)
        if target in local_defs or leaf in local_defs:
            return True
        evidence = "\n".join(
            [
                state.get("satd_comment") or "",
                original_code or "",
                self._analysis_plan_summary(state),
                self._repair_context_summary(state),
            ]
        )
        return target in evidence or leaf in evidence

    def _local_definition_names(self, code: str) -> set[str]:
        tree = self._parse_tree(code)
        if tree is None:
            return set()
        return {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }

    def _common_method_names(self) -> set[str]:
        return {
            "append",
            "clear",
            "copy",
            "count",
            "decode",
            "encode",
            "endswith",
            "extend",
            "format",
            "get",
            "index",
            "insert",
            "items",
            "join",
            "keys",
            "lower",
            "pop",
            "remove",
            "replace",
            "setdefault",
            "sort",
            "split",
            "startswith",
            "strip",
            "update",
            "upper",
            "values",
        }

    def _normalize_gate_decision(self, raw_decision: Any, raw_approved: Any = None) -> str:
        decision = str(raw_decision or "").strip().lower()
        if decision in {"accept", "approve", "approved", "accepted", "pass"}:
            return "accept"
        if decision in {"reject", "rejected", "drop", "fail", "failed"}:
            return "reject"
        if decision in {"retry", "revise", "revision", "uncertain", "maybe"}:
            return "accept"
        if isinstance(raw_approved, bool):
            return "accept" if raw_approved else "reject"
        return "accept"

    def _normalize_failure_modes(self, raw: Any) -> list[str]:
        allowed = {
            "invalid_or_noop",
            "wrong_target",
            "unsupported_invention",
            "overbroad_change",
            "semantic_drift",
        }
        aliases = {
            "wrong_or_incomplete_target": "wrong_target",
            "unsupported_reference": "unsupported_invention",
            "ungrounded_behavior": "unsupported_invention",
        }
        result = []
        if not isinstance(raw, list):
            raw = [raw] if raw else []
        for item in raw:
            cleaned = str(item or "").strip().lower()
            cleaned = aliases.get(cleaned, cleaned)
            if cleaned in allowed and cleaned not in result:
                result.append(cleaned)
            if len(result) >= 4:
                break
        return result or ["wrong_target"]

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
