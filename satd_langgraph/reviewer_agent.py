from __future__ import annotations

import ast
import difflib
import json
import re
import textwrap
from typing import Any

from .openai_client import OpenAICompatClient
from .schema import GraphState, MethodInquiryResult, RepairAttempt, RetrievedMethodContext, ReviewResult, preprocess_python_code


class OpenAIReviewer:
    LLM_FAILED_CHECKS = {
        "comment_only_without_satd_support",
        "anchor_not_modified",
        "unrelated_change",
        "unsupported_control_flow_change",
        "over_expanded_change",
        "over_scoped_change",
        "not_addressing_satd",
        "semantic_drift_risk",
        "context_conflict",
        "reviewer_uncertain",
    }
    HARD_FAIL_CHECKS = {"syntax_error", "empty_repair", "no_effective_change"}
    CHECK_CONSTRAINTS = {
        "syntax_error": ["return_valid_python_code"],
        "empty_repair": ["return_non_empty_repaired_code"],
        "no_effective_change": ["modify_code_to_address_satd"],
        "comment_only_without_satd_support": ["avoid_comment_only_change_unless_satd_requests_it"],
        "unsupported_signature_change": ["preserve_original_signature"],
        "new_helper_without_evidence": ["avoid_new_helpers"],
        "unsupported_control_flow_change": ["avoid_new_control_flow_paths"],
        "over_scoped_change": ["preserve_unrelated_code_and_formatting"],
        "over_expanded_change": ["avoid_large_additions"],
        "anchor_not_modified": ["modify_satd_anchor_region"],
        "unrelated_change": ["avoid_unrelated_rewrite"],
        "not_addressing_satd": ["make_one_direct_change_matching_satd"],
        "semantic_drift_risk": ["preserve_existing_behavior"],
        "context_conflict": ["follow_retrieved_context_only"],
        "reviewer_uncertain": ["make_smallest_local_edit"],
    }

    def __init__(self, client: OpenAICompatClient) -> None:
        self.client = client

    def run(self, state: GraphState) -> ReviewResult:
        repair = state["latest_repair"]
        assert repair is not None

        evidence = self._structural_evidence_payload(state, repair)
        local_failed_checks = self._local_failed_checks(state, repair)
        local_constraints = self._constraints_for_checks(local_failed_checks)
        structural_risk_checks = [] if local_failed_checks else self._structural_risk_checks(state, repair, evidence)
        payload: dict[str, Any] = {}

        if not local_failed_checks:
            system_prompt, user_prompt = self._build_checklist_prompts(state, repair, evidence)
            payload = self.client.generate_json(
                system_prompt,
                user_prompt,
                request_label=f"review:task_{state['task_id']}:round_{repair.round_id}",
            )

        llm_failed_checks = self._normalize_failed_checks(payload.get("failed_checks", []), allowed=self.LLM_FAILED_CHECKS)
        llm_constraints = self._normalize_constraints(payload.get("repair_constraints", []))
        llm_retry_hint = self._normalize_retry_hint(payload.get("retry_hint"))
        approved_by_llm = bool(payload.get("approved"))
        if not local_failed_checks and not structural_risk_checks and not approved_by_llm and not llm_failed_checks:
            llm_failed_checks = ["reviewer_uncertain"]

        anchor_supported = self._has_satd_anchor_support(evidence)
        anchor_protected = self._has_satd_anchor_protection(evidence)
        instruction_aligned = self._has_instruction_aligned_support(evidence)
        if anchor_supported or instruction_aligned:
            structural_risk_checks = self._clear_supported_risks(structural_risk_checks, evidence)
            llm_failed_checks = self._clear_supported_risks(llm_failed_checks, evidence)
            llm_constraints = [
                item
                for item in llm_constraints
                if item not in {"make_one_direct_change_matching_satd", "modify_satd_anchor_region"}
            ]
        structural_risk_checks = self._clear_inapplicable_risks(state, structural_risk_checks, evidence)
        llm_failed_checks = self._clear_inapplicable_risks(state, llm_failed_checks, evidence)

        failed_checks = self._dedupe_limit([*local_failed_checks, *structural_risk_checks, *llm_failed_checks], limit=3)
        if self._reviewer_uncertain_is_viable(repair, failed_checks):
            failed_checks = []
            llm_constraints = []
            llm_retry_hint = ""

        repair_constraints = self._dedupe_limit(
            [*local_constraints, *llm_constraints, *self._constraints_for_checks(failed_checks)],
            limit=4,
        )
        approved = not failed_checks and (
            approved_by_llm or anchor_protected or anchor_supported or instruction_aligned or repair.round_id >= 2
        )
        failure_anchor = "" if approved else self._failure_anchor(failed_checks)
        compat_score = 1.0 if approved else 0.0
        retry_hint = "" if approved else self._retry_hint_for_repair(
            failed_checks=failed_checks,
            evidence=evidence,
            llm_retry_hint=llm_retry_hint,
        )
        return ReviewResult(
            round_id=repair.round_id,
            approved=approved,
            issues=failed_checks,
            candidate_mode=getattr(repair, "candidate_mode", "single"),
            review_score=compat_score,
            problem_alignment=compat_score,
            minimality=compat_score,
            semantic_preservation=compat_score,
            internal_consistency=compat_score,
            revision_advice="" if approved else "constraints:" + ",".join(repair_constraints),
            reject_type=failure_anchor or None,
            rationale="approved" if approved else f"failed_checks:{','.join(failed_checks)}",
            failed_checks=failed_checks,
            repair_constraints=repair_constraints,
            failure_anchor=failure_anchor,
            retry_hint=retry_hint,
        )

    def _build_checklist_prompts(
        self,
        state: GraphState,
        repair: RepairAttempt,
        evidence: dict[str, Any],
    ) -> tuple[str, str]:
        system_prompt = (
            "You are the reviewer agent in a SATD repair workflow. "
            "Act as a viability gate, not as a repair generator or precision-only filter. "
            "Use only the SATD comment, original code, repaired code, and repair method context. "
            "Do not use analyzer decisions or exact-match assumptions. "
            "Identify concrete failure evidence after repair, then approve only repairs that remain viable. "
            "Treat clear SATD-anchor edits as strong positive evidence. "
            "Treat local edits that follow explicit SATD instructions such as replace, change, rename, enable, uncomment, remove, hack, workaround, or use-instead as positive evidence. "
            "Return strict JSON only with keys: approved, failed_checks, repair_constraints, failure_anchor, retry_hint."
        )
        allowed_checks = ", ".join(sorted(self.LLM_FAILED_CHECKS))
        user_prompt = (
            "Review the repaired code with this fixed checklist:\n"
            "1. SATD alignment: does the diff modify the SATD anchor region or the same local logic?\n"
            "2. Scope: does the diff avoid unrelated rewrites, unrelated deletion, and broad formatting churn?\n"
            "3. Semantics: does the diff avoid unsupported helper, return, raise, exception, or branch paths?\n"
            "4. Context: does the diff avoid direct contradiction with retrieved method context?\n\n"
            f"Use only these LLM-level failed_checks when rejecting: {allowed_checks}.\n"
            "Set approved=false when concrete failure evidence exists. Use reviewer_uncertain only when evidence is inconclusive.\n"
            "Do not use a fixed discard ratio, labels, or exact-match assumptions.\n"
            "Do not use deterministic checks such as no_effective_change or syntax_error; those are handled before this review.\n"
            "Use anchor_not_modified when the SATD anchor exists, the repair changes elsewhere, and the changed region is not local to the anchor.\n"
            "Use unrelated_change when the repair deletes or rewrites clearly unrelated executable logic.\n"
            "Use over_expanded_change when the repair adds many new lines, helpers, branches, imports, or implementation detail beyond a minimal SATD fix.\n"
            "Use unsupported_control_flow_change when the repair adds return, raise, exception, branch, or helper paths without SATD/context support.\n"
            "Do not reject local replacements, renames, exception changes, enabled calls, removed hacks, or workaround cleanup when those operations are explicitly requested by the SATD.\n"
            "Use not_addressing_satd only when the repair clearly ignores the SATD and has no plausible path to success.\n"
            "Do not use not_addressing_satd when the diff removes or directly edits the SATD anchor region, especially an if False, temporary, hack, obsolete, TODO, FIXME, or XXX block.\n"
            "Comment/docstring-only changes can be valid when the SATD is about documentation, descriptions, TODO/FIXME markers, comments, or cleanup.\n"
            "Incomplete or missing method context is not by itself a rejection reason, but a direct context contradiction is.\n"
            "Do not reject solely because formatting, comments, or docstrings changed. "
            "retry_hint is optional and used only for a second repair attempt. "
            "If provided, keep it one short sentence under 30 words. "
            "Do not propose replacement code, new APIs, or requirements not present in the SATD/context. "
            "Do not provide a long rationale.\n\n"
            f"Round: {repair.round_id}\n"
            f"Repository owner: {state['user']}\n"
            f"Repository name: {state['project']}\n"
            f"File path: {state['file_path']}\n"
            f"SATD comment:\n{state['satd_comment']}\n\n"
            f"Original code:\n{state['original_code']}\n\n"
            f"Repaired code:\n{repair.repaired_code}\n\n"
            f"Structural evidence:\n{json.dumps(evidence, ensure_ascii=False)}\n\n"
            f"Repair context:\n{self._repair_context_summary(state)}\n"
        )
        return system_prompt, user_prompt

    def _repair_context_summary(self, state: GraphState) -> str:
        method_inquiry = state.get("method_inquiry")
        required_methods = []
        if isinstance(method_inquiry, MethodInquiryResult):
            required_methods = method_inquiry.required_methods
        elif isinstance(method_inquiry, dict):
            required_methods = list(method_inquiry.get("required_methods") or [])
        retrieved = list(state.get("retrieved_method_contexts") or [])
        missing = list(state.get("missing_method_names") or [])
        lines = [
            f"required_methods: {', '.join(str(item) for item in required_methods[:5]) or '[none]'}",
            f"missing_method_names: {', '.join(str(item) for item in missing[:5]) or '[none]'}",
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

    def _local_failed_checks(self, state: GraphState, repair: RepairAttempt) -> list[str]:
        original_code = state["original_code"] or ""
        repaired_code = repair.repaired_code or ""
        if not repaired_code.strip():
            return ["empty_repair"]
        if not self._syntax_valid(repaired_code):
            return ["syntax_error"]
        if original_code.strip() == repaired_code.strip() or self._fallback_or_noop_repair(repair):
            return ["no_effective_change"]
        return []

    def _structural_evidence_payload(self, state: GraphState, repair: RepairAttempt) -> dict[str, Any]:
        original_code = state["original_code"] or ""
        repaired_code = repair.repaired_code or ""
        original_profile = self._structure_profile(original_code)
        repaired_profile = self._structure_profile(repaired_code)
        normalized_original = preprocess_python_code(original_code)
        normalized_repaired = preprocess_python_code(repaired_code)
        changed_line_count = self._changed_line_count(original_code, repaired_code)
        original_line_count = len(original_code.splitlines())
        repaired_line_count = len(repaired_code.splitlines())
        added_line_count, deleted_line_count = self._line_delta_counts(original_code, repaired_code)
        anchor_evidence = self._satd_anchor_evidence(state["satd_comment"], original_code, repaired_code)
        instruction_evidence = self._satd_instruction_evidence(state["satd_comment"], original_code, repaired_code)
        changed_ratio = changed_line_count / max(1, original_line_count)
        return {
            "changed_scope": repair.changed_scope,
            "changed_line_count": changed_line_count,
            "original_line_count": original_line_count,
            "repaired_line_count": repaired_line_count,
            "added_line_count": added_line_count,
            "deleted_line_count": deleted_line_count,
            "line_count_delta": repaired_line_count - original_line_count,
            "changed_line_ratio": round(changed_ratio, 3),
            "raw_code_changed": original_code.strip() != repaired_code.strip(),
            "normalized_code_changed": normalized_original != normalized_repaired,
            "comment_or_format_only_change": normalized_original == normalized_repaired and original_code.strip() != repaired_code.strip(),
            "satd_mentions_comment_or_documentation": self._satd_mentions_comment_or_documentation(state["satd_comment"]),
            "fallback_or_noop_repair": self._fallback_or_noop_repair(repair),
            "signature_changed": self._signature_changed(original_code, repaired_code),
            "signature_change_kind": self._signature_change_kind(original_code, repaired_code),
            "definitions_delta": repaired_profile["definitions"] - original_profile["definitions"],
            "returns_delta": repaired_profile["returns"] - original_profile["returns"],
            "raises_delta": repaired_profile["raises"] - original_profile["raises"],
            "branches_delta": repaired_profile["branches"] - original_profile["branches"],
            "imports_delta": repaired_profile["imports"] - original_profile["imports"],
            **anchor_evidence,
            **instruction_evidence,
        }

    def _structural_risk_checks(self, state: GraphState, repair: RepairAttempt, evidence: dict[str, Any]) -> list[str]:
        checks: list[str] = []
        anchor_supported = self._has_satd_anchor_support(evidence)
        satd_comment = state.get("satd_comment") or ""
        if evidence.get("comment_or_format_only_change") and not evidence.get("satd_mentions_comment_or_documentation"):
            checks.append("comment_only_without_satd_support")
        if evidence.get("satd_anchor_found") and evidence.get("raw_code_changed") and not anchor_supported and not evidence.get("changed_near_satd_anchor"):
            checks.append("anchor_not_modified")
        if self._has_unrelated_change_risk(evidence):
            checks.append("unrelated_change")
        if self._has_over_expanded_change_risk(state, evidence):
            checks.append("over_expanded_change")
        if self._has_unsupported_control_flow_risk(satd_comment, evidence, anchor_supported):
            checks.append("unsupported_control_flow_change")
        if evidence.get("signature_changed") and not self._signature_change_allowed(satd_comment, original_code=state["original_code"], repaired_code=repair.repaired_code) and not self._has_instruction_aligned_support(evidence):
            checks.append("unsupported_signature_change")
        if self._has_new_helper_risk(state, evidence, anchor_supported):
            checks.append("new_helper_without_evidence")
        return self._dedupe_limit(checks, limit=3)

    def _fallback_or_noop_repair(self, repair: RepairAttempt) -> bool:
        haystack = " ".join(
            [
                str(getattr(repair, "repair_plan", "") or ""),
                str(getattr(repair, "notes", "") or ""),
                str(getattr(repair, "changed_scope", "") or ""),
            ]
        ).lower()
        return "fallback no-op" in haystack or "fixer_exception" in haystack or "scope=none" in haystack

    def _has_satd_anchor_protection(self, evidence: dict[str, Any]) -> bool:
        if evidence.get("fallback_or_noop_repair") or not evidence.get("raw_code_changed"):
            return False
        return bool(
            evidence.get("satd_anchor_removed")
            or evidence.get("satd_block_removed")
            or evidence.get("dead_or_temporary_block_removed")
        )

    def _has_satd_anchor_support(self, evidence: dict[str, Any]) -> bool:
        if evidence.get("fallback_or_noop_repair") or not evidence.get("raw_code_changed"):
            return False
        return bool(
            evidence.get("satd_anchor_changed")
            or evidence.get("satd_anchor_removed")
            or evidence.get("satd_block_removed")
            or evidence.get("dead_or_temporary_block_removed")
            or evidence.get("changed_near_satd_anchor")
        )

    def _has_instruction_aligned_support(self, evidence: dict[str, Any]) -> bool:
        return bool(evidence.get("raw_code_changed") and evidence.get("instruction_aligned_change"))

    def _clear_supported_risks(self, checks: list[str], evidence: dict[str, Any]) -> list[str]:
        clearable = {
            "anchor_not_modified",
            "not_addressing_satd",
            "over_scoped_change",
            "semantic_drift_risk",
            "reviewer_uncertain",
        }
        if evidence.get("instruction_aligned_change"):
            clearable.update({"unsupported_signature_change", "unsupported_control_flow_change", "unrelated_change"})
        return [check for check in checks if check not in clearable]

    def _clear_inapplicable_risks(
        self,
        state: GraphState,
        checks: list[str],
        evidence: dict[str, Any],
    ) -> list[str]:
        cleared: list[str] = []
        for check in checks:
            if check == "over_expanded_change" and not self._has_over_expanded_change_risk(state, evidence):
                continue
            cleared.append(check)
        return cleared

    def _reviewer_uncertain_is_viable(self, repair: RepairAttempt, failed_checks: list[str]) -> bool:
        return repair.round_id >= 2 and failed_checks == ["reviewer_uncertain"]

    def _has_unrelated_change_risk(self, evidence: dict[str, Any]) -> bool:
        if not evidence.get("raw_code_changed") or evidence.get("dead_or_temporary_block_removed") or evidence.get("instruction_aligned_change"):
            return False
        changed_line_count = int(evidence.get("changed_line_count") or 0)
        changed_line_ratio = float(evidence.get("changed_line_ratio") or 0.0)
        non_satd_deleted = int(evidence.get("non_satd_deleted_executable_line_count") or 0)
        if non_satd_deleted >= 3 and not evidence.get("satd_anchor_removed"):
            return True
        return changed_line_count >= 8 and changed_line_ratio >= 0.6 and not evidence.get("satd_anchor_changed")

    def _has_over_expanded_change_risk(self, state: GraphState, evidence: dict[str, Any]) -> bool:
        if not evidence.get("raw_code_changed") or evidence.get("fallback_or_noop_repair"):
            return False
        if self._satd_mentions_comment_or_documentation(state.get("satd_comment") or ""):
            return False
        added_line_count = int(evidence.get("added_line_count") or 0)
        original_line_count = max(1, int(evidence.get("original_line_count") or 0))
        repaired_line_count = int(evidence.get("repaired_line_count") or 0)
        line_count_delta = int(evidence.get("line_count_delta") or 0)
        if line_count_delta <= 0 or repaired_line_count <= original_line_count:
            return False
        structure_expanded = bool(
            int(evidence.get("definitions_delta") or 0) > 0
            or int(evidence.get("imports_delta") or 0) > 0
            or int(evidence.get("branches_delta") or 0) >= 2
            or (int(evidence.get("returns_delta") or 0) + int(evidence.get("raises_delta") or 0)) >= 2
        )
        return (
            added_line_count >= 10
            or line_count_delta >= 8
            or (added_line_count >= 6 and added_line_count / original_line_count >= 0.50)
            or (added_line_count >= 4 and structure_expanded and not evidence.get("instruction_aligned_change"))
        )

    def _has_unsupported_control_flow_risk(self, satd_comment: str, evidence: dict[str, Any], anchor_supported: bool) -> bool:
        if anchor_supported or evidence.get("instruction_aligned_change") or self._satd_mentions_control_flow(satd_comment):
            return False
        return bool(
            int(evidence.get("returns_delta") or 0) > 0
            or int(evidence.get("raises_delta") or 0) > 0
            or int(evidence.get("branches_delta") or 0) > 0
        )

    def _has_new_helper_risk(self, state: GraphState, evidence: dict[str, Any], anchor_supported: bool) -> bool:
        if anchor_supported or evidence.get("instruction_aligned_change") or int(evidence.get("definitions_delta") or 0) <= 0:
            return False
        return not bool(state.get("retrieved_method_contexts") or [])

    def _satd_anchor_evidence(self, satd_comment: str, original_code: str, repaired_code: str) -> dict[str, Any]:
        original_lines = (original_code or "").splitlines()
        repaired_lines = (repaired_code or "").splitlines()
        changed_indices, deleted_indices = self._changed_original_line_indices(original_lines, repaired_lines)
        anchor_indices = [idx for idx, line in enumerate(original_lines) if self._line_matches_satd_anchor(line, satd_comment)]
        touched_anchor_indices = [idx for idx in anchor_indices if idx in changed_indices]
        removed_anchor_indices = [idx for idx in anchor_indices if idx in deleted_indices]
        deleted_lines = [original_lines[idx] for idx in sorted(deleted_indices) if 0 <= idx < len(original_lines)]
        nearest_changed_distance = self._nearest_index_distance(anchor_indices, changed_indices)
        non_satd_deleted_executable_lines = [
            original_lines[idx]
            for idx in sorted(deleted_indices)
            if idx not in set(anchor_indices) and 0 <= idx < len(original_lines) and self._is_executable_line(original_lines[idx])
        ]
        satd_block_removed = bool(removed_anchor_indices and len(deleted_indices) > 1)
        dead_or_temporary_block_removed = bool(
            satd_block_removed and any(self._line_has_dead_or_temporary_marker(line) for line in deleted_lines)
        )
        return {
            "satd_anchor_found": bool(anchor_indices),
            "satd_anchor_changed": bool(touched_anchor_indices),
            "satd_anchor_removed": bool(removed_anchor_indices),
            "satd_block_removed": satd_block_removed,
            "dead_or_temporary_block_removed": dead_or_temporary_block_removed,
            "non_satd_deleted_executable_line_count": len(non_satd_deleted_executable_lines),
            "nearest_changed_distance_to_satd_anchor": nearest_changed_distance,
            "changed_near_satd_anchor": nearest_changed_distance is not None and nearest_changed_distance <= 2,
        }

    def _satd_instruction_evidence(self, satd_comment: str, original_code: str, repaired_code: str) -> dict[str, Any]:
        lowered = (satd_comment or "").lower()
        changed_text = self._changed_text(original_code, repaired_code).lower()
        instruction_aligned = False
        if re.search(r"\b(remove|delete|drop|disable|revert|cleanup|clean up)\b", lowered):
            instruction_aligned = self._line_delta_counts(original_code, repaired_code)[1] > 0 or "pass" in changed_text
        elif re.search(r"\b(replace|rename|change|use|honou?r|default|return|raise|annotat|document|description)\b", lowered):
            instruction_aligned = bool(changed_text.strip())
        elif re.search(r"\b(hack|workaround|temporary|temp|obsolete|deprecated)\b", lowered):
            instruction_aligned = bool(changed_text.strip())
        return {"instruction_aligned_change": instruction_aligned}

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

    def _structure_profile(self, code: str) -> dict[str, int]:
        source = textwrap.dedent(code or "")
        try:
            tree = ast.parse(source)
            return {
                "definitions": sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree)),
                "returns": sum(isinstance(node, ast.Return) for node in ast.walk(tree)),
                "raises": sum(isinstance(node, ast.Raise) for node in ast.walk(tree)),
                "branches": sum(isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)) for node in ast.walk(tree)),
                "imports": sum(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)),
            }
        except SyntaxError:
            return {
                "definitions": len(re.findall(r"^\s*(?:async\s+def|def|class)\s+", source, flags=re.MULTILINE)),
                "returns": len(re.findall(r"\breturn\b", source)),
                "raises": len(re.findall(r"\braise\b", source)),
                "branches": len(re.findall(r"\b(if|for|while|try|with)\b", source)),
                "imports": len(re.findall(r"^\s*(?:import|from)\s+", source, flags=re.MULTILINE)),
            }

    def _first_signature_line(self, code: str) -> str:
        for line in (code or "").splitlines():
            stripped = line.strip()
            if stripped.startswith(("def ", "async def ", "class ")):
                return stripped
        return ""

    def _signature_changed(self, original_code: str, repaired_code: str) -> bool:
        original_signature = self._first_signature_line(original_code)
        repaired_signature = self._first_signature_line(repaired_code)
        return bool(original_signature and repaired_signature and original_signature != repaired_signature)

    def _signature_change_allowed(self, satd_comment: str, original_code: str, repaired_code: str) -> bool:
        lowered = (satd_comment or "").lower()
        return bool(re.search(r"\b(type|annotat|signature|parameter|return type)\b", lowered))

    def _signature_change_kind(self, original_code: str, repaired_code: str) -> str:
        if not self._signature_changed(original_code, repaired_code):
            return "unchanged"
        return "changed"

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

    def _changed_original_line_indices(self, original_lines: list[str], repaired_lines: list[str]) -> tuple[set[int], set[int]]:
        changed: set[int] = set()
        deleted: set[int] = set()
        matcher = difflib.SequenceMatcher(a=original_lines, b=repaired_lines)
        for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            changed.update(range(i1, i2))
            if tag in {"delete", "replace"}:
                deleted.update(range(i1, i2))
        return changed, deleted

    def _line_matches_satd_anchor(self, line: str, satd_comment: str) -> bool:
        lowered = line.lower()
        if any(marker in lowered for marker in ("todo", "fixme", "xxx", "hack", "workaround", "temporary", "deprecated", "obsolete")):
            return True
        tokens = self._code_tokens(satd_comment)
        if not tokens:
            return False
        line_tokens = self._code_tokens(line)
        return bool(tokens & line_tokens)

    def _code_tokens(self, text: str) -> set[str]:
        ignored = {"todo", "fixme", "xxx", "this", "that", "with", "from", "after", "before", "should", "would"}
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text or "")
            if token.lower() not in ignored
        }

    def _changed_text(self, original_code: str, repaired_code: str) -> str:
        diff = difflib.ndiff((original_code or "").splitlines(), (repaired_code or "").splitlines())
        return "\n".join(line for line in diff if line.startswith(("+ ", "- ")))

    def _nearest_index_distance(self, anchors: list[int], changed: set[int]) -> int | None:
        if not anchors or not changed:
            return None
        return min(abs(anchor - item) for anchor in anchors for item in changed)

    def _is_executable_line(self, line: str) -> bool:
        stripped = line.strip()
        return bool(stripped and not stripped.startswith("#") and not stripped.startswith(("'''", '"""')))

    def _line_has_dead_or_temporary_marker(self, line: str) -> bool:
        lowered = line.lower()
        return bool(re.search(r"\b(todo|fixme|xxx|hack|workaround|temporary|temp|obsolete|deprecated|remove later)\b", lowered))

    def _satd_mentions_comment_or_documentation(self, satd_comment: str) -> bool:
        return bool(re.search(r"\b(comment|docstring|document|description|docs?|todo|fixme|xxx)\b", satd_comment or "", flags=re.IGNORECASE))

    def _satd_mentions_control_flow(self, satd_comment: str) -> bool:
        return bool(re.search(r"\b(return|raise|exception|if|branch|loop|try|fallback)\b", satd_comment or "", flags=re.IGNORECASE))

    def _normalize_failed_checks(self, raw: Any, allowed: set[str] | None = None) -> list[str]:
        if not isinstance(raw, list):
            raw = [raw] if raw else []
        allowed = allowed or (self.LLM_FAILED_CHECKS | self.HARD_FAIL_CHECKS)
        checks = []
        for item in raw:
            for part in self._split_failed_check_item(item):
                tag = self._normalize_tag(part)
                if tag:
                    checks.append(tag if tag in allowed else "reviewer_uncertain")
        return self._dedupe_limit(checks, limit=3)

    def _normalize_constraints(self, raw: Any) -> list[str]:
        if not isinstance(raw, list):
            raw = [raw] if raw else []
        return self._dedupe_limit([self._normalize_tag(item) for item in raw], limit=4)

    def _normalize_retry_hint(self, raw: Any) -> str:
        hint = " ".join(str(raw or "").split())
        if not hint or "```" in hint or "\n" in hint:
            return ""
        words = hint.split()
        if len(words) > 30:
            hint = " ".join(words[:30]).rstrip(" ,;:")
        return hint

    def _retry_hint_for_repair(self, *, failed_checks: list[str], evidence: dict[str, Any], llm_retry_hint: str = "") -> str:
        if llm_retry_hint:
            return llm_retry_hint
        hints = {
            "syntax_error": "Return a complete valid Python snippet while preserving the original signature.",
            "empty_repair": "Return the full repaired snippet, not an empty response.",
            "no_effective_change": "Make one concrete local edit that directly addresses the SATD.",
            "comment_only_without_satd_support": "Change executable code unless the SATD is explicitly about comments or documentation.",
            "anchor_not_modified": "Modify the SATD anchor region or the same local logic.",
            "unrelated_change": "Avoid unrelated rewrites; change only the SATD-relevant local block.",
            "unsupported_control_flow_change": "Avoid new return, raise, branch, or helper paths unless directly requested.",
            "over_expanded_change": "Make a smaller local edit with no speculative helpers or broad rewrites.",
            "not_addressing_satd": "Make one direct edit matching the SATD instruction.",
            "semantic_drift_risk": "Preserve existing behavior while making the SATD-local edit.",
            "context_conflict": "Do not contradict retrieved method context.",
            "reviewer_uncertain": "Make the smallest local edit that clearly addresses the SATD.",
        }
        return hints.get(failed_checks[0] if failed_checks else "reviewer_uncertain", hints["reviewer_uncertain"])

    def _normalize_tag(self, value: Any) -> str:
        tag = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
        aliases = {
            "not_addressing_request": "not_addressing_satd",
            "semantic_risk": "semantic_drift_risk",
            "scope_risk": "over_scoped_change",
        }
        return aliases.get(tag, tag)

    def _split_failed_check_item(self, value: Any) -> list[str]:
        text = str(value or "").strip()
        if not text:
            return []
        parts = re.split(r"\s*(?:\||,|;|/|\band\b)\s*", text, flags=re.IGNORECASE)
        return [part for part in parts if part]

    def _constraints_for_checks(self, failed_checks: list[str]) -> list[str]:
        constraints: list[str] = []
        for check in failed_checks:
            constraints.extend(self.CHECK_CONSTRAINTS.get(check, []))
        return self._dedupe_limit(constraints, limit=4)

    def _failure_anchor(self, failed_checks: list[str]) -> str:
        return failed_checks[0] if failed_checks else ""

    def _dedupe_limit(self, items: list[str], limit: int) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            cleaned = str(item or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            result.append(cleaned)
            if len(result) >= limit:
                break
        return result

    def _truncate(self, text: str, limit: int) -> str:
        compact = str(text or "").strip()
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."
