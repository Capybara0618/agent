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
    def __init__(self, client: OpenAICompatClient) -> None:
        self.client = client

    def run(self, state: GraphState) -> ReviewResult:
        repair = state["latest_repair"]
        assert repair is not None

        validity_issue = self._validity_issue(state, repair)
        if validity_issue:
            return self._invalid_result(repair, validity_issue)

        system_prompt, user_prompt = self._build_review_prompts(state, repair)
        payload = self.client.generate_json(
            system_prompt,
            user_prompt,
            request_label=f"review:task_{state['task_id']}:round_{repair.round_id}",
        )
        return self._coerce_review(payload, state, repair)

    def _build_review_prompts(self, state: GraphState, repair: RepairAttempt) -> tuple[str, str]:
        evidence_profile = self._diff_evidence_profile(state, repair)
        system_prompt = (
            "You are the reviewer agent in a SATD repair workflow.\n"
            "Judge whether the candidate repair should remain a viable SATD repair.\n"
            "Act as an evidence-grounded viability gate, not as a precision-only filter or style reviewer.\n"
            "Do not repair the code. Do not propose replacement code.\n"
            "Use only the SATD comment, original code, repaired code, and retrieved context.\n"
            "Return JSON only."
        )
        user_prompt = (
            "Review this SATD repair:\n\n"
            f"SATD comment:\n{state['satd_comment']}\n\n"
            f"Original code:\n```python\n{state['original_code']}\n```\n\n"
            f"Repaired code:\n```python\n{repair.repaired_code}\n```\n\n"
            f"Retrieved context:\n{self._repair_context_summary(state)}\n\n"
            f"Diff summary:\n{self._diff_summary(state, repair)}\n\n"
            f"Diff evidence profile:\n{json.dumps(evidence_profile, ensure_ascii=False, indent=2)}\n\n"
            "Use a viability gate, not a style review.\n"
            "Evaluate problem_alignment, behavioral_preservation, scope_control, and evidence_grounding.\n"
            "Approve repairs that directly resolve the SATD obligation while preserving unrelated behavior.\n"
            "Use \"revise\" for plausible but incomplete local repairs.\n"
            "Use \"reject\" only for concrete failure evidence: no SATD alignment, unrelated rewrite, unsupported behavior choice, broad scope, invalid code, or semantic drift.\n"
            "Treat the diff evidence profile as calibration evidence: structural expansion, unsupported control-flow/signature changes, or no target overlap should lower scope_control or evidence_grounding unless directly justified by the SATD and context.\n"
            "Do not reject solely because the changed region is a whole function; judge whether the meaningful changes are necessary and evidence-grounded.\n"
            "Do not require external proof when the candidate is conservative and supported by the SATD, original code, or retrieved context.\n\n"
            "Decision:\n"
            '- "approve": the repair directly satisfies the evidence-supported SATD obligation, keeps unrelated behavior stable, and is locally scoped.\n'
            '- "revise": the repair is aimed at the right obligation but is incomplete, too broad, weakly grounded, or likely fixable with a smaller evidence-supported attempt.\n'
            '- "reject": the repair does not address the obligation, relies on unsupported behavior choices, changes unrelated behavior, or is invalid as a repair.\n\n'
            "Return exactly:\n"
            "{\n"
            '  "approved": true,\n'
            '  "decision": "approve" | "revise" | "reject",\n'
            '  "confidence": 0.0,\n'
            '  "problem_alignment": "high" | "partial" | "low",\n'
            '  "behavioral_preservation": "high" | "partial" | "low",\n'
            '  "scope_control": "high" | "partial" | "low",\n'
            '  "evidence_grounding": "high" | "partial" | "low",\n'
            '  "issues": ["short issue"],\n'
            '  "repair_constraints": ["short constraint for retry"],\n'
            '  "retry_hint": "one short sentence",\n'
            '  "rationale": "one short sentence"\n'
            "}\n"
        )
        return system_prompt, user_prompt

    def _coerce_review(self, payload: dict[str, Any], state: GraphState, repair: RepairAttempt) -> ReviewResult:
        evidence_profile = self._diff_evidence_profile(state, repair)
        decision = self._normalize_decision(payload.get("decision"), payload.get("approved"))
        problem_alignment = self._normalize_level(payload.get("problem_alignment"))
        behavioral_preservation = self._normalize_level(payload.get("behavioral_preservation"))
        minimality = self._normalize_level(payload.get("scope_control") or payload.get("minimality"))
        evidence_grounding = self._normalize_level(payload.get("evidence_grounding"))
        problem_alignment, behavioral_preservation, minimality, evidence_grounding = self._calibrate_review_levels(
            problem_alignment=problem_alignment,
            behavioral_preservation=behavioral_preservation,
            scope_control=minimality,
            evidence_grounding=evidence_grounding,
            evidence_profile=evidence_profile,
        )
        problem_score = self._level_score(problem_alignment)
        preservation_score = self._level_score(behavioral_preservation)
        minimality_score = self._level_score(minimality)
        grounding_score = self._level_score(evidence_grounding)
        score = round((problem_score + preservation_score + minimality_score + grounding_score) / 4, 4)
        confidence = self._clamp_float(payload.get("confidence"), score)
        approved = self._calibrated_approval(
            decision=decision,
            problem_alignment=problem_alignment,
            behavioral_preservation=behavioral_preservation,
            scope_control=minimality,
            evidence_grounding=evidence_grounding,
            score=score,
            evidence_profile=evidence_profile,
        )
        issues = [] if approved else self._normalize_string_list(payload.get("issues"), default=["reviewer_uncertain"])
        repair_constraints = [] if approved else self._normalize_string_list(payload.get("repair_constraints"), default=["make_smallest_evidence_grounded_edit"])
        retry_hint = "" if approved else self._one_line(payload.get("retry_hint"))
        rationale = self._one_line(payload.get("rationale")) or ("approved" if approved else "Reviewer requested revision.")
        reject_type = None if approved else decision
        return ReviewResult(
            round_id=repair.round_id,
            approved=approved,
            issues=issues,
            candidate_mode=getattr(repair, "candidate_mode", "single"),
            review_score=confidence if approved else score,
            problem_alignment=problem_score,
            minimality=minimality_score,
            semantic_preservation=preservation_score,
            internal_consistency=grounding_score,
            revision_advice="" if approved else "constraints:" + ",".join(repair_constraints),
            reject_type=reject_type,
            rationale=rationale,
            failed_checks=issues,
            repair_constraints=repair_constraints,
            failure_anchor=reject_type or "",
            retry_hint=retry_hint,
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

    def _invalid_result(self, repair: RepairAttempt, issue: str) -> ReviewResult:
        constraints = {
            "empty_repair": "return_non_empty_repaired_code",
            "syntax_error": "return_valid_python_code",
            "no_effective_change": "make_evidence_grounded_code_change",
        }
        hints = {
            "empty_repair": "Return a non-empty repaired code snippet.",
            "syntax_error": "Return syntactically valid Python code.",
            "no_effective_change": "Make a concrete code change that addresses the SATD evidence.",
        }
        return ReviewResult(
            round_id=repair.round_id,
            approved=False,
            issues=[issue],
            candidate_mode=getattr(repair, "candidate_mode", "single"),
            review_score=0.0,
            problem_alignment=0.0,
            minimality=0.0,
            semantic_preservation=0.0,
            internal_consistency=0.0,
            revision_advice="constraints:" + constraints[issue],
            reject_type=issue,
            rationale=f"Invalid repair output: {issue}.",
            failed_checks=[issue],
            repair_constraints=[constraints[issue]],
            failure_anchor=issue,
            retry_hint=hints[issue],
        )

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
        return "\n".join(
            [
                f"original_line_count: {original_lines}",
                f"repaired_line_count: {repaired_lines}",
                f"changed_line_estimate: {changed}",
                f"added_lines: {added}",
                f"deleted_lines: {deleted}",
                f"candidate_changed_scope: {getattr(repair, 'changed_scope', '')}",
            ]
        )

    def _diff_evidence_profile(self, state: GraphState, repair: RepairAttempt) -> dict[str, Any]:
        original_code = state["original_code"] or ""
        repaired_code = repair.repaired_code or ""
        original_lines = max(1, len(original_code.splitlines()))
        repaired_lines = len(repaired_code.splitlines())
        changed_line_count = self._changed_line_count(original_code, repaired_code)
        added_lines, deleted_lines = self._line_delta_counts(original_code, repaired_code)
        changed_ratio = changed_line_count / original_lines
        original_profile = self._structure_profile(original_code)
        repaired_profile = self._structure_profile(repaired_code)
        target_terms = self._target_terms_from_text(state.get("satd_comment") or "")
        changed_text = self._changed_text(original_code, repaired_code)
        comment_or_docstring_only_change = (
            preprocess_python_code(original_code) == preprocess_python_code(repaired_code)
            and original_code.strip() != repaired_code.strip()
        )
        target_overlap_terms = [
            term for term in target_terms if self._term_in_text(term, changed_text)
        ][:8]
        structural_expansion_risk = self._structural_expansion_risk(
            added_lines=added_lines,
            line_count_delta=repaired_lines - original_lines,
            original_lines=original_lines,
            original_profile=original_profile,
            repaired_profile=repaired_profile,
        )
        if comment_or_docstring_only_change:
            structural_expansion_risk = False
        return {
            "syntax_valid": self._syntax_valid(repaired_code),
            "has_effective_change": not self._same_code(original_code, repaired_code),
            "original_line_count": original_lines,
            "repaired_line_count": repaired_lines,
            "changed_line_count": changed_line_count,
            "changed_line_ratio": round(changed_ratio, 3),
            "added_line_count": added_lines,
            "deleted_line_count": deleted_lines,
            "line_count_delta": repaired_lines - original_lines,
            "candidate_changed_scope": getattr(repair, "changed_scope", ""),
            "comment_or_docstring_only_change": comment_or_docstring_only_change,
            "signature_changed": self._signature_changed(original_code, repaired_code),
            "definitions_delta": repaired_profile["definitions"] - original_profile["definitions"],
            "imports_delta": repaired_profile["imports"] - original_profile["imports"],
            "branches_delta": repaired_profile["branches"] - original_profile["branches"],
            "returns_delta": repaired_profile["returns"] - original_profile["returns"],
            "raises_delta": repaired_profile["raises"] - original_profile["raises"],
            "satd_target_terms": target_terms[:8],
            "changed_text_target_overlap_terms": target_overlap_terms,
            "changed_text_has_target_overlap": bool(target_overlap_terms),
            "broad_change_risk": self._broad_change_risk(
                changed_line_count=changed_line_count,
                changed_ratio=changed_ratio,
                added_lines=added_lines,
                original_lines=original_lines,
            ),
            "structural_expansion_risk": structural_expansion_risk,
            "unsupported_control_flow_delta": bool(
                repaired_profile["branches"] > original_profile["branches"]
                or repaired_profile["returns"] > original_profile["returns"]
                or repaired_profile["raises"] > original_profile["raises"]
            ),
            "large_scope_without_target_overlap": bool(
                getattr(repair, "changed_scope", "") in {"class", "file"}
                and not target_overlap_terms
                and changed_line_count >= 12
                and added_lines >= deleted_lines
            ),
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

    def _changed_text(self, original_code: str, repaired_code: str) -> str:
        diff = difflib.ndiff((original_code or "").splitlines(), (repaired_code or "").splitlines())
        return "\n".join(line for line in diff if line.startswith(("+ ", "- ")))

    def _structure_profile(self, code: str) -> dict[str, int]:
        source = textwrap.dedent(code or "")
        try:
            tree = ast.parse(source)
            return {
                "definitions": sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree)),
                "imports": sum(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)),
                "branches": sum(isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)) for node in ast.walk(tree)),
                "returns": sum(isinstance(node, ast.Return) for node in ast.walk(tree)),
                "raises": sum(isinstance(node, ast.Raise) for node in ast.walk(tree)),
            }
        except SyntaxError:
            return {
                "definitions": len(re.findall(r"^\s*(?:async\s+def|def|class)\s+", source, flags=re.MULTILINE)),
                "imports": len(re.findall(r"^\s*(?:import|from)\s+", source, flags=re.MULTILINE)),
                "branches": len(re.findall(r"\b(if|for|while|try|with)\b", source)),
                "returns": len(re.findall(r"\breturn\b", source)),
                "raises": len(re.findall(r"\braise\b", source)),
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

    def _broad_change_risk(
        self,
        *,
        changed_line_count: int,
        changed_ratio: float,
        added_lines: int,
        original_lines: int,
    ) -> bool:
        return changed_line_count >= 8 and changed_ratio >= 0.60 or (added_lines >= 8 and added_lines / max(1, original_lines) >= 0.50)

    def _structural_expansion_risk(
        self,
        *,
        added_lines: int,
        line_count_delta: int,
        original_lines: int,
        original_profile: dict[str, int],
        repaired_profile: dict[str, int],
    ) -> bool:
        if line_count_delta <= 0:
            return False
        structure_expanded = bool(
            repaired_profile["definitions"] > original_profile["definitions"]
            or repaired_profile["imports"] > original_profile["imports"]
            or repaired_profile["branches"] - original_profile["branches"] >= 2
            or (repaired_profile["returns"] - original_profile["returns"] + repaired_profile["raises"] - original_profile["raises"]) >= 2
        )
        return bool(
            added_lines >= 10
            or line_count_delta >= 8
            or (added_lines >= 6 and added_lines / max(1, original_lines) >= 0.50)
            or (added_lines >= 4 and structure_expanded)
        )

    def _target_terms_from_text(self, text: str) -> list[str]:
        terms: list[str] = []
        patterns = [
            r"`([^`]{2,80})`",
            r"['\"]([A-Za-z_][A-Za-z0-9_.-]{2,})['\"]",
            r"\b[A-Za-z_][A-Za-z0-9_]*(?:[._][A-Za-z_][A-Za-z0-9_]*)+\b",
            r"\b[a-z][a-z0-9]*_[A-Za-z0-9_]+\b",
            r"\b[A-Z][A-Za-z0-9_]{2,}\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text or ""):
                value = next((group for group in match.groups() if group), match.group(0))
                normalized = self._normalize_target_term(value)
                if normalized and normalized not in self._generic_target_terms() and normalized not in terms:
                    terms.append(normalized)
                if len(terms) >= 12:
                    return terms
        return terms

    def _term_in_text(self, term: str, text: str) -> bool:
        lowered = str(text or "").lower()
        variants = {
            term,
            term.replace(" ", "_"),
            term.replace("_", " "),
            term.replace("-", "_"),
            term.split(".")[-1],
        }
        for variant in variants:
            cleaned = self._normalize_target_term(variant)
            if len(cleaned) < 3:
                continue
            pattern = r"(?<![A-Za-z0-9_])" + re.escape(cleaned.lower()) + r"(?![A-Za-z0-9_])"
            if re.search(pattern, lowered):
                return True
        return False

    def _normalize_target_term(self, value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip().strip("`'\"")).lower()
        text = re.sub(r"[^a-z0-9_.\-\s]", "", text)
        return text.strip(" ._-")

    def _generic_target_terms(self) -> set[str]:
        return {
            "todo",
            "fixme",
            "xxx",
            "this",
            "that",
            "these",
            "those",
            "function",
            "method",
            "class",
            "code",
            "value",
            "values",
            "data",
            "api",
            "implementation",
        }

    def _normalize_decision(self, raw_decision: Any, raw_approved: Any) -> str:
        decision = str(raw_decision or "").strip().lower()
        if decision in {"approve", "approved", "accept", "accepted", "pass"}:
            return "approve"
        if decision in {"reject", "rejected", "fail", "drop"}:
            return "reject"
        if decision in {"revise", "revision", "retry", "uncertain"}:
            return "revise"
        if isinstance(raw_approved, bool):
            return "approve" if raw_approved else "revise"
        return "revise"

    def _normalize_level(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"high", "partial", "low"}:
            return text
        if text in {"medium", "med"}:
            return "partial"
        return "partial"

    def _level_score(self, value: str) -> float:
        return {"high": 0.85, "partial": 0.55, "low": 0.05}.get(value, 0.55)

    def _calibrate_review_levels(
        self,
        *,
        problem_alignment: str,
        behavioral_preservation: str,
        scope_control: str,
        evidence_grounding: str,
        evidence_profile: dict[str, Any],
    ) -> tuple[str, str, str, str]:
        if evidence_profile.get("broad_change_risk") or evidence_profile.get("structural_expansion_risk"):
            scope_control = self._min_level(scope_control, "partial")
        if evidence_profile.get("structural_expansion_risk"):
            scope_control = "low"
            evidence_grounding = self._min_level(evidence_grounding, "partial")
        if evidence_profile.get("large_scope_without_target_overlap"):
            scope_control = "low"
            evidence_grounding = self._min_level(evidence_grounding, "partial")
        if evidence_profile.get("signature_changed") and evidence_grounding != "high":
            behavioral_preservation = self._min_level(behavioral_preservation, "partial")
        if evidence_profile.get("signature_changed") and not evidence_profile.get("changed_text_has_target_overlap"):
            behavioral_preservation = "low"
        if evidence_profile.get("unsupported_control_flow_delta") and evidence_grounding != "high":
            behavioral_preservation = self._min_level(behavioral_preservation, "partial")
            scope_control = self._min_level(scope_control, "partial")
        if evidence_profile.get("unsupported_control_flow_delta") and not evidence_profile.get("changed_text_has_target_overlap"):
            behavioral_preservation = "low"
        if evidence_profile.get("satd_target_terms") and not evidence_profile.get("changed_text_has_target_overlap"):
            evidence_grounding = self._min_level(evidence_grounding, "partial")
        if evidence_profile.get("comment_or_docstring_only_change") and not evidence_profile.get("changed_text_has_target_overlap"):
            problem_alignment = self._min_level(problem_alignment, "partial")
        return problem_alignment, behavioral_preservation, scope_control, evidence_grounding

    def _min_level(self, current: str, cap: str) -> str:
        order = {"low": 0, "partial": 1, "high": 2}
        reverse = {0: "low", 1: "partial", 2: "high"}
        return reverse[min(order.get(current, 1), order.get(cap, 1))]

    def _calibrated_approval(
        self,
        *,
        decision: str,
        problem_alignment: str,
        behavioral_preservation: str,
        scope_control: str,
        evidence_grounding: str,
        score: float,
        evidence_profile: dict[str, Any],
    ) -> bool:
        critical = [problem_alignment, behavioral_preservation, scope_control, evidence_grounding]
        low_count = sum(1 for item in critical if item == "low")
        high_count = sum(1 for item in critical if item == "high")

        if problem_alignment == "low" or behavioral_preservation == "low":
            return False
        if scope_control == "low" and evidence_grounding != "high":
            return False
        if low_count >= 2:
            return False
        if evidence_profile.get("broad_change_risk") and evidence_grounding != "high":
            return False
        if evidence_profile.get("structural_expansion_risk"):
            return False
        if evidence_profile.get("large_scope_without_target_overlap"):
            return False
        if evidence_grounding == "low" and not evidence_profile.get("changed_text_has_target_overlap"):
            return False
        if decision == "approve":
            return high_count >= 1 and evidence_grounding != "low" and scope_control != "low" and score >= 0.66
        if decision == "revise":
            return low_count == 0 and high_count >= 3 and score >= 0.74
        return False

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

    def _clamp_float(self, value: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    def _truncate(self, text: str, limit: int) -> str:
        compact = str(text or "").strip()
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."
