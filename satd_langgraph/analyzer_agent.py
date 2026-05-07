from __future__ import annotations

import re
from typing import Any

from .openai_client import OpenAICompatClient
from .schema import AnalysisResult, EditConstraint, GraphState, MethodInquiryResult, RetrievedMethodContext


class OpenAIAnalyzer:
    def __init__(self, client: OpenAICompatClient) -> None:
        self.client = client

    def run(self, state: GraphState, method_context_block: str = "[none]", evidence_block: str = "[none]") -> AnalysisResult:
        method_inquiry = state.get("method_inquiry")
        required_methods = method_inquiry.required_methods if isinstance(method_inquiry, MethodInquiryResult) else []
        missing_method_names = [str(item) for item in (state.get("missing_method_names") or []) if str(item).strip()]
        target_grounding = self._target_grounding_profile(state)

        system_prompt = (
            "You are the analyzer agent in a SATD repair workflow.\n"
            "Decide whether this SATD item should enter the automatic fixer.\n"
            "Act as a calibrated evidence gate, not as a brainstorming assistant.\n"
            "Do not repair the code. Do not propose replacement code.\n"
            "Use only the SATD comment, code snippet, and evidence cards.\n"
            "Return JSON only."
        )
        user_prompt = (
            "Analyze this SATD:\n\n"
            f"SATD comment:\n{state['satd_comment']}\n\n"
            f"Code:\n```python\n{state['original_code']}\n```\n\n"
            f"Required methods:\n{self._format_list(required_methods)}\n\n"
            f"Evidence cards:\n{evidence_block or method_context_block or '[none]'}\n\n"
            f"Missing method context:\n{self._format_list(missing_method_names)}\n\n"
            f"Target grounding evidence:\n{self._format_target_grounding(target_grounding)}\n\n"
            "Use a precision-aware evidence gate.\n"
            "Assess target_clarity, locality, outcome_clarity, and context_sufficiency.\n"
            "Use \"uncertain\" as the default when evidence is mixed.\n"
            "Choose \"pass\" only when the target, local edit region, and acceptable outcome are all determined by the input.\n"
            "Choose \"uncertain\" when the target is local and a conservative attempt is plausible, but implementation details are incomplete.\n"
            "Choose \"drop\" when the workflow would need to invent the target behavior, product/design choice, replacement API, timing decision, "
            "future-version assumption, broad refactor, investigation result, performance strategy, or missing end state.\n"
            "Do not treat a TODO/FIXME marker or nearby editable code as sufficient by itself; the SATD and code must constrain what should change.\n"
            "Do not drop merely because retrieved method context is absent when the snippet itself gives a concrete local transformation.\n\n"
            "Decision:\n"
            '- "pass": the evidence determines a bounded repair target, local edit region, and acceptable outcome.\n'
            '- "uncertain": a local target and conservative acceptable outcome are inferable, but implementation support is incomplete.\n'
            '- "drop": the evidence does not determine the target or acceptable outcome, or the repair would require choosing policy, design, timing, external API behavior, or broad surrounding behavior not present in the input.\n\n'
            "Return exactly:\n"
            "{\n"
            '  "decision": "pass" | "uncertain" | "drop",\n'
            '  "confidence": 0.0,\n'
            '  "target_clarity": "high" | "partial" | "low",\n'
            '  "locality": "high" | "partial" | "low",\n'
            '  "outcome_clarity": "high" | "partial" | "low",\n'
            '  "context_sufficiency": "high" | "partial" | "low",\n'
            '  "repair_mode": "local_only" | "evidence_guided" | "no_context_fallback",\n'
            '  "repair_constraints": ["short constraint"],\n'
            '  "comment_evidence": "short phrase from SATD comment",\n'
            '  "code_evidence": "short phrase from code/context",\n'
            '  "reason": "one short sentence"\n'
            "}\n"
        )
        payload = self.client.generate_json(system_prompt, user_prompt, request_label=f"analyze:task_{state['task_id']}")
        return self._coerce_analysis(payload, state)

    def build_rule_drop_analysis(self, notes: str) -> AnalysisResult:
        return self._build_result(
            decision="drop",
            confidence=1.0,
            operation_concrete="low",
            localizable="low",
            local_scope="low",
            end_state_clear="low",
            context_sufficiency="low",
            notes=notes,
            comment_evidence="",
            code_evidence="invalid analyzer input",
            satd_type="context_required",
        )

    def _format_list(self, items: list[str]) -> str:
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        return ", ".join(cleaned) if cleaned else "[none]"

    def _format_analyzer_constraints(self, edit_constraints: list[EditConstraint]) -> str:
        lines = [
            "- Make the smallest plausible local edit.",
            "- Preserve the existing function/class signature unless the SATD explicitly asks for a signature-local fix.",
            "- Do not add new helpers, new control flow, or unrelated rewrites without evidence.",
        ]
        for item in edit_constraints[:3]:
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

    def _coerce_analysis(self, payload: dict[str, Any], state: GraphState) -> AnalysisResult:
        requested_decision = self._normalize_decision(payload.get("decision"))
        confidence = self._clamp_float(payload.get("confidence"), 0.0)
        target_clarity = self._normalize_level(payload.get("target_clarity") or payload.get("operation_concrete"))
        locality = self._normalize_level(payload.get("locality") or payload.get("local_scope") or payload.get("localizable"))
        outcome_clarity = self._normalize_level(payload.get("outcome_clarity") or payload.get("end_state_clear"))
        context_sufficiency = self._normalize_level(payload.get("context_sufficiency"))
        repair_mode = self._normalize_repair_mode(payload.get("repair_mode"), state)
        repair_constraints = self._coerce_constraints(payload.get("repair_constraints"))
        notes = self._one_line(
            payload.get("reason")
            or payload.get("notes")
            or payload.get("drop_reason")
            or payload.get("evidence_summary")
            or ""
        )
        comment_evidence = self._one_line(payload.get("comment_evidence"))
        code_evidence = self._one_line(payload.get("code_evidence"))
        decision = self._calibrated_decision(
            requested_decision=requested_decision,
            target_clarity=target_clarity,
            locality=locality,
            outcome_clarity=outcome_clarity,
            context_sufficiency=context_sufficiency,
            grounded_target=self._target_grounding_profile(state)["grounded_target"],
        )
        return self._build_result(
            decision=decision,
            confidence=confidence,
            operation_concrete=target_clarity,
            localizable=locality,
            local_scope=locality,
            end_state_clear=outcome_clarity,
            context_sufficiency=context_sufficiency,
            repair_mode=repair_mode,
            evidence_used=bool(state.get("evidence_cards")),
            repair_constraints=repair_constraints,
            notes=notes,
            comment_evidence=comment_evidence,
            code_evidence=code_evidence,
            satd_type=str(state.get("satd_route_type") or "context_required"),
        )

    def _build_result(
        self,
        *,
        decision: str,
        confidence: float,
        operation_concrete: str | None,
        localizable: str | None,
        local_scope: str | None,
        end_state_clear: str | None,
        context_sufficiency: str | None = None,
        repair_mode: str = "local_only",
        evidence_used: bool = False,
        repair_constraints: list[str] | None = None,
        notes: str,
        comment_evidence: str,
        code_evidence: str,
        satd_type: str,
    ) -> AnalysisResult:
        decision = "drop" if decision == "drop" else decision if decision == "uncertain" else "pass"
        repairable = decision != "drop"
        op_score = self._level_score(operation_concrete)
        loc_score = self._level_score(localizable)
        scope_score = self._level_score(local_scope)
        end_score = self._level_score(end_state_clear)
        ctx_score = self._level_score(context_sufficiency)
        analyze_score = round((op_score + loc_score + scope_score + end_score + ctx_score) / 5, 4)
        risk = "low" if analyze_score >= 0.72 else "medium" if analyze_score >= 0.45 else "high"
        reason = notes or ("Suitable for repair." if repairable else "Dropped by analyzer.")
        return AnalysisResult(
            decision=decision,
            repairable=repairable,
            confidence=self._clamp_float(confidence, analyze_score),
            reason=reason,
            repairability_score=analyze_score,
            intent_clarity=op_score,
            change_locality=scope_score,
            semantic_risk=round(1.0 - min(scope_score, end_score), 4),
            context_sufficiency=ctx_score,
            verifiability=end_score,
            analyze_score=analyze_score,
            satd_type=satd_type,
            evidence_summary=reason,
            risk_level=risk,
            context_score=ctx_score,
            clarity_score=round((op_score + end_score) / 2, 4),
            scope_radius="function",
            operation_concrete=operation_concrete,
            localizable=localizable,
            local_scope=local_scope,
            end_state_clear=end_state_clear,
            comment_evidence=comment_evidence,
            code_evidence=code_evidence,
            validation_signals=[decision],
            context_gaps=[] if ctx_score >= 0.5 else ["weak_method_context"],
            followup_context_requests=[],
            repair_strategy="Pass to fixer for smallest local edit." if repairable else "Do not attempt automatic repair.",
            github_evidence_strength="method_context_only",
            repair_mode=repair_mode,
            evidence_used=evidence_used,
            repair_constraints=list(repair_constraints or []),
            drop_reason=reason if not repairable else "",
            notes=reason,
        )

    def _normalize_decision(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"drop", "filter", "filtered", "reject", "no"}:
            return "drop"
        if text in {"uncertain", "maybe", "partial"}:
            return "uncertain"
        return "pass"

    def _normalize_level(self, value: Any) -> str | None:
        text = str(value or "").strip().lower()
        if text in {"high", "partial", "low"}:
            return text
        if text in {"medium", "med"}:
            return "partial"
        return None

    def _normalize_repair_mode(self, value: Any, state: GraphState) -> str:
        text = str(value or "").strip().lower()
        if text in {"local_only", "evidence_guided", "no_context_fallback"}:
            return text
        if state.get("evidence_cards"):
            return "evidence_guided"
        return "local_only"

    def _coerce_constraints(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        constraints: list[str] = []
        for item in value:
            text = self._one_line(item)
            if text and text not in constraints:
                constraints.append(text)
            if len(constraints) >= 4:
                break
        return constraints

    def _level_score(self, value: str | None) -> float:
        return {"high": 0.8, "partial": 0.499, "low": 0.04}.get(value or "", 0.499)

    def _calibrated_decision(
        self,
        *,
        requested_decision: str,
        target_clarity: str | None,
        locality: str | None,
        outcome_clarity: str | None,
        context_sufficiency: str | None,
        grounded_target: bool = False,
    ) -> str:
        critical = [target_clarity, locality, outcome_clarity]
        low_critical = sum(1 for item in critical if item == "low")
        high_critical = sum(1 for item in critical if item == "high")
        known_critical = [item for item in critical if item is not None]
        avg_score = (
            self._level_score(target_clarity)
            + self._level_score(locality)
            + self._level_score(outcome_clarity)
            + self._level_score(context_sufficiency)
        ) / 4

        if target_clarity == "low" and outcome_clarity == "low" and not grounded_target:
            return "drop"
        if low_critical >= 2 and not grounded_target:
            return "drop"
        if requested_decision == "drop":
            if grounded_target or (low_critical == 0 and high_critical >= 2):
                return "uncertain"
            return "drop"
        if locality == "low" and (target_clarity == "low" or outcome_clarity == "low") and not grounded_target:
            return "drop"
        if requested_decision == "pass":
            if low_critical == 1 or avg_score < 0.62:
                return "uncertain"
            return "pass"
        if requested_decision == "uncertain":
            if len(known_critical) == 3 and high_critical == 3 and context_sufficiency != "low":
                return "pass"
            return "uncertain"
        return requested_decision

    def _target_grounding_profile(self, state: GraphState) -> dict[str, Any]:
        comment = str(state.get("satd_comment") or "")
        code = str(state.get("original_code") or "")
        terms = self._target_terms_from_comment(comment)
        evidence = self._target_evidence_text(state, code)
        evidence_lower = evidence.lower()
        grounded_terms = [
            term
            for term in terms
            if self._term_in_evidence(term, evidence_lower)
        ]
        satd_anchor_present = self._satd_anchor_present(comment, code)
        return {
            "target_terms": terms[:8],
            "grounded_terms": grounded_terms[:8],
            "satd_anchor_present": satd_anchor_present,
            "grounded_target": bool(grounded_terms),
            "retrieved_context_count": len(state.get("retrieved_method_contexts") or []),
        }

    def _format_target_grounding(self, profile: dict[str, Any]) -> str:
        target_terms = self._format_list(list(profile.get("target_terms") or []))
        grounded_terms = self._format_list(list(profile.get("grounded_terms") or []))
        return "\n".join(
            [
                f"target_terms: {target_terms}",
                f"grounded_terms: {grounded_terms}",
                f"satd_anchor_present: {bool(profile.get('satd_anchor_present'))}",
                f"grounded_target: {bool(profile.get('grounded_target'))}",
                f"retrieved_context_count: {int(profile.get('retrieved_context_count') or 0)}",
            ]
        )

    def _target_terms_from_comment(self, comment: str) -> list[str]:
        text = str(comment or "")
        terms: list[str] = []
        patterns = [
            r"`([^`]{2,80})`",
            r"['\"]([A-Za-z_][A-Za-z0-9_.-]{2,})['\"]",
            r"\b[A-Za-z_][A-Za-z0-9_]*(?:[._][A-Za-z_][A-Za-z0-9_]*)+\b",
            r"\b[a-z][a-z0-9]*_[A-Za-z0-9_]+\b",
            r"\b[A-Z][A-Za-z0-9_]{2,}\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                value = next((group for group in match.groups() if group), match.group(0))
                normalized = self._normalize_target_term(value)
                if normalized and normalized not in self._generic_target_terms() and normalized not in terms:
                    terms.append(normalized)
                if len(terms) >= 12:
                    return terms
        return terms

    def _target_evidence_text(self, state: GraphState, code: str) -> str:
        parts = [str(code or "")]
        method_inquiry = state.get("method_inquiry")
        if isinstance(method_inquiry, MethodInquiryResult):
            parts.extend(str(name) for name in method_inquiry.required_methods or [])
        for item in state.get("retrieved_method_contexts") or []:
            if isinstance(item, RetrievedMethodContext):
                parts.extend([item.method_name, item.signature, item.source, item.callsite_slice, item.evidence_slice])
            elif isinstance(item, dict):
                parts.extend(
                    str(item.get(key) or "")
                    for key in ("method_name", "signature", "source", "callsite_slice", "evidence_slice")
                )
        return "\n".join(part for part in parts if part)

    def _term_in_evidence(self, term: str, evidence_lower: str) -> bool:
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
            if re.search(pattern, evidence_lower):
                return True
        return False

    def _satd_anchor_present(self, comment: str, code: str) -> bool:
        lowered_code = str(code or "").lower()
        normalized_comment = " ".join(str(comment or "").lower().split())
        if normalized_comment and normalized_comment in " ".join(lowered_code.split()):
            return True
        comment_tokens = {
            token
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(comment or "").lower())
            if token not in self._generic_target_terms()
        }
        if not comment_tokens:
            return False
        anchor_pattern = re.compile(
            r"\b(todo|fixme|xxx|hack|workaround|temporary|temp|obsolete|deprecated)\b",
            flags=re.IGNORECASE,
        )
        for line in str(code or "").splitlines():
            if not anchor_pattern.search(line):
                continue
            line_tokens = {
                token
                for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", line.lower())
                if token not in self._generic_target_terms()
            }
            if comment_tokens & line_tokens:
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

    def _one_line(self, value: Any) -> str:
        return " ".join(str(value or "").split())[:240]

    def _clamp_float(self, value: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default
