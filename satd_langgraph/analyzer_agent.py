from __future__ import annotations

import re
from typing import Any

from .openai_client import OpenAICompatClient
from .schema import AnalysisResult, EditConstraint, GraphState, MethodInquiryResult, RetrievedMethodContext


class OpenAIAnalyzer:
    def __init__(self, client: OpenAICompatClient) -> None:
        self.client = client

    def run(self, state: GraphState, method_context_block: str = "[none]") -> AnalysisResult:
        method_inquiry = state.get("method_inquiry")
        required_methods = method_inquiry.required_methods if isinstance(method_inquiry, MethodInquiryResult) else []
        missing_method_names = [str(item) for item in (state.get("missing_method_names") or []) if str(item).strip()]
        retrieved_method_count = len(state.get("retrieved_method_contexts") or [])
        missing_method_count = len(missing_method_names)
        edit_constraints = [
            item for item in (state.get("edit_constraints") or []) if isinstance(item, EditConstraint)
        ]
        repair_evidence_mode = "strong" if retrieved_method_count else "weak"
        system_prompt = (
            "You are the analyzer agent in a SATD repair workflow. "
            "Return valid JSON only with keys: decision, confidence, operation_concrete, localizable, "
            "local_scope, end_state_clear, context_sufficiency, method_context_used, drop_reason, "
            "comment_evidence, code_evidence, notes. "
            "Your job is to decide whether the SATD should enter the fixer agent. "
            "Do not repair the code. Do not output repaired_code. Do not propose replacement code."
        )
        user_prompt = (
            "Analyze whether the SATD is suitable for automatic local repair.\n\n"
            f"### SATD comment:\n{state['satd_comment']}\n\n"
            f"### Code:\n```python\n{state['original_code']}\n```\n\n"
            f"### Required methods:\n{self._format_list(required_methods)}\n\n"
            f"### Supporting evidence:\n{method_context_block or '[none]'}\n\n"
            f"### Missing methods:\n{self._format_list(missing_method_names)}\n\n"
            f"### Edit constraints:\n{self._format_analyzer_constraints(edit_constraints)}\n\n"
            "### Context quality:\n"
            f"retrieved_method_count: {retrieved_method_count}\n"
            f"missing_method_count: {missing_method_count}\n"
            f"repair_evidence_mode: {repair_evidence_mode}\n\n"
            "### Hard rules:\n"
            "- Return decision=\"pass\" when the SATD has a clear local repair target and the supporting evidence makes a small automatic repair plausible.\n"
            "- Return decision=\"uncertain\" when the SATD may be repairable but the exact edit, target, or supporting method context is incomplete.\n"
            "- Return decision=\"drop\" only when the SATD is clearly unsuitable for automatic local repair.\n"
            "- Prefer uncertain over drop when evidence is mixed.\n"
            "- Do not drop only because a method is missing.\n"
            "- Do not drop only because the repair looks difficult.\n"
            "- Drop broad, open-ended, architectural, migration, investigation, redesign, or product-decision tasks.\n"
            "- Use supporting evidence only to judge target existence, locality, context sufficiency, and repairability.\n"
            "- Do not let retrieved method context override an unclear SATD comment.\n"
            "- If the SATD asks to decide, investigate, figure out, redesign, rewrite, refactor, implement an unspecified behavior, or optimize broadly, return decision=\"drop\" unless the code and evidence expose a specific small edit.\n"
            "- Return decision=\"drop\" when both the requested operation and the desired end state are too unclear to write a local patch.\n"
            "- Return decision=\"drop\" when supporting evidence only proves that related methods exist but does not identify what should change.\n"
            "- For context-required SATD, pass requires both a concrete operation and a local target; method context alone is insufficient.\n"
            "- Do not generate repaired code.\n"
            "- Do not invent repository facts not present in the code or supporting evidence.\n\n"
            "Return JSON with exactly:\n"
            "{\n"
            '  "decision": "pass" | "uncertain" | "drop",\n'
            '  "confidence": 0.0,\n'
            '  "operation_concrete": "high" | "partial" | "low",\n'
            '  "localizable": "high" | "partial" | "low",\n'
            '  "local_scope": "high" | "partial" | "low",\n'
            '  "end_state_clear": "high" | "partial" | "low",\n'
            '  "context_sufficiency": "high" | "partial" | "low",\n'
            '  "method_context_used": true,\n'
            '  "drop_reason": "",\n'
            '  "comment_evidence": "short phrase from the SATD comment",\n'
            '  "code_evidence": "short phrase from code or supporting evidence",\n'
            '  "notes": "one short sentence"\n'
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
            code_evidence="rule-based missing input",
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
        operation_concrete = self._normalize_level(payload.get("operation_concrete"))
        localizable = self._normalize_level(payload.get("localizable"))
        local_scope = self._normalize_level(payload.get("local_scope"))
        end_state_clear = self._normalize_level(payload.get("end_state_clear"))
        context_sufficiency = self._normalize_level(payload.get("context_sufficiency"))
        operation_concrete, localizable, local_scope, end_state_clear = self._apply_existing_target_floors(
            state=state,
            operation_concrete=operation_concrete,
            localizable=localizable,
            local_scope=local_scope,
            end_state_clear=end_state_clear,
        )
        notes = self._one_line(
            payload.get("notes")
            or payload.get("drop_reason")
            or payload.get("reason")
            or payload.get("evidence_summary")
            or ""
        )
        comment_evidence = self._one_line(payload.get("comment_evidence"))
        code_evidence = self._one_line(payload.get("code_evidence"))
        decision = self._decision_from_checks(
            requested_decision=requested_decision,
            operation_concrete=operation_concrete,
            localizable=localizable,
            local_scope=local_scope,
            end_state_clear=end_state_clear,
        )
        strict_drop_reason = self._strict_drop_reason(
            state=state,
            operation_concrete=operation_concrete,
            localizable=localizable,
            local_scope=local_scope,
            end_state_clear=end_state_clear,
        )
        if strict_drop_reason:
            decision = "drop"
            confidence = max(confidence, 0.70)
            if strict_drop_reason == "unclear_operation_and_end_state":
                operation_concrete = "low"
                end_state_clear = "low"
            elif strict_drop_reason == "target_not_localizable":
                localizable = "low"
            elif strict_drop_reason == "not_local_repair":
                local_scope = "low"
            notes = self._append_note(notes, f"Analyzer strict drop: {strict_drop_reason}.")
        elif requested_decision == "drop" and self._should_keep_as_uncertain(
            state=state,
            operation_concrete=operation_concrete,
            localizable=localizable,
            local_scope=local_scope,
            end_state_clear=end_state_clear,
        ):
            decision = "uncertain"
            notes = self._append_note(notes, "Kept as uncertain because a local target signal exists.")
        return self._build_result(
            decision=decision,
            confidence=confidence,
            operation_concrete=operation_concrete,
            localizable=localizable,
            local_scope=local_scope,
            end_state_clear=end_state_clear,
            context_sufficiency=context_sufficiency,
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

    def _level_score(self, value: str | None) -> float:
        return {"high": 0.8, "partial": 0.499, "low": 0.04}.get(value or "", 0.499)

    def _decision_from_checks(
        self,
        *,
        requested_decision: str,
        operation_concrete: str | None,
        localizable: str | None,
        local_scope: str | None,
        end_state_clear: str | None,
    ) -> str:
        checks = [operation_concrete, localizable, local_scope, end_state_clear]
        known = [item for item in checks if item is not None]
        low_count = sum(1 for item in known if item == "low")
        high_count = sum(1 for item in known if item == "high")
        if requested_decision == "drop" and low_count >= 2 and high_count == 0:
            return "drop"
        if localizable == "high" and local_scope == "high" and (
            operation_concrete == "high" or end_state_clear == "high"
        ):
            return "pass"
        if operation_concrete == "low" and localizable == "low" and end_state_clear == "low":
            return "drop"
        if localizable == "low" and local_scope == "low" and end_state_clear == "low":
            return "drop"
        if low_count >= 3 and high_count == 0:
            return "drop"
        return "uncertain"

    def _apply_existing_target_floors(
        self,
        *,
        state: GraphState,
        operation_concrete: str | None,
        localizable: str | None,
        local_scope: str | None,
        end_state_clear: str | None,
    ) -> tuple[str | None, str | None, str | None, str | None]:
        comment = str(state.get("satd_comment") or "")
        code = str(state.get("original_code") or "")
        if not self._has_existing_target_hint(comment, code):
            return operation_concrete, localizable, local_scope, end_state_clear
        if self._looks_open_ended_task(comment):
            return operation_concrete, localizable, local_scope, end_state_clear
        if localizable == "low":
            localizable = "partial"
        if operation_concrete == "low":
            operation_concrete = "partial"
        if end_state_clear == "low":
            end_state_clear = "partial"
        return operation_concrete, localizable, local_scope, end_state_clear

    def _strict_drop_reason(
        self,
        *,
        state: GraphState,
        operation_concrete: str | None,
        localizable: str | None,
        local_scope: str | None,
        end_state_clear: str | None,
    ) -> str:
        comment = str(state.get("satd_comment") or "")
        code = str(state.get("original_code") or "")
        open_ended_comment = self._looks_open_ended_task(comment)
        if self._should_keep_as_uncertain(
            state=state,
            operation_concrete=operation_concrete,
            localizable=localizable,
            local_scope=local_scope,
            end_state_clear=end_state_clear,
        ):
            return ""
        if operation_concrete == "low" and end_state_clear == "low":
            return "unclear_operation_and_end_state"
        if localizable == "low" and (operation_concrete == "low" or end_state_clear == "low" or open_ended_comment):
            return "target_not_localizable"
        if local_scope == "low" and (operation_concrete == "low" or end_state_clear == "low" or open_ended_comment):
            return "not_local_repair"
        if open_ended_comment and not self._has_strong_local_edit_signal(comment, code):
            return "open_ended_without_specific_local_edit"
        return ""

    def _has_local_target_signal(self, state: GraphState) -> bool:
        comment = str(state.get("satd_comment") or "")
        code = str(state.get("original_code") or "")
        if not self._grounded_override_action_allowed(comment):
            return False
        if self._has_grounded_local_target(state, comment, code):
            return True
        tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", comment)
            if token.lower() not in {"todo", "fixme", "xxx", "this", "that", "with", "from", "after", "before"}
        }
        lowered_code = code.lower()
        return any(token in lowered_code for token in tokens)

    def _should_keep_as_uncertain(
        self,
        *,
        state: GraphState,
        operation_concrete: str | None,
        localizable: str | None,
        local_scope: str | None,
        end_state_clear: str | None,
    ) -> bool:
        comment = str(state.get("satd_comment") or "")
        code = str(state.get("original_code") or "")
        if not self._grounded_override_action_allowed(comment):
            return False
        if not self._has_grounded_local_target(state, comment, code):
            return False
        if local_scope == "low" and localizable == "low" and operation_concrete == "low" and end_state_clear == "low":
            return True
        if localizable in {"high", "partial"} and local_scope in {"high", "partial"}:
            return True
        return operation_concrete == "low" and end_state_clear == "low"

    def _has_satd_anchor_marker(self, code: str) -> bool:
        return bool(re.search(r"\b(todo|fixme|xxx|hack|workaround|temporary|temp|obsolete|deprecated)\b", code or "", flags=re.IGNORECASE))

    def _has_repair_action_hint(self, state: GraphState) -> bool:
        comment = str(state.get("satd_comment") or "")
        return bool(
            re.search(
                r"\b(remove|delete|drop|disable|revert|cleanup|replace|rename|change|use|return|raise|annotat|document|default|fix)\b",
                comment,
                flags=re.IGNORECASE,
            )
        )

    def _grounded_override_action_allowed(self, comment: str) -> bool:
        lowered = (comment or "").lower()
        if not lowered.strip():
            return False
        broad_blockers = [
            r"\bdecide\b",
            r"\bshould\s+we\b",
            r"\bfigure out\b",
            r"\binvestigat",
            r"\blook into\b",
            r"\brewrite\b",
            r"\brefactor\b",
            r"\bredesign\b",
            r"\boptimi[sz]e\b",
            r"\bperformance\b",
            r"\barchitecture\b",
            r"\bglobal\b",
            r"\bclean up\b",
            r"\bimprove\b",
        ]
        if any(re.search(pattern, lowered) for pattern in broad_blockers):
            return False
        broad_action_patterns = [
            r"\bdeprecat",
            r"\bprevent\b",
            r"\bset\b.{0,40}\bdefault\b",
            r"\bdefault\b.{0,40}\bto\b",
            r"\bhonor\b",
            r"\breturn\b",
            r"\braise\b",
        ]
        if any(re.search(pattern, lowered) for pattern in broad_action_patterns):
            return True
        symbol_required_patterns = [
            r"\bimplement\b",
            r"\badd\b",
            r"\buncomment\b",
            r"\bhandle\b",
            r"\bparse\b",
            r"\bcheck\b",
            r"\bremove\b",
            r"\bdelete\b",
            r"\breplace\b",
            r"\brename\b",
        ]
        return (
            any(re.search(pattern, lowered) for pattern in symbol_required_patterns)
            and self._has_explicit_symbolic_target(comment)
        )

    def _has_explicit_symbolic_target(self, comment: str) -> bool:
        text = str(comment or "")
        if re.search(r"`[^`]{2,80}`|['\"][A-Za-z_][A-Za-z0-9_.-]{2,}['\"]", text):
            return True
        if re.search(r"\b[A-Za-z_][A-Za-z0-9_]*(?:[._][A-Za-z_][A-Za-z0-9_]*)+\b", text):
            return True
        if re.search(r"\b[a-z][a-z0-9]*_[a-zA-Z0-9_]+\b", text):
            return True
        ignored = self._generic_target_terms() | {
            "to",
            "todo",
            "fixme",
            "xxx",
            "implement",
            "uncomment",
            "handle",
            "check",
            "parse",
            "add",
            "prevent",
            "deprecate",
            "when",
            "once",
        }
        for match in re.finditer(r"\b[A-Z][a-z][A-Za-z0-9_]{2,}\b", text):
            token = match.group(0).lower()
            if token not in ignored:
                return True
        return False

    def _has_grounded_local_target(self, state: GraphState, comment: str, code: str) -> bool:
        terms = self._local_target_terms_from_comment(comment)
        if not terms:
            return False
        evidence = self._local_target_evidence_text(state, code)
        if not evidence.strip():
            return False
        lowered_evidence = evidence.lower()
        for term in terms:
            normalized = self._normalize_target_term(term)
            if not normalized or normalized in self._generic_target_terms():
                continue
            variants = {
                normalized,
                normalized.replace(" ", "_"),
                normalized.replace("_", " "),
                normalized.replace("-", "_"),
            }
            for variant in variants:
                if len(variant.strip("_ ")) < 3:
                    continue
                pattern = r"(?<![A-Za-z0-9_])" + re.escape(variant.lower()) + r"(?![A-Za-z0-9_])"
                if re.search(pattern, lowered_evidence):
                    return True
        return False

    def _local_target_terms_from_comment(self, comment: str) -> set[str]:
        text = str(comment or "")
        terms: set[str] = set()
        for match in re.finditer(r"`([^`]{2,80})`|['\"]([^'\"]{2,80})['\"]", text):
            terms.add(match.group(1) or match.group(2) or "")
        for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*(?:[._][A-Za-z_][A-Za-z0-9_]*)+\b", text):
            terms.add(match.group(0))
        for match in re.finditer(r"\b[A-Z][A-Za-z0-9_]{2,}\b", text):
            terms.add(match.group(0))
        for match in re.finditer(r"\b[a-z][a-z0-9]*_[a-zA-Z0-9_]+\b", text):
            terms.add(match.group(0))
        action_pattern = r"\b(?:implement|add|prevent|deprecat\w*|honor|handle|parse|uncomment|return|raise|check|remove|delete|replace|rename)\s+(?:the\s+|a\s+|an\s+)?([A-Za-z_][A-Za-z0-9_]{2,})\b"
        for match in re.finditer(action_pattern, text, flags=re.IGNORECASE):
            terms.add(match.group(1))
        return {
            normalized
            for normalized in (self._normalize_target_term(term) for term in terms)
            if normalized and normalized not in self._generic_target_terms()
        }

    def _local_target_evidence_text(self, state: GraphState, code: str) -> str:
        parts = [str(code or "")]
        method_inquiry = state.get("method_inquiry")
        if isinstance(method_inquiry, MethodInquiryResult):
            parts.extend(str(name) for name in method_inquiry.required_methods or [])
        for item in state.get("retrieved_method_contexts") or []:
            if isinstance(item, RetrievedMethodContext):
                parts.extend([item.method_name, item.signature, item.source, item.callsite_slice, item.evidence_slice])
            elif isinstance(item, dict):
                parts.extend(str(item.get(key) or "") for key in ("method_name", "signature", "source", "callsite_slice", "evidence_slice"))
        return "\n".join(part for part in parts if part)

    def _normalize_target_term(self, value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip().strip("`'\"")).lower()
        text = re.sub(r"[^a-z0-9_.\-\s]", "", text)
        return text.strip(" ._-")

    def _generic_target_terms(self) -> set[str]:
        return {
            "todo",
            "fixme",
            "xxx",
            "implement",
            "implementation",
            "add",
            "prevent",
            "check",
            "parse",
            "handle",
            "for",
            "into",
            "have",
            "this",
            "that",
            "these",
            "those",
            "it",
            "this function",
            "function",
            "method",
            "class",
            "code",
            "support",
            "default",
            "value",
            "values",
            "data",
            "api",
            "sdk",
        }

    def _has_existing_target_hint(self, comment: str, code: str) -> bool:
        return self._has_strong_local_edit_signal(comment, code) or self._has_grounded_local_target({}, comment, code)

    def _looks_open_ended_task(self, comment: str) -> bool:
        lowered = (comment or "").lower()
        return bool(
            re.search(
                r"\b(should we|would we|do we need|figure out|investigat|look into|rewrite|refactor|redesign|optimi[sz]e|performance|architecture|global|clean up|improve)\b",
                lowered,
            )
        )

    def _has_strong_local_edit_signal(self, comment: str, code: str) -> bool:
        lowered = (comment or "").lower()
        if not lowered.strip() or not (code or "").strip():
            return False
        strong_patterns = [
            r"\bremove\b",
            r"\bdelete\b",
            r"\breplace\b",
            r"\brename\b",
            r"\bswitch to\b",
            r"\buse .{1,80}\binstead\b",
            r"\binstead of\b",
            r"\bdeprecated\b.{0,80}\buse\b",
            r"\bannotat",
            r"\bdocument\b",
            r"\bdocstring\b",
            r"\bmissing doc\b",
            r"\bupdate description\b",
            r"\bchange (?:the )?(?:default|value|return|exception|error|message|type)\b",
            r"\badd (?:a |an |the )?missing (?:argument|parameter|annotation|doc|string|check|guard)\b",
            r"\bhandle (?:a |an |the )?(?:missing|none|null|empty|exception|error)\b",
            r"\braise (?:a |an |the )?(?:specific )?(?:exception|error)\b",
            r"\breturn (?:a |an |the )?(?:specific |default |empty |none|null|false|true)",
        ]
        return any(re.search(pattern, lowered) for pattern in strong_patterns)

    def _append_note(self, notes: str, addition: str) -> str:
        notes = self._one_line(notes)
        addition = self._one_line(addition)
        if not notes:
            return addition
        if addition in notes:
            return notes
        return self._one_line(f"{notes} {addition}")

    def _one_line(self, value: Any) -> str:
        return " ".join(str(value or "").split())[:240]

    def _clamp_float(self, value: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default
