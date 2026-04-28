from __future__ import annotations

import ast
import builtins
import copy
import difflib
import json
import re
import os
import textwrap
import time
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

from .github_tools import GitHubToolbox
from .local_settings import OPENAI_API_KEY as LOCAL_OPENAI_API_KEY
from .local_settings import OPENAI_BASE_URL as LOCAL_OPENAI_BASE_URL
from .schema import (
    AnalysisResult,
    EditConstraint,
    GraphState,
    MethodInquiryResult,
    RepairAttempt,
    RetrievedMethodContext,
    ReviewResult,
    UncertaintyItem,
    preprocess_python_code,
)

MODEL_ALIASES = {"gpt-4o-mini-global": "gpt-4o-mini"}


class OpenAICompatClient:
    def __init__(self, model: str = "gpt-4o-mini", verbose: bool = False) -> None:
        api_key = os.environ.get("OPENAI_API_KEY") or LOCAL_OPENAI_API_KEY
        if api_key == "PASTE_YOUR_OPENAI_COMPAT_KEY_HERE":
            api_key = None
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        base_url = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or LOCAL_OPENAI_BASE_URL
            or None
        )
        if base_url == "https://your-openai-compatible-host/v1":
            base_url = None
        self.request_timeout = float(os.environ.get("OPENAI_TIMEOUT_SECONDS") or 60)
        self.max_attempts = max(1, int(os.environ.get("OPENAI_MAX_ATTEMPTS") or 3))
        sdk_max_retries = max(0, int(os.environ.get("OPENAI_SDK_MAX_RETRIES") or 0))
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.request_timeout,
            max_retries=sdk_max_retries,
        )
        self.requested_model = model
        self.model = self._normalize_model_name(model)
        self.verbose = bool(verbose)
        self.toolbox = GitHubToolbox()

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        request_label: str = "",
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        sanitized = False
        active_system_prompt = self._coerce_prompt_text(system_prompt)
        active_user_prompt = self._coerce_prompt_text(user_prompt)
        active_model = self.model
        label = request_label or "llm_request"
        for attempt in range(self.max_attempts):
            attempt_started = time.time()
            self._emit_log(
                f"[llm] start label={label} attempt={attempt + 1}/{self.max_attempts} model={active_model} timeout={self.request_timeout:.0f}s"
            )
            try:
                request_kwargs: dict[str, Any] = {
                    "model": active_model,
                    "messages": [
                        {"role": "system", "content": active_system_prompt},
                        {"role": "user", "content": active_user_prompt},
                    ],
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                }
                if max_tokens is not None:
                    request_kwargs["max_tokens"] = max_tokens
                response = self.client.chat.completions.create(**request_kwargs)
                content = response.choices[0].message.content or "{}"
                elapsed = time.time() - attempt_started
                self._emit_log(f"[llm] success label={label} attempt={attempt + 1}/{self.max_attempts} elapsed={elapsed:.2f}s")
                return json.loads(content)
            except Exception as exc:
                last_error = exc
                elapsed = time.time() - attempt_started
                self._emit_log(
                    f"[llm] error label={label} attempt={attempt + 1}/{self.max_attempts} "
                    f"elapsed={elapsed:.2f}s type={type(exc).__name__} message={self._short_error(exc)}"
                )
                fallback_model = self._fallback_model_for_error(exc, active_model)
                if fallback_model and fallback_model != active_model:
                    self._emit_log(f"[llm] fallback_model label={label} from={active_model} to={fallback_model}")
                    active_model = fallback_model
                    self.model = fallback_model
                    continue
                if self._is_content_filter_error(exc) and not sanitized:
                    self._emit_log(f"[llm] sanitize_prompts label={label}")
                    active_system_prompt, active_user_prompt = self._sanitize_prompts(system_prompt, user_prompt)
                    sanitized = True
                    continue
                if attempt < self.max_attempts - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("OpenAI-compatible request failed unexpectedly.")

    def _emit_log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _short_error(self, exc: Exception) -> str:
        message = " ".join(str(exc).split())
        if len(message) > 220:
            return message[:217] + "..."
        return message

    def _normalize_model_name(self, model: str) -> str:
        cleaned = (model or "gpt-4o-mini").strip()
        return MODEL_ALIASES.get(cleaned, cleaned)

    def _fallback_model_for_error(self, exc: Exception, active_model: str) -> str | None:
        message = str(exc).lower()
        normalized = self._normalize_model_name(active_model)
        if "unknown model" in message and normalized != active_model:
            return normalized
        if "unknown model" in message and active_model.endswith("-global"):
            return active_model[: -len("-global")]
        return None

    def _is_content_filter_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "content_filter" in message or "content management policy" in message

    def _sanitize_prompts(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:
        compact_system = self._coerce_prompt_text(system_prompt) + " Prefer concise evidence use and avoid unnecessary raw excerpts."
        compact_user = self._coerce_prompt_text(user_prompt)
        compact_user = re.sub(
            r"(Base \+ repair \+ review context:\n)[\s\S]*",
            r"\1[context compacted for safety; rely on metadata and direct task evidence]",
            compact_user,
        )
        compact_user = re.sub(
            r"(Unified GitHub context:\n)[\s\S]*",
            r"\1[context compacted for safety; rely on metadata and direct task evidence]",
            compact_user,
        )
        compact_user = re.sub(
            r"(Base \+ repair context:\n)[\s\S]*",
            r"\1[context compacted for safety; rely on metadata and direct task evidence]",
            compact_user,
        )
        compact_user = re.sub(
            r"(Base GitHub context:\n)[\s\S]*",
            r"\1[context compacted for safety; rely on metadata and direct task evidence]",
            compact_user,
        )
        compact_user = re.sub(
            r"(Original code block:\n)([\s\S]*?)(\n\n(?:Repaired code block:|Repair plan:|Base \+ repair context:|Base \+ repair \+ review context:))",
            lambda m: m.group(1) + m.group(2)[:1200] + m.group(3),
            compact_user,
        )
        compact_user = re.sub(
            r"(Current code snippet:\n)([\s\S]*?)(\n\nBase GitHub context:)",
            lambda m: m.group(1) + m.group(2)[:1200] + m.group(3),
            compact_user,
        )
        compact_user = re.sub(
            r"(Repaired code block:\n)([\s\S]*?)(\n\nRepair plan:)",
            lambda m: m.group(1) + m.group(2)[:1200] + m.group(3),
            compact_user,
        )
        compact_user = re.sub(r"\n{3,}", "\n\n", compact_user)
        return compact_system, compact_user

    def _coerce_prompt_text(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, (list, tuple)):
            return "".join(self._coerce_prompt_text(item) for item in prompt)
        if prompt is None:
            return ""
        return str(prompt)

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
            f"### Required methods:\n{self._format_required_methods(required_methods)}\n\n"
            f"### Supporting evidence:\n{method_context_block or '[none]'}\n\n"
            f"### Missing methods:\n{self._format_missing_methods(missing_method_names)}\n\n"
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
            "- For generic SATD, pass requires both a concrete operation and a local target; method context alone is insufficient.\n"
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
        return self._coerce_analysis(payload, state, source="llm_stage1")

    def _format_required_methods(self, methods: list[str]) -> str:
        items = [str(item).strip() for item in methods if str(item).strip()]
        return ", ".join(items) if items else "[none]"

    def _format_missing_methods(self, methods: list[str]) -> str:
        items = [str(item).strip() for item in methods if str(item).strip()]
        return ", ".join(items) if items else "[none]"

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

    def _build_analyzer_context_summary(self, bundle: dict[str, Any] | None) -> str:
        if not bundle:
            return ""
        base = bundle.get("base_context") or {}
        repair = bundle.get("repair_context") or {}
        metadata = bundle.get("metadata") or {}
        target_function = (base.get("target_function") or {}) if isinstance(base, dict) else {}
        module_symbols = (repair.get("module_symbols") or {}) if isinstance(repair, dict) else {}
        same_file_helpers = (repair.get("same_file_helpers") or {}) if isinstance(repair, dict) else {}
        same_class_evidence = (repair.get("same_class_evidence") or {}) if isinstance(repair, dict) else {}
        same_file_pattern = (repair.get("same_file_pattern") or {}) if isinstance(repair, dict) else {}
        targeted_callsite = (repair.get("targeted_callsite_snippet") or {}) if isinstance(repair, dict) else {}
        callsite_item = targeted_callsite.get("item") or {}

        summary = {
            "definition_context": {
                "target_function_found": target_function.get("found"),
                "symbol_name": target_function.get("symbol_name"),
                "class_name": target_function.get("class_name"),
                "matched_by": target_function.get("matched_by"),
                "module_symbols_count": module_symbols.get("count", 0),
                "module_symbols_preview": [
                    {
                        "symbol_name": item.get("symbol_name"),
                        "kind": item.get("kind"),
                    }
                    for item in (module_symbols.get("items") or [])[:5]
                ],
            },
            "usage_context": {
                "call_sites_count": metadata.get("call_sites_count", 0),
                "targeted_callsite_found": bool(targeted_callsite.get("used") and callsite_item),
                "targeted_callsite_path": callsite_item.get("path"),
                "targeted_callsite_excerpt": (callsite_item.get("excerpt") or "")[:400],
            },
            "replacement_context": {
                "same_file_helpers_count": same_file_helpers.get("count", 0),
                "same_class_methods_count": same_class_evidence.get("related_methods_count", 0),
                "same_class_attributes_count": same_class_evidence.get("related_attributes_count", 0),
                "same_file_pattern_count": same_file_pattern.get("count", 0),
            },
        }
        return json.dumps(summary, ensure_ascii=False)

    def build_easy_route_analysis(self, route_type: str) -> AnalysisResult:
        note = f"Easy SATD route '{route_type}' bypassed the generic analyzer and passed directly."
        return self._build_result(
            decision="pass",
            confidence=1.0,
            operation_concrete="high",
            localizable="high",
            local_scope="high",
            end_state_clear="high",
            notes=note,
            comment_evidence=route_type,
            code_evidence="easy route bypass",
            satd_type=route_type,
            source="rule_easy_route",
            scope_radius="function",
        )

    def build_rule_drop_analysis(self, notes: str) -> AnalysisResult:
        return self._build_result(
            decision="drop",
            confidence=1.0,
            operation_concrete="low",
            localizable="low",
            local_scope="low",
            end_state_clear="low",
            notes=notes,
            comment_evidence="",
            code_evidence="rule-based missing input",
            satd_type="generic",
            source="rule_drop",
            scope_radius="function",
        )

    def _coerce_analysis(self, payload: dict[str, Any], state: GraphState, source: str) -> AnalysisResult:
        requested_decision = self._normalize_decision(payload.get("decision"))
        confidence = self._clamp_float(payload.get("confidence"), 0.0)
        operation_concrete = self._normalize_level(payload.get("operation_concrete"))
        localizable = self._normalize_level(payload.get("localizable"))
        local_scope = self._normalize_level(payload.get("local_scope"))
        end_state_clear = self._normalize_level(payload.get("end_state_clear"))
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
        satd_type = str(state.get("satd_route_type") or "generic")
        decision = self._decision_from_checks(
            requested_decision=requested_decision,
            operation_concrete=operation_concrete,
            localizable=localizable,
            local_scope=local_scope,
            end_state_clear=end_state_clear,
        )
        preserve_grounded_target = self._should_preserve_grounded_local_target(
            state=state,
            operation_concrete=operation_concrete,
            localizable=localizable,
            local_scope=local_scope,
            end_state_clear=end_state_clear,
        )
        strict_drop_reason = self._strict_generic_drop_reason(
            state=state,
            operation_concrete=operation_concrete,
            localizable=localizable,
            local_scope=local_scope,
            end_state_clear=end_state_clear,
        )
        if strict_drop_reason:
            decision = "drop"
            confidence = max(confidence, 0.70)
            if strict_drop_reason == "open_ended_without_specific_local_edit":
                operation_concrete = "low"
                end_state_clear = "low"
            elif strict_drop_reason == "not_local_repair":
                local_scope = "low"
            elif strict_drop_reason == "target_not_localizable":
                localizable = "low"
            notes = self._append_analysis_note(notes, f"Analyzer strict drop: {strict_drop_reason}.")
        elif decision == "drop" and preserve_grounded_target:
            decision = "uncertain"
            if localizable == "low":
                localizable = "partial"
            if local_scope == "low":
                local_scope = "partial"
            notes = self._append_analysis_note(notes, "Analyzer kept item: grounded_local_target.")
        return self._build_result(
            decision=decision,
            confidence=confidence,
            operation_concrete=operation_concrete,
            localizable=localizable,
            local_scope=local_scope,
            end_state_clear=end_state_clear,
            notes=notes,
            comment_evidence=comment_evidence,
            code_evidence=code_evidence,
            satd_type=satd_type,
            source=source,
            scope_radius=self._infer_scope_radius_from_code(state.get("original_code") or ""),
        )

    def _strict_generic_drop_reason(
        self,
        *,
        state: GraphState,
        operation_concrete: str | None,
        localizable: str | None,
        local_scope: str | None,
        end_state_clear: str | None,
    ) -> str:
        if str(state.get("satd_route_type") or "generic").strip().lower() != "generic":
            return ""
        comment = str(state.get("satd_comment") or "")
        code = str(state.get("original_code") or "")
        open_ended_comment = self._looks_open_ended_task(comment)
        if self._should_preserve_grounded_local_target(
            state=state,
            operation_concrete=operation_concrete,
            localizable=localizable,
            local_scope=local_scope,
            end_state_clear=end_state_clear,
        ):
            return ""
        if operation_concrete == "low" and end_state_clear == "low":
            return "unclear_operation_and_end_state"
        if localizable == "low" and (
            operation_concrete == "low" or end_state_clear == "low" or open_ended_comment
        ):
            return "target_not_localizable"
        if local_scope == "low" and (
            operation_concrete == "low" or end_state_clear == "low" or open_ended_comment
        ):
            return "not_local_repair"
        if open_ended_comment and not self._has_strong_local_edit_signal(comment, code):
            return "open_ended_without_specific_local_edit"
        return ""

    def _should_preserve_grounded_local_target(
        self,
        *,
        state: GraphState,
        operation_concrete: str | None,
        localizable: str | None,
        local_scope: str | None,
        end_state_clear: str | None,
    ) -> bool:
        if str(state.get("satd_route_type") or "generic").strip().lower() != "generic":
            return False
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
        target_nouns = (
            "column",
            "field",
            "key",
            "argument",
            "parameter",
            "param",
            "method",
            "function",
            "class",
            "attribute",
            "variable",
            "option",
            "setting",
            "flag",
            "encoding",
            "url",
            "name",
            "symbol",
            "structure",
        )
        noun_pattern = "|".join(target_nouns)
        for match in re.finditer(rf"\b([A-Za-z][A-Za-z0-9_-]{{2,}}(?:\s+[A-Za-z][A-Za-z0-9_-]{{2,}}){{0,2}})\s+({noun_pattern})\b", text, flags=re.IGNORECASE):
            phrase = " ".join(part for part in match.groups() if part)
            terms.add(phrase)
            terms.add(match.group(1))
            preceding_words = match.group(1).split()
            if preceding_words:
                terms.add(f"{preceding_words[-1]} {match.group(2)}")
        action_pattern = r"\b(?:implement|add|prevent|deprecat\w*|honor|handle|parse|uncomment|return|raise|check)\s+(?:the\s+|a\s+|an\s+)?([A-Za-z_][A-Za-z0-9_]{2,})\b"
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
                parts.extend(
                    [
                        item.method_name,
                        item.signature,
                        item.source,
                        item.callsite_slice,
                        item.evidence_slice,
                    ]
                )
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

    def _append_analysis_note(self, notes: str, addition: str) -> str:
        base = self._one_line(notes)
        extra = self._one_line(addition)
        if not base:
            return extra
        if not extra:
            return base
        return self._one_line(f"{base} {extra}")

    def _build_result(
        self,
        *,
        decision: str,
        confidence: float,
        operation_concrete: str | None,
        localizable: str | None,
        local_scope: str | None,
        end_state_clear: str | None,
        notes: str,
        comment_evidence: str,
        code_evidence: str,
        satd_type: str,
        source: str,
        scope_radius: str,
    ) -> AnalysisResult:
        repairable = decision != "drop"
        confidence = self._clamp_float(confidence, 0.0)
        operation_concrete = operation_concrete if operation_concrete is not None else ("high" if decision == "pass" else "low" if decision == "drop" else "partial")
        localizable = localizable if localizable is not None else ("high" if decision == "pass" else "low" if decision == "drop" else "partial")
        local_scope = local_scope if local_scope is not None else ("high" if decision == "pass" else "low" if decision == "drop" else "partial")
        end_state_clear = end_state_clear if end_state_clear is not None else ("high" if decision == "pass" else "low" if decision == "drop" else "partial")

        operation_score = self._level_score(operation_concrete)
        localizable_score = self._level_score(localizable)
        local_scope_score = self._level_score(local_scope)
        end_state_score = self._level_score(end_state_clear)
        if decision == "pass":
            repairability_score = max(0.70, confidence or 0.80)
            intent_clarity = 0.25 + (0.65 * operation_score)
            change_locality = 0.25 + (0.65 * local_scope_score)
            semantic_risk = 0.28
            context_sufficiency = 0.20 + (0.35 * localizable_score) + (0.25 * end_state_score)
            verifiability = 0.68
            analyze_score = max(0.72, repairability_score)
            risk_level = "medium" if satd_type == "generic" else "low"
            context_score = context_sufficiency
            clarity_score = intent_clarity
            context_gaps: list[str] = []
            repair_strategy = "Pass this item directly to the repair stage."
        elif decision == "uncertain":
            repairability_score = max(0.45, confidence or 0.55)
            intent_clarity = 0.20 + (0.45 * operation_score)
            change_locality = 0.20 + (0.45 * local_scope_score)
            semantic_risk = 0.50
            context_sufficiency = 0.12 + (0.28 * localizable_score) + (0.18 * end_state_score)
            verifiability = 0.46
            analyze_score = max(0.46, repairability_score)
            risk_level = "medium"
            context_score = context_sufficiency
            clarity_score = intent_clarity
            context_gaps = self._derive_context_gaps(operation_concrete, localizable, local_scope, end_state_clear)
            repair_strategy = "Allow repair to try, but treat this item as ambiguous."
        else:
            repairability_score = 0.0
            intent_clarity = 0.08 + (0.22 * operation_score)
            change_locality = 0.06 + (0.18 * local_scope_score)
            semantic_risk = 0.82
            context_sufficiency = 0.04 + (0.16 * localizable_score) + (0.10 * end_state_score)
            verifiability = 0.18
            analyze_score = min(0.30, confidence)
            risk_level = "high"
            context_score = context_sufficiency
            clarity_score = intent_clarity
            context_gaps = self._derive_context_gaps(operation_concrete, localizable, local_scope, end_state_clear)
            repair_strategy = "Drop this item before repair."

        return AnalysisResult(
            decision=decision,
            repairable=repairable,
            repairability_score=repairability_score,
            intent_clarity=intent_clarity,
            change_locality=change_locality,
            semantic_risk=semantic_risk,
            context_sufficiency=context_sufficiency,
            verifiability=verifiability,
            analyze_score=analyze_score,
            confidence=confidence,
            satd_type=satd_type,
            reason=notes or decision,
            evidence_summary=notes or decision,
            risk_level=risk_level,
            context_score=context_score,
            clarity_score=clarity_score,
            scope_radius=scope_radius,
            operation_concrete=operation_concrete,
            localizable=localizable,
            local_scope=local_scope,
            end_state_clear=end_state_clear,
            comment_evidence=comment_evidence,
            code_evidence=code_evidence,
            validation_signals=[
                f"simple_analyzer:{source}",
                f"decision:{decision}",
                f"operation_concrete:{operation_concrete}",
                f"localizable:{localizable}",
                f"local_scope:{local_scope}",
                f"end_state_clear:{end_state_clear}",
            ],
            context_gaps=context_gaps,
            followup_context_requests=[],
            repair_strategy=repair_strategy,
            historical_snapshot_mismatch=False,
            github_evidence_strength="low",
        )

    def _infer_scope_radius_from_code(self, code: str) -> str:
        lowered = (code or "").lower()
        if "class " in lowered:
            return "class"
        if "def " in lowered or "async def " in lowered:
            return "function"
        if len([line for line in (code or "").splitlines() if line.strip()]) <= 2:
            return "line"
        return "function"

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

        if localizable == "high" and local_scope == "high" and (operation_concrete == "high" or end_state_clear == "high"):
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

    def _has_existing_target_hint(self, comment: str, code: str) -> bool:
        lowered = (comment or "").lower()
        if not lowered.strip() or not (code or "").strip():
            return False
        anchor_patterns = [
            r"\bthis\b",
            r"\bthese\b",
            r"\bthat\b",
            r"\bcurrent\b",
            r"\bexisting\b",
            r"\balready\b",
            r"\bhere\b",
            r"\bapi\b",
            r"\bparameter\b",
            r"\bparam\b",
            r"\bargument\b",
            r"\bkey\b",
            r"\bkeys\b",
            r"\bpath\b",
            r"\bbranch\b",
            r"\bblock\b",
            r"\bvalue\b",
            r"\bfield\b",
            r"\bindex name\b",
            r"\bline\b",
            r"\bobject\b",
        ]
        return any(re.search(pattern, lowered) for pattern in anchor_patterns)

    def _looks_open_ended_task(self, comment: str) -> bool:
        lowered = (comment or "").lower()
        open_ended_patterns = [
            r"\bimplement\b",
            r"\bdecide\b",
            r"\bdecide whether\b",
            r"\bconsider\b",
            r"\bfigure out\b",
            r"\bmechanism\b",
            r"\bdetermine\b",
            r"\binvestigat",
            r"\blook into\b",
            r"\bdebug\b",
            r"\brefactor\b",
            r"\brewrite\b",
            r"\bredesign\b",
            r"\bdeprecat",
            r"\bclean up\b",
            r"\bfrom scratch\b",
            r"\boptimi[sz]e\b",
            r"\bperformance\b",
            r"\bmemory\b",
            r"\bexploding memory\b",
            r"\badd this\b",
            r"\b1:1\b",
            r"\b1:many\b",
            r"\bglobal\b",
            r"\barchitecture\b",
            r"\bintegration\b",
        ]
        return any(re.search(pattern, lowered) for pattern in open_ended_patterns)

    def _maybe_compress_uncertain(self, state: GraphState, analysis: AnalysisResult) -> AnalysisResult:
        if analysis.decision != "uncertain":
            return analysis
        if not self._is_second_stage_uncertain_candidate(state, analysis):
            return analysis

        context_summary = self._build_analyzer_context_summary(state.get("github_context"))
        system_prompt = (
            "You are a second-stage SATD triage analyzer for generic uncertain SATD items.\n"
            "Your job is to decide whether an uncertain item should remain uncertain or be dropped after examining compact repository context.\n"
            "Use only the SATD comment, current code snippet, first-stage structured summary, and the compact definition/usage context.\n"
            "Prefer keeping the item as UNCERTAIN unless the context strongly shows that this is not a closed local edit task.\n"
            "A DROP should be reserved for tasks that still lack a credible existing target, lack local usage evidence, or remain fundamentally open-ended after context is considered.\n"
            "If the context shows an existing object, local usage, or a plausible replacement path, prefer UNCERTAIN.\n"
            "If the context still indicates broad integration, migration, validation strategy, or open-ended improvement work with weak usage locality and weak replacement evidence, prefer DROP.\n"
            "Return strict JSON only."
        )
        user_prompt = (
            f"SATD comment:\n{state['satd_comment']}\n\n"
            f"Current code snippet:\n```python\n{state['original_code']}\n```\n\n"
            "First-stage summary:\n"
            f"- operation_concrete: {analysis.operation_concrete}\n"
            f"- localizable: {analysis.localizable}\n"
            f"- local_scope: {analysis.local_scope}\n"
            f"- end_state_clear: {analysis.end_state_clear}\n"
            f"- comment_evidence: {analysis.comment_evidence or '(none)'}\n"
            f"- code_evidence: {analysis.code_evidence or '(none)'}\n\n"
            + (f"Compact definition/usage context:\n{context_summary}\n\n" if context_summary else "")
            +
            "Return JSON with exactly:\n"
            "{\n"
            '  "decision": "drop" | "uncertain",\n'
            '  "confidence": 0.0,\n'
            '  "target_existence": "high" | "partial" | "low",\n'
            '  "usage_locality": "high" | "partial" | "low",\n'
            '  "replacement_evidence": "high" | "partial" | "low",\n'
            '  "task_closedness": "high" | "partial" | "low",\n'
            '  "notes": "one short sentence"\n'
            "}\n"
        )
        payload = self.client.generate_json(
            system_prompt,
            user_prompt,
            request_label=f"analyze_stage2:task_{state['task_id']}",
        )
        return self._coerce_uncertain_compression(payload, analysis, state)

    def _is_second_stage_uncertain_candidate(self, state: GraphState, analysis: AnalysisResult) -> bool:
        if str(state.get("satd_route_type") or "generic") != "generic":
            return False
        if analysis.decision != "uncertain":
            return False
        bundle = state.get("github_context") or {}
        base = bundle.get("base_context") or {}
        repair = bundle.get("repair_context") or {}
        has_definition = bool((base.get("target_function") or {}).get("found")) or bool((repair.get("module_symbols") or {}).get("count"))
        has_usage = bool((repair.get("targeted_callsite_snippet") or {}).get("used")) or bool((bundle.get("metadata") or {}).get("call_sites_count"))
        has_replacement = bool((repair.get("same_file_helpers") or {}).get("count")) or bool((repair.get("same_file_pattern") or {}).get("count"))
        if has_definition or has_usage or has_replacement:
            return True
        weak_local_shape = (
            analysis.localizable == "high"
            and analysis.local_scope == "high"
            and analysis.operation_concrete in {"low", "partial"}
            and analysis.end_state_clear in {"low", "partial"}
        )
        return weak_local_shape

    def _coerce_uncertain_compression(
        self,
        payload: dict[str, Any],
        base: AnalysisResult,
        state: GraphState,
    ) -> AnalysisResult:
        requested = self._normalize_decision(payload.get("decision"))
        confidence = self._clamp_float(payload.get("confidence"), 0.0)
        target_existence = self._normalize_level(payload.get("target_existence"))
        usage_locality = self._normalize_level(payload.get("usage_locality"))
        replacement_evidence = self._normalize_level(payload.get("replacement_evidence"))
        task_closedness = self._normalize_level(payload.get("task_closedness"))
        notes = self._one_line(payload.get("notes"))
        open_ended_comment = self._looks_open_ended_task(str(state.get("satd_comment") or ""))

        low_count = sum(
            1 for item in (target_existence, usage_locality, replacement_evidence, task_closedness) if item == "low"
        )
        should_drop_by_open_task = (
            requested == "drop"
            and task_closedness == "low"
            and usage_locality == "low"
            and replacement_evidence != "high"
            and confidence >= 0.55
        )
        should_drop_by_missing_target = (
            requested == "drop"
            and target_existence == "low"
            and usage_locality == "low"
            and confidence >= 0.55
        )
        should_drop_by_broad_context = (
            requested == "drop"
            and task_closedness == "low"
            and usage_locality != "high"
            and replacement_evidence == "low"
            and confidence >= 0.58
        )
        should_drop_by_partial_target_but_nonlocal = (
            requested == "drop"
            and target_existence == "partial"
            and usage_locality == "low"
            and task_closedness == "low"
            and replacement_evidence != "high"
            and confidence >= 0.60
        )
        should_drop_by_open_ended_weak_spec = (
            open_ended_comment
            and base.operation_concrete == "low"
            and base.end_state_clear == "low"
            and base.localizable == "high"
            and base.local_scope == "high"
            and replacement_evidence != "high"
        )
        should_drop = (
            should_drop_by_open_task
            or should_drop_by_missing_target
            or should_drop_by_broad_context
            or should_drop_by_partial_target_but_nonlocal
            or should_drop_by_open_ended_weak_spec
            or (
                requested == "drop"
                and target_existence == "low"
                and task_closedness == "low"
                and low_count >= 3
                and confidence >= 0.60
            )
        )
        if not should_drop:
            if notes:
                base.reason = notes
                base.evidence_summary = notes
            base.validation_signals.append(
                "simple_analyzer:stage2_keep:"
                f"{target_existence or 'unknown'}:{usage_locality or 'unknown'}:"
                f"{replacement_evidence or 'unknown'}:{task_closedness or 'unknown'}"
            )
            return base

        updated = copy.deepcopy(base)
        updated.decision = "drop"
        updated.repairable = False
        updated.confidence = max(base.confidence, confidence)
        updated.repairability_score = 0.0
        updated.semantic_risk = max(updated.semantic_risk, 0.82)
        updated.verifiability = min(updated.verifiability, 0.18)
        updated.analyze_score = min(updated.analyze_score, 0.30)
        updated.risk_level = "high"
        updated.reason = notes or "Second-stage analyzer judged this uncertain item to be fundamentally open-ended."
        updated.evidence_summary = updated.reason
        updated.repair_strategy = "Drop this item after second-stage uncertain compression."
        updated.validation_signals.append(
            f"simple_analyzer:stage2_drop:{target_existence}:{usage_locality}:{replacement_evidence}:{task_closedness}"
        )
        return updated

    def _normalize_decision(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"pass", "uncertain", "drop"}:
            return text
        if "drop" in text:
            return "drop"
        if "uncertain" in text or "need" in text:
            return "uncertain"
        return "pass"

    def _normalize_level(self, value: Any) -> str | None:
        if isinstance(value, bool):
            return "high" if value else "low"
        text = str(value or "").strip().lower()
        if text in {"high", "strong", "clear"}:
            return "high"
        if text in {"partial", "medium", "mixed", "somewhat"}:
            return "partial"
        if text in {"low", "weak", "unclear"}:
            return "low"
        if text in {"true", "yes", "1"}:
            return "high"
        if text in {"false", "no", "0"}:
            return "low"
        return None

    def _derive_context_gaps(
        self,
        operation_concrete: str | None,
        localizable: str | None,
        local_scope: str | None,
        end_state_clear: str | None,
    ) -> list[str]:
        gaps: list[str] = []
        if operation_concrete == "low":
            gaps.append("not_operation_concrete")
        elif operation_concrete == "partial":
            gaps.append("partially_operation_concrete")
        if localizable == "low":
            gaps.append("not_localizable")
        elif localizable == "partial":
            gaps.append("partially_localizable")
        if local_scope == "low":
            gaps.append("not_local_scope")
        elif local_scope == "partial":
            gaps.append("partially_local_scope")
        if end_state_clear == "low":
            gaps.append("not_end_state_clear")
        elif end_state_clear == "partial":
            gaps.append("partially_end_state_clear")
        return gaps

    def _level_score(self, value: str | None) -> float:
        if value == "high":
            return 1.0
        if value == "partial":
            return 0.55
        if value == "low":
            return 0.0
        return 0.45

    def _one_line(self, value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        return text[:240]

    def _clamp_float(self, value: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default


class OpenAIFixer:
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
    ) -> tuple[RepairAttempt, MethodInquiryResult, list[RetrievedMethodContext], list[str], list[UncertaintyItem], list[EditConstraint]]:
        round_id = state["round_id"] + 1
        (
            method_inquiry,
            found_method_contexts,
            missing_method_names,
            uncertainty_items,
            edit_constraints,
            _method_context_block,
        ) = self._method_context_from_state(state, candidate_mode=candidate_mode)
        effective_candidate_mode = self._effective_candidate_mode(candidate_mode, found_method_contexts)
        self._checkpoint(
            state,
            stage="generation_start",
            payload={
                "round_id": round_id,
                "candidate_mode": candidate_mode,
                "required_methods": list(method_inquiry.required_methods),
                "method_notes": self._serialize_method_notes(method_inquiry.method_notes),
                "uncertainty_items": self._serialize_uncertainty_items(uncertainty_items),
                "retrieved_method_contexts": self._serialize_retrieved_method_contexts(found_method_contexts),
                "missing_method_names": list(missing_method_names),
                "edit_constraints": self._serialize_edit_constraints(edit_constraints),
            },
        )
        system_prompt, user_prompt = self._build_repair_prompts(
            state=state,
            round_id=round_id,
            method_inquiry=method_inquiry,
            retrieved_method_contexts=found_method_contexts,
            missing_method_names=missing_method_names,
            edit_constraints=edit_constraints,
        )
        payload = self.client.generate_json(
            system_prompt,
            user_prompt,
            request_label=f"repair:task_{state['task_id']}:round_{round_id}:candidate_{effective_candidate_mode}",
            max_tokens=self._repair_max_tokens(),
        )
        repaired_code = str(payload.get("repaired_code") or state["original_code"])
        repair_plan = self._default_repair_plan(state)
        changed_scope = self._infer_changed_scope(state["original_code"], repaired_code)
        confidence = self._default_repair_confidence(state, repaired_code, found_method_contexts)
        notes = "repair metadata generated locally"
        if method_inquiry.required_methods and not found_method_contexts:
            notes = f"{notes} | no_method_context_found"
        elif found_method_contexts:
            notes = f"{notes} | method_contexts={len(found_method_contexts)}"
        else:
            notes = f"{notes} | no_required_methods_identified"

        em_risk_notes = self._em_risk_notes(state["original_code"], repaired_code)
        if em_risk_notes:
            confidence = min(confidence, 0.42)
            notes = f"{notes} | em_risk={','.join(em_risk_notes)}"
        self._log(
            state,
            f"repair output scope={changed_scope} confidence={confidence:.2f} mode={effective_candidate_mode}"
        )
        self._checkpoint(
            state,
            stage="generation_done",
            payload={
                "round_id": round_id,
                "candidate_mode": effective_candidate_mode,
                "repair_plan": repair_plan,
                "repaired_code": repaired_code,
                "changed_scope": changed_scope,
                "confidence": confidence,
                "notes": notes,
            },
        )

        return (
            RepairAttempt(
                round_id=round_id,
                repair_plan=repair_plan,
                repaired_code=repaired_code,
                changed_scope=changed_scope,
                confidence=confidence,
                notes=notes,
                candidate_mode=effective_candidate_mode,
            ),
            method_inquiry,
            found_method_contexts,
            missing_method_names,
            uncertainty_items,
            edit_constraints,
        )

    def prepare_method_context(
        self,
        state: GraphState,
        candidate_mode: str = "baseline_context",
    ) -> tuple[MethodInquiryResult, list[RetrievedMethodContext], list[str], list[UncertaintyItem], list[EditConstraint], str]:
        round_id = state["round_id"] + 1
        use_method_context = self._candidate_uses_method_context(candidate_mode)
        self._checkpoint(
            state,
            stage="question_start",
            payload={"round_id": round_id, "candidate_mode": candidate_mode},
        )
        if use_method_context:
            method_inquiry = self.identify_required_methods(state)
            self._log(
                state,
                f"context questions={self._format_method_list(method_inquiry.required_methods)} "
                f"reason={method_inquiry.reason or 'unspecified'}"
            )
        else:
            method_inquiry = MethodInquiryResult(reason="candidate_mode_without_method_context")
            self._log(state, f"context questions=skipped mode={candidate_mode}")
        effective_use_method_context = use_method_context and bool(method_inquiry.required_methods)
        uncertainty_items = list(method_inquiry.uncertainty_items or [])
        self._checkpoint(
            state,
            stage="question_done",
            payload={
                "round_id": round_id,
                "candidate_mode": candidate_mode,
                "required_methods": list(method_inquiry.required_methods),
                "reason": method_inquiry.reason,
                "method_notes": self._serialize_method_notes(method_inquiry.method_notes),
                "uncertainty_items": self._serialize_uncertainty_items(uncertainty_items),
            },
        )
        self._checkpoint(
            state,
            stage="retrieval_start",
            payload={
                "round_id": round_id,
                "candidate_mode": candidate_mode,
                "required_methods": list(method_inquiry.required_methods),
            },
        )
        if effective_use_method_context:
            method_location_hints = self.resolve_method_locations(state, method_inquiry.required_methods)
            self._checkpoint(
                state,
                stage="retrieval_route_done",
                payload={
                    "round_id": round_id,
                    "candidate_mode": candidate_mode,
                    "required_methods": list(method_inquiry.required_methods),
                    "method_location_hints": method_location_hints,
                },
            )
            retrieved_method_contexts = self._retrieve_method_contexts(
                state,
                method_inquiry.required_methods,
                method_location_hints=method_location_hints,
            )
            retrieved_method_contexts = self._enrich_retrieved_method_contexts(
                state=state,
                method_inquiry=method_inquiry,
                retrieved_method_contexts=retrieved_method_contexts,
            )
            missing_method_names = [item.method_name for item in retrieved_method_contexts if not item.found]
            found_method_contexts = [item for item in retrieved_method_contexts if item.found]
            found_method_contexts = self._apply_method_context_gate(
                state=state,
                method_inquiry=method_inquiry,
                retrieved_method_contexts=found_method_contexts,
            )
            self._log(
                state,
                "context results "
                f"found={len(found_method_contexts)}/{len(method_inquiry.required_methods)} "
                f"missing={self._format_method_list(missing_method_names, empty='none')}"
            )
        else:
            method_location_hints = {}
            retrieved_method_contexts = []
            missing_method_names = []
            found_method_contexts = []
            if use_method_context:
                self._log(state, "context results=skipped reason=no_indispensable_methods")
            else:
                self._log(state, f"context results=skipped mode={candidate_mode}")
            self._checkpoint(
                state,
                stage="retrieval_route_done",
                payload={
                    "round_id": round_id,
                    "candidate_mode": candidate_mode,
                    "required_methods": [],
                    "method_location_hints": {},
                },
            )
        edit_constraints = self._synthesize_edit_constraints(
            state=state,
            method_inquiry=method_inquiry,
            retrieved_method_contexts=found_method_contexts,
        )
        effective_candidate_mode = self._effective_candidate_mode(candidate_mode, found_method_contexts)
        self._checkpoint(
            state,
            stage="retrieval_done",
            payload={
                "round_id": round_id,
                "candidate_mode": candidate_mode,
                "required_methods": list(method_inquiry.required_methods),
                "method_notes": self._serialize_method_notes(method_inquiry.method_notes),
                "uncertainty_items": self._serialize_uncertainty_items(uncertainty_items),
                "method_location_hints": method_location_hints,
                "retrieved_method_contexts": self._serialize_retrieved_method_contexts(retrieved_method_contexts),
                "missing_method_names": list(missing_method_names),
                "edit_constraints": self._serialize_edit_constraints(edit_constraints),
            },
        )
        method_context_block = "[none]"
        if self._candidate_uses_method_context(candidate_mode):
            method_context_block = self._format_method_context_block(
                state=state,
                method_inquiry=method_inquiry,
                retrieved_method_contexts=found_method_contexts,
                missing_method_names=missing_method_names,
            )

        return (
            method_inquiry,
            found_method_contexts,
            missing_method_names,
            uncertainty_items,
            edit_constraints,
            method_context_block,
        )

    def _method_context_from_state(
        self,
        state: GraphState,
        candidate_mode: str,
    ) -> tuple[MethodInquiryResult, list[RetrievedMethodContext], list[str], list[UncertaintyItem], list[EditConstraint], str]:
        if not self._candidate_uses_method_context(candidate_mode):
            method_inquiry = MethodInquiryResult(reason="candidate_mode_without_method_context")
            return method_inquiry, [], [], [], [], "[none]"
        method_inquiry = state.get("method_inquiry")
        if not isinstance(method_inquiry, MethodInquiryResult):
            method_inquiry = MethodInquiryResult(reason="analyzer_method_context_not_available")
        retrieved_method_contexts = [
            item
            for item in (state.get("retrieved_method_contexts") or [])
            if isinstance(item, RetrievedMethodContext)
        ]
        missing_method_names = [str(item) for item in (state.get("missing_method_names") or []) if str(item).strip()]
        uncertainty_items = [
            item
            for item in (state.get("uncertainty_items") or [])
            if isinstance(item, UncertaintyItem)
        ]
        edit_constraints = [
            item
            for item in (state.get("edit_constraints") or [])
            if isinstance(item, EditConstraint)
        ]
        method_context_block = self._format_method_context_block(
            state=state,
            method_inquiry=method_inquiry,
            retrieved_method_contexts=retrieved_method_contexts,
            missing_method_names=missing_method_names,
        )
        return (
            method_inquiry,
            retrieved_method_contexts,
            missing_method_names,
            uncertainty_items,
            edit_constraints,
            method_context_block,
        )

    def identify_required_methods(self, state: GraphState) -> MethodInquiryResult:
        if self.repair_context_mode not in {"method_query", "clone_treesitter"}:
            return MethodInquiryResult()

        satd_route_type = str(state.get("satd_route_type") or "generic").strip().lower()
        satd_comment = state.get("satd_comment") or ""
        uncertainty_items = self._extract_uncertainty_candidates_from_satd_code(
            satd_comment=satd_comment,
            original_code=state["original_code"],
        )

        if self._skip_method_context_by_rule(satd_route_type, satd_comment):
            return MethodInquiryResult(
                required_methods=[],
                reason=f"rule_without_method_context:{satd_route_type or 'generic'}",
                uncertainty_items=uncertainty_items,
            )

        retrieval_candidates = [
            item
            for item in uncertainty_items
            if item.retrieval_eligible and item.kind in {"method", "symbol"}
        ]
        candidates = [
            {
                "raw_call": item.raw_name or item.name,
                "normalized_name": item.normalized_name or self._normalize_method_name(item.name),
                "line": item.line,
                "kind": item.kind,
            }
            for item in retrieval_candidates
            if item.normalized_name or self._normalize_method_name(item.name)
        ]
        if not candidates:
            return MethodInquiryResult(
                required_methods=[],
                reason="no_retrieval_eligible_method_symbol",
                uncertainty_items=uncertainty_items,
            )

        method_line = "、".join(
            str(item.get("normalized_name") or "").strip()
            for item in candidates
            if str(item.get("normalized_name") or "").strip()
        ) or "[none]"

        system_prompt = textwrap.dedent(
            """
            You will repair this SATD in the next step.
            Before that, select only the methods that are strongly relevant to exact-match repair.

            Rules:
            - Return only methods that you must understand before repairing.
            - Favor false negatives over false positives.
            - You must only choose from the provided methods.
            - For every selected method, explain why it is strongly related to the repair.

            Return JSON with:
            {
              "required_methods": [
                {"method_name": "...", "selection_reason": "..."}
              ],
              "reason": "short explanation"
            }
            """
        ).strip()
        user_prompt = textwrap.dedent(
            f"""
            You will repair this SATD next.
            Before that, you may ask for method context.

            Methods:
            {method_line}

            Comment:
            {satd_comment}

            Code:
            {state["original_code"]}

            Output the strongly relevant methods and their selection reasons.
            """
        ).strip()
        payload = self.client.generate_json(
            system_prompt,
            user_prompt,
            temperature=0.0,
            request_label=f"method_inquiry:task_{state.get('task_id', '?')}",
        )
        inquiry = self._coerce_method_inquiry(
            payload,
            allowed_methods=[str(item.get("normalized_name") or "") for item in candidates],
        )
        if not inquiry.reason:
            inquiry.reason = "llm_required_method_selection"
        inquiry.uncertainty_items = uncertainty_items
        return inquiry

    def _skip_method_context_by_rule(self, satd_route_type: str, satd_comment: str) -> bool:
        route = str(satd_route_type or "").strip().lower()
        if route in {"remove_temporary", "type_annotation", "replace_symbol", "document"}:
            return True
        return False

    def _coerce_method_inquiry(
        self,
        payload: dict[str, Any],
        allowed_methods: list[str] | None = None,
    ) -> MethodInquiryResult:
        raw_methods = payload.get("required_methods")
        required_methods: list[str] = []
        method_notes: list[dict[str, str]] = []
        seen: set[str] = set()
        allowed = set(allowed_methods or [])
        if isinstance(raw_methods, list):
            for item in raw_methods:
                selection_reason = ""
                raw_name = item
                if isinstance(item, dict):
                    raw_name = item.get("method_name") or item.get("name") or item.get("method") or ""
                    selection_reason = str(
                        item.get("selection_reason")
                        or item.get("reason")
                        or item.get("relation")
                        or item.get("why")
                        or ""
                    ).strip()
                normalized = self._normalize_method_name(raw_name)
                if not normalized or normalized in seen:
                    continue
                if allowed and normalized not in allowed:
                    continue
                seen.add(normalized)
                required_methods.append(normalized)
                method_notes.append(
                    {
                        "method_name": normalized,
                        "why_blocking": selection_reason or f"`{normalized}` may change the exact repair behavior.",
                        "what_to_learn": selection_reason or f"Understand how `{normalized}` affects the local SATD fix.",
                    }
                )
        reason = str(payload.get("reason") or "").strip()
        return MethodInquiryResult(required_methods=required_methods, reason=reason, method_notes=method_notes)

    def _extract_uncertainty_candidates_from_satd_code(self, satd_comment: str, original_code: str) -> list[UncertaintyItem]:
        items: list[UncertaintyItem] = []
        seen: set[tuple[str, str]] = set()

        def add(
            kind: str,
            name: str,
            line: int | None = None,
            retrieval_eligible: bool = False,
            why: str = "",
            *,
            raw_name: str = "",
            normalized_name: str = "",
            source_excerpt: str = "",
        ) -> None:
            normalized_name = (normalized_name or name or "").strip()
            if not normalized_name:
                return
            key = (kind, normalized_name)
            if key in seen:
                return
            seen.add(key)
            items.append(
                UncertaintyItem(
                    kind=kind,
                    name=normalized_name,
                    line=line,
                    retrieval_eligible=retrieval_eligible,
                    why_this_matters=why,
                    raw_name=(raw_name or name or "").strip(),
                    normalized_name=normalized_name,
                    source_excerpt=(source_excerpt or "").strip(),
                )
            )

        for candidate in self._extract_method_candidates_from_satd_code(original_code):
            name = str(candidate.get("normalized_name") or "")
            add(
                "method",
                name,
                candidate.get("line"),
                True,
                "Method call appears in the edit region.",
                raw_name=str(candidate.get("raw_call") or name),
                normalized_name=name,
                source_excerpt=str(candidate.get("source_excerpt") or ""),
            )

        comment = satd_comment or ""
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_\.]+", comment):
            token = token.strip(".")
            if not token:
                continue
            lower = token.lower()
            if lower in {"todo", "fixme", "none", "true", "false"}:
                continue
            normalized_token = self._normalize_method_name(token) or token
            if self._looks_like_retrieval_symbol(token):
                add(
                    "symbol",
                    normalized_token,
                    None,
                    True,
                    "Comment explicitly names an API-like symbol.",
                    raw_name=token,
                    normalized_name=normalized_token,
                    source_excerpt=comment.strip(),
                )
            elif "." in token:
                add(
                    "value",
                    token,
                    None,
                    False,
                    "Comment references a value-like qualified name.",
                    raw_name=token,
                    normalized_name=token,
                    source_excerpt=comment.strip(),
                )

        try:
            tree = ast.parse(textwrap.dedent((original_code or "").strip("\n")))
        except (SyntaxError, ValueError):
            return items

        class LocalSignalCollector(ast.NodeVisitor):
            def visit_If(self, node: ast.If) -> Any:
                snippet = ast.get_source_segment(original_code, node.test) or ""
                cleaned = snippet.strip()
                add(
                    "condition",
                    cleaned,
                    getattr(node, "lineno", None),
                    False,
                    "Condition may determine the patch shape.",
                    raw_name=cleaned,
                    normalized_name=cleaned,
                    source_excerpt=cleaned,
                )
                self.generic_visit(node)

            def visit_Return(self, node: ast.Return) -> Any:
                snippet = ast.get_source_segment(original_code, node.value) or "return"
                cleaned = snippet.strip()
                add(
                    "return",
                    cleaned,
                    getattr(node, "lineno", None),
                    False,
                    "Return semantics may matter.",
                    raw_name=cleaned,
                    normalized_name=cleaned,
                    source_excerpt=cleaned,
                )
                self.generic_visit(node)

            def visit_Assign(self, node: ast.Assign) -> Any:
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                        name = f"self.{target.attr}"
                        add(
                            "state_write",
                            name,
                            getattr(node, "lineno", None),
                            False,
                            "State mutation may matter.",
                            raw_name=name,
                            normalized_name=name,
                            source_excerpt=(ast.get_source_segment(original_code, node) or name).strip(),
                        )
                self.generic_visit(node)

        LocalSignalCollector().visit(tree)
        return items

    def _looks_like_retrieval_symbol(self, token: str) -> bool:
        cleaned = str(token or "").strip()
        if not cleaned:
            return False
        normalized = self._normalize_method_name(cleaned)
        if not normalized:
            return False
        tail = normalized.split(".")[-1]
        tail_segments = self._method_name_segments(tail)
        if len(tail) < 3:
            return False
        if tail.lower() in {"default", "backend", "config", "true", "false", "none"}:
            return False
        callable_markers = {
            "send",
            "close",
            "open",
            "load",
            "save",
            "reset",
            "resolve",
            "create",
            "build",
            "update",
            "remove",
            "delete",
            "write",
            "read",
        }
        if any(marker in tail.lower() for marker in {"default", "backend", "config"}) and not any(
            word in tail_segments for word in callable_markers
        ):
            return False
        return (
            "." in normalized
            or "_" in tail
            or any(ch.isupper() for ch in tail)
            or any(word in tail_segments for word in callable_markers)
        )

    def _method_name_segments(self, value: str) -> set[str]:
        text = str(value or "").strip()
        if not text:
            return set()
        tokens: set[str] = set()
        for chunk in text.replace(".", "_").split("_"):
            for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", chunk):
                tokens.add(part.lower())
        return tokens

    def _candidate_uses_method_context(self, candidate_mode: str) -> bool:
        mode = str(candidate_mode or "").strip().lower()
        if not mode:
            return True
        return not mode.endswith("no_context")

    def _effective_candidate_mode(
        self,
        candidate_mode: str,
        found_method_contexts: list[RetrievedMethodContext],
    ) -> str:
        mode = str(candidate_mode or "").strip()
        if not mode:
            return mode
        if not self._candidate_uses_method_context(mode):
            return mode
        if found_method_contexts:
            return mode
        if mode.endswith("_context"):
            return mode[: -len("_context")] + "_no_context"
        return mode

    def _extract_method_candidates_from_satd_code(self, code: str) -> list[dict[str, Any]]:
        snippet = textwrap.dedent((code or "").strip("\n"))
        if not snippet.strip():
            return []
        try:
            tree = ast.parse(snippet)
        except (SyntaxError, ValueError):
            return []

        builtin_names = set(dir(builtins))
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()

        class CallCollector(ast.NodeVisitor):
            def __init__(self, outer: "OpenAIFixer") -> None:
                self.outer = outer

            def visit_Call(self, node: ast.Call) -> Any:
                raw_call = self.outer._call_chain_from_node(node.func)
                normalized = self.outer._normalize_method_name(raw_call)
                if normalized and normalized not in seen:
                    tail = normalized.split(".")[-1]
                    if raw_call.count(".") == 0 and tail in builtin_names:
                        self.generic_visit(node)
                        return
                    seen.add(normalized)
                    candidates.append(
                        {
                            "raw_call": raw_call,
                            "normalized_name": normalized,
                            "line": getattr(node, "lineno", None),
                            "source_excerpt": (ast.get_source_segment(snippet, node) or "").strip(),
                        }
                    )
                self.generic_visit(node)

        CallCollector(self).visit(tree)
        candidates.sort(
            key=lambda item: (
                item.get("line") if isinstance(item.get("line"), int) else 10**9,
                item.get("raw_call") or "",
            )
        )
        return candidates

    def _call_chain_from_node(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = self._call_chain_from_node(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        if isinstance(node, ast.Call):
            return self._call_chain_from_node(node.func)
        return ""

    def _serialize_method_notes(self, method_notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for item in method_notes or []:
            if not isinstance(item, dict):
                continue
            serialized.append(
                {
                    "method_name": str(item.get("method_name") or "").strip(),
                    "why_blocking": str(item.get("why_blocking") or "").strip(),
                    "what_to_learn": str(item.get("what_to_learn") or "").strip(),
                }
            )
        return serialized

    def _serialize_uncertainty_items(self, items: list[UncertaintyItem]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for item in items or []:
            normalized = item.normalized_name or self._normalize_method_name(item.name) or item.name
            serialized.append(
                {
                    "kind": item.kind,
                    "name": item.name,
                    "raw_name": item.raw_name or item.name,
                    "normalized_name": normalized,
                    "line": item.line,
                    "source_excerpt": item.source_excerpt,
                    "retrieval_eligible": item.retrieval_eligible,
                    "why_this_matters": item.why_this_matters,
                }
            )
        return serialized

    def _serialize_edit_constraints(self, items: list[EditConstraint]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for item in items or []:
            serialized.append(
                {
                    "focus_point": item.focus_point,
                    "required_fact": item.required_fact,
                    "must_do": item.must_do,
                    "must_not_do": item.must_not_do,
                    "supporting_point_kind": item.supporting_point_kind,
                    "target_kind": item.target_kind,
                    "target_hint": item.target_hint,
                    "allowed_edit_radius": item.allowed_edit_radius,
                    "must_preserve_signature": item.must_preserve_signature,
                    "must_not_add_helper": item.must_not_add_helper,
                    "must_not_expand_control_flow": item.must_not_expand_control_flow,
                    "must_not_rewrite_unrelated_lines": item.must_not_rewrite_unrelated_lines,
                    "supporting_symbol": item.supporting_symbol,
                    "supporting_evidence": item.supporting_evidence,
                    "confidence": item.confidence,
                }
            )
        return serialized

    def _serialize_retrieved_method_contexts(self, items: list[RetrievedMethodContext]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for item in items or []:
            serialized.append(
                {
                    "method_name": item.method_name,
                    "path": item.path,
                    "class_name": item.class_name,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "source": item.source,
                    "found": item.found,
                    "signature": item.signature,
                    "callsite_slice": item.callsite_slice,
                    "evidence_slice": item.evidence_slice,
                    "match_score": item.match_score,
                    "confidence": item.confidence,
                    "confidence_label": item.confidence_label,
                }
            )
        return serialized

    def _normalize_method_name(self, value: Any) -> str:
        text = str(value or "").strip().strip("`").strip()
        if not text:
            return ""
        text = text.split("(", 1)[0].strip()
        parts = [part for part in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text) if part]
        while len(parts) > 1 and parts[0] in {"self", "cls", "super"}:
            parts = parts[1:]
        if not parts:
            return ""
        normalized = ".".join(parts) if len(parts) > 1 else parts[0]
        return "" if self._is_low_quality_method_name(normalized) else normalized

    def _is_low_quality_method_name(self, value: str) -> bool:
        text = str(value or "").strip()
        if not text:
            return True
        generic_terms = {
            "time", "get", "set", "info", "values", "isinstance", "append", "wait", "super",
            "len", "print", "list", "dict", "str", "int", "float", "bool", "type",
        }
        parts = [part for part in text.split(".") if part]
        if not parts:
            return True
        if len(parts) >= 2 and parts[-1].lower() in {"info", "debug", "warning", "error", "exception", "critical"}:
            return True
        if len(parts) > 1:
            return False
        tail = parts[-1].lower()
        return tail in generic_terms
    def _retrieve_method_contexts(
        self,
        state: GraphState,
        method_names: list[str],
        method_location_hints: dict[str, list[str]] | None = None,
    ) -> list[RetrievedMethodContext]:
        if not method_names:
            return []
        payloads = self.client.toolbox.fetch_method_contexts(
            state["user"],
            state["project"],
            state["file_path"],
            method_names,
            ref=(state.get("commit") or "").strip() or None,
            log_prefix=self._task_prefix(state),
            path_hints_by_method=method_location_hints or {},
        )
        contexts: list[RetrievedMethodContext] = []
        for item in payloads:
            contexts.append(
                RetrievedMethodContext(
                    method_name=str(item.get("method_name") or ""),
                    path=str(item.get("path") or ""),
                    class_name=str(item.get("class_name")) if item.get("class_name") is not None else None,
                    start_line=item.get("start_line"),
                    end_line=item.get("end_line"),
                    source=str(item.get("source") or ""),
                    found=bool(item.get("found")),
                    signature=str(item.get("signature") or ""),
                    callsite_slice=str(item.get("callsite_slice") or ""),
                    evidence_slice=str(item.get("evidence_slice") or ""),
                    match_score=int(item.get("match_score") or 0),
                    confidence=float(item.get("confidence") or 0.0),
                    confidence_label=str(item.get("confidence_label") or ""),
                )
            )
        return contexts

    def _enrich_retrieved_method_contexts(
        self,
        *,
        state: GraphState,
        method_inquiry: MethodInquiryResult,
        retrieved_method_contexts: list[RetrievedMethodContext],
    ) -> list[RetrievedMethodContext]:
        if not retrieved_method_contexts:
            return []
        comment_refs = self._extract_comment_reference_names(state.get("satd_comment") or "")
        required_refs = {
            self._normalize_method_name(name)
            for name in (method_inquiry.required_methods or [])
            if self._normalize_method_name(name)
        }
        enriched: list[RetrievedMethodContext] = []
        for item in retrieved_method_contexts:
            signature = item.signature or self._extract_signature_line(item.source)
            callsite_slice = item.callsite_slice or self._extract_callsite_slice(state["original_code"], item.method_name)
            evidence_slice = item.evidence_slice or self._extract_relevant_definition_slice(item.source)
            match_score = item.match_score or self._score_retrieved_method_context(
                item=item,
                comment_refs=comment_refs,
                required_refs=required_refs,
                callsite_slice=callsite_slice,
                evidence_slice=evidence_slice,
            )
            confidence = item.confidence or self._confidence_from_match_score(match_score)
            confidence_label = item.confidence_label or self._confidence_label(confidence)
            enriched.append(
                RetrievedMethodContext(
                    method_name=item.method_name,
                    path=item.path,
                    class_name=item.class_name,
                    start_line=item.start_line,
                    end_line=item.end_line,
                    source=item.source,
                    found=item.found,
                    signature=signature,
                    callsite_slice=callsite_slice,
                    evidence_slice=evidence_slice,
                    match_score=match_score,
                    confidence=confidence,
                    confidence_label=confidence_label,
                )
            )
        return enriched

    def _score_retrieved_method_context(
        self,
        *,
        item: RetrievedMethodContext,
        comment_refs: set[str],
        required_refs: set[str],
        callsite_slice: str,
        evidence_slice: str,
    ) -> int:
        method_name = self._normalize_method_name(item.method_name)
        tail = method_name.split(".")[-1] if method_name else ""
        comment_tail_refs = {name.split(".")[-1] for name in comment_refs if name}
        required_tail_refs = {name.split(".")[-1] for name in required_refs if name}
        score = 0
        if method_name in comment_refs:
            score += 1200
        elif tail in comment_tail_refs:
            score += 350
        if method_name in required_refs:
            score += 800
        elif tail in required_tail_refs:
            score += 220
        if callsite_slice:
            score += 900
        if evidence_slice:
            score += 300
        if item.path:
            score += 80
        if item.class_name:
            score += 30
        if self._is_low_signal_evidence_method(tail):
            score -= 850
        return max(0, score)

    def _confidence_from_match_score(self, match_score: int) -> float:
        if match_score >= 1800:
            return 0.99
        if match_score >= 1200:
            return 0.82
        if match_score >= 800:
            return 0.62
        return 0.35

    def _confidence_label(self, confidence: float) -> str:
        if confidence >= 0.8:
            return "high"
        if confidence >= 0.6:
            return "medium"
        return "low"

    def _is_low_signal_evidence_method(self, tail: str) -> bool:
        lowered = str(tail or "").lower()
        return lowered in {"gradcheck", "gradgradcheck", "assertequal", "sum", "backward", "product", "run_test"}

    def _apply_method_context_gate(
        self,
        *,
        state: GraphState,
        method_inquiry: MethodInquiryResult,
        retrieved_method_contexts: list[RetrievedMethodContext],
    ) -> list[RetrievedMethodContext]:
        if not retrieved_method_contexts:
            return []
        retrieved_method_contexts = self._enrich_retrieved_method_contexts(
            state=state,
            method_inquiry=method_inquiry,
            retrieved_method_contexts=retrieved_method_contexts,
        )

        comment_refs = self._extract_comment_reference_names(state.get("satd_comment") or "")
        required_refs = {
            self._normalize_method_name(name)
            for name in (method_inquiry.required_methods or [])
            if self._normalize_method_name(name)
        }
        required_tail_refs = {name.split(".")[-1] for name in required_refs if name}
        ranked: list[tuple[tuple[int, int, int, int], RetrievedMethodContext]] = []
        for item, pack in zip(retrieved_method_contexts, self._build_method_evidence_packs(state, retrieved_method_contexts), strict=False):
            method_name = self._normalize_method_name(item.method_name)
            tail = method_name.split(".")[-1] if method_name else ""
            if item.confidence_label == "low" or item.confidence < 0.6:
                continue
            exact_comment_hit = int(bool(method_name and (method_name in comment_refs or tail in comment_refs)))
            has_callsite = int(bool(pack.get("callsite_slice")))
            has_evidence = int(bool(pack.get("evidence_slice")))
            required_match = int(bool(method_name and (method_name in required_refs or tail in required_tail_refs)))
            match_score = int(item.match_score or 0)
            if not (exact_comment_hit or has_callsite or required_match):
                continue
            ranked.append(((exact_comment_hit, has_callsite, has_evidence, match_score + required_match), item))

        if not ranked:
            return []
        ranked.sort(key=lambda entry: entry[0], reverse=True)
        return [item for _, item in ranked[: min(2, self.max_method_contexts)]]

    def _build_method_evidence_packs(
        self,
        state: GraphState,
        retrieved_method_contexts: list[RetrievedMethodContext],
    ) -> list[dict[str, Any]]:
        packs: list[dict[str, Any]] = []
        for item in retrieved_method_contexts:
            local_call = self._extract_local_call_line(
                state["original_code"],
                item.method_name,
                existing_slice=item.callsite_slice,
            )
            packs.append(
                {
                    "method_name": item.method_name,
                    "path": item.path,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "local_call": local_call,
                    "callsite_slice": item.callsite_slice or self._extract_callsite_slice(state["original_code"], item.method_name),
                    "method_behavior": self._build_method_behavior(
                        source=item.source,
                        method_name=item.method_name,
                        local_call=local_call,
                    ),
                    "evidence_slice": item.evidence_slice or self._extract_relevant_definition_slice(item.source),
                    "match_score": item.match_score,
                    "confidence": item.confidence,
                    "confidence_label": item.confidence_label,
                }
            )
        return packs

    def _extract_signature_line(self, source: str) -> str:
        for line in (source or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("async def ") or stripped.startswith("class "):
                return stripped
        return ""

    def _extract_callsite_slice(self, original_code: str, method_name: str) -> str:
        lines = (original_code or "").splitlines()
        if not lines:
            return ""
        token = (method_name or "").split(".")[-1]
        token_pattern = re.compile(rf"\b{re.escape(token)}\b") if token else None
        hit_indexes = [
            idx
            for idx, line in enumerate(lines)
            if token_pattern and token_pattern.search(line)
        ]
        if not hit_indexes:
            return ""
        center = hit_indexes[0]
        start = max(0, center - 2)
        end = min(len(lines), center + 3)
        return "\n".join(lines[start:end]).strip()

    def _extract_local_call_line(self, original_code: str, method_name: str, existing_slice: str = "") -> str:
        lines = (original_code or "").splitlines()
        token = (method_name or "").split(".")[-1]
        token_pattern = re.compile(rf"\b{re.escape(token)}\b") if token else None
        for line in lines:
            stripped = line.rstrip()
            if token_pattern and token_pattern.search(stripped):
                return stripped.strip()
        for line in (existing_slice or "").splitlines():
            stripped = line.strip()
            if token_pattern and token_pattern.search(stripped):
                return stripped
        return ""

    def _build_method_behavior(self, source: str, method_name: str, local_call: str = "") -> str:
        lines = (source or "").splitlines()
        if not lines:
            return ""
        if len(lines) <= 8:
            return "\n".join(lines).strip()

        signature_index = 0
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("def ", "async def ", "class ")):
                signature_index = index
                break

        anchor_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", local_call or "")
            if token.lower() not in {"self", "cls", "super", "true", "false", "none"}
        }
        method_tail = (method_name or "").split(".")[-1].lower()
        if method_tail:
            anchor_tokens.add(method_tail)

        best_index = min(signature_index + 1, len(lines) - 1)
        best_score = -1
        keyword_tokens = ("return", "raise", "yield", "await", "if ", "elif ", "else:", "except", "finally:")
        for index in range(signature_index + 1, len(lines)):
            stripped = lines[index].strip()
            lowered = stripped.lower()
            score = 0
            if any(token in lowered for token in anchor_tokens):
                score += 4
            if any(token in lowered for token in keyword_tokens):
                score += 3
            if "self." in stripped and "=" in stripped:
                score += 2
            if "=" in stripped:
                score += 1
            if score > best_score:
                best_score = score
                best_index = index

        body_window = 8
        body_start = max(signature_index + 1, best_index - 2)
        for index in range(best_index, signature_index, -1):
            stripped = lines[index].strip().lower()
            if stripped.startswith(("if ", "elif ", "else:", "except", "finally:")):
                body_start = index
                break
        max_body_start = max(signature_index + 1, len(lines) - body_window)
        body_start = min(body_start, max_body_start)
        body_end = min(len(lines), body_start + body_window)
        if body_end - body_start < 4 and len(lines) - (signature_index + 1) >= 4:
            body_end = min(len(lines), signature_index + 1 + 4)
            body_start = signature_index + 1

        snippet_lines = [lines[signature_index]]
        snippet_lines.extend(lines[body_start:body_end])
        return "\n".join(snippet_lines).strip()

    def _extract_relevant_definition_slice(self, source: str) -> str:
        lines = (source or "").splitlines()
        if not lines:
            return ""
        if len(lines) <= 12:
            return "\n".join(lines[:10]).strip()
        selected: list[str] = []
        keywords = ("return", "raise", "yield", "await", "if ", "elif ", "else:", "except", "finally:", "config", "default", "none", "true", "false")
        for line in lines:
            stripped = line.strip()
            lowered = stripped.lower()
            if stripped.startswith(("def ", "async def ", "class ")):
                selected.append(line)
                continue
            if "self." in stripped and "=" in stripped:
                selected.append(line)
                continue
            if any(token in lowered for token in keywords):
                selected.append(line)
            if len(selected) >= 6:
                break
        if not selected:
            selected = lines[:10]
        return "\n".join(selected[:6]).strip()

    def _synthesize_edit_constraints(
        self,
        *,
        state: GraphState,
        method_inquiry: MethodInquiryResult,
        retrieved_method_contexts: list[RetrievedMethodContext],
    ) -> list[EditConstraint]:
        constraints: list[EditConstraint] = []
        seen: set[str] = set()
        satd_comment = str(state.get("satd_comment") or "").strip()
        inquiry_lookup = {
            (item.normalized_name or self._normalize_method_name(item.name)): item
            for item in (method_inquiry.uncertainty_items or [])
            if (item.normalized_name or self._normalize_method_name(item.name))
        }
        for method_name in method_inquiry.required_methods[:2]:
            normalized = self._normalize_method_name(method_name)
            item = inquiry_lookup.get(normalized)
            focus = normalized or method_name
            if not focus or focus in seen:
                continue
            seen.add(focus)
            target_hint = (
                item.source_excerpt
                if item and item.source_excerpt
                else focus
            )
            constraints.append(
                EditConstraint(
                    focus_point=focus,
                    required_fact=(
                        f"The current edit point must satisfy the SATD intent ({satd_comment}) while staying consistent with `{focus}`."
                    ),
                    must_do=f"Use `{focus}` only as local supporting evidence for the current edit point.",
                    must_not_do="Do not treat retrieved evidence as permission to redesign the surrounding function.",
                    supporting_point_kind="method",
                    target_kind="method",
                    target_hint=target_hint,
                    supporting_symbol=focus,
                    supporting_evidence=(item.source_excerpt if item else ""),
                    confidence=0.45,
                )
            )
        for context in retrieved_method_contexts[:1]:
            focus = self._normalize_method_name(context.method_name) or context.method_name
            if not focus or focus in seen:
                continue
            seen.add(focus)
            constraints.append(
                EditConstraint(
                    focus_point=focus,
                    required_fact=(
                        f"The current edit point must satisfy the SATD intent ({satd_comment}) while staying consistent with `{focus}`."
                    ),
                    must_do=f"Use `{focus}` only as local supporting evidence for the current edit point.",
                    must_not_do="Do not treat retrieved evidence as permission to redesign the surrounding function.",
                    supporting_point_kind="method",
                    target_kind="method",
                    target_hint=context.callsite_slice.splitlines()[2].strip() if context.callsite_slice and len(context.callsite_slice.splitlines()) >= 3 else focus,
                    supporting_symbol=focus,
                    supporting_evidence=context.evidence_slice or context.callsite_slice,
                    confidence=context.confidence or 0.45,
                )
            )
        if not constraints:
            for item in method_inquiry.uncertainty_items or []:
                focus = item.name
                if not focus or focus in seen or item.kind not in {"condition", "return", "state_write", "value"}:
                    continue
                seen.add(focus)
                constraints.append(
                    EditConstraint(
                        focus_point=focus,
                        required_fact=item.why_this_matters or f"Preserve the local semantics around {focus}.",
                        must_do="Stay with the smallest local edit near the SATD comment.",
                        must_not_do="Do not introduce unrelated structural rewrites.",
                        supporting_point_kind=item.kind,
                        target_kind=item.kind,
                        target_hint=item.source_excerpt or focus,
                        supporting_symbol=item.normalized_name or focus,
                        supporting_evidence=item.source_excerpt,
                        confidence=0.45,
                    )
                )
                if len(constraints) >= 2:
                    break
        return constraints[:3]

    def resolve_method_locations(self, state: GraphState, method_names: list[str]) -> dict[str, list[str]]:
        if not method_names:
            return {}
        file_payload = self.client.toolbox.fetch_repo_file(
            state["user"],
            state["project"],
            state["file_path"],
            ref=(state.get("commit") or "").strip() or None,
        )
        file_content = file_payload.get("full_content") or "" if file_payload.get("ok") else ""
        tree_payload = self.client.toolbox.fetch_repo_tree(
            state["user"],
            state["project"],
            ref=(state.get("commit") or "").strip() or None,
        )
        repo_paths = [
            str(item.get("path") or "")
            for item in tree_payload.get("entries", [])
            if item.get("type") == "file"
        ]
        hints = self.client.toolbox.infer_method_path_hints(
            current_path=state["file_path"],
            method_names=method_names,
            current_content=file_content,
            repo_paths=repo_paths,
        )
        return hints

    def _log(self, state: GraphState, message: str) -> None:
        if not self.logger:
            return
        self.logger(f"{self._task_prefix(state)} {message}")

    def _format_method_list(self, method_names: list[str], empty: str = "none") -> str:
        items = [str(item).strip() for item in method_names if str(item).strip()]
        if not items:
            return empty
        if len(items) <= 3:
            return ", ".join(items)
        return ", ".join(items[:3]) + f", ... (+{len(items) - 3})"

    def _checkpoint(self, state: GraphState, stage: str, payload: dict[str, Any]) -> None:
        if not self.checkpoint_callback:
            return
        self.checkpoint_callback(state, stage, payload)

    def _task_prefix(self, state: GraphState) -> str:
        task_id = state.get("task_id", "?")
        task_index = state.get("task_index") or 0
        task_total = state.get("task_total") or 0
        if task_index and task_total:
            return f"[task {task_id} {task_index}/{task_total}]"
        return f"[task {task_id}]"

    def _build_repair_prompts(
        self,
        *,
        state: GraphState,
        round_id: int,
        method_inquiry: MethodInquiryResult,
        retrieved_method_contexts: list[RetrievedMethodContext],
        missing_method_names: list[str],
        edit_constraints: list[EditConstraint],
    ) -> tuple[str, str]:
        repair_feedback = state.get("repair_feedback") or {}
        reviewer_feedback_block = self._format_reviewer_feedback_block(repair_feedback)
        candidate_mode = str(state.get("candidate_mode") or "single")
        method_context_block = "[none]"
        if self._candidate_uses_method_context(candidate_mode):
            method_context_block = self._format_method_context_block(
                state=state,
                method_inquiry=method_inquiry,
                retrieved_method_contexts=retrieved_method_contexts,
                missing_method_names=missing_method_names,
            )
        if method_context_block == "[none]":
            return self._build_no_context_repair_prompts(
                state=state,
                reviewer_feedback_block=reviewer_feedback_block,
            )
        return self._build_generic_repair_prompts(
            state=state,
            method_context_block=method_context_block,
            edit_constraints=edit_constraints,
            reviewer_feedback_block=reviewer_feedback_block,
        )

    def _build_no_context_repair_prompts(
        self,
        *,
        state: GraphState,
        reviewer_feedback_block: str = "",
    ) -> tuple[str, str]:
        system_prompt = (
            "You are the fixer agent in a SATD repair workflow. "
        )
        user_prompt = (
            "Repair the code according to the SATD comment. Respond in JSON with key repaired_code.\n\n"
            f"### SATD comment:\n{state['satd_comment']}\n\n"
            f"Code:\n{self._prompt_code_block(state['original_code'])}\n\n"
            "- Use normal indentation in repaired_code; do not add tab characters, column-alignment padding, or excessive whitespace.\n"
            "- When the SATD requests deleting or removing something, delete the corresponding executable code or statement, not only the SATD comment.\n"
            f"{reviewer_feedback_block}"
        )
        return system_prompt, user_prompt

    def _build_generic_repair_prompts(
        self,
        *,
        state: GraphState,
        method_context_block: str,
        edit_constraints: list[EditConstraint],
        reviewer_feedback_block: str,
    ) -> tuple[str, str]:
        system_prompt = (
            "You are the fixer agent in a SATD repair workflow. "
            "Return valid JSON only with key: repaired_code. "
        )
        hard_rules = self._format_generic_constraint_block(edit_constraints)
        user_prompt = (
            "Repair the code according to the SATD comment. Respond in JSON with key repaired_code.\n\n"
            f"### SATD comment:\n{state['satd_comment']}\n\n"
            f"### Code:\n{self._prompt_code_block(state['original_code'])}\n\n"
            f"### Supporting evidence:\n{method_context_block}\n\n"
            f"### Hard rules:\n{hard_rules}\n"
            "- Use supporting evidence only to validate the smallest local repair.\n"
            "- Use normal indentation in repaired_code; do not add tab characters, column-alignment padding, or excessive whitespace.\n"
            "- When the SATD requests deleting or removing something, delete the corresponding executable code or statement, not only the SATD comment.\n"
            "- Edit only the smallest local block nearest to the SATD comment.\n"
            f"{reviewer_feedback_block}"
        )
        return system_prompt, user_prompt

    def _prompt_code_block(self, code: str) -> str:
        lines = (code or "").expandtabs(4).splitlines()
        return "\n".join(line.rstrip() for line in lines)

    def _repair_max_tokens(self) -> int:
        raw = os.environ.get("OPENAI_REPAIR_MAX_TOKENS")
        if raw:
            try:
                return max(512, int(raw))
            except ValueError:
                pass
        return 4096

    def _default_repair_confidence(
        self,
        state: GraphState,
        repaired_code: str,
        found_method_contexts: list[RetrievedMethodContext],
    ) -> float:
        if preprocess_python_code(state["original_code"]) == preprocess_python_code(repaired_code):
            return 0.35
        if found_method_contexts:
            return 0.60
        return 0.50

    def _default_repair_plan(self, state: GraphState) -> str:
        comment = " ".join(str(state.get("satd_comment") or "").split())
        if not comment:
            return "Apply the smallest local repair required by the SATD."
        if len(comment) > 160:
            comment = comment[:157] + "..."
        return f"Apply the smallest local repair required by the SATD comment: {comment}"

    def _infer_changed_scope(self, original_code: str, repaired_code: str) -> str:
        if (original_code or "") == (repaired_code or ""):
            return "line"
        original_lines = (original_code or "").splitlines()
        repaired_lines = (repaired_code or "").splitlines()
        changed_line_count = abs(len(original_lines) - len(repaired_lines))
        for before, after in zip(original_lines, repaired_lines):
            if before != after:
                changed_line_count += 1
        if changed_line_count <= 2:
            return "line"
        if changed_line_count <= 12:
            return "function"
        if changed_line_count <= 40:
            return "class"
        return "file"

    def _format_method_context_block(
        self,
        *,
        state: GraphState,
        method_inquiry: MethodInquiryResult,
        retrieved_method_contexts: list[RetrievedMethodContext],
        missing_method_names: list[str],
    ) -> str:
        if not retrieved_method_contexts:
            return "[none]"
        lines: list[str] = []
        note_lookup: dict[str, str] = {}
        for item in method_inquiry.method_notes or []:
            if not isinstance(item, dict):
                continue
            method_name = self._normalize_method_name(item.get("method_name"))
            if not method_name:
                continue
            selection_reason = str(item.get("why_blocking") or item.get("what_to_learn") or "").strip()
            if selection_reason:
                note_lookup[method_name] = selection_reason
        for pack in self._build_method_evidence_packs(state, retrieved_method_contexts[:2]):
            method_name = str(pack["method_name"])
            lines.append(f"- method: {method_name}")
            if pack.get("local_call"):
                lines.append("  local call:")
                lines.append(f"    {pack['local_call']}")
            if pack.get("method_behavior"):
                lines.append("  method behavior:")
                for line in str(pack["method_behavior"]).splitlines():
                    lines.append(f"    {line}")
            selection_reason = note_lookup.get(self._normalize_method_name(method_name))
            if selection_reason:
                lines.append("  selection reason:")
                lines.append(f"    {selection_reason}")
        return "\n".join(lines)

    def _format_generic_constraint_block(self, edit_constraints: list[EditConstraint]) -> str:
        lines = [
            "- Make the smallest plausible local edit.",
            "- Preserve the existing function/class signature unless the SATD explicitly asks for a signature-local fix.",
            "- Do not add new helpers, new control flow, or unrelated rewrites.",
            "- Keep unchanged lines unchanged whenever possible.",
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

    def _extract_comment_reference_names(self, comment: str) -> set[str]:
        refs: set[str] = set()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_\.]+", comment or ""):
            normalized = self._normalize_method_name(token)
            if normalized:
                refs.add(normalized)
        return refs

    def _format_reviewer_feedback_block(self, repair_feedback: dict[str, Any]) -> str:
        if not repair_feedback:
            return ""
        repair_constraints = [str(item).strip() for item in repair_feedback.get("repair_constraints", []) if str(item).strip()]
        retry_hint = " ".join(str(repair_feedback.get("retry_hint") or "").split())
        lines = [
            "### Reviewer feedback from previous attempt:",
            "Use this only to avoid the previous failed pattern; do not treat it as a new requirement.",
            "The SATD comment and retrieved method context remain the primary repair source.",
        ]
        if repair_constraints:
            lines.append(f"Repair constraints: {', '.join(repair_constraints[:4])}")
        if retry_hint:
            lines.append(f"Retry hint: {retry_hint}")
        return "\n".join(lines) + "\n\n"

    def _em_risk_notes(self, original_code: str, repaired_code: str) -> list[str]:
        risks = []
        original_signature = self._first_signature_line(original_code)
        repaired_signature = self._first_signature_line(repaired_code)
        if original_signature and repaired_signature and original_signature != repaired_signature:
            risks.append("signature_changed")
        if self._count_regex_matches(r"^\s*(?:async\s+def|def|class)\s+", repaired_code) > self._count_regex_matches(r"^\s*(?:async\s+def|def|class)\s+", original_code):
            risks.append("new_helper_or_definition")
        if self._count_regex_matches(r"\breturn\b", repaired_code) > self._count_regex_matches(r"\breturn\b", original_code):
            risks.append("new_return_path")
        return risks

    def _first_signature_line(self, code: str) -> str:
        for line in (code or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("async def ") or stripped.startswith("class "):
                return stripped
        return ""

    def _count_regex_matches(self, pattern: str, code: str) -> int:
        return len(re.findall(pattern, code or "", flags=re.MULTILINE))


class OpenAIReviewer:
    ALLOWED_FAILED_CHECKS = {
        "syntax_error",
        "empty_repair",
        "no_effective_change",
        "comment_only_without_satd_support",
        "unsupported_signature_change",
        "new_helper_without_evidence",
        "unsupported_return_path",
        "unsupported_exception_path",
        "unsupported_control_flow_change",
        "over_scoped_change",
        "over_expanded_change",
        "anchor_not_modified",
        "unrelated_change",
        "not_addressing_satd",
        "semantic_drift_risk",
        "context_conflict",
        "reviewer_uncertain",
    }
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
    HARD_FAIL_CHECKS = {
        "syntax_error",
        "empty_repair",
        "no_effective_change",
    }
    FINAL_BLOCKING_CHECKS = {
        *HARD_FAIL_CHECKS,
        "comment_only_without_satd_support",
        "anchor_not_modified",
        "unrelated_change",
        "unsupported_control_flow_change",
        "over_expanded_change",
        "not_addressing_satd",
        "context_conflict",
    }
    FINAL_NONBLOCKING_CHECKS = {
        "over_scoped_change",
        "semantic_drift_risk",
        "reviewer_uncertain",
    }
    CHECK_CONSTRAINTS = {
        "syntax_error": ["return_valid_python_code"],
        "empty_repair": ["return_non_empty_repaired_code"],
        "no_effective_change": ["modify_code_to_address_satd"],
        "comment_only_without_satd_support": ["avoid_comment_only_change_unless_satd_requests_it"],
        "unsupported_signature_change": ["preserve_original_signature"],
        "new_helper_without_evidence": ["avoid_new_helpers"],
        "unsupported_return_path": ["avoid_new_return_paths"],
        "unsupported_exception_path": ["avoid_new_exception_paths"],
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

        structural_evidence = self._structural_evidence_payload(state, repair)
        local_failed_checks = self._local_failed_checks(state, repair)
        local_constraints = self._constraints_for_checks(local_failed_checks)
        structural_risk_checks = [] if local_failed_checks else self._structural_risk_checks(state, repair, structural_evidence)
        payload: dict[str, Any] = {}

        if not local_failed_checks:
            system_prompt, user_prompt = self._build_checklist_prompts(state, repair)
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

        protected_anchor_alignment = self._has_satd_anchor_protection(structural_evidence)
        anchor_supported = self._has_satd_anchor_support(structural_evidence)
        instruction_aligned = self._has_instruction_aligned_support(structural_evidence)
        if anchor_supported or instruction_aligned:
            structural_risk_checks = self._clear_supported_risks(structural_risk_checks, structural_evidence)
            llm_failed_checks = self._clear_supported_risks(llm_failed_checks, structural_evidence)
            llm_constraints = [
                item
                for item in llm_constraints
                if item not in {"make_one_direct_change_matching_satd", "modify_satd_anchor_region"}
            ]
        structural_risk_checks = self._clear_inapplicable_risks(state, structural_risk_checks, structural_evidence)
        llm_failed_checks = self._clear_inapplicable_risks(state, llm_failed_checks, structural_evidence)

        failed_checks = self._dedupe_limit([*local_failed_checks, *structural_risk_checks, *llm_failed_checks], limit=3)
        if self._reviewer_uncertain_is_viable(repair, failed_checks):
            failed_checks = []
            llm_constraints = []
            llm_retry_hint = ""
        repair_constraints = self._dedupe_limit(
            [
                *local_constraints,
                *llm_constraints,
                *self._constraints_for_checks(failed_checks),
            ],
            limit=4,
        )
        approved = not failed_checks and (approved_by_llm or protected_anchor_alignment or anchor_supported or instruction_aligned or repair.round_id >= 2)
        failure_anchor = "" if approved else self._failure_anchor(failed_checks)
        compat_score = 1.0 if approved else 0.0
        revision_advice = "" if approved else "constraints:" + ",".join(repair_constraints)
        rationale = "approved" if approved else f"failed_checks:{','.join(failed_checks)}"
        retry_hint = "" if approved else self._retry_hint_for_repair(
            failed_checks=failed_checks,
            evidence=structural_evidence,
            llm_retry_hint=llm_retry_hint,
        )

        return ReviewResult(
            round_id=repair.round_id,
            approved=approved,
            review_score=compat_score,
            problem_alignment=compat_score,
            minimality=compat_score,
            semantic_preservation=compat_score,
            internal_consistency=compat_score,
            issues=failed_checks,
            revision_advice=revision_advice,
            reject_type=failure_anchor or None,
            rationale=rationale,
            softened_gate_used=False,
            candidate_mode=getattr(repair, "candidate_mode", "single"),
            failed_checks=failed_checks,
            repair_constraints=repair_constraints,
            failure_anchor=failure_anchor,
            retry_hint=retry_hint,
        )

    def _build_checklist_prompts(self, state: GraphState, repair: RepairAttempt) -> tuple[str, str]:
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
        context_summary = self._repair_context_summary(state)
        structural_evidence = self._structural_evidence(state, repair)
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
            "Do not use not_addressing_satd when the diff removes or directly edits the SATD anchor region, "
            "especially an if False, temporary, hack, obsolete, TODO, FIXME, or XXX block.\n"
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
            f"Structural evidence:\n{structural_evidence}\n\n"
            f"Repair context:\n{context_summary}\n"
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
        if not required_methods and not retrieved and not missing:
            repair_context = ((state.get("github_context") or {}).get("repair_context") or {}) if isinstance(state.get("github_context"), dict) else {}
            method_inquiry_payload = repair_context.get("method_inquiry", {}) if isinstance(repair_context, dict) else {}
            required_methods = list(method_inquiry_payload.get("required_methods") or [])
            retrieved = list(repair_context.get("retrieved_methods") or [])
            missing = list(repair_context.get("missing_method_names") or [])

        lines = [
            f"required_methods: {', '.join(str(item) for item in required_methods[:5]) or '[none]'}",
            f"missing_method_names: {', '.join(str(item) for item in missing[:5]) or '[none]'}",
            f"retrieved_method_count: {len(retrieved)}",
        ]
        for item in retrieved[:2]:
            if isinstance(item, RetrievedMethodContext):
                method_name = item.method_name
                path = item.path
                source = item.source
            elif isinstance(item, dict):
                method_name = str(item.get("method_name") or "")
                path = str(item.get("path") or "")
                source = str(item.get("source") or "")
            else:
                continue
            lines.append(f"- method: {method_name} path: {path}")
            if source:
                lines.append(self._truncate(source, 1200))
        return "\n".join(lines)

    def _local_failed_checks(self, state: GraphState, repair: RepairAttempt) -> list[str]:
        original_code = state["original_code"] or ""
        repaired_code = repair.repaired_code or ""
        checks: list[str] = []
        if not repaired_code.strip():
            checks.append("empty_repair")
            return checks
        if not self._syntax_valid(repaired_code):
            checks.append("syntax_error")
            return checks

        if original_code.strip() == repaired_code.strip() or self._fallback_or_noop_repair(repair):
            checks.append("no_effective_change")

        return self._dedupe_limit(checks, limit=3)

    def _structural_evidence(self, state: GraphState, repair: RepairAttempt) -> str:
        return json.dumps(self._structural_evidence_payload(state, repair), ensure_ascii=False)

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
        instruction_evidence = self._satd_instruction_evidence(state["satd_comment"], original_code, repaired_code, anchor_evidence)
        changed_ratio = changed_line_count / max(1, original_line_count)
        structural_delta = {
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
        return structural_delta

    def _structural_risk_checks(self, state: GraphState, repair: RepairAttempt, evidence: dict[str, Any]) -> list[str]:
        checks: list[str] = []
        anchor_supported = self._has_satd_anchor_support(evidence)
        satd_comment = state.get("satd_comment") or ""

        if evidence.get("comment_or_format_only_change") and not evidence.get("satd_mentions_comment_or_documentation"):
            checks.append("comment_only_without_satd_support")

        if (
            evidence.get("satd_anchor_found")
            and evidence.get("raw_code_changed")
            and not anchor_supported
            and not evidence.get("changed_near_satd_anchor")
        ):
            checks.append("anchor_not_modified")

        if self._has_unrelated_change_risk(evidence):
            checks.append("unrelated_change")

        if self._has_over_expanded_change_risk(state, evidence):
            checks.append("over_expanded_change")

        if self._has_unsupported_control_flow_risk(satd_comment, evidence, anchor_supported):
            checks.append("unsupported_control_flow_change")

        if (
            evidence.get("signature_changed")
            and not self._signature_change_allowed(satd_comment, state["original_code"], repair.repaired_code)
            and not self._has_instruction_aligned_support(evidence)
        ):
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
        if evidence.get("fallback_or_noop_repair") or not evidence.get("raw_code_changed"):
            return False
        return bool(evidence.get("instruction_aligned_change"))

    def _clear_supported_risks(self, checks: list[str], evidence: dict[str, Any]) -> list[str]:
        clearable = {
            "anchor_not_modified",
            "not_addressing_satd",
            "over_scoped_change",
            "semantic_drift_risk",
            "reviewer_uncertain",
        }
        if evidence.get("instruction_aligned_change"):
            clearable.update(
                {
                    "unsupported_signature_change",
                    "unsupported_control_flow_change",
                    "unrelated_change",
                }
            )
        cleared: list[str] = []
        for check in checks:
            if check in clearable:
                continue
            if check == "unrelated_change" and self._anchor_edit_can_explain_scope(evidence):
                continue
            cleared.append(check)
        return cleared

    def _clear_inapplicable_risks(self, state: GraphState, checks: list[str], evidence: dict[str, Any]) -> list[str]:
        cleared: list[str] = []
        for check in checks:
            if check == "over_expanded_change" and not self._has_over_expanded_change_risk(state, evidence):
                continue
            cleared.append(check)
        return cleared

    def _anchor_edit_can_explain_scope(self, evidence: dict[str, Any]) -> bool:
        if evidence.get("dead_or_temporary_block_removed"):
            return True
        return bool(
            evidence.get("comment_or_format_only_change")
            and evidence.get("satd_mentions_comment_or_documentation")
        )

    def _reviewer_uncertain_is_viable(self, repair: RepairAttempt, failed_checks: list[str]) -> bool:
        return repair.round_id >= 2 and failed_checks == ["reviewer_uncertain"]

    def _has_unrelated_change_risk(self, evidence: dict[str, Any]) -> bool:
        if (
            not evidence.get("raw_code_changed")
            or evidence.get("dead_or_temporary_block_removed")
            or evidence.get("instruction_aligned_change")
        ):
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
        satd_comment = state.get("satd_comment") or ""
        if self._satd_mentions_comment_or_documentation(satd_comment):
            return False
        added_line_count = int(evidence.get("added_line_count") or 0)
        original_line_count = max(1, int(evidence.get("original_line_count") or 0))
        repaired_line_count = int(evidence.get("repaired_line_count") or 0)
        line_count_delta = int(evidence.get("line_count_delta") or 0)
        if line_count_delta <= 0 or repaired_line_count <= original_line_count:
            return False
        added_ratio = added_line_count / original_line_count
        structure_expanded = bool(
            int(evidence.get("definitions_delta") or 0) > 0
            or int(evidence.get("imports_delta") or 0) > 0
            or int(evidence.get("branches_delta") or 0) >= 2
            or (int(evidence.get("returns_delta") or 0) + int(evidence.get("raises_delta") or 0)) >= 2
        )
        if added_line_count >= 10:
            return True
        if line_count_delta >= 8:
            return True
        if added_line_count >= 6 and added_ratio >= 0.50:
            return True
        if added_line_count >= 5 and repaired_line_count >= int(original_line_count * 1.6):
            return True
        return added_line_count >= 4 and structure_expanded and not evidence.get("instruction_aligned_change")

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
        retrieved = state.get("retrieved_method_contexts") or []
        return not bool(retrieved)

    def _satd_anchor_evidence(self, satd_comment: str, original_code: str, repaired_code: str) -> dict[str, Any]:
        original_lines = (original_code or "").splitlines()
        repaired_lines = (repaired_code or "").splitlines()
        changed_indices, deleted_indices = self._changed_original_line_indices(original_lines, repaired_lines)
        anchor_indices = [idx for idx, line in enumerate(original_lines) if self._line_matches_satd_anchor(line, satd_comment)]
        touched_anchor_indices = [idx for idx in anchor_indices if idx in changed_indices]
        removed_anchor_indices = [idx for idx in anchor_indices if idx in deleted_indices]
        deleted_lines = [original_lines[idx] for idx in sorted(deleted_indices) if 0 <= idx < len(original_lines)]
        touched_anchor_lines = [original_lines[idx] for idx in touched_anchor_indices[:3]]
        nearest_changed_distance = self._nearest_index_distance(anchor_indices, changed_indices)
        non_satd_deleted_executable_lines = [
            original_lines[idx]
            for idx in sorted(deleted_indices)
            if idx not in set(anchor_indices) and 0 <= idx < len(original_lines) and self._is_executable_line(original_lines[idx])
        ]
        satd_block_removed = bool(removed_anchor_indices and len(deleted_indices) > 1)
        dead_or_temporary_block_removed = bool(
            satd_block_removed
            and any(self._line_has_dead_or_temporary_marker(line) for line in deleted_lines + touched_anchor_lines)
        )
        return {
            "satd_anchor_found": bool(anchor_indices),
            "satd_anchor_changed": bool(touched_anchor_indices),
            "satd_anchor_removed": bool(removed_anchor_indices),
            "satd_block_removed": satd_block_removed,
            "dead_or_temporary_block_removed": dead_or_temporary_block_removed,
            "deleted_line_count": len(deleted_indices),
            "non_satd_deleted_executable_line_count": len(non_satd_deleted_executable_lines),
            "nearest_changed_distance_to_satd_anchor": nearest_changed_distance,
            "changed_near_satd_anchor": nearest_changed_distance is not None and nearest_changed_distance <= 2,
            "satd_anchor_line_sample": self._truncate("\n".join(touched_anchor_lines), 240),
        }

    def _satd_instruction_evidence(
        self,
        satd_comment: str,
        original_code: str,
        repaired_code: str,
        anchor_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        instruction_types = self._satd_instruction_types(satd_comment)
        satd_tokens = self._satd_code_tokens(satd_comment)
        changed_text = self._changed_text(original_code, repaired_code)
        changed_tokens = self._code_tokens(changed_text)
        overlap = sorted(satd_tokens & changed_tokens)
        local_support = bool(
            anchor_evidence.get("satd_anchor_changed")
            or anchor_evidence.get("satd_anchor_removed")
            or anchor_evidence.get("satd_block_removed")
            or anchor_evidence.get("dead_or_temporary_block_removed")
            or anchor_evidence.get("changed_near_satd_anchor")
        )
        symbol_support = bool(overlap)
        signature_change_kind = self._signature_change_kind(original_code, repaired_code)
        signature_support = signature_change_kind != "none" and bool(
            {"rename", "replace", "change", "override", "use_instead", "remove", "cleanup"} & instruction_types
        )
        control_flow_support = bool(
            {"change", "replace", "enable", "uncomment", "remove", "cleanup", "hack_cleanup", "use_instead"} & instruction_types
        ) and (local_support or symbol_support)
        instruction_aligned = bool(
            instruction_types
            and (
                local_support
                or symbol_support
                or signature_support
                or control_flow_support
            )
        )
        return {
            "satd_instruction_types": sorted(instruction_types),
            "satd_code_tokens": sorted(satd_tokens)[:12],
            "changed_code_token_overlap": overlap[:12],
            "changed_text_sample": self._truncate(changed_text, 360),
            "instruction_local_support": local_support,
            "instruction_symbol_support": symbol_support,
            "instruction_signature_support": signature_support,
            "instruction_control_flow_support": control_flow_support,
            "instruction_aligned_change": instruction_aligned,
        }

    def _satd_instruction_types(self, satd_comment: str) -> set[str]:
        comment = (satd_comment or "").strip().lower()
        types: set[str] = set()
        patterns = {
            "replace": (r"\breplace\b", r"\breplace\s+.+\s+with\b", r"\binstead\s+of\b"),
            "change": (r"\bchange\b", r"\bconvert\b", r"\bswitch\b", r"\buse\b"),
            "rename": (r"\brename\b", r"\bchange\s+name\b"),
            "enable": (r"\benable\b", r"\bre-enable\b", r"\bactivate\b"),
            "uncomment": (r"\buncomment\b",),
            "remove": (r"\bremove\b", r"\bdelete\b", r"\bdrop\b"),
            "cleanup": (r"\bcleanup\b", r"\bclean\s+up\b", r"\bobsolete\b", r"\bdeprecated\b"),
            "hack_cleanup": (r"\bhack\b", r"\bworkaround\b", r"\btemporary\b", r"\btemporarily\b"),
            "override": (r"\boverride\b",),
            "use_instead": (r"\bcall\b", r"\binstead\b", r"\buse\s+.+\binstead\b"),
            "exception_change": (r"\bvalueerror\b", r"\btypeerror\b", r"\bexception\b", r"\berror\b"),
        }
        for kind, regexes in patterns.items():
            if any(re.search(pattern, comment) for pattern in regexes):
                types.add(kind)
        return types

    def _satd_code_tokens(self, satd_comment: str) -> set[str]:
        flattened = re.findall(r"[A-Za-z_][A-Za-z0-9_\.]*", satd_comment or "")
        backtick_tokens = re.findall(r"`([^`]+)`", satd_comment or "")
        flattened.extend(backtick_tokens)
        tokens: set[str] = set()
        stopwords = {
            "todo",
            "fixme",
            "xxx",
            "this",
            "that",
            "with",
            "from",
            "into",
            "once",
            "maybe",
            "instead",
            "because",
            "temporary",
            "temporarily",
            "remove",
            "delete",
            "change",
            "replace",
            "rename",
            "enable",
            "uncomment",
            "call",
            "use",
            "hack",
            "workaround",
            "month",
            "added",
        }
        for token in flattened:
            for part in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(token)):
                lowered = part.lower()
                if len(lowered) < 3 or lowered in stopwords:
                    continue
                tokens.add(lowered)
        return tokens

    def _code_tokens(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text or "")
            if len(token) >= 3
        }

    def _changed_text(self, original_code: str, repaired_code: str) -> str:
        original_lines = (original_code or "").splitlines()
        repaired_lines = (repaired_code or "").splitlines()
        parts: list[str] = []
        matcher = difflib.SequenceMatcher(a=original_lines, b=repaired_lines)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            parts.extend(original_lines[i1:i2])
            parts.extend(repaired_lines[j1:j2])
        return "\n".join(parts)

    def _nearest_index_distance(self, anchors: list[int], changed: set[int]) -> int | None:
        if not anchors or not changed:
            return None
        return min(abs(anchor - changed_index) for anchor in anchors for changed_index in changed)

    def _is_executable_line(self, line: str) -> bool:
        stripped = str(line or "").strip()
        return bool(stripped and not stripped.startswith("#") and not re.match(r"^[rubfRUBF]*['\"]{3}", stripped))

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
        line_norm = self._normalize_anchor_text(line)
        comment_norm = self._normalize_anchor_text(satd_comment)
        if not comment_norm or not line_norm:
            return False
        if comment_norm in line_norm or line_norm in comment_norm:
            return True
        comment_tokens = [token for token in comment_norm.split() if len(token) >= 3]
        if not comment_tokens:
            return False
        overlap = sum(1 for token in comment_tokens if token in line_norm)
        return overlap >= min(3, len(comment_tokens))

    def _normalize_anchor_text(self, text: str) -> str:
        cleaned = str(text or "").lower()
        cleaned = re.sub(r"['\"`#:/\\()[\]{}.,;!?]+", " ", cleaned)
        cleaned = re.sub(r"\b(todo|fixme|xxx|hack|note|pyre_fixme|pyre-fixme)\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _line_has_dead_or_temporary_marker(self, line: str) -> bool:
        lowered = str(line or "").lower()
        markers = (
            "if false",
            "todo",
            "fixme",
            "xxx",
            "hack",
            "temporary",
            "workaround",
            "obsolete",
            "remove",
            "deprecated",
            "would we still need",
        )
        return any(marker in lowered for marker in markers)

    def _satd_mentions_comment_or_documentation(self, satd_comment: str) -> bool:
        comment = (satd_comment or "").strip().lower()
        markers = (
            "comment",
            "doc",
            "docs",
            "docstring",
            "document",
            "documentation",
            "description",
            "cleanup",
            "clean up",
            "obsolete",
            "remove todo",
            "delete todo",
            "drop todo",
            "remove fixme",
            "delete fixme",
            "drop fixme",
            "remove hack",
            "delete hack",
            "drop hack",
        )
        return any(marker in comment for marker in markers)

    def _satd_mentions_control_flow(self, satd_comment: str) -> bool:
        comment = (satd_comment or "").strip().lower()
        markers = (
            "return",
            "raise",
            "exception",
            "error",
            "fallback",
            "if ",
            "when ",
            "case",
            "branch",
            "skip",
            "break",
            "continue",
        )
        return any(marker in comment for marker in markers)

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
            if stripped.startswith("def ") or stripped.startswith("async def ") or stripped.startswith("class "):
                return stripped
        return ""

    def _signature_changed(self, original_code: str, repaired_code: str) -> bool:
        original_signature = self._first_signature_line(original_code)
        repaired_signature = self._first_signature_line(repaired_code)
        return bool(original_signature and repaired_signature and original_signature != repaired_signature)

    def _comment_only_change_allowed(self, satd_comment: str, original_code: str, repaired_code: str) -> bool:
        if preprocess_python_code(original_code) != preprocess_python_code(repaired_code):
            return False
        comment = (satd_comment or "").strip().lower()
        removal_markers = ("remove", "delete", "drop", "cleanup", "obsolete")
        comment_markers = ("todo", "fixme", "hack", "comment", "docstring")
        return any(marker in comment for marker in removal_markers) and any(marker in comment for marker in comment_markers)

    def _signature_change_allowed(self, satd_comment: str, original_code: str, repaired_code: str) -> bool:
        comment = (satd_comment or "").strip().lower()
        annotation_markers = ("type", "annotation", "typing", "mypy", "pyre", "return type")
        return any(marker in comment for marker in annotation_markers) and self._signature_change_kind(original_code, repaired_code) == "annotation_only"

    def _signature_change_kind(self, original_code: str, repaired_code: str) -> str:
        original_signature = self._first_signature_line(original_code)
        repaired_signature = self._first_signature_line(repaired_code)
        if not original_signature or not repaired_signature or original_signature == repaired_signature:
            return "none"
        original_skeleton = re.sub(r"\s+", "", re.sub(r":[^,)=]+", "", re.sub(r"->\s*[^:]+", "", original_signature)))
        repaired_skeleton = re.sub(r"\s+", "", re.sub(r":[^,)=]+", "", re.sub(r"->\s*[^:]+", "", repaired_signature)))
        return "annotation_only" if original_skeleton == repaired_skeleton else "shape_changed"

    def _changed_line_count(self, original_code: str, repaired_code: str) -> int:
        original_lines = (original_code or "").splitlines()
        repaired_lines = (repaired_code or "").splitlines()
        changed = abs(len(original_lines) - len(repaired_lines))
        for before, after in zip(original_lines, repaired_lines):
            if before != after:
                changed += 1
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

    def _normalize_failed_checks(self, raw: Any, allowed: set[str] | None = None) -> list[str]:
        if not isinstance(raw, list):
            raw = [raw] if raw else []
        allowed_checks = allowed or self.ALLOWED_FAILED_CHECKS
        checks = []
        for item in raw:
            tag = self._normalize_tag(item)
            if not tag:
                continue
            checks.append(tag if tag in allowed_checks else "reviewer_uncertain")
        return self._dedupe_limit(checks, limit=3)

    def _normalize_constraints(self, raw: Any) -> list[str]:
        if not isinstance(raw, list):
            raw = [raw] if raw else []
        return self._dedupe_limit([self._normalize_tag(item) for item in raw], limit=4)

    def _normalize_retry_hint(self, raw: Any) -> str:
        hint = " ".join(str(raw or "").split())
        if not hint:
            return ""
        if "```" in hint or "\n" in hint:
            return ""
        forbidden = ("replace with", "use this code", "new requirement", "ignore the satd")
        if any(token in hint.lower() for token in forbidden):
            return ""
        words = hint.split()
        if len(words) > 30:
            hint = " ".join(words[:30]).rstrip(" ,;:")
        return hint

    def _retry_hint_for_repair(
        self,
        *,
        failed_checks: list[str],
        evidence: dict[str, Any],
        llm_retry_hint: str = "",
    ) -> str:
        template_hint = self._template_retry_hint(failed_checks, evidence)
        if template_hint:
            return template_hint
        return llm_retry_hint or self._default_retry_hint(failed_checks)

    def _template_retry_hint(self, failed_checks: list[str], evidence: dict[str, Any]) -> str:
        if not failed_checks:
            return ""
        checks = set(failed_checks)
        anchor_found = bool(evidence.get("satd_anchor_found"))
        anchor_near = bool(evidence.get("changed_near_satd_anchor"))
        if "syntax_error" in checks:
            return "Return the complete repaired snippet as valid Python while preserving the original signature."
        if "empty_repair" in checks:
            return "Return the full repaired snippet, not an empty response."
        if "no_effective_change" in checks:
            return "Make one concrete local edit; do not return the original snippet unchanged."
        if "unsupported_signature_change" in checks:
            return "Preserve the original signature unless the SATD explicitly asks for annotation changes."
        if "context_conflict" in checks:
            return "Align the edit with retrieved method context and avoid contradicting it."
        if "anchor_not_modified" in checks:
            return "Move the edit to the SATD anchor region or the adjacent executable statement."
        if "comment_only_without_satd_support" in checks:
            if anchor_found:
                return "Edit code or docstring near the SATD; do not only rewrite the comment."
            return "Make a concrete code or docstring edit; do not only rewrite comments."
        if "unsupported_control_flow_change" in checks or "unsupported_return_path" in checks or "unsupported_exception_path" in checks:
            if anchor_near:
                return "Keep the original control flow shape; revise the existing statement near the SATD."
            return "Avoid new branches or return paths; prefer a smaller edit inside existing control flow."
        if "over_expanded_change" in checks:
            return "Remove added implementation detail and keep only the smallest local SATD-related edit."
        if "unrelated_change" in checks or "over_scoped_change" in checks:
            if anchor_found:
                return "Revert unrelated rewrites and make one local edit near the SATD anchor."
            return "Revert unrelated rewrites and make one focused local edit."
        if "new_helper_without_evidence" in checks:
            return "Avoid adding helpers; revise the existing local code instead."
        if "not_addressing_satd" in checks:
            if anchor_found:
                return "Make the next edit directly target the SATD anchor region."
            return "Make the next edit directly target the SATD request."
        if "semantic_drift_risk" in checks:
            return "Preserve existing behavior while making the smallest SATD-related edit."
        return ""

    def _default_retry_hint(self, failed_checks: list[str]) -> str:
        check = failed_checks[0] if failed_checks else "reviewer_uncertain"
        hints = {
            "syntax_error": "Return a complete valid Python snippet while preserving the original signature.",
            "empty_repair": "Return the full repaired snippet, not an empty response.",
            "no_effective_change": "Make one concrete local edit that directly addresses the SATD.",
            "comment_only_without_satd_support": "Do not only edit the comment; make the smallest change that directly addresses the SATD.",
            "unsupported_signature_change": "Preserve the original signature unless the SATD explicitly asks for annotation changes.",
            "new_helper_without_evidence": "Avoid adding helpers; revise the existing local code instead.",
            "unsupported_return_path": "Avoid adding new return paths; stay within the existing control flow.",
            "unsupported_exception_path": "Avoid adding new exception paths; stay within the existing control flow.",
            "unsupported_control_flow_change": "Avoid new branches or return paths; prefer a smaller edit inside existing control flow.",
            "over_scoped_change": "Make a smaller local edit and preserve unrelated lines.",
            "over_expanded_change": "Remove added implementation detail and keep only the smallest local SATD-related edit.",
            "anchor_not_modified": "Revise the code near the SATD anchor instead of changing unrelated logic elsewhere.",
            "unrelated_change": "Avoid unrelated rewrites; focus the edit on the SATD location.",
            "not_addressing_satd": "Make the next edit directly target the SATD request.",
            "semantic_drift_risk": "Preserve existing behavior while making the smallest SATD-related edit.",
            "context_conflict": "Align the edit with the retrieved method context and avoid contradicting it.",
            "reviewer_uncertain": "Make the smallest local edit that clearly addresses the SATD.",
        }
        return hints.get(check, hints["reviewer_uncertain"])

    def _normalize_tag(self, value: Any) -> str:
        tag = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
        return tag

    def _constraints_for_checks(self, failed_checks: list[str]) -> list[str]:
        constraints: list[str] = []
        for check in failed_checks:
            constraints.extend(self.CHECK_CONSTRAINTS.get(check, []))
        return self._dedupe_limit(constraints, limit=4)

    def _failure_anchor(self, failed_checks: list[str]) -> str:
        anchor_map = {
            "syntax_error": "syntax",
            "empty_repair": "noop",
            "no_effective_change": "noop",
            "comment_only_without_satd_support": "comment_only",
            "unsupported_signature_change": "signature",
            "new_helper_without_evidence": "helper",
            "unsupported_return_path": "control_flow",
            "unsupported_exception_path": "control_flow",
            "unsupported_control_flow_change": "control_flow",
            "over_scoped_change": "scope",
            "over_expanded_change": "scope",
            "anchor_not_modified": "alignment",
            "unrelated_change": "scope",
            "not_addressing_satd": "alignment",
            "semantic_drift_risk": "semantic",
            "context_conflict": "context",
            "reviewer_uncertain": "uncertain",
        }
        for check in failed_checks:
            if check in anchor_map:
                return anchor_map[check]
        return "review"

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

