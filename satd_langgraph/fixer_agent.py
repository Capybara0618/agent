from __future__ import annotations  

import os
import queue
import re
import threading
import time
from typing import Any

from .openai_client import OpenAICompatClient
from .schema import (
    EditConstraint,  
    GraphState,
    MethodInquiryResult,
    RepairAttempt,
    RepositoryEvidenceContext,
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
        repository_evidence_prompt_mode: str = "append",
        repair_response_format: str = "json",
        logger: Any | None = None,
        checkpoint_callback: Any | None = None,
    ) -> None:
        self.client = client
        self.repair_context_mode = repair_context_mode
        self.max_method_contexts = max(1, int(max_method_contexts))
        self.repository_evidence_prompt_mode = str(repository_evidence_prompt_mode or "append").strip().lower()
        self.repair_response_format = self._normalize_repair_response_format(repair_response_format)
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
        effective_mode = self._effective_mode(
            candidate_mode,
            contexts,
            has_repository_evidence=self._has_repository_evidence(state),
        )

        self._checkpoint(
            state,
            stage="generation_start",
            payload={
                "round_id": round_id,
                "candidate_mode": effective_mode,
                "required_methods": list(method_inquiry.required_methods),
                "retrieved_method_count": len(contexts),
                "missing_method_names": missing,
            },
        )

        system_prompt, user_prompt = self._build_prompts(state, method_inquiry, contexts, missing, effective_mode)
        if self.repair_response_format == "plain_text":
            raw_response = self._generate_plain_text(
                system_prompt,
                user_prompt,
                request_label=f"repair:task_{state['task_id']}:round_{round_id}:candidate_{effective_mode}",
                max_tokens=self._repair_max_tokens(),
            )
            repaired_code = self._extract_code(raw_response) or state["original_code"]
        else:
            payload = self.client.generate_json(
                system_prompt,
                user_prompt,
                request_label=f"repair:task_{state['task_id']}:round_{round_id}:candidate_{effective_mode}",
                max_tokens=self._repair_max_tokens(),
            )
            repaired_code = str(payload.get("repaired_code") or state["original_code"])

        repair = RepairAttempt(
            round_id=round_id,
            repair_plan=self._repair_plan(state),
            repaired_code=repaired_code,
            changed_scope=self._changed_scope(state["original_code"], repaired_code),
            confidence=self._confidence(state, repaired_code, contexts),
            notes=self._notes(method_inquiry, contexts, missing),
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

        system_prompt = self._system_prompt()
        context_block = self._context_block(method_inquiry, contexts, missing)
        repository_evidence = [
            item
            for item in (state.get("retrieved_repository_evidence") or [])
            if isinstance(item, RepositoryEvidenceContext)
        ]
        repository_block = self._repository_context_block(repository_evidence)
        repository_rules = self._repository_evidence_rules(repository_evidence)
        repository_guidance = str(state.get("repository_evidence_guidance") or "").strip() or "[none]"
        hard_rules = self._format_generic_constraint_block(
            state.get("edit_constraints") or [],
            allow_evidence_backed_signature_change=bool(repository_guidance != "[none]"),
        )
        optional_method_block = f"\n### Supporting method evidence:\n{context_block}\n" if context_block != "[none]" else ""
        optional_guidance_block = (
            f"\n### Evidence-backed repair guidance:\n{repository_guidance}\n"
            if repository_guidance != "[none]"
            else ""
        )
        user_prompt = (
            "How to update the following code to resolve the SATD?\n\n"
            f"### Code:\n{self._code_block(state['original_code'])}\n\n"
            f"### SATD comment:\n{state['satd_comment']}\n\n"
            f"### Repository-local evidence from the target commit:\n{repository_block}\n"
            f"{optional_method_block}"
            f"{optional_guidance_block}\n"
            "### Consider the following questions in your answer:\n"
            "Shortly explain how to resolve the SATD.\n"
            "Provide the updated code.\n\n"
            "Repair constraints:\n"
            f"{hard_rules}\n"
            "- Return the complete updated version of the entire Code block in repaired_code, not only changed lines.\n"
            "- Make the smallest local edit that resolves the SATD.\n"
            "- Use supporting evidence only to validate the smallest local repair.\n"
            f"{repository_rules}"
            "- Prefer direct sibling implementations, replacement APIs, and tests over broad definitions or generic examples.\n"
            "- Use normal indentation in repaired_code; do not add tab characters, column-alignment padding, or excessive whitespace.\n"
            "- When the SATD requests deleting or removing something, delete the corresponding executable code or statement, not only the SATD comment.\n"
            "- Do not add a nested helper function or new abstraction unless direct repository evidence shows that exact pattern.\n"
            "- Preserve unrelated behavior and control flow.\n"
            f"{self._response_instruction()}"
            f"{feedback_block}"
        )
        return system_prompt, user_prompt

    def _build_no_context_prompts(self, state: GraphState, feedback_block: str = "") -> tuple[str, str]:
        system_prompt = self._system_prompt()
        user_prompt = (
            "How to update the following code to resolve the SATD?\n\n"
            f"### Code:\n{self._code_block(state['original_code'])}\n\n"
            f"### SATD comment:\n{state['satd_comment']}\n\n"
            "### Consider the following questions in your answer:\n"
            "Shortly explain how to resolve the SATD.\n"
            "Provide the updated code.\n\n"
            "Repair constraints:\n"
            "- Return the complete updated version of the entire Code block in repaired_code, not only changed lines.\n"
            "- Make the smallest local edit that resolves the SATD.\n"
            "- Do not add a nested helper function or new abstraction unless the original code already requires that shape.\n"
            "- Preserve unrelated behavior and control flow.\n"
            "- If the SATD asks to remove a temporary workaround, remove only that workaround and keep the surrounding logic intact.\n"
            "- Use normal indentation in repaired_code; do not add tab characters, column-alignment padding, or excessive whitespace.\n"
            "- When the SATD requests deleting or removing something, delete the corresponding executable code or statement, not only the SATD comment.\n"
            f"{self._response_instruction()}"
            f"{feedback_block}"
        )
        return system_prompt, user_prompt

    def _system_prompt(self) -> str:
        prompt = "You are an expert software engineer specialized in technical debt refactoring."
        if self.repair_response_format == "json":
            prompt += " Return valid JSON only with key: repaired_code."
        return prompt

    def _response_instruction(self) -> str:
        if self.repair_response_format == "json":
            return "- Respond in JSON only with key repaired_code.\n"
        return "- In your final answer, include the complete updated code. Prefer a single Python code block.\n"

    def _format_generic_constraint_block(
        self,
        edit_constraints: list[EditConstraint],
        *,
        allow_evidence_backed_signature_change: bool = False,
    ) -> str:
        lines = [
            "- Make the smallest plausible local edit.",
            (
                "- Preserve the existing function/class signature unless the SATD explicitly asks for a signature-local fix "
                "or repository evidence-backed guidance requires one."
                if allow_evidence_backed_signature_change
                else "- Preserve the existing function/class signature unless the SATD explicitly asks for a signature-local fix."
            ),
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

    def _repository_context_block(self, items: list[RepositoryEvidenceContext]) -> str:
        if not items:
            return "[none]"
        lines: list[str] = []
        for item in items:
            lines.append(
                f"- {item.evidence_type}/{item.evidence_subtype} "
                f"support={item.support_level} score={item.score:.2f}"
            )
            if item.source_path:
                lines.append(f"  location: {item.source_path}:{item.span}")
            if item.why_relevant:
                lines.append(f"  why: {item.why_relevant}")
            source = str(item.content or "").strip()
            if source:
                lines.append("  source:")
                lines.extend(f"    {line}" for line in source.splitlines()[:60])
        return "\n".join(lines)

    def _repository_evidence_rules(self, items: list[RepositoryEvidenceContext]) -> str:
        if not items or self.repository_evidence_prompt_mode != "constrained":
            return ""
        return (
            "- Ground any new symbol, API replacement, or behavioral assumption in the repository evidence when possible.\n"
            "- Prefer evidence marked direct over supporting evidence; ignore weak evidence unless it confirms a local pattern.\n"
            "- When evidence-backed repair guidance lists concrete edit steps, implement the supported steps instead of only the smallest subset.\n"
            "- If the evidence does not justify a broader rewrite, keep the repair local instead of inventing a larger refactor.\n"
        )

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
            "The SATD comment and retrieved method context remain the primary repair source.",
        ]
        if constraints:
            lines.append("Repair constraints: " + ", ".join(constraints[:4]))
        if retry_hint:
            lines.append("Retry hint: " + retry_hint)
        return "\n".join(lines) + "\n"

    def _uses_context(self, candidate_mode: str) -> bool:
        mode = str(candidate_mode or "").strip().lower()
        if mode in {"baseline_guarded", "baseline_no_context"}:
            return False
        return not mode.endswith("no_context") and self.repair_context_mode in {"method_query", "clone_treesitter"}

    def _effective_mode(
        self,
        candidate_mode: str,
        contexts: list[RetrievedMethodContext],
        *,
        has_repository_evidence: bool = False,
    ) -> str:
        mode = str(candidate_mode or "").strip() or "single"
        if not self._uses_context(mode):
            return "baseline_guarded"
        if contexts or has_repository_evidence:
            return "evidence_guarded"
        return "baseline_guarded"

    def _has_repository_evidence(self, state: GraphState) -> bool:
        return any(
            isinstance(item, RepositoryEvidenceContext)
            for item in (state.get("retrieved_repository_evidence") or [])
        )

    def _repair_max_tokens(self) -> int:
        try:
            return max(512, int(os.environ.get("OPENAI_REPAIR_MAX_TOKENS") or 4096))
        except ValueError:
            return 4096

    def _normalize_repair_response_format(self, value: str | None) -> str:
        mode = str(value or "json").strip().lower()
        if mode not in {"json", "plain_text"}:
            raise ValueError("repair_response_format must be 'json' or 'plain_text'.")
        return mode

    def _generate_plain_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        request_label: str,
        max_tokens: int | None = None,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(self.client.max_attempts):
            started = time.time()
            self.client._emit_log(
                f"[llm] start label={request_label} attempt={attempt + 1}/{self.client.max_attempts} "
                f"model={self.client.model} timeout={self.client.request_timeout:.0f}s"
            )
            try:
                result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

                def invoke() -> None:
                    try:
                        request_kwargs: dict[str, Any] = {
                            "model": self.client.model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": 0.0,
                        }
                        if max_tokens is not None:
                            request_kwargs["max_tokens"] = max_tokens
                        result_queue.put(("ok", self.client.client.chat.completions.create(**request_kwargs)))
                    except Exception as exc:
                        result_queue.put(("error", exc))

                worker = threading.Thread(target=invoke, daemon=True)
                worker.start()
                status, payload = result_queue.get(timeout=self.client.request_timeout + 5)
                if status == "error":
                    raise payload
                elapsed = time.time() - started
                self.client._emit_log(f"[llm] success label={request_label} attempt={attempt + 1}/{self.client.max_attempts} elapsed={elapsed:.2f}s")
                return payload.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                self.client._emit_log(
                    f"[llm] error label={request_label} attempt={attempt + 1}/{self.client.max_attempts} "
                    f"type={type(exc).__name__} message={self.client._short_error(exc)}"
                )
                if attempt < self.client.max_attempts - 1:
                    time.sleep(2 * (attempt + 1))
        if last_error:
            raise last_error
        return ""

    def _extract_code(self, raw_response: str) -> str:
        python_fence = re.search(r"```python\s*(.*?)```", raw_response or "", flags=re.IGNORECASE | re.DOTALL)
        if python_fence:
            return python_fence.group(1).strip()
        any_fence = re.search(r"```\s*(.*?)```", raw_response or "", flags=re.DOTALL)
        if any_fence:
            return any_fence.group(1).strip()
        lines = (raw_response or "").splitlines()
        for index, line in enumerate(lines):
            if line.lstrip().startswith(("def ", "async def ", "class ")):
                return "\n".join(lines[index:]).strip()
        return (raw_response or "").strip()

    def _repair_plan(self, state: GraphState) -> str:
        comment = " ".join(str(state.get("satd_comment") or "").split())
        if len(comment) > 160:
            comment = comment[:157] + "..."
        return f"Apply the smallest local repair required by the SATD comment: {comment}"

    def _confidence(self, state: GraphState, repaired_code: str, contexts: list[RetrievedMethodContext]) -> float:
        if preprocess_python_code(state["original_code"]) == preprocess_python_code(repaired_code):
            return 0.35
        return 0.60 if contexts else 0.50

    def _notes(
        self,
        method_inquiry: MethodInquiryResult,
        contexts: list[RetrievedMethodContext],
        missing: list[str],
    ) -> str:
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
