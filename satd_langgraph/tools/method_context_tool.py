from __future__ import annotations

import re
import textwrap
from typing import Any

from .method_retriever import MethodRetrievalTool
from ..schema import (
    EditConstraint,
    GraphState,
    MethodInquiryResult,
    RetrievedMethodContext,
    UncertaintyItem,
)


class MethodContextTool:
    """Prepare method-level repository context for context-required SATD items."""

    def __init__(
        self,
        client: Any,
        method_retriever: MethodRetrievalTool,
        repair_context_mode: str = "clone_treesitter",
        max_method_contexts: int = 2,
        logger: Any | None = None,
        checkpoint_callback: Any | None = None,
    ) -> None:
        self.client = client
        self.method_retriever = method_retriever
        self.repair_context_mode = repair_context_mode
        self.max_method_contexts = max(1, int(max_method_contexts))
        self.logger = logger
        self.checkpoint_callback = checkpoint_callback

    def prepare_method_context(
        self,
        state: GraphState,
        candidate_mode: str = "baseline_context",
    ) -> tuple[
        MethodInquiryResult,
        list[RetrievedMethodContext],
        list[str],
        list[UncertaintyItem],
        list[EditConstraint],
        str,
    ]:
        round_id = int(state["round_id"]) + 1
        self._checkpoint(state, "question_start", {"round_id": round_id, "candidate_mode": candidate_mode})

        if not self._candidate_uses_method_context(candidate_mode):
            inquiry = MethodInquiryResult(reason="candidate_mode_without_method_context")
            self._checkpoint(
                state,
                "question_done",
                {"round_id": round_id, "candidate_mode": candidate_mode, "required_methods": []},
            )
            return inquiry, [], [], [], [], "[none]"

        inquiry = self.identify_required_methods(state)
        self._checkpoint(
            state,
            "question_done",
            {
                "round_id": round_id,
                "candidate_mode": candidate_mode,
                "required_methods": list(inquiry.required_methods),
                "reason": inquiry.reason,
                "method_notes": self._serialize_method_notes(inquiry.method_notes),
                "uncertainty_items": list(inquiry.uncertainty_items or []),
            },
        )

        self._checkpoint(
            state,
            "retrieval_start",
            {
                "round_id": round_id,
                "candidate_mode": candidate_mode,
                "required_methods": list(inquiry.required_methods),
            },
        )
        contexts = self._retrieve_method_contexts(state, inquiry.required_methods) if inquiry.required_methods else []
        found_contexts = [item for item in contexts if item.found][: self.max_method_contexts]
        missing_methods = [item.method_name for item in contexts if not item.found]
        block = self._format_method_context_block(inquiry, found_contexts)
        effective_mode = candidate_mode if found_contexts else self._no_context_fallback_mode(candidate_mode)

        self._checkpoint(
            state,
            "retrieval_done",
            {
                "round_id": round_id,
                "candidate_mode": effective_mode,
                "retrieved_method_contexts": self._serialize_retrieved_method_contexts(contexts),
                "missing_method_names": list(missing_methods),
            },
        )
        self._log(
            state,
            f"context results found={len(found_contexts)}/{len(inquiry.required_methods)} mode={effective_mode}",
        )
        return inquiry, found_contexts, missing_methods, [], [], block

    def identify_required_methods(self, state: GraphState) -> MethodInquiryResult:
        if self.repair_context_mode not in {"method_query", "clone_treesitter"}:
            return MethodInquiryResult(reason="repair_context_mode_disabled")

        candidates = self._method_candidates(state)
        candidate_block = self._format_candidates(candidates)
        system_prompt = textwrap.dedent(
            """
            You will repair this SATD in the next step.
            Before that, select only the methods, functions, or classes that are strongly relevant to the smallest local repair.

            Return JSON only. Ask for no more than the methods whose behavior directly affects the smallest local repair.
            Favor false negatives over false positives, but remember this item was routed as context-required.
            Prefer one or two concrete local candidates when external method behavior could change the edit.
            Do not ask for generic libraries, decorators, assertions, logging helpers, or obvious builtins unless their
            behavior is the specific target of the SATD.
            """
        ).strip()
        user_prompt = textwrap.dedent(
            f"""
            You will repair this SATD next.
            Before that, you may ask for method context.

            SATD comment:
            {state["satd_comment"]}

            Current file path:
            {state["file_path"]}

            Code snippet:
            {self._prompt_code_block(state["original_code"])}

            Candidate methods and symbols from the snippet:
            {candidate_block}

            Select only from the candidate list unless the SATD names an explicit method or class not listed.
            Good selections are symbols named by the SATD, methods called in the target lines, or helpers whose behavior determines whether a local edit is safe.
            Bad selections are broad framework APIs, incidental calls, obvious accessors, constructors, or methods merely adjacent to the SATD.

            Return JSON with:
            {{
              "required_methods": [
                {{"method_name": "...", "selection_reason": "..."}}
              ],
              "reason": "short explanation"
            }}

            If no candidate would change the repair, return an empty required_methods list.
            """
        ).strip()
        payload = self.client.generate_json(
            system_prompt,
            user_prompt,
            temperature=0.0,
            request_label=f"method_inquiry:task_{state.get('task_id', '?')}",
            max_tokens=1024,
        )
        inquiry = self._coerce_method_inquiry(payload)
        if not inquiry.reason:
            inquiry.reason = "llm_required_method_selection"
        return inquiry

    def _method_candidates(self, state: GraphState) -> list[str]:
        code = str(state.get("original_code") or "")
        comment = str(state.get("satd_comment") or "")
        candidates: list[str] = []
        seen: set[str] = set()

        def add(name: str) -> None:
            cleaned = self._normalize_method_name(name)
            if not cleaned or cleaned in seen:
                return
            leaf = cleaned.split(".")[-1]
            if leaf in self._ignored_candidate_names():
                return
            seen.add(cleaned)
            candidates.append(cleaned)

        for pattern in (
            r"\bself\.([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            r"\bcls\.([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\s*\(",
        ):
            for match in re.finditer(pattern, code):
                add(match.group(1))
        for line in code.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("def ", "async def ", "class ")):
                continue
            for match in re.finditer(r"(?<![\.\w])([A-Za-z_][A-Za-z0-9_]*)\s*\(", line):
                add(match.group(1))

        for match in re.finditer(r"`([^`]{2,80})`|['\"]([A-Za-z_][A-Za-z0-9_.]{2,80})['\"]", comment):
            add(match.group(1) or match.group(2) or "")
        for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", comment):
            add(match.group(0))

        return candidates[:12]

    def _format_candidates(self, candidates: list[str]) -> str:
        if not candidates:
            return "[none]"
        return "\n".join(f"- {name}" for name in candidates)

    def _ignored_candidate_names(self) -> set[str]:
        return {
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "set",
            "tuple",
            "len",
            "range",
            "enumerate",
            "zip",
            "print",
            "super",
            "isinstance",
            "issubclass",
            "getattr",
            "setattr",
            "hasattr",
            "open",
            "append",
            "extend",
            "items",
            "keys",
            "values",
            "format",
        }

    def _retrieve_method_contexts(
        self,
        state: GraphState,
        method_names: list[str],
    ) -> list[RetrievedMethodContext]:
        if not method_names:
            return []
        payloads = self.method_retriever.fetch_method_contexts(
            owner=state["user"],
            repo=state["project"],
            current_path=state["file_path"],
            method_names=method_names[: self.max_method_contexts],
            ref=state.get("commit") or None,
            log_prefix=self._task_prefix(state),
        )
        return [self._coerce_retrieved_method_context(item) for item in payloads]

    def _coerce_method_inquiry(self, payload: dict[str, Any]) -> MethodInquiryResult:
        raw_methods = payload.get("required_methods")
        required_methods: list[str] = []
        method_notes: list[dict[str, str]] = []
        seen: set[str] = set()

        if isinstance(raw_methods, list):
            for item in raw_methods:
                raw_name = item
                reason = ""
                if isinstance(item, dict):
                    raw_name = item.get("method_name") or item.get("name") or item.get("method") or ""
                    reason = str(
                        item.get("selection_reason")
                        or item.get("reason")
                        or item.get("why")
                        or ""
                    ).strip()
                name = self._normalize_method_name(raw_name)
                if not name or name in seen:
                    continue
                seen.add(name)
                required_methods.append(name)
                method_notes.append(
                    {
                        "method_name": name,
                        "why_blocking": reason or f"{name} may affect analyzer filtering.",
                        "what_to_learn": reason or f"Understand {name} before filtering.",
                    }
                )

        return MethodInquiryResult(
            required_methods=required_methods,
            reason=str(payload.get("reason") or "").strip(),
            method_notes=method_notes,
        )

    def _coerce_retrieved_method_context(self, payload: dict[str, Any]) -> RetrievedMethodContext:
        return RetrievedMethodContext(
            method_name=str(payload.get("method_name") or ""),
            path=str(payload.get("path") or ""),
            class_name=payload.get("class_name"),
            start_line=self._coerce_int(payload.get("start_line")),
            end_line=self._coerce_int(payload.get("end_line")),
            source=str(payload.get("source") or ""),
            found=bool(payload.get("found")),
            signature=str(payload.get("signature") or ""),
            callsite_slice=str(payload.get("callsite_slice") or ""),
            evidence_slice=str(payload.get("evidence_slice") or ""),
            match_score=int(payload.get("match_score") or 0),
            confidence=self._coerce_float(payload.get("confidence")),
            confidence_label=str(payload.get("confidence_label") or ""),
        )

    def _format_method_context_block(
        self,
        method_inquiry: MethodInquiryResult,
        retrieved_method_contexts: list[RetrievedMethodContext],
    ) -> str:
        if not retrieved_method_contexts:
            return "[none]"

        note_by_name = {
            self._normalize_method_name(item.get("method_name")): str(
                item.get("why_blocking") or item.get("what_to_learn") or ""
            ).strip()
            for item in (method_inquiry.method_notes or [])
            if isinstance(item, dict)
        }
        lines: list[str] = []
        for context in retrieved_method_contexts[: self.max_method_contexts]:
            lines.append(f"- method: {context.method_name}")
            if context.path:
                line_span = ""
                if context.start_line:
                    line_span = f":{context.start_line}"
                    if context.end_line and context.end_line != context.start_line:
                        line_span += f"-{context.end_line}"
                lines.append(f"  location: {context.path}{line_span}")
            note = note_by_name.get(self._normalize_method_name(context.method_name))
            if note:
                lines.append(f"  why needed: {note}")
            source = (context.evidence_slice or context.source or "").strip()
            if source:
                lines.append("  source:")
                lines.extend(f"    {line}" for line in source.splitlines()[:80])

        return "\n".join(lines) if lines else "[none]"

    def _candidate_uses_method_context(self, candidate_mode: str) -> bool:
        mode = str(candidate_mode or "").strip().lower()
        if mode.endswith("no_context"):
            return False
        return self.repair_context_mode in {"method_query", "clone_treesitter"}

    def _no_context_fallback_mode(self, candidate_mode: str) -> str:
        mode = str(candidate_mode or "").strip() or "single"
        if mode == "baseline_context":
            return "baseline_no_context"
        return f"{mode}_no_context"

    def _serialize_method_notes(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [dict(item) for item in items if isinstance(item, dict)]

    def _serialize_retrieved_method_contexts(self, items: list[RetrievedMethodContext]) -> list[dict[str, Any]]:
        return [
            {
                "method_name": item.method_name,
                "path": item.path,
                "class_name": item.class_name,
                "start_line": item.start_line,
                "end_line": item.end_line,
                "found": item.found,
                "signature": item.signature,
                "match_score": item.match_score,
                "confidence": item.confidence,
                "confidence_label": item.confidence_label,
            }
            for item in items
        ]

    def _normalize_method_name(self, value: Any) -> str:
        text = str(value or "").strip().strip("`'\"")
        text = re.sub(r"\(.*\)$", "", text)
        text = text.replace("\\", ".").replace("/", ".")
        text = re.sub(r"[^A-Za-z0-9_.$]+", "", text)
        text = text.strip(".")
        text = re.sub(r"^(self|cls)\.", "", text)
        return text.strip(".")

    def _prompt_code_block(self, code: str) -> str:
        lines = (code or "").expandtabs(4).splitlines()
        return "\n".join(line.rstrip() for line in lines)

    def _coerce_int(self, value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _coerce_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

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
