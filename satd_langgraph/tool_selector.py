from __future__ import annotations

import re
import textwrap
from typing import Any

from .openai_client import OpenAICompatClient
from .schema import GraphState, PlannerResult, RetrievalQuery


ACTION_TO_TOOL = {
    "search_definition": ("symbol_definition", "definition"),
    "search_usage": ("callsite_usage", "usage"),
    "search_sibling_pattern": ("sibling_pattern", "sibling_pattern"),
}
ALLOWED_ACTIONS = set(ACTION_TO_TOOL) | {"no_query"}
ALLOWED_SCOPES = {"current_file", "same_directory", "same_project"}


class OpenAIToolSelector:
    """Select one code evidence tool through OpenAI tool calling."""

    def __init__(self, client: OpenAICompatClient) -> None:
        self.client = client

    def select(self, state: GraphState, plan: PlannerResult) -> tuple[RetrievalQuery | None, str]:
        tool_call = self.client.generate_tool_call(
            self._system_prompt(),
            self._user_prompt(state, plan),
            tools=self._tools(),
            temperature=0.0,
            request_label=f"tool_selector:task_{state.get('task_id', '?')}",
            max_tokens=450,
        )
        return self._coerce_selection(tool_call.get("name"), tool_call.get("arguments"), plan)

    def _system_prompt(self) -> str:
        return "Select exactly one tool for SATD repair evidence."

    def _user_prompt(self, state: GraphState, plan: PlannerResult) -> str:
        return textwrap.dedent(
            f"""
            SATD comment:
            {state["satd_comment"]}

            Code:
            {state["original_code"]}

            Patch decision:
            {plan.patch_decision}

            Evidence need:
            {plan.evidence_need}
            """
        ).strip()

    def _tools(self) -> list[dict[str, Any]]:
        scope = {"type": "string", "enum": ["current_file", "same_directory", "same_project"]}
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_definition",
                    "description": (
                        "Find the declaration or definition of a concrete code symbol: function, method, class, "
                        "constructor, constant, field, or attribute. Use only when the repair depends on a signature, "
                        "available field, enum/constant value, or constructor shape. The symbol must be copied exactly "
                        "from the SATD comment, code, imports, visible identifier, attribute access, or function call; "
                        "do not invent or paraphrase symbol names."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Exact symbol, dotted symbol, or visible expression copied from the input."},
                            "scope": scope,
                            "reason": {"type": "string", "description": "Why this definition can affect the patch."},
                            "grounding": {"type": "string", "description": "Short copied text showing where the symbol came from."},
                        },
                        "required": ["symbol", "scope", "reason"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_usage",
                    "description": (
                        "Find how a concrete symbol or literal code pattern is called, assigned, passed, compared, "
                        "returned, or consumed. Prefer this for variables, fields, parameters, return values, and API calls. "
                        "Use the most specific copied expression when available; do not generalize a precise expression "
                        "into a broad word."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Concrete symbol or expression copied exactly from the input, if available."},
                            "patterns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Literal code tokens or short patterns copied from SATD/code when no single symbol is enough.",
                            },
                            "scope": scope,
                            "reason": {"type": "string", "description": "What usage fact the repair needs."},
                            "grounding": {"type": "string", "description": "Short copied text showing where the symbol or pattern came from."},
                        },
                        "required": ["scope", "reason"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_sibling_pattern",
                    "description": (
                        "Find nearby or same-directory code that should be mirrored for style or behavior. Use when the "
                        "repair needs an analogous implementation pattern rather than a definition. Anchor and patterns "
                        "must be copied from the visible SATD/code and should be short enough to find related code."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "anchor": {"type": "string", "description": "Local anchor symbol or code phrase copied from the input."},
                            "patterns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Literal tokens copied from SATD/code that similar code should contain.",
                            },
                            "scope": scope,
                            "reason": {"type": "string", "description": "What pattern should be mirrored."},
                            "grounding": {"type": "string", "description": "Short copied text showing where the anchor or pattern came from."},
                        },
                        "required": ["scope", "reason"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "no_query",
                    "description": (
                        "Use when the evidence need cannot be answered from the current repository code, or when there "
                        "is no concrete code symbol, expression, or literal pattern to search."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string", "description": "Why no repository-code query is reliable."},
                            "fallback": {"type": "string", "enum": ["local_only", "continue_without_context", "drop"]},
                        },
                        "required": ["reason", "fallback"],
                    },
                },
            },
        ]

    def _coerce_selection(
        self,
        action: Any,
        arguments: Any,
        plan: PlannerResult,
    ) -> tuple[RetrievalQuery | None, str]:
        action = self._one_line(action)
        payload = dict(arguments or {}) if isinstance(arguments, dict) else {}
        reason = self._one_line(payload.get("reason"))
        if action not in ALLOWED_ACTIONS or action == "no_query":
            return None, reason or "Selector chose no_query."
        target = self._coerce_target(payload)
        if not target:
            return None, reason or "Selector did not produce a searchable target."
        scope = self._one_line(payload.get("scope")) or "same_project"
        if scope not in ALLOWED_SCOPES:
            scope = "same_project"
        tool, need_type = ACTION_TO_TOOL[action]
        return (
            RetrievalQuery(
                id="q1",
                need_type=need_type,
                target=target,
                tool=tool,
                scope=scope,
                decision=plan.patch_decision,
                expected_patch_use=plan.patch_decision,
                why=plan.evidence_need,
                prevents_wrong_repair=plan.evidence_need,
                fallback_if_not_found="continue_without_context",
            ),
            reason,
        )

    def _coerce_target(self, value: Any) -> dict[str, Any]:
        raw = dict(value or {}) if isinstance(value, dict) else {}
        symbol = self._one_line(raw.get("symbol"))
        patterns = self._patterns(raw.get("patterns"))
        anchor = self._one_line(raw.get("anchor"))
        if symbol:
            return {"symbol": symbol}
        if patterns:
            return {"patterns": patterns}
        if anchor:
            return {"anchor": anchor}
        return {}

    def _patterns(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [self._one_line(item) for item in value if self._one_line(item)][:5]
        text = self._one_line(value)
        return [text] if text else []

    def _one_line(self, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())[:260]
