from __future__ import annotations

import ast
import re
import textwrap
from typing import Any

from .openai_client import OpenAICompatClient
from .schema import BlockingUnknown, GraphState, PlannerResult, RetrievalQuery


ALLOWED_NEED_TYPES = {"definition", "usage", "sibling_pattern"}
NEED_TYPE_TO_TOOL = {
    "definition": "symbol_definition",
    "usage": "callsite_usage",
    "sibling_pattern": "sibling_pattern",
}
ALLOWED_SCOPES = {"current_file", "same_directory", "same_project"}
ALLOWED_FALLBACKS = {"continue_without_context", "local_only", "drop"}
ALLOWED_TARGET_ROLES = {"introduce_or_preserve", "delete_anchor", "disambiguate"}


class OpenAIPlanner:
    """Plan executable context retrieval for SATD repair."""

    def __init__(self, client: OpenAICompatClient, max_queries: int = 1) -> None:
        self.client = client
        self.max_queries = max(1, min(1, int(max_queries)))

    def run(self, state: GraphState) -> PlannerResult:
        payload = self.client.generate_json(
            self._system_prompt(),
            self._user_prompt(state),
            temperature=0.0,
            request_label=f"planner:task_{state.get('task_id', '?')}",
            max_tokens=1800,
        )
        return self._coerce_plan(payload, state)

    def _system_prompt(self) -> str:
        return textwrap.dedent(
            """
            You are the Planner Agent in a SATD repair workflow.
            Decide whether context lookup is necessary.
            Default to context_needed=false.
            Query only facts that directly affect the patch.
            Prefer usage when context is needed.
            Return JSON only.
            """
        ).strip()

    def _user_prompt(self, state: GraphState) -> str:
        return textwrap.dedent(
            f"""
            Plan context retrieval for this SATD repair.

            SATD comment:
            {state["satd_comment"]}

            Code:
            {state["original_code"]}


            Return exactly:
            {{
              "context_needed": true or false,
              "queries": [
                {{
                  "id": "q1",
                  "need_type": "definition" | "usage" | "sibling_pattern",
                  "target": {{}},
                  "scope": "current_file" | "same_directory" | "same_project",
                  "expected_patch_use": "how this evidence may affect repaired_code",
                  "grounding_tokens": ["literal tokens copied from SATD/code/imports"]
                }}
              ],
              "no_context_reason": "short reason when context_needed is false"
            }}

            Need types:
            - definition: signature, constructor, field, constant, or declaration.
            - usage: calling or consuming pattern for a symbol.
            - sibling_pattern: nearby or same-directory code pattern.
            Use definition only when the patch depends on a signature, field, constant, constructor, or declaration.
            Use sibling_pattern only when the patch should mirror nearby code.

            Target shape must use only these keys: {{"symbol": "..."}}, {{"anchor": "..."}}, {{"patterns": ["..."]}}.
            Targets must be grounded in SATD comment, code, imports, or an explicit replacement target.
            Use at most {self.max_queries} queries.
            """
        ).strip()

    def _coerce_plan(self, payload: dict[str, Any], state: GraphState) -> PlannerResult:
        context_needed = bool(payload.get("context_needed"))
        rejected: list[dict[str, Any]] = []
        known_symbols = self._known_symbols(state)
        literal_tokens = self._literal_tokens(state)
        blocking_unknowns: list[BlockingUnknown] = []
        total_queries = 0

        raw_queries = payload.get("queries")
        if isinstance(raw_queries, list):
            queries: list[RetrievalQuery] = []
            for raw_query in raw_queries:
                if not isinstance(raw_query, dict):
                    continue
                query = self._coerce_query(raw_query)
                reason = self._query_rejection_reason(query, known_symbols, literal_tokens)
                if reason:
                    rejected.append({"query": raw_query, "reason": reason})
                    continue
                queries.append(query)
                total_queries += 1
                if total_queries >= self.max_queries:
                    break
            if queries:
                blocking_unknowns.append(BlockingUnknown(unknown="planner_queries", queries=queries))

        raw_unknowns = payload.get("blocking_unknowns") or []
        if not blocking_unknowns and isinstance(raw_unknowns, list):
            for raw_unknown in raw_unknowns:
                if not isinstance(raw_unknown, dict):
                    continue
                queries: list[RetrievalQuery] = []
                for raw_query in raw_unknown.get("queries") or []:
                    if not isinstance(raw_query, dict):
                        continue
                    query = self._coerce_query(raw_query)
                    reason = self._query_rejection_reason(query, known_symbols, literal_tokens)
                    if reason:
                        rejected.append({"query": raw_query, "reason": reason})
                        continue
                    queries.append(query)
                    total_queries += 1
                    if total_queries >= self.max_queries:
                        break
                if queries:
                    blocking_unknowns.append(
                        BlockingUnknown(
                            unknown=self._one_line(raw_unknown.get("unknown")),
                            why_blocking=self._one_line(raw_unknown.get("why_blocking")),
                            queries=queries,
                        )
                    )
                if total_queries >= self.max_queries:
                    break

        if not blocking_unknowns:
            context_needed = False

        return PlannerResult(
            context_needed=context_needed,
            satd_intent=self._one_line(payload.get("satd_intent")),
            local_repair_plan=self._one_line(payload.get("local_repair_plan")),
            blocking_unknowns=blocking_unknowns if context_needed else [],
            no_context_reason=self._one_line(payload.get("no_context_reason"))
            or ("No executable high-value context query was produced." if not context_needed else ""),
            raw_queries_rejected=rejected,
        )

    def _coerce_query(self, raw: dict[str, Any]) -> RetrievalQuery:
        need_type = str(raw.get("need_type") or "").strip()
        tool = self._tool_for_need_type(need_type)
        grounding_tokens = self._coerce_grounding_tokens(raw.get("grounding_tokens"))
        return RetrievalQuery(
            id=self._query_id(raw.get("id")),
            need_type=need_type,
            target=self._coerce_target(raw.get("target")),
            tool=tool,
            scope=str(raw.get("scope") or "same_project").strip(),
            required=bool(raw.get("required")),
            decision=self._one_line(raw.get("decision") or raw.get("expected_patch_use")),
            target_role=str(raw.get("target_role") or "disambiguate").strip(),
            expected_patch_use=self._one_line(raw.get("expected_patch_use")),
            grounding_tokens=grounding_tokens,
            why=self._one_line(raw.get("why")),
            prevents_wrong_repair=self._one_line(raw.get("prevents_wrong_repair") or raw.get("expected_patch_use")),
            fallback_if_not_found=str(raw.get("fallback_if_not_found") or "continue_without_context").strip(),
        )

    def _tool_for_need_type(self, need_type: str) -> str:
        return NEED_TYPE_TO_TOOL.get(str(need_type or "").strip(), "")

    def _coerce_target(self, value: Any) -> dict[str, Any]:
        raw = dict(value or {}) if isinstance(value, dict) else {}
        target: dict[str, Any] = {
            str(key): val
            for key, val in raw.items()
            if str(key) not in {"symbol", "anchor", "patterns"}
        }
        symbol = self._one_line(raw.get("symbol"))
        anchor = self._one_line(raw.get("anchor"))
        patterns = self._coerce_patterns(raw.get("patterns"))
        if symbol:
            target["symbol"] = symbol
        if anchor:
            target["anchor"] = anchor
        if patterns:
            target["patterns"] = patterns
        return target

    def _coerce_patterns(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [self._one_line(item) for item in value if self._one_line(item)][:5]
        text = self._one_line(value)
        return [text] if text else []

    def _query_rejection_reason(
        self,
        query: RetrievalQuery,
        known_symbols: set[str],
        literal_tokens: set[str],
    ) -> str:
        return (
            self._validate_query_options(query)
            or self._validate_query_required_fields(query)
            or self._validate_grounding_tokens(query, literal_tokens)
            or self._validate_target_shape(query, known_symbols, literal_tokens)
        )

    def _validate_query_options(self, query: RetrievalQuery) -> str:
        if query.need_type not in ALLOWED_NEED_TYPES:
            return "unsupported_need_type" if query.need_type else "missing_need_type"
        if not query.tool:
            return "missing_derived_tool"
        if query.scope not in ALLOWED_SCOPES:
            return "unsupported_scope"
        if query.fallback_if_not_found not in ALLOWED_FALLBACKS:
            return "unsupported_fallback"
        if query.target_role not in ALLOWED_TARGET_ROLES:
            return "unsupported_target_role"
        return ""

    def _validate_query_required_fields(self, query: RetrievalQuery) -> str:
        if not query.expected_patch_use:
            return "missing_expected_patch_use"
        if not query.grounding_tokens:
            return "missing_grounding_tokens"
        return ""

    def _validate_grounding_tokens(self, query: RetrievalQuery, literal_tokens: set[str]) -> str:
        for token in query.grounding_tokens:
            if not self._pattern_is_literal(token, literal_tokens):
                return "grounding_token_not_grounded"
        return ""

    def _validate_target_shape(
        self,
        query: RetrievalQuery,
        known_symbols: set[str],
        literal_tokens: set[str],
    ) -> str:
        unsupported_key = self._unsupported_target_key(query.target)
        if unsupported_key:
            return f"unsupported_target_key:{unsupported_key}"
        symbol = str(query.target.get("symbol") or "").strip()
        if query.tool in {"symbol_definition", "callsite_usage"}:
            if not symbol:
                return "missing_symbol"
            if not self._symbol_is_known(symbol, known_symbols):
                return "symbol_not_grounded"

        patterns = query.target.get("patterns") or []
        if patterns and not isinstance(patterns, list):
            return "patterns_not_list"
        for pattern in patterns:
            if not self._pattern_is_literal(str(pattern), literal_tokens):
                return "pattern_not_grounded"

        if query.tool == "sibling_pattern":
            anchor = str(query.target.get("anchor") or "").strip()
            if anchor and not self._pattern_is_literal(anchor, literal_tokens):
                return "anchor_not_grounded"
            if not patterns and not anchor:
                return "missing_sibling_pattern"
        return ""

    def _unsupported_target_key(self, target: dict[str, Any]) -> str:
        allowed = {"symbol", "anchor", "patterns"}
        for key in target:
            if key not in allowed:
                return str(key)
        return ""

    def _coerce_grounding_tokens(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        tokens: list[str] = []
        for item in value:
            text = self._one_line(item)
            if text and text not in tokens:
                tokens.append(text)
            if len(tokens) >= 6:
                break
        return tokens

    def _known_symbols(self, state: GraphState) -> set[str]:
        text = "\n".join([str(state.get("satd_comment") or ""), str(state.get("original_code") or "")])
        symbols = {
            match.group(0).strip("`'\"")
            for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\b", text)
        }
        try:
            tree = ast.parse(str(state.get("original_code") or ""))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    symbols.add(node.name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        symbols.add(alias.asname or alias.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        symbols.add(alias.asname or alias.name.split(".")[0])
        except (SyntaxError, ValueError):
            pass
        return {item for item in symbols if item}

    def _literal_tokens(self, state: GraphState) -> set[str]:
        text = "\n".join([str(state.get("satd_comment") or ""), str(state.get("original_code") or "")])
        tokens = {token for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.]{1,}", text)}
        quoted = {match.group(1) or match.group(2) for match in re.finditer(r"`([^`]+)`|['\"]([^'\"]+)['\"]", text)}
        return {item for item in tokens | quoted if item}

    def _symbol_is_known(self, symbol: str, known_symbols: set[str]) -> bool:
        parts = [part for part in re.split(r"[.\s]+", symbol) if part]
        if symbol in known_symbols:
            return True
        if any(
            item.startswith(f"{symbol}.")
            or item.endswith(f".{symbol}")
            or symbol.startswith(f"{item}.")
            or symbol.endswith(f".{item}")
            for item in known_symbols
        ):
            return True
        return any(part in known_symbols for part in parts)

    def _pattern_is_literal(self, pattern: str, literal_tokens: set[str]) -> bool:
        text = str(pattern or "").strip()
        if not text:
            return False
        if text in literal_tokens:
            return True
        if any(token.startswith(f"{text}.") or text.startswith(f"{token}.") for token in literal_tokens):
            return True
        parts = [part for part in re.findall(r"[A-Za-z_][A-Za-z0-9_.]{1,}", text) if part]
        return bool(parts) and any(
            part in literal_tokens
            or any(token.startswith(f"{part}.") for token in literal_tokens)
            or any(token.endswith(f".{part}") for token in literal_tokens)
            for part in parts
        )

    def _query_id(self, value: Any) -> str:
        text = re.sub(r"[^A-Za-z0-9_-]+", "", str(value or "").strip())
        return text or "q1"

    def _one_line(self, value: Any) -> str:
        return " ".join(str(value or "").split())[:260]
