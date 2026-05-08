from __future__ import annotations

import ast
import re
from pathlib import PurePosixPath
from typing import Any

from ..openai_client import OpenAICompatClient
from ..schema import EvidenceCard, GraphState, PlannerResult, RetrievedMethodContext, RetrievalQuery
from .method_retriever import MethodRetrievalTool


class PlannedContextTool:
    """Dispatch Planner queries to a small context tool layer and build evidence cards."""

    def __init__(
        self,
        method_retriever: MethodRetrievalTool,
        max_results_per_query: int = 3,
        evidence_client: OpenAICompatClient | None = None,
        logger: Any | None = None,
    ) -> None:
        self.method_retriever = method_retriever
        self.max_results_per_query = max(1, int(max_results_per_query))
        self.evidence_client = evidence_client
        self.logger = logger

    def prepare_context(
        self,
        state: GraphState,
        planner_result: PlannerResult,
    ) -> tuple[list[RetrievedMethodContext], list[str], list[EvidenceCard], str]:
        if not planner_result.context_needed:
            return [], [], [], "[none]"

        method_contexts: list[RetrievedMethodContext] = []
        missing_symbols: list[str] = []
        cards: list[EvidenceCard] = []
        for query in self._queries(planner_result):
            if query.tool == "symbol_definition":
                contexts = self._symbol_definition(state, query)
                method_contexts.extend(contexts)
                if not contexts:
                    missing_symbols.append(self._target_label(query))
                cards.extend(self._cards_from_method_contexts(state, query, contexts))
            elif query.tool == "callsite_usage":
                cards.extend(self._callsite_usage(state, query))
            elif query.tool == "sibling_pattern":
                cards.extend(self._sibling_pattern(state, query))
            if len(cards) >= self.max_results_per_query * 3:
                break

        self._build_answer_evidence(state, cards)
        evidence_block = self.format_evidence_cards(cards)
        return method_contexts, missing_symbols, cards, evidence_block

    def format_evidence_cards(self, cards: list[EvidenceCard], max_cards: int = 1) -> str:
        selected = [
            card
            for card in cards
            if card.answer and card.relevance in {"high", "medium"} and card.polarity != "weak"
        ][:max_cards]
        if not selected:
            return "[none]"
        blocks: list[str] = []
        for card in selected:
            blocks.append(
                "\n".join(
                    [
                        f"- [{card.tool}] {card.target}",
                        f"  polarity: {card.polarity}",
                        f"  decision: {card.decision}",
                        f"  answer: {card.answer}",
                        f"  edit_hint: {card.edit_hint}",
                        "  snippet:",
                        *[f"    {line}" for line in self._short_snippet(card.snippet).splitlines()],
                    ]
                )
            )
        return "\n".join(blocks)

    def _queries(self, planner_result: PlannerResult) -> list[RetrievalQuery]:
        queries: list[RetrievalQuery] = []
        for unknown in planner_result.blocking_unknowns:
            queries.extend(unknown.queries)
        return queries[:3]

    def _symbol_definition(self, state: GraphState, query: RetrievalQuery) -> list[RetrievedMethodContext]:
        symbol = self._target_symbol(query)
        if not symbol:
            return []
        payloads = self.method_retriever.fetch_method_contexts(
            owner=state["user"],
            repo=state["project"],
            current_path=state["file_path"],
            method_names=[symbol],
            ref=state.get("commit") or None,
            log_prefix=self._task_prefix(state),
        )
        contexts = [self._coerce_method_context(item) for item in payloads]
        return [item for item in contexts if item.found][: self.max_results_per_query]

    def _cards_from_method_contexts(
        self,
        state: GraphState,
        query: RetrievalQuery,
        contexts: list[RetrievedMethodContext],
    ) -> list[EvidenceCard]:
        cards: list[EvidenceCard] = []
        for context in contexts[: self.max_results_per_query]:
            snippet = context.evidence_slice or context.source or ""
            if query.need_type == "definition":
                snippet = self._definition_fact_snippet(snippet)
            cards.append(
                EvidenceCard(
                    query_id=query.id,
                    tool=query.tool,
                    target=context.method_name or self._target_label(query),
                    relevance="high",
                    polarity=self._polarity(state, snippet, query),
                    summary=self._summary(query, f"Definition found in {context.path}:{context.start_line or ''}."),
                    snippet=snippet,
                    decision=query.decision,
                    edit_hint=query.expected_patch_use,
                )
            )
        return cards

    def _definition_fact_snippet(self, snippet: str) -> str:
        lines = [line.rstrip() for line in str(snippet or "").splitlines() if line.strip()]
        if not lines:
            return ""
        first = lines[0].lstrip()
        if first.startswith(("def ", "async def ")):
            return lines[0]
        if first.startswith("class "):
            return "\n".join(lines[: min(len(lines), 6)])
        return lines[0]

    def _callsite_usage(self, state: GraphState, query: RetrievalQuery) -> list[EvidenceCard]:
        symbol = self._target_symbol(query)
        patterns = self._patterns(query)
        if not symbol:
            return self._pattern_usage(state, query, patterns)
        cards: list[EvidenceCard] = []
        for path, content in self._usage_candidate_files(state, query, symbol):
            for line_no, snippet, kind, score in self._usage_windows(content, symbol, patterns):
                if path == state["file_path"] and self._overlaps_original_code(state, snippet):
                    continue
                cards.append(
                    EvidenceCard(
                        query_id=query.id,
                        tool=query.tool,
                        target=f"{symbol} usage",
                        relevance="high" if score >= 80 else "medium",
                        polarity=self._polarity(state, snippet, query),
                        summary=self._summary(query, f"{kind} usage found in {path}:{line_no}."),
                        snippet=snippet,
                        decision=query.decision,
                        edit_hint=query.expected_patch_use,
                    )
                )
                if len(cards) >= self.max_results_per_query:
                    return cards
        return cards

    def _pattern_usage(self, state: GraphState, query: RetrievalQuery, patterns: list[str]) -> list[EvidenceCard]:
        if not patterns:
            return []
        cards: list[EvidenceCard] = []
        for path, content in self._candidate_files(state, query):
            for line_no, snippet in self._matching_windows(content, patterns=patterns, any_match=True):
                if path == state["file_path"] and self._overlaps_original_code(state, snippet):
                    continue
                cards.append(
                    EvidenceCard(
                        query_id=query.id,
                        tool=query.tool,
                        target="pattern usage",
                        relevance="high" if path == state["file_path"] else "medium",
                        polarity=self._polarity(state, snippet, query),
                        summary=self._summary(query, f"Pattern usage found in {path}:{line_no}."),
                        snippet=snippet,
                        decision=query.decision,
                        edit_hint=query.expected_patch_use,
                    )
                )
                if len(cards) >= self.max_results_per_query:
                    return cards
        return cards

    def _usage_candidate_files(
        self,
        state: GraphState,
        query: RetrievalQuery,
        symbol: str,
    ) -> list[tuple[str, str]]:
        paths = self._usage_candidate_paths(state, query, symbol)
        files: list[tuple[str, str]] = []
        for path in paths:
            payload = self.method_retriever._fetch_repo_file(
                state["user"], state["project"], path, state.get("commit") or None
            )
            content = str(payload.get("full_content") or "") if payload.get("ok") else ""
            if content and self._content_may_contain_usage(content, symbol, self._patterns(query)):
                files.append((path, content))
            if len(files) >= 80:
                break
        return files

    def _usage_candidate_paths(
        self,
        state: GraphState,
        query: RetrievalQuery,
        symbol: str,
    ) -> list[str]:
        current_path = state["file_path"]
        paths: list[str] = [current_path]
        scope = query.scope or "same_project"
        if scope == "current_file":
            return self._dedupe_paths(paths)

        repo_paths = self._repo_source_paths(state)
        current_dir = str(PurePosixPath(current_path).parent).strip(".")
        same_dir_paths = [
            path
            for path in repo_paths
            if str(PurePosixPath(path).parent).strip(".") == current_dir and path != current_path
        ]
        paths.extend(same_dir_paths)
        if scope == "same_directory":
            return self._dedupe_paths(paths)

        current_payload = self.method_retriever._fetch_repo_file(
            state["user"], state["project"], current_path, state.get("commit") or None
        )
        current_content = str(current_payload.get("full_content") or "") if current_payload.get("ok") else ""
        query_obj = self.method_retriever._normalize_method_query(symbol)
        import_index = self.method_retriever._build_import_index(current_content, current_path)
        paths.extend(self.method_retriever._candidate_paths_from_imports(query_obj, import_index, current_path))
        paths.extend(self._symbol_index_candidate_paths(state, query_obj, repo_paths, current_path))
        paths.extend(self.method_retriever._score_repo_paths_for_query(query_obj, repo_paths, current_path))
        paths.extend(repo_paths)
        return self._dedupe_paths(paths)[:160]

    def _symbol_index_candidate_paths(
        self,
        state: GraphState,
        query_obj: dict[str, Any],
        repo_paths: list[str],
        current_path: str,
    ) -> list[str]:
        try:
            symbol_index = self.method_retriever._load_symbol_index(
                state["user"],
                state["project"],
                state.get("commit") or None,
                repo_paths,
                log_prefix=self._task_prefix(state),
            )
            return self.method_retriever._candidate_paths_from_symbol_index(query_obj, symbol_index, current_path)
        except Exception:
            return []

    def _usage_windows(
        self,
        content: str,
        symbol: str,
        patterns: list[str] | None = None,
    ) -> list[tuple[int, str, str, int]]:
        lines = content.splitlines()
        try:
            tree = ast.parse(content)
        except (SyntaxError, ValueError):
            return [(line_no, snippet, "text", 45) for line_no, snippet in self._matching_windows(content, symbol=symbol, patterns=patterns)]

        parent_map = self._ast_parent_map(tree)
        hits: list[tuple[int, str, str, int]] = []
        for node in ast.walk(tree):
            match = self._usage_match(node, symbol, parent_map)
            if not match:
                continue
            line_no = int(getattr(node, "lineno", 0) or 0)
            if line_no <= 0:
                continue
            snippet = self._usage_snippet(lines, line_no, self._enclosing_block(node, parent_map))
            score = self._usage_score(match, snippet, patterns)
            hits.append((line_no, snippet, match, score))
        return self._rank_usage_hits(hits)

    def _usage_match(self, node: ast.AST, symbol: str, parent_map: dict[ast.AST, ast.AST]) -> str:
        if self._inside_import(node, parent_map):
            return ""
        if isinstance(node, ast.Call):
            return "call" if self._ref_matches_symbol(self._ast_ref(node.func), symbol) else ""
        if isinstance(node, ast.Attribute):
            parent = parent_map.get(node)
            if isinstance(parent, ast.Call) and parent.func is node:
                return ""
            if not isinstance(getattr(node, "ctx", None), ast.Load):
                return ""
            return "attribute" if self._ref_matches_symbol(self._ast_ref(node), symbol) else ""
        if isinstance(node, ast.Name):
            parent = parent_map.get(node)
            if isinstance(parent, ast.Attribute):
                return ""
            if not isinstance(getattr(node, "ctx", None), ast.Load):
                return ""
            return "name" if self._ref_matches_symbol(node.id, symbol) else ""
        return ""

    def _usage_score(self, kind: str, snippet: str, patterns: list[str] | None = None) -> int:
        score = {"call": 75, "attribute": 60, "name": 50, "text": 35}.get(kind, 35)
        for pattern in [item for item in patterns or [] if item][:2]:
            if pattern in snippet:
                score += 12
        return score

    def _rank_usage_hits(self, hits: list[tuple[int, str, str, int]]) -> list[tuple[int, str, str, int]]:
        deduped: dict[str, tuple[int, str, str, int]] = {}
        for line_no, snippet, kind, score in hits:
            key = re.sub(r"\s+", " ", snippet).strip()
            if not key:
                continue
            existing = deduped.get(key)
            if existing is None or score > existing[3]:
                deduped[key] = (line_no, snippet, kind, score)
        ranked = sorted(deduped.values(), key=lambda item: (-item[3], item[0]))
        return ranked[: max(self.max_results_per_query * 4, self.max_results_per_query)]

    def _usage_snippet(self, lines: list[str], line_no: int, enclosing: ast.AST | None) -> str:
        snippet = self._window_around(lines, line_no, before=3, after=4)
        enclosing_line = int(getattr(enclosing, "lineno", 0) or 0) if enclosing is not None else 0
        if enclosing_line and enclosing_line < line_no - 3:
            header = lines[enclosing_line - 1].rstrip()
            if header and header not in snippet:
                snippet = f"{header}\n...\n{snippet}"
        return snippet

    def _enclosing_block(self, node: ast.AST, parent_map: dict[ast.AST, ast.AST]) -> ast.AST | None:
        current = parent_map.get(node)
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return current
            current = parent_map.get(current)
        return None

    def _ast_parent_map(self, tree: ast.AST) -> dict[ast.AST, ast.AST]:
        return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    def _inside_import(self, node: ast.AST, parent_map: dict[ast.AST, ast.AST]) -> bool:
        current: ast.AST | None = node
        while current is not None:
            if isinstance(current, (ast.Import, ast.ImportFrom)):
                return True
            current = parent_map.get(current)
        return False

    def _ast_ref(self, node: ast.AST | None) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            owner = self._ast_ref(node.value)
            return f"{owner}.{node.attr}" if owner else node.attr
        if isinstance(node, ast.Call):
            return self._ast_ref(node.func)
        if isinstance(node, ast.Subscript):
            return self._ast_ref(node.value)
        return ""

    def _ref_matches_symbol(self, reference: str, symbol: str) -> bool:
        ref_parts = self._identifier_parts(reference)
        symbol_parts = self._identifier_parts(symbol)
        if not ref_parts or not symbol_parts:
            return False
        if ref_parts == symbol_parts:
            return True
        if len(ref_parts) >= len(symbol_parts) and ref_parts[-len(symbol_parts) :] == symbol_parts:
            return True
        return ref_parts[-1] == symbol_parts[-1]

    def _identifier_parts(self, text: str) -> list[str]:
        return [part for part in re.split(r"[^A-Za-z0-9_]+", str(text or "")) if part]

    def _content_may_contain_usage(self, content: str, symbol: str, patterns: list[str] | None = None) -> bool:
        parts = self._identifier_parts(symbol)
        needles = [symbol, parts[-1] if parts else "", *(patterns or [])]
        return any(needle and needle in content for needle in needles)

    def _dedupe_paths(self, paths: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for path in paths:
            normalized = (path or "").replace("\\", "/").strip("/")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _overlaps_original_code(self, state: GraphState, snippet: str) -> bool:
        original = self._meaningful_lines(str(state.get("original_code") or ""))
        candidate = set(self._meaningful_lines(snippet))
        if not original or not candidate:
            return False
        overlap = sum(1 for line in original if line in candidate)
        if len(original) == 1:
            return overlap == 1
        return overlap >= 2 and overlap / max(1, len(original)) >= 0.6

    def _meaningful_lines(self, text: str) -> list[str]:
        lines: list[str] = []
        for line in str(text or "").splitlines():
            stripped = re.sub(r"\s+", " ", line.strip())
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(stripped)
        return lines

    def _sibling_pattern(self, state: GraphState, query: RetrievalQuery) -> list[EvidenceCard]:
        patterns = self._patterns(query)
        anchor = str(query.target.get("anchor") or "").strip()
        cards = []
        for path, content in self._candidate_files(state, query, prefer_same_directory=True):
            for line_no, snippet in self._matching_windows(content, symbol=anchor, patterns=patterns, any_match=True):
                cards.append(
                    EvidenceCard(
                        query_id=query.id,
                        tool=query.tool,
                        target=anchor or "sibling pattern",
                        relevance="high" if path == state["file_path"] else "medium",
                        polarity=self._polarity(state, snippet, query),
                        summary=self._summary(query, f"Similar local pattern found in {path}:{line_no}."),
                        snippet=snippet,
                        decision=query.decision,
                        edit_hint=query.expected_patch_use,
                    )
                )
                if len(cards) >= self.max_results_per_query:
                    return cards
        return cards

    def _candidate_files(
        self,
        state: GraphState,
        query: RetrievalQuery,
        prefer_same_directory: bool = False,
    ) -> list[tuple[str, str]]:
        paths = [state["file_path"]]
        scope = query.scope or "same_project"
        if scope != "current_file":
            repo_paths = self._repo_source_paths(state)
            current_dir = str(PurePosixPath(state["file_path"]).parent).strip(".")
            if prefer_same_directory or scope == "same_directory":
                paths.extend(path for path in repo_paths if str(PurePosixPath(path).parent).strip(".") == current_dir)
            if scope == "same_project" and not prefer_same_directory:
                paths.extend(repo_paths)
        seen: set[str] = set()
        files: list[tuple[str, str]] = []
        for path in paths:
            normalized = (path or "").replace("\\", "/").strip("/")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            payload = self.method_retriever._fetch_repo_file(
                state["user"], state["project"], normalized, state.get("commit") or None
            )
            if payload.get("ok") and str(payload.get("full_content") or "").strip():
                files.append((normalized, str(payload.get("full_content") or "")))
            if len(files) >= 80:
                break
        return files

    def _repo_source_paths(self, state: GraphState) -> list[str]:
        ref = state.get("commit") or None
        payload = (
            self.method_retriever._fetch_repo_tree_for_ref(state["user"], state["project"], "", ref)
            if ref
            else self.method_retriever._fetch_repo_tree(state["user"], state["project"], "", ref)
        )
        paths = [
            str(item.get("path") or "")
            for item in payload.get("entries", [])
            if item.get("type") == "file" and self.method_retriever._looks_like_source_file(str(item.get("path") or ""))
        ]
        current = state["file_path"]
        current_dir = str(PurePosixPath(current).parent).strip(".")
        scored = sorted(
            paths,
            key=lambda path: (
                0 if path == current else 1 if str(PurePosixPath(path).parent).strip(".") == current_dir else 2,
                path.count("/"),
                path,
            ),
        )
        return scored[:120]

    def _matching_windows(
        self,
        content: str,
        *,
        symbol: str = "",
        patterns: list[str] | None = None,
        any_match: bool = False,
    ) -> list[tuple[int, str]]:
        lines = content.splitlines()
        patterns = [p for p in patterns or [] if p]
        needles = [symbol, *patterns] if symbol else patterns
        needles = [needle for needle in needles if needle]
        if not needles:
            return []
        results: list[tuple[int, str]] = []
        for idx, line in enumerate(lines):
            haystack = line
            if any_match:
                matched = any(needle in haystack for needle in needles)
            else:
                matched = symbol in haystack and all(pattern in "\n".join(lines[max(0, idx - 2) : idx + 3]) for pattern in patterns[:2])
            if not matched:
                continue
            start = max(0, idx - 3)
            end = min(len(lines), idx + 4)
            snippet = "\n".join(lines[start:end]).strip()
            results.append((idx + 1, snippet))
            if len(results) >= self.max_results_per_query:
                break
        return results

    def _window_around(self, lines: list[str], line_no: int, before: int = 3, after: int = 4) -> str:
        idx = max(0, int(line_no or 1) - 1)
        start = max(0, idx - before)
        end = min(len(lines), idx + after + 1)
        return "\n".join(lines[start:end]).strip()

    def _polarity(self, state: GraphState, snippet: str, query: RetrievalQuery | None = None) -> str:
        if query is not None and query.target_role == "delete_anchor":
            return "anchor_to_delete"
        comment = str(state.get("satd_comment") or "").lower()
        if any(marker in comment for marker in ("remove", "delete", "drop", "get rid of", "debug", "temporary")):
            return "anchor_to_delete"
        if not snippet.strip():
            return "weak"
        return "support"

    def _coerce_method_context(self, payload: dict[str, Any]) -> RetrievedMethodContext:
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

    def _target_symbol(self, query: RetrievalQuery) -> str:
        return str(query.target.get("symbol") or "").strip()

    def _target_label(self, query: RetrievalQuery) -> str:
        return self._target_symbol(query) or str(query.target.get("anchor") or query.tool)

    def _patterns(self, query: RetrievalQuery) -> list[str]:
        raw = query.target.get("patterns") or []
        if not isinstance(raw, list):
            return []
        return [str(item).strip() for item in raw if str(item).strip()][:5]

    def _summary(self, query: RetrievalQuery, base: str) -> str:
        parts = [base]
        if query.need_type:
            parts.append(f"Need: {query.need_type}.")
        if query.decision:
            parts.append(f"Decision: {query.decision}")
        if query.expected_patch_use:
            parts.append(f"Patch use: {query.expected_patch_use}")
        return " ".join(parts)[:500]

    def _build_answer_evidence(self, state: GraphState, cards: list[EvidenceCard], max_cards: int = 3) -> None:
        for card in cards[:max_cards]:
            if self.evidence_client is None:
                card.answer = card.answer or self._fallback_answer(card)
                card.edit_hint = card.edit_hint or "Use this evidence only if it directly answers the repair decision."
                self._downgrade_if_unhelpful(card)
                continue
            try:
                payload = self.evidence_client.generate_json(
                    "You turn one retrieved code snippet into one concise repair-decision answer. Return JSON only.",
                    self._evidence_prompt(state, card),
                    temperature=0.0,
                    request_label=f"evidence_builder:task_{state.get('task_id', '?')}:{card.query_id}",
                    max_tokens=350,
                )
                card.answer = self._one_line(payload.get("answer"))
                card.edit_hint = self._one_line(payload.get("edit_hint")) or card.edit_hint
                relevance = self._one_line(payload.get("relevance"))
                if relevance in {"high", "medium", "low"}:
                    card.relevance = relevance
            except Exception:
                card.answer = card.answer or ""
            self._downgrade_if_unhelpful(card)

    def _evidence_prompt(self, state: GraphState, card: EvidenceCard) -> str:
        return (
            "Answer only with a concrete fact supported by the retrieved snippet. "
            "Do not restate the decision. If the snippet does not answer it, leave answer empty.\n\n"
            f"SATD comment:\n{state.get('satd_comment') or ''}\n\n"
            f"Code to repair:\n{state.get('original_code') or ''}\n\n"
            f"Decision:\n{card.decision}\n\n"
            f"Retrieved snippet:\n{self._short_snippet(card.snippet, max_lines=8)}\n\n"
            'Return exactly: {"answer": "...", "edit_hint": "...", "relevance": "high|medium|low"}'
        )

    def _fallback_answer(self, card: EvidenceCard) -> str:
        summary = str(card.summary or "")
        first = summary.split(". Patch use:", 1)[0]
        first = first.split(". Decision:", 1)[0]
        return " ".join(first.split())[:220]

    def _downgrade_if_unhelpful(self, card: EvidenceCard) -> None:
        if self._useful_answer(card):
            return
        card.answer = ""
        card.edit_hint = ""
        card.relevance = "low"
        card.polarity = "weak"

    def _useful_answer(self, card: EvidenceCard) -> bool:
        answer = self._normalized(card.answer)
        if not answer:
            return False
        return answer not in {self._normalized(card.decision), self._normalized(card.edit_hint)}

    def _normalized(self, value: Any) -> str:
        return re.sub(r"\W+", " ", str(value or "").lower()).strip()

    def _one_line(self, value: Any) -> str:
        return " ".join(str(value or "").split())[:260]

    def _short_snippet(self, snippet: str, max_lines: int = 8) -> str:
        lines = [line.rstrip() for line in str(snippet or "").splitlines() if line.strip()]
        text = "\n".join(lines[:max_lines])
        return text[:1200]

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

    def _task_prefix(self, state: GraphState) -> str:
        task_id = state.get("task_id", "?")
        task_index = state.get("task_index") or 0
        task_total = state.get("task_total") or 0
        if task_index and task_total:
            return f"[task {task_id} {task_index}/{task_total}]"
        return f"[task {task_id}]"
