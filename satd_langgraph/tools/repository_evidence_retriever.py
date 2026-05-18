from __future__ import annotations

import ast
import os
import re
from pathlib import PurePosixPath
from typing import Any

from ..schema import RepositoryEvidenceContext
from .github_downloader import GitHubDownloadTool


class RepositoryEvidenceRetriever:
    """Retrieve diverse repository-local evidence beyond method definitions."""

    def __init__(self, downloader: GitHubDownloadTool | None = None) -> None:
        self.downloader = downloader or GitHubDownloadTool()
        self._file_cache: dict[tuple[str, str, str | None, str], str] = {}

    def retrieve(
        self,
        owner: str,
        repo: str,
        current_path: str,
        satd_comment: str,
        original_code: str,
        ref: str | None = None,
        max_items: int = 5,
    ) -> list[RepositoryEvidenceContext]:
        queries = self._query_seeds(satd_comment, original_code)
        tree = self.downloader._fetch_repo_tree_for_ref(owner, repo, "", ref)
        entries = list(tree.get("entries") or []) if tree.get("ok") else []
        repo_paths = [str(item.get("path") or "") for item in entries if item.get("type") == "file"]
        if not repo_paths:
            return []

        candidates: list[RepositoryEvidenceContext] = []
        candidates.extend(self._definition_candidates(owner, repo, ref, current_path, repo_paths, queries, original_code))
        candidates.extend(self._test_candidates(owner, repo, ref, current_path, repo_paths, queries))
        candidates.extend(self._doc_candidates(owner, repo, ref, current_path, repo_paths, queries))
        candidates.extend(self._usage_candidates(owner, repo, ref, current_path, repo_paths, queries))
        candidates.extend(self._config_candidates(owner, repo, ref, current_path, repo_paths, queries))
        candidates.extend(self._sibling_candidates(owner, repo, ref, current_path, original_code))
        return self._diversify(candidates, max_items=max(1, max_items))

    def _definition_candidates(
        self,
        owner: str,
        repo: str,
        ref: str | None,
        current_path: str,
        repo_paths: list[str],
        queries: list[str],
        original_code: str,
    ) -> list[RepositoryEvidenceContext]:
        results: list[RepositoryEvidenceContext] = []
        current_symbol = self._enclosing_symbol(original_code)
        source_paths = [path for path in repo_paths if self._looks_like_source(path)]
        for query in queries:
            tail = query.split(".")[-1]
            if tail == current_symbol or self._is_low_value_query(tail):
                continue
            for path in self._rank_paths(source_paths, current_path, [tail], limit=16):
                definition = self._find_definition(self._read(owner, repo, ref, path), tail)
                if definition is None:
                    continue
                results.append(
                    RepositoryEvidenceContext(
                        evidence_type="symbol_or_api_definition",
                        evidence_subtype="direct_definition",
                        support_level="direct",
                        source_path=path,
                        span=f"{definition['start']}-{definition['end']}",
                        content=definition["source"],
                        retrieval_method="ast_definition",
                        query_origin="snippet_or_comment",
                        score=0.95,
                        why_relevant="Defines a project-local symbol named by the SATD or target snippet.",
                        query=query,
                    )
                )
                break
        return results

    def _test_candidates(self, owner: str, repo: str, ref: str | None, current_path: str, repo_paths: list[str], queries: list[str]) -> list[RepositoryEvidenceContext]:
        return self._text_candidates(
            owner,
            repo,
            ref,
            current_path,
            [path for path in repo_paths if self._is_test_path(path)],
            queries,
            "related_tests_or_expected_behavior",
            "test_oracle",
            "Tests may encode the expected behavior for the SATD target.",
            0.88,
        )

    def _doc_candidates(self, owner: str, repo: str, ref: str | None, current_path: str, repo_paths: list[str], queries: list[str]) -> list[RepositoryEvidenceContext]:
        return self._text_candidates(
            owner,
            repo,
            ref,
            current_path,
            [path for path in repo_paths if self._is_doc_path(path) or path == current_path],
            queries,
            "documentation_or_docstring",
            "doc_reference",
            "Documentation may state the intended local behavior.",
            0.78,
        )

    def _usage_candidates(self, owner: str, repo: str, ref: str | None, current_path: str, repo_paths: list[str], queries: list[str]) -> list[RepositoryEvidenceContext]:
        return self._text_candidates(
            owner,
            repo,
            ref,
            current_path,
            [path for path in repo_paths if self._looks_like_source(path) and path != current_path and not self._is_test_path(path)],
            queries,
            "project_usage_examples",
            "usage_example",
            "Existing usage shows how the project applies the relevant API.",
            0.82,
        )

    def _config_candidates(self, owner: str, repo: str, ref: str | None, current_path: str, repo_paths: list[str], queries: list[str]) -> list[RepositoryEvidenceContext]:
        return self._text_candidates(
            owner,
            repo,
            ref,
            current_path,
            [path for path in repo_paths if self._is_config_path(path) or self._looks_like_source(path)],
            queries,
            "config_or_project_convention",
            "convention_example",
            "Config or nearby convention constrains the accepted repair shape.",
            0.72,
        )

    def _sibling_candidates(
        self,
        owner: str,
        repo: str,
        ref: str | None,
        current_path: str,
        original_code: str,
    ) -> list[RepositoryEvidenceContext]:
        current_symbol = self._enclosing_symbol(original_code)
        current_content = self._read(owner, repo, ref, current_path)
        if not current_symbol or not current_content:
            return []
        siblings = self._extract_siblings(current_content, current_symbol)
        siblings = self._rank_siblings_by_similarity(siblings, original_code)
        return [
            RepositoryEvidenceContext(
                evidence_type="sibling_implementation",
                evidence_subtype="same_file_sibling",
                support_level="supporting",
                source_path=current_path,
                span=f"{item['start']}-{item['end']}",
                content=item["source"],
                retrieval_method="ast_same_file_sibling",
                query_origin="enclosing_symbol",
                score=0.86,
                why_relevant="Nearby sibling implementation may reveal the project-local pattern.",
                query=current_symbol,
            )
            for item in siblings[:4]
        ]

    def _text_candidates(
        self,
        owner: str,
        repo: str,
        ref: str | None,
        current_path: str,
        paths: list[str],
        queries: list[str],
        evidence_type: str,
        subtype: str,
        why_relevant: str,
        base_score: float,
    ) -> list[RepositoryEvidenceContext]:
        results: list[RepositoryEvidenceContext] = []
        useful_queries = [query for query in queries if not self._is_low_value_query(query)]
        ranked = self._rank_paths(paths, current_path, useful_queries, limit=18)
        for path in ranked:
            content = self._read(owner, repo, ref, path)
            lines = content.splitlines()
            lowered = [line.lower() for line in lines]
            for query in useful_queries:
                needle = query.lower()
                for index, line in enumerate(lowered):
                    if needle not in line:
                        continue
                    start = max(0, index - 2)
                    end = min(len(lines), index + 3)
                    results.append(
                        RepositoryEvidenceContext(
                            evidence_type=evidence_type,
                            evidence_subtype=subtype,
                            support_level="supporting" if base_score >= 0.8 else "weak",
                            source_path=path,
                            span=f"{start + 1}-{end}",
                            content="\n".join(lines[start:end]).strip(),
                            retrieval_method="ranked_text_match",
                            query_origin="snippet_or_comment",
                            score=base_score,
                            why_relevant=why_relevant,
                            query=query,
                        )
                    )
                    break
                if results and results[-1].source_path == path:
                    break
            if len(results) >= 6:
                break
        return results

    def _diversify(self, candidates: list[RepositoryEvidenceContext], max_items: int) -> list[RepositoryEvidenceContext]:
        ranked = sorted(candidates, key=lambda item: (-item.score, item.evidence_type, item.source_path, item.span))
        chosen: list[RepositoryEvidenceContext] = []
        seen_locations: set[tuple[str, str]] = set()
        used_types: set[str] = set()
        for candidate in ranked:
            location = (candidate.source_path, candidate.span)
            if location in seen_locations:
                continue
            if candidate.evidence_type in used_types and len(chosen) < min(3, max_items):
                continue
            chosen.append(candidate)
            seen_locations.add(location)
            used_types.add(candidate.evidence_type)
            if len(chosen) >= max_items:
                break
        if len(chosen) < max_items:
            for candidate in ranked:
                location = (candidate.source_path, candidate.span)
                if location in seen_locations:
                    continue
                chosen.append(candidate)
                seen_locations.add(location)
                if len(chosen) >= max_items:
                    break
        return chosen

    def _query_seeds(self, satd_comment: str, original_code: str) -> list[str]:
        values: list[str] = []
        current_symbol = self._enclosing_symbol(original_code)
        for candidate in self._calls(original_code):
            if candidate and candidate not in values and not self._is_low_value_query(candidate):
                values.append(candidate)
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_\.]{2,}", satd_comment or ""):
            if token != current_symbol and not self._is_low_value_query(token) and token not in values:
                values.append(token)
        return values[:10]

    def _calls(self, code: str) -> list[str]:
        values: list[str] = []
        try:
            tree = ast.parse(code or "")
        except SyntaxError:
            return values
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = self._call_name(node.func)
            if name and name not in values:
                values.append(name)
        return values

    def _call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _find_definition(self, content: str, name: str) -> dict[str, Any] | None:
        if not content or not name:
            return None
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None
        lines = content.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                start = int(getattr(node, "lineno", 1))
                end = int(getattr(node, "end_lineno", start))
                return {"start": start, "end": end, "source": "\n".join(lines[start - 1 : end]).strip()}
        return None

    def _enclosing_symbol(self, code: str) -> str:
        try:
            tree = ast.parse(code or "")
        except SyntaxError:
            return ""
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        return ""

    def _extract_siblings(self, content: str, current_symbol: str) -> list[dict[str, Any]]:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._extract_siblings_text_fallback(content, current_symbol)
        lines = content.splitlines()
        results: list[dict[str, Any]] = []
        current_node, parent = self._find_symbol_with_parent(tree, current_symbol)
        if current_node is None:
            return results
        search_space = parent.body if parent is not None else tree.body
        for node in search_space:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != current_symbol:
                start = int(getattr(node, "lineno", 1))
                end = int(getattr(node, "end_lineno", start))
                results.append(
                    {
                        "start": start,
                        "end": end,
                        "source": "\n".join(lines[start - 1 : end]).strip(),
                        "distance": abs(start - int(getattr(current_node, "lineno", start))),
                    }
                )
        results.sort(key=lambda item: (item["distance"], item["start"]))
        return results

    def _extract_siblings_text_fallback(self, content: str, current_symbol: str) -> list[dict[str, Any]]:
        lines = content.splitlines()
        pattern = re.compile(r"^(?P<indent>[ \t]*)def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
        class_pattern = re.compile(r"^(?P<indent>[ \t]*)class\s+[A-Za-z_][A-Za-z0-9_]*")
        defs: list[dict[str, Any]] = []
        for index, line in enumerate(lines, start=1):
            match = pattern.match(line)
            if not match:
                continue
            defs.append(
                {
                    "name": match.group("name"),
                    "start": index,
                    "indent": len(match.group("indent").replace("\t", "    ")),
                }
            )
        current = next((item for item in defs if item["name"] == current_symbol), None)
        if current is None:
            return []
        class_start, class_end = 1, len(lines) + 1
        for index in range(current["start"] - 1, 0, -1):
            match = class_pattern.match(lines[index - 1])
            if not match:
                continue
            indent = len(match.group("indent").replace("\t", "    "))
            if indent < current["indent"]:
                class_start = index
                for next_index in range(index + 1, len(lines) + 1):
                    next_match = class_pattern.match(lines[next_index - 1])
                    if not next_match:
                        continue
                    next_indent = len(next_match.group("indent").replace("\t", "    "))
                    if next_indent <= indent:
                        class_end = next_index
                        break
                break
        results: list[dict[str, Any]] = []
        for position, item in enumerate(defs):
            if item["name"] == current_symbol or item["indent"] != current["indent"]:
                continue
            if not (class_start <= item["start"] < class_end):
                continue
            next_start = next(
                (
                    candidate["start"]
                    for candidate in defs[position + 1 :]
                    if candidate["indent"] <= item["indent"]
                ),
                len(lines) + 1,
            )
            end = max(item["start"], next_start - 1)
            results.append(
                {
                    "start": item["start"],
                    "end": end,
                    "source": "\n".join(lines[item["start"] - 1 : end]).strip(),
                    "distance": abs(item["start"] - current["start"]),
                }
            )
        results.sort(key=lambda item: (item["distance"], item["start"]))
        return results

    def _rank_siblings_by_similarity(self, siblings: list[dict[str, Any]], original_code: str) -> list[dict[str, Any]]:
        anchor_tokens = self._meaningful_tokens(original_code)
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        for item in siblings:
            overlap = len(anchor_tokens & self._meaningful_tokens(str(item.get("source") or "")))
            ranked.append((-overlap, int(item.get("distance") or 0), item))
        ranked.sort(key=lambda row: (row[0], row[1], int(row[2].get("start") or 0)))
        return [item for _, _, item in ranked]

    def _meaningful_tokens(self, text: str) -> set[str]:
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text or ""))
        return {token for token in tokens if not self._is_low_value_query(token)}

    def _find_symbol_with_parent(self, tree: ast.AST, name: str) -> tuple[ast.AST | None, ast.AST | None]:
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == name:
                    return child, node
        return None, None

    def _rank_paths(self, paths: list[str], current_path: str, queries: list[str], limit: int) -> list[str]:
        current_dir = os.path.dirname((current_path or "").replace("\\", "/"))
        module_stem = PurePosixPath(current_path or "").stem.lower()
        tokens = [query.lower().split(".")[-1] for query in queries if query]
        scored: list[tuple[int, str]] = []
        for path in paths:
            lowered = path.lower()
            score = 0
            if current_dir and lowered.startswith(current_dir.lower() + "/"):
                score += 5
            if module_stem and module_stem in lowered:
                score += 4
            if any(token in lowered for token in tokens):
                score += 3
            if path == current_path:
                score += 8
            scored.append((score, path))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [path for _, path in scored[:limit]]

    def _read(self, owner: str, repo: str, ref: str | None, path: str) -> str:
        key = (owner, repo, ref, path)
        if key not in self._file_cache:
            payload = self.downloader.fetch_repo_file(owner, repo, path, ref)
            self._file_cache[key] = str(payload.get("full_content") or "") if payload.get("ok") else ""
        return self._file_cache[key]

    def _looks_like_source(self, path: str) -> bool:
        return path.lower().endswith((".py", ".pyi"))

    def _is_test_path(self, path: str) -> bool:
        lowered = path.lower()
        return any(token in lowered for token in ("/test", "tests/", "_test.py", "test_"))

    def _is_doc_path(self, path: str) -> bool:
        lowered = path.lower()
        name = PurePosixPath(path).name.lower()
        return lowered.endswith((".md", ".rst", ".txt")) or name.startswith("readme")

    def _is_config_path(self, path: str) -> bool:
        lowered = path.lower()
        return lowered.endswith((".ini", ".cfg", ".toml", ".yaml", ".yml", ".json")) or any(
            token in lowered for token in ("config", "settings", "logging")
        )

    def _is_low_value_query(self, query: str) -> bool:
        value = str(query or "").strip().lower().split(".")[-1]
        if len(value) < 4:
            return True
        return value in {
            "todo",
            "fixme",
            "hack",
            "remove",
            "append",
            "print",
            "error",
            "debug",
            "version",
            "handle",
            "check",
            "where",
            "with",
            "from",
            "this",
            "that",
            "call",
            "calls",
            "save",
            "name",
            "item",
            "items",
            "data",
            "list",
            "test",
            "init",
        }
