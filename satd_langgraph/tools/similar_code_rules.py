from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..bootstrap import bootstrap_vendor

bootstrap_vendor()

from tree_sitter import Language, Parser
import tree_sitter_python


class SimilarCodeRules:
    """Rules for finding likely method definitions and similar code locations."""

    def __init__(self, parse_cache_dir: Path | None = None) -> None:
        if parse_cache_dir is None:
            parse_cache_dir = Path(__file__).resolve().parent.parent.parent / ".repo_cache" / "__parse_cache__"
        self.parse_cache_dir = parse_cache_dir
        self._python_language = Language(tree_sitter_python.language())
        self._python_parser = Parser(self._python_language)

    def _safe_cache_key(self, value: str) -> str:
        return hashlib.sha1((value or "").encode("utf-8", errors="replace")).hexdigest()

    def _global_parse_cache_path(self, cache_key: str) -> Path:
        return self.parse_cache_dir / f"{self._safe_cache_key(cache_key)}.json"

    def _read_json_file(self, path: Path) -> dict[str, Any] | list[Any] | None:
        try:
            if not path.exists():
                return None
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None

    def _write_json_file(self, path: Path, payload: dict[str, Any] | list[Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

    def _sanitize_source_text(self, text: str) -> str:
        if not text:
            return ""
        return text.replace("\x00", "")

    def _is_test_path(self, path: str) -> bool:
        lowered = (path or "").lower()
        return any(token in lowered for token in ("/test", "tests/", "_test.py", "test_"))

    def _candidate_paths_from_symbol_index(
        self,
        query: dict[str, Any],
        symbol_index: dict[str, list[dict[str, Any]]],
        current_path: str,
    ) -> list[str]:
        identifiers = [str(item or "").strip() for item in (query.get("identifiers") or []) if str(item or "").strip()]
        preferred = str(query.get("preferred") or "").strip()
        lookup_keys: list[str] = []
        if identifiers:
            lookup_keys.append(".".join(identifiers))
        if preferred:
            lookup_keys.append(preferred)
        if len(identifiers) >= 2:
            lookup_keys.append(f"{identifiers[-2]}.{identifiers[-1]}")

        candidates: dict[str, int] = {}
        normalized_full = self._normalize_identifier(".".join(identifiers))
        normalized_owner = self._normalize_identifier(identifiers[-2]) if len(identifiers) >= 2 else ""
        normalized_preferred = self._normalize_identifier(preferred)
        current_norm = (current_path or "").replace("\\", "/").strip("/")
        receiver_paths = [
            str(item or "").replace("\\", "/").strip("/")
            for item in (query.get("receiver_paths") or [])
            if str(item or "").strip()
        ]

        for key in lookup_keys:
            for item in symbol_index.get(key, []):
                path = str(item.get("path") or "").replace("\\", "/").strip("/")
                if not path:
                    continue
                score = 0
                symbol_name = str(item.get("symbol_name") or "")
                class_name = str(item.get("class_name") or "")
                qualified_name = str(item.get("qualified_name") or "")
                normalized_symbol = self._normalize_identifier(symbol_name)
                normalized_class = self._normalize_identifier(class_name)
                normalized_qualified = self._normalize_identifier(qualified_name)
                if normalized_symbol == normalized_preferred:
                    score += 40
                if normalized_full and normalized_qualified == normalized_full:
                    score += 100
                elif normalized_full and normalized_qualified.endswith(normalized_full):
                    score += 60
                if normalized_owner and normalized_class == normalized_owner:
                    score += 35
                if path == current_norm:
                    score += 5
                if path in receiver_paths:
                    score += 80
                if score <= 0:
                    continue
                candidates[path] = max(candidates.get(path, 0), score)

        ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
        return [path for path, _ in ranked[:24]]

    def infer_method_path_hints(
        self,
        current_path: str,
        method_names: list[str],
        current_content: str,
        repo_paths: list[str],
        explicit_hints: dict[str, list[str]] | None = None,
    ) -> dict[str, list[str]]:
        import_index = self._build_import_index(current_content, current_path)
        call_targets = self._extract_called_references(current_content)
        hints: dict[str, list[str]] = {}
        for raw_name in method_names:
            query = self._normalize_method_query(raw_name)
            query = self._refine_query_with_call_targets(query, call_targets)
            candidates: list[str] = []
            if explicit_hints:
                for hint in explicit_hints.get(raw_name, []) or []:
                    normalized = (hint or "").replace("\\", "/").strip("/")
                    if normalized and normalized not in candidates:
                        candidates.append(normalized)
            for path in self._candidate_paths_from_imports(query, import_index, current_path):
                if path not in candidates:
                    candidates.append(path)
            normalized_current = (current_path or "").replace("\\", "/").strip("/")
            if normalized_current and normalized_current not in candidates:
                candidates.append(normalized_current)
            scored_repo_paths = self._score_repo_paths_for_query(query, repo_paths, current_path)
            for path in scored_repo_paths:
                if path not in candidates:
                    candidates.append(path)
                if len(candidates) >= 24:
                    break
            hints[raw_name] = candidates
        return hints

    def _refine_query_with_call_targets(self, query: dict[str, Any], call_targets: list[dict[str, Any]]) -> dict[str, Any]:
        preferred = str(query.get("preferred") or "")
        if not preferred:
            return query
        matches = []
        for item in call_targets:
            qualified_name = str(item.get("qualified_name") or "")
            identifiers = list(item.get("identifiers") or [])
            if not qualified_name or not identifiers:
                continue
            if identifiers[-1] == preferred:
                matches.append((qualified_name, identifiers))
        if len(matches) != 1:
            return query
        qualified_name, identifiers = matches[0]
        return {
            **query,
            "raw": qualified_name,
            "display_name": qualified_name,
            "preferred": identifiers[-1],
            "identifiers": identifiers,
            "refined_from_usage": True,
        }

    def _candidate_paths_from_imports(
        self,
        query: dict[str, Any],
        import_index: dict[str, list[str]],
        current_path: str,
    ) -> list[str]:
        candidates: list[str] = []
        identifiers = list(query.get("identifiers") or [])
        keys = []
        if identifiers:
            keys.append(".".join(identifiers))
            keys.extend(identifiers)
        preferred = str(query.get("preferred") or "")
        if preferred:
            keys.append(preferred)
        for key in keys:
            for candidate in import_index.get(key, []):
                normalized = self._prepend_repo_prefix_if_needed(candidate, current_path)
                if normalized and normalized not in candidates:
                    candidates.append(normalized)
        return candidates

    def _score_repo_paths_for_query(self, query: dict[str, Any], repo_paths: list[str], current_path: str) -> list[str]:
        scored: list[tuple[int, str]] = []
        preferred = str(query.get("preferred") or "")
        identifiers = list(query.get("identifiers") or [])
        for path in repo_paths:
            score = self._score_path_for_method(path, preferred, identifiers, current_path)
            if score <= 0:
                continue
            scored.append((score, path))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [path for _, path in scored[:24]]

    def _score_path_for_method(self, path: str, preferred: str, identifiers: list[str], current_path: str) -> int:
        lowered_path = (path or "").lower()
        basename = os.path.basename(lowered_path)
        current_prefix = os.path.dirname((current_path or "").replace("\\", "/").lower())
        score = 0
        if self._is_test_path(lowered_path):
            score -= 2
        if preferred and preferred.lower() in basename:
            score += 8
        elif preferred and preferred.lower() in lowered_path:
            score += 4
        if identifiers:
            for token in identifiers[:-1]:
                lowered = token.lower()
                if lowered and lowered in lowered_path:
                    score += 3
        if current_prefix and lowered_path.startswith(current_prefix):
            score += 2
        return score

    def _build_import_index(self, content: str, current_path: str) -> dict[str, list[str]]:
        clean_content = self._sanitize_source_text(content)
        try:
            tree = ast.parse(clean_content)
        except (SyntaxError, ValueError):
            return {}
        index: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name or ""
                    alias_name = alias.asname or module.split(".")[-1]
                    self._add_import_candidate(index, alias_name, self._module_to_repo_path(module, current_path))
                    self._add_import_candidate(index, module, self._module_to_repo_path(module, current_path))
            elif isinstance(node, ast.ImportFrom):
                module = self._resolve_import_module(current_path, node.module or "", getattr(node, "level", 0))
                module_path = self._module_to_repo_path(module, current_path)
                for alias in node.names:
                    imported_name = alias.name or ""
                    alias_name = alias.asname or imported_name
                    direct_path = self._module_to_repo_path(f"{module}.{imported_name}".strip("."), current_path)
                    package_init_path = self._module_to_repo_init_path(module, current_path)
                    self._add_import_candidate(index, alias_name, module_path)
                    self._add_import_candidate(index, imported_name, module_path)
                    self._add_import_candidate(index, alias_name, direct_path)
                    self._add_import_candidate(index, imported_name, direct_path)
                    self._add_import_candidate(index, alias_name, package_init_path)
                    self._add_import_candidate(index, imported_name, package_init_path)
                    self._add_import_candidate(index, f"{module}.{imported_name}".strip("."), direct_path)
                    self._add_import_candidate(index, f"{alias_name}.{imported_name}".strip("."), direct_path)
                    self._add_import_candidate(index, module, module_path)
        return index

    def _add_import_candidate(self, index: dict[str, list[str]], key: str, path: str) -> None:
        normalized_key = (key or "").strip()
        normalized_path = (path or "").replace("\\", "/").strip("/")
        if not normalized_key or not normalized_path:
            return
        index.setdefault(normalized_key, [])
        if normalized_path not in index[normalized_key]:
            index[normalized_key].append(normalized_path)

    def _module_to_repo_path(self, module: str, current_path: str) -> str:
        dotted = (module or "").strip(".")
        if not dotted:
            return ""
        module_path = dotted.replace(".", "/")
        candidate = f"{module_path}.py"
        return self._prepend_repo_prefix_if_needed(candidate, current_path)

    def _module_to_repo_init_path(self, module: str, current_path: str) -> str:
        dotted = (module or "").strip(".")
        if not dotted:
            return ""
        module_path = dotted.replace(".", "/")
        candidate = f"{module_path}/__init__.py"
        return self._prepend_repo_prefix_if_needed(candidate, current_path)

    def _resolve_import_module(self, current_path: str, module: str, level: int) -> str:
        if level <= 0:
            return (module or "").strip(".")
        current_parts = [part for part in (current_path or "").replace("\\", "/").split("/") if part]
        if current_parts:
            current_parts = current_parts[:-1]
        if level > 1:
            current_parts = current_parts[: max(0, len(current_parts) - (level - 1))]
        base = ".".join(current_parts)
        suffix = (module or "").strip(".")
        if base and suffix:
            return f"{base}.{suffix}"
        return base or suffix

    def _prepend_repo_prefix_if_needed(self, path: str, current_path: str) -> str:
        normalized = (path or "").replace("\\", "/").strip("/")
        if not normalized:
            return ""
        top_level = normalized.split("/", 1)[0]
        current_parts = [part for part in (current_path or "").replace("\\", "/").split("/") if part]
        if top_level and top_level in current_parts:
            index = current_parts.index(top_level)
            prefix = "/".join(current_parts[:index])
            if prefix:
                return f"{prefix}/{normalized}"
        return normalized

    def _extract_called_references(self, code: str) -> list[dict[str, Any]]:
        source = self._sanitize_source_text(code)
        tree = self._parse_python_tree(source)
        if tree is None:
            return []
        source_bytes = source.encode("utf-8", errors="replace")
        results: list[dict[str, Any]] = []
        seen: set[str] = set()

        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == "call":
                function_node = node.child_by_field_name("function")
                raw_call = self._node_text(source_bytes, function_node)
                normalized = self._normalize_method_reference(raw_call)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    results.append(
                        {
                            "raw_call": raw_call,
                            "qualified_name": normalized,
                            "identifiers": normalized.split("."),
                            "line": node.start_point.row + 1,
                        }
                    )
            stack.extend(reversed(node.children))
        results.sort(key=lambda item: (int(item.get("line") or 0), str(item.get("qualified_name") or "")))
        return results

    def _build_local_receiver_type_index(
        self,
        content: str,
        current_path: str,
        import_index: dict[str, list[str]],
    ) -> dict[str, dict[str, Any]]:
        clean_content = self._sanitize_source_text(content)
        try:
            tree = ast.parse(clean_content)
        except (SyntaxError, ValueError):
            return {}
        index: dict[str, dict[str, Any]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                inferred = self._infer_receiver_type(node.value, None, current_path, import_index)
                if not inferred:
                    continue
                for target in node.targets:
                    target_name = self._receiver_target_name(target)
                    if target_name:
                        index[target_name] = self._receiver_type_record(inferred, target_name, current_path, clean_content, node)
            elif isinstance(node, ast.AnnAssign):
                inferred = self._infer_receiver_type(node.value, node.annotation, current_path, import_index)
                if not inferred:
                    continue
                target_name = self._receiver_target_name(node.target)
                if target_name:
                    index[target_name] = self._receiver_type_record(inferred, target_name, current_path, clean_content, node)
        return index

    def _infer_receiver_type(
        self,
        value: ast.AST | None,
        annotation: ast.AST | None,
        current_path: str,
        import_index: dict[str, list[str]],
    ) -> dict[str, Any] | None:
        type_name = ""
        receiver_paths: list[str] = []
        if isinstance(value, ast.Call):
            callee_text = self._normalize_method_reference(ast.unparse(value.func) if hasattr(ast, "unparse") else "")
            if callee_text:
                callee_parts = [part for part in callee_text.split(".") if part]
                if callee_parts:
                    type_name = callee_parts[-1]
                    receiver_paths = self._candidate_paths_from_imports(
                        {
                            "preferred": type_name,
                            "identifiers": callee_parts,
                            "receiver_paths": [],
                        },
                        import_index,
                        current_path,
                    )
        if not type_name and annotation is not None:
            annotation_text = self._normalize_method_reference(ast.unparse(annotation) if hasattr(ast, "unparse") else "")
            if annotation_text:
                annotation_parts = [part for part in annotation_text.split(".") if part]
                if annotation_parts:
                    type_name = annotation_parts[-1]
                    receiver_paths = self._candidate_paths_from_imports(
                        {
                            "preferred": type_name,
                            "identifiers": annotation_parts,
                            "receiver_paths": [],
                        },
                        import_index,
                        current_path,
                    )
        if not type_name:
            return None
        return {
            "type_name": type_name,
            "receiver_paths": receiver_paths,
        }

    def _receiver_target_name(self, target: ast.AST | None) -> str:
        if isinstance(target, ast.Name):
            return str(target.id or "").strip()
        if isinstance(target, ast.Attribute):
            normalized = self._normalize_method_reference(ast.unparse(target) if hasattr(ast, "unparse") else "")
            if normalized:
                return normalized
        return ""

    def _receiver_type_record(
        self,
        inferred: dict[str, Any],
        target_name: str,
        current_path: str,
        content: str,
        node: ast.AST,
    ) -> dict[str, Any]:
        source = ""
        if hasattr(ast, "get_source_segment"):
            source = ast.get_source_segment(content, node) or ""
        return {
            **inferred,
            "target_name": target_name,
            "path": current_path,
            "start_line": getattr(node, "lineno", None),
            "end_line": getattr(node, "end_lineno", getattr(node, "lineno", None)),
            "source": source.strip(),
        }

    def _refine_query_with_receiver_types(
        self,
        query: dict[str, Any],
        receiver_type_index: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        identifiers = [str(item or "").strip() for item in (query.get("identifiers") or []) if str(item or "").strip()]
        if len(identifiers) < 2:
            return query
        owner = identifiers[-2]
        receiver_info = receiver_type_index.get(owner) or receiver_type_index.get(f"self.{owner}")
        if not isinstance(receiver_info, dict):
            return query
        type_name = str(receiver_info.get("type_name") or "").strip()
        if not self._looks_like_type_name(type_name):
            return query
        method_name = identifiers[-1]
        refined_identifiers = [type_name, method_name]
        return {
            **query,
            "raw": ".".join(refined_identifiers),
            "display_name": ".".join(refined_identifiers),
            "preferred": method_name,
            "identifiers": refined_identifiers,
            "receiver_paths": list(receiver_info.get("receiver_paths") or []),
            "refined_from_usage": True,
        }

    def _parse_python_tree(self, source: str) -> Any | None:
        if not source.strip():
            return None
        try:
            return self._python_parser.parse(source.encode("utf-8", errors="replace"))
        except Exception:
            return None

    def _node_text(self, source_bytes: bytes, node: Any | None) -> str:
        if node is None:
            return ""
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def _normalize_method_reference(self, value: str) -> str:
        text = str(value or "").strip().strip("`").strip()
        if not text:
            return ""
        text = text.split("(", 1)[0].strip()
        parts = [part for part in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text) if part]
        while len(parts) > 1 and parts[0] in {"self", "cls", "super"}:
            parts = parts[1:]
        if not parts:
            return ""
        return ".".join(parts)

    def _normalize_method_query(self, method_name: str) -> dict[str, Any]:
        raw = (method_name or "").strip()
        cleaned = re.sub(r"\(.*?\)", "", raw)
        cleaned = cleaned.strip()
        identifiers = [part for part in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", cleaned) if part]
        display_name = identifiers[-1] if identifiers else cleaned or raw
        preferred = display_name
        if "." in cleaned and identifiers:
            preferred = identifiers[-1]
        elif identifiers:
            preferred = identifiers[-1]
        normalized_identifiers = []
        for item in identifiers or ([preferred] if preferred else []):
            if item not in normalized_identifiers:
                normalized_identifiers.append(item)
        return {
            "raw": cleaned or raw,
            "display_name": preferred or raw,
            "preferred": preferred or raw,
            "identifiers": normalized_identifiers,
            "refined_from_usage": False,
        }

    def _find_best_method_match(
        self,
        content: str,
        query: dict[str, Any],
        log_prefix: str,
        scope: str,
        parse_cache_key: str | None = None,
    ) -> dict[str, Any] | None:
        exact_matches = self._extract_exact_method_matches(content, query, parse_cache_key=parse_cache_key)
        candidates = [item for item in exact_matches if self._candidate_is_precise_for_query(query, item)]
        if not candidates:
            return None
        best: dict[str, Any] | None = None
        best_score = -1
        for candidate in candidates:
            score = self._score_method_candidate(query, candidate)
            if best is None or score > best_score:
                best = candidate
                best_score = score
        threshold = 900 if scope.startswith("hint:") else 850
        if best_score < threshold:
            return None
        return best

    def _extract_exact_method_matches(self, content: str, query: dict[str, Any], parse_cache_key: str | None = None) -> list[dict[str, Any]]:
        preferred = str(query.get("preferred") or "")
        if not preferred:
            return []
        matches = [item for item in self._extract_tree_sitter_method_matches(content, parse_cache_key=parse_cache_key) if item.get("symbol_name") == preferred]
        if matches:
            return matches
        return self._extract_regex_method_matches(self._sanitize_source_text(content), preferred=preferred)

    def _extract_tree_sitter_method_matches(self, content: str, parse_cache_key: str | None = None) -> list[dict[str, Any]]:
        source = self._sanitize_source_text(content)
        cache_key = parse_cache_key or hashlib.sha1(source.encode("utf-8", errors="replace")).hexdigest()
        cache_path = self._global_parse_cache_path(cache_key)
        cached_payload = self._read_json_file(cache_path)
        if isinstance(cached_payload, list):
            return [item for item in cached_payload if isinstance(item, dict)]
        tree = self._parse_python_tree(source)
        if tree is None:
            return []
        source_bytes = source.encode("utf-8", errors="replace")
        matches: list[dict[str, Any]] = []

        stack: list[tuple[Any, str | None]] = [(tree.root_node, None)]
        while stack:
            node, enclosing_class = stack.pop()
            node_type = node.type
            if node_type == "class_definition":
                name_node = node.child_by_field_name("name")
                symbol_name = self._node_text(source_bytes, name_node)
                matches.append(
                    {
                        "symbol_name": symbol_name,
                        "qualified_name": symbol_name,
                        "class_name": None,
                        "start_line": node.start_point.row + 1,
                        "end_line": node.end_point.row + 1,
                        "source": self._node_text(source_bytes, node),
                    }
                )
                body = node.child_by_field_name("body")
                if body is not None:
                    for child in reversed(body.named_children):
                        stack.append((child, symbol_name or enclosing_class))
                continue
            if node_type in {"function_definition", "async_function_definition"}:
                name_node = node.child_by_field_name("name")
                symbol_name = self._node_text(source_bytes, name_node)
                qualified_name = ".".join(part for part in [enclosing_class, symbol_name] if part)
                matches.append(
                    {
                        "symbol_name": symbol_name,
                        "qualified_name": qualified_name or symbol_name,
                        "class_name": enclosing_class,
                        "start_line": node.start_point.row + 1,
                        "end_line": node.end_point.row + 1,
                        "source": self._node_text(source_bytes, node),
                    }
                )
            for child in reversed(node.children):
                stack.append((child, enclosing_class))
        self._write_json_file(cache_path, matches)
        return matches

    def _extract_regex_method_matches(self, content: str, preferred: str | None = None) -> list[dict[str, Any]]:
        lines = content.splitlines()
        if not lines:
            return []
        definition_pattern = re.compile(r"^(?P<indent>[ \t]*)(?:(?:async)\s+def|def|class)\s+(?P<name>[A-Za-z_]\w*)\b")
        entries: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            match = definition_pattern.match(line)
            if not match:
                continue
            name = match.group("name")
            if preferred and name != preferred:
                continue
            indent = self._indent_width(match.group("indent"))
            kind = "class" if line.lstrip().startswith("class ") else "function"
            start_line = index + 1
            decorator_index = index - 1
            while decorator_index >= 0:
                decorator_line = lines[decorator_index]
                stripped = decorator_line.strip()
                if not stripped:
                    decorator_index -= 1
                    continue
                if stripped.startswith("@") and self._indent_width(decorator_line) == indent:
                    start_line = decorator_index + 1
                    decorator_index -= 1
                    continue
                break
            entries.append(
                {
                    "symbol_name": name,
                    "kind": kind,
                    "indent": indent,
                    "start_line": start_line,
                    "line_index": index,
                }
            )
        if not entries:
            return []
        for idx, entry in enumerate(entries):
            end_line = len(lines)
            base_indent = int(entry["indent"])
            line_index = int(entry["line_index"])
            for probe in range(line_index + 1, len(lines)):
                probe_line = lines[probe]
                stripped = probe_line.strip()
                if not stripped:
                    continue
                if self._indent_width(probe_line) <= base_indent and not stripped.startswith("#"):
                    end_line = probe
                    break
            entry["end_line"] = end_line
        class_entries = [entry for entry in entries if entry["kind"] == "class"]
        matches: list[dict[str, Any]] = []
        for entry in entries:
            class_name = None
            if entry["kind"] != "class":
                for class_entry in reversed(class_entries):
                    if int(class_entry["start_line"]) <= int(entry["start_line"]) <= int(class_entry["end_line"]):
                        if int(class_entry["indent"]) < int(entry["indent"]):
                            class_name = str(class_entry["symbol_name"])
                            break
            start_line = int(entry["start_line"])
            end_line = int(entry["end_line"])
            matches.append(
                {
                    "symbol_name": str(entry["symbol_name"]),
                    "class_name": class_name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "source": "\n".join(lines[start_line - 1 : end_line]),
                }
            )
        return matches

    def _indent_width(self, line: str) -> int:
        expanded = line.replace("\t", "    ")
        return len(expanded) - len(expanded.lstrip(" "))

    def _score_method_candidate(self, query: dict[str, Any], candidate: dict[str, Any]) -> int:
        preferred = str(query.get("preferred") or "")
        raw = str(query.get("raw") or "")
        identifiers = list(query.get("identifiers") or [])
        symbol_name = str(candidate.get("symbol_name") or "")
        class_name = str(candidate.get("class_name") or "")
        qualified_name = str(candidate.get("qualified_name") or "")
        normalized_symbol = self._normalize_identifier(symbol_name)
        normalized_preferred = self._normalize_identifier(preferred)
        joined_candidate = ".".join(part for part in [class_name, symbol_name] if part)
        normalized_joined = self._normalize_identifier(joined_candidate)
        normalized_qualified = self._normalize_identifier(qualified_name)

        score = 0
        if symbol_name == preferred:
            score += 1200
        if normalized_symbol == normalized_preferred and normalized_symbol:
            score += 900
        if normalized_qualified and identifiers and normalized_qualified.endswith(".".join(self._normalize_identifier(item) for item in identifiers if item)):
            score += 400
        if preferred and preferred.lower() == symbol_name.lower():
            score += 300
        if normalized_preferred and normalized_preferred in normalized_symbol:
            score += 180
        if normalized_symbol and normalized_symbol in normalized_preferred:
            score += 120
        if identifiers:
            normalized_ids = [self._normalize_identifier(item) for item in identifiers if item]
            overlap = sum(1 for item in normalized_ids if item and (item == normalized_symbol or item == normalized_joined))
            score += overlap * 120
            if class_name and len(normalized_ids) >= 2 and self._normalize_identifier(class_name) == normalized_ids[-2]:
                score += 260
        source = str(candidate.get("source") or "")
        source_norm = self._normalize_identifier(source)
        if normalized_preferred and normalized_preferred in source_norm:
            score += 70
        score += int(SequenceMatcher(None, normalized_preferred or raw.lower(), normalized_symbol).ratio() * 100)
        return score

    def _candidate_is_precise_for_query(self, query: dict[str, Any], candidate: dict[str, Any]) -> bool:
        preferred = str(query.get("preferred") or "")
        symbol_name = str(candidate.get("symbol_name") or "")
        if self._normalize_identifier(symbol_name) != self._normalize_identifier(preferred):
            return False

        identifiers = [str(item or "").strip() for item in (query.get("identifiers") or []) if str(item or "").strip()]
        if len(identifiers) < 2:
            return True

        owner_token = identifiers[-2]
        class_name = str(candidate.get("class_name") or "").strip()
        qualified_name = str(candidate.get("qualified_name") or "").strip()
        owner_norm = self._normalize_identifier(owner_token)
        class_norm = self._normalize_identifier(class_name)
        qualified_parts = [self._normalize_identifier(part) for part in qualified_name.split(".") if part]

        if self._looks_like_type_name(owner_token):
            if class_norm:
                return class_norm == owner_norm
            return len(qualified_parts) >= 2 and qualified_parts[-2] == owner_norm

        return True

    def _normalize_identifier(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", (text or "").lower())

    def _looks_like_type_name(self, text: str) -> bool:
        token = str(text or "").strip()
        if not token:
            return False
        return bool(re.match(r"[A-Z][A-Za-z0-9_]*$", token))
