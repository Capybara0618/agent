from __future__ import annotations

import hashlib
import os
import time
from typing import Any

from .github_downloader import GitHubDownloadTool
from .similar_code_rules import SimilarCodeRules


class MethodRetrievalTool:
    """Retrieve method source contexts using GitHub downloads and similarity rules."""

    def __init__(
        self,
        downloader: GitHubDownloadTool,
        rules: SimilarCodeRules,
        logger: Any | None = None,
    ) -> None:
        self.downloader = downloader
        self.rules = rules
        self.logger = logger
        self.symbol_index_heartbeat_seconds = max(5, int(os.environ.get("SATD_SYMBOL_INDEX_HEARTBEAT_SECONDS") or 10))

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.rules, name):
            return getattr(self.rules, name)
        if hasattr(self.downloader, name):
            return getattr(self.downloader, name)
        raise AttributeError(name)

    def fetch_method_contexts(
        self,
        owner: str,
        repo: str,
        current_path: str,
        method_names: list[str],
        ref: str | None = None,
        log_prefix: str = "",
        path_hints_by_method: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        contexts: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        current_payload = self._fetch_repo_file(owner, repo, current_path, ref)
        current_content = current_payload.get("full_content") or "" if current_payload.get("ok") else ""
        tree_payload = self._fetch_repo_tree_for_ref(owner, repo, "", ref) if ref else self._fetch_repo_tree(owner, repo, "", ref)
        repo_paths = [
            str(item.get("path") or "")
            for item in tree_payload.get("entries", [])
            if item.get("type") == "file" and self._looks_like_source_file(str(item.get("path") or ""))
        ]
        symbol_index = self._load_symbol_index(owner, repo, ref, repo_paths, log_prefix=log_prefix)
        call_targets = self._extract_called_references(current_content)
        import_index = self._build_import_index(current_content, current_path)
        receiver_type_index = self._build_local_receiver_type_index(current_content, current_path, import_index)
        auto_hints = self.infer_method_path_hints(
            current_path=current_path,
            method_names=method_names,
            current_content=current_content,
            repo_paths=repo_paths,
            explicit_hints=path_hints_by_method,
        )
        for raw_name in method_names:
            method_name = (raw_name or "").strip()
            if not method_name or method_name in seen_names:
                continue
            seen_names.add(method_name)
            hints = list(auto_hints.get(method_name) or [])
            query = self._refine_query_with_call_targets(self._normalize_method_query(method_name), call_targets)
            query = self._refine_query_with_receiver_types(query, receiver_type_index)
            for symbol_path in self._candidate_paths_from_symbol_index(query, symbol_index, current_path):
                if symbol_path not in hints:
                    hints.append(symbol_path)
            context = (
                self._fetch_single_method_context(
                    owner,
                    repo,
                    current_path,
                    method_name,
                    ref,
                    log_prefix,
                    hints,
                    query_override=query,
                )
            )
            contexts.append(context)
            self._log(self._method_result_log(context, hints), log_prefix)
        return contexts

    def _log(self, message: str, prefix: str = "") -> None:
        if callable(self.logger):
            text = f"{prefix} {message}".strip() if prefix else message
            self.logger(text)

    def _missing_method_context(self, display_name: str) -> dict[str, Any]:
        return {
            "method_name": display_name,
            "path": "",
            "class_name": None,
            "start_line": None,
            "end_line": None,
            "source": "",
            "found": False,
        }

    def _method_result_log(self, context: dict[str, Any], hints: list[str]) -> str:
        method_name = str(context.get("method_name") or "?")
        searched_count = len([item for item in hints if str(item).strip()])
        if context.get("found"):
            path = str(context.get("path") or "?")
            start_line = context.get("start_line")
            end_line = context.get("end_line")
            line_part = ""
            if isinstance(start_line, int):
                line_part = f":{start_line}"
                if isinstance(end_line, int) and end_line != start_line:
                    line_part += f"-{end_line}"
            return f"search {method_name} -> hit {path}{line_part} candidates={searched_count}"
        return f"search {method_name} -> miss candidates={searched_count}"

    def _fetch_single_method_context(
        self,
        owner: str,
        repo: str,
        current_path: str,
        method_name: str,
        ref: str | None,
        log_prefix: str = "",
        path_hints: list[str] | None = None,
        query_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        query = query_override or self._normalize_method_query(method_name)
        candidate_paths: list[str] = []
        for hint in path_hints or []:
            normalized = (hint or "").replace("\\", "/").strip("/")
            if normalized and normalized not in candidate_paths:
                candidate_paths.append(normalized)
        if current_path not in candidate_paths:
            candidate_paths.append(current_path)
        for path in candidate_paths:
            payload = self._fetch_repo_file(owner, repo, path, ref)
            if not payload.get("ok"):
                continue
            content = payload.get("full_content") or ""
            parse_cache_key = str(payload.get("sha") or f"{path}:{hashlib.sha1(content.encode('utf-8', errors='replace')).hexdigest()}")
            best = self._find_best_method_match(content, query, log_prefix, f"hint:{path}", parse_cache_key=parse_cache_key)
            if not best:
                continue
            resolved_name = query["display_name"]
            if query.get("refined_from_usage"):
                resolved_name = best.get("qualified_name") or resolved_name
            return {
                "method_name": resolved_name,
                "path": path,
                "class_name": best.get("class_name"),
                "start_line": best.get("start_line"),
                "end_line": best.get("end_line"),
                "source": best.get("source") or "",
                "found": True,
                "searched_paths": candidate_paths,
            }
        missing = self._missing_method_context(query["display_name"])
        missing["searched_paths"] = candidate_paths
        return missing

    def _load_symbol_index(
        self,
        owner: str,
        repo: str,
        ref: str | None,
        repo_paths: list[str],
        log_prefix: str = "",
    ) -> dict[str, list[dict[str, Any]]]:
        resolved_ref = (ref or "").strip() or "HEAD"
        cache_path = self._repo_symbol_index_cache_path(owner, repo, resolved_ref)
        cached_payload = self._read_json_file(cache_path)
        if isinstance(cached_payload, dict):
            symbols = cached_payload.get("symbols")
            if isinstance(symbols, dict):
                return {
                    str(key): [item for item in value if isinstance(item, dict)]
                    for key, value in symbols.items()
                    if isinstance(value, list)
                }
        self._log(
            f"repo symbol index build repo={owner}/{repo} ref={resolved_ref[:12]} files={len(repo_paths)}",
            log_prefix,
        )
        symbols = self._build_symbol_index(owner, repo, ref, repo_paths, log_prefix=log_prefix)
        payload = {
            "ok": True,
            "ref": ref,
            "resolved_ref": resolved_ref,
            "symbol_count": sum(len(items) for items in symbols.values()),
            "symbols": symbols,
        }
        self._write_json_file(cache_path, payload)
        self._log(
            f"repo symbol index ready repo={owner}/{repo} ref={resolved_ref[:12]} symbols={payload['symbol_count']}",
            log_prefix,
        )
        return symbols

    def _build_symbol_index(
        self,
        owner: str,
        repo: str,
        ref: str | None,
        repo_paths: list[str],
        log_prefix: str = "",
    ) -> dict[str, list[dict[str, Any]]]:
        index: dict[str, list[dict[str, Any]]] = {}
        seen_records: dict[str, set[tuple[str, str, str, str, str]]] = {}
        total_files = len(repo_paths)
        scanned_files = 0
        total_symbols = 0
        next_heartbeat = time.time() + self.symbol_index_heartbeat_seconds
        for path in repo_paths:
            scanned_files += 1
            payload = self._fetch_repo_file(owner, repo, path, ref)
            if not payload.get("ok"):
                if time.time() >= next_heartbeat:
                    self._log(
                        f"repo symbol index scanning repo={owner}/{repo} files={scanned_files}/{total_files} symbols={total_symbols}",
                        log_prefix,
                    )
                    next_heartbeat = time.time() + self.symbol_index_heartbeat_seconds
                continue
            content = payload.get("full_content") or ""
            parse_cache_key = str(
                payload.get("sha")
                or f"{path}:{hashlib.sha1(content.encode('utf-8', errors='replace')).hexdigest()}"
            )
            file_symbols = 0
            for item in self._extract_tree_sitter_method_matches(content, parse_cache_key=parse_cache_key):
                symbol_name = str(item.get("symbol_name") or "").strip()
                if not symbol_name:
                    continue
                record = {
                    "path": str(path),
                    "symbol_name": symbol_name,
                    "qualified_name": str(item.get("qualified_name") or symbol_name),
                    "class_name": str(item.get("class_name") or "").strip() or None,
                    "start_line": item.get("start_line"),
                    "end_line": item.get("end_line"),
                }
                record_key = (
                    str(record["path"]),
                    str(record["symbol_name"]),
                    str(record["qualified_name"]),
                    str(record.get("start_line") or ""),
                    str(record.get("end_line") or ""),
                )
                keys = {
                    symbol_name,
                    str(record["qualified_name"]),
                }
                if record.get("class_name"):
                    keys.add(f"{record['class_name']}.{symbol_name}")
                for key in [k for k in keys if str(k).strip()]:
                    index.setdefault(key, [])
                    seen_for_key = seen_records.setdefault(str(key), set())
                    if record_key not in seen_for_key:
                        seen_for_key.add(record_key)
                        index[key].append(record)
                file_symbols += 1
            total_symbols += file_symbols
            if time.time() >= next_heartbeat:
                self._log(
                    f"repo symbol index scanning repo={owner}/{repo} files={scanned_files}/{total_files} symbols={total_symbols}",
                    log_prefix,
                )
                next_heartbeat = time.time() + self.symbol_index_heartbeat_seconds
        return index
