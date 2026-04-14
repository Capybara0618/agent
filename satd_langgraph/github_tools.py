from __future__ import annotations

import ast
import base64
import functools
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .bootstrap import bootstrap_vendor
from .local_settings import GITHUB_TOKEN as LOCAL_GITHUB_TOKEN

bootstrap_vendor()

from tree_sitter import Language, Parser
import tree_sitter_python


README_LIMIT = 1600
FILE_LIMIT = 12000
ISSUE_LIMIT = 1600
SEARCH_LIMIT = 8
COMMENT_LIMIT = 6
COMMIT_LIMIT = 5
TREE_LIMIT = 40
SNIPPET_WINDOW = 12
SNIPPET_CHAR_LIMIT = 2200
EXPANDED_ITEM_LIMIT = 4


class GitHubToolbox:
    REPO_CACHE_SCHEMA_VERSION = 1
    SNAPSHOT_CACHE_SCHEMA_VERSION = 1

    def __init__(self) -> None:
        token = os.environ.get("GITHUB_TOKEN") or LOCAL_GITHUB_TOKEN
        self.github_token = None if token == "PASTE_YOUR_GITHUB_TOKEN_HERE" else token
        self.user_agent = "satd-langgraph-agent"
        self.logger = None
        cache_root = os.environ.get("SATD_REPO_CACHE_DIR")
        if cache_root:
            self.repo_cache_dir = Path(cache_root).expanduser()
        else:
            self.repo_cache_dir = Path(__file__).resolve().parent.parent / ".repo_cache"
        self.repo_cache_dir.mkdir(parents=True, exist_ok=True)
        self._python_language = Language(tree_sitter_python.language())
        self._python_parser = Parser(self._python_language)
        self.clone_failure_ttl_seconds = max(0, int(os.environ.get("SATD_REPO_FAILURE_TTL_SECONDS") or 21600))
        self.git_progress_heartbeat_seconds = max(3, int(os.environ.get("SATD_GIT_PROGRESS_HEARTBEAT_SECONDS") or 8))
        self.snapshot_progress_heartbeat_seconds = max(10, int(os.environ.get("SATD_SNAPSHOT_PROGRESS_HEARTBEAT_SECONDS") or 10))
        self.symbol_index_heartbeat_seconds = max(5, int(os.environ.get("SATD_SYMBOL_INDEX_HEARTBEAT_SECONDS") or 10))
        self._snapshot_lock_guard = threading.Lock()
        self._snapshot_locks: dict[str, threading.Lock] = {}
        self._snapshot_cache_hit_logged: set[str] = set()

    def fetch_repo_readme(self, owner: str, repo: str, ref: str | None = None) -> dict[str, Any]:
        return self._fetch_repo_readme(owner, repo, ref)

    def fetch_readme_or_module_docs(self, owner: str, repo: str, path_prefix: str, ref: str | None = None) -> dict[str, Any]:
        return self._fetch_readme_or_module_docs(owner, repo, path_prefix, ref)

    def fetch_repo_file(self, owner: str, repo: str, path: str, ref: str | None = None) -> dict[str, Any]:
        return self._fetch_repo_file(owner, repo, path, ref)

    def fetch_repo_file_full_or_window(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
        max_chars: int = FILE_LIMIT,
    ) -> dict[str, Any]:
        file_payload = self._fetch_repo_file(owner, repo, path, ref)
        if not file_payload.get("ok"):
            return file_payload

        content = file_payload.get("full_content") or ""
        lines = content.splitlines()
        if start_line and end_line and start_line > 0 and end_line >= start_line:
            excerpt_lines = lines[start_line - 1 : end_line]
            excerpt = "\n".join(excerpt_lines)
        else:
            excerpt = content
            start_line = 1 if lines else None
            end_line = len(lines) if lines else None

        excerpt = excerpt[:max_chars]
        return {
            "ok": True,
            "path": file_payload.get("path"),
            "sha": file_payload.get("sha"),
            "download_url": file_payload.get("download_url"),
            "total_lines": len(lines),
            "start_line": start_line,
            "end_line": end_line,
            "content_excerpt": excerpt,
        }

    def extract_satd_window(self, file_content: str, satd_comment: str, window: int = 20) -> dict[str, Any]:
        lines = (file_content or "").splitlines()
        if not lines:
            return {"found": False, "error": "empty_file"}

        satd_line = self._locate_satd_line(lines, satd_comment)
        if satd_line is None:
            return {"found": False, "error": "satd_comment_not_found"}

        start_line = max(1, satd_line - window)
        end_line = min(len(lines), satd_line + window)
        numbered_excerpt = "\n".join(f"{idx}: {lines[idx - 1]}" for idx in range(start_line, end_line + 1))
        return {
            "found": True,
            "satd_line": satd_line,
            "start_line": start_line,
            "end_line": end_line,
            "matched_line": lines[satd_line - 1],
            "excerpt": numbered_excerpt,
        }

    def extract_enclosing_symbol(self, file_content: str, satd_comment: str) -> dict[str, Any]:
        clean_content = self._sanitize_source_text(file_content)
        satd_window = self.extract_satd_window(clean_content, satd_comment, window=0)
        satd_line = satd_window.get("satd_line") if satd_window.get("found") else None
        if satd_line is None:
            return {"found": False, "error": "satd_comment_not_found"}

        try:
            tree = ast.parse(clean_content)
        except (SyntaxError, ValueError) as exc:
            return {"found": False, "error": f"syntax_error: {exc}"}

        best_match: ast.AST | None = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                if start is None or end is None:
                    continue
                if start <= satd_line <= end:
                    if best_match is None:
                        best_match = node
                    else:
                        current_span = getattr(best_match, "end_lineno") - getattr(best_match, "lineno")
                        node_span = end - start
                        if node_span < current_span:
                            best_match = node

        if best_match is None:
            return {"found": False, "error": "no_enclosing_symbol"}

        source_lines = clean_content.splitlines()
        start = getattr(best_match, "lineno")
        end = getattr(best_match, "end_lineno")
        return {
            "found": True,
            "symbol_type": type(best_match).__name__,
            "symbol_name": getattr(best_match, "name", None),
            "start_line": start,
            "end_line": end,
            "source": "\n".join(source_lines[start - 1 : end]),
        }

    def extract_imports(self, file_content: str) -> dict[str, Any]:
        clean_content = self._sanitize_source_text(file_content)
        if not clean_content.strip():
            return {"imports": []}
        try:
            tree = ast.parse(clean_content)
        except (SyntaxError, ValueError) as exc:
            return {"imports": [], "error": f"syntax_error: {exc}"}

        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend(f"{module}:{alias.name}" for alias in node.names)
        return {"imports": imports[:30]}

    def search_code(self, owner: str, repo: str, query: str, per_page: int = SEARCH_LIMIT) -> dict[str, Any]:
        return self._search_code(owner, repo, query, per_page)

    def find_related_tests(
        self,
        owner: str,
        repo: str,
        symbol_name_or_path: str,
        per_page: int = SEARCH_LIMIT,
        ref: str | None = None,
    ) -> dict[str, Any]:
        return self._find_related_tests(owner, repo, symbol_name_or_path, per_page, ref)

    def find_call_sites(
        self,
        owner: str,
        repo: str,
        symbol_name: str,
        per_page: int = SEARCH_LIMIT,
        ref: str | None = None,
        current_path: str = "",
    ) -> dict[str, Any]:
        return self._find_call_sites(owner, repo, symbol_name, per_page, ref, current_path)

    def fetch_commits_for_path(self, owner: str, repo: str, path: str, limit: int = COMMIT_LIMIT) -> dict[str, Any]:
        return self._fetch_commits_for_path(owner, repo, path, limit)

    def search_closed_prs_or_issues(self, owner: str, repo: str, keywords: str, per_page: int = SEARCH_LIMIT) -> dict[str, Any]:
        return self._search_closed_prs_or_issues(owner, repo, keywords, per_page)

    def fetch_issue_or_pr(self, owner: str, repo: str, number: int) -> dict[str, Any]:
        return self._fetch_issue_or_pr(owner, repo, number)

    def fetch_pr_files(self, owner: str, repo: str, pr_number: int) -> dict[str, Any]:
        return self._fetch_pr_files(owner, repo, pr_number)

    def fetch_issue_comments(self, owner: str, repo: str, number: int) -> dict[str, Any]:
        return self._fetch_issue_comments(owner, repo, number)

    def fetch_repo_tree(self, owner: str, repo: str, prefix: str = "", ref: str | None = None) -> dict[str, Any]:
        return self._fetch_repo_tree(owner, repo, prefix, ref)

    def fetch_blame_or_last_commit_for_line(self, owner: str, repo: str, path: str, satd_line: int | None) -> dict[str, Any]:
        commits = self._fetch_commits_for_path(owner, repo, path, 1)
        return {
            "ok": commits.get("ok", False),
            "path": path,
            "satd_line": satd_line,
            "last_commit": commits.get("commits", [None])[0] if commits.get("commits") else None,
            "note": "line-level blame is approximated with the latest path commit",
            "error": commits.get("error"),
        }

    def fetch_code_snippet(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str | None = None,
        anchor_text: str = "",
        symbol_name: str = "",
        prefer_assert: bool = False,
        window: int = SNIPPET_WINDOW,
        max_chars: int = SNIPPET_CHAR_LIMIT,
    ) -> dict[str, Any]:
        return self._fetch_code_snippet(owner, repo, path, ref, anchor_text, symbol_name, prefer_assert, window, max_chars)

    @functools.lru_cache(maxsize=256)
    def _fetch_repo_readme(self, owner: str, repo: str, ref: str | None) -> dict[str, Any]:
        endpoint = self._with_ref(f"https://api.github.com/repos/{owner}/{repo}/readme", ref)
        try:
            payload = self._github_json(endpoint)
            decoded = self._decode_content(payload)
            return {
                "ok": True,
                "path": payload.get("path"),
                "download_url": payload.get("download_url"),
                "content_excerpt": decoded[:README_LIMIT],
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @functools.lru_cache(maxsize=256)
    def _fetch_readme_or_module_docs(self, owner: str, repo: str, path_prefix: str, ref: str | None) -> dict[str, Any]:
        candidates = []
        cleaned_prefix = (path_prefix or "").strip("/")
        if cleaned_prefix:
            candidates.extend([
                f"{cleaned_prefix}/README.md",
                f"{cleaned_prefix}/README.rst",
                f"{cleaned_prefix}/docs/README.md",
            ])
        for candidate in candidates:
            payload = self._fetch_repo_file(owner, repo, candidate, ref)
            if payload.get("ok"):
                return {
                    "ok": True,
                    "path": payload.get("path"),
                    "download_url": payload.get("download_url"),
                    "content_excerpt": (payload.get("full_content") or "")[:README_LIMIT],
                }
        return self._fetch_repo_readme(owner, repo, ref)

    @functools.lru_cache(maxsize=1024)
    def _fetch_repo_file(self, owner: str, repo: str, path: str, ref: str | None) -> dict[str, Any]:
        normalized_path = (path or "").replace("\\", "/").strip("/")
        cache_ref = (ref or "").strip()
        if cache_ref and normalized_path:
            cache_path = self._repo_file_cache_path(owner, repo, cache_ref, normalized_path)
            cached_payload = self._read_json_file(cache_path)
            if isinstance(cached_payload, dict) and cached_payload.get("ok"):
                return dict(cached_payload)
        encoded_path = "/".join(urllib.parse.quote(part) for part in path.split("/"))
        endpoint = self._with_ref(f"https://api.github.com/repos/{owner}/{repo}/contents/{encoded_path}", ref)
        try:
            payload = self._github_json(endpoint)
            decoded = self._decode_content(payload)
            result = {
                "ok": True,
                "path": payload.get("path"),
                "sha": payload.get("sha"),
                "download_url": payload.get("download_url"),
                "content_excerpt": decoded[:FILE_LIMIT],
                "full_content": decoded,
                "ref": ref,
                "resolved_ref": cache_ref or ref,
            }
            if cache_ref and normalized_path:
                self._write_json_file(cache_path, result)
            return result
        except Exception as exc:
            error_text = str(exc)
            if ref:
                if self._ref_exists(owner, repo, ref):
                    error_text = f"historical_file_missing: {error_text}"
                else:
                    error_text = f"historical_ref_missing: {error_text}"
            return {"ok": False, "error": error_text, "path": path, "ref": ref}

    @functools.lru_cache(maxsize=512)
    def _search_code(self, owner: str, repo: str, query: str, per_page: int) -> dict[str, Any]:
        search_query = f"repo:{owner}/{repo} {query}"
        encoded_query = urllib.parse.quote(search_query)
        endpoint = f"https://api.github.com/search/code?q={encoded_query}&per_page={per_page}"
        try:
            payload = self._github_json(endpoint)
            items = []
            for item in payload.get("items", [])[:per_page]:
                items.append(
                    {
                        "name": item.get("name"),
                        "path": item.get("path"),
                        "sha": item.get("sha"),
                        "html_url": item.get("html_url"),
                        "score": item.get("score"),
                    }
                )
            return {"ok": True, "query": query, "count": payload.get("total_count", len(items)), "items": items}
        except Exception as exc:
            return {"ok": False, "query": query, "error": str(exc), "items": []}

    @functools.lru_cache(maxsize=256)
    def _find_related_tests(
        self,
        owner: str,
        repo: str,
        symbol_name_or_path: str,
        per_page: int,
        ref: str | None,
    ) -> dict[str, Any]:
        token = (symbol_name_or_path or "").strip()
        if not token:
            return {"ok": True, "query": "", "count": 0, "items": []}
        basename = os.path.basename(token).replace(".py", "")
        symbol_name = basename if basename and basename != token else ""
        if ref:
            items = self._scan_historical_tree_for_matches(
                owner,
                repo,
                ref,
                query_token=basename or token,
                symbol_name=symbol_name,
                per_page=per_page,
                candidate_kind="test",
            )
            return {"ok": True, "query": basename or token, "count": len(items), "items": items}
        raw = self._search_code(owner, repo, f'{basename} test', per_page)
        items = self._expand_search_items(
            owner,
            repo,
            [item for item in raw.get("items", []) if self._is_test_path(item.get("path", ""))],
            anchor_text=basename,
            symbol_name=symbol_name,
            prefer_assert=True,
        )
        return {"ok": raw.get("ok", False), "query": raw.get("query"), "count": len(items), "items": items}

    @functools.lru_cache(maxsize=256)
    def _find_call_sites(
        self,
        owner: str,
        repo: str,
        symbol_name: str,
        per_page: int,
        ref: str | None,
        current_path: str,
    ) -> dict[str, Any]:
        token = (symbol_name or "").strip()
        if not token:
            return {"ok": True, "query": "", "count": 0, "items": []}
        if ref:
            items = self._scan_historical_tree_for_matches(
                owner,
                repo,
                ref,
                query_token=token,
                symbol_name=token,
                per_page=per_page,
                candidate_kind="callsite",
                current_path=current_path,
            )
            return {"ok": True, "query": token, "count": len(items), "items": items}
        raw = self._search_code(owner, repo, f'"{token}("', per_page)
        items = self._expand_search_items(owner, repo, raw.get("items", []), anchor_text=f"{token}(", symbol_name=token)
        return {"ok": raw.get("ok", False), "query": raw.get("query"), "count": len(items), "items": items}

    @functools.lru_cache(maxsize=256)
    def _fetch_commits_for_path(self, owner: str, repo: str, path: str, limit: int) -> dict[str, Any]:
        encoded_path = urllib.parse.quote(path)
        endpoint = f"https://api.github.com/repos/{owner}/{repo}/commits?path={encoded_path}&per_page={limit}"
        try:
            payload = self._github_json(endpoint)
            commits = []
            for item in payload[:limit]:
                commit = item.get("commit", {})
                commits.append(
                    {
                        "sha": item.get("sha"),
                        "html_url": item.get("html_url"),
                        "message": (commit.get("message") or "")[:300],
                        "author": ((commit.get("author") or {}).get("name")),
                        "date": ((commit.get("author") or {}).get("date")),
                    }
                )
            return {"ok": True, "path": path, "count": len(commits), "commits": commits}
        except Exception as exc:
            return {"ok": False, "path": path, "error": str(exc), "commits": []}

    @functools.lru_cache(maxsize=256)
    def _search_closed_prs_or_issues(self, owner: str, repo: str, keywords: str, per_page: int) -> dict[str, Any]:
        token = (keywords or "").strip()
        if not token:
            return {"ok": True, "query": "", "count": 0, "items": []}
        search_query = f"repo:{owner}/{repo} is:closed {token}"
        endpoint = f"https://api.github.com/search/issues?q={urllib.parse.quote(search_query)}&per_page={per_page}"
        try:
            payload = self._github_json(endpoint)
            items = []
            for item in payload.get("items", [])[:per_page]:
                items.append(
                    {
                        "number": item.get("number"),
                        "title": item.get("title"),
                        "state": item.get("state"),
                        "html_url": item.get("html_url"),
                        "is_pull_request": bool(item.get("pull_request")),
                    }
                )
            return {"ok": True, "query": token, "count": payload.get("total_count", len(items)), "items": items}
        except Exception as exc:
            return {"ok": False, "query": token, "error": str(exc), "items": []}

    @functools.lru_cache(maxsize=256)
    def _fetch_issue_or_pr(self, owner: str, repo: str, number: int) -> dict[str, Any]:
        endpoint = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"
        try:
            payload = self._github_json(endpoint)
            return {
                "ok": True,
                "number": payload.get("number"),
                "title": payload.get("title"),
                "state": payload.get("state"),
                "html_url": payload.get("html_url"),
                "body_excerpt": (payload.get("body") or "")[:ISSUE_LIMIT],
                "is_pull_request": bool(payload.get("pull_request")),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "number": number}

    @functools.lru_cache(maxsize=256)
    def _fetch_pr_files(self, owner: str, repo: str, pr_number: int) -> dict[str, Any]:
        endpoint = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files?per_page={SEARCH_LIMIT}"
        try:
            payload = self._github_json(endpoint)
            files = []
            for item in payload[:SEARCH_LIMIT]:
                files.append(
                    {
                        "filename": item.get("filename"),
                        "status": item.get("status"),
                        "additions": item.get("additions"),
                        "deletions": item.get("deletions"),
                        "changes": item.get("changes"),
                        "patch_excerpt": (item.get("patch") or "")[:SNIPPET_CHAR_LIMIT],
                    }
                )
            return {"ok": True, "number": pr_number, "count": len(files), "files": files}
        except Exception as exc:
            return {"ok": False, "number": pr_number, "error": str(exc), "files": []}

    @functools.lru_cache(maxsize=256)
    def _fetch_issue_comments(self, owner: str, repo: str, number: int) -> dict[str, Any]:
        endpoint = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}/comments?per_page={COMMENT_LIMIT}"
        try:
            payload = self._github_json(endpoint)
            comments = []
            for item in payload[:COMMENT_LIMIT]:
                comments.append(
                    {
                        "user": ((item.get("user") or {}).get("login")),
                        "created_at": item.get("created_at"),
                        "body_excerpt": (item.get("body") or "")[:ISSUE_LIMIT],
                    }
                )
            return {"ok": True, "number": number, "count": len(comments), "comments": comments}
        except Exception as exc:
            return {"ok": False, "number": number, "error": str(exc), "comments": []}

    @functools.lru_cache(maxsize=256)
    def _fetch_repo_tree(self, owner: str, repo: str, prefix: str, ref: str | None) -> dict[str, Any]:
        cleaned_prefix = prefix.strip("/")
        if ref:
            payload = self._fetch_repo_tree_for_ref(owner, repo, cleaned_prefix, ref)
            if not payload.get("ok"):
                return payload
            entries = list(payload.get("entries", []))[:TREE_LIMIT]
            return {
                "ok": True,
                "prefix": cleaned_prefix,
                "count": len(entries),
                "entries": entries,
                "ref": ref,
            }

        encoded_prefix = "/".join(urllib.parse.quote(part) for part in cleaned_prefix.split("/")) if cleaned_prefix else ""
        endpoint = f"https://api.github.com/repos/{owner}/{repo}/contents/{encoded_prefix}" if encoded_prefix else f"https://api.github.com/repos/{owner}/{repo}/contents"
        try:
            payload = self._github_json(endpoint)
            if isinstance(payload, dict):
                payload = [payload]
            entries = []
            for item in payload[:TREE_LIMIT]:
                entries.append(
                    {
                        "name": item.get("name"),
                        "path": item.get("path"),
                        "type": item.get("type"),
                    }
                )
            return {"ok": True, "prefix": cleaned_prefix, "count": len(entries), "entries": entries}
        except Exception as exc:
            return {"ok": False, "prefix": cleaned_prefix, "error": str(exc), "entries": []}

    @functools.lru_cache(maxsize=1024)
    def _fetch_code_snippet(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str | None,
        anchor_text: str,
        symbol_name: str,
        prefer_assert: bool,
        window: int,
        max_chars: int,
    ) -> dict[str, Any]:
        payload = self._fetch_repo_file(owner, repo, path, ref)
        if not payload.get("ok"):
            return {"ok": False, "path": path, "error": payload.get("error")}

        content = self._sanitize_source_text(payload.get("full_content") or "")
        snippet = self._extract_best_snippet(
            content,
            anchor_text=anchor_text,
            symbol_name=symbol_name,
            prefer_assert=prefer_assert,
            window=window,
            max_chars=max_chars,
        )
        snippet.update(
            {
                "ok": True,
                "path": payload.get("path"),
                "sha": payload.get("sha"),
                "download_url": payload.get("download_url"),
            }
        )
        return snippet

    def _repo_cache_path(self, owner: str, repo: str) -> Path:
        return self.repo_cache_dir / owner / repo

    def _repo_cache_meta_dir(self, owner: str, repo: str) -> Path:
        return self.repo_cache_dir / owner / f"{repo}.__satd_cache__"

    def _repo_failure_marker_path(self, owner: str, repo: str) -> Path:
        return self._repo_cache_meta_dir(owner, repo) / "clone_failure.json"

    def _repo_complete_marker_path(self, owner: str, repo: str) -> Path:
        return self._repo_cache_meta_dir(owner, repo) / "clone_complete.json"

    def _repo_ref_cache_path(self, owner: str, repo: str) -> Path:
        return self._repo_cache_meta_dir(owner, repo) / "refs.json"

    def _repo_snapshot_cache_dir(self, owner: str, repo: str, ref: str) -> Path:
        ref_key = self._safe_cache_key(ref)
        return self._repo_cache_meta_dir(owner, repo) / "snapshots" / ref_key

    def _repo_snapshot_marker_path(self, owner: str, repo: str, ref: str) -> Path:
        return self._repo_snapshot_cache_dir(owner, repo, ref) / "snapshot_complete.json"

    def _repo_snapshot_zip_path(self, owner: str, repo: str, ref: str) -> Path:
        return self._repo_snapshot_cache_dir(owner, repo, ref) / "archive.zip"

    def _repo_snapshot_root_path(self, owner: str, repo: str, ref: str) -> Path:
        return self._repo_snapshot_cache_dir(owner, repo, ref) / "root"

    def _snapshot_cache_key(self, owner: str, repo: str, ref: str) -> str:
        return f"{owner}/{repo}@{(ref or '').strip()}"

    def _repo_tree_cache_path(self, owner: str, repo: str, resolved_ref: str, prefix: str = "") -> Path:
        ref_key = self._safe_cache_key(resolved_ref)
        prefix_key = self._safe_cache_key(prefix or "__root__")
        return self._repo_cache_meta_dir(owner, repo) / "trees" / ref_key / f"{prefix_key}.json"

    def _repo_file_cache_path(self, owner: str, repo: str, resolved_ref: str, path: str) -> Path:
        ref_key = self._safe_cache_key(resolved_ref)
        normalized_path = (path or "").replace("\\", "/").strip("/")
        return self._repo_cache_meta_dir(owner, repo) / "files" / ref_key / Path(normalized_path).with_suffix(Path(normalized_path).suffix + ".json")

    def _repo_parse_cache_path(self, owner: str, repo: str, cache_key: str) -> Path:
        return self._repo_cache_meta_dir(owner, repo) / "parses" / f"{self._safe_cache_key(cache_key)}.json"

    def _global_parse_cache_path(self, cache_key: str) -> Path:
        return self.repo_cache_dir / "__parse_cache__" / f"{self._safe_cache_key(cache_key)}.json"

    def _repo_symbol_index_cache_path(self, owner: str, repo: str, resolved_ref: str) -> Path:
        ref_key = self._safe_cache_key(resolved_ref)
        return self._repo_cache_meta_dir(owner, repo) / "symbols" / f"{ref_key}.json"

    def _safe_cache_key(self, value: str) -> str:
        text = (value or "").strip()
        digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
        return digest

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

    def _should_skip_clone_due_to_failure_cache(self, owner: str, repo: str) -> str | None:
        marker = self._repo_failure_marker_path(owner, repo)
        cached = self._read_json_file(marker)
        if not isinstance(cached, dict):
            return None
        error_text = str(cached.get("error") or "")
        local_cache_markers = (
            "destination path",
            "cleanup did not remove",
            "repo cache purge failed",
            "git_dir_mismatch",
        )
        if any(marker_text in error_text for marker_text in local_cache_markers):
            try:
                marker.unlink()
            except Exception:
                pass
            return None
        timestamp = float(cached.get("timestamp") or 0)
        if self.clone_failure_ttl_seconds <= 0:
            return None
        if (time.time() - timestamp) > self.clone_failure_ttl_seconds:
            return None
        return error_text or "cached_clone_failure"

    def _record_clone_failure(self, owner: str, repo: str, error: str) -> None:
        local_cache_markers = (
            "destination path",
            "cleanup did not remove",
            "repo cache purge failed",
            "git_dir_mismatch",
        )
        if any(marker_text in str(error or "") for marker_text in local_cache_markers):
            return
        marker = self._repo_failure_marker_path(owner, repo)
        self._write_json_file(
            marker,
            {
                "timestamp": time.time(),
                "error": str(error or "clone_failed"),
            },
        )

    def _clear_clone_failure(self, owner: str, repo: str) -> None:
        marker = self._repo_failure_marker_path(owner, repo)
        if marker.exists():
            marker.unlink()

    def _record_repo_complete(self, owner: str, repo: str, remote_url: str) -> None:
        marker = self._repo_complete_marker_path(owner, repo)
        self._write_json_file(
            marker,
            {
                "schema_version": self.REPO_CACHE_SCHEMA_VERSION,
                "timestamp": time.time(),
                "remote_url": remote_url,
            },
        )

    def _read_repo_complete_marker(self, owner: str, repo: str) -> dict[str, Any] | None:
        marker = self._repo_complete_marker_path(owner, repo)
        payload = self._read_json_file(marker)
        if isinstance(payload, dict):
            return payload
        return None

    def _read_snapshot_complete_marker(self, owner: str, repo: str, ref: str) -> dict[str, Any] | None:
        marker = self._repo_snapshot_marker_path(owner, repo, ref)
        payload = self._read_json_file(marker)
        if isinstance(payload, dict):
            return payload
        return None

    def _record_snapshot_complete(self, owner: str, repo: str, ref: str, archive_url: str) -> None:
        marker = self._repo_snapshot_marker_path(owner, repo, ref)
        self._write_json_file(
            marker,
            {
                "schema_version": self.SNAPSHOT_CACHE_SCHEMA_VERSION,
                "timestamp": time.time(),
                "ref": ref,
                "archive_url": archive_url,
            },
        )

    def _normalize_resolved_path(self, path: Path) -> str:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path.absolute()
        return str(resolved).replace("\\", "/").rstrip("/")

    def _format_bytes(self, size_bytes: int | None) -> str:
        if not size_bytes or size_bytes <= 0:
            return "unknown"
        units = ["B", "KiB", "MiB", "GiB"]
        value = float(size_bytes)
        unit = units[0]
        for candidate in units:
            unit = candidate
            if value < 1024 or candidate == units[-1]:
                break
            value /= 1024.0
        if unit == "B":
            return f"{int(value)} {unit}"
        return f"{value:.1f} {unit}"

    def _snapshot_archive_url(self, owner: str, repo: str, ref: str) -> str:
        return f"https://api.github.com/repos/{owner}/{repo}/zipball/{urllib.parse.quote(ref)}"

    def _snapshot_lock_for(self, owner: str, repo: str, ref: str) -> threading.Lock:
        cache_key = self._snapshot_cache_key(owner, repo, ref)
        with self._snapshot_lock_guard:
            lock = self._snapshot_locks.get(cache_key)
            if lock is None:
                lock = threading.Lock()
                self._snapshot_locks[cache_key] = lock
            return lock

    def _purge_snapshot_cache(self, owner: str, repo: str, ref: str) -> None:
        snapshot_dir = self._repo_snapshot_cache_dir(owner, repo, ref)
        if not snapshot_dir.exists():
            return
        shutil.rmtree(snapshot_dir, ignore_errors=False)

    def _download_snapshot_archive(self, owner: str, repo: str, ref: str, destination: Path) -> str:
        archive_url = self._snapshot_archive_url(owner, repo, ref)
        request = urllib.request.Request(archive_url, headers=self._headers())
        destination.parent.mkdir(parents=True, exist_ok=True)
        started_at = time.time()
        self._log(f"repo snapshot download repo={owner}/{repo} ref={ref[:12]}", "")
        temp_destination = destination.with_suffix(destination.suffix + ".part")
        if temp_destination.exists():
            temp_destination.unlink()
        stop_event = threading.Event()

        def heartbeat() -> None:
            while not stop_event.wait(self.snapshot_progress_heartbeat_seconds):
                elapsed = max(1, int(time.time() - started_at))
                self._log(f"repo snapshot downloading repo={owner}/{repo} ref={ref[:12]} elapsed={elapsed}s", "")

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temp_destination.open("wb") as handle:
                final_url = response.geturl()
                archive_size: int | None = None
                try:
                    archive_size = int(response.headers.get("Content-Length") or "0")
                except Exception:
                    archive_size = None
                self._log(
                    f"repo snapshot size repo={owner}/{repo} ref={ref[:12]} size={self._format_bytes(archive_size)}",
                    "",
                )
                downloaded = 0
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
            if archive_size and downloaded != archive_size:
                try:
                    temp_destination.unlink()
                except Exception:
                    pass
                raise RuntimeError(
                    f"incomplete_snapshot_download: expected={archive_size} downloaded={downloaded} repo={owner}/{repo} ref={ref}"
                )
            try:
                with zipfile.ZipFile(temp_destination) as archive:
                    if not archive.namelist():
                        raise RuntimeError("empty_snapshot_archive")
            except Exception as exc:
                try:
                    temp_destination.unlink()
                except Exception:
                    pass
                raise RuntimeError(f"invalid_snapshot_archive: {exc}") from exc
            temp_destination.replace(destination)
            return final_url or archive_url
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=0.2)

    def _extract_snapshot_archive(self, archive_path: Path, snapshot_root: Path) -> None:
        staging_dir = snapshot_root.parent / "_extract"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if snapshot_root.exists():
            shutil.rmtree(snapshot_root)
        staging_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(staging_dir)
        children = [item for item in staging_dir.iterdir() if item.exists()]
        if len(children) != 1 or not children[0].is_dir():
            raise RuntimeError(f"unexpected_snapshot_layout: {archive_path}")
        shutil.move(str(children[0]), str(snapshot_root))
        shutil.rmtree(staging_dir, ignore_errors=True)

    def _ensure_snapshot_for_ref(self, owner: str, repo: str, ref: str) -> tuple[Path, str]:
        wanted_ref = (ref or "").strip()
        if not wanted_ref:
            raise RuntimeError("snapshot_ref_required")
        cache_key = self._snapshot_cache_key(owner, repo, wanted_ref)
        with self._snapshot_lock_for(owner, repo, wanted_ref):
            snapshot_root = self._repo_snapshot_root_path(owner, repo, wanted_ref)
            marker = self._read_snapshot_complete_marker(owner, repo, wanted_ref)
            if (
                marker
                and int(marker.get("schema_version") or 0) == self.SNAPSHOT_CACHE_SCHEMA_VERSION
                and snapshot_root.exists()
            ):
                if cache_key not in self._snapshot_cache_hit_logged:
                    self._log(f"repo snapshot cache hit repo={owner}/{repo} ref={wanted_ref[:12]}", "")
                    self._snapshot_cache_hit_logged.add(cache_key)
                return snapshot_root, wanted_ref
            try:
                self._purge_snapshot_cache(owner, repo, wanted_ref)
            except Exception:
                pass
            archive_path = self._repo_snapshot_zip_path(owner, repo, wanted_ref)
            final_url = self._download_snapshot_archive(owner, repo, wanted_ref, archive_path)
            self._extract_snapshot_archive(archive_path, snapshot_root)
            self._record_snapshot_complete(owner, repo, wanted_ref, final_url)
            self._log(f"repo snapshot ready repo={owner}/{repo} ref={wanted_ref[:12]}", "")
            return snapshot_root, wanted_ref

    def _probe_repo_cache(self, repo_dir: Path) -> dict[str, Any]:
        if not repo_dir.exists():
            return {"ok": False, "reason": "missing_repo_dir"}
        worktree_git_dir = repo_dir / ".git"
        bare_git_markers = [repo_dir / "HEAD", repo_dir / "objects", repo_dir / "refs"]
        if not worktree_git_dir.exists() and not all(path.exists() for path in bare_git_markers):
            return {"ok": False, "reason": "missing_git_metadata"}

        git_dir_probe = self._run_git(repo_dir, ["rev-parse", "--absolute-git-dir"], check=False)
        if git_dir_probe.returncode != 0:
            return {"ok": False, "reason": "git_dir_probe_failed"}
        git_dir_text = (git_dir_probe.stdout or "").strip()
        if not git_dir_text:
            return {"ok": False, "reason": "empty_git_dir"}

        bare_probe = self._run_git(repo_dir, ["rev-parse", "--is-bare-repository"], check=False)
        if bare_probe.returncode != 0:
            return {"ok": False, "reason": "bare_probe_failed"}
        is_bare = (bare_probe.stdout or "").strip().lower() == "true"

        actual_git_dir = Path(git_dir_text)
        if not actual_git_dir.is_absolute():
            actual_git_dir = (repo_dir / actual_git_dir).resolve()
        expected_git_dir = repo_dir.resolve() if is_bare else (repo_dir / ".git").resolve()
        if self._normalize_resolved_path(actual_git_dir) != self._normalize_resolved_path(expected_git_dir):
            return {
                "ok": False,
                "reason": "git_dir_mismatch",
                "actual_git_dir": str(actual_git_dir),
                "expected_git_dir": str(expected_git_dir),
            }

        origin_probe = self._run_git(repo_dir, ["config", "--get", "remote.origin.url"], check=False)
        if origin_probe.returncode != 0:
            return {"ok": False, "reason": "missing_origin_remote"}
        head_probe = self._run_git(repo_dir, ["rev-parse", "--verify", "HEAD^{commit}"], check=False)
        if head_probe.returncode != 0:
            return {"ok": False, "reason": "missing_head_commit"}
        return {
            "ok": True,
            "reason": "ok",
            "remote_url": (origin_probe.stdout or "").strip(),
            "is_bare": is_bare,
        }

    def _purge_repo_cache(self, owner: str, repo: str) -> None:
        repo_dir = self._repo_cache_path(owner, repo)
        meta_dir = self._repo_cache_meta_dir(owner, repo)
        failures: list[str] = []

        def _on_rm_error(func: Any, path: str, exc_info: Any) -> None:
            try:
                os.chmod(path, 0o777)
                func(path)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                failures.append(f"{path}: {exc}")

        for path in (repo_dir, meta_dir):
            if not path.exists():
                continue
            removed = False
            for _attempt in range(3):
                try:
                    shutil.rmtree(path, onerror=_on_rm_error)
                except Exception as exc:
                    failures.append(f"{path}: {exc}")
                if not path.exists():
                    removed = True
                    break
                time.sleep(0.2)
            if not removed and path.exists():
                failures.append(f"{path}: still exists after cleanup")
        if repo_dir.exists() or meta_dir.exists() or failures:
            detail = "; ".join(failures) if failures else "cache directories still exist after cleanup"
            raise RuntimeError(f"repo cache purge failed for {owner}/{repo}: {detail}")

    def _is_repo_cache_complete(self, owner: str, repo: str, repo_dir: Path) -> bool:
        marker = self._read_repo_complete_marker(owner, repo)
        probe = self._probe_repo_cache(repo_dir)
        if not probe.get("ok"):
            return False
        if marker and int(marker.get("schema_version") or 0) == self.REPO_CACHE_SCHEMA_VERSION:
            return True
        # Upgrade older but healthy cache layouts in place so they can be reused.
        self._record_repo_complete(owner, repo, str(probe.get("remote_url") or "unknown"))
        return True

    def _remote_urls(self, owner: str, repo: str) -> list[str]:
        urls: list[str] = []
        template = (os.environ.get("SATD_GIT_REMOTE_TEMPLATE") or "").strip()
        if template:
            urls.append(template.format(owner=owner, repo=repo))
        base = (os.environ.get("SATD_GIT_REMOTE_BASE") or "").strip().rstrip("/")
        if base:
            urls.append(f"{base}/{owner}/{repo}.git")
        default_urls = [
            f"https://mirrors.tuna.tsinghua.edu.cn/git/github.com/{owner}/{repo}.git",
            f"https://github.com/{owner}/{repo}.git",
        ]
        for item in default_urls:
            if item not in urls:
                urls.append(item)
        return urls

    def _ensure_local_repo(self, owner: str, repo: str) -> Path:
        repo_dir = self._repo_cache_path(owner, repo)
        if repo_dir.exists():
            if self._is_repo_cache_complete(owner, repo, repo_dir):
                self._clear_clone_failure(owner, repo)
                return repo_dir
            probe = self._probe_repo_cache(repo_dir)
            reason = str(probe.get("reason") or "unknown")
            self._log(f"repo cache invalid repo={owner}/{repo} reason={reason}; rebuilding cache", "")
            self._purge_repo_cache(owner, repo)
            repo_dir = self._repo_cache_path(owner, repo)
            if repo_dir.exists():
                raise RuntimeError(f"repo cache cleanup did not remove {repo_dir}")
        cached_failure = self._should_skip_clone_due_to_failure_cache(owner, repo)
        if cached_failure:
            raise RuntimeError(cached_failure)
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        if repo_dir.exists():
            self._purge_repo_cache(owner, repo)
            repo_dir = self._repo_cache_path(owner, repo)
            if repo_dir.exists():
                raise RuntimeError(f"repo cache cleanup did not remove {repo_dir}")
        remote_urls = self._remote_urls(owner, repo)
        last_error = ""
        for remote_url in remote_urls:
            self._log(f"repo cache clone repo={owner}/{repo} remote={remote_url}", "")
            try:
                self._run_git_with_progress(
                    None,
                    ["clone", "--progress", "--filter=blob:none", "--no-checkout", remote_url, str(repo_dir)],
                    progress_label=f"repo clone {owner}/{repo}",
                )
                self._record_repo_complete(owner, repo, remote_url)
                self._clear_clone_failure(owner, repo)
                return repo_dir
            except Exception as exc:
                last_error = str(exc)
                self._log(f"repo cache clone failed repo={owner}/{repo} remote={remote_url} error={last_error}", "")
                try:
                    self._purge_repo_cache(owner, repo)
                except Exception as purge_exc:
                    raise RuntimeError(f"{last_error} | {purge_exc}") from purge_exc
                repo_dir = self._repo_cache_path(owner, repo)
                continue
        if last_error:
            self._record_clone_failure(owner, repo, last_error)
            raise RuntimeError(last_error)
        self._clear_clone_failure(owner, repo)
        return repo_dir

    def _run_git(self, repo_dir: Path | None, args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        command = ["git", *args]
        return subprocess.run(
            command,
            cwd=str(repo_dir) if repo_dir else None,
            check=check,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )

    def _run_git_with_progress(
        self,
        repo_dir: Path | None,
        args: list[str],
        progress_label: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = ["git", *args]
        process = subprocess.Popen(
            command,
            cwd=str(repo_dir) if repo_dir else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        stderr_chunks: list[str] = []
        progress_buffer = ""
        stop_event = threading.Event()
        started_at = time.time()

        def heartbeat() -> None:
            while not stop_event.wait(self.git_progress_heartbeat_seconds):
                elapsed = int(time.time() - started_at)
                self._log(f"{progress_label} still running... {elapsed}s elapsed", "")

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        assert process.stderr is not None
        try:
            while True:
                chunk = process.stderr.read(1)
                if chunk == "":
                    break
                stderr_chunks.append(chunk)
                progress_buffer += chunk
                if chunk not in {"\r", "\n"}:
                    continue
                message = progress_buffer.strip()
                progress_buffer = ""
                if not message:
                    continue
                self._log(f"{progress_label} {message}", "")
            if progress_buffer.strip():
                self._log(f"{progress_label} {progress_buffer.strip()}", "")
            stdout_text = ""
            if process.stdout is not None:
                stdout_text = process.stdout.read()
            return_code = process.wait()
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=0.2)
        completed = subprocess.CompletedProcess(
            command,
            return_code,
            stdout_text,
            "".join(stderr_chunks),
        )
        if check and return_code != 0:
            raise subprocess.CalledProcessError(
                return_code,
                command,
                output=stdout_text,
                stderr=completed.stderr,
            )
        return completed

    def _git_stdout(self, repo_dir: Path, args: list[str]) -> str:
        completed = self._run_git(repo_dir, args)
        return (completed.stdout or "").strip()

    def _local_ref_exists(self, repo_dir: Path, ref: str) -> bool:
        if not ref:
            return False
        completed = self._run_git(repo_dir, ["rev-parse", "--verify", f"{ref}^{{commit}}"], check=False)
        return completed.returncode == 0

    def _resolve_ref_for_repo(self, owner: str, repo: str, ref: str | None) -> tuple[Path, str]:
        repo_dir = self._ensure_local_repo(owner, repo)
        wanted_ref = (ref or "").strip() or "HEAD"
        ref_cache_path = self._repo_ref_cache_path(owner, repo)
        ref_cache = self._read_json_file(ref_cache_path)
        if isinstance(ref_cache, dict):
            cached_resolved = str(ref_cache.get(wanted_ref) or "").strip()
            if cached_resolved and self._local_ref_exists(repo_dir, cached_resolved):
                return repo_dir, cached_resolved
        if self._local_ref_exists(repo_dir, wanted_ref):
            if isinstance(ref_cache, dict):
                ref_cache[wanted_ref] = wanted_ref
            else:
                ref_cache = {wanted_ref: wanted_ref}
            self._write_json_file(ref_cache_path, ref_cache)
            return repo_dir, wanted_ref
        fetch_specs: list[list[str]] = []
        if ref:
            fetch_specs.append(["fetch", "--depth", "1", "origin", ref])
            fetch_specs.append(["fetch", "origin", ref])
        else:
            fetch_specs.append(["fetch", "--depth", "1", "origin"])
            fetch_specs.append(["fetch", "origin"])
        last_error = ""
        for fetch_args in fetch_specs:
            self._log(f"repo cache fetch repo={owner}/{repo} ref={wanted_ref[:12]}", "")
            progress_args = list(fetch_args)
            if "--progress" not in progress_args:
                progress_args.insert(1, "--progress")
            completed = self._run_git_with_progress(
                repo_dir,
                progress_args,
                progress_label=f"repo fetch {owner}/{repo}",
                check=False,
            )
            last_error = (completed.stderr or completed.stdout or "").strip()
            if completed.returncode == 0:
                if self._local_ref_exists(repo_dir, wanted_ref):
                    cache_payload = ref_cache if isinstance(ref_cache, dict) else {}
                    cache_payload[wanted_ref] = wanted_ref
                    self._write_json_file(ref_cache_path, cache_payload)
                    return repo_dir, wanted_ref
                if self._local_ref_exists(repo_dir, "FETCH_HEAD"):
                    fetched_ref = self._git_stdout(repo_dir, ["rev-parse", "FETCH_HEAD"])
                    if fetched_ref:
                        cache_payload = ref_cache if isinstance(ref_cache, dict) else {}
                        cache_payload[wanted_ref] = fetched_ref
                        self._write_json_file(ref_cache_path, cache_payload)
                        return repo_dir, fetched_ref
        if ref:
            raise RuntimeError(f"historical_ref_missing: {last_error or ref}")
        raise RuntimeError(last_error or "unable to fetch repository HEAD")

    def _fetch_repo_file_from_snapshot(self, owner: str, repo: str, path: str, ref: str) -> dict[str, Any]:
        snapshot_root, resolved_ref = self._ensure_snapshot_for_ref(owner, repo, ref)
        normalized_path = (path or "").replace("\\", "/").strip("/")
        cache_path = self._repo_file_cache_path(owner, repo, resolved_ref, normalized_path)
        cached_payload = self._read_json_file(cache_path)
        if isinstance(cached_payload, dict) and cached_payload.get("ok"):
            return dict(cached_payload)
        file_path = snapshot_root / Path(normalized_path)
        if not file_path.exists() or not file_path.is_file():
            return {"ok": False, "error": f"historical_file_missing: {normalized_path}", "path": normalized_path, "ref": ref}
        decoded = self._sanitize_source_text(file_path.read_text(encoding="utf-8", errors="replace"))
        sha = hashlib.sha1(decoded.encode("utf-8", errors="replace")).hexdigest()
        payload = {
            "ok": True,
            "path": normalized_path,
            "sha": sha,
            "download_url": f"https://github.com/{owner}/{repo}/blob/{resolved_ref}/{normalized_path}",
            "content_excerpt": decoded[:FILE_LIMIT],
            "full_content": decoded,
            "ref": ref,
            "resolved_ref": resolved_ref,
        }
        self._write_json_file(cache_path, payload)
        return payload

    def _fetch_repo_file_from_local_repo(self, owner: str, repo: str, path: str, ref: str | None) -> dict[str, Any]:
        repo_dir, resolved_ref = self._resolve_ref_for_repo(owner, repo, ref)
        normalized_path = (path or "").replace("\\", "/").strip("/")
        cache_path = self._repo_file_cache_path(owner, repo, resolved_ref, normalized_path)
        cached_payload = self._read_json_file(cache_path)
        if isinstance(cached_payload, dict) and cached_payload.get("ok"):
            return dict(cached_payload)
        blob_spec = f"{resolved_ref}:{normalized_path}"
        blob_proc = self._run_git(repo_dir, ["show", blob_spec], check=False)
        if blob_proc.returncode != 0:
            error_text = (blob_proc.stderr or blob_proc.stdout or "").strip()
            if ref:
                if self._local_ref_exists(repo_dir, resolved_ref):
                    error_text = f"historical_file_missing: {error_text or normalized_path}"
                else:
                    error_text = f"historical_ref_missing: {error_text or resolved_ref}"
            return {"ok": False, "error": error_text, "path": normalized_path, "ref": ref}
        sha = self._git_stdout(repo_dir, ["rev-parse", blob_spec])
        decoded = self._sanitize_source_text(blob_proc.stdout or "")
        payload = {
            "ok": True,
            "path": normalized_path,
            "sha": sha or None,
            "download_url": f"https://github.com/{owner}/{repo}/blob/{resolved_ref}/{normalized_path}",
            "content_excerpt": decoded[:FILE_LIMIT],
            "full_content": decoded,
            "ref": ref,
            "resolved_ref": resolved_ref,
        }
        self._write_json_file(cache_path, payload)
        return payload

    def _fetch_repo_tree_for_ref_from_snapshot(self, owner: str, repo: str, prefix: str, ref: str) -> dict[str, Any]:
        snapshot_root, resolved_ref = self._ensure_snapshot_for_ref(owner, repo, ref)
        normalized_prefix = prefix.strip("/")
        cache_path = self._repo_tree_cache_path(owner, repo, resolved_ref, normalized_prefix)
        cached_payload = self._read_json_file(cache_path)
        if isinstance(cached_payload, dict) and cached_payload.get("ok"):
            return dict(cached_payload)
        entries = []
        for file_path in snapshot_root.rglob("*"):
            if not file_path.is_file():
                continue
            relative_path = file_path.relative_to(snapshot_root).as_posix()
            if normalized_prefix and not (relative_path == normalized_prefix or relative_path.startswith(normalized_prefix + "/")):
                continue
            entries.append(
                {
                    "name": file_path.name,
                    "path": relative_path,
                    "type": "file",
                }
            )
        payload = {"ok": True, "prefix": normalized_prefix, "count": len(entries), "entries": entries, "ref": ref, "resolved_ref": resolved_ref}
        self._write_json_file(cache_path, payload)
        return payload

    def _fetch_repo_tree_for_ref_from_local_repo(self, owner: str, repo: str, prefix: str, ref: str) -> dict[str, Any]:
        repo_dir, resolved_ref = self._resolve_ref_for_repo(owner, repo, ref)
        normalized_prefix = prefix.strip("/")
        cache_path = self._repo_tree_cache_path(owner, repo, resolved_ref, normalized_prefix)
        cached_payload = self._read_json_file(cache_path)
        if isinstance(cached_payload, dict) and cached_payload.get("ok"):
            return dict(cached_payload)
        completed = self._run_git(repo_dir, ["ls-tree", "-r", "--name-only", resolved_ref], check=False)
        if completed.returncode != 0:
            error_text = (completed.stderr or completed.stdout or "").strip()
            return {"ok": False, "prefix": normalized_prefix, "error": error_text, "entries": [], "ref": ref}
        entries = []
        for raw_path in (completed.stdout or "").splitlines():
            clean_path = raw_path.strip()
            if not clean_path:
                continue
            if normalized_prefix and not (clean_path == normalized_prefix or clean_path.startswith(normalized_prefix + "/")):
                continue
            entries.append(
                {
                    "name": os.path.basename(clean_path),
                    "path": clean_path,
                    "type": "file",
                }
            )
        payload = {"ok": True, "prefix": normalized_prefix, "count": len(entries), "entries": entries, "ref": ref, "resolved_ref": resolved_ref}
        self._write_json_file(cache_path, payload)
        return payload

    def _github_json(self, url: str) -> Any:
        request = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(request, timeout=20) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return json.loads(response.read().decode(charset, errors="replace"))

    def _with_ref(self, url: str, ref: str | None) -> str:
        token = (ref or "").strip()
        if not token:
            return url
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}ref={urllib.parse.quote(token)}"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": self.user_agent,
        }
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        return headers

    def _sanitize_source_text(self, text: str) -> str:
        if not text:
            return ""
        return text.replace("\x00", "")

    def _decode_content(self, payload: dict[str, Any]) -> str:
        content = payload.get("content", "")
        if not content:
            return ""
        decoded = base64.b64decode(content).decode("utf-8", errors="replace")
        return self._sanitize_source_text(decoded)

    def _expand_search_items(
        self,
        owner: str,
        repo: str,
        items: list[dict[str, Any]],
        anchor_text: str,
        symbol_name: str = "",
        prefer_assert: bool = False,
        ref: str | None = None,
    ) -> list[dict[str, Any]]:
        enriched = []
        seen_paths: set[str] = set()
        for item in items:
            path = item.get("path") or ""
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            snippet = self._fetch_code_snippet(
                owner,
                repo,
                path,
                ref,
                anchor_text=anchor_text,
                symbol_name=symbol_name,
                prefer_assert=prefer_assert,
                window=SNIPPET_WINDOW,
                max_chars=SNIPPET_CHAR_LIMIT,
            )
            enriched.append(
                {
                    **item,
                    "snippet_ok": snippet.get("ok", False),
                    "match_reason": snippet.get("match_reason"),
                    "start_line": snippet.get("start_line"),
                    "end_line": snippet.get("end_line"),
                    "excerpt": snippet.get("excerpt"),
                    "symbol_name": snippet.get("symbol_name"),
                    "symbol_type": snippet.get("symbol_type"),
                }
            )
            if len(enriched) >= EXPANDED_ITEM_LIMIT:
                break
        return enriched

    @functools.lru_cache(maxsize=128)
    def _fetch_repo_tree_for_ref(self, owner: str, repo: str, prefix: str, ref: str) -> dict[str, Any]:
        normalized_prefix = prefix.strip("/")
        cache_ref = (ref or "").strip()
        if cache_ref:
            cache_path = self._repo_tree_cache_path(owner, repo, cache_ref, normalized_prefix)
            cached_payload = self._read_json_file(cache_path)
            if isinstance(cached_payload, dict) and cached_payload.get("ok"):
                return dict(cached_payload)
        endpoint = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{urllib.parse.quote(ref)}?recursive=1"
        try:
            payload = self._github_json(endpoint)
            tree = payload.get("tree", []) if isinstance(payload, dict) else []
            entries = []
            for item in tree:
                path = (item.get("path") or "").strip("/")
                if not path:
                    continue
                if normalized_prefix and not (path == normalized_prefix or path.startswith(normalized_prefix + "/")):
                    continue
                item_type = item.get("type")
                entries.append(
                    {
                        "name": os.path.basename(path),
                        "path": path,
                        "type": "file" if item_type == "blob" else "dir" if item_type == "tree" else item_type,
                    }
                )
            result = {"ok": True, "prefix": normalized_prefix, "count": len(entries), "entries": entries, "ref": ref, "resolved_ref": cache_ref or ref}
            if cache_ref:
                self._write_json_file(cache_path, result)
            return result
        except Exception as exc:
            error_text = str(exc)
            error_text = f"historical_ref_missing: {error_text}" if not self._ref_exists(owner, repo, ref) else error_text
            return {"ok": False, "prefix": prefix, "error": error_text, "entries": [], "ref": ref}

    @functools.lru_cache(maxsize=512)
    def _ref_exists(self, owner: str, repo: str, ref: str) -> bool:
        marker = self._read_snapshot_complete_marker(owner, repo, ref)
        snapshot_root = self._repo_snapshot_root_path(owner, repo, ref)
        if (
            marker
            and int(marker.get("schema_version") or 0) == self.SNAPSHOT_CACHE_SCHEMA_VERSION
            and snapshot_root.exists()
        ):
            return True
        endpoint = f"https://api.github.com/repos/{owner}/{repo}/commits/{urllib.parse.quote(ref)}"
        try:
            self._github_json(endpoint)
            return True
        except Exception:
            return False

    def _scan_historical_tree_for_matches(
        self,
        owner: str,
        repo: str,
        ref: str,
        query_token: str,
        symbol_name: str,
        per_page: int,
        candidate_kind: str,
        current_path: str = "",
    ) -> list[dict[str, Any]]:
        tree_payload = self._fetch_repo_tree_for_ref(owner, repo, "", ref)
        if not tree_payload.get("ok"):
            return []

        scored_paths = []
        for item in tree_payload.get("entries", []):
            path = item.get("path") or ""
            if item.get("type") != "file" or not self._looks_like_source_file(path):
                continue
            score = self._historical_candidate_score(path, query_token, symbol_name, candidate_kind, current_path)
            if score <= 0:
                continue
            scored_paths.append((score, path))

        scored_paths.sort(key=lambda pair: (-pair[0], pair[1]))
        candidate_budget = max(per_page * 3, EXPANDED_ITEM_LIMIT)
        anchor_text = f"{symbol_name}(" if candidate_kind == "callsite" and symbol_name else query_token
        prefer_assert = candidate_kind == "test"

        enriched = []
        for _, path in scored_paths[:candidate_budget]:
            snippet = self._fetch_code_snippet(
                owner,
                repo,
                path,
                ref,
                anchor_text=anchor_text,
                symbol_name=symbol_name if candidate_kind == "callsite" else "",
                prefer_assert=prefer_assert,
                window=SNIPPET_WINDOW,
                max_chars=SNIPPET_CHAR_LIMIT,
            )
            excerpt = snippet.get("excerpt") or ""
            if candidate_kind == "callsite" and symbol_name and symbol_name not in excerpt:
                continue
            if candidate_kind == "test" and not (snippet.get("match_reason") != "file_start" or query_token.lower() in path.lower()):
                continue
            enriched.append(
                {
                    "name": os.path.basename(path),
                    "path": path,
                    "sha": snippet.get("sha"),
                    "html_url": f"https://github.com/{owner}/{repo}/blob/{ref}/{path}",
                    "score": None,
                    "snippet_ok": snippet.get("ok", False),
                    "match_reason": snippet.get("match_reason"),
                    "start_line": snippet.get("start_line"),
                    "end_line": snippet.get("end_line"),
                    "excerpt": excerpt,
                    "symbol_name": snippet.get("symbol_name"),
                    "symbol_type": snippet.get("symbol_type"),
                }
            )
            if len(enriched) >= min(per_page, EXPANDED_ITEM_LIMIT):
                break
        return enriched

    def _historical_candidate_score(
        self,
        path: str,
        query_token: str,
        symbol_name: str,
        candidate_kind: str,
        current_path: str,
    ) -> int:
        lowered_path = (path or "").lower()
        token = (query_token or "").lower()
        symbol = (symbol_name or "").lower()
        current = (current_path or "").replace("\\", "/").lower()
        basename = os.path.basename(lowered_path)
        score = 0

        if candidate_kind == "test":
            if not self._is_test_path(path):
                return 0
            score += 3
            if token and token in lowered_path:
                score += 4
            if symbol and symbol in lowered_path:
                score += 2
            if basename.startswith("test_") or basename.endswith("_test.py"):
                score += 1
            return score

        if current and lowered_path == current:
            return 0
        if self._is_test_path(path):
            score -= 1
        if symbol and symbol in basename:
            score += 5
        elif symbol and symbol in lowered_path:
            score += 3
        if token and token in lowered_path:
            score += 2
        if current:
            current_prefix = os.path.dirname(current)
            if current_prefix and lowered_path.startswith(current_prefix):
                score += 2
        return score

    def _looks_like_source_file(self, path: str) -> bool:
        lowered = (path or "").lower()
        return lowered.endswith((".py", ".pyi")) or self._is_test_path(lowered)

    def _extract_best_snippet(
        self,
        content: str,
        anchor_text: str,
        symbol_name: str,
        prefer_assert: bool,
        window: int,
        max_chars: int,
    ) -> dict[str, Any]:
        content = self._sanitize_source_text(content)
        lines = content.splitlines()
        if not lines:
            return {"match_reason": "empty_file", "start_line": None, "end_line": None, "excerpt": "", "symbol_name": None, "symbol_type": None}

        symbol_snippet = self._extract_symbol_snippet(content, symbol_name)
        if symbol_snippet:
            symbol_snippet["excerpt"] = (symbol_snippet.get("excerpt") or "")[:max_chars]
            return symbol_snippet

        anchor_line = self._find_anchor_line(lines, anchor_text)
        if anchor_line is not None:
            return self._snippet_from_line(lines, anchor_line, window, "anchor_text", max_chars)

        if prefer_assert:
            assert_line = self._find_anchor_line(lines, "assert")
            if assert_line is not None:
                return self._snippet_from_line(lines, assert_line, window, "assert_window", max_chars)

        return self._snippet_from_line(lines, 1, min(window, 20), "file_start", max_chars)

    def _extract_symbol_snippet(self, content: str, symbol_name: str) -> dict[str, Any] | None:
        token = (symbol_name or "").strip()
        if not token:
            return None
        content = self._sanitize_source_text(content)
        try:
            tree = ast.parse(content)
        except (SyntaxError, ValueError):
            return None

        best: ast.AST | None = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and getattr(node, "name", "") == token:
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                if start is None or end is None:
                    continue
                if best is None:
                    best = node
                    continue
                current_span = getattr(best, "end_lineno") - getattr(best, "lineno")
                node_span = end - start
                if node_span < current_span:
                    best = node

        if best is None:
            return None

        source_lines = content.splitlines()
        start = getattr(best, "lineno")
        end = getattr(best, "end_lineno")
        return {
            "match_reason": "symbol_match",
            "start_line": start,
            "end_line": end,
            "excerpt": "\n".join(source_lines[start - 1 : end]),
            "symbol_name": getattr(best, "name", None),
            "symbol_type": type(best).__name__,
        }

    def _find_anchor_line(self, lines: list[str], anchor_text: str) -> int | None:
        target = self._normalize_search_text(anchor_text)
        if not target:
            return None
        for index, line in enumerate(lines, start=1):
            candidate = self._normalize_search_text(line)
            if candidate and (target in candidate or candidate in target):
                return index
        return None

    def _normalize_search_text(self, text: str) -> str:
        normalized = (text or "").strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _snippet_from_line(
        self,
        lines: list[str],
        line_number: int,
        window: int,
        match_reason: str,
        max_chars: int,
    ) -> dict[str, Any]:
        start_line = max(1, line_number - window)
        end_line = min(len(lines), line_number + window)
        excerpt = "\n".join(f"{idx}: {lines[idx - 1]}" for idx in range(start_line, end_line + 1))
        return {
            "match_reason": match_reason,
            "start_line": start_line,
            "end_line": end_line,
            "excerpt": excerpt[:max_chars],
            "symbol_name": None,
            "symbol_type": None,
        }

    def _locate_satd_line(self, lines: list[str], satd_comment: str) -> int | None:
        target = self._normalize_comment_text(satd_comment)
        if not target:
            return None
        for index, line in enumerate(lines, start=1):
            candidate = self._normalize_comment_text(line)
            if candidate and (target in candidate or candidate in target):
                return index
        return None

    def _normalize_comment_text(self, text: str) -> str:
        normalized = (text or "").strip().lower()
        normalized = re.sub(r"^[#/*\-\s]+", "", normalized)
        normalized = normalized.replace("*/", " ")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _is_test_path(self, path: str) -> bool:
        lowered = (path or "").lower()
        return any(token in lowered for token in ("/test", "tests/", "_test.py", "test_"))


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
                    "path": path,
                    "symbol_name": symbol_name,
                    "qualified_name": str(item.get("qualified_name") or symbol_name),
                    "class_name": item.get("class_name"),
                    "start_line": item.get("start_line"),
                    "end_line": item.get("end_line"),
                }
                keys = {
                    symbol_name,
                    str(record["qualified_name"]),
                }
                if record.get("class_name"):
                    keys.add(f"{record['class_name']}.{symbol_name}")
                for key in [k for k in keys if str(k).strip()]:
                    index.setdefault(key, [])
                    if record not in index[key]:
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

    def extract_relevant_method_names(self, code: str, max_methods: int) -> list[str]:
        calls = self._extract_called_references(code)
        selected: list[str] = []
        seen: set[str] = set()
        for item in calls:
            normalized = self._normalize_method_reference(item.get("qualified_name") or item.get("raw_call") or "")
            if not normalized or normalized in seen or self._is_low_quality_method_reference(normalized):
                continue
            seen.add(normalized)
            selected.append(normalized)
            if len(selected) >= max_methods:
                break
        return selected

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

        def visit(node: Any) -> None:
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
            for child in node.children:
                visit(child)

        visit(tree.root_node)
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

    def _is_low_quality_method_reference(self, value: str) -> bool:
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
        return parts[-1].lower() in generic_terms

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

    def _extract_all_method_matches(self, content: str) -> list[dict[str, Any]]:
        matches = self._extract_tree_sitter_method_matches(content)
        if matches:
            return matches
        return self._extract_regex_method_matches(self._sanitize_source_text(content))

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

        def visit(node: Any, enclosing_class: str | None = None) -> None:
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
                    for child in body.named_children:
                        visit(child, symbol_name or enclosing_class)
                return
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
            for child in node.children:
                visit(child, enclosing_class)

        visit(tree.root_node)
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

    def _find_enclosing_class_name(self, tree: ast.AST, target: ast.AST) -> str | None:
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for child in getattr(node, "body", []):
                if child is target:
                    return node.name
        return None

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

