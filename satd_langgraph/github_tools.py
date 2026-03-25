from __future__ import annotations

import ast
import base64
import functools
import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any

from .local_settings import GITHUB_TOKEN as LOCAL_GITHUB_TOKEN


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
    def __init__(self) -> None:
        token = os.environ.get("GITHUB_TOKEN") or LOCAL_GITHUB_TOKEN
        self.github_token = None if token == "PASTE_YOUR_GITHUB_TOKEN_HERE" else token
        self.user_agent = "satd-langgraph-agent"

    def fetch_repo_readme(self, owner: str, repo: str) -> dict[str, Any]:
        return self._fetch_repo_readme(owner, repo)

    def fetch_readme_or_module_docs(self, owner: str, repo: str, path_prefix: str) -> dict[str, Any]:
        return self._fetch_readme_or_module_docs(owner, repo, path_prefix)

    def fetch_repo_file(self, owner: str, repo: str, path: str) -> dict[str, Any]:
        return self._fetch_repo_file(owner, repo, path)

    def fetch_repo_file_full_or_window(
        self,
        owner: str,
        repo: str,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        max_chars: int = FILE_LIMIT,
    ) -> dict[str, Any]:
        file_payload = self._fetch_repo_file(owner, repo, path)
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

    def find_related_tests(self, owner: str, repo: str, symbol_name_or_path: str, per_page: int = SEARCH_LIMIT) -> dict[str, Any]:
        return self._find_related_tests(owner, repo, symbol_name_or_path, per_page)

    def find_call_sites(self, owner: str, repo: str, symbol_name: str, per_page: int = SEARCH_LIMIT) -> dict[str, Any]:
        return self._find_call_sites(owner, repo, symbol_name, per_page)

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

    def fetch_repo_tree(self, owner: str, repo: str, prefix: str = "") -> dict[str, Any]:
        return self._fetch_repo_tree(owner, repo, prefix)

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
        anchor_text: str = "",
        symbol_name: str = "",
        prefer_assert: bool = False,
        window: int = SNIPPET_WINDOW,
        max_chars: int = SNIPPET_CHAR_LIMIT,
    ) -> dict[str, Any]:
        return self._fetch_code_snippet(owner, repo, path, anchor_text, symbol_name, prefer_assert, window, max_chars)

    @functools.lru_cache(maxsize=256)
    def _fetch_repo_readme(self, owner: str, repo: str) -> dict[str, Any]:
        endpoint = f"https://api.github.com/repos/{owner}/{repo}/readme"
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
    def _fetch_readme_or_module_docs(self, owner: str, repo: str, path_prefix: str) -> dict[str, Any]:
        candidates = []
        cleaned_prefix = (path_prefix or "").strip("/")
        if cleaned_prefix:
            candidates.extend([
                f"{cleaned_prefix}/README.md",
                f"{cleaned_prefix}/README.rst",
                f"{cleaned_prefix}/docs/README.md",
            ])
        for candidate in candidates:
            payload = self._fetch_repo_file(owner, repo, candidate)
            if payload.get("ok"):
                return {
                    "ok": True,
                    "path": payload.get("path"),
                    "download_url": payload.get("download_url"),
                    "content_excerpt": (payload.get("full_content") or "")[:README_LIMIT],
                }
        return self._fetch_repo_readme(owner, repo)

    @functools.lru_cache(maxsize=1024)
    def _fetch_repo_file(self, owner: str, repo: str, path: str) -> dict[str, Any]:
        encoded_path = "/".join(urllib.parse.quote(part) for part in path.split("/"))
        endpoint = f"https://api.github.com/repos/{owner}/{repo}/contents/{encoded_path}"
        try:
            payload = self._github_json(endpoint)
            decoded = self._decode_content(payload)
            return {
                "ok": True,
                "path": payload.get("path"),
                "sha": payload.get("sha"),
                "download_url": payload.get("download_url"),
                "content_excerpt": decoded[:FILE_LIMIT],
                "full_content": decoded,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "path": path}

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
    def _find_related_tests(self, owner: str, repo: str, symbol_name_or_path: str, per_page: int) -> dict[str, Any]:
        token = (symbol_name_or_path or "").strip()
        if not token:
            return {"ok": True, "query": "", "count": 0, "items": []}
        basename = os.path.basename(token).replace(".py", "")
        symbol_name = basename if basename and basename != token else ""
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
    def _find_call_sites(self, owner: str, repo: str, symbol_name: str, per_page: int) -> dict[str, Any]:
        token = (symbol_name or "").strip()
        if not token:
            return {"ok": True, "query": "", "count": 0, "items": []}
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
    def _fetch_repo_tree(self, owner: str, repo: str, prefix: str) -> dict[str, Any]:
        cleaned_prefix = prefix.strip("/")
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
        anchor_text: str,
        symbol_name: str,
        prefer_assert: bool,
        window: int,
        max_chars: int,
    ) -> dict[str, Any]:
        payload = self._fetch_repo_file(owner, repo, path)
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

    def _github_json(self, url: str) -> Any:
        request = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(request, timeout=20) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return json.loads(response.read().decode(charset, errors="replace"))

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
