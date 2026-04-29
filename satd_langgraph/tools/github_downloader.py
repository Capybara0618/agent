from __future__ import annotations

import base64
import functools
import hashlib
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from ..local_settings import GITHUB_TOKEN as LOCAL_GITHUB_TOKEN


FILE_LIMIT = 12000
TREE_LIMIT = 40


class GitHubDownloadTool:
    """Download repository files and trees from GitHub with local cache support."""

    def __init__(self) -> None:
        token = os.environ.get("GITHUB_TOKEN") or LOCAL_GITHUB_TOKEN
        self.github_token = None if token == "PASTE_YOUR_GITHUB_TOKEN_HERE" else token
        self.user_agent = "satd-langgraph-agent"
        cache_root = os.environ.get("SATD_REPO_CACHE_DIR")
        if cache_root:
            self.repo_cache_dir = Path(cache_root).expanduser()
        else:
            self.repo_cache_dir = Path(__file__).resolve().parent.parent.parent / ".repo_cache"
        self.repo_cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_repo_file(self, owner: str, repo: str, path: str, ref: str | None = None) -> dict[str, Any]:
        return self._fetch_repo_file(owner, repo, path, ref)

    def fetch_repo_tree(self, owner: str, repo: str, prefix: str = "", ref: str | None = None) -> dict[str, Any]:
        return self._fetch_repo_tree(owner, repo, prefix, ref)

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

    def _repo_cache_meta_dir(self, owner: str, repo: str) -> Path:
        return self.repo_cache_dir / owner / f"{repo}.__satd_cache__"

    def _repo_tree_cache_path(self, owner: str, repo: str, resolved_ref: str, prefix: str = "") -> Path:
        ref_key = self._safe_cache_key(resolved_ref)
        prefix_key = self._safe_cache_key(prefix or "__root__")
        return self._repo_cache_meta_dir(owner, repo) / "trees" / ref_key / f"{prefix_key}.json"

    def _repo_file_cache_path(self, owner: str, repo: str, resolved_ref: str, path: str) -> Path:
        ref_key = self._safe_cache_key(resolved_ref)
        normalized_path = (path or "").replace("\\", "/").strip("/")
        return self._repo_cache_meta_dir(owner, repo) / "files" / ref_key / Path(normalized_path).with_suffix(Path(normalized_path).suffix + ".json")

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

    def _ref_exists(self, owner: str, repo: str, ref: str) -> bool:
        endpoint = f"https://api.github.com/repos/{owner}/{repo}/commits/{urllib.parse.quote(ref)}"
        try:
            self._github_json(endpoint)
            return True
        except Exception:
            return False

    def _looks_like_source_file(self, path: str) -> bool:
        lowered = (path or "").lower()
        return lowered.endswith((".py", ".pyi")) or self._is_test_path(lowered)

    def _is_test_path(self, path: str) -> bool:
        lowered = (path or "").lower()
        return any(token in lowered for token in ("/test", "tests/", "_test.py", "test_"))

