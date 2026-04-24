from __future__ import annotations

import ast
import builtins
import copy
import json
import re
import os
import re
import textwrap
import time
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

from .github_tools import GitHubToolbox
from .local_settings import OPENAI_API_KEY as LOCAL_OPENAI_API_KEY
from .local_settings import OPENAI_BASE_URL as LOCAL_OPENAI_BASE_URL
from .schema import (
    AnalysisResult,
    EditConstraint,
    GraphState,
    MethodInquiryResult,
    RepairAttempt,
    RetrievedMethodContext,
    ReviewResult,
    SelectorDecision,
    UncertaintyItem,
    preprocess_python_code,
)

MODEL_ALIASES = {"gpt-4o-mini-global": "gpt-4o-mini"}


class OpenAICompatClient:
    def __init__(self, model: str = "gpt-4o-mini", verbose: bool = False) -> None:
        api_key = os.environ.get("OPENAI_API_KEY") or LOCAL_OPENAI_API_KEY
        if api_key == "PASTE_YOUR_OPENAI_COMPAT_KEY_HERE":
            api_key = None
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        base_url = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or LOCAL_OPENAI_BASE_URL
            or None
        )
        if base_url == "https://your-openai-compatible-host/v1":
            base_url = None
        self.request_timeout = float(os.environ.get("OPENAI_TIMEOUT_SECONDS") or 60)
        self.max_attempts = max(1, int(os.environ.get("OPENAI_MAX_ATTEMPTS") or 1))
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.request_timeout)
        self.requested_model = model
        self.model = self._normalize_model_name(model)
        self.verbose = bool(verbose)
        self.toolbox = GitHubToolbox()

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        request_label: str = "",
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        sanitized = False
        active_system_prompt = self._coerce_prompt_text(system_prompt)
        active_user_prompt = self._coerce_prompt_text(user_prompt)
        active_model = self.model
        label = request_label or "llm_request"
        for attempt in range(self.max_attempts):
            attempt_started = time.time()
            self._emit_log(
                f"[llm] start label={label} attempt={attempt + 1}/{self.max_attempts} model={active_model} timeout={self.request_timeout:.0f}s"
            )
            try:
                response = self.client.chat.completions.create(
                    model=active_model,
                    messages=[
                        {"role": "system", "content": active_system_prompt},
                        {"role": "user", "content": active_user_prompt},
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                elapsed = time.time() - attempt_started
                self._emit_log(f"[llm] success label={label} attempt={attempt + 1}/{self.max_attempts} elapsed={elapsed:.2f}s")
                return json.loads(content)
            except Exception as exc:
                last_error = exc
                elapsed = time.time() - attempt_started
                self._emit_log(
                    f"[llm] error label={label} attempt={attempt + 1}/{self.max_attempts} "
                    f"elapsed={elapsed:.2f}s type={type(exc).__name__} message={self._short_error(exc)}"
                )
                fallback_model = self._fallback_model_for_error(exc, active_model)
                if fallback_model and fallback_model != active_model:
                    self._emit_log(f"[llm] fallback_model label={label} from={active_model} to={fallback_model}")
                    active_model = fallback_model
                    self.model = fallback_model
                    continue
                if self._is_content_filter_error(exc) and not sanitized:
                    self._emit_log(f"[llm] sanitize_prompts label={label}")
                    active_system_prompt, active_user_prompt = self._sanitize_prompts(system_prompt, user_prompt)
                    sanitized = True
                    continue
                if attempt < self.max_attempts - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("OpenAI-compatible request failed unexpectedly.")

    def _emit_log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _short_error(self, exc: Exception) -> str:
        message = " ".join(str(exc).split())
        if len(message) > 220:
            return message[:217] + "..."
        return message

    def _normalize_model_name(self, model: str) -> str:
        cleaned = (model or "gpt-4o-mini").strip()
        return MODEL_ALIASES.get(cleaned, cleaned)

    def _fallback_model_for_error(self, exc: Exception, active_model: str) -> str | None:
        message = str(exc).lower()
        normalized = self._normalize_model_name(active_model)
        if "unknown model" in message and normalized != active_model:
            return normalized
        if "unknown model" in message and active_model.endswith("-global"):
            return active_model[: -len("-global")]
        return None

    def _is_content_filter_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "content_filter" in message or "content management policy" in message

    def _sanitize_prompts(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:
        compact_system = self._coerce_prompt_text(system_prompt) + " Prefer concise evidence use and avoid unnecessary raw excerpts."
        compact_user = self._coerce_prompt_text(user_prompt)
        compact_user = re.sub(
            r"(Base \+ repair \+ review context:\n)[\s\S]*",
            r"\1[context compacted for safety; rely on metadata and direct task evidence]",
            compact_user,
        )
        compact_user = re.sub(
            r"(Unified GitHub context:\n)[\s\S]*",
            r"\1[context compacted for safety; rely on metadata and direct task evidence]",
            compact_user,
        )
        compact_user = re.sub(
            r"(Base \+ repair context:\n)[\s\S]*",
            r"\1[context compacted for safety; rely on metadata and direct task evidence]",
            compact_user,
        )
        compact_user = re.sub(
            r"(Base GitHub context:\n)[\s\S]*",
            r"\1[context compacted for safety; rely on metadata and direct task evidence]",
            compact_user,
        )
        compact_user = re.sub(
            r"(Original code block:\n)([\s\S]*?)(\n\n(?:Repaired code block:|Repair plan:|Base \+ repair context:|Base \+ repair \+ review context:))",
            lambda m: m.group(1) + m.group(2)[:1200] + m.group(3),
            compact_user,
        )
        compact_user = re.sub(
            r"(Current code snippet:\n)([\s\S]*?)(\n\nBase GitHub context:)",
            lambda m: m.group(1) + m.group(2)[:1200] + m.group(3),
            compact_user,
        )
        compact_user = re.sub(
            r"(Repaired code block:\n)([\s\S]*?)(\n\nRepair plan:)",
            lambda m: m.group(1) + m.group(2)[:1200] + m.group(3),
            compact_user,
        )
        compact_user = re.sub(r"\n{3,}", "\n\n", compact_user)
        return compact_system, compact_user

    def _coerce_prompt_text(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, (list, tuple)):
            return "".join(self._coerce_prompt_text(item) for item in prompt)
        if prompt is None:
            return ""
        return str(prompt)
    def build_base_context(self, state: GraphState, existing_bundle: dict[str, Any] | None = None) -> dict[str, Any]:
        bundle = self._ensure_bundle(existing_bundle, state)
        metadata = bundle.setdefault("metadata", {})
        active_commit = (state.get("commit") or "").strip()
        cached_commit = str(metadata.get("context_commit") or bundle.get("commit") or "").strip()
        if cached_commit != active_commit:
            bundle["base_context"] = {}
            bundle["repair_context"] = {}
            bundle["review_context"] = {}
            metadata["base_cached"] = False
            metadata["repair_cached"] = False
            metadata["review_cached"] = False
        if metadata.get("base_cached") and bundle.get("base_context"):
            if not metadata.get("base_cache_source"):
                metadata["base_cache_source"] = "memory"
            return self._augment_bundle_metadata(bundle)

        owner = state["user"]
        repo = state["project"]
        path = state["file_path"]
        ref = (state.get("commit") or "").strip() or None
        file_payload = self.toolbox.fetch_repo_file(owner, repo, path, ref=ref)
        file_content = file_payload.get("full_content", "") if file_payload.get("ok") else ""
        target_function = self._resolve_target_function(file_content, state)

        bundle["base_context"] = {
            "file_path": path,
            "original_signature": self._extract_original_signature(state["original_code"]),
            "target_file": self._summarize_file_payload(file_payload),
            "target_function": target_function,
        }
        metadata.update(
            {
                "base_cached": True,
                "base_context_fetched_at": self._timestamp(),
                "base_cache_source": metadata.get("base_cache_source") or "github_fetch",
                "context_commit": state.get("commit") or "",
                "symbol_name": target_function.get("symbol_name"),
                "satd_line": target_function.get("satd_line"),
            }
        )
        return self._augment_bundle_metadata(bundle)

    def ensure_repair_context(self, state: GraphState, bundle: dict[str, Any] | None) -> dict[str, Any]:
        bundle = self.build_base_context(state, bundle)
        metadata = bundle.setdefault("metadata", {})
        if metadata.get("repair_cached") and bundle.get("repair_context"):
            if not metadata.get("repair_cache_source"):
                metadata["repair_cache_source"] = "memory"
            return self._augment_bundle_metadata(bundle)

        owner = state["user"]
        repo = state["project"]
        path = state["file_path"]
        ref = (state.get("commit") or "").strip() or None
        file_payload = self.toolbox.fetch_repo_file(owner, repo, path, ref=ref)
        file_content = file_payload.get("full_content", "") if file_payload.get("ok") else ""
        target_function = ((bundle.get("base_context") or {}).get("target_function") or {}) if isinstance(bundle.get("base_context"), dict) else {}
        symbol_name = str(target_function.get("symbol_name") or metadata.get("symbol_name") or self._fallback_symbol_name(state))

        bundle["repair_context"] = {
            "context_strategy": "function_external_evidence",
            "same_file_helpers": self._extract_same_file_helpers(file_content, target_function, state["satd_comment"]),
            "same_class_evidence": self._extract_same_class_evidence(file_content, target_function, state["satd_comment"]),
            "module_symbols": self._extract_module_symbols(file_content, target_function, state["satd_comment"]),
            "same_file_pattern": self._extract_same_file_pattern(file_content, target_function, state["satd_comment"]),
            "targeted_test_snippet": self._extract_targeted_test_snippet(owner, repo, ref, state, symbol_name),
            "targeted_callsite_snippet": self._extract_targeted_callsite_snippet(owner, repo, path, ref, state, symbol_name),
        }
        metadata.update(
            {
                "repair_cached": True,
                "repair_context_fetched_at": self._timestamp(),
                "repair_cache_source": metadata.get("repair_cache_source") or "github_fetch",
                "context_commit": state.get("commit") or "",
            }
        )
        return self._augment_bundle_metadata(bundle)

    def ensure_review_context(
        self,
        state: GraphState,
        bundle: dict[str, Any] | None,
        analysis: AnalysisResult,
        repair: RepairAttempt,
    ) -> dict[str, Any]:
        bundle = self.ensure_repair_context(state, bundle)
        metadata = bundle.setdefault("metadata", {})
        if metadata.get("review_cached") and bundle.get("review_context"):
            if not metadata.get("review_cache_source"):
                metadata["review_cache_source"] = "memory"
            return self._augment_bundle_metadata(bundle)

        repair_context = bundle.get("repair_context", {}) or {}
        base_context = bundle.get("base_context", {}) or {}
        call_sites = repair_context.get("call_sites", {}) if isinstance(repair_context, dict) else {}
        related_tests = repair_context.get("related_tests", {}) if isinstance(repair_context, dict) else {}
        commits_for_path = repair_context.get("commits_for_path", {}) if isinstance(repair_context, dict) else {}
        history_snippets = repair_context.get("history_snippets", {}) if isinstance(repair_context, dict) else {}
        symbol = base_context.get("enclosing_symbol", {}) if isinstance(base_context, dict) else {}

        risk_indicators = []
        if analysis.risk_level == "high":
            risk_indicators.append("analysis_marked_high_risk")
        if analysis.historical_snapshot_mismatch:
            risk_indicators.append("historical_snapshot_mismatch")
        if (call_sites.get("count") or 0) > 5:
            risk_indicators.append("many_call_sites")
        if not (related_tests.get("count") or 0):
            risk_indicators.append("no_related_tests_found")
        if analysis.scope_radius == "multi_file":
            risk_indicators.append("multi_file_scope")
        if metadata.get("repair_evidence_mode") == "weak":
            risk_indicators.append("weak_repair_evidence")

        bundle["review_context"] = {
            "validation_signals_summary": {
                "analysis_validation_signals": analysis.validation_signals,
                "related_tests_count": related_tests.get("count", 0),
                "call_sites_count": call_sites.get("count", 0),
                "commits_count": commits_for_path.get("count", 0),
                "history_snippets_count": history_snippets.get("count", 0),
                "github_evidence_strength": analysis.github_evidence_strength,
                "snapshot_alignment_status": metadata.get("snapshot_alignment_status"),
                "repair_evidence_mode": metadata.get("repair_evidence_mode"),
            },
            "risk_indicators": risk_indicators,
            "change_scope_evidence": {
                "analysis_scope_radius": analysis.scope_radius,
                "repair_changed_scope": repair.changed_scope,
                "symbol_name": symbol.get("symbol_name"),
                "symbol_type": symbol.get("symbol_type"),
                "historical_snapshot_mismatch": analysis.historical_snapshot_mismatch,
            },
            "em_risk_guardrails": {
                "disallow_signature_changes_without_strong_evidence": True,
                "disallow_new_helpers_without_strong_evidence": True,
                "disallow_control_flow_rewrites_without_strong_evidence": True,
                "repair_evidence_mode": metadata.get("repair_evidence_mode"),
                "snapshot_alignment_status": metadata.get("snapshot_alignment_status"),
            },
        }
        metadata.update(
            {
                "review_cached": True,
                "review_context_fetched_at": self._timestamp(),
                "review_cache_source": metadata.get("review_cache_source") or "derived",
            }
        )
        return self._augment_bundle_metadata(bundle)

    def format_context(self, bundle: dict[str, Any] | None, layers: tuple[str, ...] | None = None) -> str:
        if not bundle:
            return ""
        selected_layers = layers or ("base_context", "repair_context", "review_context")
        parts = []
        metadata = bundle.get("metadata")
        if metadata:
            parts.append("[CONTEXT_METADATA]\n" + json.dumps(metadata, ensure_ascii=False))
        for layer_name in selected_layers:
            payload = bundle.get(layer_name)
            if payload:
                parts.append(f"[{layer_name.upper()}]\n" + json.dumps(payload, ensure_ascii=False))
        return "\n\n".join(parts)

    def compact_context_for_stage(self, bundle: dict[str, Any] | None, stage: str) -> str:
        if not bundle:
            return ""
        metadata = bundle.get("metadata") or {}
        base = bundle.get("base_context") or {}
        repair = bundle.get("repair_context") or {}
        review = bundle.get("review_context") or {}
        compact = {
            "metadata": {
                "context_strategy": metadata.get("context_strategy"),
                "historical_snapshot_mismatch": metadata.get("historical_snapshot_mismatch"),
                "snapshot_alignment_status": metadata.get("snapshot_alignment_status"),
                "repair_evidence_mode": metadata.get("repair_evidence_mode"),
                "github_evidence_strength": metadata.get("github_evidence_strength"),
                "target_file_ok": metadata.get("target_file_ok"),
                "target_function_found": metadata.get("target_function_found"),
                "same_file_helpers_count": metadata.get("same_file_helpers_count"),
                "same_class_methods_count": metadata.get("same_class_methods_count"),
                "same_class_attributes_count": metadata.get("same_class_attributes_count"),
                "module_symbols_count": metadata.get("module_symbols_count"),
                "same_file_pattern_count": metadata.get("same_file_pattern_count"),
                "decisive_external_evidence_count": metadata.get("decisive_external_evidence_count"),
                "targeted_test_snippet_count": metadata.get("targeted_test_snippet_count"),
                "targeted_callsite_snippet_count": metadata.get("targeted_callsite_snippet_count"),
                "retrieved_test_snippets_count": metadata.get("retrieved_test_snippets_count"),
                "retrieved_callsite_snippets_count": metadata.get("retrieved_callsite_snippets_count"),
                "symbol_name": metadata.get("symbol_name"),
                "satd_line": metadata.get("satd_line"),
            },
            "base_context": {
                "file_path": base.get("file_path"),
                "original_signature": base.get("original_signature"),
                "target_function": {
                    "found": (base.get("target_function") or {}).get("found"),
                    "symbol_name": (base.get("target_function") or {}).get("symbol_name"),
                    "class_name": (base.get("target_function") or {}).get("class_name"),
                    "start_line": (base.get("target_function") or {}).get("start_line"),
                    "end_line": (base.get("target_function") or {}).get("end_line"),
                    "matched_by": (base.get("target_function") or {}).get("matched_by"),
                },
            },
        }
        if stage in {"repair", "review"}:
            evidence_cards = self._build_repair_evidence_cards(repair)
            compact["repair_context"] = {
                "evidence_cards": evidence_cards,
                "same_file_helpers": {"count": (repair.get("same_file_helpers") or {}).get("count", 0)},
                "same_class_evidence": {
                    "related_methods_count": (repair.get("same_class_evidence") or {}).get("related_methods_count", 0),
                    "related_attributes_count": (repair.get("same_class_evidence") or {}).get("related_attributes_count", 0),
                },
                "module_symbols": {"count": (repair.get("module_symbols") or {}).get("count", 0)},
                "same_file_pattern": {"count": (repair.get("same_file_pattern") or {}).get("count", 0)},
                "targeted_test_snippet": {"used": bool((repair.get("targeted_test_snippet") or {}).get("used"))},
                "targeted_callsite_snippet": {"used": bool((repair.get("targeted_callsite_snippet") or {}).get("used"))},
            }
        if stage == "review":
            compact["review_context"] = review
        return json.dumps(compact, ensure_ascii=False)

    def compact_shared_context(self, bundle: dict[str, Any] | None) -> str:
        return self.compact_context_for_stage(bundle, "repair")

    def _ensure_bundle(self, bundle: dict[str, Any] | None, state: GraphState) -> dict[str, Any]:
        current = dict(bundle or {})
        current.setdefault("task_id", state["task_id"])
        current.setdefault("repo_owner", state["user"])
        current.setdefault("repo_name", state["project"])
        current.setdefault("file_path", state["file_path"])
        current.setdefault("commit", state.get("commit") or "")
        current.setdefault("metadata", {})
        current.setdefault("base_context", {})
        current.setdefault("repair_context", {})
        current.setdefault("review_context", {})
        return current

    def _extract_issue_numbers(self, text: str) -> list[int]:
        return [int(match) for match in re.findall(r"#(\d+)", text or "")]

    def _module_prefix(self, path: str) -> str:
        cleaned = (path or "").replace("\\", "/")
        if "/" not in cleaned:
            return ""
        return cleaned.rsplit("/", 1)[0]

    def _fallback_symbol_name(self, state: GraphState) -> str:
        match = re.search(r"(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", state["original_code"] or "")
        return match.group(1) if match else ""

    def _build_history_query(self, satd_comment: str, symbol_name: str) -> str:
        cleaned_comment = re.sub(r"[^a-zA-Z0-9_\s]", " ", satd_comment or "")
        words = [word for word in cleaned_comment.split() if len(word) > 3][:5]
        if symbol_name:
            words.insert(0, symbol_name)
        return " ".join(words[:6])

    def _summarize_file_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not payload.get("ok"):
            return {
                "ok": False,
                "error": payload.get("error"),
                "path": payload.get("path"),
                "ref": payload.get("ref"),
            }
        full_content = payload.get("full_content") or ""
        return {
            "ok": True,
            "path": payload.get("path"),
            "sha": payload.get("sha"),
            "download_url": payload.get("download_url"),
            "content_excerpt": payload.get("content_excerpt"),
            "total_lines": len(full_content.splitlines()),
            "ref": payload.get("ref"),
        }

    def _neighbor_files(
        self,
        owner: str,
        repo: str,
        repo_tree: dict[str, Any],
        current_path: str,
        symbol_name: str,
        ref: str | None,
    ) -> dict[str, Any]:
        entries = repo_tree.get("entries", []) if isinstance(repo_tree, dict) else []
        neighbors = []
        current = (current_path or "").replace("\\", "/")
        anchor_text = symbol_name or os.path.basename(current).replace(".py", "")
        for item in entries:
            path = (item.get("path") or "").replace("\\", "/")
            if not path or path == current or item.get("type") != "file":
                continue
            snippet = self.toolbox.fetch_code_snippet(
                owner,
                repo,
                path,
                ref=ref,
                anchor_text=anchor_text,
                symbol_name=symbol_name,
            )
            neighbors.append(
                {
                    "name": item.get("name"),
                    "path": path,
                    "match_reason": snippet.get("match_reason"),
                    "start_line": snippet.get("start_line"),
                    "end_line": snippet.get("end_line"),
                    "excerpt": snippet.get("excerpt"),
                }
            )
            if len(neighbors) >= 5:
                break
        return {"count": len(neighbors), "items": neighbors}

    def _history_snippets(
        self,
        owner: str,
        repo: str,
        current_path: str,
        symbol_name: str,
        related_pr_files: list[dict[str, Any]],
        ref: str | None,
    ) -> dict[str, Any]:
        snippets = []
        seen_paths = set()
        current_basename = os.path.basename(current_path or "")
        current_prefix = self._module_prefix(current_path)
        for pr in related_pr_files:
            for file_item in pr.get("files", []) if isinstance(pr, dict) else []:
                filename = (file_item.get("filename") or "").replace("\\", "/")
                if not filename or filename in seen_paths:
                    continue
                if filename != current_path and os.path.basename(filename) != current_basename and self._module_prefix(filename) != current_prefix:
                    continue
                seen_paths.add(filename)
                snippet = self.toolbox.fetch_code_snippet(
                    owner,
                    repo,
                    filename,
                    ref=ref,
                    anchor_text=symbol_name or current_basename,
                    symbol_name=symbol_name,
                )
                snippets.append(
                    {
                        "path": filename,
                        "match_reason": snippet.get("match_reason"),
                        "start_line": snippet.get("start_line"),
                        "end_line": snippet.get("end_line"),
                        "excerpt": snippet.get("excerpt"),
                        "patch_excerpt": file_item.get("patch_excerpt"),
                    }
                )
                if len(snippets) >= 3:
                    return {"count": len(snippets), "items": snippets}
        return {"count": len(snippets), "items": snippets}

    def _extract_original_signature(self, code: str) -> str:
        for line in (code or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("async def ") or stripped.startswith("class "):
                return stripped
        return ""

    def _comment_keywords(self, satd_comment: str) -> set[str]:
        stopwords = {
            "todo",
            "fixme",
            "xxx",
            "temporary",
            "temporarily",
            "this",
            "that",
            "these",
            "those",
            "when",
            "with",
            "from",
            "into",
            "only",
            "then",
            "than",
            "they",
            "them",
            "their",
            "there",
            "here",
            "should",
            "would",
            "could",
            "must",
            "need",
            "needs",
            "line",
            "code",
            "function",
            "test",
            "tests",
            "comment",
            "below",
            "above",
            "entire",
            "environment",
            "remove",
            "support",
            "default",
        }
        raw_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", satd_comment or "")
        keywords: set[str] = set()
        for token in raw_tokens:
            lowered = token.lower()
            if lowered in stopwords or len(lowered) < 3:
                continue
            keywords.add(lowered)
            if lowered.endswith("s") and len(lowered) > 4:
                keywords.add(lowered[:-1])
        return keywords

    def _name_tokens(self, name: str) -> set[str]:
        if not name:
            return set()
        pieces = re.split(r"[_\W]+", name)
        exploded: list[str] = []
        for piece in pieces:
            if not piece:
                continue
            exploded.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|$)|\d+", piece))
        return {part.lower() for part in exploded if len(part) >= 2}

    def _keyword_overlap(self, text: str, keywords: set[str]) -> int:
        lowered = (text or "").lower()
        return sum(1 for keyword in keywords if keyword in lowered)

    def _distinct_match_tokens(self, text: str) -> set[str]:
        generic = {"test", "tests", "case", "cases", "bug", "func", "function", "method", "class", "file", "py"}
        return {token for token in self._name_tokens(text) if token not in generic}

    def _clip_source(self, text: str, limit: int = 420) -> str:
        clipped = (text or "").strip()
        if len(clipped) <= limit:
            return clipped
        return clipped[: limit - 3].rstrip() + "..."

    def _format_relevance_reason(self, reasons: list[str]) -> str:
        if not reasons:
            return "directly referenced by the target function"
        reason = reasons[0].replace("_", " ")
        reason = re.sub(r"=\d+", "", reason)
        return reason

    def _build_repair_evidence_cards(self, repair_context: dict[str, Any]) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []

        for item in (repair_context.get("same_file_helpers") or {}).get("items") or []:
            cards.append(
                {
                    "kind": "same_file_helper",
                    "symbol_name": item.get("symbol_name"),
                    "location": f"{item.get('start_line')}:{item.get('end_line')}",
                    "why": self._format_relevance_reason(item.get("relevance") or []),
                    "snippet": self._clip_source(item.get("source") or ""),
                }
            )

        same_class = repair_context.get("same_class_evidence") or {}
        for item in same_class.get("related_methods") or []:
            cards.append(
                {
                    "kind": "same_class_method",
                    "symbol_name": item.get("symbol_name"),
                    "location": f"{item.get('start_line')}:{item.get('end_line')}",
                    "why": self._format_relevance_reason(item.get("relevance") or []),
                    "snippet": self._clip_source(item.get("source") or ""),
                }
            )
        for item in same_class.get("related_attributes") or []:
            cards.append(
                {
                    "kind": "same_class_attribute",
                    "symbol_name": item.get("attribute"),
                    "location": f"{item.get('start_line')}:{item.get('end_line')}",
                    "why": self._format_relevance_reason(item.get("relevance") or []),
                    "snippet": self._clip_source(item.get("snippet") or ""),
                }
            )

        for item in (repair_context.get("module_symbols") or {}).get("items") or []:
            cards.append(
                {
                    "kind": f"module_{item.get('kind')}",
                    "symbol_name": item.get("symbol_name"),
                    "location": f"{item.get('start_line')}:{item.get('end_line')}",
                    "why": self._format_relevance_reason(item.get("relevance") or []),
                    "snippet": self._clip_source(item.get("source") or ""),
                }
            )

        test_item = (repair_context.get("targeted_test_snippet") or {}).get("item")
        if test_item:
            cards.append(
                {
                    "kind": "targeted_test",
                    "symbol_name": test_item.get("path"),
                    "location": f"{test_item.get('start_line')}:{test_item.get('end_line')}",
                    "why": "targeted test snippet selected by rule",
                    "snippet": self._clip_source(test_item.get("excerpt") or ""),
                }
            )

        callsite_item = (repair_context.get("targeted_callsite_snippet") or {}).get("item")
        if callsite_item:
            cards.append(
                {
                    "kind": "targeted_callsite",
                    "symbol_name": callsite_item.get("path"),
                    "location": f"{callsite_item.get('start_line')}:{callsite_item.get('end_line')}",
                    "why": "targeted callsite snippet selected by rule",
                    "snippet": self._clip_source(callsite_item.get("excerpt") or ""),
                }
            )

        return cards[:6]

    def _score_helper_candidate(
        self,
        item: dict[str, Any],
        current_name: str,
        direct_calls: set[str],
        self_calls: set[str],
        comment_keywords: set[str],
        class_name: str | None,
    ) -> tuple[int, list[str]]:
        score = 0
        reasons: list[str] = []
        symbol_name = item["symbol_name"]
        name_tokens = self._name_tokens(symbol_name)
        overlap = len(name_tokens.intersection(comment_keywords))
        if symbol_name in direct_calls or symbol_name in self_calls:
            score += 10
            reasons.append("direct_call")
        if overlap:
            score += 4 * overlap
            reasons.append(f"keyword_name_overlap={overlap}")
        source_hits = self._keyword_overlap(item["source"], comment_keywords)
        if source_hits:
            score += source_hits
            reasons.append(f"keyword_source_overlap={source_hits}")
        if class_name and item.get("class_name") == class_name:
            score += 1
            reasons.append("same_class")
        if symbol_name == current_name:
            score = -1
        return score, reasons

    def _strong_targeted_item(
        self,
        item: dict[str, Any] | None,
        symbol_name: str,
        satd_comment: str,
        current_path: str,
        expected_kind: str,
    ) -> bool:
        if not item or not item.get("snippet_ok"):
            return False
        match_reason = str(item.get("match_reason") or "")
        if match_reason in {"file_start", "assert_window"} and expected_kind != "test":
            return False

        keywords = self._comment_keywords(satd_comment)
        symbol_tokens = self._distinct_match_tokens(symbol_name)
        path_tokens = self._distinct_match_tokens(os.path.basename(item.get("path") or ""))
        current_tokens = self._distinct_match_tokens(os.path.basename(current_path or ""))
        excerpt = str(item.get("excerpt") or "")
        excerpt_lower = excerpt.lower()

        score = 0
        if match_reason in {"symbol_match", "anchor_text"}:
            score += 3
        if symbol_name and symbol_name in excerpt:
            score += 4
        if symbol_tokens.intersection(path_tokens):
            score += 3
        if current_tokens.intersection(path_tokens):
            score += 2
        keyword_hits = sum(1 for keyword in keywords if keyword in excerpt_lower)
        score += min(3, keyword_hits)
        if expected_kind == "test" and self.toolbox._is_test_path(item.get("path", "")):
            score += 1
            if match_reason == "assert_window" and not (
                symbol_tokens.intersection(path_tokens) or current_tokens.intersection(path_tokens)
            ):
                return False
        return score >= 4

    def _first_signature_line(self, code: str) -> str:
        for line in (code or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("async def ") or stripped.startswith("class "):
                return stripped
        return ""

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _augment_bundle_metadata(self, bundle: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(bundle.get("metadata", {}))
        base_context = bundle.get("base_context", {}) or {}
        repair_context = bundle.get("repair_context", {}) or {}

        target_file = base_context.get("target_file", {}) if isinstance(base_context, dict) else {}
        target_function = base_context.get("target_function", {}) if isinstance(base_context, dict) else {}
        same_file_helpers = repair_context.get("same_file_helpers", {}) if isinstance(repair_context, dict) else {}
        same_class_evidence = repair_context.get("same_class_evidence", {}) if isinstance(repair_context, dict) else {}
        module_symbols = repair_context.get("module_symbols", {}) if isinstance(repair_context, dict) else {}
        same_file_pattern = repair_context.get("same_file_pattern", {}) if isinstance(repair_context, dict) else {}
        targeted_test_snippet = repair_context.get("targeted_test_snippet", {}) if isinstance(repair_context, dict) else {}
        targeted_callsite_snippet = repair_context.get("targeted_callsite_snippet", {}) if isinstance(repair_context, dict) else {}
        context_commit = str(metadata.get("context_commit") or bundle.get("commit") or "")

        target_file_ok = bool(target_file.get("ok"))
        target_function_found = bool(target_function.get("found"))
        same_file_helpers_count = int(same_file_helpers.get("count") or 0)
        same_class_methods_count = int(same_class_evidence.get("related_methods_count") or 0)
        same_class_attributes_count = int(same_class_evidence.get("related_attributes_count") or 0)
        module_symbols_count = int(module_symbols.get("count") or 0)
        same_file_pattern_count = int(same_file_pattern.get("count") or 0)
        targeted_test_snippet_count = 1 if targeted_test_snippet.get("used") and targeted_test_snippet.get("item") else 0
        targeted_callsite_snippet_count = 1 if targeted_callsite_snippet.get("used") and targeted_callsite_snippet.get("item") else 0
        retrieved_test_snippets_count = targeted_test_snippet_count
        retrieved_callsite_snippets_count = targeted_callsite_snippet_count
        retrieved_history_snippets_count = 0

        target_file_error = str(target_file.get("error") or "")
        if target_file_ok and target_function_found:
            snapshot_alignment_status = "aligned"
        elif context_commit and self._is_missing_ref_error(target_file_error):
            snapshot_alignment_status = "historical_ref_missing"
        elif context_commit and not target_file_ok:
            snapshot_alignment_status = "historical_file_missing"
        else:
            snapshot_alignment_status = "true_mismatch"

        historical_snapshot_mismatch = snapshot_alignment_status != "aligned"
        external_evidence_count = (
            same_file_helpers_count
            + same_class_methods_count
            + same_class_attributes_count
            + module_symbols_count
            + same_file_pattern_count
            + targeted_test_snippet_count
            + targeted_callsite_snippet_count
        )
        decisive_evidence_count = (
            same_file_helpers_count
            + same_class_methods_count
            + targeted_test_snippet_count
            + targeted_callsite_snippet_count
        )
        repair_evidence_mode = "strong" if (
            target_function_found and decisive_evidence_count > 0
        ) else "weak"

        positive_signals = sum(
            1
            for flag in (
                target_file_ok,
                target_function_found,
                same_file_helpers_count > 0,
                same_class_methods_count > 0,
                same_class_attributes_count > 0,
                module_symbols_count > 0 and decisive_evidence_count > 0,
                same_file_pattern_count > 0,
                targeted_test_snippet_count > 0,
                targeted_callsite_snippet_count > 0,
            )
            if flag
        )
        if positive_signals >= 4 and repair_evidence_mode == "strong":
            github_evidence_strength = "high"
        elif positive_signals >= 2 or (target_file_ok and decisive_evidence_count > 0):
            github_evidence_strength = "medium"
        else:
            github_evidence_strength = "low"

        metadata.update(
            {
                "target_file_ok": target_file_ok,
                "context_commit": context_commit,
                "context_strategy": "function_external_evidence",
                "target_function_found": target_function_found,
                "same_file_helpers_count": same_file_helpers_count,
                "same_class_methods_count": same_class_methods_count,
                "same_class_attributes_count": same_class_attributes_count,
                "module_symbols_count": module_symbols_count,
                "same_file_pattern_count": same_file_pattern_count,
                "decisive_external_evidence_count": decisive_evidence_count,
                "targeted_test_snippet_count": targeted_test_snippet_count,
                "targeted_callsite_snippet_count": targeted_callsite_snippet_count,
                "satd_window_found": False,
                "enclosing_symbol_found": target_function_found,
                "related_tests_count": targeted_test_snippet_count,
                "call_sites_count": targeted_callsite_snippet_count,
                "commits_count": 0,
                "similar_history_count": 0,
                "retrieved_test_snippets_count": retrieved_test_snippets_count,
                "retrieved_callsite_snippets_count": retrieved_callsite_snippets_count,
                "retrieved_history_snippets_count": retrieved_history_snippets_count,
                "snapshot_alignment_status": snapshot_alignment_status,
                "repair_evidence_mode": repair_evidence_mode,
                "historical_snapshot_mismatch": historical_snapshot_mismatch,
                "github_evidence_strength": github_evidence_strength,
                "base_cached": bool(bundle.get("base_context")),
                "repair_cached": bool(bundle.get("repair_context")),
                "review_cached": bool(bundle.get("review_context")),
            }
        )
        bundle["metadata"] = metadata
        return bundle

    def _resolve_target_function(self, file_content: str, state: GraphState) -> dict[str, Any]:
        contexts = self._collect_function_contexts(file_content)
        if not contexts:
            return {"found": False, "error": "function_not_found"}

        signature_name = self._fallback_symbol_name(state)
        satd_window = self.toolbox.extract_satd_window(file_content, state["satd_comment"], window=0)
        satd_line = satd_window.get("satd_line") if satd_window.get("found") else None
        original_norm = preprocess_python_code(state["original_code"])
        best = None
        best_score = -1

        for item in contexts:
            score = 0
            if signature_name and item["symbol_name"] == signature_name:
                score += 5
            if satd_line and item["start_line"] <= satd_line <= item["end_line"]:
                score += 7
            candidate_norm = preprocess_python_code(item["source"])
            if original_norm and candidate_norm == original_norm:
                score += 20
            elif original_norm and self._first_signature_line(item["source"]) == self._first_signature_line(state["original_code"]):
                score += 4
            if score > best_score:
                best = item
                best_score = score

        if not best or best_score <= 0:
            return {"found": False, "error": "function_not_found", "satd_line": satd_line}

        matched_by = "normalized_source" if preprocess_python_code(best["source"]) == original_norm else "signature_or_comment"
        return {
            "found": True,
            "symbol_name": best["symbol_name"],
            "class_name": best["class_name"],
            "start_line": best["start_line"],
            "end_line": best["end_line"],
            "satd_line": satd_line,
            "matched_by": matched_by,
            "source": best["source"],
            "calls": sorted(best["calls"]),
            "self_calls": sorted(best["self_calls"]),
            "self_attrs": sorted(best["self_attrs"]),
            "global_uses": sorted(best["global_uses"]),
        }

    def _collect_function_contexts(self, file_content: str) -> list[dict[str, Any]]:
        clean_content = self.toolbox._sanitize_source_text(file_content)
        if not clean_content.strip():
            return []
        try:
            tree = ast.parse(clean_content)
        except (SyntaxError, ValueError):
            return []

        lines = clean_content.splitlines()
        parent_map: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_map[child] = parent

        builtin_names = set(dir(builtins))
        contexts: list[dict[str, Any]] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start is None or end is None:
                continue
            parent = parent_map.get(node)
            class_name = parent.name if isinstance(parent, ast.ClassDef) else None
            usage = self._collect_function_usage(node, builtin_names)
            contexts.append(
                {
                    "node": node,
                    "class_name": class_name,
                    "symbol_name": node.name,
                    "start_line": start,
                    "end_line": end,
                    "source": "\n".join(lines[start - 1 : end]),
                    "calls": usage["calls"],
                    "self_calls": usage["self_calls"],
                    "self_attrs": usage["self_attrs"],
                    "global_uses": usage["global_uses"],
                }
            )
        return contexts

    def _collect_function_usage(self, node: ast.AST, builtin_names: set[str]) -> dict[str, set[str]]:
        arg_names = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs):
                arg_names.add(arg.arg)
            if node.args.vararg:
                arg_names.add(node.args.vararg.arg)
            if node.args.kwarg:
                arg_names.add(node.args.kwarg.arg)

        local_names = set(arg_names)
        calls: set[str] = set()
        self_calls: set[str] = set()
        self_attrs: set[str] = set()
        global_uses: set[str] = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name):
                    calls.add(func.id)
                elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
                    self_calls.add(func.attr)
                    self_attrs.add(func.attr)
            elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name) and child.value.id == "self":
                self_attrs.add(child.attr)
            elif isinstance(child, ast.Name):
                if isinstance(child.ctx, ast.Store):
                    local_names.add(child.id)
                elif isinstance(child.ctx, ast.Load) and child.id not in local_names and child.id not in builtin_names and child.id != "self":
                    global_uses.add(child.id)

        return {
            "calls": calls,
            "self_calls": self_calls,
            "self_attrs": self_attrs,
            "global_uses": global_uses,
        }

    def _extract_same_file_helpers(self, file_content: str, target_function: dict[str, Any], satd_comment: str) -> dict[str, Any]:
        if not target_function.get("found"):
            return {"count": 0, "items": []}
        contexts = self._collect_function_contexts(file_content)
        direct_calls = set(target_function.get("calls") or [])
        self_calls = set(target_function.get("self_calls") or [])
        class_name = target_function.get("class_name")
        decisive = direct_calls | self_calls
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for item in contexts:
            symbol_name = item["symbol_name"]
            if symbol_name not in decisive:
                continue
            score = 12
            reasons = ["direct_call_reference"]
            if class_name and item.get("class_name") == class_name:
                score += 2
                reasons.append("same_class")
            scored.append((score, item["start_line"], {
                "symbol_name": symbol_name,
                "class_name": item["class_name"],
                "start_line": item["start_line"],
                "end_line": item["end_line"],
                "relevance": reasons,
                "source": item["source"],
            }))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        items = [payload for _, _, payload in scored[:3]]
        return {"count": len(items), "items": items}

    def _extract_same_class_evidence(self, file_content: str, target_function: dict[str, Any], satd_comment: str) -> dict[str, Any]:
        if not target_function.get("found") or not target_function.get("class_name"):
            return {"class_name": None, "related_methods_count": 0, "related_attributes_count": 0, "related_methods": [], "related_attributes": []}

        contexts = self._collect_function_contexts(file_content)
        class_name = target_function.get("class_name")
        current_name = target_function.get("symbol_name")
        self_calls = set(target_function.get("self_calls") or [])
        self_attrs = set(target_function.get("self_attrs") or [])

        scored_methods: list[tuple[int, int, dict[str, Any]]] = []
        for item in contexts:
            if item["class_name"] != class_name or item["symbol_name"] == current_name:
                continue
            score = 0
            reasons: list[str] = []
            if item["symbol_name"] in self_calls:
                score += 10
                reasons.append("direct_self_call")
            if score < 10:
                continue
            scored_methods.append((score, item["start_line"], {
                "symbol_name": item["symbol_name"],
                "start_line": item["start_line"],
                "end_line": item["end_line"],
                "relevance": reasons,
                "source": item["source"],
            }))

        related_methods = [payload for _, _, payload in sorted(scored_methods, key=lambda pair: (-pair[0], pair[1]))[:2]]
        related_attributes = self._extract_class_attribute_evidence(file_content, class_name, self_attrs, satd_comment)
        return {
            "class_name": class_name,
            "related_methods_count": len(related_methods),
            "related_attributes_count": len(related_attributes),
            "related_methods": related_methods,
            "related_attributes": related_attributes,
        }

    def _extract_class_attribute_evidence(self, file_content: str, class_name: str, attr_names: set[str], satd_comment: str) -> list[dict[str, Any]]:
        if not attr_names:
            return []
        clean_content = self.toolbox._sanitize_source_text(file_content)
        try:
            tree = ast.parse(clean_content)
        except (SyntaxError, ValueError):
            return []
        lines = clean_content.splitlines()
        results: list[tuple[int, int, dict[str, Any]]] = []
        seen_attrs: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != class_name:
                continue
            for child in ast.walk(node):
                targets = list(child.targets) if isinstance(child, ast.Assign) else [child.target] if isinstance(child, ast.AnnAssign) else []
                for target in targets:
                    if not (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr in attr_names
                        and target.attr not in seen_attrs
                    ):
                        continue
                    start = max(1, getattr(child, "lineno", 1) - 1)
                    end = min(len(lines), getattr(child, "end_lineno", getattr(child, "lineno", 1)) + 1)
                    snippet = "\n".join(lines[start - 1 : end])
                    score = 8
                    reasons = ["direct_attribute_assignment"]
                    results.append((score, start, {
                        "attribute": target.attr,
                        "start_line": start,
                        "end_line": end,
                        "relevance": reasons,
                        "snippet": snippet,
                    }))
                    seen_attrs.add(target.attr)
        return [payload for _, _, payload in sorted(results, key=lambda pair: (-pair[0], pair[1]))[:3]]

    def _extract_module_symbols(self, file_content: str, target_function: dict[str, Any], satd_comment: str) -> dict[str, Any]:
        if not target_function.get("found"):
            return {"count": 0, "items": []}
        clean_content = self.toolbox._sanitize_source_text(file_content)
        try:
            tree = ast.parse(clean_content)
        except (SyntaxError, ValueError):
            return {"count": 0, "items": []}

        lines = clean_content.splitlines()
        used_names = set(target_function.get("global_uses") or [])
        if not used_names:
            return {"count": 0, "items": []}

        items = []
        seen_names: set[str] = set()
        for node in getattr(tree, "body", []):
            symbol_name = None
            symbol_kind = None
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", start)
            source = ""
            if isinstance(node, ast.Import):
                for alias in node.names:
                    candidate = alias.asname or alias.name.split(".")[0]
                    if candidate in used_names and candidate not in seen_names:
                        symbol_name = candidate
                        symbol_kind = "import"
                        break
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    candidate = alias.asname or alias.name
                    if candidate in used_names and candidate not in seen_names:
                        symbol_name = candidate
                        symbol_kind = "import_from"
                        break
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in used_names and target.id not in seen_names:
                        symbol_name = target.id
                        symbol_kind = "assign"
                        break
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in used_names and node.target.id not in seen_names:
                symbol_name = node.target.id
                symbol_kind = "annassign"
            elif isinstance(node, ast.ClassDef) and node.name in used_names and node.name not in seen_names:
                symbol_name = node.name
                symbol_kind = "class"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in used_names and node.name not in seen_names:
                symbol_name = node.name
                symbol_kind = "function"

            if not symbol_name or start is None:
                continue
            source = source or "\n".join(lines[start - 1 : end])
            seen_names.add(symbol_name)
            end = end or start
            items.append(
                {
                    "symbol_name": symbol_name,
                    "kind": symbol_kind,
                    "start_line": start,
                    "end_line": end,
                    "relevance": ["direct_global_use"],
                    "source": source,
                }
            )
        items = items[:3]
        return {"count": len(items), "items": items}

    def _extract_same_file_pattern(self, file_content: str, target_function: dict[str, Any], satd_comment: str) -> dict[str, Any]:
        return {"count": 0, "items": []}

    def _extract_targeted_test_snippet(self, owner: str, repo: str, ref: str | None, state: GraphState, symbol_name: str) -> dict[str, Any]:
        if not self._should_use_test_context(state["satd_comment"]):
            return {"used": False, "reason": "rule_not_triggered", "item": None}
        payload = self.toolbox.find_related_tests(owner, repo, symbol_name or state["file_path"], per_page=1, ref=ref)
        item = ((payload.get("items") or [None])[0] if isinstance(payload, dict) else None)
        if not self._strong_targeted_item(item, symbol_name, state["satd_comment"], state["file_path"], expected_kind="test"):
            return {"used": False, "reason": "weak_match_filtered", "item": None}
        return {"used": True, "reason": "keyword_triggered", "item": item}

    def _extract_targeted_callsite_snippet(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str | None,
        state: GraphState,
        symbol_name: str,
    ) -> dict[str, Any]:
        if not self._should_use_callsite_context(state["satd_comment"]):
            return {"used": False, "reason": "rule_not_triggered", "item": None}
        payload = self.toolbox.find_call_sites(owner, repo, symbol_name, per_page=1, ref=ref, current_path=path)
        item = ((payload.get("items") or [None])[0] if isinstance(payload, dict) else None)
        if not self._strong_targeted_item(item, symbol_name, state["satd_comment"], state["file_path"], expected_kind="callsite"):
            return {"used": False, "reason": "weak_match_filtered", "item": None}
        return {"used": True, "reason": "keyword_triggered", "item": item}

    def _should_use_test_context(self, satd_comment: str) -> bool:
        comment = (satd_comment or "").lower()
        return any(token in comment for token in ("test", "assert", "re-enable", "reenable", "skip", "-o"))

    def _should_use_callsite_context(self, satd_comment: str) -> bool:
        comment = (satd_comment or "").lower()
        return any(token in comment for token in ("replace by", "switch to", "support", "deprecated", "legacy", "full_path", "rename", "use "))

    def _is_missing_ref_error(self, error_text: str) -> bool:
        lowered = (error_text or "").lower()
        markers = (
            "historical_ref_missing",
            "no commit found for sha",
            "invalid object requested",
            "no tree found",
            "reference does not exist",
        )
        return any(marker in lowered for marker in markers)


class OpenAIAnalyzer:
    def __init__(self, client: OpenAICompatClient) -> None:
        self.client = client

    def run(self, state: GraphState) -> AnalysisResult:
        context_summary = self._build_analyzer_context_summary(state.get("github_context"))
        system_prompt = (
            "You are a SATD triage analyzer for generic SATD items only.\n"
            "Your job is not to repair code.\n"
            "Your job is to judge whether this SATD is a good candidate for automatic local repair.\n"
            "Use only the SATD comment and the current code snippet.\n"
            "Judge the item using four questions:\n"
            "1. Does the SATD request a concrete editing operation rather than open-ended implementation, investigation, or design work?\n"
            "2. Is the repair target localizable in the snippet?\n"
            "3. Does the change appear local in scope?\n"
            "4. Is the desired end state clear enough from the comment and snippet to attempt a local repair?\n"
            "Decision policy:\n"
            "- PASS: the SATD describes a clear local repair target that is visible or strongly implied in the snippet.\n"
            "- UNCERTAIN: the SATD may still be repairable, but the target, intent, or scope is not clear enough for confident filtering.\n"
            "- DROP: the SATD is clearly not a good candidate for automatic local repair because it is refactor-like, clearly non-local, or lacks any identifiable local repair target.\n"
            "Do not assume extra repository facts.\n"
            "Do not propose repairs.\n"
            "Do not explain at length.\n"
            "Do not think about whether a human expert could eventually fix it.\n"
            "Decide only whether this is a good candidate for automatic local repair now.\n\n"
            "Important:\n"
            "- Prefer UNCERTAIN over DROP when evidence is mixed.\n"
            "- Use DROP only for strong structural reasons, not just because the task looks difficult.\n"
            "- Question-like wording, future-timing wording, or tentative wording are not automatic DROP signals.\n"
            "- If a local code target is visible, prefer PASS or UNCERTAIN rather than DROP.\n"
            "- Use the levels high / partial / low for each dimension.\n"
            "- Treat the four dimension judgments seriously: they should reflect the actual evidence in the comment and snippet.\n"
            "- Set operation_concrete=high only when the request implies a concrete edit such as remove, replace, annotate, document, adjust a return or raise path, or update a specific value or branch.\n"
            "- Set operation_concrete=partial when the edit direction exists but is still somewhat open-ended, such as adding a check or enabling a path without a fully specified edit. If the comment already points to an existing API, parameter, key, path, branch, exception path, or value replacement target, it should be at least partial.\n"
            "- Set operation_concrete=low only for open-ended implementation, design decisions, debugging, investigation, broad cleanup, or other underspecified work.\n"
            "- If the comment already refers to an existing local object, API, parameter, key, path, branch, block, value, or current code path, operation_concrete should usually be at least partial unless the task is still clearly open-ended.\n"
            "- Set localizable=high only if the snippet shows a concrete edit target such as a symbol, call, parameter, branch, return statement, exception path, variable, or doc block to change.\n"
            "- Set localizable=partial when the rough area is visible but the exact local edit target is still unclear. If the current statement, call, key, parameter, branch, or code block is already visible, it should be at least partial.\n"
            "- Set localizable=low only when the snippet does not expose any credible local target to edit.\n"
            "- Set local_scope=high only if the change appears solvable within one local block or function without broader refactoring.\n"
            "- Set local_scope=partial when the change still looks mostly local but may need a small amount of nearby surrounding context. If the discussion is still centered on the current function or block, it should be at least partial.\n"
            "- Set local_scope=low only when the task looks cross-cutting, architectural, global, or broader than a local edit.\n"
            "- Set end_state_clear=high only if the desired repaired state is reasonably clear from the comment and snippet, even if future timing or external versions are mentioned.\n"
            "- Set end_state_clear=partial when the intended direction is visible but the exact repaired state is not fully pinned down.\n"
            "- Set end_state_clear=low only when the final intended state is genuinely ambiguous, requires product or design choices, or depends on unspecified broader behavior.\n"
            "- If the comment points to an existing local object or current code path, end_state_clear should usually be at least partial unless the request is still fundamentally open-ended or design-level.\n"
            "Return strict JSON only."
        )
        user_prompt = (
            f"SATD comment:\n{state['satd_comment']}\n\n"
            f"Current code snippet:\n```python\n{state['original_code']}\n```\n\n"
            "Return JSON with exactly:\n"
            "{\n"
            '  "decision": "pass" | "uncertain" | "drop",\n'
            '  "confidence": 0.0,\n'
            '  "operation_concrete": "high" | "partial" | "low",\n'
            '  "localizable": "high" | "partial" | "low",\n'
            '  "local_scope": "high" | "partial" | "low",\n'
            '  "end_state_clear": "high" | "partial" | "low",\n'
            '  "comment_evidence": "short quote or phrase from the comment",\n'
            '  "code_evidence": "short phrase describing the visible target or missing target",\n'
            '  "notes": "one short sentence"\n'
            "}\n"
        )
        payload = self.client.generate_json(system_prompt, user_prompt, request_label=f"analyze:task_{state['task_id']}")
        return self._coerce_analysis(payload, state, source="llm_stage1")

    def _build_analyzer_context_summary(self, bundle: dict[str, Any] | None) -> str:
        if not bundle:
            return ""
        base = bundle.get("base_context") or {}
        repair = bundle.get("repair_context") or {}
        metadata = bundle.get("metadata") or {}
        target_function = (base.get("target_function") or {}) if isinstance(base, dict) else {}
        module_symbols = (repair.get("module_symbols") or {}) if isinstance(repair, dict) else {}
        same_file_helpers = (repair.get("same_file_helpers") or {}) if isinstance(repair, dict) else {}
        same_class_evidence = (repair.get("same_class_evidence") or {}) if isinstance(repair, dict) else {}
        same_file_pattern = (repair.get("same_file_pattern") or {}) if isinstance(repair, dict) else {}
        targeted_callsite = (repair.get("targeted_callsite_snippet") or {}) if isinstance(repair, dict) else {}
        callsite_item = targeted_callsite.get("item") or {}

        summary = {
            "definition_context": {
                "target_function_found": target_function.get("found"),
                "symbol_name": target_function.get("symbol_name"),
                "class_name": target_function.get("class_name"),
                "matched_by": target_function.get("matched_by"),
                "module_symbols_count": module_symbols.get("count", 0),
                "module_symbols_preview": [
                    {
                        "symbol_name": item.get("symbol_name"),
                        "kind": item.get("kind"),
                    }
                    for item in (module_symbols.get("items") or [])[:5]
                ],
            },
            "usage_context": {
                "call_sites_count": metadata.get("call_sites_count", 0),
                "targeted_callsite_found": bool(targeted_callsite.get("used") and callsite_item),
                "targeted_callsite_path": callsite_item.get("path"),
                "targeted_callsite_excerpt": (callsite_item.get("excerpt") or "")[:400],
            },
            "replacement_context": {
                "same_file_helpers_count": same_file_helpers.get("count", 0),
                "same_class_methods_count": same_class_evidence.get("related_methods_count", 0),
                "same_class_attributes_count": same_class_evidence.get("related_attributes_count", 0),
                "same_file_pattern_count": same_file_pattern.get("count", 0),
            },
        }
        return json.dumps(summary, ensure_ascii=False)

    def build_easy_route_analysis(self, route_type: str) -> AnalysisResult:
        note = f"Easy SATD route '{route_type}' bypassed the generic analyzer and passed directly."
        return self._build_result(
            decision="pass",
            confidence=1.0,
            operation_concrete="high",
            localizable="high",
            local_scope="high",
            end_state_clear="high",
            notes=note,
            comment_evidence=route_type,
            code_evidence="easy route bypass",
            satd_type=route_type,
            source="rule_easy_route",
            scope_radius="function",
        )

    def build_rule_drop_analysis(self, notes: str) -> AnalysisResult:
        return self._build_result(
            decision="drop",
            confidence=1.0,
            operation_concrete="low",
            localizable="low",
            local_scope="low",
            end_state_clear="low",
            notes=notes,
            comment_evidence="",
            code_evidence="rule-based missing input",
            satd_type="generic",
            source="rule_drop",
            scope_radius="function",
        )

    def _coerce_analysis(self, payload: dict[str, Any], state: GraphState, source: str) -> AnalysisResult:
        requested_decision = self._normalize_decision(payload.get("decision"))
        confidence = self._clamp_float(payload.get("confidence"), 0.0)
        operation_concrete = self._normalize_level(payload.get("operation_concrete"))
        localizable = self._normalize_level(payload.get("localizable"))
        local_scope = self._normalize_level(payload.get("local_scope"))
        end_state_clear = self._normalize_level(payload.get("end_state_clear"))
        operation_concrete, localizable, local_scope, end_state_clear = self._apply_existing_target_floors(
            state=state,
            operation_concrete=operation_concrete,
            localizable=localizable,
            local_scope=local_scope,
            end_state_clear=end_state_clear,
        )
        notes = self._one_line(payload.get("notes") or payload.get("reason") or payload.get("evidence_summary") or "")
        comment_evidence = self._one_line(payload.get("comment_evidence"))
        code_evidence = self._one_line(payload.get("code_evidence"))
        satd_type = str(state.get("satd_route_type") or "generic")
        decision = self._decision_from_checks(
            requested_decision=requested_decision,
            operation_concrete=operation_concrete,
            localizable=localizable,
            local_scope=local_scope,
            end_state_clear=end_state_clear,
        )
        return self._build_result(
            decision=decision,
            confidence=confidence,
            operation_concrete=operation_concrete,
            localizable=localizable,
            local_scope=local_scope,
            end_state_clear=end_state_clear,
            notes=notes,
            comment_evidence=comment_evidence,
            code_evidence=code_evidence,
            satd_type=satd_type,
            source=source,
            scope_radius=self._infer_scope_radius_from_code(state.get("original_code") or ""),
        )

    def _build_result(
        self,
        *,
        decision: str,
        confidence: float,
        operation_concrete: str | None,
        localizable: str | None,
        local_scope: str | None,
        end_state_clear: str | None,
        notes: str,
        comment_evidence: str,
        code_evidence: str,
        satd_type: str,
        source: str,
        scope_radius: str,
    ) -> AnalysisResult:
        repairable = decision != "drop"
        confidence = self._clamp_float(confidence, 0.0)
        operation_concrete = operation_concrete if operation_concrete is not None else ("high" if decision == "pass" else "low" if decision == "drop" else "partial")
        localizable = localizable if localizable is not None else ("high" if decision == "pass" else "low" if decision == "drop" else "partial")
        local_scope = local_scope if local_scope is not None else ("high" if decision == "pass" else "low" if decision == "drop" else "partial")
        end_state_clear = end_state_clear if end_state_clear is not None else ("high" if decision == "pass" else "low" if decision == "drop" else "partial")

        operation_score = self._level_score(operation_concrete)
        localizable_score = self._level_score(localizable)
        local_scope_score = self._level_score(local_scope)
        end_state_score = self._level_score(end_state_clear)
        if decision == "pass":
            repairability_score = max(0.70, confidence or 0.80)
            intent_clarity = 0.25 + (0.65 * operation_score)
            change_locality = 0.25 + (0.65 * local_scope_score)
            semantic_risk = 0.28
            context_sufficiency = 0.20 + (0.35 * localizable_score) + (0.25 * end_state_score)
            verifiability = 0.68
            analyze_score = max(0.72, repairability_score)
            risk_level = "medium" if satd_type == "generic" else "low"
            context_score = context_sufficiency
            clarity_score = intent_clarity
            context_gaps: list[str] = []
            repair_strategy = "Pass this item directly to the repair stage."
        elif decision == "uncertain":
            repairability_score = max(0.45, confidence or 0.55)
            intent_clarity = 0.20 + (0.45 * operation_score)
            change_locality = 0.20 + (0.45 * local_scope_score)
            semantic_risk = 0.50
            context_sufficiency = 0.12 + (0.28 * localizable_score) + (0.18 * end_state_score)
            verifiability = 0.46
            analyze_score = max(0.46, repairability_score)
            risk_level = "medium"
            context_score = context_sufficiency
            clarity_score = intent_clarity
            context_gaps = self._derive_context_gaps(operation_concrete, localizable, local_scope, end_state_clear)
            repair_strategy = "Allow repair to try, but treat this item as ambiguous."
        else:
            repairability_score = 0.0
            intent_clarity = 0.08 + (0.22 * operation_score)
            change_locality = 0.06 + (0.18 * local_scope_score)
            semantic_risk = 0.82
            context_sufficiency = 0.04 + (0.16 * localizable_score) + (0.10 * end_state_score)
            verifiability = 0.18
            analyze_score = min(0.30, confidence)
            risk_level = "high"
            context_score = context_sufficiency
            clarity_score = intent_clarity
            context_gaps = self._derive_context_gaps(operation_concrete, localizable, local_scope, end_state_clear)
            repair_strategy = "Drop this item before repair."

        return AnalysisResult(
            decision=decision,
            repairable=repairable,
            repairability_score=repairability_score,
            intent_clarity=intent_clarity,
            change_locality=change_locality,
            semantic_risk=semantic_risk,
            context_sufficiency=context_sufficiency,
            verifiability=verifiability,
            analyze_score=analyze_score,
            confidence=confidence,
            satd_type=satd_type,
            reason=notes or decision,
            evidence_summary=notes or decision,
            risk_level=risk_level,
            context_score=context_score,
            clarity_score=clarity_score,
            scope_radius=scope_radius,
            operation_concrete=operation_concrete,
            localizable=localizable,
            local_scope=local_scope,
            end_state_clear=end_state_clear,
            comment_evidence=comment_evidence,
            code_evidence=code_evidence,
            validation_signals=[
                f"simple_analyzer:{source}",
                f"decision:{decision}",
                f"operation_concrete:{operation_concrete}",
                f"localizable:{localizable}",
                f"local_scope:{local_scope}",
                f"end_state_clear:{end_state_clear}",
            ],
            context_gaps=context_gaps,
            followup_context_requests=[],
            repair_strategy=repair_strategy,
            historical_snapshot_mismatch=False,
            github_evidence_strength="low",
        )

    def _infer_scope_radius_from_code(self, code: str) -> str:
        lowered = (code or "").lower()
        if "class " in lowered:
            return "class"
        if "def " in lowered or "async def " in lowered:
            return "function"
        if len([line for line in (code or "").splitlines() if line.strip()]) <= 2:
            return "line"
        return "function"

    def _decision_from_checks(
        self,
        *,
        requested_decision: str,
        operation_concrete: str | None,
        localizable: str | None,
        local_scope: str | None,
        end_state_clear: str | None,
    ) -> str:
        checks = [operation_concrete, localizable, local_scope, end_state_clear]
        known = [item for item in checks if item is not None]
        low_count = sum(1 for item in known if item == "low")
        high_count = sum(1 for item in known if item == "high")

        if localizable == "high" and local_scope == "high" and (operation_concrete == "high" or end_state_clear == "high"):
            return "pass"

        if operation_concrete == "low" and localizable == "low" and end_state_clear == "low":
            return "drop"
        if localizable == "low" and local_scope == "low" and end_state_clear == "low":
            return "drop"
        if low_count >= 3 and high_count == 0:
            return "drop"

        return "uncertain"

    def _apply_existing_target_floors(
        self,
        *,
        state: GraphState,
        operation_concrete: str | None,
        localizable: str | None,
        local_scope: str | None,
        end_state_clear: str | None,
    ) -> tuple[str | None, str | None, str | None, str | None]:
        comment = str(state.get("satd_comment") or "")
        code = str(state.get("original_code") or "")
        if not self._has_existing_target_hint(comment, code):
            return operation_concrete, localizable, local_scope, end_state_clear
        if self._looks_open_ended_task(comment):
            return operation_concrete, localizable, local_scope, end_state_clear

        if localizable == "low":
            localizable = "partial"
        if operation_concrete == "low":
            operation_concrete = "partial"
        if end_state_clear == "low":
            end_state_clear = "partial"
        return operation_concrete, localizable, local_scope, end_state_clear

    def _has_existing_target_hint(self, comment: str, code: str) -> bool:
        lowered = (comment or "").lower()
        if not lowered.strip() or not (code or "").strip():
            return False
        anchor_patterns = [
            r"\bthis\b",
            r"\bthese\b",
            r"\bthat\b",
            r"\bcurrent\b",
            r"\bexisting\b",
            r"\balready\b",
            r"\bhere\b",
            r"\bapi\b",
            r"\bparameter\b",
            r"\bparam\b",
            r"\bargument\b",
            r"\bkey\b",
            r"\bkeys\b",
            r"\bpath\b",
            r"\bbranch\b",
            r"\bblock\b",
            r"\bvalue\b",
            r"\bfield\b",
            r"\bindex name\b",
            r"\bline\b",
            r"\bobject\b",
        ]
        return any(re.search(pattern, lowered) for pattern in anchor_patterns)

    def _looks_open_ended_task(self, comment: str) -> bool:
        lowered = (comment or "").lower()
        open_ended_patterns = [
            r"\bimplement\b",
            r"\bdecide\b",
            r"\bconsider\b",
            r"\binvestigat",
            r"\blook into\b",
            r"\bdebug\b",
            r"\brefactor\b",
            r"\brewrite\b",
            r"\bredesign\b",
            r"\bclean up\b",
            r"\bglobal\b",
            r"\barchitecture\b",
            r"\bintegration\b",
        ]
        return any(re.search(pattern, lowered) for pattern in open_ended_patterns)

    def _maybe_compress_uncertain(self, state: GraphState, analysis: AnalysisResult) -> AnalysisResult:
        if analysis.decision != "uncertain":
            return analysis
        if not self._is_second_stage_uncertain_candidate(state, analysis):
            return analysis

        context_summary = self._build_analyzer_context_summary(state.get("github_context"))
        system_prompt = (
            "You are a second-stage SATD triage analyzer for generic uncertain SATD items.\n"
            "Your job is to decide whether an uncertain item should remain uncertain or be dropped after examining compact repository context.\n"
            "Use only the SATD comment, current code snippet, first-stage structured summary, and the compact definition/usage context.\n"
            "Prefer keeping the item as UNCERTAIN unless the context strongly shows that this is not a closed local edit task.\n"
            "A DROP should be reserved for tasks that still lack a credible existing target, lack local usage evidence, or remain fundamentally open-ended after context is considered.\n"
            "If the context shows an existing object, local usage, or a plausible replacement path, prefer UNCERTAIN.\n"
            "If the context still indicates broad integration, migration, validation strategy, or open-ended improvement work with weak usage locality and weak replacement evidence, prefer DROP.\n"
            "Return strict JSON only."
        )
        user_prompt = (
            f"SATD comment:\n{state['satd_comment']}\n\n"
            f"Current code snippet:\n```python\n{state['original_code']}\n```\n\n"
            "First-stage summary:\n"
            f"- operation_concrete: {analysis.operation_concrete}\n"
            f"- localizable: {analysis.localizable}\n"
            f"- local_scope: {analysis.local_scope}\n"
            f"- end_state_clear: {analysis.end_state_clear}\n"
            f"- comment_evidence: {analysis.comment_evidence or '(none)'}\n"
            f"- code_evidence: {analysis.code_evidence or '(none)'}\n\n"
            + (f"Compact definition/usage context:\n{context_summary}\n\n" if context_summary else "")
            +
            "Return JSON with exactly:\n"
            "{\n"
            '  "decision": "drop" | "uncertain",\n'
            '  "confidence": 0.0,\n'
            '  "target_existence": "high" | "partial" | "low",\n'
            '  "usage_locality": "high" | "partial" | "low",\n'
            '  "replacement_evidence": "high" | "partial" | "low",\n'
            '  "task_closedness": "high" | "partial" | "low",\n'
            '  "notes": "one short sentence"\n'
            "}\n"
        )
        payload = self.client.generate_json(
            system_prompt,
            user_prompt,
            request_label=f"analyze_stage2:task_{state['task_id']}",
        )
        return self._coerce_uncertain_compression(payload, analysis, state)

    def _is_second_stage_uncertain_candidate(self, state: GraphState, analysis: AnalysisResult) -> bool:
        if str(state.get("satd_route_type") or "generic") != "generic":
            return False
        if analysis.decision != "uncertain":
            return False
        bundle = state.get("github_context") or {}
        base = bundle.get("base_context") or {}
        repair = bundle.get("repair_context") or {}
        has_definition = bool((base.get("target_function") or {}).get("found")) or bool((repair.get("module_symbols") or {}).get("count"))
        has_usage = bool((repair.get("targeted_callsite_snippet") or {}).get("used")) or bool((bundle.get("metadata") or {}).get("call_sites_count"))
        has_replacement = bool((repair.get("same_file_helpers") or {}).get("count")) or bool((repair.get("same_file_pattern") or {}).get("count"))
        if has_definition or has_usage or has_replacement:
            return True
        weak_local_shape = (
            analysis.localizable == "high"
            and analysis.local_scope == "high"
            and analysis.operation_concrete in {"low", "partial"}
            and analysis.end_state_clear in {"low", "partial"}
        )
        return weak_local_shape

    def _coerce_uncertain_compression(
        self,
        payload: dict[str, Any],
        base: AnalysisResult,
        state: GraphState,
    ) -> AnalysisResult:
        requested = self._normalize_decision(payload.get("decision"))
        confidence = self._clamp_float(payload.get("confidence"), 0.0)
        target_existence = self._normalize_level(payload.get("target_existence"))
        usage_locality = self._normalize_level(payload.get("usage_locality"))
        replacement_evidence = self._normalize_level(payload.get("replacement_evidence"))
        task_closedness = self._normalize_level(payload.get("task_closedness"))
        notes = self._one_line(payload.get("notes"))
        open_ended_comment = self._looks_open_ended_task(str(state.get("satd_comment") or ""))

        low_count = sum(
            1 for item in (target_existence, usage_locality, replacement_evidence, task_closedness) if item == "low"
        )
        should_drop_by_open_task = (
            requested == "drop"
            and task_closedness == "low"
            and usage_locality == "low"
            and replacement_evidence != "high"
            and confidence >= 0.55
        )
        should_drop_by_missing_target = (
            requested == "drop"
            and target_existence == "low"
            and usage_locality == "low"
            and confidence >= 0.55
        )
        should_drop_by_broad_context = (
            requested == "drop"
            and task_closedness == "low"
            and usage_locality != "high"
            and replacement_evidence == "low"
            and confidence >= 0.58
        )
        should_drop_by_partial_target_but_nonlocal = (
            requested == "drop"
            and target_existence == "partial"
            and usage_locality == "low"
            and task_closedness == "low"
            and replacement_evidence != "high"
            and confidence >= 0.60
        )
        should_drop_by_open_ended_weak_spec = (
            open_ended_comment
            and base.operation_concrete == "low"
            and base.end_state_clear == "low"
            and base.localizable == "high"
            and base.local_scope == "high"
            and replacement_evidence != "high"
        )
        should_drop = (
            should_drop_by_open_task
            or should_drop_by_missing_target
            or should_drop_by_broad_context
            or should_drop_by_partial_target_but_nonlocal
            or should_drop_by_open_ended_weak_spec
            or (
                requested == "drop"
                and target_existence == "low"
                and task_closedness == "low"
                and low_count >= 3
                and confidence >= 0.60
            )
        )
        if not should_drop:
            if notes:
                base.reason = notes
                base.evidence_summary = notes
            base.validation_signals.append(
                "simple_analyzer:stage2_keep:"
                f"{target_existence or 'unknown'}:{usage_locality or 'unknown'}:"
                f"{replacement_evidence or 'unknown'}:{task_closedness or 'unknown'}"
            )
            return base

        updated = copy.deepcopy(base)
        updated.decision = "drop"
        updated.repairable = False
        updated.confidence = max(base.confidence, confidence)
        updated.repairability_score = 0.0
        updated.semantic_risk = max(updated.semantic_risk, 0.82)
        updated.verifiability = min(updated.verifiability, 0.18)
        updated.analyze_score = min(updated.analyze_score, 0.30)
        updated.risk_level = "high"
        updated.reason = notes or "Second-stage analyzer judged this uncertain item to be fundamentally open-ended."
        updated.evidence_summary = updated.reason
        updated.repair_strategy = "Drop this item after second-stage uncertain compression."
        updated.validation_signals.append(
            f"simple_analyzer:stage2_drop:{target_existence}:{usage_locality}:{replacement_evidence}:{task_closedness}"
        )
        return updated

    def _normalize_decision(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"pass", "uncertain", "drop"}:
            return text
        if "drop" in text:
            return "drop"
        if "uncertain" in text or "need" in text:
            return "uncertain"
        return "pass"

    def _normalize_level(self, value: Any) -> str | None:
        if isinstance(value, bool):
            return "high" if value else "low"
        text = str(value or "").strip().lower()
        if text in {"high", "strong", "clear"}:
            return "high"
        if text in {"partial", "medium", "mixed", "somewhat"}:
            return "partial"
        if text in {"low", "weak", "unclear"}:
            return "low"
        if text in {"true", "yes", "1"}:
            return "high"
        if text in {"false", "no", "0"}:
            return "low"
        return None

    def _derive_context_gaps(
        self,
        operation_concrete: str | None,
        localizable: str | None,
        local_scope: str | None,
        end_state_clear: str | None,
    ) -> list[str]:
        gaps: list[str] = []
        if operation_concrete == "low":
            gaps.append("not_operation_concrete")
        elif operation_concrete == "partial":
            gaps.append("partially_operation_concrete")
        if localizable == "low":
            gaps.append("not_localizable")
        elif localizable == "partial":
            gaps.append("partially_localizable")
        if local_scope == "low":
            gaps.append("not_local_scope")
        elif local_scope == "partial":
            gaps.append("partially_local_scope")
        if end_state_clear == "low":
            gaps.append("not_end_state_clear")
        elif end_state_clear == "partial":
            gaps.append("partially_end_state_clear")
        return gaps

    def _level_score(self, value: str | None) -> float:
        if value == "high":
            return 1.0
        if value == "partial":
            return 0.55
        if value == "low":
            return 0.0
        return 0.45

    def _one_line(self, value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        return text[:240]

    def _clamp_float(self, value: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default


class OpenAIFixer:
    def __init__(
        self,
        client: OpenAICompatClient,
        repair_context_mode: str = "clone_treesitter",
        max_method_contexts: int = 2,
        logger: Any | None = None,
        checkpoint_callback: Any | None = None,
    ) -> None:
        self.client = client
        self.repair_context_mode = repair_context_mode
        self.max_method_contexts = max(1, int(max_method_contexts))
        self.logger = logger
        self.checkpoint_callback = checkpoint_callback

    def run(
        self,
        state: GraphState,
        candidate_mode: str = "baseline_context",
    ) -> tuple[RepairAttempt, MethodInquiryResult, list[RetrievedMethodContext], list[str], list[UncertaintyItem], list[EditConstraint]]:
        round_id = state["round_id"] + 1
        use_method_context = self._candidate_uses_method_context(candidate_mode)
        self._checkpoint(
            state,
            stage="question_start",
            payload={"round_id": round_id, "candidate_mode": candidate_mode},
        )
        if use_method_context:
            method_inquiry = self.identify_required_methods(state)
            self._log(
                state,
                f"context questions={self._format_method_list(method_inquiry.required_methods)} "
                f"reason={method_inquiry.reason or 'unspecified'}"
            )
        else:
            method_inquiry = MethodInquiryResult(reason="candidate_mode_without_method_context")
            self._log(state, f"context questions=skipped mode={candidate_mode}")
        effective_use_method_context = use_method_context and bool(method_inquiry.required_methods)
        uncertainty_items = list(method_inquiry.uncertainty_items or [])
        self._checkpoint(
            state,
            stage="question_done",
            payload={
                "round_id": round_id,
                "candidate_mode": candidate_mode,
                "required_methods": list(method_inquiry.required_methods),
                "reason": method_inquiry.reason,
                "method_notes": self._serialize_method_notes(method_inquiry.method_notes),
                "uncertainty_items": self._serialize_uncertainty_items(uncertainty_items),
            },
        )
        self._checkpoint(
            state,
            stage="retrieval_start",
            payload={
                "round_id": round_id,
                "candidate_mode": candidate_mode,
                "required_methods": list(method_inquiry.required_methods),
            },
        )
        if effective_use_method_context:
            method_location_hints = self.resolve_method_locations(state, method_inquiry.required_methods)
            self._checkpoint(
                state,
                stage="retrieval_route_done",
                payload={
                    "round_id": round_id,
                    "candidate_mode": candidate_mode,
                    "required_methods": list(method_inquiry.required_methods),
                    "method_location_hints": method_location_hints,
                },
            )
            retrieved_method_contexts = self._retrieve_method_contexts(
                state,
                method_inquiry.required_methods,
                method_location_hints=method_location_hints,
            )
            retrieved_method_contexts = self._enrich_retrieved_method_contexts(
                state=state,
                method_inquiry=method_inquiry,
                retrieved_method_contexts=retrieved_method_contexts,
            )
            missing_method_names = [item.method_name for item in retrieved_method_contexts if not item.found]
            found_method_contexts = [item for item in retrieved_method_contexts if item.found]
            found_method_contexts = self._apply_method_context_gate(
                state=state,
                method_inquiry=method_inquiry,
                retrieved_method_contexts=found_method_contexts,
            )
            self._log(
                state,
                "context results "
                f"found={len(found_method_contexts)}/{len(method_inquiry.required_methods)} "
                f"missing={self._format_method_list(missing_method_names, empty='none')}"
            )
        else:
            method_location_hints = {}
            retrieved_method_contexts = []
            missing_method_names = []
            found_method_contexts = []
            if use_method_context:
                self._log(state, "context results=skipped reason=no_indispensable_methods")
            else:
                self._log(state, f"context results=skipped mode={candidate_mode}")
            self._checkpoint(
                state,
                stage="retrieval_route_done",
                payload={
                    "round_id": round_id,
                    "candidate_mode": candidate_mode,
                    "required_methods": [],
                    "method_location_hints": {},
                },
            )
        edit_constraints = self._synthesize_edit_constraints(
            state=state,
            method_inquiry=method_inquiry,
            retrieved_method_contexts=found_method_contexts,
        )
        effective_candidate_mode = self._effective_candidate_mode(candidate_mode, found_method_contexts)
        self._checkpoint(
            state,
            stage="retrieval_done",
            payload={
                "round_id": round_id,
                "candidate_mode": candidate_mode,
                "required_methods": list(method_inquiry.required_methods),
                "method_notes": self._serialize_method_notes(method_inquiry.method_notes),
                "uncertainty_items": self._serialize_uncertainty_items(uncertainty_items),
                "method_location_hints": method_location_hints,
                "retrieved_method_contexts": self._serialize_retrieved_method_contexts(retrieved_method_contexts),
                "missing_method_names": list(missing_method_names),
                "edit_constraints": self._serialize_edit_constraints(edit_constraints),
            },
        )
        self._checkpoint(
            state,
            stage="generation_start",
            payload={
                "round_id": round_id,
                "candidate_mode": candidate_mode,
                "required_methods": list(method_inquiry.required_methods),
                "method_notes": self._serialize_method_notes(method_inquiry.method_notes),
                "uncertainty_items": self._serialize_uncertainty_items(uncertainty_items),
                "retrieved_method_contexts": self._serialize_retrieved_method_contexts(retrieved_method_contexts),
                "missing_method_names": list(missing_method_names),
                "edit_constraints": self._serialize_edit_constraints(edit_constraints),
            },
        )
        system_prompt, user_prompt = self._build_repair_prompts(
            state=state,
            round_id=round_id,
            method_inquiry=method_inquiry,
            retrieved_method_contexts=found_method_contexts,
            missing_method_names=missing_method_names,
            edit_constraints=edit_constraints,
        )
        payload = self.client.generate_json(
            system_prompt,
            user_prompt,
            request_label=f"repair:task_{state['task_id']}:round_{round_id}:candidate_{effective_candidate_mode}",
        )
        repaired_code = str(payload.get("repaired_code") or state["original_code"])
        repair_plan = self._default_repair_plan(state)
        changed_scope = self._infer_changed_scope(state["original_code"], repaired_code)
        confidence = self._default_repair_confidence(state, repaired_code, found_method_contexts)
        notes = "repair metadata generated locally"
        if method_inquiry.required_methods and not found_method_contexts:
            notes = f"{notes} | no_method_context_found"
        elif found_method_contexts:
            notes = f"{notes} | method_contexts={len(found_method_contexts)}"
        else:
            notes = f"{notes} | no_required_methods_identified"

        em_risk_notes = self._em_risk_notes(state["original_code"], repaired_code)
        if em_risk_notes:
            confidence = min(confidence, 0.42)
            notes = f"{notes} | em_risk={','.join(em_risk_notes)}"
        self._log(
            state,
            f"repair output scope={changed_scope} confidence={confidence:.2f} mode={effective_candidate_mode}"
        )
        self._checkpoint(
            state,
            stage="generation_done",
            payload={
                "round_id": round_id,
                "candidate_mode": effective_candidate_mode,
                "repair_plan": repair_plan,
                "repaired_code": repaired_code,
                "changed_scope": changed_scope,
                "confidence": confidence,
                "notes": notes,
            },
        )

        return (
            RepairAttempt(
                round_id=round_id,
                repair_plan=repair_plan,
                repaired_code=repaired_code,
                changed_scope=changed_scope,
                confidence=confidence,
                notes=notes,
                candidate_mode=effective_candidate_mode,
            ),
            method_inquiry,
            found_method_contexts,
            missing_method_names,
            uncertainty_items,
            edit_constraints,
        )

    def identify_required_methods(self, state: GraphState) -> MethodInquiryResult:
        if self.repair_context_mode not in {"method_query", "clone_treesitter"}:
            return MethodInquiryResult()

        satd_route_type = str(state.get("satd_route_type") or "generic").strip().lower()
        satd_comment = state.get("satd_comment") or ""
        uncertainty_items = self._extract_uncertainty_candidates_from_satd_code(
            satd_comment=satd_comment,
            original_code=state["original_code"],
        )

        if self._skip_method_context_by_rule(satd_route_type, satd_comment):
            return MethodInquiryResult(
                required_methods=[],
                reason=f"rule_without_method_context:{satd_route_type or 'generic'}",
                uncertainty_items=uncertainty_items,
            )

        retrieval_candidates = [
            item
            for item in uncertainty_items
            if item.retrieval_eligible and item.kind in {"method", "symbol"}
        ]
        candidates = [
            {
                "raw_call": item.raw_name or item.name,
                "normalized_name": item.normalized_name or self._normalize_method_name(item.name),
                "line": item.line,
                "kind": item.kind,
            }
            for item in retrieval_candidates
            if item.normalized_name or self._normalize_method_name(item.name)
        ]
        if not candidates:
            return MethodInquiryResult(
                required_methods=[],
                reason="no_retrieval_eligible_method_symbol",
                uncertainty_items=uncertainty_items,
            )

        method_line = "、".join(
            str(item.get("normalized_name") or "").strip()
            for item in candidates
            if str(item.get("normalized_name") or "").strip()
        ) or "[none]"

        system_prompt = textwrap.dedent(
            """
            You will repair this SATD in the next step.
            Before that, select only the methods that are strongly relevant to exact-match repair.

            Rules:
            - Return only methods that you must understand before repairing.
            - Favor false negatives over false positives.
            - You must only choose from the provided methods.
            - For every selected method, explain why it is strongly related to the repair.

            Return JSON with:
            {
              "required_methods": [
                {"method_name": "...", "selection_reason": "..."}
              ],
              "reason": "short explanation"
            }
            """
        ).strip()
        user_prompt = textwrap.dedent(
            f"""
            You will repair this SATD next.
            Before that, you may ask for method context.

            Methods:
            {method_line}

            Comment:
            {satd_comment}

            Code:
            {state["original_code"]}

            Output the strongly relevant methods and their selection reasons.
            """
        ).strip()
        payload = self.client.generate_json(
            system_prompt,
            user_prompt,
            temperature=0.0,
            request_label=f"method_inquiry:task_{state.get('task_id', '?')}",
        )
        inquiry = self._coerce_method_inquiry(
            payload,
            allowed_methods=[str(item.get("normalized_name") or "") for item in candidates],
        )
        if not inquiry.reason:
            inquiry.reason = "llm_required_method_selection"
        inquiry.uncertainty_items = uncertainty_items
        return inquiry

    def _skip_method_context_by_rule(self, satd_route_type: str, satd_comment: str) -> bool:
        route = str(satd_route_type or "").strip().lower()
        if route in {"remove_temporary", "type_annotation", "replace_symbol", "document"}:
            return True
        return False

    def _coerce_method_inquiry(
        self,
        payload: dict[str, Any],
        allowed_methods: list[str] | None = None,
    ) -> MethodInquiryResult:
        raw_methods = payload.get("required_methods")
        required_methods: list[str] = []
        method_notes: list[dict[str, str]] = []
        seen: set[str] = set()
        allowed = set(allowed_methods or [])
        if isinstance(raw_methods, list):
            for item in raw_methods:
                selection_reason = ""
                raw_name = item
                if isinstance(item, dict):
                    raw_name = item.get("method_name") or item.get("name") or item.get("method") or ""
                    selection_reason = str(
                        item.get("selection_reason")
                        or item.get("reason")
                        or item.get("relation")
                        or item.get("why")
                        or ""
                    ).strip()
                normalized = self._normalize_method_name(raw_name)
                if not normalized or normalized in seen:
                    continue
                if allowed and normalized not in allowed:
                    continue
                seen.add(normalized)
                required_methods.append(normalized)
                method_notes.append(
                    {
                        "method_name": normalized,
                        "why_blocking": selection_reason or f"`{normalized}` may change the exact repair behavior.",
                        "what_to_learn": selection_reason or f"Understand how `{normalized}` affects the local SATD fix.",
                    }
                )
        reason = str(payload.get("reason") or "").strip()
        return MethodInquiryResult(required_methods=required_methods, reason=reason, method_notes=method_notes)

    def _extract_uncertainty_candidates_from_satd_code(self, satd_comment: str, original_code: str) -> list[UncertaintyItem]:
        items: list[UncertaintyItem] = []
        seen: set[tuple[str, str]] = set()

        def add(
            kind: str,
            name: str,
            line: int | None = None,
            retrieval_eligible: bool = False,
            why: str = "",
            *,
            raw_name: str = "",
            normalized_name: str = "",
            source_excerpt: str = "",
        ) -> None:
            normalized_name = (normalized_name or name or "").strip()
            if not normalized_name:
                return
            key = (kind, normalized_name)
            if key in seen:
                return
            seen.add(key)
            items.append(
                UncertaintyItem(
                    kind=kind,
                    name=normalized_name,
                    line=line,
                    retrieval_eligible=retrieval_eligible,
                    why_this_matters=why,
                    raw_name=(raw_name or name or "").strip(),
                    normalized_name=normalized_name,
                    source_excerpt=(source_excerpt or "").strip(),
                )
            )

        for candidate in self._extract_method_candidates_from_satd_code(original_code):
            name = str(candidate.get("normalized_name") or "")
            add(
                "method",
                name,
                candidate.get("line"),
                True,
                "Method call appears in the edit region.",
                raw_name=str(candidate.get("raw_call") or name),
                normalized_name=name,
                source_excerpt=str(candidate.get("source_excerpt") or ""),
            )

        comment = satd_comment or ""
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_\.]+", comment):
            token = token.strip(".")
            if not token:
                continue
            lower = token.lower()
            if lower in {"todo", "fixme", "none", "true", "false"}:
                continue
            normalized_token = self._normalize_method_name(token) or token
            if self._looks_like_retrieval_symbol(token):
                add(
                    "symbol",
                    normalized_token,
                    None,
                    True,
                    "Comment explicitly names an API-like symbol.",
                    raw_name=token,
                    normalized_name=normalized_token,
                    source_excerpt=comment.strip(),
                )
            elif "." in token:
                add(
                    "value",
                    token,
                    None,
                    False,
                    "Comment references a value-like qualified name.",
                    raw_name=token,
                    normalized_name=token,
                    source_excerpt=comment.strip(),
                )

        try:
            tree = ast.parse(textwrap.dedent((original_code or "").strip("\n")))
        except (SyntaxError, ValueError):
            return items

        class LocalSignalCollector(ast.NodeVisitor):
            def visit_If(self, node: ast.If) -> Any:
                snippet = ast.get_source_segment(original_code, node.test) or ""
                cleaned = snippet.strip()
                add(
                    "condition",
                    cleaned,
                    getattr(node, "lineno", None),
                    False,
                    "Condition may determine the patch shape.",
                    raw_name=cleaned,
                    normalized_name=cleaned,
                    source_excerpt=cleaned,
                )
                self.generic_visit(node)

            def visit_Return(self, node: ast.Return) -> Any:
                snippet = ast.get_source_segment(original_code, node.value) or "return"
                cleaned = snippet.strip()
                add(
                    "return",
                    cleaned,
                    getattr(node, "lineno", None),
                    False,
                    "Return semantics may matter.",
                    raw_name=cleaned,
                    normalized_name=cleaned,
                    source_excerpt=cleaned,
                )
                self.generic_visit(node)

            def visit_Assign(self, node: ast.Assign) -> Any:
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                        name = f"self.{target.attr}"
                        add(
                            "state_write",
                            name,
                            getattr(node, "lineno", None),
                            False,
                            "State mutation may matter.",
                            raw_name=name,
                            normalized_name=name,
                            source_excerpt=(ast.get_source_segment(original_code, node) or name).strip(),
                        )
                self.generic_visit(node)

        LocalSignalCollector().visit(tree)
        return items

    def _looks_like_retrieval_symbol(self, token: str) -> bool:
        cleaned = str(token or "").strip()
        if not cleaned:
            return False
        normalized = self._normalize_method_name(cleaned)
        if not normalized:
            return False
        tail = normalized.split(".")[-1]
        tail_segments = self._method_name_segments(tail)
        if len(tail) < 3:
            return False
        if tail.lower() in {"default", "backend", "config", "true", "false", "none"}:
            return False
        callable_markers = {
            "send",
            "close",
            "open",
            "load",
            "save",
            "reset",
            "resolve",
            "create",
            "build",
            "update",
            "remove",
            "delete",
            "write",
            "read",
        }
        if any(marker in tail.lower() for marker in {"default", "backend", "config"}) and not any(
            word in tail_segments for word in callable_markers
        ):
            return False
        return (
            "." in normalized
            or "_" in tail
            or any(ch.isupper() for ch in tail)
            or any(word in tail_segments for word in callable_markers)
        )

    def _method_name_segments(self, value: str) -> set[str]:
        text = str(value or "").strip()
        if not text:
            return set()
        tokens: set[str] = set()
        for chunk in text.replace(".", "_").split("_"):
            for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", chunk):
                tokens.add(part.lower())
        return tokens

    def _candidate_uses_method_context(self, candidate_mode: str) -> bool:
        mode = str(candidate_mode or "").strip().lower()
        if not mode:
            return True
        return not mode.endswith("no_context")

    def _effective_candidate_mode(
        self,
        candidate_mode: str,
        found_method_contexts: list[RetrievedMethodContext],
    ) -> str:
        mode = str(candidate_mode or "").strip()
        if not mode:
            return mode
        if not self._candidate_uses_method_context(mode):
            return mode
        if found_method_contexts:
            return mode
        if mode.endswith("_context"):
            return mode[: -len("_context")] + "_no_context"
        return mode

    def _extract_method_candidates_from_satd_code(self, code: str) -> list[dict[str, Any]]:
        snippet = textwrap.dedent((code or "").strip("\n"))
        if not snippet.strip():
            return []
        try:
            tree = ast.parse(snippet)
        except (SyntaxError, ValueError):
            return []

        builtin_names = set(dir(builtins))
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()

        class CallCollector(ast.NodeVisitor):
            def __init__(self, outer: "OpenAIFixer") -> None:
                self.outer = outer

            def visit_Call(self, node: ast.Call) -> Any:
                raw_call = self.outer._call_chain_from_node(node.func)
                normalized = self.outer._normalize_method_name(raw_call)
                if normalized and normalized not in seen:
                    tail = normalized.split(".")[-1]
                    if raw_call.count(".") == 0 and tail in builtin_names:
                        self.generic_visit(node)
                        return
                    seen.add(normalized)
                    candidates.append(
                        {
                            "raw_call": raw_call,
                            "normalized_name": normalized,
                            "line": getattr(node, "lineno", None),
                            "source_excerpt": (ast.get_source_segment(snippet, node) or "").strip(),
                        }
                    )
                self.generic_visit(node)

        CallCollector(self).visit(tree)
        candidates.sort(
            key=lambda item: (
                item.get("line") if isinstance(item.get("line"), int) else 10**9,
                item.get("raw_call") or "",
            )
        )
        return candidates

    def _call_chain_from_node(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = self._call_chain_from_node(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        if isinstance(node, ast.Call):
            return self._call_chain_from_node(node.func)
        return ""

    def _serialize_method_notes(self, method_notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for item in method_notes or []:
            if not isinstance(item, dict):
                continue
            serialized.append(
                {
                    "method_name": str(item.get("method_name") or "").strip(),
                    "why_blocking": str(item.get("why_blocking") or "").strip(),
                    "what_to_learn": str(item.get("what_to_learn") or "").strip(),
                }
            )
        return serialized

    def _serialize_uncertainty_items(self, items: list[UncertaintyItem]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for item in items or []:
            normalized = item.normalized_name or self._normalize_method_name(item.name) or item.name
            serialized.append(
                {
                    "kind": item.kind,
                    "name": item.name,
                    "raw_name": item.raw_name or item.name,
                    "normalized_name": normalized,
                    "line": item.line,
                    "source_excerpt": item.source_excerpt,
                    "retrieval_eligible": item.retrieval_eligible,
                    "why_this_matters": item.why_this_matters,
                }
            )
        return serialized

    def _serialize_edit_constraints(self, items: list[EditConstraint]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for item in items or []:
            serialized.append(
                {
                    "focus_point": item.focus_point,
                    "required_fact": item.required_fact,
                    "must_do": item.must_do,
                    "must_not_do": item.must_not_do,
                    "supporting_point_kind": item.supporting_point_kind,
                    "target_kind": item.target_kind,
                    "target_hint": item.target_hint,
                    "allowed_edit_radius": item.allowed_edit_radius,
                    "must_preserve_signature": item.must_preserve_signature,
                    "must_not_add_helper": item.must_not_add_helper,
                    "must_not_expand_control_flow": item.must_not_expand_control_flow,
                    "must_not_rewrite_unrelated_lines": item.must_not_rewrite_unrelated_lines,
                    "supporting_symbol": item.supporting_symbol,
                    "supporting_evidence": item.supporting_evidence,
                    "confidence": item.confidence,
                }
            )
        return serialized

    def _serialize_retrieved_method_contexts(self, items: list[RetrievedMethodContext]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for item in items or []:
            serialized.append(
                {
                    "method_name": item.method_name,
                    "path": item.path,
                    "class_name": item.class_name,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "source": item.source,
                    "found": item.found,
                    "signature": item.signature,
                    "callsite_slice": item.callsite_slice,
                    "evidence_slice": item.evidence_slice,
                    "match_score": item.match_score,
                    "confidence": item.confidence,
                    "confidence_label": item.confidence_label,
                }
            )
        return serialized

    def _normalize_method_name(self, value: Any) -> str:
        text = str(value or "").strip().strip("`").strip()
        if not text:
            return ""
        text = text.split("(", 1)[0].strip()
        parts = [part for part in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text) if part]
        while len(parts) > 1 and parts[0] in {"self", "cls", "super"}:
            parts = parts[1:]
        if not parts:
            return ""
        normalized = ".".join(parts) if len(parts) > 1 else parts[0]
        return "" if self._is_low_quality_method_name(normalized) else normalized

    def _is_low_quality_method_name(self, value: str) -> bool:
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
        tail = parts[-1].lower()
        return tail in generic_terms
    def _retrieve_method_contexts(
        self,
        state: GraphState,
        method_names: list[str],
        method_location_hints: dict[str, list[str]] | None = None,
    ) -> list[RetrievedMethodContext]:
        if not method_names:
            return []
        payloads = self.client.toolbox.fetch_method_contexts(
            state["user"],
            state["project"],
            state["file_path"],
            method_names,
            ref=(state.get("commit") or "").strip() or None,
            log_prefix=self._task_prefix(state),
            path_hints_by_method=method_location_hints or {},
        )
        contexts: list[RetrievedMethodContext] = []
        for item in payloads:
            contexts.append(
                RetrievedMethodContext(
                    method_name=str(item.get("method_name") or ""),
                    path=str(item.get("path") or ""),
                    class_name=str(item.get("class_name")) if item.get("class_name") is not None else None,
                    start_line=item.get("start_line"),
                    end_line=item.get("end_line"),
                    source=str(item.get("source") or ""),
                    found=bool(item.get("found")),
                    signature=str(item.get("signature") or ""),
                    callsite_slice=str(item.get("callsite_slice") or ""),
                    evidence_slice=str(item.get("evidence_slice") or ""),
                    match_score=int(item.get("match_score") or 0),
                    confidence=float(item.get("confidence") or 0.0),
                    confidence_label=str(item.get("confidence_label") or ""),
                )
            )
        return contexts

    def _enrich_retrieved_method_contexts(
        self,
        *,
        state: GraphState,
        method_inquiry: MethodInquiryResult,
        retrieved_method_contexts: list[RetrievedMethodContext],
    ) -> list[RetrievedMethodContext]:
        if not retrieved_method_contexts:
            return []
        comment_refs = self._extract_comment_reference_names(state.get("satd_comment") or "")
        required_refs = {
            self._normalize_method_name(name)
            for name in (method_inquiry.required_methods or [])
            if self._normalize_method_name(name)
        }
        enriched: list[RetrievedMethodContext] = []
        for item in retrieved_method_contexts:
            signature = item.signature or self._extract_signature_line(item.source)
            callsite_slice = item.callsite_slice or self._extract_callsite_slice(state["original_code"], item.method_name)
            evidence_slice = item.evidence_slice or self._extract_relevant_definition_slice(item.source)
            match_score = item.match_score or self._score_retrieved_method_context(
                item=item,
                comment_refs=comment_refs,
                required_refs=required_refs,
                callsite_slice=callsite_slice,
                evidence_slice=evidence_slice,
            )
            confidence = item.confidence or self._confidence_from_match_score(match_score)
            confidence_label = item.confidence_label or self._confidence_label(confidence)
            enriched.append(
                RetrievedMethodContext(
                    method_name=item.method_name,
                    path=item.path,
                    class_name=item.class_name,
                    start_line=item.start_line,
                    end_line=item.end_line,
                    source=item.source,
                    found=item.found,
                    signature=signature,
                    callsite_slice=callsite_slice,
                    evidence_slice=evidence_slice,
                    match_score=match_score,
                    confidence=confidence,
                    confidence_label=confidence_label,
                )
            )
        return enriched

    def _score_retrieved_method_context(
        self,
        *,
        item: RetrievedMethodContext,
        comment_refs: set[str],
        required_refs: set[str],
        callsite_slice: str,
        evidence_slice: str,
    ) -> int:
        method_name = self._normalize_method_name(item.method_name)
        tail = method_name.split(".")[-1] if method_name else ""
        comment_tail_refs = {name.split(".")[-1] for name in comment_refs if name}
        required_tail_refs = {name.split(".")[-1] for name in required_refs if name}
        score = 0
        if method_name in comment_refs:
            score += 1200
        elif tail in comment_tail_refs:
            score += 350
        if method_name in required_refs:
            score += 800
        elif tail in required_tail_refs:
            score += 220
        if callsite_slice:
            score += 900
        if evidence_slice:
            score += 300
        if item.path:
            score += 80
        if item.class_name:
            score += 30
        if self._is_low_signal_evidence_method(tail):
            score -= 850
        return max(0, score)

    def _confidence_from_match_score(self, match_score: int) -> float:
        if match_score >= 1800:
            return 0.99
        if match_score >= 1200:
            return 0.82
        if match_score >= 800:
            return 0.62
        return 0.35

    def _confidence_label(self, confidence: float) -> str:
        if confidence >= 0.8:
            return "high"
        if confidence >= 0.6:
            return "medium"
        return "low"

    def _is_low_signal_evidence_method(self, tail: str) -> bool:
        lowered = str(tail or "").lower()
        return lowered in {"gradcheck", "gradgradcheck", "assertequal", "sum", "backward", "product", "run_test"}

    def _apply_method_context_gate(
        self,
        *,
        state: GraphState,
        method_inquiry: MethodInquiryResult,
        retrieved_method_contexts: list[RetrievedMethodContext],
    ) -> list[RetrievedMethodContext]:
        if not retrieved_method_contexts:
            return []
        retrieved_method_contexts = self._enrich_retrieved_method_contexts(
            state=state,
            method_inquiry=method_inquiry,
            retrieved_method_contexts=retrieved_method_contexts,
        )

        comment_refs = self._extract_comment_reference_names(state.get("satd_comment") or "")
        required_refs = {
            self._normalize_method_name(name)
            for name in (method_inquiry.required_methods or [])
            if self._normalize_method_name(name)
        }
        required_tail_refs = {name.split(".")[-1] for name in required_refs if name}
        ranked: list[tuple[tuple[int, int, int, int], RetrievedMethodContext]] = []
        for item, pack in zip(retrieved_method_contexts, self._build_method_evidence_packs(state, retrieved_method_contexts), strict=False):
            method_name = self._normalize_method_name(item.method_name)
            tail = method_name.split(".")[-1] if method_name else ""
            if item.confidence_label == "low" or item.confidence < 0.6:
                continue
            exact_comment_hit = int(bool(method_name and (method_name in comment_refs or tail in comment_refs)))
            has_callsite = int(bool(pack.get("callsite_slice")))
            has_evidence = int(bool(pack.get("evidence_slice")))
            required_match = int(bool(method_name and (method_name in required_refs or tail in required_tail_refs)))
            match_score = int(item.match_score or 0)
            if not (exact_comment_hit or has_callsite or required_match):
                continue
            ranked.append(((exact_comment_hit, has_callsite, has_evidence, match_score + required_match), item))

        if not ranked:
            return []
        ranked.sort(key=lambda entry: entry[0], reverse=True)
        return [item for _, item in ranked[: min(2, self.max_method_contexts)]]

    def _build_method_evidence_packs(
        self,
        state: GraphState,
        retrieved_method_contexts: list[RetrievedMethodContext],
    ) -> list[dict[str, Any]]:
        packs: list[dict[str, Any]] = []
        for item in retrieved_method_contexts:
            local_call = self._extract_local_call_line(
                state["original_code"],
                item.method_name,
                existing_slice=item.callsite_slice,
            )
            packs.append(
                {
                    "method_name": item.method_name,
                    "path": item.path,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "local_call": local_call,
                    "callsite_slice": item.callsite_slice or self._extract_callsite_slice(state["original_code"], item.method_name),
                    "method_behavior": self._build_method_behavior(
                        source=item.source,
                        method_name=item.method_name,
                        local_call=local_call,
                    ),
                    "evidence_slice": item.evidence_slice or self._extract_relevant_definition_slice(item.source),
                    "match_score": item.match_score,
                    "confidence": item.confidence,
                    "confidence_label": item.confidence_label,
                }
            )
        return packs

    def _extract_signature_line(self, source: str) -> str:
        for line in (source or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("async def ") or stripped.startswith("class "):
                return stripped
        return ""

    def _extract_callsite_slice(self, original_code: str, method_name: str) -> str:
        lines = (original_code or "").splitlines()
        if not lines:
            return ""
        token = (method_name or "").split(".")[-1]
        token_pattern = re.compile(rf"\b{re.escape(token)}\b") if token else None
        hit_indexes = [
            idx
            for idx, line in enumerate(lines)
            if token_pattern and token_pattern.search(line)
        ]
        if not hit_indexes:
            return ""
        center = hit_indexes[0]
        start = max(0, center - 2)
        end = min(len(lines), center + 3)
        return "\n".join(lines[start:end]).strip()

    def _extract_local_call_line(self, original_code: str, method_name: str, existing_slice: str = "") -> str:
        lines = (original_code or "").splitlines()
        token = (method_name or "").split(".")[-1]
        token_pattern = re.compile(rf"\b{re.escape(token)}\b") if token else None
        for line in lines:
            stripped = line.rstrip()
            if token_pattern and token_pattern.search(stripped):
                return stripped.strip()
        for line in (existing_slice or "").splitlines():
            stripped = line.strip()
            if token_pattern and token_pattern.search(stripped):
                return stripped
        return ""

    def _build_method_behavior(self, source: str, method_name: str, local_call: str = "") -> str:
        lines = (source or "").splitlines()
        if not lines:
            return ""
        if len(lines) <= 8:
            return "\n".join(lines).strip()

        signature_index = 0
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("def ", "async def ", "class ")):
                signature_index = index
                break

        anchor_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", local_call or "")
            if token.lower() not in {"self", "cls", "super", "true", "false", "none"}
        }
        method_tail = (method_name or "").split(".")[-1].lower()
        if method_tail:
            anchor_tokens.add(method_tail)

        best_index = min(signature_index + 1, len(lines) - 1)
        best_score = -1
        keyword_tokens = ("return", "raise", "yield", "await", "if ", "elif ", "else:", "except", "finally:")
        for index in range(signature_index + 1, len(lines)):
            stripped = lines[index].strip()
            lowered = stripped.lower()
            score = 0
            if any(token in lowered for token in anchor_tokens):
                score += 4
            if any(token in lowered for token in keyword_tokens):
                score += 3
            if "self." in stripped and "=" in stripped:
                score += 2
            if "=" in stripped:
                score += 1
            if score > best_score:
                best_score = score
                best_index = index

        body_window = 8
        body_start = max(signature_index + 1, best_index - 2)
        for index in range(best_index, signature_index, -1):
            stripped = lines[index].strip().lower()
            if stripped.startswith(("if ", "elif ", "else:", "except", "finally:")):
                body_start = index
                break
        max_body_start = max(signature_index + 1, len(lines) - body_window)
        body_start = min(body_start, max_body_start)
        body_end = min(len(lines), body_start + body_window)
        if body_end - body_start < 4 and len(lines) - (signature_index + 1) >= 4:
            body_end = min(len(lines), signature_index + 1 + 4)
            body_start = signature_index + 1

        snippet_lines = [lines[signature_index]]
        snippet_lines.extend(lines[body_start:body_end])
        return "\n".join(snippet_lines).strip()

    def _extract_relevant_definition_slice(self, source: str) -> str:
        lines = (source or "").splitlines()
        if not lines:
            return ""
        if len(lines) <= 12:
            return "\n".join(lines[:10]).strip()
        selected: list[str] = []
        keywords = ("return", "raise", "yield", "await", "if ", "elif ", "else:", "except", "finally:", "config", "default", "none", "true", "false")
        for line in lines:
            stripped = line.strip()
            lowered = stripped.lower()
            if stripped.startswith(("def ", "async def ", "class ")):
                selected.append(line)
                continue
            if "self." in stripped and "=" in stripped:
                selected.append(line)
                continue
            if any(token in lowered for token in keywords):
                selected.append(line)
            if len(selected) >= 6:
                break
        if not selected:
            selected = lines[:10]
        return "\n".join(selected[:6]).strip()

    def _synthesize_edit_constraints(
        self,
        *,
        state: GraphState,
        method_inquiry: MethodInquiryResult,
        retrieved_method_contexts: list[RetrievedMethodContext],
    ) -> list[EditConstraint]:
        constraints: list[EditConstraint] = []
        seen: set[str] = set()
        satd_comment = str(state.get("satd_comment") or "").strip()
        inquiry_lookup = {
            (item.normalized_name or self._normalize_method_name(item.name)): item
            for item in (method_inquiry.uncertainty_items or [])
            if (item.normalized_name or self._normalize_method_name(item.name))
        }
        for method_name in method_inquiry.required_methods[:2]:
            normalized = self._normalize_method_name(method_name)
            item = inquiry_lookup.get(normalized)
            focus = normalized or method_name
            if not focus or focus in seen:
                continue
            seen.add(focus)
            target_hint = (
                item.source_excerpt
                if item and item.source_excerpt
                else focus
            )
            constraints.append(
                EditConstraint(
                    focus_point=focus,
                    required_fact=(
                        f"The current edit point must satisfy the SATD intent ({satd_comment}) while staying consistent with `{focus}`."
                    ),
                    must_do=f"Use `{focus}` only as local supporting evidence for the current edit point.",
                    must_not_do="Do not treat retrieved evidence as permission to redesign the surrounding function.",
                    supporting_point_kind="method",
                    target_kind="method",
                    target_hint=target_hint,
                    supporting_symbol=focus,
                    supporting_evidence=(item.source_excerpt if item else ""),
                    confidence=0.45,
                )
            )
        for context in retrieved_method_contexts[:1]:
            focus = self._normalize_method_name(context.method_name) or context.method_name
            if not focus or focus in seen:
                continue
            seen.add(focus)
            constraints.append(
                EditConstraint(
                    focus_point=focus,
                    required_fact=(
                        f"The current edit point must satisfy the SATD intent ({satd_comment}) while staying consistent with `{focus}`."
                    ),
                    must_do=f"Use `{focus}` only as local supporting evidence for the current edit point.",
                    must_not_do="Do not treat retrieved evidence as permission to redesign the surrounding function.",
                    supporting_point_kind="method",
                    target_kind="method",
                    target_hint=context.callsite_slice.splitlines()[2].strip() if context.callsite_slice and len(context.callsite_slice.splitlines()) >= 3 else focus,
                    supporting_symbol=focus,
                    supporting_evidence=context.evidence_slice or context.callsite_slice,
                    confidence=context.confidence or 0.45,
                )
            )
        if not constraints:
            for item in method_inquiry.uncertainty_items or []:
                focus = item.name
                if not focus or focus in seen or item.kind not in {"condition", "return", "state_write", "value"}:
                    continue
                seen.add(focus)
                constraints.append(
                    EditConstraint(
                        focus_point=focus,
                        required_fact=item.why_this_matters or f"Preserve the local semantics around {focus}.",
                        must_do="Stay with the smallest local edit near the SATD comment.",
                        must_not_do="Do not introduce unrelated structural rewrites.",
                        supporting_point_kind=item.kind,
                        target_kind=item.kind,
                        target_hint=item.source_excerpt or focus,
                        supporting_symbol=item.normalized_name or focus,
                        supporting_evidence=item.source_excerpt,
                        confidence=0.45,
                    )
                )
                if len(constraints) >= 2:
                    break
        return constraints[:3]

    def resolve_method_locations(self, state: GraphState, method_names: list[str]) -> dict[str, list[str]]:
        if not method_names:
            return {}
        file_payload = self.client.toolbox.fetch_repo_file(
            state["user"],
            state["project"],
            state["file_path"],
            ref=(state.get("commit") or "").strip() or None,
        )
        file_content = file_payload.get("full_content") or "" if file_payload.get("ok") else ""
        tree_payload = self.client.toolbox.fetch_repo_tree(
            state["user"],
            state["project"],
            ref=(state.get("commit") or "").strip() or None,
        )
        repo_paths = [
            str(item.get("path") or "")
            for item in tree_payload.get("entries", [])
            if item.get("type") == "file"
        ]
        hints = self.client.toolbox.infer_method_path_hints(
            current_path=state["file_path"],
            method_names=method_names,
            current_content=file_content,
            repo_paths=repo_paths,
        )
        return hints

    def _log(self, state: GraphState, message: str) -> None:
        if not self.logger:
            return
        self.logger(f"{self._task_prefix(state)} {message}")

    def _format_method_list(self, method_names: list[str], empty: str = "none") -> str:
        items = [str(item).strip() for item in method_names if str(item).strip()]
        if not items:
            return empty
        if len(items) <= 3:
            return ", ".join(items)
        return ", ".join(items[:3]) + f", ... (+{len(items) - 3})"

    def _checkpoint(self, state: GraphState, stage: str, payload: dict[str, Any]) -> None:
        if not self.checkpoint_callback:
            return
        self.checkpoint_callback(state, stage, payload)

    def _task_prefix(self, state: GraphState) -> str:
        task_id = state.get("task_id", "?")
        task_index = state.get("task_index") or 0
        task_total = state.get("task_total") or 0
        if task_index and task_total:
            return f"[task {task_id} {task_index}/{task_total}]"
        return f"[task {task_id}]"

    def _build_repair_prompts(
        self,
        *,
        state: GraphState,
        round_id: int,
        method_inquiry: MethodInquiryResult,
        retrieved_method_contexts: list[RetrievedMethodContext],
        missing_method_names: list[str],
        edit_constraints: list[EditConstraint],
    ) -> tuple[str, str]:
        repair_feedback = state.get("repair_feedback") or {}
        reviewer_feedback_block = self._format_reviewer_feedback_block(repair_feedback)
        candidate_mode = str(state.get("candidate_mode") or "single")
        method_context_block = "[none]"
        if self._candidate_uses_method_context(candidate_mode):
            method_context_block = self._format_method_context_block(
                state=state,
                method_inquiry=method_inquiry,
                retrieved_method_contexts=retrieved_method_contexts,
                missing_method_names=missing_method_names,
            )
        if method_context_block == "[none]":
            return self._build_no_context_repair_prompts(state=state)
        return self._build_generic_repair_prompts(
            state=state,
            method_context_block=method_context_block,
            edit_constraints=edit_constraints,
            reviewer_feedback_block=reviewer_feedback_block,
        )

    def _build_no_context_repair_prompts(
        self,
        *,
        state: GraphState,
    ) -> tuple[str, str]:
        system_prompt = (
            "You are the fixer agent in a SATD repair workflow. "
        )
        user_prompt = (
            "Repair the code according to the SATD comment. Respond in JSON with key repaired_code.\n\n"
            f"### SATD comment:\n{state['satd_comment']}\n\n"
            f"Code:\n{state['original_code']}\n\n"
            "- When the SATD requests deleting or removing something, delete the corresponding executable code or statement, not only the SATD comment.\n"
        )
        return system_prompt, user_prompt

    def _build_generic_repair_prompts(
        self,
        *,
        state: GraphState,
        method_context_block: str,
        edit_constraints: list[EditConstraint],
        reviewer_feedback_block: str,
    ) -> tuple[str, str]:
        system_prompt = (
            "You are the fixer agent in a SATD repair workflow. "
            "Return valid JSON only with key: repaired_code. "
        )
        hard_rules = self._format_generic_constraint_block(edit_constraints)
        user_prompt = (
            "Repair the code according to the SATD comment. Respond in JSON with key repaired_code.\n\n"
            f"### SATD comment:\n{state['satd_comment']}\n\n"
            f"### Code:\n{state['original_code']}\n\n"
            f"### Supporting evidence:\n{method_context_block}\n\n"
            f"### Hard rules:\n{hard_rules}\n"
            "- Use supporting evidence only to validate the smallest local repair.\n"
            "- When the SATD requests deleting or removing something, delete the corresponding executable code or statement, not only the SATD comment.\n"
            "- Edit only the smallest local block nearest to the SATD comment.\n"
            f"{reviewer_feedback_block}"
        )
        return system_prompt, user_prompt

    def _default_repair_confidence(
        self,
        state: GraphState,
        repaired_code: str,
        found_method_contexts: list[RetrievedMethodContext],
    ) -> float:
        if preprocess_python_code(state["original_code"]) == preprocess_python_code(repaired_code):
            return 0.35
        if found_method_contexts:
            return 0.60
        return 0.50

    def _default_repair_plan(self, state: GraphState) -> str:
        comment = " ".join(str(state.get("satd_comment") or "").split())
        if not comment:
            return "Apply the smallest local repair required by the SATD."
        if len(comment) > 160:
            comment = comment[:157] + "..."
        return f"Apply the smallest local repair required by the SATD comment: {comment}"

    def _infer_changed_scope(self, original_code: str, repaired_code: str) -> str:
        if (original_code or "") == (repaired_code or ""):
            return "line"
        original_lines = (original_code or "").splitlines()
        repaired_lines = (repaired_code or "").splitlines()
        changed_line_count = abs(len(original_lines) - len(repaired_lines))
        for before, after in zip(original_lines, repaired_lines):
            if before != after:
                changed_line_count += 1
        if changed_line_count <= 2:
            return "line"
        if changed_line_count <= 12:
            return "function"
        if changed_line_count <= 40:
            return "class"
        return "file"

    def _format_method_context_block(
        self,
        *,
        state: GraphState,
        method_inquiry: MethodInquiryResult,
        retrieved_method_contexts: list[RetrievedMethodContext],
        missing_method_names: list[str],
    ) -> str:
        if not retrieved_method_contexts:
            return "[none]"
        lines: list[str] = []
        note_lookup: dict[str, str] = {}
        for item in method_inquiry.method_notes or []:
            if not isinstance(item, dict):
                continue
            method_name = self._normalize_method_name(item.get("method_name"))
            if not method_name:
                continue
            selection_reason = str(item.get("why_blocking") or item.get("what_to_learn") or "").strip()
            if selection_reason:
                note_lookup[method_name] = selection_reason
        for pack in self._build_method_evidence_packs(state, retrieved_method_contexts[:2]):
            method_name = str(pack["method_name"])
            lines.append(f"- method: {method_name}")
            if pack.get("local_call"):
                lines.append("  local call:")
                lines.append(f"    {pack['local_call']}")
            if pack.get("method_behavior"):
                lines.append("  method behavior:")
                for line in str(pack["method_behavior"]).splitlines():
                    lines.append(f"    {line}")
            selection_reason = note_lookup.get(self._normalize_method_name(method_name))
            if selection_reason:
                lines.append("  selection reason:")
                lines.append(f"    {selection_reason}")
        return "\n".join(lines)

    def _format_generic_constraint_block(self, edit_constraints: list[EditConstraint]) -> str:
        lines = [
            "- Make the smallest plausible local edit.",
            "- Preserve the existing function/class signature unless the SATD explicitly asks for a signature-local fix.",
            "- Do not add new helpers, new control flow, or unrelated rewrites.",
            "- Keep unchanged lines unchanged whenever possible.",
        ]
        for item in edit_constraints[:3]:
            focus = str(item.focus_point or "").strip()
            if focus:
                lines.append(f"- Focus on: {focus}")
            must_do = str(item.must_do or "").strip()
            if must_do:
                lines.append(f"- Must do: {must_do}")
            must_not_do = str(item.must_not_do or "").strip()
            if must_not_do:
                lines.append(f"- Must not do: {must_not_do}")
        return "\n".join(lines)

    def _extract_comment_reference_names(self, comment: str) -> set[str]:
        refs: set[str] = set()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_\.]+", comment or ""):
            normalized = self._normalize_method_name(token)
            if normalized:
                refs.add(normalized)
        return refs

    def _format_reviewer_feedback_block(self, repair_feedback: dict[str, Any]) -> str:
        if not repair_feedback:
            return ""
        lines = ["### Reviewer feedback from previous attempt:"]
        reject_type = repair_feedback.get("reject_type")
        revision_advice = repair_feedback.get("revision_advice")
        key_issues = repair_feedback.get("key_issues") or []
        if reject_type:
            lines.append(f"Reject type: {reject_type}")
        if revision_advice:
            lines.append(f"Revision advice: {revision_advice}")
        if key_issues:
            lines.append("Key issues:")
            for issue in key_issues[:3]:
                lines.append(f"- {issue}")
        flags = []
        if repair_feedback.get("has_api_shape_drift"):
            flags.append("api_shape_drift")
        if repair_feedback.get("has_noop_change"):
            flags.append("noop_change")
        if repair_feedback.get("has_over_edit"):
            flags.append("over_edit")
        if repair_feedback.get("has_comment_only_problem"):
            flags.append("comment_only_problem")
        if flags:
            lines.append(f"Reviewer flags: {', '.join(flags)}")
        return "\n".join(lines) + "\n\n"

    def _em_risk_notes(self, original_code: str, repaired_code: str) -> list[str]:
        risks = []
        original_signature = self._first_signature_line(original_code)
        repaired_signature = self._first_signature_line(repaired_code)
        if original_signature and repaired_signature and original_signature != repaired_signature:
            risks.append("signature_changed")
        if self._count_regex_matches(r"^\s*(?:async\s+def|def|class)\s+", repaired_code) > self._count_regex_matches(r"^\s*(?:async\s+def|def|class)\s+", original_code):
            risks.append("new_helper_or_definition")
        if self._count_regex_matches(r"\breturn\b", repaired_code) > self._count_regex_matches(r"\breturn\b", original_code):
            risks.append("new_return_path")
        return risks

    def _first_signature_line(self, code: str) -> str:
        for line in (code or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("async def ") or stripped.startswith("class "):
                return stripped
        return ""

    def _count_regex_matches(self, pattern: str, code: str) -> int:
        return len(re.findall(pattern, code or "", flags=re.MULTILINE))


class OpenAIReviewer:
    def __init__(self, client: OpenAICompatClient) -> None:
        self.client = client

    def run(self, state: GraphState) -> ReviewResult:
        analysis = state["analysis"]
        repair = state["latest_repair"]
        assert analysis is not None
        assert repair is not None

        metadata = ((state.get("github_context") or {}).get("metadata") or {}) if isinstance(state.get("github_context"), dict) else {}
        repair_evidence_mode = str(metadata.get("repair_evidence_mode") or "weak")
        snapshot_alignment_status = str(metadata.get("snapshot_alignment_status") or "mismatch")
        github_context = self.client.compact_shared_context(state.get("github_context"))
        system_prompt = (
            "You are the reviewer agent in a SATD repair workflow. "
            "You are a strict EM-oriented final gate. "
            "Prefer rejecting a plausible but over-written repair over approving a change that drifts away from the original patch shape. "
            "Reject repairs that add parameters, helper functions, return paths, exception paths, or control-flow rewrites without strong evidence. "
            "Use the analyzer evidence and repair/review context, but do not infer missing APIs or broader refactors from weak evidence. "
            "Weak evidence alone is not enough to reject a small in-place fix that preserves the original structure. "
            "A comment-only change can be valid when the SATD explicitly asks to remove an obsolete TODO, comment, or hack marker. "
            "Return JSON only."
        )
        user_prompt = (
            "Return a JSON object with keys: approved (bool), review_score (float 0-1), "
            "problem_alignment (float 0-1), minimality (float 0-1), semantic_preservation (float 0-1), internal_consistency (float 0-1), "
            "issues (array of strings), revision_advice (string), reject_type (string or null), rationale (string).\n"
            "Be conservative. Reject repairs that solve the comment by changing the surrounding API shape instead of applying the smallest credible local change.\n"
            "When evidence is weak, any new parameter, helper, return path, exception path, or notable control-flow rewrite should be treated as a likely EM miss.\n"
            "Do not reject purely because repository evidence is weak if the repair is a narrow in-place change with good SATD alignment.\n"
            "If the SATD explicitly says to remove an obsolete TODO, comment, or hack marker, a comment-only deletion can be approved.\n\n"
            f"Round: {repair.round_id}\n"
            f"Repository owner: {state['user']}\n"
            f"Repository name: {state['project']}\n"
            f"File path: {state['file_path']}\n"
            f"SATD comment: {state['satd_comment']}\n"
            f"Analysis decision: {analysis.decision}\n"
            f"Analysis repairability score: {analysis.repairability_score}\n"
            f"Analysis score: {analysis.analyze_score}\n"
            f"Analysis intent clarity: {analysis.intent_clarity}\n"
            f"Analysis change locality: {analysis.change_locality}\n"
            f"Analysis semantic risk: {analysis.semantic_risk}\n"
            f"Analysis context sufficiency: {analysis.context_sufficiency}\n"
            f"Analysis verifiability: {analysis.verifiability}\n"
            f"Analysis summary: {analysis.evidence_summary}\n"
            f"Analysis strategy: {analysis.repair_strategy}\n"
            f"Analysis scope radius: {analysis.scope_radius}\n"
            f"Analysis validation signals: {analysis.validation_signals}\n"
            f"Analysis context gaps: {analysis.context_gaps}\n"
            f"Analysis historical snapshot mismatch: {analysis.historical_snapshot_mismatch}\n"
            f"Analysis github evidence strength: {analysis.github_evidence_strength}\n"
            f"Snapshot alignment status: {snapshot_alignment_status}\n"
            f"Repair evidence mode: {repair_evidence_mode}\n"
            f"Original code block:\n{state['original_code']}\n\n"
            f"Repaired code block:\n{repair.repaired_code}\n\n"
            f"Repair plan: {repair.repair_plan}\n"
            f"Repair confidence: {repair.confidence}\n\n"
            f"Unified GitHub context:\n{github_context}\n"
        )
        payload = self.client.generate_json(system_prompt, user_prompt, request_label=f"review:task_{state['task_id']}:round_{repair.round_id}")

        problem_alignment = self._clamp_float(payload.get("problem_alignment"), 0.0)
        minimality = self._clamp_float(payload.get("minimality"), 0.0)
        semantic_preservation = self._clamp_float(payload.get("semantic_preservation"), 0.0)
        internal_consistency = self._clamp_float(payload.get("internal_consistency"), 0.0)
        weighted_score = self._weighted_review_score(problem_alignment, minimality, semantic_preservation, internal_consistency)
        review_score = max(
            self._clamp_float(payload.get("review_score"), weighted_score),
            weighted_score,
        )
        issues = [str(item).strip() for item in payload.get("issues", []) if str(item).strip()]
        approved = bool(payload.get("approved"))
        reject_type = payload.get("reject_type")
        rationale = str(payload.get("rationale") or "")
        revision_advice = str(payload.get("revision_advice") or "")
        softened_gate_used = False

        structural_risks = self._structural_em_risks(state["original_code"], repair.repaired_code)
        comment_only_allowed = self._comment_only_change_allowed(analysis, state["satd_comment"], state["original_code"], repair.repaired_code)
        signature_change_allowed = self._signature_change_allowed(analysis, state["original_code"], repair.repaired_code)
        low_risk_local_candidate = self._is_low_risk_local_candidate(analysis)
        risk_messages = {
            "signature_changed": "The repair changes the original signature, which is likely to miss the manual patch shape.",
            "new_helper_function": "The repair introduces a new helper or extra definition that may drift from the smallest manual patch shape.",
            "new_return_path": "The repair adds a return path that was not present in the original code.",
            "new_exception_path": "The repair adds an exception path that was not present in the original code.",
            "control_flow_expansion": "The repair expands control flow beyond a minimal local change.",
            "comment_only_change": "The repair only changes comments/docstrings without materially changing the code behavior.",
        }
        for risk in structural_risks:
            message = risk_messages.get(risk)
            if message and message not in issues:
                issues.append(message)

        if problem_alignment < 0.62:
            approved = False
            reject_type = reject_type or "not_satd_aligned"
            rationale = rationale + " | hard_gate=problem_not_addressed"

        if minimality < 0.50:
            approved = False
            reject_type = reject_type or "over_scoped_change"
            rationale = rationale + " | hard_gate=change_too_large"

        if semantic_preservation < 0.55:
            approved = False
            reject_type = reject_type or "unsafe_semantic_change"
            rationale = rationale + " | hard_gate=semantic_drift_risk"

        if "comment_only_change" in structural_risks and not comment_only_allowed:
            approved = False
            reject_type = reject_type or "comment_only_change"
            rationale = rationale + " | hard_gate=comment_only_change"

        if "signature_changed" in structural_risks and repair_evidence_mode != "strong" and not signature_change_allowed:
            approved = False
            reject_type = reject_type or "em_shape_risk"
            rationale = rationale + " | hard_gate=unsupported_signature_change"

        if "control_flow_expansion" in structural_risks and repair_evidence_mode != "strong" and minimality < 0.65:
            approved = False
            reject_type = reject_type or "em_shape_risk"
            rationale = rationale + " | hard_gate=unsupported_control_flow_expansion"

        needs_targeted_context = (
            not approved
            and repair_evidence_mode == "weak"
            and not structural_risks
            and (metadata.get("retrieved_test_snippets_count", 0) == 0 or metadata.get("retrieved_callsite_snippets_count", 0) == 0)
        )
        if needs_targeted_context:
            reject_type = reject_type or "missing_evidence"
            retrieval_targets = []
            if metadata.get("retrieved_test_snippets_count", 0) == 0:
                retrieval_targets.append("related_tests")
            if metadata.get("retrieved_callsite_snippets_count", 0) == 0:
                retrieval_targets.append("call_sites")
            if metadata.get("retrieved_history_snippets_count", 0) == 0:
                retrieval_targets.append("history_snippets")
            if retrieval_targets and "RETRIEVE:" not in revision_advice:
                revision_advice = (revision_advice + " " if revision_advice else "") + f"RETRIEVE:{','.join(retrieval_targets)}"

        credible_local_fix = (
            analysis.scope_radius in {"line", "function", "class", "file"}
            and analysis.risk_level != "high"
            and analysis.analyze_score >= 0.60
            and repair.confidence >= 0.35
            and review_score >= 0.35
            and problem_alignment >= 0.70
            and minimality >= (0.42 if low_risk_local_candidate else 0.50)
            and semantic_preservation >= 0.55
            and internal_consistency >= 0.55
            and ("comment_only_change" not in structural_risks or comment_only_allowed)
            and ("signature_changed" not in structural_risks or signature_change_allowed)
            and reject_type not in {
                "unsafe_semantic_change",
                "high_behavioral_risk",
                "architecture_level_change",
                "not_satd_aligned",
                "over_scoped_change",
            }
        )
        if not approved and credible_local_fix:
            approved = True
            reject_type = None
            softened_gate_used = True
            rationale = rationale + " | softened_gate=accepted_as_credible_local_fix"

        model_overreach_override = (
            not approved
            and not structural_risks
            and low_risk_local_candidate
            and review_score >= 0.62
            and problem_alignment >= 0.75
            and minimality >= 0.60
            and internal_consistency >= 0.60
            and reject_type not in {
                "unsafe_semantic_change",
                "high_behavioral_risk",
                "architecture_level_change",
                "not_satd_aligned",
                "comment_only_change",
            }
        )
        if not approved and (comment_only_allowed and problem_alignment >= 0.95 and review_score >= 0.90):
            approved = True
            reject_type = None
            softened_gate_used = True
            rationale = rationale + " | override=allowed_comment_only_satd"
        elif model_overreach_override:
            approved = True
            reject_type = None
            softened_gate_used = True
            rationale = rationale + " | override=model_overreach_without_structural_risk"

        severe_structural_risk = (
            ("comment_only_change" in structural_risks and not comment_only_allowed)
            or ("signature_changed" in structural_risks and repair_evidence_mode != "strong" and not signature_change_allowed)
        )
        review_score_floor = 0.50 if low_risk_local_candidate or comment_only_allowed or signature_change_allowed else 0.55
        if approved and (review_score < review_score_floor or severe_structural_risk):
            approved = False
            reject_type = reject_type or "borderline_review_confidence"
            rationale = rationale + " | hard_gate=review_not_strong_enough"
            softened_gate_used = False

        return ReviewResult(
            round_id=repair.round_id,
            approved=approved,
            review_score=review_score,
            problem_alignment=problem_alignment,
            minimality=minimality,
            semantic_preservation=semantic_preservation,
            internal_consistency=internal_consistency,
            issues=issues,
            revision_advice=revision_advice,
            reject_type=reject_type,
            rationale=rationale,
            softened_gate_used=softened_gate_used,
            candidate_mode=getattr(repair, "candidate_mode", "single"),
        )

    def _weighted_review_score(
        self,
        problem_alignment: float,
        minimality: float,
        semantic_preservation: float,
        internal_consistency: float,
    ) -> float:
        score = (
            0.40 * problem_alignment
            + 0.25 * minimality
            + 0.20 * semantic_preservation
            + 0.15 * internal_consistency
        )
        return max(0.0, min(1.0, score))

    def _structure_profile(self, code: str) -> dict[str, int]:
        source = code or ""
        try:
            tree = ast.parse(source)
            return {
                "definitions": sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree)),
                "returns": sum(isinstance(node, ast.Return) for node in ast.walk(tree)),
                "raises": sum(isinstance(node, ast.Raise) for node in ast.walk(tree)),
                "branches": sum(isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)) for node in ast.walk(tree)),
            }
        except SyntaxError:
            return {
                "definitions": len(re.findall(r"^\s*(?:async\s+def|def|class)\s+", source, flags=re.MULTILINE)),
                "returns": len(re.findall(r"\breturn\b", source)),
                "raises": len(re.findall(r"\braise\b", source)),
                "branches": len(re.findall(r"\b(if|for|while|try|with)\b", source)),
            }

    def _first_signature_line(self, code: str) -> str:
        for line in (code or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("async def ") or stripped.startswith("class "):
                return stripped
        return ""

    def _structural_em_risks(self, original_code: str, repaired_code: str) -> list[str]:
        original_profile = self._structure_profile(original_code)
        repaired_profile = self._structure_profile(repaired_code)
        risks = []
        original_signature = self._first_signature_line(original_code)
        repaired_signature = self._first_signature_line(repaired_code)
        if original_signature and repaired_signature and original_signature != repaired_signature:
            risks.append("signature_changed")
        if repaired_profile["definitions"] > original_profile["definitions"]:
            risks.append("new_helper_function")
        if repaired_profile["returns"] > original_profile["returns"]:
            risks.append("new_return_path")
        if repaired_profile["raises"] > original_profile["raises"]:
            risks.append("new_exception_path")
        if repaired_profile["branches"] > max(original_profile["branches"] + 1, int(original_profile["branches"] * 1.5) + 1):
            risks.append("control_flow_expansion")
        if preprocess_python_code(original_code) == preprocess_python_code(repaired_code) and (original_code or "").strip() != (repaired_code or "").strip():
            risks.append("comment_only_change")
        return risks

    def _comment_only_change_allowed(
        self,
        analysis: AnalysisResult,
        satd_comment: str,
        original_code: str,
        repaired_code: str,
    ) -> bool:
        normalized_original = preprocess_python_code(original_code)
        normalized_repaired = preprocess_python_code(repaired_code)
        if normalized_original != normalized_repaired:
            return False
        comment = (satd_comment or "").strip().lower()
        satd = (analysis.satd_type or "").strip().lower()
        removal_markers = ("remove", "delete", "drop", "cleanup", "obsolete")
        comment_markers = ("todo", "fixme", "hack", "comment")
        return satd in {"todo", "cleanup", "clarification"} and any(marker in comment for marker in removal_markers) and any(
            marker in comment for marker in comment_markers
        )

    def _signature_change_allowed(self, analysis: AnalysisResult, original_code: str, repaired_code: str) -> bool:
        satd = (analysis.satd_type or "").strip().lower()
        if satd not in {"type_annotation", "pyre-fixme", "pyre_fixme"}:
            return False
        return self._signature_change_kind(original_code, repaired_code) == "annotation_only"

    def _signature_change_kind(self, original_code: str, repaired_code: str) -> str:
        original_signature = self._first_signature_line(original_code)
        repaired_signature = self._first_signature_line(repaired_code)
        if not original_signature or not repaired_signature or original_signature == repaired_signature:
            return "none"
        original_skeleton = re.sub(r"\s+", "", re.sub(r":[^,)=]+", "", re.sub(r"->\s*[^:]+", "", original_signature)))
        repaired_skeleton = re.sub(r"\s+", "", re.sub(r":[^,)=]+", "", re.sub(r"->\s*[^:]+", "", repaired_signature)))
        return "annotation_only" if original_skeleton == repaired_skeleton else "shape_changed"

    def _is_low_risk_local_candidate(self, analysis: AnalysisResult) -> bool:
        satd = (analysis.satd_type or "").strip().lower()
        return (
            analysis.scope_radius in {"line", "function", "class", "file"}
            and analysis.risk_level in {"low", "medium"}
            and analysis.analyze_score >= 0.60
            and analysis.semantic_risk <= 0.70
            and satd not in {"architecture", "multi_file"}
        )

    def _clamp_float(self, value: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default


class OpenAISelector:
    def __init__(self, client: OpenAICompatClient) -> None:
        self.client = client

    def run(self, state: GraphState) -> SelectorDecision:
        candidates = state.get("repair_candidates") or []
        round_id = (candidates[0].round_id if candidates else state["round_id"] + 1)
        satd_route_type = str(state.get("satd_route_type") or "generic")
        if not candidates:
            return SelectorDecision(
                round_id=round_id,
                satd_route_type=satd_route_type,
                selected_candidate_mode="",
                selected_index=0,
                confidence=0.0,
                rationale="No repair candidates were available.",
                candidate_scores=[],
            )
        if len(candidates) == 1:
            only = candidates[0]
            return SelectorDecision(
                round_id=round_id,
                satd_route_type=satd_route_type,
                selected_candidate_mode=only.candidate_mode,
                selected_index=0,
                confidence=max(0.45, only.confidence),
                rationale="Only one repair candidate was available.",
                candidate_scores=[{"index": 0, "candidate_mode": only.candidate_mode, "score": only.confidence}],
            )

        system_prompt = (
            "You are the selector agent in a SATD repair workflow. "
            "Choose the single candidate that is most likely to match the intended minimal human repair. "
            "Prefer candidates that stay closest to the SATD comment, preserve the original structure, and avoid unnecessary API or control-flow changes. "
            "Return JSON only."
        )
        candidate_blocks = []
        for index, candidate in enumerate(candidates):
            candidate_blocks.append(
                f"Candidate {index} ({candidate.candidate_mode})\n"
                f"Plan: {candidate.repair_plan}\n"
                f"Scope: {candidate.changed_scope}\n"
                f"Confidence: {candidate.confidence}\n"
                f"Code:\n{candidate.repaired_code}\n"
            )
        user_prompt = (
            "Return a JSON object with keys: selected_index (integer), selected_candidate_mode (string), "
            "confidence (float 0-1), rationale (string), candidate_scores (array of objects with keys: index, candidate_mode, score).\n\n"
            f"SATD route type: {satd_route_type}\n"
            f"File path: {state['file_path']}\n"
            f"SATD comment: {state['satd_comment']}\n"
            f"Original code:\n{state['original_code']}\n\n"
            "Candidates:\n"
            + "\n".join(candidate_blocks)
        )
        payload = self.client.generate_json(system_prompt, user_prompt, request_label=f"select:task_{state['task_id']}:round_{round_id}")
        try:
            selected_index = int(payload.get("selected_index", 0))
        except (TypeError, ValueError):
            selected_index = 0
        if selected_index < 0 or selected_index >= len(candidates):
            selected_index = 0
        selected_candidate_mode = str(payload.get("selected_candidate_mode") or candidates[selected_index].candidate_mode)
        try:
            confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        rationale = str(payload.get("rationale") or "Selected the candidate that best matches the SATD comment with the smallest credible edit.")
        raw_scores = payload.get("candidate_scores")
        candidate_scores: list[dict[str, Any]] = []
        if isinstance(raw_scores, list):
            for item in raw_scores:
                if not isinstance(item, dict):
                    continue
                try:
                    idx = int(item.get("index", -1))
                except (TypeError, ValueError):
                    idx = -1
                if idx < 0 or idx >= len(candidates):
                    continue
                try:
                    score = max(0.0, min(1.0, float(item.get("score", 0.0))))
                except (TypeError, ValueError):
                    score = 0.0
                candidate_scores.append(
                    {
                        "index": idx,
                        "candidate_mode": str(item.get("candidate_mode") or candidates[idx].candidate_mode),
                        "score": score,
                    }
                )
        if not candidate_scores:
            candidate_scores = [
                {"index": idx, "candidate_mode": candidate.candidate_mode, "score": candidate.confidence}
                for idx, candidate in enumerate(candidates)
            ]
        selected_index, selected_candidate_mode, confidence, rationale, candidate_scores = self._apply_route_bias(
            satd_route_type=satd_route_type,
            candidates=candidates,
            selected_index=selected_index,
            selected_candidate_mode=selected_candidate_mode,
            confidence=confidence,
            rationale=rationale,
            candidate_scores=candidate_scores,
        )
        return SelectorDecision(
            round_id=round_id,
            satd_route_type=satd_route_type,
            selected_candidate_mode=selected_candidate_mode,
            selected_index=selected_index,
            confidence=confidence,
            rationale=rationale,
            candidate_scores=candidate_scores,
        )

    def _apply_route_bias(
        self,
        satd_route_type: str,
        candidates: list[RepairAttempt],
        selected_index: int,
        selected_candidate_mode: str,
        confidence: float,
        rationale: str,
        candidate_scores: list[dict[str, Any]],
    ) -> tuple[int, str, float, str, list[dict[str, Any]]]:
        if not candidates:
            return selected_index, selected_candidate_mode, confidence, rationale, candidate_scores

        score_by_index: dict[int, float] = {}
        normalized_scores: list[dict[str, Any]] = []
        for idx, candidate in enumerate(candidates):
            raw_item = next((item for item in candidate_scores if int(item.get("index", -1)) == idx), None)
            raw_score = raw_item.get("score") if raw_item else candidate.confidence
            try:
                score = max(0.0, min(1.0, float(raw_score)))
            except (TypeError, ValueError):
                score = max(0.0, min(1.0, float(candidate.confidence)))
            score_by_index[idx] = score
            normalized_scores.append(
                {
                    "index": idx,
                    "candidate_mode": candidate.candidate_mode,
                    "score": score,
                }
            )

        top_score = max(score_by_index.values())
        preference = self._route_preference_map(satd_route_type)
        eligible = [
            idx
            for idx, score in score_by_index.items()
            if top_score - score <= self._route_bias_margin(satd_route_type)
        ]
        if len(eligible) <= 1:
            return selected_index, selected_candidate_mode, confidence, rationale, normalized_scores

        def rank_key(idx: int) -> tuple[float, float, int]:
            mode = candidates[idx].candidate_mode
            return (
                preference.get(mode, 0.0),
                score_by_index[idx],
                1 if mode.endswith("no_context") else 0,
            )

        biased_index = max(eligible, key=rank_key)
        if biased_index == selected_index:
            return selected_index, selected_candidate_mode, confidence, rationale, normalized_scores

        biased_mode = candidates[biased_index].candidate_mode
        biased_score = score_by_index[biased_index]
        selected_score = score_by_index.get(selected_index, 0.0)
        bias_note = (
            f" Route bias applied for {satd_route_type}: preferred {biased_mode} over "
            f"{selected_candidate_mode or candidates[selected_index].candidate_mode} among near-tied candidates."
        )
        updated_confidence = max(confidence, min(0.95, max(biased_score, selected_score)))
        return biased_index, biased_mode, updated_confidence, rationale + bias_note, normalized_scores

    def _route_preference_map(self, satd_route_type: str) -> dict[str, float]:
        if satd_route_type == "remove_temporary":
            return {
                "typed_no_context": 1.00,
                "baseline_no_context": 0.92,
                "baseline_context": 0.35,
                "typed_context": 0.25,
            }
        if satd_route_type == "type_annotation":
            return {
                "typed_no_context": 1.00,
                "baseline_no_context": 0.92,
                "typed_context": 0.70,
                "baseline_context": 0.52,
            }
        if satd_route_type == "replace_symbol":
            return {
                "baseline_no_context": 1.00,
                "typed_no_context": 0.88,
                "baseline_context": 0.45,
                "typed_context": 0.30,
            }
        if satd_route_type == "document":
            return {
                "baseline_no_context": 1.00,
                "typed_no_context": 0.90,
                "baseline_context": 0.40,
                "typed_context": 0.28,
            }
        if satd_route_type == "generic":
            return {
                "baseline_context": 1.00,
                "typed_context": 0.82,
                "baseline_no_context": 0.55,
                "typed_no_context": 0.40,
            }
        return {
            "baseline_context": 1.00,
            "typed_context": 0.80,
            "baseline_no_context": 0.58,
            "typed_no_context": 0.42,
        }

    def _route_bias_margin(self, satd_route_type: str) -> float:
        if satd_route_type in {"remove_temporary", "type_annotation", "replace_symbol", "document"}:
            return 0.12
        return 0.10

