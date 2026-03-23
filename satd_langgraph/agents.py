from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

from .github_tools import GitHubToolbox
from .local_settings import OPENAI_API_KEY as LOCAL_OPENAI_API_KEY
from .local_settings import OPENAI_BASE_URL as LOCAL_OPENAI_BASE_URL
from .schema import AnalysisResult, GraphState, RepairAttempt, ReviewResult


VALID_DECISIONS = {"repairable", "drop", "needs_more_context"}
FINAL_DECISIONS = {"repairable", "drop"}
VALID_SCOPE_RADII = {"line", "function", "class", "file", "multi_file"}
VALID_DROP_REASONS = {
    "intent_too_vague",
    "architecture_level_change",
    "insufficient_context",
    "missing_validation_signal",
    "high_behavioral_risk",
    "too_many_call_sites",
    "external_dependency_blocked",
    "repo_history_conflict",
}
VALID_RISK_LEVELS = {"low", "medium", "high"}
VALID_GITHUB_EVIDENCE = {"low", "medium", "high"}


class OpenAICompatClient:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
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
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
        self.model = model
        self.toolbox = GitHubToolbox()

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        sanitized = False
        active_system_prompt = system_prompt
        active_user_prompt = user_prompt
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": active_system_prompt},
                        {"role": "user", "content": active_user_prompt},
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as exc:
                last_error = exc
                if self._is_content_filter_error(exc) and not sanitized:
                    active_system_prompt, active_user_prompt = self._sanitize_prompts(system_prompt, user_prompt)
                    sanitized = True
                    continue
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("OpenAI-compatible request failed unexpectedly.")

    def _is_content_filter_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "content_filter" in message or "content management policy" in message

    def _sanitize_prompts(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:
        compact_system = system_prompt + " Prefer concise evidence use and avoid unnecessary raw excerpts."
        compact_user = user_prompt
        compact_user = re.sub(
            r"(Base \+ repair \+ review context:\n)[\s\S]*",
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
    def build_base_context(self, state: GraphState, existing_bundle: dict[str, Any] | None = None) -> dict[str, Any]:
        bundle = self._ensure_bundle(existing_bundle, state)
        metadata = bundle.setdefault("metadata", {})
        if metadata.get("base_cached") and bundle.get("base_context"):
            if not metadata.get("base_cache_source"):
                metadata["base_cache_source"] = "memory"
            return self._augment_bundle_metadata(bundle)

        owner = state["user"]
        repo = state["project"]
        path = state["file_path"]
        issue_numbers = self._extract_issue_numbers(state["satd_comment"])[:2]
        file_payload = self.toolbox.fetch_repo_file(owner, repo, path)
        file_content = file_payload.get("full_content", "") if file_payload.get("ok") else ""
        satd_window = self.toolbox.extract_satd_window(file_content, state["satd_comment"])
        enclosing_symbol = self.toolbox.extract_enclosing_symbol(file_content, state["satd_comment"])
        imports = self.toolbox.extract_imports(file_content)
        module_prefix = self._module_prefix(path)

        bundle["base_context"] = {
            "file_path": path,
            "original_code": state["original_code"],
            "satd_comment": state["satd_comment"],
            "target_file": self._summarize_file_payload(file_payload),
            "satd_window": satd_window,
            "enclosing_symbol": enclosing_symbol,
            "imports": imports,
            "issue_refs": [self.toolbox.fetch_issue_or_pr(owner, repo, number) for number in issue_numbers],
            "module_docs": self.toolbox.fetch_readme_or_module_docs(owner, repo, module_prefix)
            if module_prefix
            else {"ok": False, "error": "no_module_prefix"},
        }
        metadata.update(
            {
                "base_cached": True,
                "base_context_fetched_at": self._timestamp(),
                "base_cache_source": metadata.get("base_cache_source") or "github_fetch",
                "symbol_name": enclosing_symbol.get("symbol_name") if isinstance(enclosing_symbol, dict) else None,
                "satd_line": satd_window.get("satd_line") if isinstance(satd_window, dict) else None,
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
        symbol_name = metadata.get("symbol_name") or self._fallback_symbol_name(state)
        satd_line = metadata.get("satd_line")
        module_prefix = self._module_prefix(path)
        issue_refs = (bundle.get("base_context", {}) or {}).get("issue_refs", [])
        issue_numbers = [item.get("number") for item in issue_refs if isinstance(item, dict) and item.get("ok")]
        history_query = self._build_history_query(state["satd_comment"], symbol_name)
        similar_history = (
            self.toolbox.search_closed_prs_or_issues(owner, repo, history_query)
            if history_query
            else {"ok": True, "query": "", "count": 0, "items": []}
        )
        repo_tree = self.toolbox.fetch_repo_tree(owner, repo, module_prefix)

        bundle["repair_context"] = {
            "related_tests": self.toolbox.find_related_tests(owner, repo, symbol_name or path),
            "call_sites": self.toolbox.find_call_sites(owner, repo, symbol_name),
            "repo_tree": repo_tree,
            "neighbor_files": self._neighbor_files(repo_tree, path),
            "commits_for_path": self.toolbox.fetch_commits_for_path(owner, repo, path, limit=5),
            "similar_history": similar_history,
            "issue_comments": [self.toolbox.fetch_issue_comments(owner, repo, number) for number in issue_numbers[:2]],
            "related_pr_files": [
                self.toolbox.fetch_pr_files(owner, repo, item.get("number"))
                for item in similar_history.get("items", [])
                if item.get("is_pull_request")
            ][:2],
            "last_commit_for_line": self.toolbox.fetch_blame_or_last_commit_for_line(owner, repo, path, satd_line),
        }
        metadata.update(
            {
                "repair_cached": True,
                "repair_context_fetched_at": self._timestamp(),
                "repair_cache_source": metadata.get("repair_cache_source") or "github_fetch",
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

        bundle["review_context"] = {
            "validation_signals_summary": {
                "analysis_validation_signals": analysis.validation_signals,
                "related_tests_count": related_tests.get("count", 0),
                "call_sites_count": call_sites.get("count", 0),
                "commits_count": commits_for_path.get("count", 0),
                "github_evidence_strength": analysis.github_evidence_strength,
            },
            "risk_indicators": risk_indicators,
            "change_scope_evidence": {
                "analysis_scope_radius": analysis.scope_radius,
                "repair_changed_scope": repair.changed_scope,
                "symbol_name": symbol.get("symbol_name"),
                "symbol_type": symbol.get("symbol_type"),
                "historical_snapshot_mismatch": analysis.historical_snapshot_mismatch,
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
                "historical_snapshot_mismatch": metadata.get("historical_snapshot_mismatch"),
                "github_evidence_strength": metadata.get("github_evidence_strength"),
                "target_file_ok": metadata.get("target_file_ok"),
                "satd_window_found": metadata.get("satd_window_found"),
                "enclosing_symbol_found": metadata.get("enclosing_symbol_found"),
                "related_tests_count": metadata.get("related_tests_count"),
                "call_sites_count": metadata.get("call_sites_count"),
                "commits_count": metadata.get("commits_count"),
                "similar_history_count": metadata.get("similar_history_count"),
                "symbol_name": metadata.get("symbol_name"),
                "satd_line": metadata.get("satd_line"),
            },
            "base_context": {
                "file_path": base.get("file_path"),
                "satd_comment": base.get("satd_comment"),
                "target_file": {
                    "ok": (base.get("target_file") or {}).get("ok"),
                    "path": (base.get("target_file") or {}).get("path"),
                    "content_excerpt": (base.get("target_file") or {}).get("content_excerpt"),
                    "total_lines": (base.get("target_file") or {}).get("total_lines"),
                },
                "satd_window": base.get("satd_window"),
                "enclosing_symbol": base.get("enclosing_symbol"),
                "imports": ((base.get("imports") or {}).get("imports", []) if isinstance(base.get("imports"), dict) else (base.get("imports") or []))[:20],
                "issue_refs": ((base.get("issue_refs") or []) if isinstance(base.get("issue_refs"), list) else [])[:2],
                "module_docs": base.get("module_docs"),
            },
        }
        if stage in {"repair", "review"}:
            compact["repair_context"] = {
                "related_tests": {
                    "count": (repair.get("related_tests") or {}).get("count", 0),
                    "items": (repair.get("related_tests") or {}).get("items", [])[:5],
                },
                "call_sites": {
                    "count": (repair.get("call_sites") or {}).get("count", 0),
                    "items": (repair.get("call_sites") or {}).get("items", [])[:5],
                },
                "neighbor_files": repair.get("neighbor_files"),
                "commits_for_path": {
                    "count": (repair.get("commits_for_path") or {}).get("count", 0),
                    "commits": (repair.get("commits_for_path") or {}).get("commits", [])[:2],
                },
                "similar_history": {
                    "count": (repair.get("similar_history") or {}).get("count", 0),
                    "items": (repair.get("similar_history") or {}).get("items", [])[:2],
                },
                "last_commit_for_line": repair.get("last_commit_for_line"),
            }
        if stage == "review":
            compact["review_context"] = review
        return json.dumps(compact, ensure_ascii=False)


    def _ensure_bundle(self, bundle: dict[str, Any] | None, state: GraphState) -> dict[str, Any]:
        current = dict(bundle or {})
        current.setdefault("task_id", state["task_id"])
        current.setdefault("repo_owner", state["user"])
        current.setdefault("repo_name", state["project"])
        current.setdefault("file_path", state["file_path"])
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
            return {"ok": False, "error": payload.get("error"), "path": payload.get("path")}
        full_content = payload.get("full_content") or ""
        return {
            "ok": True,
            "path": payload.get("path"),
            "sha": payload.get("sha"),
            "download_url": payload.get("download_url"),
            "content_excerpt": payload.get("content_excerpt"),
            "total_lines": len(full_content.splitlines()),
        }

    def _neighbor_files(self, repo_tree: dict[str, Any], current_path: str) -> dict[str, Any]:
        entries = repo_tree.get("entries", []) if isinstance(repo_tree, dict) else []
        neighbors = []
        current = (current_path or "").replace("\\", "/")
        for item in entries:
            path = (item.get("path") or "").replace("\\", "/")
            if not path or path == current or item.get("type") != "file":
                continue
            neighbors.append({"name": item.get("name"), "path": path})
            if len(neighbors) >= 8:
                break
        return {"count": len(neighbors), "items": neighbors}

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _augment_bundle_metadata(self, bundle: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(bundle.get("metadata", {}))
        base_context = bundle.get("base_context", {}) or {}
        repair_context = bundle.get("repair_context", {}) or {}

        target_file = base_context.get("target_file", {}) if isinstance(base_context, dict) else {}
        satd_window = base_context.get("satd_window", {}) if isinstance(base_context, dict) else {}
        enclosing_symbol = base_context.get("enclosing_symbol", {}) if isinstance(base_context, dict) else {}
        related_tests = repair_context.get("related_tests", {}) if isinstance(repair_context, dict) else {}
        call_sites = repair_context.get("call_sites", {}) if isinstance(repair_context, dict) else {}
        commits_for_path = repair_context.get("commits_for_path", {}) if isinstance(repair_context, dict) else {}
        similar_history = repair_context.get("similar_history", {}) if isinstance(repair_context, dict) else {}

        target_file_ok = bool(target_file.get("ok"))
        satd_window_found = bool(satd_window.get("found"))
        enclosing_symbol_found = bool(enclosing_symbol.get("found"))
        related_tests_count = int(related_tests.get("count") or 0)
        call_sites_count = int(call_sites.get("count") or 0)
        commits_count = int(commits_for_path.get("count") or 0)
        similar_history_count = int(similar_history.get("count") or 0)

        historical_snapshot_mismatch = not target_file_ok or not satd_window_found or not enclosing_symbol_found
        positive_signals = sum(
            1
            for flag in (
                target_file_ok,
                satd_window_found,
                enclosing_symbol_found,
                related_tests_count > 0,
                call_sites_count > 0,
                commits_count > 0,
                similar_history_count > 0,
            )
            if flag
        )
        if positive_signals >= 4 and target_file_ok:
            github_evidence_strength = "high"
        elif positive_signals >= 2 or target_file_ok:
            github_evidence_strength = "medium"
        else:
            github_evidence_strength = "low"

        metadata.update(
            {
                "target_file_ok": target_file_ok,
                "satd_window_found": satd_window_found,
                "enclosing_symbol_found": enclosing_symbol_found,
                "related_tests_count": related_tests_count,
                "call_sites_count": call_sites_count,
                "commits_count": commits_count,
                "similar_history_count": similar_history_count,
                "historical_snapshot_mismatch": historical_snapshot_mismatch,
                "github_evidence_strength": github_evidence_strength,
                "base_cached": bool(bundle.get("base_context")),
                "repair_cached": bool(bundle.get("repair_context")),
                "review_cached": bool(bundle.get("review_context")),
            }
        )
        bundle["metadata"] = metadata
        return bundle


class OpenAIAnalyzer:
    def __init__(self, client: OpenAICompatClient) -> None:
        self.client = client

    def run(self, state: GraphState) -> AnalysisResult:
        context_bundle = state.get("github_context") or self.client.build_base_context(state)
        github_context = self.client.format_context(context_bundle, layers=("base_context",))
        system_prompt = (
            "You are the analyzer agent in a SATD repair workflow. "
            "Your job is to decide whether a SATD item is a good candidate for automatic repair, not whether it is theoretically fixable in some ideal setting. "
            "Use SATD comment, original code, and file path as the primary evidence. "
            "Use current GitHub repository state only as auxiliary evidence because the SATD may come from an older snapshot that has already been repaired. "
            "Missing GitHub files, missing symbols, missing tests, or missing history must be treated as neutral missing evidence, not as automatic reasons to drop. "
            "Score the task on five general dimensions: intent_clarity, change_locality, semantic_risk, context_sufficiency, and verifiability. "
            "Prefer tasks that are clear, local, low-risk, and reviewable. "
            "Drop only when the primary evidence itself indicates vague intent, architecture-level change, high behavioral risk, or an external blocking dependency. "
            "Return JSON only."
        )
        user_prompt = (
            "Return a JSON object with keys: "
            "decision (repairable|drop|needs_more_context), repairability_score (float 0-1), confidence (float 0-1), "
            "satd_type (string), evidence_summary (string), risk_level (low|medium|high), context_score (float 0-1), "
            "clarity_score (float 0-1), scope_radius (line|function|class|file|multi_file), validation_signals (array of strings), "
            "context_gaps (array of strings), repair_strategy (string), drop_reason (one of: intent_too_vague, architecture_level_change, insufficient_context, "
            "missing_validation_signal, high_behavioral_risk, too_many_call_sites, external_dependency_blocked, repo_history_conflict, or null), "
            "followup_context_requests (array of strings), intent_clarity (float 0-1), change_locality (float 0-1), semantic_risk (float 0-1), "
            "primary_context_sufficiency (float 0-1), external_context_support (float 0-1), context_sufficiency (float 0-1), verifiability (float 0-1), analyze_score (float 0-1). "
            "Judge whether the SATD is worth sending into an automatic fixer, not whether the final repair is guaranteed to exactly match human code. "
            "Keep evidence_summary concise and evidence-based. If repository evidence is stale or missing but the task still looks clear and local from SATD comment plus original code, do not drop solely for that reason.\n\n"
            f"Repository owner: {state['user']}\n"
            f"Repository name: {state['project']}\n"
            f"File path: {state['file_path']}\n"
            f"SATD comment: {state['satd_comment']}\n"
            f"Current code snippet:\n{state['original_code']}\n\n"
            f"Base GitHub context:\n{github_context}\n"
        )
        payload = self.client.generate_json(system_prompt, user_prompt)
        return self._coerce_analysis(payload, state, context_bundle)

    def _coerce_analysis(
        self,
        payload: dict[str, Any],
        state: GraphState,
        context_bundle: dict[str, Any],
    ) -> AnalysisResult:
        metadata = (context_bundle.get("metadata") or {}) if isinstance(context_bundle, dict) else {}
        historical_snapshot_mismatch = bool(metadata.get("historical_snapshot_mismatch"))
        github_evidence_strength = self._normalize_choice(
            metadata.get("github_evidence_strength"), VALID_GITHUB_EVIDENCE, "low"
        )

        decision = self._normalize_decision(payload.get("decision"))
        repairability_score = self._clamp_float(payload.get("repairability_score"), 0.0)
        confidence = self._clamp_float(payload.get("confidence"), 0.0)
        satd_type = str(payload.get("satd_type") or "unknown")
        evidence_summary = str(payload.get("evidence_summary") or payload.get("reason") or "")
        risk_level = self._normalize_choice(payload.get("risk_level"), VALID_RISK_LEVELS, "medium")
        context_score = self._clamp_float(payload.get("context_score"), 0.0)
        clarity_score = self._clamp_float(payload.get("clarity_score"), 0.0)
        scope_radius = self._normalize_choice(payload.get("scope_radius"), VALID_SCOPE_RADII, "file")
        validation_signals = self._normalize_list(payload.get("validation_signals"))
        context_gaps = self._normalize_list(payload.get("context_gaps"))
        followup_context_requests = self._normalize_list(payload.get("followup_context_requests"))
        repair_strategy = str(payload.get("repair_strategy") or "")
        drop_reason = self._normalize_drop_reason(payload.get("drop_reason"), decision)

        intent_clarity = self._clamp_float(payload.get("intent_clarity"), clarity_score)
        change_locality = self._clamp_float(payload.get("change_locality"), self._scope_to_locality(scope_radius))
        semantic_risk = self._clamp_float(payload.get("semantic_risk"), self._risk_to_numeric(risk_level))
        primary_context_sufficiency = self._clamp_float(
            payload.get("primary_context_sufficiency"),
            context_score,
        )
        external_context_support = self._clamp_float(
            payload.get("external_context_support"),
            self._estimate_external_context_support(metadata),
        )
        context_sufficiency = self._clamp_float(
            payload.get("context_sufficiency"),
            primary_context_sufficiency,
        )
        verifiability = self._clamp_float(
            payload.get("verifiability"),
            self._estimate_verifiability(scope_radius, validation_signals, context_gaps),
        )
        analyze_score = self._clamp_float(
            payload.get("analyze_score"),
            self._weighted_analyze_score(
                intent_clarity,
                change_locality,
                semantic_risk,
                primary_context_sufficiency,
                verifiability,
            ),
        )

        if historical_snapshot_mismatch and "github_snapshot_mismatch" not in validation_signals:
            validation_signals.append("github_snapshot_mismatch")

        primary_clear = self._primary_evidence_clear(
            state,
            intent_clarity=intent_clarity,
            change_locality=change_locality,
            context_sufficiency=primary_context_sufficiency,
            analyze_score=analyze_score,
            scope_radius=scope_radius,
        )
        local_enough = scope_radius in {"line", "function", "class", "file"} and change_locality >= 0.52
        hard_drop_reason = drop_reason in {
            "intent_too_vague",
            "architecture_level_change",
            "high_behavioral_risk",
            "external_dependency_blocked",
        }
        weak_drop_reason = drop_reason in {
            None,
            "insufficient_context",
            "missing_validation_signal",
            "too_many_call_sites",
            "repo_history_conflict",
        }
        forced_non_drop = (
            scope_radius in {"line", "function", "class", "file"}
            and intent_clarity >= 0.55
            and change_locality >= 0.60
            and primary_context_sufficiency >= 0.55
            and semantic_risk <= 0.65
        )
        strong_repairable = (
            analyze_score >= 0.68
            and intent_clarity >= 0.60
            and change_locality >= 0.60
            and semantic_risk <= 0.70
        )

        if decision == "needs_more_context" and (
            primary_clear
            and local_enough
            and intent_clarity >= 0.65
            and change_locality >= 0.65
            and primary_context_sufficiency >= 0.60
            and semantic_risk <= 0.55
            and analyze_score >= 0.70
        ):
            decision = "repairable"
            repairability_score = max(repairability_score, analyze_score, 0.70)
            confidence = max(confidence, 0.60)
            drop_reason = None
            followup_context_requests = []

        if decision == "drop" and weak_drop_reason:
            if (
                scope_radius in {"line", "function", "class"}
                and intent_clarity >= 0.72
                and change_locality >= 0.70
                and semantic_risk <= 0.50
                and analyze_score >= 0.72
            ):
                decision = "repairable"
                repairability_score = max(repairability_score, analyze_score, 0.72)
                confidence = max(confidence, 0.64 if historical_snapshot_mismatch else 0.62)
                drop_reason = None
            elif analyze_score >= 0.50 and not hard_drop_reason:
                decision = "needs_more_context"
                drop_reason = "insufficient_context"

        if (
            decision == "drop"
            and drop_reason == "architecture_level_change"
            and scope_radius in {"line", "function", "class"}
            and primary_context_sufficiency >= 0.55
            and semantic_risk <= 0.65
        ):
            decision = "needs_more_context"
            repairability_score = max(repairability_score, analyze_score, 0.58)
            confidence = max(confidence, 0.57)

        if decision == "drop" and forced_non_drop and not hard_drop_reason:
            decision = "repairable" if strong_repairable else "needs_more_context"
            repairability_score = max(repairability_score, analyze_score, 0.62 if decision == "repairable" else 0.56)
            confidence = max(confidence, 0.58)
            if decision == "repairable":
                drop_reason = None

        if strong_repairable:
            decision = "repairable"
            repairability_score = max(repairability_score, analyze_score, 0.68)
            drop_reason = None
        elif decision != "drop" and analyze_score >= 0.50:
            decision = "needs_more_context"

        if decision == "repairable" and primary_clear and local_enough and semantic_risk <= 0.70:
            repairability_score = max(repairability_score, analyze_score, 0.68)
            confidence = max(confidence, 0.58)
            drop_reason = None

        if decision == "repairable":
            repair_strategy = repair_strategy or "Apply a local, behavior-preserving fix within the smallest stable scope."
            followup_context_requests = []
        elif decision == "needs_more_context":
            repair_strategy = repair_strategy or "Gather only the missing context needed to localize the change."
            if not followup_context_requests:
                followup_context_requests = ["clarify_missing_local_context"]
        else:
            repair_strategy = repair_strategy or "Do not attempt automatic repair."
            if not drop_reason:
                drop_reason = "insufficient_context"

        return AnalysisResult(
            decision=decision,
            repairable=decision == "repairable",
            repairability_score=repairability_score,
            intent_clarity=intent_clarity,
            change_locality=change_locality,
            semantic_risk=semantic_risk,
            context_sufficiency=context_sufficiency,
            primary_context_sufficiency=primary_context_sufficiency,
            external_context_support=external_context_support,
            verifiability=verifiability,
            analyze_score=analyze_score,
            confidence=confidence,
            satd_type=satd_type,
            reason=evidence_summary,
            evidence_summary=evidence_summary,
            risk_level=risk_level,
            context_score=context_score,
            clarity_score=clarity_score,
            scope_radius=scope_radius,
            validation_signals=validation_signals,
            context_gaps=context_gaps,
            followup_context_requests=followup_context_requests,
            repair_strategy=repair_strategy,
            drop_reason=drop_reason,
            historical_snapshot_mismatch=historical_snapshot_mismatch,
            github_evidence_strength=github_evidence_strength,
        )

    def _primary_evidence_clear(
        self,
        state: GraphState,
        intent_clarity: float,
        change_locality: float,
        context_sufficiency: float,
        analyze_score: float,
        scope_radius: str,
    ) -> bool:
        comment = (state.get("satd_comment") or "").strip()
        code = (state.get("original_code") or "").strip()
        comment_tokens = len(re.findall(r"[A-Za-z_]+", comment))
        code_lines = len([line for line in code.splitlines() if line.strip()])
        has_local_structure = bool(re.search(r"\b(def|class|return|if|for|while|try|except|with|raise)\b", code))
        scope_local = scope_radius in {"line", "function", "class", "file"}
        return (
            comment_tokens >= 3
            and code_lines >= 1
            and has_local_structure
            and scope_local
            and intent_clarity >= 0.55
            and change_locality >= 0.55
            and context_sufficiency >= 0.50
            and analyze_score >= 0.60
        )

    def _scope_to_locality(self, scope_radius: str) -> float:
        return {
            "line": 0.95,
            "function": 0.85,
            "class": 0.70,
            "file": 0.58,
            "multi_file": 0.28,
        }.get(scope_radius, 0.50)

    def _risk_to_numeric(self, risk_level: str) -> float:
        return {"low": 0.22, "medium": 0.55, "high": 0.86}.get(risk_level, 0.55)

    def _estimate_verifiability(self, scope_radius: str, validation_signals: list[str], context_gaps: list[str]) -> float:
        score = 0.45
        if scope_radius in {"line", "function", "class"}:
            score += 0.10
        if validation_signals:
            score += min(0.25, 0.05 * len(validation_signals))
        if context_gaps:
            score -= min(0.20, 0.04 * len(context_gaps))
        return max(0.0, min(1.0, score))

    def _estimate_external_context_support(self, metadata: dict[str, Any]) -> float:
        evidence = 0.0
        if metadata.get("target_file_ok"):
            evidence += 0.22
        if metadata.get("satd_window_found"):
            evidence += 0.18
        if metadata.get("enclosing_symbol_found"):
            evidence += 0.15
        if (metadata.get("related_tests_count") or 0) > 0:
            evidence += 0.18
        if (metadata.get("call_sites_count") or 0) > 0:
            evidence += 0.12
        if (metadata.get("commits_count") or 0) > 0:
            evidence += 0.08
        if (metadata.get("similar_history_count") or 0) > 0:
            evidence += 0.07
        return max(0.0, min(1.0, evidence))

    def _weighted_analyze_score(
        self,
        intent_clarity: float,
        change_locality: float,
        semantic_risk: float,
        context_sufficiency: float,
        verifiability: float,
    ) -> float:
        score = (
            0.30 * intent_clarity
            + 0.25 * change_locality
            + 0.20 * context_sufficiency
            + 0.15 * verifiability
            + 0.10 * (1.0 - semantic_risk)
        )
        return max(0.0, min(1.0, score))

    def _normalize_decision(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in VALID_DECISIONS:
            return text
        if "repair" in text:
            return "repairable"
        if "need" in text or "more_context" in text or "context" in text:
            return "needs_more_context"
        if "drop" in text or "reject" in text:
            return "drop"
        return "needs_more_context"

    def _normalize_choice(self, value: Any, valid: set[str], default: str) -> str:
        text = str(value or "").strip().lower()
        return text if text in valid else default

    def _normalize_drop_reason(self, value: Any, decision: str) -> str | None:
        if decision == "repairable":
            return None
        text = str(value or "").strip().lower()
        if text in VALID_DROP_REASONS:
            return text
        keyword_map = {
            "vague": "intent_too_vague",
            "arch": "architecture_level_change",
            "context": "insufficient_context",
            "validation": "missing_validation_signal",
            "risk": "high_behavioral_risk",
            "call": "too_many_call_sites",
            "external": "external_dependency_blocked",
            "history": "repo_history_conflict",
        }
        for keyword, mapped in keyword_map.items():
            if keyword in text:
                return mapped
        return None

    def _normalize_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            pieces = re.split(r"[\n,;|]", value)
            return [piece.strip() for piece in pieces if piece.strip()]
        return [str(value).strip()]

    def _clamp_float(self, value: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default


class OpenAIFixer:
    def __init__(self, client: OpenAICompatClient) -> None:
        self.client = client

    def run(self, state: GraphState) -> RepairAttempt:
        analysis = state["analysis"]
        latest_review = state["latest_review"]
        assert analysis is not None

        round_id = state["round_id"] + 1
        github_context = self.client.compact_context_for_stage(state.get("github_context"), "repair")
        system_prompt = (
            "You are the fixer agent in a SATD repair workflow. "
            "Produce repaired code, not a patch description. "
            "Use the repair context to localize the smallest valid repair. "
            "Prefer the enclosing symbol, related tests, call sites, and similar history when available. "
            "Keep the change as local and behavior-aware as possible. "
            "Do not perform architecture rewrites. Return JSON only."
        )
        user_prompt = (
            "Return a JSON object with keys: "
            "repair_plan (string), repaired_code (string), changed_scope (string), "
            "confidence (float 0-1), notes (string).\n"
            "repaired_code must be the complete repaired version of the provided original code block.\n"
            "First infer the smallest plausible fix strategy, then produce the repaired code.\n\n"
            f"Round: {round_id}\n"
            f"Repository owner: {state['user']}\n"
            f"Repository name: {state['project']}\n"
            f"File path: {state['file_path']}\n"
            f"SATD comment: {state['satd_comment']}\n"
            f"Analyzer decision: {analysis.decision}\n"
            f"Analyzer type: {analysis.satd_type}\n"
            f"Analyzer risk: {analysis.risk_level}\n"
            f"Analyzer confidence: {analysis.confidence}\n"
            f"Analyzer repairability score: {analysis.repairability_score}\n"
            f"Analyzer scope radius: {analysis.scope_radius}\n"
            f"Analyzer validation signals: {analysis.validation_signals}\n"
            f"Analyzer context gaps: {analysis.context_gaps}\n"
            f"Analyzer historical snapshot mismatch: {analysis.historical_snapshot_mismatch}\n"
            f"Analyzer github evidence strength: {analysis.github_evidence_strength}\n"
            f"Analyzer evidence summary: {analysis.evidence_summary}\n"
            f"Analyzer strategy: {analysis.repair_strategy}\n"
            f"Previous review advice: {latest_review.revision_advice if latest_review else 'None'}\n"
            f"Original code block:\n{state['original_code']}\n\n"
            f"Base + repair context:\n{github_context}\n"
        )
        payload = self.client.generate_json(system_prompt, user_prompt)
        repair_plan = str(payload.get("repair_plan") or analysis.repair_strategy or "Apply the smallest plausible local fix.")
        repaired_code = str(payload.get("repaired_code") or state["original_code"])
        changed_scope = str(payload.get("changed_scope") or analysis.scope_radius or "function")
        if changed_scope not in {"line", "function", "class", "file", "multi_file"}:
            changed_scope = analysis.scope_radius or "function"
        try:
            confidence = max(0.0, min(1.0, float(payload.get("confidence", analysis.confidence or 0.45))))
        except (TypeError, ValueError):
            confidence = max(0.0, min(1.0, float(analysis.confidence or 0.45)))
        notes = str(payload.get("notes") or "model response normalized with fixer fallback")
        return RepairAttempt(
            round_id=round_id,
            repair_plan=repair_plan,
            repaired_code=repaired_code,
            changed_scope=changed_scope,
            confidence=confidence,
            notes=notes,
        )


class OpenAIReviewer:
    def __init__(self, client: OpenAICompatClient) -> None:
        self.client = client

    def run(self, state: GraphState) -> ReviewResult:
        analysis = state["analysis"]
        repair = state["latest_repair"]
        assert analysis is not None
        assert repair is not None

        github_context = self.client.compact_context_for_stage(state.get("github_context"), "review")
        system_prompt = (
            "You are the reviewer agent in a SATD repair workflow. "
            "You are a moderately strict final gate. "
            "Judge whether the repaired code is a focused, minimal, and correct response to the SATD. "
            "Use the analyzer evidence and repair/review context, but do not reject solely because repository evidence is incomplete or stale. "
            "If historical snapshot mismatch is present, rely more on SATD comment and original code semantics. "
            "Prioritize whether the repair actually solves the SATD's core problem over whether it merely looks like generally reasonable code. "
            "Return JSON only."
        )
        user_prompt = (
            "Return a JSON object with keys: approved (bool), review_score (float 0-1), "
            "problem_alignment (float 0-1), minimality (float 0-1), semantic_preservation (float 0-1), internal_consistency (float 0-1), "
            "issues (array of strings), revision_advice (string), reject_type (string or null), rationale (string).\n"
            "Use these meanings: problem_alignment = whether the repair truly responds to the SATD; minimality = whether the change stays within the smallest necessary scope; "
            "semantic_preservation = whether the repair keeps the original behavior boundaries unless the SATD explicitly asks to change them; internal_consistency = whether the repaired code is self-consistent and fits the context.\n"
            "Be moderately conservative. Approve strong local repairs, but reject repairs that fail to address the SATD, expand the scope too much, or create plausible semantic drift.\n\n"
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
            f"Original code block:\n{state['original_code']}\n\n"
            f"Repaired code block:\n{repair.repaired_code}\n\n"
            f"Repair plan: {repair.repair_plan}\n"
            f"Repair confidence: {repair.confidence}\n\n"
            f"Base + repair + review context:\n{github_context}\n"
        )
        payload = self.client.generate_json(system_prompt, user_prompt)

        problem_alignment = self._clamp_float(payload.get("problem_alignment"), 0.0)
        minimality = self._clamp_float(payload.get("minimality"), 0.0)
        semantic_preservation = self._clamp_float(payload.get("semantic_preservation"), 0.0)
        internal_consistency = self._clamp_float(payload.get("internal_consistency"), 0.0)
        change_compactness = self._clamp_float(
            payload.get("change_compactness"),
            self._estimate_change_compactness(state["original_code"], repair.repaired_code),
        )
        review_score = self._weighted_review_score(problem_alignment, minimality, semantic_preservation, internal_consistency)
        issues = [str(item).strip() for item in payload.get("issues", []) if str(item).strip()]
        reject_type = payload.get("reject_type")
        rationale = str(payload.get("rationale") or "")
        softened_gate_used = False
        strong_reject_types = {
            "unsafe_semantic_change",
            "high_behavioral_risk",
            "over_scoped_change",
            "not_satd_aligned",
        }

        approved = (
            review_score >= 0.75
            and problem_alignment >= 0.75
            and minimality >= 0.60
            and semantic_preservation >= 0.60
        )

        if problem_alignment < 0.58:
            approved = False
            reject_type = reject_type or "not_satd_aligned"
            rationale = rationale + " | hard_gate=problem_not_addressed"

        if minimality < 0.42:
            approved = False
            reject_type = reject_type or "over_scoped_change"
            rationale = rationale + " | hard_gate=change_too_large"

        if semantic_preservation < 0.45:
            approved = False
            reject_type = reject_type or "unsafe_semantic_change"
            rationale = rationale + " | hard_gate=semantic_drift_risk"

        if reject_type in strong_reject_types:
            approved = False

        credible_local_fix = (
            analysis.scope_radius in {"line", "function"}
            and analysis.risk_level == "low"
            and analysis.analyze_score >= 0.70
            and repair.confidence >= 0.75
            and review_score >= 0.78
            and problem_alignment >= 0.80
            and minimality >= 0.72
            and semantic_preservation >= 0.70
            and change_compactness >= 0.65
            and len(issues) == 0
            and reject_type is None
        )
        if not approved and credible_local_fix:
            approved = True
            reject_type = None
            softened_gate_used = True
            rationale = rationale + " | softened_gate=accepted_as_credible_local_fix"

        if approved and review_score < 0.60 and (problem_alignment < 0.62 or len(issues) >= 2):
            approved = False
            reject_type = reject_type or "borderline_review_confidence"
            rationale = rationale + " | softened_gate=reverted_due_to_borderline_confidence"
            softened_gate_used = False

        return ReviewResult(
            round_id=repair.round_id,
            approved=approved,
            review_score=review_score,
            problem_alignment=problem_alignment,
            minimality=minimality,
            semantic_preservation=semantic_preservation,
            internal_consistency=internal_consistency,
            change_compactness=change_compactness,
            issues=issues,
            revision_advice=str(payload.get("revision_advice") or ""),
            reject_type=reject_type,
            rationale=rationale,
            softened_gate_used=softened_gate_used,
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

    def _estimate_change_compactness(self, original_code: str, repaired_code: str) -> float:
        original_lines = [line.strip() for line in (original_code or "").splitlines() if line.strip()]
        repaired_lines = [line.strip() for line in (repaired_code or "").splitlines() if line.strip()]
        if not original_lines and not repaired_lines:
            return 1.0
        baseline = max(len(original_lines), 1)
        delta = abs(len(repaired_lines) - len(original_lines))
        overlap = len(set(original_lines) & set(repaired_lines))
        overlap_ratio = overlap / max(len(set(original_lines)), 1)
        size_penalty = min(1.0, delta / baseline)
        score = 0.7 * overlap_ratio + 0.3 * (1.0 - size_penalty)
        return max(0.0, min(1.0, score))

    def _clamp_float(self, value: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

