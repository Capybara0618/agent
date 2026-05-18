from __future__ import annotations

import csv
import difflib
import json
import os
import re
import shutil
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from .analyzer_agent import OpenAIAnalyzer
from .context_policy_agent import OpenAIContextPolicyAgent
from .fixer_agent import OpenAIFixer
from .llm_judge import LLMRepairJudge, llm_judge_result_to_row
from .openai_client import OpenAICompatClient
from .reviewer_agent import OpenAIReviewer
from .csv_loader import load_satd_csv
from .repair_metrics import (
    RepairMetricContext,
    average_metric_rows,
    build_metric_context,
    calculate_repair_metrics,
    metric_result_to_row,
)
from .schema import (
    AnalysisResult,
    ContextNeedDecision,
    ContextPolicyDecision,
    GraphState,
    MethodInquiryResult,
    RepairAttempt,
    RetrievedMethodContext,
    ReviewResult,
    SATDRecord,
    preprocess_python_code,
    record_to_graph_input,
    trace_from_state,
)
from .tools import GitHubDownloadTool, MethodContextTool, MethodRetrievalTool, RepositoryEvidenceRetriever, SimilarCodeRules


class LangGraphSATDWorkflow:
    def __init__(
        self,
        max_rounds: int = 2,
        model: str = "gpt-4o-mini",
        verbose: bool = False,
        write_batch_size: int = 10,
        repair_context_mode: str = "clone_treesitter",
        max_method_contexts: int = 2,
        repository_evidence_mode: str = "none",
        max_repository_evidence: int = 0,
        repository_evidence_prompt_mode: str = "append",
        repository_evidence_guidance_mode: str = "none",
        repository_evidence_rerank_mode: str = "none",
        policy_candidate_strategy: str = "single",
        policy_injection_strategy: str = "llm",
        repair_response_format: str = "json",
        analysis_only: bool = False,
        fixer_only: bool = False,
        force_route: str | None = None,
        enable_llm_judge: bool = True,
        judge_model: str | None = None,
    ) -> None:
        self.max_rounds = max(2, int(max_rounds))
        self.model = model
        self.verbose = verbose
        self.write_batch_size = write_batch_size
        self.repair_context_mode = repair_context_mode
        self.max_method_contexts = max(1, int(max_method_contexts))
        self.repository_evidence_mode = self._normalize_repository_evidence_mode(repository_evidence_mode)
        self.max_repository_evidence = max(0, int(max_repository_evidence))
        self.repository_evidence_prompt_mode = self._normalize_repository_evidence_prompt_mode(
            repository_evidence_prompt_mode
        )
        self.repository_evidence_guidance_mode = self._normalize_repository_evidence_guidance_mode(
            repository_evidence_guidance_mode
        )
        self.repository_evidence_rerank_mode = self._normalize_repository_evidence_rerank_mode(
            repository_evidence_rerank_mode
        )
        self.policy_candidate_strategy = self._normalize_policy_candidate_strategy(policy_candidate_strategy)
        self.policy_injection_strategy = self._normalize_policy_injection_strategy(policy_injection_strategy)
        self.repair_response_format = self._normalize_repair_response_format(repair_response_format)
        self.analysis_only = analysis_only
        self.fixer_only = fixer_only
        self.force_route = self._normalize_force_route(force_route)
        self.enable_llm_judge = bool(enable_llm_judge)
        self.judge_model = (judge_model or model or "gpt-4o-mini").strip()

        client = OpenAICompatClient(model=model, verbose=verbose)
        self.context_client = client
        self.github_downloader = GitHubDownloadTool()
        self.similar_code_rules = SimilarCodeRules()
        self.method_retriever = MethodRetrievalTool(
            self.github_downloader,
            self.similar_code_rules,
            logger=self._log,
        )
        self.repository_evidence_retriever = RepositoryEvidenceRetriever(self.github_downloader)
        self.context_policy_agent = OpenAIContextPolicyAgent(client, logger=self._log)
        self.method_context_tool = MethodContextTool(
            client,
            method_retriever=self.method_retriever,
            repair_context_mode=repair_context_mode,
            max_method_contexts=self.max_method_contexts,
            logger=self._log,
            checkpoint_callback=self._record_repair_debug_checkpoint,
        )
        self.analyzer = OpenAIAnalyzer(client)
        self.fixer = OpenAIFixer(
            client,
            repair_context_mode=repair_context_mode,
            max_method_contexts=self.max_method_contexts,
            repository_evidence_prompt_mode=self.repository_evidence_prompt_mode,
            repair_response_format=self.repair_response_format,
            logger=self._log,
            checkpoint_callback=self._record_repair_debug_checkpoint,
        )
        self.reviewer = OpenAIReviewer(client)
        self.llm_judge = (
            LLMRepairJudge(OpenAICompatClient(model=self.judge_model, verbose=verbose))
            if self.enable_llm_judge
            else None
        )
        self._llm_judge_cache: dict[tuple[str, str], Any] = {}
        self._current_output_dir: Path | None = None
        self._metric_context: RepairMetricContext = build_metric_context([])

    def run_record(self, record: SATDRecord, task_index: int = 0, task_total: int = 0):
        state = record_to_graph_input(record, self.max_rounds)
        state["task_index"] = task_index
        state["task_total"] = task_total
        self._log(
            f"{self._task_label(state)} start project={record.project} file={record.file_path} "
            f"commit={(record.commit or '')[:12]} satd={self._compact_satd_comment(record.satd_comment)}"
        )

        context_decision = self._route_context_need(state)
        state["context_decision"] = context_decision
        state["satd_route_type"] = context_decision.route

        if self.fixer_only:
            state = self._run_fixer_only(state)
            self._log(f"{self._task_label(state)} end status={state['status']} fixer_only=True")
            return trace_from_state(state, record.em_label)

        if context_decision.route == "no_context":
            state["analysis"] = self._bypass_analysis(state, "no_context", "context_router_no_context | analyzer_skipped")
            state["status"] = "repairable"
        else:
            state = self._run_analyzer_stage(state)

        if self.analysis_only:
            state["status"] = "repairable"
            self._log(f"{self._task_label(state)} end status={state['status']} analysis_only=True")
            return trace_from_state(state, record.em_label)

        state = self._run_repair_review_loop(state)
        self._log(f"{self._task_label(state)} end status={state['status']}")
        return trace_from_state(state, record.em_label)

    def _run_analyzer_stage(self, state: GraphState) -> GraphState:
        self._log(f"{self._task_label(state)} analyze start")
        context_bundle = self._load_or_build_base_context(state)
        method_inquiry = MethodInquiryResult(reason="analyzer_method_context_not_run")
        retrieved_method_contexts: list[RetrievedMethodContext] = []
        missing_method_names: list[str] = []
        uncertainty_items = []
        edit_constraints = []
        method_context_block = "[none]"

        try:
            (
                method_inquiry,
                retrieved_method_contexts,
                missing_method_names,
                uncertainty_items,
                edit_constraints,
                method_context_block,
            ) = self.method_context_tool.prepare_method_context(state, candidate_mode="baseline_context")
            context_bundle = self._attach_method_context(
                context_bundle,
                method_inquiry=method_inquiry,
                retrieved_method_contexts=retrieved_method_contexts,
                missing_method_names=missing_method_names,
            )
            analyzer_state = {
                **state,
                "github_context": context_bundle,
                "method_inquiry": method_inquiry,
                "retrieved_method_contexts": retrieved_method_contexts,
                "missing_method_names": missing_method_names,
                "uncertainty_items": uncertainty_items,
                "edit_constraints": edit_constraints,
            }
            analysis = self.analyzer.run(analyzer_state, method_context_block=method_context_block)
        except Exception as exc:
            if self.context_client._is_content_filter_error(exc):
                reason = "analyzer_content_filter"
            else:
                reason = f"analyzer_exception:{type(exc).__name__}"
            self._log(f"{self._task_label(state)} analyze failed reason={reason}; continuing with fallback plan")
            analysis = self._fallback_analysis(state, reason=reason)

        updated = {
            **state,
            "analysis": analysis,
            "github_context": context_bundle,
            "method_inquiry": method_inquiry,
            "uncertainty_items": uncertainty_items,
            "edit_constraints": edit_constraints,
            "retrieved_method_contexts": retrieved_method_contexts,
            "missing_method_names": missing_method_names,
            "status": "repairable",
        }
        updated = self._attach_repository_evidence_for_repair(updated)
        self._log(
            f"{self._task_label(updated)} analyze done decision={analysis.decision} "
            f"repairable={analysis.repairable} context={self._compact_satd_comment(analysis.context_summary)}"
        )
        return updated

    def _run_fixer_only(self, state: GraphState) -> GraphState:
        route = str(state.get("satd_route_type") or "context_required")
        self._log(f"{self._task_label(state)} fixer-only context start route={route}")
        context_bundle = self._load_or_build_base_context(state)
        method_inquiry = MethodInquiryResult(reason="fixer_only_context_not_required")
        retrieved_method_contexts: list[RetrievedMethodContext] = []
        missing_method_names: list[str] = []
        uncertainty_items = []
        edit_constraints = []

        if route != "no_context" and self.repository_evidence_mode != "policy":
            try:
                (
                    method_inquiry,
                    retrieved_method_contexts,
                    missing_method_names,
                    uncertainty_items,
                    edit_constraints,
                    _method_context_block,
                ) = self.method_context_tool.prepare_method_context(state, candidate_mode="baseline_context")
            except Exception as exc:
                reason = (
                    "fixer_only_context_content_filter"
                    if self.context_client._is_content_filter_error(exc)
                    else f"fixer_only_context_exception:{type(exc).__name__}"
                )
                self._log(f"{self._task_label(state)} fixer-only context failed reason={reason}; continuing snippet-only")
                method_inquiry = MethodInquiryResult(reason=reason)

        context_bundle = self._attach_method_context(
            context_bundle,
            method_inquiry=method_inquiry,
            retrieved_method_contexts=retrieved_method_contexts,
            missing_method_names=missing_method_names,
        )
        analysis = self._bypass_analysis(state, route, "fixer_only_analyzer_skipped")
        repair_state = {
            **state,
            "analysis": analysis,
            "github_context": context_bundle,
            "method_inquiry": method_inquiry,
            "retrieved_method_contexts": retrieved_method_contexts,
            "missing_method_names": missing_method_names,
            "uncertainty_items": uncertainty_items,
            "edit_constraints": edit_constraints,
            "status": "repairable",
        }
        repair_state = self._attach_repository_evidence_for_repair(repair_state)

        if self._should_run_policy_dual_candidate(repair_state):
            return self._run_policy_dual_candidate_repair(repair_state)

        candidate_mode = "baseline_no_context" if route == "no_context" else "baseline_context"
        self._log(f"{self._task_label(repair_state)} fixer-only repair start mode={candidate_mode}")
        repair, method_inquiry, contexts, missing, uncertainty_items, edit_constraints = self._run_repair(
            repair_state,
            1,
            candidate_mode,
        )
        context_bundle = self._attach_method_context(
            repair_state.get("github_context") or self._load_or_build_base_context(repair_state),
            method_inquiry=method_inquiry,
            retrieved_method_contexts=contexts,
            missing_method_names=missing,
        )
        return {
            **repair_state,
            "github_context": context_bundle,
            "method_inquiry": method_inquiry,
            "retrieved_method_contexts": contexts,
            "missing_method_names": missing,
            "uncertainty_items": uncertainty_items,
            "edit_constraints": edit_constraints,
            "repair_context_used": self._repair_attempt_uses_any_evidence(repair, contexts, repair_state),
            "round_id": repair.round_id,
            "latest_repair": repair,
            "latest_review": None,
            "repair_feedback": None,
            "review_strict_gate_result": "skipped_fixer_only",
            "repairs": [*repair_state["repairs"], repair],
            "reviews": repair_state["reviews"],
            "status": "accepted",
            "final_repaired_code": repair.repaired_code,
        }

    def _should_run_policy_dual_candidate(self, state: GraphState) -> bool:
        if self.repository_evidence_mode != "policy":
            return False
        if self.policy_candidate_strategy == "single":
            return False
        decision = state.get("context_policy_decision")
        if not isinstance(decision, ContextPolicyDecision):
            return False
        if not decision.inject_evidence or not state.get("retrieved_repository_evidence"):
            return False
        if self.policy_candidate_strategy == "dual_all":
            return True
        risk = str(decision.context_risk or "").strip().lower()
        if risk != "low":
            return True
        selected = list(state.get("retrieved_repository_evidence") or [])
        has_direct = any(str(item.support_level or "").strip().lower() == "direct" for item in selected)
        return not has_direct or len(selected) < 2

    def _run_policy_dual_candidate_repair(self, state: GraphState) -> GraphState:
        self._log(f"{self._task_label(state)} policy dual-candidate repair start")
        baseline_state = {
            **state,
            "retrieved_repository_evidence": [],
            "repository_evidence_guidance": "",
            "retrieved_method_contexts": [],
            "missing_method_names": [],
            "repair_context_used": False,
        }
        baseline_repair, _, _, _, _, _ = self._run_repair(
            baseline_state,
            1,
            "baseline_no_context",
        )
        evidence_repair, method_inquiry, contexts, missing, uncertainty_items, edit_constraints = self._run_repair(
            state,
            2,
            "baseline_context",
        )
        selected_key, selector_reason = self._select_policy_repair_candidate(
            state,
            baseline_repair,
            evidence_repair,
        )
        selected_repair = evidence_repair if selected_key == "evidence_guarded" else baseline_repair
        if isinstance(state.get("context_policy_decision"), ContextPolicyDecision):
            decision = state["context_policy_decision"]
            decision.injection_decision_reason = " | ".join(
                item
                for item in [
                    decision.injection_decision_reason,
                    f"candidate_selector={selected_key}: {selector_reason}",
                ]
                if item
            )

        context_bundle = self._attach_method_context(
            state.get("github_context") or self._load_or_build_base_context(state),
            method_inquiry=method_inquiry,
            retrieved_method_contexts=contexts,
            missing_method_names=missing,
        )
        selected_uses_context = selected_repair is evidence_repair and self._repair_attempt_uses_any_evidence(
            evidence_repair,
            contexts,
            state,
        )
        self._log(
            f"{self._task_label(state)} policy dual-candidate selected={selected_key} "
            f"reason={selector_reason[:140]}"
        )
        return {
            **state,
            "github_context": context_bundle,
            "method_inquiry": method_inquiry,
            "retrieved_method_contexts": contexts if selected_repair is evidence_repair else [],
            "missing_method_names": missing if selected_repair is evidence_repair else [],
            "uncertainty_items": uncertainty_items,
            "edit_constraints": edit_constraints,
            "repair_context_used": selected_uses_context,
            "round_id": selected_repair.round_id,
            "latest_repair": selected_repair,
            "latest_review": None,
            "repair_feedback": None,
            "review_strict_gate_result": "skipped_fixer_only",
            "repairs": [*state["repairs"], baseline_repair, evidence_repair],
            "reviews": state["reviews"],
            "status": "accepted",
            "final_repaired_code": selected_repair.repaired_code,
        }

    def _select_policy_repair_candidate(
        self,
        state: GraphState,
        baseline_repair: RepairAttempt,
        evidence_repair: RepairAttempt,
    ) -> tuple[str, str]:
        if self._normalized_code_equal(baseline_repair.repaired_code, evidence_repair.repaired_code):
            return "baseline_guarded", "both candidates are equivalent after normalization"
        evidence_summary = self._selector_evidence_summary(state.get("retrieved_repository_evidence") or [])
        baseline_diff = self._compact_unified_diff(state["original_code"], baseline_repair.repaired_code)
        evidence_diff = self._compact_unified_diff(state["original_code"], evidence_repair.repaired_code)
        system_prompt = (
            "You select the final SATD repair candidate. "
            "Use only the SATD, original code, repository evidence, and candidate diffs. "
            "Do not use any hidden or manual answer. Return JSON only."
        )
        user_prompt = (
            "Choose which candidate should be submitted for exact-match SATD repair.\n\n"
            f"### SATD comment:\n{state['satd_comment']}\n\n"
            f"### Original code:\n```python\n{state['original_code']}\n```\n\n"
            f"### Repository evidence available to the evidence candidate:\n{evidence_summary}\n\n"
            f"### Candidate A: baseline_guarded diff\n```diff\n{baseline_diff}\n```\n\n"
            f"### Candidate B: evidence_guarded diff\n```diff\n{evidence_diff}\n```\n\n"
            "Selection rules:\n"
            "- Prefer the candidate that makes the smallest complete edit that resolves the SATD.\n"
            "- Choose evidence_guarded only when its extra edit is concretely supported by the repository evidence.\n"
            "- Choose baseline_guarded when evidence is weak, the evidence candidate rewrites unrelated code, or both candidates seem plausible.\n"
            "- Do not reward adding comments/docstrings unless the SATD specifically asks for documentation.\n\n"
            "Return JSON:\n"
            "{\n"
            '  "selected_candidate": "baseline_guarded" or "evidence_guarded",\n'
            '  "reason": "one concise reason"\n'
            "}\n"
        )
        try:
            payload = self.context_client.generate_json(
                system_prompt,
                user_prompt,
                temperature=0.0,
                request_label=f"policy_candidate_selector:task_{state['task_id']}",
                max_tokens=512,
            )
            selected = str(payload.get("selected_candidate") or "").strip().lower()
            reason = " ".join(str(payload.get("reason") or "").split())
            if selected not in {"baseline_guarded", "evidence_guarded"}:
                selected = "baseline_guarded"
            return selected, reason or "selector returned no reason"
        except Exception as exc:
            if str(evidence_repair.changed_scope or "") != "no_change" and str(baseline_repair.changed_scope or "") == "no_change":
                return "evidence_guarded", f"selector_exception:{type(exc).__name__}; evidence changed while baseline did not"
            return "baseline_guarded", f"selector_exception:{type(exc).__name__}; conservative fallback"

    def _run_repair_review_loop(self, state: GraphState) -> GraphState:
        state = self._attach_repository_evidence_for_repair(state)
        next_round = int(state["round_id"]) + 1
        route = str(state.get("satd_route_type") or "context_required")
        candidate_mode = "baseline_no_context" if route == "no_context" else "baseline_context"
        self._log(f"{self._task_label(state)} repair start round={next_round} mode={candidate_mode}")
        repair, method_inquiry, contexts, missing, uncertainty_items, edit_constraints = self._run_repair(
            state,
            next_round,
            candidate_mode,
        )
        context_bundle = self._attach_method_context(
            state.get("github_context") or self._load_or_build_base_context(state),
            method_inquiry=method_inquiry,
            retrieved_method_contexts=contexts,
            missing_method_names=missing,
        )
        state = {
            **state,
            "github_context": context_bundle,
            "method_inquiry": method_inquiry,
            "retrieved_method_contexts": contexts,
            "missing_method_names": missing,
            "uncertainty_items": uncertainty_items,
            "edit_constraints": edit_constraints,
            "repair_context_used": self._repair_attempt_uses_any_evidence(repair, contexts, state),
            "round_id": repair.round_id,
            "latest_repair": repair,
            "status": "repairing",
        }

        review = self._run_review(state)
        state = {
            **state,
            "repair_feedback": None,
            "review_strict_gate_result": review.gate_decision,
            "latest_review": review,
            "repairs": [*state["repairs"], repair],
            "reviews": [*state["reviews"], review],
            "status": "accepted" if review.approved else "dropped_after_review",
            "final_repaired_code": repair.repaired_code if review.approved else state["final_repaired_code"],
        }
        self._log(
            f"{self._task_label(state)} review done round={review.round_id} "
            f"gate={review.gate_decision} approved={review.approved} reject_type={review.reject_type or ''}"
        )
        return {**state, "status": "accepted" if review.approved else "dropped_after_review"}

    def _run_repair(
        self,
        state: GraphState,
        round_id: int,
        candidate_mode: str,
    ) -> tuple[RepairAttempt, MethodInquiryResult, list[RetrievedMethodContext], list[str], list, list]:
        repair_state = {**state, "candidate_mode": candidate_mode, "round_id": max(0, int(round_id) - 1)}
        try:
            return self.fixer.run(repair_state, candidate_mode=candidate_mode)
        except Exception as exc:
            error_message = " ".join(str(exc).split())
            error_traceback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=8))
            self._record_repair_debug_checkpoint(
                repair_state,
                "generation_error",
                {
                    "round_id": round_id,
                    "candidate_mode": candidate_mode,
                    "error_type": type(exc).__name__,
                    "error_message": error_message[:500],
                    "error_traceback": error_traceback[-4000:],
                },
            )
            if self.context_client._is_content_filter_error(exc):
                reason = f"fixer_content_filter:{candidate_mode}"
            else:
                reason = f"fixer_exception:{candidate_mode}:{type(exc).__name__}"
            repair = self._fallback_repair(state, round_id, reason=reason)
            repair.candidate_mode = candidate_mode
            return repair, state.get("method_inquiry") or MethodInquiryResult(), [], [], [], []

    def _run_review(self, state: GraphState) -> ReviewResult:
        try:
            return self.reviewer.run(state)
        except Exception as exc:
            if self.context_client._is_content_filter_error(exc):
                reason = "review_content_filter"
            else:
                reason = f"review_exception:{type(exc).__name__}"
            self._log(f"{self._task_label(state)} review failed reason={reason}; using fallback reject")
            return self._fallback_review(state, reason=reason)

    def _route_context_need(self, state: GraphState) -> ContextNeedDecision:
        if self.force_route:
            decision = ContextNeedDecision(
                route=self.force_route,
                context_required=self.force_route == "context_required",
                confidence=1.0,
                reason=f"forced_route:{self.force_route}",
                blocking_unknowns=[],
            )
            self._log(
                f"{self._task_label(state)} context routing forced route={decision.route} "
                f"confidence={decision.confidence:.2f}"
            )
            return decision
        self._log(f"{self._task_label(state)} context routing start")
        try:
            system_prompt = (
                "You are a context router for SATD repair. "
                "Decide whether the fixer needs repository context outside the shown snippet. "
                "Do not repair or suggest code. Return JSON only."
            )
            user_prompt = (
                "Decide whether this SATD can be reliably repaired from the shown snippet alone.\n\n"
                f"### SATD comment:\n{state['satd_comment']}\n\n"
                f"### Code:\n```python\n{state['original_code']}\n```\n\n"
                "Route by evidence sufficiency for conservative local repair.\n\n"
                "Choose \"no_context\" when the snippet alone makes a small local edit plausible, especially deleting obsolete "
                "or temporary code, removing an obvious SATD anchor, updating documentation, adding simple annotations, "
                "or making a clearly specified rename, literal, default, enable/disable, or local replacement.\n\n"
                "Choose \"context_required\" when reliable repair depends on external method behavior, missing symbols, "
                "repository conventions, an unclear replacement API, broad refactor or optimization intent, or an unclear target/end state.\n\n"
                "Return JSON:\n"
                "{\n"
                '  "route": "no_context" or "context_required",\n'
                '  "context_required": true or false,\n'
                '  "confidence": number from 0.0 to 1.0,\n'
                '  "reason": "one short sentence",\n'
                '  "blocking_unknowns": []\n'
                "}\n"
            )
            payload = self.context_client.generate_json(
                system_prompt,
                user_prompt,
                temperature=0.0,
                request_label=f"context_router:task_{state['task_id']}",
                max_tokens=512,
            )
            decision = self._coerce_context_need_decision(payload)
        except Exception as exc:
            reason = "context_router_content_filter" if self.context_client._is_content_filter_error(exc) else f"context_router_exception:{type(exc).__name__}"
            decision = ContextNeedDecision(
                route="context_required",
                context_required=True,
                confidence=0.0,
                reason=reason,
                blocking_unknowns=[reason],
            )
        self._log(
            f"{self._task_label(state)} context routing done route={decision.route} "
            f"confidence={decision.confidence:.2f}"
        )
        return decision

    def _coerce_context_need_decision(self, payload: dict[str, Any]) -> ContextNeedDecision:
        raw_route = str(payload.get("route") or "").strip().lower()
        raw_required = payload.get("context_required")
        confidence = self._clamp_float(payload.get("confidence"), 0.0)
        reason = " ".join(str(payload.get("reason") or "").split())
        blocking_unknowns: list[str] = []
        raw_unknowns = payload.get("blocking_unknowns")
        if isinstance(raw_unknowns, list):
            for item in raw_unknowns:
                cleaned = " ".join(str(item).split())
                if cleaned and cleaned not in blocking_unknowns:
                    blocking_unknowns.append(cleaned)
                if len(blocking_unknowns) >= 5:
                    break
        if isinstance(raw_required, bool):
            context_required = raw_required
        elif isinstance(raw_required, str):
            context_required = raw_required.strip().lower() in {"true", "yes", "1", "context_required"}
        else:
            context_required = raw_route == "context_required"
        route = "context_required" if context_required or raw_route != "no_context" or confidence < 0.55 else "no_context"
        if not reason:
            reason = "LLM context router selected this route."
        return ContextNeedDecision(
            route=route,
            context_required=route == "context_required",
            confidence=confidence,
            reason=reason,
            blocking_unknowns=blocking_unknowns,
        )

    def _normalize_force_route(self, value: str | None) -> str | None:
        route = str(value or "").strip().lower()
        if not route:
            return None
        if route not in {"no_context", "context_required"}:
            raise ValueError("force_route must be 'no_context', 'context_required', or empty.")
        return route

    def _normalize_repository_evidence_mode(self, value: str | None) -> str:
        mode = str(value or "none").strip().lower()
        if mode not in {"none", "fixed", "adaptive", "two_stage", "policy"}:
            raise ValueError("repository_evidence_mode must be one of: none, fixed, adaptive, two_stage, policy.")
        return mode

    def _normalize_repository_evidence_prompt_mode(self, value: str | None) -> str:
        mode = str(value or "append").strip().lower()
        if mode not in {"append", "constrained"}:
            raise ValueError("repository_evidence_prompt_mode must be 'append' or 'constrained'.")
        return mode

    def _normalize_repository_evidence_guidance_mode(self, value: str | None) -> str:
        mode = str(value or "none").strip().lower()
        if mode not in {"none", "summarize"}:
            raise ValueError("repository_evidence_guidance_mode must be 'none' or 'summarize'.")
        return mode

    def _normalize_repository_evidence_rerank_mode(self, value: str | None) -> str:
        mode = str(value or "none").strip().lower()
        if mode not in {"none", "llm"}:
            raise ValueError("repository_evidence_rerank_mode must be 'none' or 'llm'.")
        return mode

    def _normalize_policy_candidate_strategy(self, value: str | None) -> str:
        mode = str(value or "single").strip().lower()
        if mode not in {"single", "dual_on_uncertain", "dual_all"}:
            raise ValueError("policy_candidate_strategy must be one of: single, dual_on_uncertain, dual_all.")
        return mode

    def _normalize_policy_injection_strategy(self, value: str | None) -> str:
        mode = str(value or "llm").strip().lower()
        if mode not in {"llm", "top"}:
            raise ValueError("policy_injection_strategy must be 'llm' or 'top'.")
        return mode

    def _normalize_repair_response_format(self, value: str | None) -> str:
        mode = str(value or "json").strip().lower()
        if mode not in {"json", "plain_text"}:
            raise ValueError("repair_response_format must be 'json' or 'plain_text'.")
        return mode

    def run_csv(self, input_path: Path, output_dir: Path, limit: int | None = None, resume: bool = False) -> dict:
        records = load_satd_csv(input_path, limit=limit)
        self._metric_context = build_metric_context(record.manual_code for record in records)
        total = len(records)
        summary: dict | None = None
        output_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self._acquire_run_lock(output_dir)
        try:
            self._current_output_dir = output_dir
            if not resume:
                self._reset_output_dir_for_fresh_run(output_dir)
            self._context_cache_dir().mkdir(parents=True, exist_ok=True)
            if not resume:
                self._write_task_progress_csv(output_dir / "task_progress.csv", [])

            existing_rows = self._load_existing_rows(output_dir) if resume else self._empty_existing_rows()
            if resume:
                existing_rows = self._backfill_existing_metric_columns(output_dir, existing_rows)
            completed_ids = {row.get("task_id") for row in existing_rows["results"] if row.get("task_id")}
            pending_records = [record for record in records if record.task_id not in completed_ids]
            new_traces = []
            pending_flush_traces = []

            if self.verbose and resume:
                print(f"[resume] found {len(completed_ids)} completed tasks in {output_dir}")
                print(f"[resume] remaining {len(pending_records)}/{total} tasks to run")

            for offset, record in enumerate(pending_records, start=1):
                overall_index = len(completed_ids) + offset
                if self.verbose:
                    print(f"[progress] {overall_index}/{total} task_id={record.task_id} begin")
                trace = self.run_record(record, task_index=overall_index, task_total=total)
                new_traces.append(trace)
                pending_flush_traces.append(trace)
                self._append_task_progress_csv(output_dir / "task_progress.csv", [trace], overall_index, total)
                if self.verbose:
                    print(f"[progress] {overall_index}/{total} task_id={record.task_id} status={trace.status} rounds={trace.rounds_used} exact={trace.exact_match}")

                if offset % self.write_batch_size == 0 or offset == len(pending_records):
                    summary = self._with_summary_metadata(
                        self._summarize_from_existing_and_new(existing_rows, new_traces),
                        input_path=input_path,
                        input_limit=limit,
                    )
                    if resume and completed_ids:
                        self._append_outputs(output_dir, pending_flush_traces)
                    else:
                        self._write_outputs(output_dir, new_traces, summary)
                    self._write_summary_csv(output_dir / "summary.csv", summary)
                    pending_flush_traces = []
                    if self.verbose:
                        print(f"[flush] wrote {len(completed_ids) + len(new_traces)}/{total} tasks to {output_dir}")

            if not pending_records:
                summary = self._with_summary_metadata(
                    self._summarize_from_existing_and_new(existing_rows, []),
                    input_path=input_path,
                    input_limit=limit,
                )
                self._write_summary_csv(output_dir / "summary.csv", summary)

            assert summary is not None
            return summary
        finally:
            self._current_output_dir = None
            self._release_run_lock(lock_path)

    def _with_summary_metadata(
        self,
        summary: dict[str, Any],
        *,
        input_path: Path | None = None,
        input_limit: int | None = None,
    ) -> dict[str, Any]:
        if input_path is not None:
            summary["input_path"] = str(input_path)
        summary["input_limit"] = "" if input_limit is None else input_limit
        summary["agent_mode"] = "openai"
        summary["model"] = self.model
        summary["max_rounds"] = self.max_rounds
        summary["written_tasks"] = summary.get("input_satd_count", 0)
        summary["write_batch_size"] = self.write_batch_size
        summary["use_analyzer"] = not self.fixer_only
        summary["use_reviewer"] = not self.analysis_only and not self.fixer_only
        summary["analysis_only"] = self.analysis_only
        summary["fixer_only"] = self.fixer_only
        summary["repair_prompt_mode"] = "lightweight"
        summary["repair_context_mode"] = self.repair_context_mode
        summary["max_method_contexts"] = self.max_method_contexts
        summary["repository_evidence_mode"] = self.repository_evidence_mode
        summary["max_repository_evidence"] = self.max_repository_evidence
        summary["repository_evidence_prompt_mode"] = self.repository_evidence_prompt_mode
        summary["repository_evidence_guidance_mode"] = self.repository_evidence_guidance_mode
        summary["repository_evidence_rerank_mode"] = self.repository_evidence_rerank_mode
        summary["policy_candidate_strategy"] = self.policy_candidate_strategy
        summary["policy_injection_strategy"] = self.policy_injection_strategy
        summary["repair_response_format"] = self.repair_response_format
        summary["single_repair_path"] = self.policy_candidate_strategy == "single"
        summary["method_inquiry_enabled"] = (
            self.repair_context_mode in {"method_query", "clone_treesitter"}
            and not (self.fixer_only and self.repository_evidence_mode == "policy")
        )
        summary["context_router_enabled"] = self.force_route is None
        summary["force_route"] = self.force_route or ""
        summary["llm_judge_enabled"] = self.enable_llm_judge
        summary["judge_model"] = self.judge_model if self.enable_llm_judge else ""
        return summary

    def _empty_existing_rows(self) -> dict[str, list[dict[str, str]]]:
        return {"results": [], "trajectory": []}

    def _load_existing_rows(self, output_dir: Path) -> dict[str, list[dict[str, str]]]:
        return {
            "results": self._read_csv_rows(output_dir / "results.csv"),
            "trajectory": self._read_csv_rows(output_dir / "trajectory_overview.csv"),
        }

    def _backfill_existing_metric_columns(
        self,
        output_dir: Path,
        existing_rows: dict[str, list[dict[str, str]]],
    ) -> dict[str, list[dict[str, str]]]:
        result_rows = [self._ensure_metric_row(row) for row in existing_rows.get("results", [])]
        trajectory_rows = [self._ensure_metric_row(row) for row in existing_rows.get("trajectory", [])]
        if result_rows:
            self._write_csv_rows(output_dir / "results.csv", self._results_fieldnames(), result_rows)
        if trajectory_rows:
            self._write_csv_rows(output_dir / "trajectory_overview.csv", self._trajectory_fieldnames(), trajectory_rows)
        return {"results": result_rows, "trajectory": trajectory_rows}

    def _summarize_from_existing_and_new(self, existing_rows: dict[str, list[dict[str, str]]], new_traces: list) -> dict:
        result_rows = [
            *[self._ensure_metric_row(row) for row in existing_rows.get("results", [])],
            *[self._results_row(trace) for trace in new_traces],
        ]
        total = len(result_rows)
        analyze_filtered_count = 0
        analyzer_pass_count = total
        review_rejected_count = sum(1 for row in result_rows if row.get("status") == "dropped_after_review")
        workflow_output_count = sum(1 for row in result_rows if row.get("status") == "accepted")
        successful_repair_count = sum(1 for row in result_rows if str(row.get("exact_match")).lower() == "true")
        policy_injected_count = sum(1 for row in result_rows if str(row.get("context_policy_inject_evidence")).lower() == "true")
        evidence_aware_count = sum(1 for row in result_rows if int(float(row.get("repository_evidence_count") or 0)) > 0)
        snippet_only_count = total - evidence_aware_count
        historical_yes_success_count = sum(
            1
            for row in result_rows
            if str(row.get("em_label")).strip().upper() == "YES" and str(row.get("exact_match")).lower() == "true"
        )
        historical_no_success_count = sum(
            1
            for row in result_rows
            if str(row.get("em_label")).strip().upper() == "NO" and str(row.get("exact_match")).lower() == "true"
        )
        precision = round(successful_repair_count / workflow_output_count, 4) if workflow_output_count else 0.0
        recall = round(successful_repair_count / total, 4) if total else 0.0
        accepted_metrics = average_metric_rows(result_rows, accepted_only=True)
        llm_judge_stats = self._llm_judge_summary_stats(result_rows)
        return {
            "input_satd_count": total,
            "analyze_filtered_count": analyze_filtered_count,
            "analyzer_pass_count": analyzer_pass_count,
            "review_rejected_count": review_rejected_count,
            "workflow_output_count": workflow_output_count,
            "successful_repair_count": successful_repair_count,
            "policy_injected_count": policy_injected_count,
            "evidence_aware_count": evidence_aware_count,
            "snippet_only_count": snippet_only_count,
            "historical_yes_success_count": historical_yes_success_count,
            "historical_no_success_count": historical_no_success_count,
            "precision": precision,
            "recall": recall,
            "avg_BLEU_diff": accepted_metrics["avg_bleu_diff"],
            "avg_CrystalBLEU_diff": accepted_metrics["avg_crystalbleu_diff"],
            "avg_LEMOD": accepted_metrics["avg_lemod"],
            "avg_LLM_as_judge": accepted_metrics["avg_llm_as_judge"],
            **llm_judge_stats,
        }

    def _write_outputs(self, output_dir: Path, traces: list, summary: dict) -> None:
        self._write_trajectory_overview_csv(output_dir / "trajectory_overview.csv", traces)
        self._write_results_csv(output_dir / "results.csv", traces)
        self._write_repairs_csv(output_dir / "repairs.csv", traces)
        self._write_reviews_csv(output_dir / "reviews.csv", traces)
        self._write_github_context_csv(output_dir / "github_context.csv", traces)
        self._write_context_cache_csv(output_dir / "context_cache.csv", traces)
        self._write_summary_csv(output_dir / "summary.csv", summary)

    def _append_outputs(self, output_dir: Path, traces: list) -> None:
        self._append_trajectory_overview_csv(output_dir / "trajectory_overview.csv", traces)
        self._append_results_csv(output_dir / "results.csv", traces)
        self._append_repairs_csv(output_dir / "repairs.csv", traces)
        self._append_reviews_csv(output_dir / "reviews.csv", traces)
        self._append_github_context_csv(output_dir / "github_context.csv", traces)
        self._append_context_cache_csv(output_dir / "context_cache.csv", traces)

    def _trajectory_fieldnames(self) -> list[str]:
        return [
            "task_id", "project", "file_path", "commit", "context_commit", "status", "workflow_output", "drop_stage", "trajectory_summary", "rounds_used", "em_label", "exact_match", "satd_comment", "satd_route_type", "context_route", "context_required", "context_confidence", "context_reason", "context_blocking_unknowns",
            *self._metric_output_fields(),
            *self._analysis_output_fields("analysis_passed"),
            "repair_evidence_mode", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count",
            "identified_method_names", "retrieved_method_names", "missing_method_names", "retrieved_method_count", "method_context_json",
            *self._context_policy_output_fields(),
            "repository_evidence_count", "repository_evidence_json", "repository_evidence_guidance", "uncertainty_items_json", "edit_constraints_json",
            "repair_context_used", "review_strict_gate_result", "original_code", "processed_manual_code", "processed_final_repaired_code",
            *self._round_output_fields(1),
            *self._round_output_fields(2),
        ]

    def _results_fieldnames(self) -> list[str]:
        return [
            "task_id", "project", "file_path", "commit", "context_commit", "satd_comment", "status", "rounds_used", "em_label", "exact_match", "satd_route_type", "context_route", "context_required", "context_confidence", "context_reason", "context_blocking_unknowns",
            *self._metric_output_fields(),
            *self._analysis_output_fields("analysis_repairable"),
            "repair_evidence_mode", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count",
            "identified_method_names", "retrieved_method_names", "missing_method_names", "retrieved_method_count", "method_context_json",
            *self._context_policy_output_fields(),
            "repository_evidence_count", "repository_evidence_json", "repository_evidence_guidance", "uncertainty_items_json", "edit_constraints_json",
            "repair_context_used", "review_strict_gate_result", "original_code", "processed_manual_code", "processed_final_repaired_code",
            *self._round_output_fields(1),
            *self._round_output_fields(2),
        ]

    def _metric_output_fields(self) -> list[str]:
        return [
            "BLEU_diff",
            "CrystalBLEU_diff",
            "LEMOD",
            "LLM_as_judge",
        ]

    def _analysis_output_fields(self, repairable_field: str) -> list[str]:
        return [
            "analysis_decision",
            repairable_field,
            "analysis_reason",
            "analysis_repair_plan",
            "analysis_target_summary",
            "analysis_context_summary",
            "analysis_intent_type",
            "analysis_target_clarity",
            "analysis_expected_edit_shape",
            "analysis_evidence_requirement",
            "analysis_risk_note",
        ]

    def _context_policy_output_fields(self) -> list[str]:
        return [
            "context_policy_json",
            "context_policy_use_repository_context",
            "context_policy_needed_evidence_types",
            "context_policy_context_risk",
            "context_policy_inject_evidence",
            "context_policy_selected_evidence_indices",
            "context_policy_candidate_evidence_count",
            "context_policy_selected_evidence_count",
            "context_policy_reason",
            "context_policy_injection_reason",
        ]

    def _round_output_fields(self, round_id: int) -> list[str]:
        prefix = f"round_{round_id}"
        return [
            f"{prefix}_candidate_mode", f"{prefix}_repair_plan", f"{prefix}_repaired_code", f"{prefix}_changed_scope", f"{prefix}_fix_confidence",
            f"{prefix}_review_approved", f"{prefix}_review_gate_decision", f"{prefix}_review_failure_modes", f"{prefix}_review_score", f"{prefix}_review_problem_alignment", f"{prefix}_review_minimality", f"{prefix}_review_semantic_preservation", f"{prefix}_review_internal_consistency", f"{prefix}_review_softened_gate_used", f"{prefix}_review_reject_type", f"{prefix}_review_issues", f"{prefix}_review_failed_checks", f"{prefix}_review_repair_constraints", f"{prefix}_review_failure_anchor", f"{prefix}_revision_advice",
        ]

    def _write_trajectory_overview_csv(self, path: Path, traces: list) -> None:
        self._write_csv_rows(path, self._trajectory_fieldnames(), [self._trajectory_row(trace) for trace in traces])

    def _append_trajectory_overview_csv(self, path: Path, traces: list) -> None:
        self._append_csv_rows(path, self._trajectory_fieldnames(), [self._trajectory_row(trace) for trace in traces])

    def _write_results_csv(self, path: Path, traces: list) -> None:
        self._write_csv_rows(path, self._results_fieldnames(), [self._results_row(trace) for trace in traces])

    def _append_results_csv(self, path: Path, traces: list) -> None:
        self._append_csv_rows(path, self._results_fieldnames(), [self._results_row(trace) for trace in traces])

    def _trajectory_row(self, trace) -> dict[str, Any]:
        return self._base_output_row(trace, include_workflow=True)

    def _results_row(self, trace) -> dict[str, Any]:
        return self._base_output_row(trace, include_workflow=False)

    def _base_output_row(self, trace, include_workflow: bool) -> dict[str, Any]:
        analysis = trace.analysis or {}
        repairs = trace.repairs or []
        reviews = trace.reviews or []
        context = trace.github_context or {}
        metadata = context.get("metadata") or {}
        context_policy = trace.context_policy_decision or {}
        row: dict[str, Any] = {
            "task_id": trace.task_id,
            "project": trace.project,
            "file_path": trace.file_path,
            "commit": trace.commit,
            "context_commit": metadata.get("context_commit") or trace.commit,
            "status": trace.status,
            "satd_comment": trace.satd_comment,
            "rounds_used": trace.rounds_used,
            "em_label": trace.em_label,
            "exact_match": trace.exact_match,
            "satd_route_type": trace.satd_route_type,
            "context_route": trace.context_route,
            "context_required": trace.context_required,
            "context_confidence": trace.context_confidence,
            "context_reason": trace.context_reason,
            "context_blocking_unknowns": " | ".join(trace.context_blocking_unknowns or []),
            "repair_evidence_mode": self._repair_evidence_mode(trace),
            "retrieved_test_snippets_count": 0,
            "retrieved_callsite_snippets_count": 0,
            "retrieved_history_snippets_count": 0,
            "identified_method_names": " | ".join((trace.method_inquiry or {}).get("required_methods") or []),
            "retrieved_method_names": " | ".join(item.get("method_name", "") for item in trace.retrieved_method_contexts or [] if item.get("found")),
            "missing_method_names": " | ".join(trace.missing_method_names or []),
            "retrieved_method_count": len([item for item in trace.retrieved_method_contexts or [] if item.get("found")]),
            "method_context_json": json.dumps(trace.retrieved_method_contexts or [], ensure_ascii=False),
            "context_policy_json": json.dumps(context_policy, ensure_ascii=False),
            "context_policy_use_repository_context": context_policy.get("use_repository_context", ""),
            "context_policy_needed_evidence_types": " | ".join(context_policy.get("needed_evidence_types") or []),
            "context_policy_context_risk": context_policy.get("context_risk", ""),
            "context_policy_inject_evidence": context_policy.get("inject_evidence", ""),
            "context_policy_selected_evidence_indices": " | ".join(
                str(item) for item in (context_policy.get("selected_evidence_indices") or [])
            ),
            "context_policy_candidate_evidence_count": context_policy.get("candidate_evidence_count", ""),
            "context_policy_selected_evidence_count": context_policy.get("selected_evidence_count", ""),
            "context_policy_reason": context_policy.get("decision_reason", ""),
            "context_policy_injection_reason": context_policy.get("injection_decision_reason", ""),
            "repository_evidence_count": len(trace.retrieved_repository_evidence or []),
            "repository_evidence_json": json.dumps(trace.retrieved_repository_evidence or [], ensure_ascii=False),
            "repository_evidence_guidance": trace.repository_evidence_guidance,
            "uncertainty_items_json": json.dumps(trace.uncertainty_items or [], ensure_ascii=False),
            "edit_constraints_json": json.dumps(trace.edit_constraints or [], ensure_ascii=False),
            "repair_context_used": trace.repair_context_used,
            "review_strict_gate_result": trace.review_strict_gate_result,
            "original_code": trace.original_code,
            "processed_manual_code": trace.processed_manual_code,
            "processed_final_repaired_code": trace.processed_final_repaired_code or "",
        }
        row.update(
            metric_result_to_row(
                calculate_repair_metrics(
                    trace.original_code,
                    trace.processed_manual_code,
                    trace.processed_final_repaired_code,
                    self._metric_context,
                )
            )
        )
        row.update(self._llm_judge_row(trace, row))
        if include_workflow:
            row["workflow_output"] = "YES" if trace.status == "accepted" else "NO"
            row["drop_stage"] = self._drop_stage(trace)
            row["trajectory_summary"] = self._trajectory_summary(trace)
        self._merge_analysis_row(row, analysis, "analysis_passed" if include_workflow else "analysis_repairable")
        for round_id in (1, 2):
            repair = next((item for item in repairs if int(item.get("round_id") or 0) == round_id), None)
            review = next((item for item in reviews if int(item.get("round_id") or 0) == round_id), None)
            row.update(self._round_row(round_id, repair, review))
        return row

    def _ensure_metric_row(self, row: dict[str, Any]) -> dict[str, Any]:
        if (
            row.get("BLEU_diff") not in (None, "")
            and row.get("CrystalBLEU_diff") not in (None, "")
            and row.get("LEMOD") not in (None, "")
            and (not self.enable_llm_judge or row.get("LLM_as_judge") not in (None, ""))
        ):
            return row
        updated = dict(row)
        if (
            updated.get("BLEU_diff") in (None, "")
            or updated.get("CrystalBLEU_diff") in (None, "")
            or updated.get("LEMOD") in (None, "")
        ):
            updated.update(
                metric_result_to_row(
                    calculate_repair_metrics(
                        updated.get("original_code"),
                        updated.get("processed_manual_code"),
                        updated.get("processed_final_repaired_code"),
                        self._metric_context,
                    )
                )
            )
        if self.enable_llm_judge and updated.get("LLM_as_judge") in (None, ""):
            updated.update(
                self._judge_codes_to_row(
                    original_code=updated.get("original_code"),
                    manual_code=updated.get("processed_manual_code"),
                    candidate_code=updated.get("processed_final_repaired_code"),
                    satd_comment=updated.get("satd_comment"),
                    exact_match=str(updated.get("exact_match")).strip().lower() in {"true", "1", "yes"},
                )
            )
        return updated

    def _llm_judge_row(self, trace, row: dict[str, Any]) -> dict[str, Any]:
        if not self.enable_llm_judge:
            return llm_judge_result_to_row(None)
        return self._judge_codes_to_row(
            original_code=trace.original_code,
            manual_code=trace.processed_manual_code,
            candidate_code=trace.processed_final_repaired_code,
            satd_comment=trace.satd_comment,
            exact_match=bool(row.get("exact_match")),
        )

    def _judge_codes_to_row(
        self,
        original_code: str | None,
        manual_code: str | None,
        candidate_code: str | None,
        satd_comment: str | None,
        exact_match: bool = False,
    ) -> dict[str, Any]:
        if not self.enable_llm_judge or self.llm_judge is None:
            return llm_judge_result_to_row(None)
        if exact_match:
            return {"LLM_as_judge": 1.0}
        try:
            cache_key = (str(satd_comment or ""), str(candidate_code or ""))
            if cache_key in self._llm_judge_cache:
                return {"LLM_as_judge": self._llm_judge_cache[cache_key]}
            result = self.llm_judge.judge(original_code, manual_code, candidate_code, satd_comment)
            row = llm_judge_result_to_row(result)
            self._llm_judge_cache[cache_key] = row.get("LLM_as_judge", "")
            return row
        except Exception as exc:
            self._log(f"[llm_judge] failed task metric type={type(exc).__name__} message={exc}")
            return {"LLM_as_judge": ""}

    def _merge_analysis_row(self, row: dict[str, Any], analysis: dict[str, Any], repairable_field: str) -> None:
        row.update(
            {
                "analysis_decision": analysis.get("decision", ""),
                repairable_field: analysis.get("repairable", ""),
                "analysis_reason": analysis.get("reason", ""),
                "analysis_repair_plan": analysis.get("repair_plan", ""),
                "analysis_target_summary": analysis.get("target_summary", ""),
                "analysis_context_summary": analysis.get("context_summary", ""),
                "analysis_intent_type": analysis.get("intent_type", ""),
                "analysis_target_clarity": analysis.get("target_clarity", ""),
                "analysis_expected_edit_shape": analysis.get("expected_edit_shape", ""),
                "analysis_evidence_requirement": analysis.get("evidence_requirement", ""),
                "analysis_risk_note": analysis.get("risk_note", ""),
            }
        )

    def _round_row(self, round_id: int, repair: dict[str, Any] | None, review: dict[str, Any] | None) -> dict[str, Any]:
        prefix = f"round_{round_id}"
        repair = repair or {}
        review = review or {}
        return {
            f"{prefix}_candidate_mode": repair.get("candidate_mode", ""),
            f"{prefix}_repair_plan": repair.get("repair_plan", ""),
            f"{prefix}_repaired_code": repair.get("repaired_code", ""),
            f"{prefix}_changed_scope": repair.get("changed_scope", ""),
            f"{prefix}_fix_confidence": repair.get("confidence", ""),
            f"{prefix}_review_approved": review.get("approved", ""),
            f"{prefix}_review_gate_decision": review.get("gate_decision", ""),
            f"{prefix}_review_failure_modes": json.dumps(review.get("failure_modes") or [], ensure_ascii=False),
            f"{prefix}_review_score": review.get("review_score", ""),
            f"{prefix}_review_problem_alignment": review.get("problem_alignment", ""),
            f"{prefix}_review_minimality": review.get("minimality", ""),
            f"{prefix}_review_semantic_preservation": review.get("semantic_preservation", ""),
            f"{prefix}_review_internal_consistency": review.get("internal_consistency", ""),
            f"{prefix}_review_softened_gate_used": review.get("softened_gate_used", ""),
            f"{prefix}_review_reject_type": review.get("reject_type", ""),
            f"{prefix}_review_issues": json.dumps(review.get("issues") or [], ensure_ascii=False),
            f"{prefix}_review_failed_checks": json.dumps(review.get("failed_checks") or [], ensure_ascii=False),
            f"{prefix}_review_repair_constraints": json.dumps(review.get("repair_constraints") or [], ensure_ascii=False),
            f"{prefix}_review_failure_anchor": review.get("failure_anchor", ""),
            f"{prefix}_revision_advice": review.get("revision_advice", ""),
        }

    def _write_repairs_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "round_id", "candidate_mode", "repair_plan", "repaired_code", "changed_scope", "confidence", "notes"]
        rows = []
        for trace in traces:
            for repair in trace.repairs:
                rows.append(self._select_fields({"task_id": trace.task_id, **repair}, fieldnames))
        self._write_csv_rows(path, fieldnames, rows)

    def _append_repairs_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "round_id", "candidate_mode", "repair_plan", "repaired_code", "changed_scope", "confidence", "notes"]
        rows = []
        for trace in traces:
            for repair in trace.repairs:
                rows.append(self._select_fields({"task_id": trace.task_id, **repair}, fieldnames))
        self._append_csv_rows(path, fieldnames, rows)

    def _review_output_fields(self) -> list[str]:
        return ["task_id", "round_id", "candidate_mode", "approved", "gate_decision", "failure_modes", "review_score", "problem_alignment", "minimality", "semantic_preservation", "internal_consistency", "softened_gate_used", "reject_type", "rationale", "revision_advice", "issues", "failed_checks", "repair_constraints", "failure_anchor", "retry_hint"]

    def _write_reviews_csv(self, path: Path, traces: list) -> None:
        self._write_csv_rows(path, self._review_output_fields(), self._review_rows(traces))

    def _append_reviews_csv(self, path: Path, traces: list) -> None:
        self._append_csv_rows(path, self._review_output_fields(), self._review_rows(traces))

    def _review_rows(self, traces: list) -> list[dict[str, Any]]:
        fieldnames = self._review_output_fields()
        rows = []
        for trace in traces:
            for review in trace.reviews:
                row = dict(review)
                row["failure_modes"] = json.dumps(row.get("failure_modes", []), ensure_ascii=False)
                row["issues"] = json.dumps(row.get("issues", []), ensure_ascii=False)
                row["failed_checks"] = json.dumps(row.get("failed_checks", []), ensure_ascii=False)
                row["repair_constraints"] = json.dumps(row.get("repair_constraints", []), ensure_ascii=False)
                rows.append(self._select_fields({"task_id": trace.task_id, **row}, fieldnames))
        return rows

    def _write_github_context_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "repo_owner", "repo_name", "file_path", "commit", "context_commit", "historical_snapshot_mismatch", "github_evidence_strength", "snapshot_alignment_status", "repair_evidence_mode", "target_file_ok", "satd_window_found", "enclosing_symbol_found", "symbol_name", "satd_line", "related_tests_count", "call_sites_count", "commits_count", "similar_history_count", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count", "identified_method_count", "retrieved_method_count", "missing_method_count", "repository_evidence_count", "base_context_json", "repair_context_json", "review_context_json", "method_context_json", "repository_evidence_json"]
        self._write_csv_rows(path, fieldnames, [self._github_context_row(trace) for trace in traces])

    def _append_github_context_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "repo_owner", "repo_name", "file_path", "commit", "context_commit", "historical_snapshot_mismatch", "github_evidence_strength", "snapshot_alignment_status", "repair_evidence_mode", "target_file_ok", "satd_window_found", "enclosing_symbol_found", "symbol_name", "satd_line", "related_tests_count", "call_sites_count", "commits_count", "similar_history_count", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count", "identified_method_count", "retrieved_method_count", "missing_method_count", "repository_evidence_count", "base_context_json", "repair_context_json", "review_context_json", "method_context_json", "repository_evidence_json"]
        self._append_csv_rows(path, fieldnames, [self._github_context_row(trace) for trace in traces])

    def _github_context_row(self, trace) -> dict[str, Any]:
        context = trace.github_context or {}
        metadata = context.get("metadata") or {}
        base = context.get("base_context") or {}
        repair = context.get("repair_context") or {}
        review = context.get("review_context") or {}
        return {
            "task_id": trace.task_id,
            "repo_owner": trace.project and getattr(trace, "user", ""),
            "repo_name": trace.project,
            "file_path": trace.file_path,
            "commit": trace.commit,
            "context_commit": metadata.get("context_commit") or trace.commit,
            "historical_snapshot_mismatch": False,
            "github_evidence_strength": self._github_evidence_strength(trace),
            "snapshot_alignment_status": "not_checked",
            "repair_evidence_mode": self._repair_evidence_mode(trace),
            "target_file_ok": base.get("target_file_ok"),
            "satd_window_found": base.get("satd_window_found"),
            "enclosing_symbol_found": base.get("enclosing_symbol_found"),
            "symbol_name": base.get("symbol_name", ""),
            "satd_line": base.get("satd_line", ""),
            "related_tests_count": 0,
            "call_sites_count": 0,
            "commits_count": 0,
            "similar_history_count": 0,
            "retrieved_test_snippets_count": 0,
            "retrieved_callsite_snippets_count": 0,
            "retrieved_history_snippets_count": 0,
            "identified_method_count": len((trace.method_inquiry or {}).get("required_methods") or []),
            "retrieved_method_count": len([item for item in trace.retrieved_method_contexts or [] if item.get("found")]),
            "missing_method_count": len(trace.missing_method_names or []),
            "repository_evidence_count": len(trace.retrieved_repository_evidence or []),
            "base_context_json": json.dumps(base, ensure_ascii=False),
            "repair_context_json": json.dumps(repair, ensure_ascii=False),
            "review_context_json": json.dumps(review, ensure_ascii=False),
            "method_context_json": json.dumps(trace.retrieved_method_contexts or [], ensure_ascii=False),
            "repository_evidence_json": json.dumps(trace.retrieved_repository_evidence or [], ensure_ascii=False),
        }

    def _write_context_cache_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "commit", "context_commit", "cache_file", "base_cached", "base_cache_source", "base_context_fetched_at", "repair_cached", "repair_cache_source", "repair_context_fetched_at", "review_cached", "review_cache_source", "review_context_fetched_at", "historical_snapshot_mismatch", "github_evidence_strength", "snapshot_alignment_status", "repair_evidence_mode", "target_file_ok", "satd_window_found", "enclosing_symbol_found", "related_tests_count", "call_sites_count", "commits_count", "similar_history_count", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count", "identified_method_count", "retrieved_method_count", "missing_method_count", "repository_evidence_count"]
        self._write_csv_rows(path, fieldnames, [self._context_cache_row(trace) for trace in traces])

    def _append_context_cache_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "commit", "context_commit", "cache_file", "base_cached", "base_cache_source", "base_context_fetched_at", "repair_cached", "repair_cache_source", "repair_context_fetched_at", "review_cached", "review_cache_source", "review_context_fetched_at", "historical_snapshot_mismatch", "github_evidence_strength", "snapshot_alignment_status", "repair_evidence_mode", "target_file_ok", "satd_window_found", "enclosing_symbol_found", "related_tests_count", "call_sites_count", "commits_count", "similar_history_count", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count", "identified_method_count", "retrieved_method_count", "missing_method_count", "repository_evidence_count"]
        self._append_csv_rows(path, fieldnames, [self._context_cache_row(trace) for trace in traces])

    def _context_cache_row(self, trace) -> dict[str, Any]:
        context = trace.github_context or {}
        metadata = context.get("metadata") or {}
        cache_file = f"context_cache/{trace.task_id}.json"
        return {
            "task_id": trace.task_id,
            "commit": trace.commit,
            "context_commit": metadata.get("context_commit") or trace.commit,
            "cache_file": cache_file,
            "base_cached": True,
            "base_cache_source": "generated",
            "base_context_fetched_at": metadata.get("base_context_fetched_at", ""),
            "repair_cached": True,
            "repair_cache_source": "generated",
            "repair_context_fetched_at": metadata.get("repair_context_fetched_at", ""),
            "review_cached": bool(trace.reviews),
            "review_cache_source": "generated" if trace.reviews else "",
            "review_context_fetched_at": metadata.get("review_context_fetched_at", ""),
            "historical_snapshot_mismatch": False,
            "github_evidence_strength": self._github_evidence_strength(trace),
            "snapshot_alignment_status": "not_checked",
            "repair_evidence_mode": self._repair_evidence_mode(trace),
            "target_file_ok": True,
            "satd_window_found": True,
            "enclosing_symbol_found": "",
            "related_tests_count": 0,
            "call_sites_count": 0,
            "commits_count": 0,
            "similar_history_count": 0,
            "retrieved_test_snippets_count": 0,
            "retrieved_callsite_snippets_count": 0,
            "retrieved_history_snippets_count": 0,
            "identified_method_count": len((trace.method_inquiry or {}).get("required_methods") or []),
            "retrieved_method_count": len([item for item in trace.retrieved_method_contexts or [] if item.get("found")]),
            "missing_method_count": len(trace.missing_method_names or []),
            "repository_evidence_count": len(trace.retrieved_repository_evidence or []),
        }

    def _write_summary_csv(self, path: Path, summary: dict) -> None:
        fieldnames = ["input_path", "input_limit", "input_satd_count", "analyze_filtered_count", "analyzer_pass_count", "review_rejected_count", "workflow_output_count", "successful_repair_count", "policy_injected_count", "evidence_aware_count", "snippet_only_count", "historical_yes_success_count", "historical_no_success_count", "precision", "recall", "avg_BLEU_diff", "avg_CrystalBLEU_diff", "avg_LEMOD", "avg_LLM_as_judge", "llm_judge_count", "llm_judge_pass_count", "llm_judge_fail_count", "llm_judge_pass_rate", "agent_mode", "model", "max_rounds", "written_tasks", "write_batch_size", "use_analyzer", "use_reviewer", "analysis_only", "fixer_only", "repair_prompt_mode", "repair_response_format", "repair_context_mode", "max_method_contexts", "repository_evidence_mode", "max_repository_evidence", "repository_evidence_prompt_mode", "repository_evidence_guidance_mode", "repository_evidence_rerank_mode", "policy_candidate_strategy", "policy_injection_strategy", "single_repair_path", "method_inquiry_enabled", "context_router_enabled", "force_route", "llm_judge_enabled", "judge_model"]
        try:
            self._write_csv_rows(path, fieldnames, [summary])
        except PermissionError:
            fallback = path.with_name(f"{path.stem}.fallback{path.suffix}")
            self._write_csv_rows(fallback, fieldnames, [summary])

    def _llm_judge_summary_stats(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        judged_values: list[float] = []
        for row in rows:
            if row.get("status") != "accepted":
                continue
            value = row.get("LLM_as_judge")
            if value in (None, ""):
                continue
            try:
                judged_values.append(float(value))
            except (TypeError, ValueError):
                continue
        pass_count = sum(1 for value in judged_values if value >= 0.5)
        fail_count = len(judged_values) - pass_count
        pass_rate = round(pass_count / len(judged_values), 6) if judged_values else 0.0
        return {
            "llm_judge_count": len(judged_values),
            "llm_judge_pass_count": pass_count,
            "llm_judge_fail_count": fail_count,
            "llm_judge_pass_rate": pass_rate,
        }

    def _write_task_progress_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "status", "rounds_used", "exact_match", "progress_index", "progress_total"]
        self._write_csv_rows(path, fieldnames, [])

    def _append_task_progress_csv(self, path: Path, traces: list, progress_index: int, progress_total: int) -> None:
        fieldnames = ["task_id", "status", "rounds_used", "exact_match", "progress_index", "progress_total"]
        rows = [
            {
                "task_id": trace.task_id,
                "status": trace.status,
                "rounds_used": trace.rounds_used,
                "exact_match": trace.exact_match,
                "progress_index": progress_index,
                "progress_total": progress_total,
            }
            for trace in traces
        ]
        self._append_csv_rows(path, fieldnames, rows)

    def _write_csv_rows(self, path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(self._select_fields(row, fieldnames))

    def _append_csv_rows(self, path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            for row in rows:
                writer.writerow(self._select_fields(row, fieldnames))

    def _select_fields(self, row: dict[str, Any], fieldnames: list[str]) -> dict[str, Any]:
        return {field: row.get(field, "") for field in fieldnames}

    def _read_csv_rows(self, path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        csv.field_size_limit(1024 * 1024 * 1024)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def _load_or_build_base_context(self, state: GraphState) -> dict[str, Any]:
        cached = state.get("github_context")
        if isinstance(cached, dict) and cached:
            return cached
        metadata = {
            "context_commit": state.get("commit") or "",
            "base_context_fetched_at": self.context_client._timestamp(),
            "repair_context_fetched_at": "",
            "review_context_fetched_at": "",
        }
        base_context = {
            "target_file_ok": True,
            "satd_window_found": True,
            "enclosing_symbol_found": None,
            "symbol_name": "",
            "satd_line": "",
        }
        bundle = {"metadata": metadata, "base_context": base_context, "repair_context": {}, "review_context": {}}
        self._persist_context_cache(state["task_id"], bundle)
        return bundle

    def _attach_method_context(
        self,
        context_bundle: dict[str, Any] | None,
        *,
        method_inquiry: MethodInquiryResult,
        retrieved_method_contexts: list[RetrievedMethodContext],
        missing_method_names: list[str],
    ) -> dict[str, Any]:
        bundle = dict(context_bundle or {})
        metadata = dict(bundle.get("metadata") or {})
        metadata.setdefault("context_commit", "")
        metadata["repair_context_fetched_at"] = self.context_client._timestamp()
        bundle["metadata"] = metadata
        existing_repair_context = dict(bundle.get("repair_context") or {})
        bundle["repair_context"] = {
            **existing_repair_context,
            "context_strategy": self.repair_context_mode,
            "method_inquiry": {
                "required_methods": list(method_inquiry.required_methods),
                "reason": method_inquiry.reason,
                "method_notes": list(method_inquiry.method_notes or []),
            },
            "retrieved_method_contexts": [self._serialize_method_context(item) for item in retrieved_method_contexts],
            "missing_method_names": list(missing_method_names),
        }
        return bundle

    def _attach_repository_evidence_for_repair(self, state: GraphState) -> GraphState:
        existing = list(state.get("retrieved_repository_evidence") or [])
        if existing or self.repository_evidence_mode == "none" or self.max_repository_evidence <= 0:
            return state
        if self.repository_evidence_mode == "policy":
            return self._attach_policy_selected_repository_evidence(state)
        if not self._should_retrieve_repository_evidence(state):
            return state

        candidate_pool_limit = self._repository_candidate_pool_limit()
        first_limit = min(2, self.max_repository_evidence)
        first_pass = self.repository_evidence_retriever.retrieve(
            owner=state["user"],
            repo=state["project"],
            current_path=state["file_path"],
            satd_comment=state["satd_comment"],
            original_code=state["original_code"],
            ref=state["commit"],
            max_items=first_limit if self.repository_evidence_mode == "two_stage" else candidate_pool_limit,
        )
        retrieval_rounds = 1
        evidence = first_pass
        if self.repository_evidence_mode == "two_stage" and self._should_expand_repository_evidence(state, first_pass):
            evidence = self.repository_evidence_retriever.retrieve(
                owner=state["user"],
                repo=state["project"],
                current_path=state["file_path"],
                satd_comment=state["satd_comment"],
                original_code=state["original_code"],
                ref=state["commit"],
                max_items=candidate_pool_limit,
            )
            retrieval_rounds = 2

        evidence = self._rerank_repository_evidence(state, evidence)

        context_bundle = self._attach_repository_evidence(
            state.get("github_context") or self._load_or_build_base_context(state),
            evidence=evidence,
            retrieval_rounds=retrieval_rounds,
        )
        guidance = self._build_repository_evidence_guidance(state, evidence)
        return {
            **state,
            "github_context": context_bundle,
            "retrieved_repository_evidence": evidence,
            "repository_evidence_guidance": guidance,
        }

    def _should_retrieve_repository_evidence(self, state: GraphState) -> bool:
        if self.repository_evidence_mode == "fixed":
            return True
        if str(state.get("satd_route_type") or "") != "context_required":
            return False
        if self.repository_evidence_mode not in {"adaptive", "two_stage"}:
            return False
        analysis = state.get("analysis")
        requirement = str(getattr(analysis, "evidence_requirement", "") or "").strip()
        return requirement not in {"local_cleanup_ok", "documentation_only"}

    def _attach_policy_selected_repository_evidence(self, state: GraphState) -> GraphState:
        retrieval_decision = self.context_policy_agent.decide_retrieval(state)
        if not retrieval_decision.use_repository_context:
            context_bundle = self._attach_context_policy_decision(
                state.get("github_context") or self._load_or_build_base_context(state),
                retrieval_decision,
            )
            return {
                **state,
                "github_context": context_bundle,
                "context_policy_decision": retrieval_decision,
                "method_inquiry": MethodInquiryResult(reason="context_policy_snippet_only"),
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "retrieved_repository_evidence": [],
                "repository_evidence_guidance": "",
            }

        candidate_pool_limit = max(self.max_repository_evidence, min(10, max(6, self.max_repository_evidence * 2)))
        candidates = self.repository_evidence_retriever.retrieve(
            owner=state["user"],
            repo=state["project"],
            current_path=state["file_path"],
            satd_comment=state["satd_comment"],
            original_code=state["original_code"],
            ref=state["commit"],
            max_items=candidate_pool_limit,
        )
        useful_candidates = self._policy_candidate_evidence(candidates, retrieval_decision)
        if self.policy_injection_strategy == "top":
            selected_evidence = useful_candidates[: self.max_repository_evidence]
            injection_decision = replace(
                retrieval_decision,
                inject_evidence=bool(selected_evidence),
                selected_evidence_indices=list(range(1, len(selected_evidence) + 1)),
                candidate_evidence_count=len(useful_candidates),
                selected_evidence_count=len(selected_evidence),
                injection_decision_reason=(
                    "policy injected top direct/supporting evidence after LLM chose evidence-aware mode"
                    if selected_evidence
                    else "policy found no direct/supporting evidence after LLM chose evidence-aware mode"
                ),
            )
        else:
            injection_decision = self.context_policy_agent.decide_injection(
                state,
                retrieval_decision,
                useful_candidates,
                max_items=self.max_repository_evidence,
            )
            selected_evidence = self._select_policy_evidence(useful_candidates, injection_decision)
        context_bundle = self._attach_repository_evidence(
            state.get("github_context") or self._load_or_build_base_context(state),
            evidence=selected_evidence,
            retrieval_rounds=1,
        )
        context_bundle = self._attach_context_policy_decision(context_bundle, injection_decision)
        guidance = self._build_repository_evidence_guidance(state, selected_evidence)
        method_inquiry = (
            MethodInquiryResult(reason="context_policy_repository_evidence_only")
            if selected_evidence
            else MethodInquiryResult(reason="context_policy_no_injected_evidence")
        )
        return {
            **state,
            "github_context": context_bundle,
            "context_policy_decision": injection_decision,
            "method_inquiry": method_inquiry,
            "retrieved_method_contexts": [],
            "missing_method_names": [],
            "retrieved_repository_evidence": selected_evidence,
            "repository_evidence_guidance": guidance,
        }

    def _policy_candidate_evidence(
        self,
        candidates: list[Any],
        decision: ContextPolicyDecision,
    ) -> list[Any]:
        useful = [
            item
            for item in candidates
            if str(getattr(item, "support_level", "")).strip().lower() in {"direct", "supporting"}
        ]
        needed = set(decision.needed_evidence_types or [])
        if not needed:
            return useful
        typed = [item for item in useful if getattr(item, "evidence_type", "") in needed]
        return typed or useful

    def _select_policy_evidence(
        self,
        candidates: list[Any],
        decision: ContextPolicyDecision,
    ) -> list[Any]:
        if not decision.inject_evidence:
            return []
        selected: list[Any] = []
        for index in decision.selected_evidence_indices:
            if 1 <= index <= len(candidates):
                selected.append(candidates[index - 1])
            if len(selected) >= self.max_repository_evidence:
                break
        return selected

    def _attach_context_policy_decision(
        self,
        context_bundle: dict[str, Any] | None,
        decision: ContextPolicyDecision,
    ) -> dict[str, Any]:
        bundle = dict(context_bundle or {})
        repair_context = dict(bundle.get("repair_context") or {})
        repair_context["context_policy_decision"] = {
            "use_repository_context": decision.use_repository_context,
            "needed_evidence_types": list(decision.needed_evidence_types),
            "context_risk": decision.context_risk,
            "inject_evidence": decision.inject_evidence,
            "selected_evidence_indices": list(decision.selected_evidence_indices),
            "decision_reason": decision.decision_reason,
            "retrieval_decision_reason": decision.retrieval_decision_reason,
            "injection_decision_reason": decision.injection_decision_reason,
            "candidate_evidence_count": decision.candidate_evidence_count,
            "selected_evidence_count": decision.selected_evidence_count,
        }
        bundle["repair_context"] = repair_context
        return bundle

    def _should_expand_repository_evidence(self, state: GraphState, evidence: list[Any]) -> bool:
        if self.repository_evidence_rerank_mode == "llm":
            return len(evidence) < self._repository_candidate_pool_limit()
        if len(evidence) < min(2, self.max_repository_evidence):
            return True
        has_useful = any(str(getattr(item, "support_level", "")) in {"direct", "supporting"} for item in evidence)
        if not has_useful:
            return True
        decision = state.get("context_decision")
        return bool(decision and decision.blocking_unknowns)

    def _attach_repository_evidence(
        self,
        context_bundle: dict[str, Any] | None,
        *,
        evidence: list[Any],
        retrieval_rounds: int,
    ) -> dict[str, Any]:
        bundle = dict(context_bundle or {})
        metadata = dict(bundle.get("metadata") or {})
        metadata.setdefault("context_commit", "")
        metadata["repair_context_fetched_at"] = self.context_client._timestamp()
        bundle["metadata"] = metadata
        repair_context = dict(bundle.get("repair_context") or {})
        repair_context["repository_evidence_policy"] = {
            "mode": self.repository_evidence_mode,
            "prompt_mode": self.repository_evidence_prompt_mode,
            "max_items": self.max_repository_evidence,
            "retrieval_rounds": retrieval_rounds,
            "rerank_mode": self.repository_evidence_rerank_mode,
        }
        repair_context["retrieved_repository_evidence"] = [
            {
                "evidence_type": item.evidence_type,
                "evidence_subtype": item.evidence_subtype,
                "support_level": item.support_level,
                "source_path": item.source_path,
                "span": item.span,
                "content": item.content,
                "retrieval_method": item.retrieval_method,
                "query_origin": item.query_origin,
                "score": item.score,
                "why_relevant": item.why_relevant,
                "query": item.query,
            }
            for item in evidence
        ]
        bundle["repair_context"] = repair_context
        return bundle

    def _repository_candidate_pool_limit(self) -> int:
        if self.repository_evidence_rerank_mode == "none":
            return self.max_repository_evidence
        return max(self.max_repository_evidence, min(10, max(6, self.max_repository_evidence * 2)))

    def _rerank_repository_evidence(self, state: GraphState, evidence: list[Any]) -> list[Any]:
        if self.repository_evidence_rerank_mode != "llm":
            return evidence[: self.max_repository_evidence]
        if len(evidence) <= self.max_repository_evidence:
            return evidence

        snippets: list[str] = []
        for index, item in enumerate(evidence, start=1):
            snippets.append(
                f"[{index}] type={item.evidence_type}/{item.evidence_subtype} "
                f"support={item.support_level} score={item.score:.2f} "
                f"path={item.source_path}:{item.span}\n"
                f"why={item.why_relevant}\n"
                f"{str(item.content or '').strip()[:1200]}"
            )
        system_prompt = (
            "You select repository evidence for a SATD repair agent. "
            "Prefer snippets that can directly determine the edit, not snippets that are merely topically related. "
            "Return JSON only."
        )
        user_prompt = (
            "Choose the repository evidence most likely to help repair the target code exactly. "
            "Prefer patch-driving evidence such as sibling implementations, project usage proving required arguments, "
            "definitions of replacement APIs, tests proving expected behavior, or conventions that constrain the repair. "
            "Avoid self-evidence, generic documentation, and duplicates.\n\n"
            f"### SATD comment:\n{state['satd_comment']}\n\n"
            f"### Target code:\n```python\n{state['original_code']}\n```\n\n"
            f"### Candidate evidence:\n\n{chr(10).join(snippets)}\n\n"
            f"Return JSON with keys:\n"
            f'- "selected_indices": list of up to {self.max_repository_evidence} 1-based candidate indices in best-first order,\n'
            '- "reason": one short sentence explaining what kind of evidence mattered most.\n'
        )
        try:
            payload = self.context_client.generate_json(
                system_prompt,
                user_prompt,
                temperature=0.0,
                request_label=f"repository_rerank:task_{state['task_id']}",
                max_tokens=300,
            )
        except Exception as exc:
            self._log(f"{self._task_label(state)} repository rerank failed reason={type(exc).__name__}")
            return evidence[: self.max_repository_evidence]

        selected: list[Any] = []
        seen: set[int] = set()
        for value in payload.get("selected_indices") or []:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if index < 1 or index > len(evidence) or index in seen:
                continue
            selected.append(evidence[index - 1])
            seen.add(index)
            if len(selected) >= self.max_repository_evidence:
                break
        if not selected:
            return evidence[: self.max_repository_evidence]
        return selected

    def _build_repository_evidence_guidance(self, state: GraphState, evidence: list[Any]) -> str:
        if self.repository_evidence_guidance_mode != "summarize" or not evidence:
            return ""
        snippets = []
        for index, item in enumerate(evidence[:5], start=1):
            snippets.append(
                f"[{index}] {item.evidence_type}/{item.evidence_subtype} {item.source_path}:{item.span}\n"
                f"{str(item.content or '').strip()[:1800]}"
            )
        system_prompt = (
            "You summarize repository evidence for a SATD repair agent. "
            "Use only the evidence shown. Return JSON only."
        )
        user_prompt = (
            "Given the SATD, target code, and repository evidence, extract only concrete repair guidance that is directly "
            "supported by the evidence. Prefer actionable edit steps over broad summaries. When sibling implementations "
            "or usage examples are present, compare them against the target code and identify missing parameters, missing "
            "assignments, missing calls, or replacement patterns that the sibling proves. Do not write repaired code.\n\n"
            f"### SATD comment:\n{state['satd_comment']}\n\n"
            f"### Target code:\n```python\n{state['original_code']}\n```\n\n"
            f"### Repository evidence:\n\n{chr(10).join(snippets)}\n\n"
            "Return JSON with keys:\n"
            '- "summary": one short sentence about the strongest evidence,\n'
            '- "edit_steps": list of concrete evidence-backed edits the fixer should consider, including signature changes when sibling or usage evidence demonstrates them,\n'
            '- "must_use": list of concrete symbols, arguments, or operations supported by evidence,\n'
            '- "must_preserve": list of important existing behaviors supported by evidence,\n'
            '- "avoid": list of unsupported edits the fixer should avoid.\n'
        )
        try:
            payload = self.context_client.generate_json(
                system_prompt,
                user_prompt,
                temperature=0.0,
                request_label=f"repository_guidance:task_{state['task_id']}",
                max_tokens=600,
            )
        except Exception as exc:
            self._log(f"{self._task_label(state)} repository guidance failed reason={type(exc).__name__}")
            return ""
        lines: list[str] = []
        summary = " ".join(str(payload.get("summary") or "").split())
        if summary:
            lines.append(f"- summary: {summary}")
        for key, label in (
            ("edit_steps", "evidence-backed edit steps"),
            ("must_use", "must use"),
            ("must_preserve", "must preserve"),
            ("avoid", "avoid"),
        ):
            values = payload.get(key)
            if not isinstance(values, list):
                continue
            cleaned = [" ".join(str(item).split()) for item in values if " ".join(str(item).split())]
            if cleaned:
                lines.append(f"- {label}: " + " | ".join(cleaned[:4]))
        return "\n".join(lines)

    def _serialize_method_context(self, item: RetrievedMethodContext) -> dict[str, Any]:
        return {
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

    def _persist_context_cache(self, task_id: str, bundle: dict[str, Any]) -> None:
        if self._current_output_dir is None:
            return
        path = self._context_cache_dir() / f"{task_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(bundle, handle, ensure_ascii=False, indent=2)

    def _context_cache_dir(self) -> Path:
        assert self._current_output_dir is not None
        return self._current_output_dir / "context_cache"

    def _record_repair_debug_checkpoint(self, state: GraphState, stage: str, payload: dict[str, Any]) -> None:
        if self._current_output_dir is None:
            return
        task_dir = self._current_output_dir / "repair_debug" / str(state.get("task_id", "unknown"))
        task_dir.mkdir(parents=True, exist_ok=True)
        round_id = payload.get("round_id") or int(state.get("round_id") or 0) + 1
        candidate = payload.get("candidate_mode") or state.get("candidate_mode") or ""
        safe_candidate = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(candidate or "candidate"))
        path = task_dir / f"round_{round_id}_{safe_candidate}_{stage}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _fallback_analysis(self, state: GraphState, reason: str) -> AnalysisResult:
        return self.analyzer.fallback_analysis(state, reason)

    def _bypass_analysis(self, state: GraphState, satd_route_type: str, reason: str) -> AnalysisResult:
        return self.analyzer.fallback_analysis(state, reason)

    def _fallback_repair(self, state: GraphState, round_id: int, reason: str = "fixer_fallback") -> RepairAttempt:
        return RepairAttempt(
            round_id=round_id,
            repair_plan=f"Fallback no-op repair because the fixer request failed ({reason}).",
            repaired_code=state["original_code"],
            changed_scope="none",
            confidence=0.0,
            notes=reason,
        )

    def _fallback_review(self, state: GraphState, reason: str = "review_fallback") -> ReviewResult:
        candidate_mode = getattr(state.get("latest_repair"), "candidate_mode", "single") if isinstance(state, dict) else "single"
        return ReviewResult(
            round_id=state["round_id"],
            approved=False,
            issues=[f"Reviewer fallback triggered: {reason}"],
            candidate_mode=candidate_mode,
            gate_decision="reject",
            failure_modes=["invalid_or_noop"],
            review_score=0.0,
            problem_alignment=0.0,
            minimality=0.0,
            semantic_preservation=0.0,
            internal_consistency=0.0,
            revision_advice=f"constraints:make_smallest_clear_local_edit; reason:{reason}",
            reject_type="reviewer_uncertain",
            rationale=f"reviewer fallback: {reason}",
            failed_checks=["reviewer_uncertain"],
            repair_constraints=["make_smallest_clear_local_edit"],
            failure_anchor="reviewer_uncertain",
            retry_hint=f"Retry because reviewer failed: {reason}.",
        )

    def _build_repair_feedback(self, review: ReviewResult) -> dict | None:
        repair_constraints = []
        for item in getattr(review, "repair_constraints", None) or []:
            cleaned = str(item).strip()
            if cleaned and cleaned not in repair_constraints:
                repair_constraints.append(cleaned)
            if len(repair_constraints) >= 4:
                break
        retry_hint = " ".join(str(getattr(review, "retry_hint", "") or "").split())
        if not repair_constraints and not retry_hint:
            repair_constraints = ["make_smallest_local_edit"]
            retry_hint = "Make the smallest local edit that directly addresses the SATD comment."
        feedback = {"repair_constraints": repair_constraints, "retry_hint": retry_hint or None}
        feedback["can_retry"] = bool(feedback["repair_constraints"] or feedback["retry_hint"])
        return feedback if feedback["can_retry"] else None

    def _should_retry_after_review(self, state: GraphState) -> bool:
        latest_review = state.get("latest_review")
        if latest_review is None or latest_review.approved:
            return False
        if latest_review.gate_decision != "retry":
            return False
        if state["round_id"] >= min(state["max_rounds"], 2):
            return False
        if self._repair_failed_due_to_infrastructure(state.get("latest_repair")):
            return False
        return True

    def _repair_failed_due_to_infrastructure(self, repair: RepairAttempt | None) -> bool:
        if repair is None:
            return False
        haystack = " ".join([str(getattr(repair, "repair_plan", "") or ""), str(getattr(repair, "notes", "") or ""), str(getattr(repair, "changed_scope", "") or "")]).lower()
        if "fallback no-op repair" not in haystack and "fixer_exception" not in haystack:
            return False
        return any(marker in haystack for marker in ("apitimeouterror", "apiconnectionerror", "timeout", "connection", "rate limit", "ratelimit"))

    def _repair_attempt_uses_method_evidence(self, repair: RepairAttempt | None, retrieved_method_contexts: list[RetrievedMethodContext]) -> bool:
        if repair is None or not retrieved_method_contexts:
            return False
        return not str(repair.candidate_mode or "").strip().lower().endswith("no_context")

    def _repair_attempt_uses_any_evidence(
        self,
        repair: RepairAttempt | None,
        retrieved_method_contexts: list[RetrievedMethodContext],
        state: GraphState,
    ) -> bool:
        if self._repair_attempt_uses_method_evidence(repair, retrieved_method_contexts):
            return True
        if repair is None or str(repair.candidate_mode or "").strip().lower().endswith("no_context"):
            return False
        return bool(state.get("retrieved_repository_evidence"))

    def _repair_evidence_mode(self, trace) -> str:
        has_method = bool(trace.retrieved_method_contexts)
        has_repository = bool(trace.retrieved_repository_evidence)
        if has_method and has_repository:
            return "method_plus_repository_context"
        if has_repository:
            return "repository_context"
        return "method_context" if has_method else "snippet_only"

    def _github_evidence_strength(self, trace) -> str:
        has_method = bool(trace.retrieved_method_contexts)
        has_repository = bool(trace.retrieved_repository_evidence)
        if has_method and has_repository:
            return "method_plus_repository_context"
        if has_repository:
            return "repository_context_only"
        return "method_context_only" if has_method else "low"

    def _drop_stage(self, trace) -> str:
        if trace.status == "dropped_after_review":
            return f"review_round_{trace.rounds_used or 1}"
        return ""

    def _trajectory_summary(self, trace) -> str:
        if trace.status == "accepted":
            return "accepted"
        if trace.status == "dropped_after_review":
            return "dropped after review"
        return trace.status

    def _reset_output_dir_for_fresh_run(self, output_dir: Path) -> None:
        if output_dir.exists():
            for name in [
                "trajectory_overview.csv", "results.csv", "repairs.csv", "reviews.csv", "github_context.csv",
                "context_cache.csv", "summary.csv", "task_progress.csv",
            ]:
                path = output_dir / name
                if path.exists():
                    path.unlink()
            for dirname in ["context_cache", "repair_debug"]:
                path = output_dir / dirname
                if path.exists():
                    shutil.rmtree(path)

    def _acquire_run_lock(self, output_dir: Path) -> Path:
        lock_path = output_dir / ".run.lock"
        lock_path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
        return lock_path

    def _release_run_lock(self, lock_path: Path) -> None:
        try:
            if lock_path.exists():
                lock_path.unlink()
        except OSError:
            pass

    def _clamp_float(self, value: object, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, number))

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _task_label(self, state: GraphState | dict) -> str:
        task_id = state.get("task_id", "?")
        task_index = state.get("task_index") or 0
        task_total = state.get("task_total") or 0
        if task_index and task_total:
            return f"[task {task_id} {task_index}/{task_total}]"
        return f"[task {task_id}]"

    def _compact_satd_comment(self, comment: str, limit: int = 80) -> str:
        text = re.sub(r"\s+", " ", (comment or "").strip())
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _normalized_code_equal(self, left: str, right: str) -> bool:
        return preprocess_python_code(left or "") == preprocess_python_code(right or "")

    def _compact_unified_diff(self, original: str, candidate: str, limit: int = 5000) -> str:
        original_lines = str(original or "").splitlines()
        candidate_lines = str(candidate or "").splitlines()
        diff = "\n".join(
            difflib.unified_diff(
                original_lines,
                candidate_lines,
                fromfile="original",
                tofile="candidate",
                lineterm="",
                n=3,
            )
        )
        if not diff:
            diff = "[no textual change]"
        if len(diff) > limit:
            return diff[:limit] + "\n...[truncated]"
        return diff

    def _selector_evidence_summary(self, evidence: list) -> str:
        if not evidence:
            return "[none]"
        lines: list[str] = []
        for index, item in enumerate(evidence[:5], start=1):
            evidence_type = getattr(item, "evidence_type", "")
            subtype = getattr(item, "evidence_subtype", "")
            support = getattr(item, "support_level", "")
            source_path = getattr(item, "source_path", "")
            span = getattr(item, "span", "")
            why = " ".join(str(getattr(item, "why_relevant", "") or "").split())
            content = " ".join(str(getattr(item, "content", "") or "").split())
            if len(content) > 800:
                content = content[:800] + "..."
            lines.append(
                f"[{index}] type={evidence_type}/{subtype} support={support} source={source_path}:{span}\n"
                f"why={why or '[none]'}\n"
                f"content={content or '[empty]'}"
            )
        return "\n\n".join(lines)
