from __future__ import annotations

import csv
import json
import os
import re
import shutil
import traceback
from pathlib import Path
from typing import Any

from .analyzer_agent import OpenAIAnalyzer
from .fixer_agent import OpenAIFixer
from .openai_client import OpenAICompatClient
from .reviewer_agent import OpenAIReviewer
from .csv_loader import load_satd_csv
from .schema import (
    AnalysisResult,
    ContextNeedDecision,
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
from .tools import GitHubDownloadTool, MethodContextTool, MethodRetrievalTool, SimilarCodeRules


class LangGraphSATDWorkflow:
    def __init__(
        self,
        max_rounds: int = 2,
        model: str = "gpt-4o-mini",
        verbose: bool = False,
        write_batch_size: int = 10,
        repair_context_mode: str = "clone_treesitter",
        max_method_contexts: int = 2,
    ) -> None:
        self.max_rounds = max(2, int(max_rounds))
        self.model = model
        self.verbose = verbose
        self.write_batch_size = write_batch_size
        self.repair_context_mode = repair_context_mode
        self.max_method_contexts = max(1, int(max_method_contexts))

        client = OpenAICompatClient(model=model, verbose=verbose)
        self.context_client = client
        self.github_downloader = GitHubDownloadTool()
        self.similar_code_rules = SimilarCodeRules()
        self.method_retriever = MethodRetrievalTool(
            self.github_downloader,
            self.similar_code_rules,
            logger=self._log,
        )
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
            logger=self._log,
            checkpoint_callback=self._record_repair_debug_checkpoint,
        )
        self.reviewer = OpenAIReviewer(client)
        self._current_output_dir: Path | None = None

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

        if context_decision.route == "no_context":
            state["analysis"] = self._bypass_analysis(state, "no_context", "context_router_no_context | analyzer_skipped")
            state["status"] = "repairable"
        else:
            state = self._run_analyzer_stage(state)

        if not state["analysis"] or not state["analysis"].repairable:
            state["status"] = "dropped_by_analyzer"
            self._log(f"{self._task_label(state)} end status={state['status']}")
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

        rule_drop = self._rule_based_analyzer_drop(state)
        if rule_drop is not None:
            analysis = self.analyzer.build_rule_drop_analysis(notes=rule_drop["notes"])
        else:
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
                self._log(f"{self._task_label(state)} analyze failed reason={reason}; using fallback drop")
                analysis = self._fallback_analysis(reason=reason)

        updated = {
            **state,
            "analysis": analysis,
            "github_context": context_bundle,
            "method_inquiry": method_inquiry,
            "uncertainty_items": uncertainty_items,
            "edit_constraints": edit_constraints,
            "retrieved_method_contexts": retrieved_method_contexts,
            "missing_method_names": missing_method_names,
            "status": "repairable" if analysis.repairable else "dropped_by_analyzer",
        }
        self._log(
            f"{self._task_label(updated)} analyze done decision={analysis.decision} "
            f"repairable={analysis.repairable} score={analysis.repairability_score:.2f}"
        )
        return updated

    def _run_repair_review_loop(self, state: GraphState) -> GraphState:
        while state["round_id"] < min(state["max_rounds"], self.max_rounds):
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
                "repair_context_used": self._repair_attempt_uses_method_evidence(repair, contexts),
                "round_id": repair.round_id,
                "latest_repair": repair,
                "status": "repairing",
            }

            review = self._run_review(state)
            repair_feedback = None if review.approved else self._build_repair_feedback(review)
            state = {
                **state,
                "repair_feedback": repair_feedback,
                "review_strict_gate_result": "approved" if review.approved else "rejected",
                "latest_review": review,
                "repairs": [*state["repairs"], repair],
                "reviews": [*state["reviews"], review],
                "status": "accepted" if review.approved else "review_failed",
                "final_repaired_code": repair.repaired_code if review.approved else state["final_repaired_code"],
            }
            self._log(
                f"{self._task_label(state)} review done round={review.round_id} "
                f"approved={review.approved} reject_type={review.reject_type or ''}"
            )
            if review.approved:
                return {**state, "status": "accepted", "repair_feedback": None}
            if not self._should_retry_after_review(state):
                return {**state, "status": "dropped_after_review"}
        return {**state, "status": "dropped_after_review"}

    def _run_repair(
        self,
        state: GraphState,
        round_id: int,
        candidate_mode: str,
    ) -> tuple[RepairAttempt, MethodInquiryResult, list[RetrievedMethodContext], list[str], list, list]:
        repair_state = {**state, "candidate_mode": candidate_mode}
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
                "Choose \"no_context\" when the snippet alone makes a local edit obvious, especially removing obsolete or temporary code, "
                "updating documentation, adding simple type annotations, or making a clearly specified local rename/literal/default replacement.\n\n"
                "Choose \"context_required\" only if reliable repair depends on external method behavior, missing symbols, "
                "repository conventions, an external or unclear replacement API, broad refactor/optimization intent, or an unclear target/end state.\n\n"
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

    def run_csv(self, input_path: Path, output_dir: Path, limit: int | None = None, resume: bool = False) -> dict:
        records = load_satd_csv(input_path, limit=limit)
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
                    summary = self._with_summary_metadata(self._summarize_from_existing_and_new(existing_rows, new_traces))
                    if resume and completed_ids:
                        self._append_outputs(output_dir, pending_flush_traces)
                    else:
                        self._write_outputs(output_dir, new_traces, summary)
                    self._write_summary_csv(output_dir / "summary.csv", summary)
                    pending_flush_traces = []
                    if self.verbose:
                        print(f"[flush] wrote {len(completed_ids) + len(new_traces)}/{total} tasks to {output_dir}")

            if not pending_records:
                summary = self._with_summary_metadata(self._summarize_from_existing_and_new(existing_rows, []))
                self._write_summary_csv(output_dir / "summary.csv", summary)

            assert summary is not None
            return summary
        finally:
            self._current_output_dir = None
            self._release_run_lock(lock_path)

    def _with_summary_metadata(self, summary: dict[str, Any]) -> dict[str, Any]:
        summary["agent_mode"] = "openai"
        summary["model"] = self.model
        summary["max_rounds"] = self.max_rounds
        summary["written_tasks"] = summary.get("input_satd_count", 0)
        summary["write_batch_size"] = self.write_batch_size
        summary["use_analyzer"] = True
        summary["use_reviewer"] = True
        summary["analysis_only"] = False
        summary["repair_prompt_mode"] = "lightweight"
        summary["repair_context_mode"] = self.repair_context_mode
        summary["max_method_contexts"] = self.max_method_contexts
        summary["single_repair_path"] = True
        summary["method_inquiry_enabled"] = self.repair_context_mode in {"method_query", "clone_treesitter"}
        summary["context_router_enabled"] = True
        return summary

    def _empty_existing_rows(self) -> dict[str, list[dict[str, str]]]:
        return {"results": [], "trajectory": []}

    def _load_existing_rows(self, output_dir: Path) -> dict[str, list[dict[str, str]]]:
        return {
            "results": self._read_csv_rows(output_dir / "results.csv"),
            "trajectory": self._read_csv_rows(output_dir / "trajectory_overview.csv"),
        }

    def _summarize_from_existing_and_new(self, existing_rows: dict[str, list[dict[str, str]]], new_traces: list) -> dict:
        result_rows = [*existing_rows.get("results", []), *[self._results_row(trace) for trace in new_traces]]
        total = len(result_rows)
        analyze_filtered_count = sum(1 for row in result_rows if row.get("status") == "dropped_by_analyzer")
        review_rejected_count = sum(1 for row in result_rows if row.get("status") == "dropped_after_review")
        workflow_output_count = sum(1 for row in result_rows if row.get("status") == "accepted")
        successful_repair_count = sum(1 for row in result_rows if str(row.get("exact_match")).lower() == "true")
        precision = round(successful_repair_count / workflow_output_count, 4) if workflow_output_count else 0.0
        recall = round(successful_repair_count / total, 4) if total else 0.0
        return {
            "input_satd_count": total,
            "analyze_filtered_count": analyze_filtered_count,
            "analyzer_pass_count": 0,
            "review_rejected_count": review_rejected_count,
            "workflow_output_count": workflow_output_count,
            "successful_repair_count": successful_repair_count,
            "precision": precision,
            "recall": recall,
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
            *self._analysis_output_fields("analysis_passed"),
            "repair_evidence_mode", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count",
            "identified_method_names", "retrieved_method_names", "missing_method_names", "retrieved_method_count", "method_context_json", "uncertainty_items_json", "edit_constraints_json",
            "repair_context_used", "review_strict_gate_result", "original_code", "processed_manual_code", "processed_final_repaired_code",
            *self._round_output_fields(1),
            *self._round_output_fields(2),
        ]

    def _results_fieldnames(self) -> list[str]:
        return [
            "task_id", "project", "file_path", "commit", "context_commit", "satd_comment", "status", "rounds_used", "em_label", "exact_match", "satd_route_type", "context_route", "context_required", "context_confidence", "context_reason", "context_blocking_unknowns",
            *self._analysis_output_fields("analysis_repairable"),
            "repair_evidence_mode", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count",
            "identified_method_names", "retrieved_method_names", "missing_method_names", "retrieved_method_count", "method_context_json", "uncertainty_items_json", "edit_constraints_json",
            "repair_context_used", "review_strict_gate_result", "original_code", "processed_manual_code", "processed_final_repaired_code",
            *self._round_output_fields(1),
            *self._round_output_fields(2),
        ]

    def _analysis_output_fields(self, repairable_field: str) -> list[str]:
        return [
            "analysis_decision", repairable_field, "analysis_repairability_score", "analysis_confidence", "analysis_satd_type", "analysis_risk_level", "analysis_scope_radius",
            "analysis_intent_clarity", "analysis_change_locality", "analysis_semantic_risk", "analysis_context_sufficiency", "analysis_verifiability", "analysis_analyze_score",
            "analysis_context_score", "analysis_clarity_score", "analysis_validation_signals", "analysis_context_gaps", "analysis_followup_context_requests", "analysis_evidence_summary",
            "analysis_repair_strategy", "analysis_operation_concrete", "analysis_localizable", "analysis_local_scope", "analysis_end_state_clear", "analysis_comment_evidence", "analysis_code_evidence", "analysis_historical_snapshot_mismatch", "analysis_github_evidence_strength", "analysis_snapshot_alignment_status",
        ]

    def _round_output_fields(self, round_id: int) -> list[str]:
        prefix = f"round_{round_id}"
        return [
            f"{prefix}_candidate_mode", f"{prefix}_repair_plan", f"{prefix}_repaired_code", f"{prefix}_changed_scope", f"{prefix}_fix_confidence",
            f"{prefix}_review_approved", f"{prefix}_review_score", f"{prefix}_review_problem_alignment", f"{prefix}_review_minimality", f"{prefix}_review_semantic_preservation", f"{prefix}_review_internal_consistency", f"{prefix}_review_softened_gate_used", f"{prefix}_review_reject_type", f"{prefix}_review_issues", f"{prefix}_review_failed_checks", f"{prefix}_review_repair_constraints", f"{prefix}_review_failure_anchor", f"{prefix}_revision_advice",
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
            "uncertainty_items_json": json.dumps(trace.uncertainty_items or [], ensure_ascii=False),
            "edit_constraints_json": json.dumps(trace.edit_constraints or [], ensure_ascii=False),
            "repair_context_used": trace.repair_context_used,
            "review_strict_gate_result": trace.review_strict_gate_result,
            "original_code": trace.original_code,
            "processed_manual_code": trace.processed_manual_code,
            "processed_final_repaired_code": trace.processed_final_repaired_code or "",
        }
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

    def _merge_analysis_row(self, row: dict[str, Any], analysis: dict[str, Any], repairable_field: str) -> None:
        row.update(
            {
                "analysis_decision": analysis.get("decision", ""),
                repairable_field: analysis.get("repairable", ""),
                "analysis_repairability_score": analysis.get("repairability_score", ""),
                "analysis_confidence": analysis.get("confidence", ""),
                "analysis_satd_type": analysis.get("satd_type", ""),
                "analysis_risk_level": analysis.get("risk_level", ""),
                "analysis_scope_radius": analysis.get("scope_radius", ""),
                "analysis_intent_clarity": analysis.get("intent_clarity", ""),
                "analysis_change_locality": analysis.get("change_locality", ""),
                "analysis_semantic_risk": analysis.get("semantic_risk", ""),
                "analysis_context_sufficiency": analysis.get("context_sufficiency", ""),
                "analysis_verifiability": analysis.get("verifiability", ""),
                "analysis_analyze_score": analysis.get("analyze_score", ""),
                "analysis_context_score": analysis.get("context_score", ""),
                "analysis_clarity_score": analysis.get("clarity_score", ""),
                "analysis_validation_signals": " | ".join(analysis.get("validation_signals") or []),
                "analysis_context_gaps": " | ".join(analysis.get("context_gaps") or []),
                "analysis_followup_context_requests": " | ".join(analysis.get("followup_context_requests") or []),
                "analysis_evidence_summary": analysis.get("evidence_summary", "") or analysis.get("reason", ""),
                "analysis_repair_strategy": analysis.get("repair_strategy", ""),
                "analysis_operation_concrete": analysis.get("operation_concrete", ""),
                "analysis_localizable": analysis.get("localizable", ""),
                "analysis_local_scope": analysis.get("local_scope", ""),
                "analysis_end_state_clear": analysis.get("end_state_clear", ""),
                "analysis_comment_evidence": analysis.get("comment_evidence", ""),
                "analysis_code_evidence": analysis.get("code_evidence", ""),
                "analysis_historical_snapshot_mismatch": analysis.get("historical_snapshot_mismatch", ""),
                "analysis_github_evidence_strength": analysis.get("github_evidence_strength", ""),
                "analysis_snapshot_alignment_status": analysis.get("snapshot_alignment_status", ""),
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
        return ["task_id", "round_id", "candidate_mode", "approved", "review_score", "problem_alignment", "minimality", "semantic_preservation", "internal_consistency", "softened_gate_used", "reject_type", "rationale", "revision_advice", "issues", "failed_checks", "repair_constraints", "failure_anchor", "retry_hint"]

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
                row["issues"] = json.dumps(row.get("issues", []), ensure_ascii=False)
                row["failed_checks"] = json.dumps(row.get("failed_checks", []), ensure_ascii=False)
                row["repair_constraints"] = json.dumps(row.get("repair_constraints", []), ensure_ascii=False)
                rows.append(self._select_fields({"task_id": trace.task_id, **row}, fieldnames))
        return rows

    def _write_github_context_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "repo_owner", "repo_name", "file_path", "commit", "context_commit", "historical_snapshot_mismatch", "github_evidence_strength", "snapshot_alignment_status", "repair_evidence_mode", "target_file_ok", "satd_window_found", "enclosing_symbol_found", "symbol_name", "satd_line", "related_tests_count", "call_sites_count", "commits_count", "similar_history_count", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count", "identified_method_count", "retrieved_method_count", "missing_method_count", "base_context_json", "repair_context_json", "review_context_json", "method_context_json"]
        self._write_csv_rows(path, fieldnames, [self._github_context_row(trace) for trace in traces])

    def _append_github_context_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "repo_owner", "repo_name", "file_path", "commit", "context_commit", "historical_snapshot_mismatch", "github_evidence_strength", "snapshot_alignment_status", "repair_evidence_mode", "target_file_ok", "satd_window_found", "enclosing_symbol_found", "symbol_name", "satd_line", "related_tests_count", "call_sites_count", "commits_count", "similar_history_count", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count", "identified_method_count", "retrieved_method_count", "missing_method_count", "base_context_json", "repair_context_json", "review_context_json", "method_context_json"]
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
            "github_evidence_strength": "method_context_only" if trace.retrieved_method_contexts else "low",
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
            "base_context_json": json.dumps(base, ensure_ascii=False),
            "repair_context_json": json.dumps(repair, ensure_ascii=False),
            "review_context_json": json.dumps(review, ensure_ascii=False),
            "method_context_json": json.dumps(trace.retrieved_method_contexts or [], ensure_ascii=False),
        }

    def _write_context_cache_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "commit", "context_commit", "cache_file", "base_cached", "base_cache_source", "base_context_fetched_at", "repair_cached", "repair_cache_source", "repair_context_fetched_at", "review_cached", "review_cache_source", "review_context_fetched_at", "historical_snapshot_mismatch", "github_evidence_strength", "snapshot_alignment_status", "repair_evidence_mode", "target_file_ok", "satd_window_found", "enclosing_symbol_found", "related_tests_count", "call_sites_count", "commits_count", "similar_history_count", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count", "identified_method_count", "retrieved_method_count", "missing_method_count"]
        self._write_csv_rows(path, fieldnames, [self._context_cache_row(trace) for trace in traces])

    def _append_context_cache_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "commit", "context_commit", "cache_file", "base_cached", "base_cache_source", "base_context_fetched_at", "repair_cached", "repair_cache_source", "repair_context_fetched_at", "review_cached", "review_cache_source", "review_context_fetched_at", "historical_snapshot_mismatch", "github_evidence_strength", "snapshot_alignment_status", "repair_evidence_mode", "target_file_ok", "satd_window_found", "enclosing_symbol_found", "related_tests_count", "call_sites_count", "commits_count", "similar_history_count", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count", "identified_method_count", "retrieved_method_count", "missing_method_count"]
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
            "github_evidence_strength": "method_context_only" if trace.retrieved_method_contexts else "low",
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
        }

    def _write_summary_csv(self, path: Path, summary: dict) -> None:
        fieldnames = ["input_satd_count", "analyze_filtered_count", "analyzer_pass_count", "review_rejected_count", "workflow_output_count", "successful_repair_count", "precision", "recall", "agent_mode", "model", "max_rounds", "written_tasks", "write_batch_size", "use_analyzer", "use_reviewer", "analysis_only", "repair_prompt_mode", "repair_context_mode", "max_method_contexts", "single_repair_path", "method_inquiry_enabled", "context_router_enabled"]
        self._write_csv_rows(path, fieldnames, [summary])

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
        bundle["repair_context"] = {
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

    def _fallback_analysis(self, reason: str) -> AnalysisResult:
        return AnalysisResult(
            decision="drop",
            repairable=False,
            confidence=0.0,
            reason=reason,
            repairability_score=0.0,
            intent_clarity=0.0,
            change_locality=0.0,
            semantic_risk=1.0,
            context_sufficiency=0.0,
            verifiability=0.0,
            analyze_score=0.0,
            satd_type="context_required",
            evidence_summary=reason,
            risk_level="high",
            context_score=0.0,
            clarity_score=0.0,
            scope_radius="unknown",
            operation_concrete="low",
            localizable="low",
            local_scope="low",
            end_state_clear="low",
            code_evidence="fallback",
            validation_signals=[reason],
            repair_strategy="Do not attempt automatic repair.",
            github_evidence_strength="low",
        )

    def _bypass_analysis(self, state: GraphState, satd_route_type: str, reason: str) -> AnalysisResult:
        return AnalysisResult(
            decision="pass",
            repairable=True,
            confidence=0.60,
            reason=reason,
            repairability_score=0.60,
            intent_clarity=0.70,
            change_locality=0.70,
            semantic_risk=0.30,
            context_sufficiency=0.60,
            verifiability=0.60,
            analyze_score=0.60,
            satd_type=satd_route_type,
            evidence_summary=reason,
            risk_level="medium",
            context_score=0.60,
            clarity_score=0.70,
            scope_radius="snippet",
            operation_concrete="partial",
            localizable="partial",
            local_scope="partial",
            end_state_clear="partial",
            comment_evidence=state.get("satd_comment", ""),
            code_evidence="snippet-only route",
            validation_signals=["context_router_no_context", "analyzer_skipped"],
            repair_strategy="Pass directly to fixer for snippet-only repair.",
            github_evidence_strength="none",
        )

    def _rule_based_analyzer_drop(self, state: GraphState) -> dict | None:
        comment = (state.get("satd_comment") or "").strip()
        code = (state.get("original_code") or "").strip()
        if not comment:
            return {"notes": "SATD comment is empty, so there is no concrete repair request."}
        if not code:
            return {"notes": "Code snippet is empty, so the repair target cannot be localized."}
        return None

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

    def _repair_evidence_mode(self, trace) -> str:
        return "method_context" if trace.retrieved_method_contexts else "snippet_only"

    def _drop_stage(self, trace) -> str:
        if trace.status == "dropped_by_analyzer":
            return "analyzer"
        if trace.status == "dropped_after_review":
            return f"review_round_{trace.rounds_used or 1}"
        return ""

    def _trajectory_summary(self, trace) -> str:
        if trace.status == "accepted":
            return "accepted"
        if trace.status == "dropped_by_analyzer":
            return "dropped by analyzer"
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
