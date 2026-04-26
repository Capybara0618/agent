from __future__ import annotations

import csv
import json
import os
import re
import shutil
import traceback
from pathlib import Path

from .bootstrap import bootstrap_vendor

bootstrap_vendor()

from langgraph.graph import END, START, StateGraph

from .agents import OpenAIAnalyzer, OpenAICompatClient, OpenAIFixer, OpenAIReviewer, OpenAISelector
from .csv_loader import load_satd_csv
from .schema import (
    AnalysisResult,
    GraphState,
    MethodInquiryResult,
    RepairAttempt,
    RetrievedMethodContext,
    ReviewResult,
    SATDRecord,
    SelectorDecision,
    preprocess_python_code,
    record_to_graph_input,
    trace_from_state,
)


class LangGraphSATDWorkflow:
    def __init__(
        self,
        max_rounds: int = 2,
        model: str = "gpt-4o-mini",
        verbose: bool = False,
        write_batch_size: int = 10,
        use_analyzer: bool = True,
        use_reviewer: bool = False,
        analysis_only: bool = False,
        use_selector: bool = False,
        dual_repair_candidates: bool = False,
        repair_context_mode: str = "clone_treesitter",
        max_method_contexts: int = 2,
    ) -> None:
        self.max_rounds = max_rounds
        self.model = model
        self.verbose = verbose
        self.write_batch_size = write_batch_size
        self.use_analyzer = use_analyzer
        self.use_reviewer = use_reviewer
        self.analysis_only = analysis_only
        self.use_selector = use_selector
        self.dual_repair_candidates = dual_repair_candidates
        self.repair_context_mode = repair_context_mode
        self.max_method_contexts = max(1, int(max_method_contexts))
        client = OpenAICompatClient(model=model, verbose=verbose)
        self.context_client = client
        self.context_client.toolbox.logger = self._log
        self.analyzer = OpenAIAnalyzer(client)
        self.fixer = OpenAIFixer(
            client,
            repair_context_mode=repair_context_mode,
            max_method_contexts=self.max_method_contexts,
            logger=self._log,
            checkpoint_callback=self._record_repair_debug_checkpoint,
        )
        self.selector = OpenAISelector(client)
        self.reviewer = OpenAIReviewer(client)
        self.graph = self._build_graph()
        self._current_output_dir: Path | None = None

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("analyze", self._analyze_node)
        graph.add_node("accept", self._accept_node)
        graph.add_node("drop", self._drop_node)
        graph.add_edge(START, "analyze")
        if self.analysis_only:
            graph.add_conditional_edges("analyze", self._route_after_analysis, {"accept": "accept", "drop": "drop"})
        else:
            graph.add_node("repair", self._repair_node)
            graph.add_node("select", self._select_node)
            graph.add_node("review", self._review_node)
            graph.add_conditional_edges("analyze", self._route_after_analysis, {"repair": "repair", "drop": "drop"})
            graph.add_edge("repair", "select")
            graph.add_edge("select", "review")
            graph.add_conditional_edges("review", self._route_after_review, {"accept": "accept", "repair": "repair", "drop": "drop"})
        graph.add_edge("accept", END)
        graph.add_edge("drop", END)
        return graph.compile()

    def _analyze_node(self, state: GraphState) -> dict:
        self._log(f"{self._task_label(state)} analyze start")
        satd_route_type = self._infer_satd_route_type(state)
        context_bundle = state.get("github_context")
        method_inquiry = MethodInquiryResult(reason="analyzer_method_context_not_run")
        retrieved_method_contexts: list[RetrievedMethodContext] = []
        missing_method_names: list[str] = []
        uncertainty_items = []
        edit_constraints = []
        method_context_block = "[none]"
        if satd_route_type == "generic":
            rule_drop = self._rule_based_analyzer_drop(state)
            if rule_drop is None:
                context_bundle = self._load_or_build_shared_context(state)
        else:
            rule_drop = None
        if self.use_analyzer:
            if satd_route_type != "generic":
                analysis = self.analyzer.build_easy_route_analysis(satd_route_type)
            elif rule_drop is not None:
                analysis = self.analyzer.build_rule_drop_analysis(notes=rule_drop["notes"])
            else:
                try:
                    analyzer_state = {**state, "satd_route_type": satd_route_type, "github_context": context_bundle}
                    try:
                        (
                            method_inquiry,
                            retrieved_method_contexts,
                            missing_method_names,
                            uncertainty_items,
                            edit_constraints,
                            method_context_block,
                        ) = self.fixer.prepare_method_context(analyzer_state, candidate_mode="baseline_context")
                        context_bundle = self._attach_method_context(
                            context_bundle,
                            method_inquiry=method_inquiry,
                            retrieved_method_contexts=retrieved_method_contexts,
                            missing_method_names=missing_method_names,
                        )
                    except Exception as context_exc:
                        self._log(
                            f"{self._task_label(state)} analyzer method-context exception "
                            f"type={type(context_exc).__name__}; continuing without method context"
                        )
                        method_inquiry = MethodInquiryResult(reason=f"analyzer_method_context_exception:{type(context_exc).__name__}")
                        retrieved_method_contexts = []
                        missing_method_names = []
                        uncertainty_items = []
                        edit_constraints = []
                        method_context_block = "[none]"
                    analyzer_state = {
                        **analyzer_state,
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
                        self._log(f"{self._task_label(state)} analyze content-filtered; using fallback drop")
                        analysis = self._fallback_analysis(reason="analyzer_content_filter")
                    else:
                        self._log(f"{self._task_label(state)} analyze exception type={type(exc).__name__}; using fallback drop")
                        analysis = self._fallback_analysis(reason=f"analyzer_exception:{type(exc).__name__}")
        else:
            analysis = self._bypass_analysis(state, satd_route_type)
        self._log(
            f"{self._task_label(state)} analyze done decision={analysis.decision} "
            f"repairable={analysis.repairable} score={analysis.repairability_score:.2f}"
        )
        return {
            "analysis": analysis,
            "satd_route_type": satd_route_type,
            "github_context": context_bundle,
            "method_inquiry": method_inquiry,
            "uncertainty_items": uncertainty_items,
            "edit_constraints": edit_constraints,
            "retrieved_method_contexts": retrieved_method_contexts,
            "missing_method_names": missing_method_names,
            "status": "repairable" if analysis.repairable else "dropped_by_analyzer",
        }

    def _repair_node(self, state: GraphState) -> dict:
        next_round = state["round_id"] + 1
        self._log(f"{self._task_label(state)} repair start round={next_round}")
        context_bundle = state.get("github_context") or self._load_or_build_base_context(state)
        satd_route_type = self._infer_satd_route_type(state)
        candidates, method_inquiry, retrieved_method_contexts, missing_method_names, uncertainty_items, edit_constraints = self._run_repair_candidates(
            state,
            context_bundle,
            next_round,
            satd_route_type,
        )
        provisional = self._select_candidate_without_selector(candidates)
        context_bundle = self._attach_method_context(
            context_bundle,
            method_inquiry=method_inquiry,
            retrieved_method_contexts=retrieved_method_contexts,
            missing_method_names=missing_method_names,
        )
        self._log(
            f"{self._task_label(state)} repair done round={provisional.round_id} "
            f"candidate={provisional.candidate_mode} scope={provisional.changed_scope} conf={provisional.confidence:.2f}"
        )
        return {
            "github_context": context_bundle,
            "satd_route_type": satd_route_type,
            "method_inquiry": method_inquiry,
            "uncertainty_items": uncertainty_items,
            "edit_constraints": edit_constraints,
            "retrieved_method_contexts": retrieved_method_contexts,
            "missing_method_names": missing_method_names,
            "repair_candidates": candidates,
            "candidate_repairs": [*state["candidate_repairs"], *candidates],
            "repair_context_used": self._repair_attempt_uses_method_evidence(
                satd_route_type,
                provisional,
                retrieved_method_contexts,
            ),
            "repair_feedback": state.get("repair_feedback"),
            "status": "repairing",
            "round_id": provisional.round_id,
            "latest_repair": provisional,
        }

    def _select_node(self, state: GraphState) -> dict:
        self._log(f"{self._task_label(state)} select start round={state['round_id']}")
        candidates = state.get("repair_candidates") or ([state["latest_repair"]] if state.get("latest_repair") else [])
        decision = self._run_selector(state, candidates)
        selected_repair = self._select_repair_from_decision(candidates, decision)
        self._log(
            f"{self._task_label(state)} select done round={decision.round_id} "
            f"route={decision.satd_route_type} candidate={selected_repair.candidate_mode} conf={decision.confidence:.2f}"
        )
        return {
            "repair_context_used": self._repair_attempt_uses_method_evidence(
                state.get("satd_route_type"),
                selected_repair,
                state.get("retrieved_method_contexts") or [],
            ),
            "latest_repair": selected_repair,
            "selector_decisions": [*state["selector_decisions"], decision],
            "status": "selected",
        }

    def _review_node(self, state: GraphState) -> dict:
        self._log(f"{self._task_label(state)} review start round={state['round_id']}")
        assert state["analysis"] is not None
        assert state["latest_repair"] is not None
        context_bundle = self._ensure_review_context_cache(state)
        if self.use_reviewer:
            try:
                review = self.reviewer.run({**state, "github_context": context_bundle})
            except Exception as exc:
                if self.context_client._is_content_filter_error(exc):
                    self._log(f"{self._task_label(state)} review content-filtered; using fallback reject")
                    review = self._fallback_review(state, reason="review_content_filter")
                else:
                    self._log(f"{self._task_label(state)} review exception type={type(exc).__name__}; using fallback reject")
                    review = self._fallback_review(state, reason=f"review_exception:{type(exc).__name__}")
        else:
            review = self._bypass_review(state)
        self._log(
            f"{self._task_label(state)} review done round={review.round_id} "
            f"candidate={state['latest_repair'].candidate_mode} approved={review.approved} score={review.review_score:.2f}"
        )
        repair_feedback = None if review.approved else self._build_repair_feedback(review)
        return {
            "github_context": context_bundle,
            "repair_candidates": [],
            "candidate_reviews": [*state["candidate_reviews"], review],
            "repair_feedback": repair_feedback,
            "review_strict_gate_result": "approved" if review.approved else "rejected",
            "status": "accepted" if review.approved else "review_failed",
            "repair_context_used": self._repair_attempt_uses_method_evidence(
                state.get("satd_route_type"),
                state.get("latest_repair"),
                state.get("retrieved_method_contexts") or [],
            ),
            "latest_review": review,
            "repairs": [*state["repairs"], state["latest_repair"]],
            "reviews": [*state["reviews"], review],
            "final_repaired_code": state["latest_repair"].repaired_code if review.approved else state["final_repaired_code"],
        }

    def _accept_node(self, state: GraphState) -> dict:
        if self.analysis_only:
            self._log(f"{self._task_label(state)} analyzer pass")
            return {
                "status": "passed_by_analyzer",
                "repair_candidates": [],
                "repair_feedback": None,
                "review_strict_gate_result": None,
                "final_repaired_code": None,
            }
        self._log(f"{self._task_label(state)} accepted after rounds={state['round_id']}")
        return {
            "status": "accepted",
            "repair_candidates": [],
            "repair_feedback": None,
            "review_strict_gate_result": state.get("review_strict_gate_result") or "approved",
            "final_repaired_code": state["latest_repair"].repaired_code if state["latest_repair"] else None,
        }

    def _drop_node(self, state: GraphState) -> dict:
        if state["analysis"] and not state["analysis"].repairable:
            self._log(
                f"{self._task_label(state)} dropped by analyzer "
                f"decision={state['analysis'].decision} notes={state['analysis'].evidence_summary}"
            )
            return {"status": "dropped_by_analyzer", "repair_candidates": [], "repair_feedback": None}
        self._log(f"{self._task_label(state)} dropped after review rounds={state['round_id']}")
        return {
            "status": "dropped_after_review",
            "repair_candidates": [],
            "repair_feedback": state.get("repair_feedback"),
            "review_strict_gate_result": state.get("review_strict_gate_result") or "rejected",
        }

    def _route_after_analysis(self, state: GraphState) -> str:
        analysis = state["analysis"]
        if analysis is None or not analysis.repairable:
            return "drop"
        if self.analysis_only:
            return "accept"
        return "repair"

    def _route_after_review(self, state: GraphState) -> str:
        latest_review = state["latest_review"]
        if latest_review is None:
            return "drop"
        if latest_review.approved:
            return "accept"
        if self._should_retry_after_review(state):
            return "repair"
        return "drop"

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
        haystack = " ".join(
            [
                str(getattr(repair, "repair_plan", "") or ""),
                str(getattr(repair, "notes", "") or ""),
                str(getattr(repair, "changed_scope", "") or ""),
            ]
        ).lower()
        if "fallback no-op repair" not in haystack and "fixer_exception" not in haystack:
            return False
        infrastructure_markers = (
            "apitimeouterror",
            "apiconnectionerror",
            "timeout",
            "connection",
            "rate limit",
            "ratelimit",
        )
        return any(marker in haystack for marker in infrastructure_markers)

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
        feedback = {
            "repair_constraints": repair_constraints,
            "retry_hint": retry_hint or None,
        }
        feedback["can_retry"] = bool(feedback["repair_constraints"] or feedback["retry_hint"])
        return feedback if feedback["can_retry"] else None

    def _run_repair_candidates(
        self,
        state: GraphState,
        context_bundle: dict,
        round_id: int,
        satd_route_type: str,
    ) -> tuple[list[RepairAttempt], MethodInquiryResult, list[RetrievedMethodContext], list[str], list, list]:
        candidates: list[RepairAttempt] = []
        candidate_modes = self._candidate_modes_for_route(satd_route_type)
        method_inquiry = MethodInquiryResult()
        retrieved_method_contexts: list[RetrievedMethodContext] = []
        missing_method_names: list[str] = []
        uncertainty_items = []
        edit_constraints = []
        for candidate_mode in candidate_modes:
            candidate_state = {
                **state,
                "satd_route_type": satd_route_type,
                "candidate_mode": candidate_mode,
                "github_context": context_bundle,
            }
            try:
                repair, inquiry, contexts, missing, uncertainty_items, edit_constraints = self.fixer.run(candidate_state, candidate_mode=candidate_mode)
                method_inquiry = inquiry
                retrieved_method_contexts = contexts
                missing_method_names = missing
            except Exception as exc:
                error_message = " ".join(str(exc).split())
                error_traceback = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__, limit=8)
                )
                self.fixer._checkpoint(
                    candidate_state,
                    stage="generation_error",
                    payload={
                        "round_id": int(candidate_state.get("round_id", 0)) + 1,
                        "candidate_mode": candidate_mode,
                        "error_type": type(exc).__name__,
                        "error_message": error_message[:500],
                        "error_traceback": error_traceback[-4000:],
                    },
                )
                retry_mode = "baseline_no_context"
                can_retry_without_context = (
                    candidate_mode != retry_mode
                    and retry_mode not in candidate_modes
                )
                if can_retry_without_context:
                    retry_state = {**candidate_state, "candidate_mode": retry_mode}
                    self._log(
                        f"{self._task_label(state)} repair candidate={candidate_mode} "
                        f"exception type={type(exc).__name__}; retrying without method context"
                    )
                    try:
                        repair, inquiry, contexts, missing, uncertainty_items, edit_constraints = self.fixer.run(
                            retry_state,
                            candidate_mode=retry_mode,
                        )
                        repair.candidate_mode = retry_mode
                        repair.notes = (
                            f"{repair.notes} | recovered_from_context_candidate_error:{type(exc).__name__}"
                        ).strip()
                        method_inquiry = inquiry
                        retrieved_method_contexts = contexts
                        missing_method_names = missing
                        candidates.append(repair)
                        continue
                    except Exception as retry_exc:
                        retry_error_message = " ".join(str(retry_exc).split())
                        retry_error_traceback = "".join(
                            traceback.format_exception(
                                type(retry_exc),
                                retry_exc,
                                retry_exc.__traceback__,
                                limit=8,
                            )
                        )
                        self.fixer._checkpoint(
                            retry_state,
                            stage="generation_retry_error",
                            payload={
                                "round_id": int(retry_state.get("round_id", 0)) + 1,
                                "candidate_mode": retry_mode,
                                "source_candidate_mode": candidate_mode,
                                "source_error_type": type(exc).__name__,
                                "error_type": type(retry_exc).__name__,
                                "error_message": retry_error_message[:500],
                                "error_traceback": retry_error_traceback[-4000:],
                            },
                        )
                        self._log(
                            f"{self._task_label(state)} repair no-context retry exception "
                            f"type={type(retry_exc).__name__} message={retry_error_message[:200]}; "
                            "using fallback no-op repair"
                        )
                if self.context_client._is_content_filter_error(exc):
                    self._log(
                        f"{self._task_label(state)} repair candidate={candidate_mode} "
                        f"content-filtered; using fallback no-op repair"
                    )
                    repair = self._fallback_repair(state, round_id, reason=f"fixer_content_filter:{candidate_mode}")
                else:
                    self._log(
                        f"{self._task_label(state)} repair candidate={candidate_mode} "
                        f"exception type={type(exc).__name__} message={error_message[:200]}; using fallback no-op repair"
                    )
                    repair = self._fallback_repair(state, round_id, reason=f"fixer_exception:{candidate_mode}:{type(exc).__name__}")
                repair.candidate_mode = candidate_mode
            candidates.append(repair)
        return candidates, method_inquiry, retrieved_method_contexts, missing_method_names, uncertainty_items, edit_constraints

    def _run_selector(self, state: GraphState, candidates: list[RepairAttempt]) -> SelectorDecision:
        if not candidates:
            return SelectorDecision(
                round_id=state["round_id"] + 1,
                satd_route_type=str(state.get("satd_route_type") or "generic"),
                selected_candidate_mode="",
                selected_index=0,
                confidence=0.0,
                rationale="No candidates available.",
                candidate_scores=[],
            )
        if not self.use_selector:
            selected = self._select_candidate_without_selector(candidates)
            index = max(0, next((i for i, item in enumerate(candidates) if item is selected), 0))
            return SelectorDecision(
                round_id=selected.round_id,
                satd_route_type=str(state.get("satd_route_type") or "generic"),
                selected_candidate_mode=selected.candidate_mode,
                selected_index=index,
                confidence=max(0.45, selected.confidence),
                rationale="Selector disabled; using workflow fallback candidate ordering.",
                candidate_scores=[{"index": i, "candidate_mode": item.candidate_mode, "score": item.confidence} for i, item in enumerate(candidates)],
            )
        try:
            return self.selector.run({**state, "repair_candidates": candidates})
        except Exception as exc:
            if self.context_client._is_content_filter_error(exc):
                self._log(f"{self._task_label(state)} selector content-filtered; using fallback selection")
            else:
                self._log(f"{self._task_label(state)} selector exception type={type(exc).__name__}; using fallback selection")
            selected = self._select_candidate_without_selector(candidates)
            index = max(0, next((i for i, item in enumerate(candidates) if item is selected), 0))
            return SelectorDecision(
                round_id=selected.round_id,
                satd_route_type=str(state.get("satd_route_type") or "generic"),
                selected_candidate_mode=selected.candidate_mode,
                selected_index=index,
                confidence=max(0.45, selected.confidence),
                rationale=f"Fallback selector used because selector request failed ({type(exc).__name__}).",
                candidate_scores=[{"index": i, "candidate_mode": item.candidate_mode, "score": item.confidence} for i, item in enumerate(candidates)],
            )

    def _select_repair_from_decision(self, candidates: list[RepairAttempt], decision: SelectorDecision) -> RepairAttempt:
        if not candidates:
            raise ValueError("repair candidates must not be empty")
        index = decision.selected_index
        if index < 0 or index >= len(candidates):
            index = 0
        return candidates[index]

    def _select_candidate_without_selector(self, candidates: list[RepairAttempt]) -> RepairAttempt:
        if not candidates:
            raise ValueError("repair candidates must not be empty")
        return max(
            candidates,
            key=lambda candidate: (
                candidate.confidence,
                1 if candidate.candidate_mode.endswith("no_context") else 0,
            ),
        )

    def _repair_attempt_uses_method_evidence(
        self,
        satd_route_type: str | None,
        repair: RepairAttempt | None,
        retrieved_method_contexts: list[RetrievedMethodContext],
    ) -> bool:
        if repair is None or not retrieved_method_contexts:
            return False
        return not str(repair.candidate_mode or "").strip().lower().endswith("no_context")

    def _infer_satd_route_type(self, state: GraphState) -> str:
        comment = (state.get("satd_comment") or "").strip().lower()
        if not comment:
            return "generic"
        if any(token in comment for token in ("pyre-fixme", "return type", "parameter must be annotated", "annotation", "annotated")):
            return "type_annotation"
        if any(token in comment for token in ("replace by", "switch to", "deprecated", "rename", "full_path")):
            return "replace_symbol"
        if self._matches_remove_temporary_route(comment):
            return "remove_temporary"
        if any(
            token in comment
            for token in (
                "document",
                "documentation",
                "docstring",
                "docs",
                "comment this",
                "commented explanation",
                "update description",
                "add description",
                "fix description",
            )
        ):
            return "document"
        return "generic"

    def _matches_remove_temporary_route(self, comment: str) -> bool:
        if any(token in comment for token in ("remove", "delete", "revert")):
            return True
        if any(token in comment for token in ("temporary", "hack", "workaround", "obsolete", "drop this")):
            return True
        return False

    def _candidate_modes_for_route(self, satd_route_type: str) -> list[str]:
        if self._dual_candidate_enabled():
            return ["baseline_no_context", "baseline_context"]
        return ["baseline_context"]

    def _dual_candidate_enabled(self) -> bool:
        return bool(getattr(self, "dual_repair_candidates", False) or getattr(self, "use_selector", False))

    def run_record(self, record: SATDRecord, task_index: int = 0, task_total: int = 0):
        initial_state = record_to_graph_input(record, self.max_rounds)
        initial_state["task_index"] = task_index
        initial_state["task_total"] = task_total
        self._log(
            f"{self._task_label(initial_state)} start project={record.project} file={record.file_path} "
            f"commit={(record.commit or '')[:12]} satd={self._compact_satd_comment(record.satd_comment)}"
        )
        final_state = self.graph.invoke(initial_state)
        self._log(f"{self._task_label(final_state)} end status={final_state['status']}")
        return trace_from_state(final_state, record.em_label)

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
                    summary = self._summarize_from_existing_and_new(existing_rows, new_traces)
                    summary["agent_mode"] = "openai_analyzer_only" if self.analysis_only else "openai"
                    summary["model"] = self.model
                    summary["max_rounds"] = self.max_rounds
                    summary["written_tasks"] = len(completed_ids) + len(new_traces)
                    summary["write_batch_size"] = self.write_batch_size
                    summary["use_analyzer"] = self.use_analyzer
                    summary["use_reviewer"] = self.use_reviewer
                    summary["analysis_only"] = self.analysis_only
                    summary["use_selector"] = self.use_selector
                    summary["repair_prompt_mode"] = "lightweight"
                    summary["dual_repair_candidates"] = self.dual_repair_candidates
                    summary["repair_context_mode"] = self.repair_context_mode
                    summary["max_method_contexts"] = self.max_method_contexts
                    summary["single_repair_path"] = not self._dual_candidate_enabled()
                    summary["method_inquiry_enabled"] = self.repair_context_mode in {"method_query", "clone_treesitter"}
                    if resume and completed_ids:
                        self._append_outputs(output_dir, pending_flush_traces)
                        self._write_summary_csv(output_dir / "summary.csv", summary)
                    else:
                        self._write_outputs(output_dir, new_traces, summary)
                    pending_flush_traces = []
                    if self.verbose:
                        print(f"[flush] wrote {len(completed_ids) + len(new_traces)}/{total} tasks to {output_dir}")

            if not pending_records:
                summary = self._summarize_from_existing_and_new(existing_rows, [])
                summary["agent_mode"] = "openai_analyzer_only" if self.analysis_only else "openai"
                summary["model"] = self.model
                summary["max_rounds"] = self.max_rounds
                summary["written_tasks"] = len(completed_ids)
                summary["write_batch_size"] = self.write_batch_size
                summary["use_analyzer"] = self.use_analyzer
                summary["use_reviewer"] = self.use_reviewer
                summary["analysis_only"] = self.analysis_only
                summary["use_selector"] = self.use_selector
                summary["repair_prompt_mode"] = "lightweight"
                summary["dual_repair_candidates"] = self.dual_repair_candidates
                summary["repair_context_mode"] = self.repair_context_mode
                summary["max_method_contexts"] = self.max_method_contexts
                summary["single_repair_path"] = not self._dual_candidate_enabled()
                summary["method_inquiry_enabled"] = self.repair_context_mode in {"method_query", "clone_treesitter"}
                self._write_summary_csv(output_dir / "summary.csv", summary)

            assert summary is not None
            return summary
        finally:
            self._release_run_lock(lock_path)

    def _acquire_run_lock(self, output_dir: Path) -> Path:
        lock_path = output_dir / ".run.lock"
        payload = json.dumps({"pid": os.getpid()}, ensure_ascii=False)
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = self._read_json_file_safely(lock_path)
            pid = existing.get("pid") if isinstance(existing, dict) else None
            if isinstance(pid, int) and not self._pid_is_running(pid):
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            else:
                raise RuntimeError(
                    f"Output directory is already locked by another run: {output_dir}. "
                    "Stop the other process or remove .run.lock if it is stale."
                )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        return lock_path

    def _release_run_lock(self, lock_path: Path) -> None:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    def _pid_is_running(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _read_json_file_safely(self, path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _reset_output_dir_for_fresh_run(self, output_dir: Path) -> None:
        for directory in (output_dir / "repair_debug", output_dir / "context_cache"):
            if directory.exists():
                shutil.rmtree(directory)
        for filename in (
            "task_progress.csv",
            "trajectory_overview.csv",
            "results.csv",
            "repairs.csv",
            "repair_candidates.csv",
            "selector_decisions.csv",
            "reviews.csv",
            "candidate_reviews.csv",
            "github_context.csv",
            "context_cache.csv",
            "summary.csv",
        ):
            path = output_dir / filename
            if path.exists():
                path.unlink()

    def _summarize(self, traces: list) -> dict:
        total = len(traces)
        analyze_filtered_count = sum(1 for trace in traces if trace.status == "dropped_by_analyzer")
        review_rejected_count = sum(1 for trace in traces if trace.status == "dropped_after_review")
        workflow_output_count = sum(1 for trace in traces if trace.status == "accepted")
        analyzer_pass_count = sum(1 for trace in traces if trace.status == "passed_by_analyzer")
        successful_repair_count = sum(1 for trace in traces if trace.status == "accepted" and trace.exact_match)

        return {
            "input_satd_count": total,
            "analyze_filtered_count": analyze_filtered_count,
            "analyzer_pass_count": analyzer_pass_count,
            "review_rejected_count": review_rejected_count,
            "workflow_output_count": workflow_output_count,
            "successful_repair_count": successful_repair_count,
            "precision": round(successful_repair_count / workflow_output_count, 4) if workflow_output_count else 0.0,
            "recall": round(successful_repair_count / total, 4) if total else 0.0,
        }

    def _write_outputs(self, output_dir: Path, traces: list, summary: dict) -> None:
        self._write_trajectory_overview_csv(output_dir / "trajectory_overview.csv", traces)
        self._write_results_csv(output_dir / "results.csv", traces)
        self._write_repairs_csv(output_dir / "repairs.csv", traces)
        self._write_repair_candidates_csv(output_dir / "repair_candidates.csv", traces)
        self._write_selector_decisions_csv(output_dir / "selector_decisions.csv", traces)
        self._write_reviews_csv(output_dir / "reviews.csv", traces)
        self._write_candidate_reviews_csv(output_dir / "candidate_reviews.csv", traces)
        self._write_github_context_csv(output_dir / "github_context.csv", traces)
        self._write_context_cache_csv(output_dir / "context_cache.csv", traces)
        self._write_summary_csv(output_dir / "summary.csv", summary)

    def _trace_context_metadata(self, trace) -> tuple[dict, dict]:
        context = trace.github_context or {}
        if not isinstance(context, dict):
            return {}, {}
        metadata = context.get("metadata", {})
        return context, metadata if isinstance(metadata, dict) else {}

    def _write_trajectory_overview_csv(self, path: Path, traces: list) -> None:
        fieldnames = [
            "task_id", "project", "file_path", "commit", "context_commit", "status", "workflow_output", "drop_stage", "trajectory_summary", "rounds_used", "em_label", "exact_match", "satd_comment", "satd_route_type",
            "analysis_decision", "analysis_passed", "analysis_repairability_score", "analysis_confidence", "analysis_satd_type", "analysis_risk_level", "analysis_scope_radius",
            "analysis_intent_clarity", "analysis_change_locality", "analysis_semantic_risk", "analysis_context_sufficiency", "analysis_verifiability", "analysis_analyze_score",
            "analysis_context_score", "analysis_clarity_score", "analysis_validation_signals", "analysis_context_gaps", "analysis_followup_context_requests", "analysis_evidence_summary",
            "analysis_repair_strategy", "analysis_operation_concrete", "analysis_localizable", "analysis_local_scope", "analysis_end_state_clear", "analysis_comment_evidence", "analysis_code_evidence", "analysis_historical_snapshot_mismatch", "analysis_github_evidence_strength", "analysis_snapshot_alignment_status",
            "repair_evidence_mode", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count",
            "identified_method_names", "retrieved_method_names", "missing_method_names", "retrieved_method_count", "method_context_json", "uncertainty_items_json", "edit_constraints_json",
            "repair_context_used", "review_strict_gate_result", "selector_selected_candidate_mode", "selector_confidence", "selector_rationale", "original_code", "processed_manual_code", "processed_final_repaired_code",
            "round_1_selector_mode", "round_1_selector_confidence", "round_1_candidate_mode", "round_1_repair_plan", "round_1_repaired_code", "round_1_changed_scope", "round_1_fix_confidence", "round_1_review_approved", "round_1_review_score", "round_1_review_problem_alignment", "round_1_review_minimality", "round_1_review_semantic_preservation", "round_1_review_internal_consistency", "round_1_review_softened_gate_used", "round_1_review_reject_type", "round_1_review_issues", "round_1_review_failed_checks", "round_1_review_repair_constraints", "round_1_review_failure_anchor", "round_1_revision_advice",
            "round_2_selector_mode", "round_2_selector_confidence", "round_2_candidate_mode", "round_2_repair_plan", "round_2_repaired_code", "round_2_changed_scope", "round_2_fix_confidence", "round_2_review_approved", "round_2_review_score", "round_2_review_problem_alignment", "round_2_review_minimality", "round_2_review_semantic_preservation", "round_2_review_internal_consistency", "round_2_review_softened_gate_used", "round_2_review_reject_type", "round_2_review_issues", "round_2_review_failed_checks", "round_2_review_repair_constraints", "round_2_review_failure_anchor", "round_2_revision_advice",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trace in traces:
                row = self._trajectory_row(trace)
                writer.writerow(self._select_fields(row, fieldnames))

    def _trajectory_row(self, trace) -> dict:
        analysis = trace.analysis or {}
        _, metadata = self._trace_context_metadata(trace)
        repairs = {item.get("round_id"): item for item in trace.repairs}
        reviews = {item.get("round_id"): item for item in trace.reviews}
        selector_decisions = {item.get("round_id"): item for item in trace.selector_decisions}
        latest_selector = trace.selector_decisions[-1] if trace.selector_decisions else {}
        method_inquiry = trace.method_inquiry or {}
        retrieved_method_names = [
            item.get("method_name")
            for item in trace.retrieved_method_contexts
            if item.get("method_name")
        ]
        row = {
            "task_id": trace.task_id,
            "project": trace.project,
            "file_path": trace.file_path,
            "commit": trace.commit,
            "context_commit": metadata.get("context_commit"),
            "status": trace.status,
            "workflow_output": (
                "YES"
                if trace.status == "accepted"
                else "ANALYZER_PASS"
                if trace.status == "passed_by_analyzer"
                else "NO"
            ),
            "drop_stage": self._drop_stage(trace),
            "trajectory_summary": self._trajectory_summary(trace),
            "rounds_used": trace.rounds_used,
            "em_label": trace.em_label,
            "exact_match": trace.exact_match,
            "satd_comment": trace.satd_comment,
            "satd_route_type": trace.satd_route_type,
            "analysis_decision": analysis.get("decision"),
            "analysis_passed": analysis.get("repairable"),
            "analysis_repairability_score": analysis.get("repairability_score"),
            "analysis_confidence": analysis.get("confidence"),
            "analysis_satd_type": analysis.get("satd_type"),
            "analysis_risk_level": analysis.get("risk_level"),
            "analysis_scope_radius": analysis.get("scope_radius"),
            "analysis_intent_clarity": analysis.get("intent_clarity"),
            "analysis_change_locality": analysis.get("change_locality"),
            "analysis_semantic_risk": analysis.get("semantic_risk"),
            "analysis_context_sufficiency": analysis.get("context_sufficiency"),
            "analysis_verifiability": analysis.get("verifiability"),
            "analysis_analyze_score": analysis.get("analyze_score"),
            "analysis_context_score": analysis.get("context_score"),
            "analysis_clarity_score": analysis.get("clarity_score"),
            "analysis_validation_signals": " | ".join(analysis.get("validation_signals", [])),
            "analysis_context_gaps": " | ".join(analysis.get("context_gaps", [])),
            "analysis_followup_context_requests": " | ".join(analysis.get("followup_context_requests", [])),
            "analysis_evidence_summary": analysis.get("evidence_summary") or analysis.get("reason"),
            "analysis_repair_strategy": analysis.get("repair_strategy"),
            "analysis_operation_concrete": analysis.get("operation_concrete"),
            "analysis_localizable": analysis.get("localizable"),
            "analysis_local_scope": analysis.get("local_scope"),
            "analysis_end_state_clear": analysis.get("end_state_clear"),
            "analysis_comment_evidence": analysis.get("comment_evidence"),
            "analysis_code_evidence": analysis.get("code_evidence"),
            "analysis_historical_snapshot_mismatch": analysis.get("historical_snapshot_mismatch"),
            "analysis_github_evidence_strength": analysis.get("github_evidence_strength"),
            "analysis_snapshot_alignment_status": metadata.get("snapshot_alignment_status"),
            "repair_evidence_mode": metadata.get("repair_evidence_mode"),
            "retrieved_test_snippets_count": metadata.get("retrieved_test_snippets_count"),
            "retrieved_callsite_snippets_count": metadata.get("retrieved_callsite_snippets_count"),
            "retrieved_history_snippets_count": metadata.get("retrieved_history_snippets_count"),
            "identified_method_names": " | ".join(method_inquiry.get("required_methods", [])),
            "retrieved_method_names": " | ".join(retrieved_method_names),
            "missing_method_names": " | ".join(trace.missing_method_names),
            "retrieved_method_count": len(trace.retrieved_method_contexts),
            "method_context_json": json.dumps(trace.retrieved_method_contexts, ensure_ascii=False),
            "uncertainty_items_json": json.dumps(trace.uncertainty_items, ensure_ascii=False),
            "edit_constraints_json": json.dumps(trace.edit_constraints, ensure_ascii=False),
            "repair_context_used": trace.repair_context_used,
            "review_strict_gate_result": trace.review_strict_gate_result,
            "selector_selected_candidate_mode": latest_selector.get("selected_candidate_mode"),
            "selector_confidence": latest_selector.get("confidence"),
            "selector_rationale": latest_selector.get("rationale"),
            "original_code": trace.original_code,
            "processed_manual_code": trace.processed_manual_code,
            "processed_final_repaired_code": trace.processed_final_repaired_code,
        }
        for round_id in (1, 2):
            repair = repairs.get(round_id, {})
            review = reviews.get(round_id, {})
            selector = selector_decisions.get(round_id, {})
            row[f"round_{round_id}_candidate_mode"] = repair.get("candidate_mode")
            row[f"round_{round_id}_repair_plan"] = repair.get("repair_plan")
            row[f"round_{round_id}_repaired_code"] = preprocess_python_code(repair.get("repaired_code")) if repair else None
            row[f"round_{round_id}_changed_scope"] = repair.get("changed_scope")
            row[f"round_{round_id}_fix_confidence"] = repair.get("confidence")
            row[f"round_{round_id}_review_approved"] = review.get("approved")
            row[f"round_{round_id}_review_score"] = review.get("review_score")
            row[f"round_{round_id}_review_problem_alignment"] = review.get("problem_alignment")
            row[f"round_{round_id}_review_minimality"] = review.get("minimality")
            row[f"round_{round_id}_review_semantic_preservation"] = review.get("semantic_preservation")
            row[f"round_{round_id}_review_internal_consistency"] = review.get("internal_consistency")
            row[f"round_{round_id}_review_softened_gate_used"] = review.get("softened_gate_used")
            row[f"round_{round_id}_review_reject_type"] = review.get("reject_type")
            row[f"round_{round_id}_review_issues"] = " | ".join(review.get("issues", [])) if review else None
            row[f"round_{round_id}_review_failed_checks"] = " | ".join(review.get("failed_checks", [])) if review else None
            row[f"round_{round_id}_review_repair_constraints"] = " | ".join(review.get("repair_constraints", [])) if review else None
            row[f"round_{round_id}_review_failure_anchor"] = review.get("failure_anchor")
            row[f"round_{round_id}_revision_advice"] = review.get("revision_advice")
            row[f"round_{round_id}_selector_mode"] = selector.get("selected_candidate_mode")
            row[f"round_{round_id}_selector_confidence"] = selector.get("confidence")
        return row

    def _select_fields(self, row: dict, fieldnames: list[str]) -> dict:
        return {field: row.get(field) for field in fieldnames}

    def _trajectory_summary(self, trace) -> str:
        steps = []
        analysis = trace.analysis or {}
        steps.append(f"analysis:{analysis.get('decision') or ('pass' if analysis.get('repairable') else 'drop')}")
        if trace.repair_context_used:
            steps.append("repair_context:used")
        selector_by_round = {item.get("round_id"): item for item in trace.selector_decisions}
        for repair in trace.repairs:
            candidate_mode = repair.get("candidate_mode")
            if candidate_mode:
                steps.append(f"repair{repair.get('round_id')}:{candidate_mode}")
            else:
                steps.append(f"repair{repair.get('round_id')}")
            selector = selector_by_round.get(repair.get("round_id"))
            if selector and selector.get("selected_candidate_mode"):
                steps.append(f"select{repair.get('round_id')}:{selector.get('selected_candidate_mode')}")
        for review in trace.reviews:
            outcome = "pass" if review.get("approved") else "reject"
            steps.append(f"review{review.get('round_id')}:{outcome}")
        if trace.status == "accepted":
            steps.append("workflow:output")
        elif trace.status == "passed_by_analyzer":
            steps.append("workflow:analyzer_pass")
        else:
            steps.append(f"workflow:drop@{self._drop_stage(trace)}")
        return " -> ".join(steps)

    def _drop_stage(self, trace) -> str:
        if trace.status == "dropped_by_analyzer":
            return "analyzer"
        if trace.status == "dropped_after_review" and trace.reviews:
            last_round = trace.reviews[-1].get("round_id")
            return f"review_round_{last_round}"
        return ""

    def _write_results_csv(self, path: Path, traces: list) -> None:
        fieldnames = [
            "task_id", "project", "file_path", "commit", "context_commit", "satd_comment", "status", "rounds_used", "em_label", "exact_match", "satd_route_type",
            "analysis_decision", "analysis_repairable", "analysis_repairability_score", "analysis_confidence", "analysis_satd_type", "analysis_risk_level", "analysis_scope_radius",
            "analysis_intent_clarity", "analysis_change_locality", "analysis_semantic_risk", "analysis_context_sufficiency", "analysis_verifiability", "analysis_analyze_score",
            "analysis_context_score", "analysis_clarity_score", "analysis_validation_signals", "analysis_context_gaps", "analysis_followup_context_requests", "analysis_evidence_summary",
            "analysis_repair_strategy", "analysis_operation_concrete", "analysis_localizable", "analysis_local_scope", "analysis_end_state_clear", "analysis_comment_evidence", "analysis_code_evidence", "analysis_historical_snapshot_mismatch", "analysis_github_evidence_strength", "analysis_snapshot_alignment_status",
            "repair_evidence_mode", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count",
            "identified_method_names", "retrieved_method_names", "missing_method_names", "retrieved_method_count", "method_context_json", "uncertainty_items_json", "edit_constraints_json",
            "repair_context_used", "review_strict_gate_result", "selector_selected_candidate_mode", "selector_confidence", "selector_rationale", "original_code", "processed_manual_code", "processed_final_repaired_code",
            "round_1_selector_mode", "round_1_selector_confidence", "round_1_candidate_mode", "round_1_repair_plan", "round_1_repaired_code", "round_1_changed_scope", "round_1_fix_confidence", "round_1_review_approved", "round_1_review_score", "round_1_review_problem_alignment", "round_1_review_minimality", "round_1_review_semantic_preservation", "round_1_review_internal_consistency", "round_1_review_softened_gate_used", "round_1_review_reject_type", "round_1_review_issues", "round_1_review_failed_checks", "round_1_review_repair_constraints", "round_1_review_failure_anchor", "round_1_revision_advice",
            "round_2_selector_mode", "round_2_selector_confidence", "round_2_candidate_mode", "round_2_repair_plan", "round_2_repaired_code", "round_2_changed_scope", "round_2_fix_confidence", "round_2_review_approved", "round_2_review_score", "round_2_review_problem_alignment", "round_2_review_minimality", "round_2_review_semantic_preservation", "round_2_review_internal_consistency", "round_2_review_softened_gate_used", "round_2_review_reject_type", "round_2_review_issues", "round_2_review_failed_checks", "round_2_review_repair_constraints", "round_2_review_failure_anchor", "round_2_revision_advice",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trace in traces:
                writer.writerow(self._select_fields(self._trajectory_row(trace), fieldnames))

    def _write_repairs_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "round_id", "candidate_mode", "repair_plan", "repaired_code", "changed_scope", "confidence", "notes"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trace in traces:
                for repair in trace.repairs:
                    row = {"task_id": trace.task_id, **repair}
                    row["repaired_code"] = preprocess_python_code(repair.get("repaired_code"))
                    writer.writerow(row)

    def _write_repair_candidates_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "round_id", "candidate_mode", "repair_plan", "repaired_code", "changed_scope", "confidence", "notes"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trace in traces:
                for repair in trace.candidate_repairs:
                    row = {"task_id": trace.task_id, **repair}
                    row["repaired_code"] = preprocess_python_code(repair.get("repaired_code"))
                    writer.writerow(row)

    def _write_selector_decisions_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "round_id", "satd_route_type", "selected_candidate_mode", "selected_index", "confidence", "rationale", "candidate_scores"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trace in traces:
                for decision in trace.selector_decisions:
                    row = dict(decision)
                    row["candidate_scores"] = json.dumps(row.get("candidate_scores", []), ensure_ascii=False)
                    writer.writerow({"task_id": trace.task_id, **row})

    def _write_reviews_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "round_id", "candidate_mode", "approved", "review_score", "problem_alignment", "minimality", "semantic_preservation", "internal_consistency", "softened_gate_used", "issues", "revision_advice", "reject_type", "rationale", "failed_checks", "repair_constraints", "failure_anchor", "retry_hint"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trace in traces:
                for review in trace.reviews:
                    row = dict(review)
                    row["issues"] = json.dumps(row.get("issues", []), ensure_ascii=False)
                    row["failed_checks"] = json.dumps(row.get("failed_checks", []), ensure_ascii=False)
                    row["repair_constraints"] = json.dumps(row.get("repair_constraints", []), ensure_ascii=False)
                    writer.writerow({"task_id": trace.task_id, **row})

    def _write_candidate_reviews_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "round_id", "candidate_mode", "approved", "review_score", "problem_alignment", "minimality", "semantic_preservation", "internal_consistency", "softened_gate_used", "issues", "revision_advice", "reject_type", "rationale", "failed_checks", "repair_constraints", "failure_anchor", "retry_hint"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trace in traces:
                for review in trace.candidate_reviews:
                    row = dict(review)
                    row["issues"] = json.dumps(row.get("issues", []), ensure_ascii=False)
                    row["failed_checks"] = json.dumps(row.get("failed_checks", []), ensure_ascii=False)
                    row["repair_constraints"] = json.dumps(row.get("repair_constraints", []), ensure_ascii=False)
                    writer.writerow({"task_id": trace.task_id, **row})

    def _write_github_context_csv(self, path: Path, traces: list) -> None:
        fieldnames = [
            "task_id", "repo_owner", "repo_name", "file_path", "commit", "context_commit", "historical_snapshot_mismatch", "github_evidence_strength", "snapshot_alignment_status", "repair_evidence_mode", "target_file_ok", "satd_window_found", "enclosing_symbol_found", "symbol_name", "satd_line",
            "related_tests_count", "call_sites_count", "commits_count", "similar_history_count", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count", "identified_method_count", "retrieved_method_count", "missing_method_count", "base_context_json", "repair_context_json", "review_context_json", "method_context_json",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trace in traces:
                context, metadata = self._trace_context_metadata(trace)
                writer.writerow({
                    "task_id": trace.task_id,
                    "repo_owner": context.get("repo_owner"),
                    "repo_name": context.get("repo_name"),
                    "file_path": context.get("file_path"),
                    "commit": trace.commit,
                    "context_commit": metadata.get("context_commit"),
                    "historical_snapshot_mismatch": metadata.get("historical_snapshot_mismatch"),
                    "github_evidence_strength": metadata.get("github_evidence_strength"),
                    "snapshot_alignment_status": metadata.get("snapshot_alignment_status"),
                    "repair_evidence_mode": metadata.get("repair_evidence_mode"),
                    "target_file_ok": metadata.get("target_file_ok"),
                    "satd_window_found": metadata.get("satd_window_found"),
                    "enclosing_symbol_found": metadata.get("enclosing_symbol_found"),
                    "symbol_name": metadata.get("symbol_name"),
                    "satd_line": metadata.get("satd_line"),
                    "related_tests_count": metadata.get("related_tests_count"),
                    "call_sites_count": metadata.get("call_sites_count"),
                    "commits_count": metadata.get("commits_count"),
                    "similar_history_count": metadata.get("similar_history_count"),
                    "retrieved_test_snippets_count": metadata.get("retrieved_test_snippets_count"),
                    "retrieved_callsite_snippets_count": metadata.get("retrieved_callsite_snippets_count"),
                    "retrieved_history_snippets_count": metadata.get("retrieved_history_snippets_count"),
                    "identified_method_count": metadata.get("identified_method_count"),
                    "retrieved_method_count": metadata.get("retrieved_method_count"),
                    "missing_method_count": metadata.get("missing_method_count"),
                    "base_context_json": json.dumps(context.get("base_context", {}), ensure_ascii=False),
                    "repair_context_json": json.dumps(context.get("repair_context", {}), ensure_ascii=False),
                    "review_context_json": json.dumps(context.get("review_context", {}), ensure_ascii=False),
                    "method_context_json": json.dumps((context.get("repair_context", {}) or {}).get("retrieved_methods", []), ensure_ascii=False),
                })

    def _write_context_cache_csv(self, path: Path, traces: list) -> None:
        fieldnames = [
            "task_id", "commit", "context_commit", "cache_file", "base_cached", "base_cache_source", "base_context_fetched_at", "repair_cached", "repair_cache_source", "repair_context_fetched_at", "review_cached", "review_cache_source", "review_context_fetched_at",
            "historical_snapshot_mismatch", "github_evidence_strength", "snapshot_alignment_status", "repair_evidence_mode", "target_file_ok", "satd_window_found", "enclosing_symbol_found", "related_tests_count", "call_sites_count", "commits_count", "similar_history_count", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count", "identified_method_count", "retrieved_method_count", "missing_method_count",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trace in traces:
                _, metadata = self._trace_context_metadata(trace)
                writer.writerow({
                    "task_id": trace.task_id,
                    "commit": trace.commit,
                    "context_commit": metadata.get("context_commit"),
                    "cache_file": str(self._context_cache_file(trace.task_id)),
                    "base_cached": metadata.get("base_cached"),
                    "base_cache_source": metadata.get("base_cache_source"),
                    "base_context_fetched_at": metadata.get("base_context_fetched_at"),
                    "repair_cached": metadata.get("repair_cached"),
                    "repair_cache_source": metadata.get("repair_cache_source"),
                    "repair_context_fetched_at": metadata.get("repair_context_fetched_at"),
                    "review_cached": metadata.get("review_cached"),
                    "review_cache_source": metadata.get("review_cache_source"),
                    "review_context_fetched_at": metadata.get("review_context_fetched_at"),
                    "historical_snapshot_mismatch": metadata.get("historical_snapshot_mismatch"),
                    "github_evidence_strength": metadata.get("github_evidence_strength"),
                    "snapshot_alignment_status": metadata.get("snapshot_alignment_status"),
                    "repair_evidence_mode": metadata.get("repair_evidence_mode"),
                    "target_file_ok": metadata.get("target_file_ok"),
                    "satd_window_found": metadata.get("satd_window_found"),
                    "enclosing_symbol_found": metadata.get("enclosing_symbol_found"),
                    "related_tests_count": metadata.get("related_tests_count"),
                    "call_sites_count": metadata.get("call_sites_count"),
                    "commits_count": metadata.get("commits_count"),
                    "similar_history_count": metadata.get("similar_history_count"),
                    "retrieved_test_snippets_count": metadata.get("retrieved_test_snippets_count"),
                    "retrieved_callsite_snippets_count": metadata.get("retrieved_callsite_snippets_count"),
                    "retrieved_history_snippets_count": metadata.get("retrieved_history_snippets_count"),
                    "identified_method_count": metadata.get("identified_method_count"),
                    "retrieved_method_count": metadata.get("retrieved_method_count"),
                    "missing_method_count": metadata.get("missing_method_count"),
                })

    def _append_outputs(self, output_dir: Path, traces: list) -> None:
        self._append_trajectory_overview_csv(output_dir / "trajectory_overview.csv", traces)
        self._append_results_csv(output_dir / "results.csv", traces)
        self._append_repairs_csv(output_dir / "repairs.csv", traces)
        self._append_repair_candidates_csv(output_dir / "repair_candidates.csv", traces)
        self._append_selector_decisions_csv(output_dir / "selector_decisions.csv", traces)
        self._append_reviews_csv(output_dir / "reviews.csv", traces)
        self._append_candidate_reviews_csv(output_dir / "candidate_reviews.csv", traces)
        self._append_github_context_csv(output_dir / "github_context.csv", traces)
        self._append_context_cache_csv(output_dir / "context_cache.csv", traces)

    def _write_task_progress_csv(self, path: Path, traces: list) -> None:
        fieldnames = [
            "task_id",
            "task_index",
            "task_total",
            "project",
            "file_path",
            "commit",
            "status",
            "rounds_used",
            "exact_match",
            "identified_method_names",
            "retrieved_method_names",
            "missing_method_names",
            "retrieved_method_count",
            "latest_stage",
            "debug_file",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._task_progress_rows(traces):
                writer.writerow(row)

    def _append_task_progress_csv(self, path: Path, traces: list, task_index: int | None = None, task_total: int | None = None) -> None:
        fieldnames = [
            "task_id",
            "task_index",
            "task_total",
            "project",
            "file_path",
            "commit",
            "status",
            "rounds_used",
            "exact_match",
            "identified_method_names",
            "retrieved_method_names",
            "missing_method_names",
            "retrieved_method_count",
            "latest_stage",
            "debug_file",
        ]
        rows = self._task_progress_rows(traces, task_index=task_index, task_total=task_total)
        self._append_csv_rows(path, fieldnames, rows)

    def _task_progress_rows(self, traces: list, task_index: int | None = None, task_total: int | None = None) -> list[dict]:
        rows = []
        for trace in traces:
            method_inquiry = trace.method_inquiry or {}
            retrieved_method_names = [
                item.get("method_name")
                for item in trace.retrieved_method_contexts
                if item.get("method_name")
            ]
            debug_path = self._repair_debug_file(trace.task_id) if self._current_output_dir is not None else None
            latest_stage = ""
            if debug_path and debug_path.exists():
                try:
                    debug_payload = json.loads(debug_path.read_text(encoding="utf-8"))
                    latest_stage = str(debug_payload.get("latest_stage") or "")
                except json.JSONDecodeError:
                    latest_stage = ""
            rows.append(
                {
                    "task_id": trace.task_id,
                    "task_index": task_index,
                    "task_total": task_total,
                    "project": trace.project,
                    "file_path": trace.file_path,
                    "commit": trace.commit,
                    "status": trace.status,
                    "rounds_used": trace.rounds_used,
                    "exact_match": trace.exact_match,
                    "identified_method_names": " | ".join(method_inquiry.get("required_methods", [])),
                    "retrieved_method_names": " | ".join(retrieved_method_names),
                    "missing_method_names": " | ".join(trace.missing_method_names),
                    "retrieved_method_count": len(trace.retrieved_method_contexts),
                    "latest_stage": latest_stage,
                    "debug_file": str(debug_path) if debug_path else "",
                }
            )
        return rows

    def _append_csv_rows(self, path: Path, fieldnames: list[str], rows: list[dict]) -> None:
        if not rows:
            return
        self._ensure_csv_schema(path, fieldnames)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(self._select_fields(row, fieldnames))

    def _ensure_csv_schema(self, path: Path, fieldnames: list[str]) -> None:
        if not path.exists() or path.stat().st_size == 0:
            return
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = reader.fieldnames or []
            if existing_fieldnames == fieldnames:
                return
            existing_rows = list(reader)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in existing_rows:
                writer.writerow(self._select_fields(row, fieldnames))

    def _append_trajectory_overview_csv(self, path: Path, traces: list) -> None:
        fieldnames = [
            "task_id", "project", "file_path", "commit", "context_commit", "status", "workflow_output", "drop_stage", "trajectory_summary", "rounds_used", "em_label", "exact_match", "satd_comment", "satd_route_type",
            "analysis_decision", "analysis_passed", "analysis_repairability_score", "analysis_confidence", "analysis_satd_type", "analysis_risk_level", "analysis_scope_radius",
            "analysis_intent_clarity", "analysis_change_locality", "analysis_semantic_risk", "analysis_context_sufficiency", "analysis_verifiability", "analysis_analyze_score",
            "analysis_context_score", "analysis_clarity_score", "analysis_validation_signals", "analysis_context_gaps", "analysis_followup_context_requests", "analysis_evidence_summary",
            "analysis_repair_strategy", "analysis_operation_concrete", "analysis_localizable", "analysis_local_scope", "analysis_end_state_clear", "analysis_comment_evidence", "analysis_code_evidence", "analysis_historical_snapshot_mismatch", "analysis_github_evidence_strength", "analysis_snapshot_alignment_status",
            "repair_evidence_mode", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count",
            "identified_method_names", "retrieved_method_names", "missing_method_names", "retrieved_method_count", "method_context_json", "uncertainty_items_json", "edit_constraints_json",
            "repair_context_used", "review_strict_gate_result", "selector_selected_candidate_mode", "selector_confidence", "selector_rationale", "original_code", "processed_manual_code", "processed_final_repaired_code",
            "round_1_selector_mode", "round_1_selector_confidence", "round_1_candidate_mode", "round_1_repair_plan", "round_1_repaired_code", "round_1_changed_scope", "round_1_fix_confidence", "round_1_review_approved", "round_1_review_score", "round_1_review_problem_alignment", "round_1_review_minimality", "round_1_review_semantic_preservation", "round_1_review_internal_consistency", "round_1_review_softened_gate_used", "round_1_review_reject_type", "round_1_review_issues", "round_1_review_failed_checks", "round_1_review_repair_constraints", "round_1_review_failure_anchor", "round_1_revision_advice",
            "round_2_selector_mode", "round_2_selector_confidence", "round_2_candidate_mode", "round_2_repair_plan", "round_2_repaired_code", "round_2_changed_scope", "round_2_fix_confidence", "round_2_review_approved", "round_2_review_score", "round_2_review_problem_alignment", "round_2_review_minimality", "round_2_review_semantic_preservation", "round_2_review_internal_consistency", "round_2_review_softened_gate_used", "round_2_review_reject_type", "round_2_review_issues", "round_2_review_failed_checks", "round_2_review_repair_constraints", "round_2_review_failure_anchor", "round_2_revision_advice",
        ]
        rows = [self._trajectory_row(trace) for trace in traces]
        self._append_csv_rows(path, fieldnames, rows)

    def _append_results_csv(self, path: Path, traces: list) -> None:
        fieldnames = [
            "task_id", "project", "file_path", "commit", "context_commit", "satd_comment", "status", "rounds_used", "em_label", "exact_match", "satd_route_type",
            "analysis_decision", "analysis_repairable", "analysis_repairability_score", "analysis_confidence", "analysis_satd_type", "analysis_risk_level", "analysis_scope_radius",
            "analysis_intent_clarity", "analysis_change_locality", "analysis_semantic_risk", "analysis_context_sufficiency", "analysis_verifiability", "analysis_analyze_score",
            "analysis_context_score", "analysis_clarity_score", "analysis_validation_signals", "analysis_context_gaps", "analysis_followup_context_requests", "analysis_evidence_summary",
            "analysis_repair_strategy", "analysis_operation_concrete", "analysis_localizable", "analysis_local_scope", "analysis_end_state_clear", "analysis_comment_evidence", "analysis_code_evidence", "analysis_historical_snapshot_mismatch", "analysis_github_evidence_strength", "analysis_snapshot_alignment_status",
            "repair_evidence_mode", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count",
            "identified_method_names", "retrieved_method_names", "missing_method_names", "retrieved_method_count", "method_context_json", "uncertainty_items_json", "edit_constraints_json",
            "repair_context_used", "review_strict_gate_result", "selector_selected_candidate_mode", "selector_confidence", "selector_rationale", "original_code", "processed_manual_code", "processed_final_repaired_code",
            "round_1_selector_mode", "round_1_selector_confidence",
            "round_1_candidate_mode", "round_1_repair_plan", "round_1_repaired_code", "round_1_changed_scope", "round_1_fix_confidence", "round_1_review_approved", "round_1_review_score", "round_1_review_problem_alignment", "round_1_review_minimality", "round_1_review_semantic_preservation", "round_1_review_internal_consistency", "round_1_review_softened_gate_used", "round_1_review_reject_type", "round_1_review_issues", "round_1_review_failed_checks", "round_1_review_repair_constraints", "round_1_review_failure_anchor", "round_1_revision_advice",
            "round_2_selector_mode", "round_2_selector_confidence",
            "round_2_candidate_mode", "round_2_repair_plan", "round_2_repaired_code", "round_2_changed_scope", "round_2_fix_confidence", "round_2_review_approved", "round_2_review_score", "round_2_review_problem_alignment", "round_2_review_minimality", "round_2_review_semantic_preservation", "round_2_review_internal_consistency", "round_2_review_softened_gate_used", "round_2_review_reject_type", "round_2_review_issues", "round_2_review_failed_checks", "round_2_review_repair_constraints", "round_2_review_failure_anchor", "round_2_revision_advice",
        ]
        rows = []
        for trace in traces:
            rows.append(self._select_fields(self._trajectory_row(trace), fieldnames))
        self._append_csv_rows(path, fieldnames, rows)

    def _append_repairs_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "round_id", "candidate_mode", "repair_plan", "repaired_code", "changed_scope", "confidence", "notes"]
        rows = []
        for trace in traces:
            for repair in trace.repairs:
                row = {"task_id": trace.task_id, **repair}
                row["repaired_code"] = preprocess_python_code(repair.get("repaired_code"))
                rows.append(row)
        self._append_csv_rows(path, fieldnames, rows)

    def _append_repair_candidates_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "round_id", "candidate_mode", "repair_plan", "repaired_code", "changed_scope", "confidence", "notes"]
        rows = []
        for trace in traces:
            for repair in trace.candidate_repairs:
                row = {"task_id": trace.task_id, **repair}
                row["repaired_code"] = preprocess_python_code(repair.get("repaired_code"))
                rows.append(row)
        self._append_csv_rows(path, fieldnames, rows)

    def _append_selector_decisions_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "round_id", "satd_route_type", "selected_candidate_mode", "selected_index", "confidence", "rationale", "candidate_scores"]
        rows = []
        for trace in traces:
            for decision in trace.selector_decisions:
                row = dict(decision)
                row["candidate_scores"] = json.dumps(row.get("candidate_scores", []), ensure_ascii=False)
                rows.append({"task_id": trace.task_id, **row})
        self._append_csv_rows(path, fieldnames, rows)

    def _append_reviews_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "round_id", "candidate_mode", "approved", "review_score", "problem_alignment", "minimality", "semantic_preservation", "internal_consistency", "softened_gate_used", "issues", "revision_advice", "reject_type", "rationale", "failed_checks", "repair_constraints", "failure_anchor", "retry_hint"]
        rows = []
        for trace in traces:
            for review in trace.reviews:
                row = dict(review)
                row["issues"] = json.dumps(row.get("issues", []), ensure_ascii=False)
                row["failed_checks"] = json.dumps(row.get("failed_checks", []), ensure_ascii=False)
                row["repair_constraints"] = json.dumps(row.get("repair_constraints", []), ensure_ascii=False)
                rows.append({"task_id": trace.task_id, **row})
        self._append_csv_rows(path, fieldnames, rows)

    def _append_candidate_reviews_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "round_id", "candidate_mode", "approved", "review_score", "problem_alignment", "minimality", "semantic_preservation", "internal_consistency", "softened_gate_used", "issues", "revision_advice", "reject_type", "rationale", "failed_checks", "repair_constraints", "failure_anchor", "retry_hint"]
        rows = []
        for trace in traces:
            for review in trace.candidate_reviews:
                row = dict(review)
                row["issues"] = json.dumps(row.get("issues", []), ensure_ascii=False)
                row["failed_checks"] = json.dumps(row.get("failed_checks", []), ensure_ascii=False)
                row["repair_constraints"] = json.dumps(row.get("repair_constraints", []), ensure_ascii=False)
                rows.append({"task_id": trace.task_id, **row})
        self._append_csv_rows(path, fieldnames, rows)

    def _append_github_context_csv(self, path: Path, traces: list) -> None:
        fieldnames = [
            "task_id", "repo_owner", "repo_name", "file_path", "commit", "context_commit", "historical_snapshot_mismatch", "github_evidence_strength", "snapshot_alignment_status", "repair_evidence_mode", "target_file_ok", "satd_window_found", "enclosing_symbol_found", "symbol_name", "satd_line",
            "related_tests_count", "call_sites_count", "commits_count", "similar_history_count", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count", "identified_method_count", "retrieved_method_count", "missing_method_count", "base_context_json", "repair_context_json", "review_context_json", "method_context_json",
        ]
        rows = []
        for trace in traces:
            context, metadata = self._trace_context_metadata(trace)
            rows.append({
                "task_id": trace.task_id,
                "repo_owner": context.get("repo_owner"),
                "repo_name": context.get("repo_name"),
                "file_path": context.get("file_path"),
                "commit": trace.commit,
                "context_commit": metadata.get("context_commit"),
                "historical_snapshot_mismatch": metadata.get("historical_snapshot_mismatch"),
                "github_evidence_strength": metadata.get("github_evidence_strength"),
                "snapshot_alignment_status": metadata.get("snapshot_alignment_status"),
                "repair_evidence_mode": metadata.get("repair_evidence_mode"),
                "target_file_ok": metadata.get("target_file_ok"),
                "satd_window_found": metadata.get("satd_window_found"),
                "enclosing_symbol_found": metadata.get("enclosing_symbol_found"),
                "symbol_name": metadata.get("symbol_name"),
                "satd_line": metadata.get("satd_line"),
                "related_tests_count": metadata.get("related_tests_count"),
                "call_sites_count": metadata.get("call_sites_count"),
                "commits_count": metadata.get("commits_count"),
                "similar_history_count": metadata.get("similar_history_count"),
                "retrieved_test_snippets_count": metadata.get("retrieved_test_snippets_count"),
                "retrieved_callsite_snippets_count": metadata.get("retrieved_callsite_snippets_count"),
                "retrieved_history_snippets_count": metadata.get("retrieved_history_snippets_count"),
                "identified_method_count": metadata.get("identified_method_count"),
                "retrieved_method_count": metadata.get("retrieved_method_count"),
                "missing_method_count": metadata.get("missing_method_count"),
                "base_context_json": json.dumps(context.get("base_context", {}), ensure_ascii=False),
                "repair_context_json": json.dumps(context.get("repair_context", {}), ensure_ascii=False),
                "review_context_json": json.dumps(context.get("review_context", {}), ensure_ascii=False),
                "method_context_json": json.dumps((context.get("repair_context", {}) or {}).get("retrieved_methods", []), ensure_ascii=False),
            })
        self._append_csv_rows(path, fieldnames, rows)

    def _append_context_cache_csv(self, path: Path, traces: list) -> None:
        fieldnames = [
            "task_id", "commit", "context_commit", "cache_file", "base_cached", "base_cache_source", "base_context_fetched_at", "repair_cached", "repair_cache_source", "repair_context_fetched_at", "review_cached", "review_cache_source", "review_context_fetched_at",
            "historical_snapshot_mismatch", "github_evidence_strength", "snapshot_alignment_status", "repair_evidence_mode", "target_file_ok", "satd_window_found", "enclosing_symbol_found", "related_tests_count", "call_sites_count", "commits_count", "similar_history_count", "retrieved_test_snippets_count", "retrieved_callsite_snippets_count", "retrieved_history_snippets_count", "identified_method_count", "retrieved_method_count", "missing_method_count",
        ]
        rows = []
        for trace in traces:
            _, metadata = self._trace_context_metadata(trace)
            rows.append({
                "task_id": trace.task_id,
                "commit": trace.commit,
                "context_commit": metadata.get("context_commit"),
                "cache_file": str(self._context_cache_file(trace.task_id)),
                "base_cached": metadata.get("base_cached"),
                "base_cache_source": metadata.get("base_cache_source"),
                "base_context_fetched_at": metadata.get("base_context_fetched_at"),
                "repair_cached": metadata.get("repair_cached"),
                "repair_cache_source": metadata.get("repair_cache_source"),
                "repair_context_fetched_at": metadata.get("repair_context_fetched_at"),
                "review_cached": metadata.get("review_cached"),
                "review_cache_source": metadata.get("review_cache_source"),
                "review_context_fetched_at": metadata.get("review_context_fetched_at"),
                "historical_snapshot_mismatch": metadata.get("historical_snapshot_mismatch"),
                "github_evidence_strength": metadata.get("github_evidence_strength"),
                "snapshot_alignment_status": metadata.get("snapshot_alignment_status"),
                "repair_evidence_mode": metadata.get("repair_evidence_mode"),
                "target_file_ok": metadata.get("target_file_ok"),
                "satd_window_found": metadata.get("satd_window_found"),
                "enclosing_symbol_found": metadata.get("enclosing_symbol_found"),
                "related_tests_count": metadata.get("related_tests_count"),
                "call_sites_count": metadata.get("call_sites_count"),
                "commits_count": metadata.get("commits_count"),
                "similar_history_count": metadata.get("similar_history_count"),
                "retrieved_test_snippets_count": metadata.get("retrieved_test_snippets_count"),
                "retrieved_callsite_snippets_count": metadata.get("retrieved_callsite_snippets_count"),
                "retrieved_history_snippets_count": metadata.get("retrieved_history_snippets_count"),
                "identified_method_count": metadata.get("identified_method_count"),
                "retrieved_method_count": metadata.get("retrieved_method_count"),
                "missing_method_count": metadata.get("missing_method_count"),
            })
        self._append_csv_rows(path, fieldnames, rows)

    def _empty_existing_rows(self) -> dict[str, list[dict]]:
        return {"results": [], "reviews": [], "context_cache": []}

    def _load_existing_rows(self, output_dir: Path) -> dict[str, list[dict]]:
        return {
            "results": self._read_csv_rows(output_dir / "results.csv"),
            "reviews": self._read_csv_rows(output_dir / "reviews.csv"),
            "context_cache": self._read_csv_rows(output_dir / "context_cache.csv"),
        }

    def _read_csv_rows(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def _summarize_from_existing_and_new(self, existing_rows: dict[str, list[dict]], new_traces: list) -> dict:
        result_rows = list(existing_rows.get("results", []))

        for trace in new_traces:
            result_rows.append({
                "task_id": trace.task_id,
                "status": trace.status,
                "exact_match": str(bool(trace.exact_match)) if trace.exact_match is not None else "",
            })

        total = len(result_rows)
        analyze_filtered_count = sum(1 for row in result_rows if row.get("status") == "dropped_by_analyzer")
        review_rejected_count = sum(1 for row in result_rows if row.get("status") == "dropped_after_review")
        workflow_output_count = sum(1 for row in result_rows if row.get("status") == "accepted")
        analyzer_pass_count = sum(1 for row in result_rows if row.get("status") == "passed_by_analyzer")
        successful_repair_count = sum(
            1
            for row in result_rows
            if row.get("status") == "accepted" and str(row.get("exact_match")).lower() == "true"
        )

        return {
            "input_satd_count": total,
            "analyze_filtered_count": analyze_filtered_count,
            "analyzer_pass_count": analyzer_pass_count,
            "review_rejected_count": review_rejected_count,
            "workflow_output_count": workflow_output_count,
            "successful_repair_count": successful_repair_count,
            "precision": round(successful_repair_count / workflow_output_count, 4) if workflow_output_count else 0.0,
            "recall": round(successful_repair_count / total, 4) if total else 0.0,
        }

    def _write_summary_csv(self, path: Path, summary: dict) -> None:
        fieldnames = list(summary.keys())
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(summary)

    def _load_or_build_base_context(self, state: GraphState) -> dict:
        cached = self._load_context_cache(state["task_id"])
        if cached and (cached.get("metadata") or {}).get("base_cached"):
            cached.setdefault("metadata", {})["base_cache_source"] = "disk_cache"
            cached = self.context_client.build_base_context(state, cached)
            self._persist_context_cache(state["task_id"], cached)
            return cached
        built = self.context_client.build_base_context(state, cached)
        self._persist_context_cache(state["task_id"], built)
        return built

    def _load_or_build_shared_context(self, state: GraphState) -> dict:
        bundle = state.get("github_context") or self._load_or_build_base_context(state)
        bundle = self.context_client.ensure_repair_context(state, bundle)
        bundle.setdefault("metadata", {})["shared_context_mode"] = True
        self._persist_context_cache(state["task_id"], bundle)
        return bundle

    def _ensure_repair_context_cache(self, state: GraphState) -> dict:
        bundle = state.get("github_context") or self._load_or_build_base_context(state)
        self._persist_context_cache(state["task_id"], bundle)
        return bundle

    def _ensure_review_context_cache(self, state: GraphState) -> dict:
        bundle = state.get("github_context") or self._load_or_build_base_context(state)
        bundle.setdefault("metadata", {})["review_cache_source"] = "shared_context"
        self._persist_context_cache(state["task_id"], bundle)
        return bundle

    def _attach_method_context(
        self,
        bundle: dict,
        *,
        method_inquiry: MethodInquiryResult,
        retrieved_method_contexts: list[RetrievedMethodContext],
        missing_method_names: list[str],
    ) -> dict:
        updated = dict(bundle or {})
        metadata = dict(updated.get("metadata", {}) or {})
        updated["metadata"] = metadata
        context_strategy = self.repair_context_mode or "clone_treesitter"
        updated["repair_context"] = {
            "context_strategy": context_strategy,
            "method_inquiry": {
                "required_methods": list(method_inquiry.required_methods),
                "reason": method_inquiry.reason,
            },
            "retrieved_methods": [
                {
                    "method_name": item.method_name,
                    "path": item.path,
                    "class_name": item.class_name,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "source": item.source,
                    "found": item.found,
                }
                for item in retrieved_method_contexts
            ],
            "missing_method_names": list(missing_method_names),
        }
        metadata.update(
            {
                "repair_cached": True,
                "repair_cache_source": context_strategy,
                "repair_context_fetched_at": metadata.get("repair_context_fetched_at") or self.context_client._timestamp(),
                "context_strategy": context_strategy,
                "identified_method_count": len(method_inquiry.required_methods),
                "retrieved_method_count": len(retrieved_method_contexts),
                "missing_method_count": len(missing_method_names),
                "repair_evidence_mode": "strong" if retrieved_method_contexts else "weak",
                "retrieved_test_snippets_count": 0,
                "retrieved_callsite_snippets_count": 0,
                "retrieved_history_snippets_count": 0,
            }
        )
        if "github_evidence_strength" not in metadata:
            metadata["github_evidence_strength"] = "medium" if retrieved_method_contexts else "low"
        task_id = str(updated.get("task_id") or "")
        if task_id:
            self._persist_context_cache(task_id, updated)
        return updated

    def _context_cache_dir(self) -> Path:
        if self._current_output_dir is None:
            raise RuntimeError("Output directory is not set for context caching.")
        return self._current_output_dir / "context_cache"

    def _repair_debug_dir(self) -> Path:
        if self._current_output_dir is None:
            raise RuntimeError("Output directory is not set for repair debug output.")
        return self._current_output_dir / "repair_debug"

    def _context_cache_file(self, task_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", task_id)
        return self._context_cache_dir() / f"{safe}.json"

    def _load_context_cache(self, task_id: str) -> dict | None:
        path = self._context_cache_file(task_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _persist_context_cache(self, task_id: str, bundle: dict[str, object]) -> None:
        path = self._context_cache_file(task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    def _repair_debug_file(self, task_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", task_id)
        return self._repair_debug_dir() / f"{safe}.json"

    def _record_repair_debug_checkpoint(self, state: GraphState | dict, stage: str, payload: dict) -> None:
        if self._current_output_dir is None:
            return
        task_id = str(state.get("task_id") or "")
        if not task_id:
            return
        path = self._repair_debug_file(task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        current: dict[str, object]
        if path.exists():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current = {}
        else:
            current = {}
        current.update(
            {
                "task_id": task_id,
                "task_index": state.get("task_index"),
                "task_total": state.get("task_total"),
                "project": state.get("project"),
                "user": state.get("user"),
                "file_path": state.get("file_path"),
                "commit": state.get("commit"),
                "satd_comment": state.get("satd_comment"),
                "latest_stage": stage,
                "updated_at": self.context_client._timestamp(),
            }
        )
        stages = current.get("stages")
        if not isinstance(stages, dict):
            stages = {}
        stages[stage] = payload
        current["stages"] = stages
        path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")

    def _fallback_analysis(self, reason: str = "analyzer_fallback") -> AnalysisResult:
        return AnalysisResult(
            decision="drop",
            repairable=False,
            repairability_score=0.0,
            intent_clarity=0.0,
            change_locality=0.0,
            semantic_risk=1.0,
            context_sufficiency=0.0,
            verifiability=0.0,
            analyze_score=0.0,
            confidence=0.0,
            satd_type="content_filtered",
            reason=reason,
            evidence_summary=f"Analyzer fallback triggered: {reason}",
            risk_level="medium",
            context_score=0.0,
            clarity_score=0.0,
            scope_radius="file",
            operation_concrete="low",
            localizable="low",
            local_scope="low",
            end_state_clear="low",
            comment_evidence="",
            code_evidence="analyzer fallback",
            validation_signals=[reason],
            context_gaps=[reason],
            followup_context_requests=[],
            repair_strategy="Do not attempt automatic repair.",
            historical_snapshot_mismatch=False,
            github_evidence_strength="low",
        )

    def _bypass_analysis(self, state: GraphState, satd_route_type: str) -> AnalysisResult:
        scope_radius = "function" if "def " in (state.get("original_code") or "") or "class " in (state.get("original_code") or "") else "file"
        return AnalysisResult(
            decision="pass",
            repairable=True,
            repairability_score=0.75,
            intent_clarity=0.70,
            change_locality=0.75,
            semantic_risk=0.35,
            context_sufficiency=0.65,
            verifiability=0.60,
            analyze_score=0.72,
            confidence=0.60,
            satd_type=satd_route_type or "generic",
            reason="Analyzer disabled; using direct pass-through analysis.",
            evidence_summary="Analyzer disabled; SATD passed directly to the next stage.",
            risk_level="medium",
            context_score=0.65,
            clarity_score=0.70,
            scope_radius=scope_radius,
            operation_concrete="high",
            localizable="high",
            local_scope="high",
            end_state_clear="high",
            comment_evidence=(state.get("satd_comment") or "").strip()[:80],
            code_evidence="pass-through analyzer disabled",
            validation_signals=["analyzer_bypassed"],
            context_gaps=[],
            followup_context_requests=[],
            repair_strategy="Pass this item to the next stage.",
            historical_snapshot_mismatch=False,
            github_evidence_strength="low",
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
            review_score=0.0,
            problem_alignment=0.0,
            minimality=0.0,
            semantic_preservation=0.0,
            internal_consistency=0.0,
            issues=[f"Reviewer fallback triggered: {reason}"],
            revision_advice=f"Stop automatic approval for this SATD because reviewer request failed ({reason}).",
            reject_type="content_filter" if "content_filter" in reason else "review_error",
            rationale=f"Reviewer fallback triggered: {reason}",
            candidate_mode=candidate_mode,
        )

    def _bypass_review(self, state: GraphState) -> ReviewResult:
        candidate_mode = getattr(state.get("latest_repair"), "candidate_mode", "single") if isinstance(state, dict) else "single"
        return ReviewResult(
            round_id=state["round_id"],
            approved=True,
            review_score=1.0,
            problem_alignment=1.0,
            minimality=1.0,
            semantic_preservation=1.0,
            internal_consistency=1.0,
            issues=[],
            revision_advice="Reviewer disabled; accepting fixer output directly.",
            reject_type=None,
            rationale="Reviewer disabled in fixer-only experiment.",
            softened_gate_used=False,
            candidate_mode=candidate_mode,
        )

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







