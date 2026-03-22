from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .bootstrap import bootstrap_vendor

bootstrap_vendor()

from langgraph.graph import END, START, StateGraph

from .agents import OpenAIAnalyzer, OpenAICompatClient, OpenAIFixer, OpenAIReviewer
from .csv_loader import load_satd_csv
from .schema import AnalysisResult, GraphState, RepairAttempt, ReviewResult, SATDRecord, preprocess_python_code, record_to_graph_input, trace_from_state


class LangGraphSATDWorkflow:
    def __init__(
        self,
        max_rounds: int = 2,
        model: str = "gpt-4o-mini",
        verbose: bool = False,
        write_batch_size: int = 10,
    ) -> None:
        self.max_rounds = max_rounds
        self.model = model
        self.verbose = verbose
        self.write_batch_size = write_batch_size
        client = OpenAICompatClient(model=model)
        self.context_client = client
        self.analyzer = OpenAIAnalyzer(client)
        self.fixer = OpenAIFixer(client)
        self.reviewer = OpenAIReviewer(client)
        self.graph = self._build_graph()
        self._current_output_dir: Path | None = None

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("analyze", self._analyze_node)
        graph.add_node("repair", self._repair_node)
        graph.add_node("review", self._review_node)
        graph.add_node("accept", self._accept_node)
        graph.add_node("drop", self._drop_node)
        graph.add_edge(START, "analyze")
        graph.add_conditional_edges("analyze", self._route_after_analysis, {"repair": "repair", "drop": "drop"})
        graph.add_edge("repair", "review")
        graph.add_conditional_edges("review", self._route_after_review, {"accept": "accept", "repair": "repair", "drop": "drop"})
        graph.add_edge("accept", END)
        graph.add_edge("drop", END)
        return graph.compile()

    def _analyze_node(self, state: GraphState) -> dict:
        self._log(f"[task {state['task_id']}] analyze start")
        context_bundle = state.get("github_context") or self._load_or_build_base_context(state)
        try:
            analysis = self.analyzer.run({**state, "github_context": context_bundle})
        except Exception as exc:
            if self.context_client._is_content_filter_error(exc):
                self._log(f"[task {state['task_id']}] analyze content-filtered; using fallback drop")
                analysis = self._fallback_analysis(context_bundle)
            else:
                raise
        self._log(
            f"[task {state['task_id']}] analyze done decision={analysis.decision} "
            f"repairable={analysis.repairable} score={analysis.repairability_score:.2f} "
            f"mismatch={analysis.historical_snapshot_mismatch} github={analysis.github_evidence_strength}"
        )
        return {"github_context": context_bundle, "analysis": analysis, "status": "repairable" if analysis.repairable else "dropped_by_analyzer"}

    def _repair_node(self, state: GraphState) -> dict:
        next_round = state["round_id"] + 1
        self._log(f"[task {state['task_id']}] repair start round={next_round}")
        context_bundle = self._ensure_repair_context_cache(state)
        try:
            repair = self.fixer.run({**state, "github_context": context_bundle})
        except Exception as exc:
            if self.context_client._is_content_filter_error(exc):
                self._log(f"[task {state['task_id']}] repair content-filtered; using fallback no-op repair")
                repair = self._fallback_repair(state, next_round)
            else:
                raise
        self._log(f"[task {state['task_id']}] repair done round={repair.round_id} scope={repair.changed_scope} conf={repair.confidence:.2f}")
        return {
            "github_context": context_bundle,
            "repair_context_used": bool((context_bundle.get("metadata") or {}).get("repair_cached")),
            "status": "repairing",
            "round_id": repair.round_id,
            "latest_repair": repair,
            "repairs": [*state["repairs"], repair],
        }

    def _review_node(self, state: GraphState) -> dict:
        self._log(f"[task {state['task_id']}] review start round={state['round_id']}")
        assert state["analysis"] is not None
        assert state["latest_repair"] is not None
        context_bundle = self._ensure_review_context_cache(state)
        try:
            review = self.reviewer.run({**state, "github_context": context_bundle})
        except Exception as exc:
            if self.context_client._is_content_filter_error(exc):
                self._log(f"[task {state['task_id']}] review content-filtered; using fallback reject")
                review = self._fallback_review(state)
            else:
                raise
        self._log(f"[task {state['task_id']}] review done round={review.round_id} approved={review.approved} score={review.review_score:.2f}")
        return {
            "github_context": context_bundle,
            "review_strict_gate_result": "approved" if review.approved else "rejected",
            "status": "accepted" if review.approved else "review_failed",
            "latest_review": review,
            "reviews": [*state["reviews"], review],
            "final_repaired_code": state["latest_repair"].repaired_code if review.approved else state["final_repaired_code"],
        }

    def _accept_node(self, state: GraphState) -> dict:
        self._log(f"[task {state['task_id']}] accepted after rounds={state['round_id']}")
        return {
            "status": "accepted",
            "review_strict_gate_result": state.get("review_strict_gate_result") or "approved",
            "final_repaired_code": state["latest_repair"].repaired_code if state["latest_repair"] else None,
        }

    def _drop_node(self, state: GraphState) -> dict:
        if state["analysis"] and not state["analysis"].repairable:
            self._log(f"[task {state['task_id']}] dropped by analyzer reason={state['analysis'].drop_reason}")
            return {"status": "dropped_by_analyzer"}
        self._log(f"[task {state['task_id']}] dropped after review rounds={state['round_id']}")
        return {"status": "dropped_after_review", "review_strict_gate_result": state.get("review_strict_gate_result") or "rejected"}

    def _route_after_analysis(self, state: GraphState) -> str:
        analysis = state["analysis"]
        if analysis is None or not analysis.repairable:
            return "drop"
        return "repair"

    def _route_after_review(self, state: GraphState) -> str:
        latest_review = state["latest_review"]
        if latest_review is None:
            return "drop"
        if latest_review.approved:
            return "accept"
        if state["round_id"] < state["max_rounds"]:
            return "repair"
        return "drop"

    def run_record(self, record: SATDRecord):
        self._log(f"[task {record.task_id}] start project={record.project} file={record.file_path}")
        initial_state = record_to_graph_input(record, self.max_rounds)
        initial_state["github_context"] = self._load_or_build_base_context(initial_state)
        final_state = self.graph.invoke(initial_state)
        self._log(f"[task {record.task_id}] end status={final_state['status']}")
        return trace_from_state(final_state, record.em_label)

    def run_csv(self, input_path: Path, output_dir: Path, limit: int | None = None, resume: bool = False) -> dict:
        records = load_satd_csv(input_path, limit=limit)
        total = len(records)
        summary: dict | None = None
        output_dir.mkdir(parents=True, exist_ok=True)
        self._current_output_dir = output_dir
        self._context_cache_dir().mkdir(parents=True, exist_ok=True)

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
            trace = self.run_record(record)
            new_traces.append(trace)
            pending_flush_traces.append(trace)
            if self.verbose:
                print(f"[progress] {overall_index}/{total} task_id={record.task_id} status={trace.status} rounds={trace.rounds_used} exact={trace.exact_match}")

            if offset % self.write_batch_size == 0 or offset == len(pending_records):
                summary = self._summarize_from_existing_and_new(existing_rows, new_traces)
                summary["agent_mode"] = "openai"
                summary["model"] = self.model
                summary["max_rounds"] = self.max_rounds
                summary["written_tasks"] = len(completed_ids) + len(new_traces)
                summary["write_batch_size"] = self.write_batch_size
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
            summary["agent_mode"] = "openai"
            summary["model"] = self.model
            summary["max_rounds"] = self.max_rounds
            summary["written_tasks"] = len(completed_ids)
            summary["write_batch_size"] = self.write_batch_size
            self._write_summary_csv(output_dir / "summary.csv", summary)

        assert summary is not None
        return summary

    def _summarize(self, traces: list) -> dict:
        total = len(traces)
        analyze_filtered_count = sum(1 for trace in traces if trace.status == "dropped_by_analyzer")
        review_rejected_count = sum(1 for trace in traces if trace.status == "dropped_after_review")
        workflow_output_count = sum(1 for trace in traces if trace.status == "accepted")
        successful_repair_count = sum(1 for trace in traces if trace.status == "accepted" and trace.exact_match)

        return {
            "input_satd_count": total,
            "analyze_filtered_count": analyze_filtered_count,
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
        self._write_reviews_csv(output_dir / "reviews.csv", traces)
        self._write_github_context_csv(output_dir / "github_context.csv", traces)
        self._write_context_cache_csv(output_dir / "context_cache.csv", traces)
        self._write_summary_csv(output_dir / "summary.csv", summary)

    def _write_trajectory_overview_csv(self, path: Path, traces: list) -> None:
        fieldnames = [
            "task_id", "project", "file_path", "status", "workflow_output", "drop_stage", "trajectory_summary", "rounds_used", "em_label", "exact_match", "satd_comment",
            "analysis_decision", "analysis_passed", "analysis_repairability_score", "analysis_confidence", "analysis_satd_type", "analysis_risk_level", "analysis_scope_radius",
            "analysis_intent_clarity", "analysis_change_locality", "analysis_semantic_risk", "analysis_context_sufficiency", "analysis_verifiability", "analysis_analyze_score",
            "analysis_context_score", "analysis_clarity_score", "analysis_validation_signals", "analysis_context_gaps", "analysis_followup_context_requests", "analysis_evidence_summary",
            "analysis_repair_strategy", "analysis_drop_reason", "analysis_historical_snapshot_mismatch", "analysis_github_evidence_strength",
            "repair_context_used", "review_strict_gate_result", "original_code", "processed_manual_code", "processed_final_repaired_code",
            "round_1_repair_plan", "round_1_repaired_code", "round_1_changed_scope", "round_1_fix_confidence", "round_1_review_approved", "round_1_review_score", "round_1_review_problem_alignment", "round_1_review_minimality", "round_1_review_semantic_preservation", "round_1_review_internal_consistency", "round_1_review_softened_gate_used", "round_1_review_reject_type", "round_1_review_issues", "round_1_revision_advice",
            "round_2_repair_plan", "round_2_repaired_code", "round_2_changed_scope", "round_2_fix_confidence", "round_2_review_approved", "round_2_review_score", "round_2_review_problem_alignment", "round_2_review_minimality", "round_2_review_semantic_preservation", "round_2_review_internal_consistency", "round_2_review_softened_gate_used", "round_2_review_reject_type", "round_2_review_issues", "round_2_revision_advice",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trace in traces:
                writer.writerow(self._trajectory_row(trace))

    def _trajectory_row(self, trace) -> dict:
        analysis = trace.analysis or {}
        repairs = {item.get("round_id"): item for item in trace.repairs}
        reviews = {item.get("round_id"): item for item in trace.reviews}
        row = {
            "task_id": trace.task_id,
            "project": trace.project,
            "file_path": trace.file_path,
            "status": trace.status,
            "workflow_output": "YES" if trace.status == "accepted" else "NO",
            "drop_stage": self._drop_stage(trace),
            "trajectory_summary": self._trajectory_summary(trace),
            "rounds_used": trace.rounds_used,
            "em_label": trace.em_label,
            "exact_match": trace.exact_match,
            "satd_comment": trace.satd_comment,
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
            "analysis_drop_reason": analysis.get("drop_reason"),
            "analysis_historical_snapshot_mismatch": analysis.get("historical_snapshot_mismatch"),
            "analysis_github_evidence_strength": analysis.get("github_evidence_strength"),
            "repair_context_used": trace.repair_context_used,
            "review_strict_gate_result": trace.review_strict_gate_result,
            "original_code": trace.original_code,
            "processed_manual_code": trace.processed_manual_code,
            "processed_final_repaired_code": trace.processed_final_repaired_code,
        }
        for round_id in (1, 2):
            repair = repairs.get(round_id, {})
            review = reviews.get(round_id, {})
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
            row[f"round_{round_id}_revision_advice"] = review.get("revision_advice")
        return row

    def _trajectory_summary(self, trace) -> str:
        steps = []
        analysis = trace.analysis or {}
        steps.append("analysis:pass" if analysis.get("repairable") else f"analysis:{analysis.get('decision') or 'drop'}")
        if trace.repair_context_used:
            steps.append("repair_context:used")
        for repair in trace.repairs:
            steps.append(f"repair{repair.get('round_id')}")
        for review in trace.reviews:
            outcome = "pass" if review.get("approved") else "reject"
            steps.append(f"review{review.get('round_id')}:{outcome}")
        if trace.status == "accepted":
            steps.append("workflow:output")
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
            "task_id", "project", "file_path", "satd_comment", "status", "rounds_used", "em_label", "exact_match",
            "analysis_decision", "analysis_repairable", "analysis_repairability_score", "analysis_confidence", "analysis_satd_type", "analysis_risk_level", "analysis_scope_radius",
            "analysis_intent_clarity", "analysis_change_locality", "analysis_semantic_risk", "analysis_context_sufficiency", "analysis_verifiability", "analysis_analyze_score",
            "analysis_context_score", "analysis_clarity_score", "analysis_validation_signals", "analysis_context_gaps", "analysis_followup_context_requests", "analysis_evidence_summary",
            "analysis_repair_strategy", "analysis_drop_reason", "analysis_historical_snapshot_mismatch", "analysis_github_evidence_strength",
            "repair_context_used", "review_strict_gate_result", "original_code", "processed_manual_code", "processed_final_repaired_code",
            "round_1_repair_plan", "round_1_repaired_code", "round_1_changed_scope", "round_1_fix_confidence", "round_1_review_approved", "round_1_review_score", "round_1_review_problem_alignment", "round_1_review_minimality", "round_1_review_semantic_preservation", "round_1_review_internal_consistency", "round_1_review_softened_gate_used", "round_1_review_reject_type", "round_1_review_issues", "round_1_revision_advice",
            "round_2_repair_plan", "round_2_repaired_code", "round_2_changed_scope", "round_2_fix_confidence", "round_2_review_approved", "round_2_review_score", "round_2_review_problem_alignment", "round_2_review_minimality", "round_2_review_semantic_preservation", "round_2_review_internal_consistency", "round_2_review_softened_gate_used", "round_2_review_reject_type", "round_2_review_issues", "round_2_revision_advice",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trace in traces:
                analysis = trace.analysis or {}
                writer.writerow({
                    "task_id": trace.task_id,
                    "project": trace.project,
                    "file_path": trace.file_path,
                    "satd_comment": trace.satd_comment,
                    "status": trace.status,
                    "rounds_used": trace.rounds_used,
                    "em_label": trace.em_label,
                    "exact_match": trace.exact_match,
                    "analysis_decision": analysis.get("decision"),
                    "analysis_repairable": analysis.get("repairable"),
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
                    "analysis_validation_signals": json.dumps(analysis.get("validation_signals", []), ensure_ascii=False),
                    "analysis_context_gaps": json.dumps(analysis.get("context_gaps", []), ensure_ascii=False),
                    "analysis_followup_context_requests": json.dumps(analysis.get("followup_context_requests", []), ensure_ascii=False),
                    "analysis_evidence_summary": analysis.get("evidence_summary") or analysis.get("reason"),
                    "analysis_repair_strategy": analysis.get("repair_strategy"),
                    "analysis_drop_reason": analysis.get("drop_reason"),
                    "analysis_historical_snapshot_mismatch": analysis.get("historical_snapshot_mismatch"),
                    "analysis_github_evidence_strength": analysis.get("github_evidence_strength"),
                    "repair_context_used": trace.repair_context_used,
                    "review_strict_gate_result": trace.review_strict_gate_result,
                    "original_code": trace.original_code,
                    "processed_manual_code": trace.processed_manual_code,
                    "processed_final_repaired_code": trace.processed_final_repaired_code,
                })

    def _write_repairs_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "round_id", "repair_plan", "repaired_code", "changed_scope", "confidence", "notes"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trace in traces:
                for repair in trace.repairs:
                    row = {"task_id": trace.task_id, **repair}
                    row["repaired_code"] = preprocess_python_code(repair.get("repaired_code"))
                    writer.writerow(row)

    def _write_reviews_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "round_id", "approved", "review_score", "problem_alignment", "minimality", "semantic_preservation", "internal_consistency", "softened_gate_used", "issues", "revision_advice", "reject_type", "rationale"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trace in traces:
                for review in trace.reviews:
                    row = dict(review)
                    row["issues"] = json.dumps(row.get("issues", []), ensure_ascii=False)
                    writer.writerow({"task_id": trace.task_id, **row})

    def _write_github_context_csv(self, path: Path, traces: list) -> None:
        fieldnames = [
            "task_id", "repo_owner", "repo_name", "file_path", "historical_snapshot_mismatch", "github_evidence_strength", "target_file_ok", "satd_window_found", "enclosing_symbol_found", "symbol_name", "satd_line",
            "related_tests_count", "call_sites_count", "commits_count", "similar_history_count", "base_context_json", "repair_context_json", "review_context_json",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trace in traces:
                context = trace.github_context or {}
                metadata = context.get("metadata", {}) if isinstance(context, dict) else {}
                writer.writerow({
                    "task_id": trace.task_id,
                    "repo_owner": context.get("repo_owner") if isinstance(context, dict) else None,
                    "repo_name": context.get("repo_name") if isinstance(context, dict) else None,
                    "file_path": context.get("file_path") if isinstance(context, dict) else None,
                    "historical_snapshot_mismatch": metadata.get("historical_snapshot_mismatch"),
                    "github_evidence_strength": metadata.get("github_evidence_strength"),
                    "target_file_ok": metadata.get("target_file_ok"),
                    "satd_window_found": metadata.get("satd_window_found"),
                    "enclosing_symbol_found": metadata.get("enclosing_symbol_found"),
                    "symbol_name": metadata.get("symbol_name"),
                    "satd_line": metadata.get("satd_line"),
                    "related_tests_count": metadata.get("related_tests_count"),
                    "call_sites_count": metadata.get("call_sites_count"),
                    "commits_count": metadata.get("commits_count"),
                    "similar_history_count": metadata.get("similar_history_count"),
                    "base_context_json": json.dumps(context.get("base_context", {}), ensure_ascii=False),
                    "repair_context_json": json.dumps(context.get("repair_context", {}), ensure_ascii=False),
                    "review_context_json": json.dumps(context.get("review_context", {}), ensure_ascii=False),
                })

    def _write_context_cache_csv(self, path: Path, traces: list) -> None:
        fieldnames = [
            "task_id", "cache_file", "base_cached", "base_cache_source", "base_context_fetched_at", "repair_cached", "repair_cache_source", "repair_context_fetched_at", "review_cached", "review_cache_source", "review_context_fetched_at",
            "historical_snapshot_mismatch", "github_evidence_strength", "target_file_ok", "satd_window_found", "enclosing_symbol_found", "related_tests_count", "call_sites_count", "commits_count", "similar_history_count",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trace in traces:
                context = trace.github_context or {}
                metadata = context.get("metadata", {}) if isinstance(context, dict) else {}
                writer.writerow({
                    "task_id": trace.task_id,
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
                    "target_file_ok": metadata.get("target_file_ok"),
                    "satd_window_found": metadata.get("satd_window_found"),
                    "enclosing_symbol_found": metadata.get("enclosing_symbol_found"),
                    "related_tests_count": metadata.get("related_tests_count"),
                    "call_sites_count": metadata.get("call_sites_count"),
                    "commits_count": metadata.get("commits_count"),
                    "similar_history_count": metadata.get("similar_history_count"),
                })

    def _append_outputs(self, output_dir: Path, traces: list) -> None:
        self._append_trajectory_overview_csv(output_dir / "trajectory_overview.csv", traces)
        self._append_results_csv(output_dir / "results.csv", traces)
        self._append_repairs_csv(output_dir / "repairs.csv", traces)
        self._append_reviews_csv(output_dir / "reviews.csv", traces)
        self._append_github_context_csv(output_dir / "github_context.csv", traces)
        self._append_context_cache_csv(output_dir / "context_cache.csv", traces)

    def _append_csv_rows(self, path: Path, fieldnames: list[str], rows: list[dict]) -> None:
        if not rows:
            return
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _append_trajectory_overview_csv(self, path: Path, traces: list) -> None:
        fieldnames = [
            "task_id", "project", "file_path", "status", "workflow_output", "drop_stage", "trajectory_summary", "rounds_used", "em_label", "exact_match", "satd_comment",
            "analysis_decision", "analysis_passed", "analysis_repairability_score", "analysis_confidence", "analysis_satd_type", "analysis_risk_level", "analysis_scope_radius",
            "analysis_intent_clarity", "analysis_change_locality", "analysis_semantic_risk", "analysis_context_sufficiency", "analysis_verifiability", "analysis_analyze_score",
            "analysis_context_score", "analysis_clarity_score", "analysis_validation_signals", "analysis_context_gaps", "analysis_followup_context_requests", "analysis_evidence_summary",
            "analysis_repair_strategy", "analysis_drop_reason", "analysis_historical_snapshot_mismatch", "analysis_github_evidence_strength",
            "repair_context_used", "review_strict_gate_result", "original_code", "processed_manual_code", "processed_final_repaired_code",
            "round_1_repair_plan", "round_1_repaired_code", "round_1_changed_scope", "round_1_fix_confidence", "round_1_review_approved", "round_1_review_score", "round_1_review_problem_alignment", "round_1_review_minimality", "round_1_review_semantic_preservation", "round_1_review_internal_consistency", "round_1_review_softened_gate_used", "round_1_review_reject_type", "round_1_review_issues", "round_1_revision_advice",
            "round_2_repair_plan", "round_2_repaired_code", "round_2_changed_scope", "round_2_fix_confidence", "round_2_review_approved", "round_2_review_score", "round_2_review_problem_alignment", "round_2_review_minimality", "round_2_review_semantic_preservation", "round_2_review_internal_consistency", "round_2_review_softened_gate_used", "round_2_review_reject_type", "round_2_review_issues", "round_2_revision_advice",
        ]
        rows = [self._trajectory_row(trace) for trace in traces]
        self._append_csv_rows(path, fieldnames, rows)

    def _append_results_csv(self, path: Path, traces: list) -> None:
        fieldnames = [
            "task_id", "project", "file_path", "satd_comment", "status", "rounds_used", "em_label", "exact_match",
            "analysis_decision", "analysis_repairable", "analysis_repairability_score", "analysis_confidence", "analysis_satd_type", "analysis_risk_level", "analysis_scope_radius",
            "analysis_intent_clarity", "analysis_change_locality", "analysis_semantic_risk", "analysis_context_sufficiency", "analysis_verifiability", "analysis_analyze_score",
            "analysis_context_score", "analysis_clarity_score", "analysis_validation_signals", "analysis_context_gaps", "analysis_followup_context_requests", "analysis_evidence_summary",
            "analysis_repair_strategy", "analysis_drop_reason", "analysis_historical_snapshot_mismatch", "analysis_github_evidence_strength",
            "repair_context_used", "review_strict_gate_result", "original_code", "processed_manual_code", "processed_final_repaired_code",
        ]
        rows = []
        for trace in traces:
            analysis = trace.analysis or {}
            rows.append({
                "task_id": trace.task_id,
                "project": trace.project,
                "file_path": trace.file_path,
                "satd_comment": trace.satd_comment,
                "status": trace.status,
                "rounds_used": trace.rounds_used,
                "em_label": trace.em_label,
                "exact_match": trace.exact_match,
                "analysis_decision": analysis.get("decision"),
                "analysis_repairable": analysis.get("repairable"),
                "analysis_repairability_score": analysis.get("repairability_score"),
                "analysis_confidence": analysis.get("confidence"),
                "analysis_satd_type": analysis.get("satd_type"),
                "analysis_risk_level": analysis.get("risk_level"),
                "analysis_scope_radius": analysis.get("scope_radius"),
                "analysis_context_score": analysis.get("context_score"),
                "analysis_clarity_score": analysis.get("clarity_score"),
                "analysis_validation_signals": json.dumps(analysis.get("validation_signals", []), ensure_ascii=False),
                "analysis_context_gaps": json.dumps(analysis.get("context_gaps", []), ensure_ascii=False),
                "analysis_followup_context_requests": json.dumps(analysis.get("followup_context_requests", []), ensure_ascii=False),
                "analysis_evidence_summary": analysis.get("evidence_summary") or analysis.get("reason"),
                "analysis_repair_strategy": analysis.get("repair_strategy"),
                "analysis_drop_reason": analysis.get("drop_reason"),
                "analysis_historical_snapshot_mismatch": analysis.get("historical_snapshot_mismatch"),
                "analysis_github_evidence_strength": analysis.get("github_evidence_strength"),
                "repair_context_used": trace.repair_context_used,
                "review_strict_gate_result": trace.review_strict_gate_result,
                "original_code": trace.original_code,
                "processed_manual_code": trace.processed_manual_code,
                "processed_final_repaired_code": trace.processed_final_repaired_code,
            })
        self._append_csv_rows(path, fieldnames, rows)

    def _append_repairs_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "round_id", "repair_plan", "repaired_code", "changed_scope", "confidence", "notes"]
        rows = []
        for trace in traces:
            for repair in trace.repairs:
                row = {"task_id": trace.task_id, **repair}
                row["repaired_code"] = preprocess_python_code(repair.get("repaired_code"))
                rows.append(row)
        self._append_csv_rows(path, fieldnames, rows)

    def _append_reviews_csv(self, path: Path, traces: list) -> None:
        fieldnames = ["task_id", "round_id", "approved", "review_score", "problem_alignment", "minimality", "semantic_preservation", "internal_consistency", "softened_gate_used", "issues", "revision_advice", "reject_type", "rationale"]
        rows = []
        for trace in traces:
            for review in trace.reviews:
                row = dict(review)
                row["issues"] = json.dumps(row.get("issues", []), ensure_ascii=False)
                rows.append({"task_id": trace.task_id, **row})
        self._append_csv_rows(path, fieldnames, rows)

    def _append_github_context_csv(self, path: Path, traces: list) -> None:
        fieldnames = [
            "task_id", "repo_owner", "repo_name", "file_path", "historical_snapshot_mismatch", "github_evidence_strength", "target_file_ok", "satd_window_found", "enclosing_symbol_found", "symbol_name", "satd_line",
            "related_tests_count", "call_sites_count", "commits_count", "similar_history_count", "base_context_json", "repair_context_json", "review_context_json",
        ]
        rows = []
        for trace in traces:
            context = trace.github_context or {}
            metadata = context.get("metadata", {}) if isinstance(context, dict) else {}
            rows.append({
                "task_id": trace.task_id,
                "repo_owner": context.get("repo_owner") if isinstance(context, dict) else None,
                "repo_name": context.get("repo_name") if isinstance(context, dict) else None,
                "file_path": context.get("file_path") if isinstance(context, dict) else None,
                "historical_snapshot_mismatch": metadata.get("historical_snapshot_mismatch"),
                "github_evidence_strength": metadata.get("github_evidence_strength"),
                "target_file_ok": metadata.get("target_file_ok"),
                "satd_window_found": metadata.get("satd_window_found"),
                "enclosing_symbol_found": metadata.get("enclosing_symbol_found"),
                "symbol_name": metadata.get("symbol_name"),
                "satd_line": metadata.get("satd_line"),
                "related_tests_count": metadata.get("related_tests_count"),
                "call_sites_count": metadata.get("call_sites_count"),
                "commits_count": metadata.get("commits_count"),
                "similar_history_count": metadata.get("similar_history_count"),
                "base_context_json": json.dumps(context.get("base_context", {}), ensure_ascii=False),
                "repair_context_json": json.dumps(context.get("repair_context", {}), ensure_ascii=False),
                "review_context_json": json.dumps(context.get("review_context", {}), ensure_ascii=False),
            })
        self._append_csv_rows(path, fieldnames, rows)

    def _append_context_cache_csv(self, path: Path, traces: list) -> None:
        fieldnames = [
            "task_id", "cache_file", "base_cached", "base_cache_source", "base_context_fetched_at", "repair_cached", "repair_cache_source", "repair_context_fetched_at", "review_cached", "review_cache_source", "review_context_fetched_at",
            "historical_snapshot_mismatch", "github_evidence_strength", "target_file_ok", "satd_window_found", "enclosing_symbol_found", "related_tests_count", "call_sites_count", "commits_count", "similar_history_count",
        ]
        rows = []
        for trace in traces:
            context = trace.github_context or {}
            metadata = context.get("metadata", {}) if isinstance(context, dict) else {}
            rows.append({
                "task_id": trace.task_id,
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
                "target_file_ok": metadata.get("target_file_ok"),
                "satd_window_found": metadata.get("satd_window_found"),
                "enclosing_symbol_found": metadata.get("enclosing_symbol_found"),
                "related_tests_count": metadata.get("related_tests_count"),
                "call_sites_count": metadata.get("call_sites_count"),
                "commits_count": metadata.get("commits_count"),
                "similar_history_count": metadata.get("similar_history_count"),
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
        successful_repair_count = sum(
            1
            for row in result_rows
            if row.get("status") == "accepted" and str(row.get("exact_match")).lower() == "true"
        )

        return {
            "input_satd_count": total,
            "analyze_filtered_count": analyze_filtered_count,
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

    def _ensure_repair_context_cache(self, state: GraphState) -> dict:
        bundle = state.get("github_context") or self._load_or_build_base_context(state)
        before = bool((bundle.get("metadata") or {}).get("repair_cached"))
        enriched = self.context_client.ensure_repair_context(state, bundle)
        if not before:
            enriched.setdefault("metadata", {})["repair_cache_source"] = "github_fetch"
        self._persist_context_cache(state["task_id"], enriched)
        return enriched

    def _ensure_review_context_cache(self, state: GraphState) -> dict:
        bundle = state.get("github_context") or self._ensure_repair_context_cache(state)
        enriched = self.context_client.ensure_review_context(state, bundle, state["analysis"], state["latest_repair"])
        self._persist_context_cache(state["task_id"], enriched)
        return enriched

    def _context_cache_dir(self) -> Path:
        if self._current_output_dir is None:
            raise RuntimeError("Output directory is not set for context caching.")
        return self._current_output_dir / "context_cache"

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

    def _fallback_analysis(self, context_bundle: dict) -> AnalysisResult:
        metadata = context_bundle.get("metadata", {}) if isinstance(context_bundle, dict) else {}
        return AnalysisResult(
            decision="drop",
            repairable=False,
            repairability_score=0.0,
            confidence=0.0,
            satd_type="content_filtered",
            reason="Analyzer prompt was blocked by provider content filtering.",
            evidence_summary="Analyzer prompt was blocked by provider content filtering.",
            risk_level="medium",
            context_score=0.0,
            clarity_score=0.0,
            scope_radius="file",
            validation_signals=["content_filter_fallback"],
            context_gaps=["provider_content_filter"],
            followup_context_requests=[],
            repair_strategy="Do not attempt automatic repair.",
            drop_reason="insufficient_context",
            historical_snapshot_mismatch=bool(metadata.get("historical_snapshot_mismatch")),
            github_evidence_strength=str(metadata.get("github_evidence_strength") or "low"),
        )

    def _fallback_repair(self, state: GraphState, round_id: int) -> RepairAttempt:
        return RepairAttempt(
            round_id=round_id,
            repair_plan="Fallback no-op repair because the fixer prompt was blocked by provider content filtering.",
            repaired_code=state["original_code"],
            changed_scope="none",
            confidence=0.0,
            notes="fixer_content_filter_fallback",
        )

    def _fallback_review(self, state: GraphState) -> ReviewResult:
        return ReviewResult(
            round_id=state["round_id"],
            approved=False,
            review_score=0.0,
            issues=["Reviewer prompt was blocked by provider content filtering."],
            revision_advice="Stop automatic approval for this SATD because reviewer prompting was content-filtered.",
            reject_type="content_filter",
            rationale="Reviewer prompt was blocked by provider content filtering.",
        )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)






