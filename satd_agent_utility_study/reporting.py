from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .io_utils import read_csv_rows, write_csv_rows


def build_reports(output_dir: Path, baseline_variant: str = "current_agent") -> None:
    rows = _collect_variant_rows(output_dir)
    write_csv_rows(output_dir / "agent_variant_results.csv", rows)
    summary_rows = _build_summary_rows(rows, baseline_variant=baseline_variant)
    write_csv_rows(output_dir / "context_utility_results.csv", summary_rows)
    write_csv_rows(output_dir / "dev_holdout_summary.csv", summary_rows)
    write_csv_rows(output_dir / "per_sample_decision_trace.csv", _decision_trace_rows(rows))
    write_csv_rows(output_dir / "failure_after_context.csv", _failure_rows(rows))
    _write_recommendation(output_dir, summary_rows)


def select_top_variants(output_dir: Path, phase: str, *, baseline_variant: str, top_k: int) -> list[str]:
    rows = _collect_variant_rows(output_dir)
    summaries = [row for row in _build_summary_rows(rows, baseline_variant=baseline_variant) if row["phase"] == phase]
    candidates = [row for row in summaries if row["variant"] != baseline_variant]
    candidates.sort(
        key=lambda row: (
            int(row["delta_exact_match_vs_baseline"]),
            -int(row["baseline_success_regressions"]),
            float(row["exact_match_rate"]),
            row["variant"],
        ),
        reverse=True,
    )
    return [row["variant"] for row in candidates[: max(0, top_k)]]


def _collect_variant_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return rows
    for phase_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        for variant_dir in sorted(path for path in phase_dir.iterdir() if path.is_dir()):
            results_path = variant_dir / "results.csv"
            if not results_path.exists():
                continue
            for row in read_csv_rows(results_path):
                rows.append(
                    {
                        "phase": phase_dir.name,
                        "variant": variant_dir.name,
                        "task_id": row.get("task_id", ""),
                        "em_label": row.get("em_label", ""),
                        "exact_match": _as_bool(row.get("exact_match")),
                        "status": row.get("status", ""),
                        "analysis_intent_type": row.get("analysis_intent_type", ""),
                        "analysis_evidence_requirement": row.get("analysis_evidence_requirement", ""),
                        "repair_evidence_mode": row.get("repair_evidence_mode", ""),
                        "retrieved_method_count": _as_int(row.get("retrieved_method_count")),
                        "repository_evidence_count": _as_int(row.get("repository_evidence_count")),
                        "context_route": row.get("context_route", ""),
                        "context_required": _as_bool(row.get("context_required")),
                    }
                )
    return rows


def _build_summary_rows(rows: list[dict[str, Any]], baseline_variant: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_phase_task_variant: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        grouped[(row["phase"], row["variant"])].append(row)
        by_phase_task_variant[(row["phase"], row["task_id"], row["variant"])] = row

    summaries: list[dict[str, Any]] = []
    for (phase, variant), items in sorted(grouped.items()):
        baseline_rows = {
            task_id: row
            for (phase_key, task_id, variant_key), row in by_phase_task_variant.items()
            if phase_key == phase and variant_key == baseline_variant
        }
        exact = sum(1 for item in items if item["exact_match"])
        failure_rescues = 0
        baseline_success_regressions = 0
        for item in items:
            baseline = baseline_rows.get(item["task_id"])
            if baseline is None:
                continue
            if not baseline["exact_match"] and item["exact_match"]:
                failure_rescues += 1
            if baseline["exact_match"] and not item["exact_match"]:
                baseline_success_regressions += 1
        baseline_exact = sum(1 for item in baseline_rows.values() if item["exact_match"])
        summaries.append(
            {
                "phase": phase,
                "variant": variant,
                "sample_count": len(items),
                "exact_match_count": exact,
                "exact_match_rate": round(exact / len(items), 4) if items else 0.0,
                "baseline_exact_match_count": baseline_exact,
                "delta_exact_match_vs_baseline": exact - baseline_exact,
                "failure_rescues_vs_baseline": failure_rescues,
                "baseline_success_regressions": baseline_success_regressions,
                "avg_repository_evidence_count": round(
                    sum(item["repository_evidence_count"] for item in items) / len(items), 4
                )
                if items
                else 0.0,
            }
        )
    return summaries


def _decision_trace_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "phase": row["phase"],
            "variant": row["variant"],
            "task_id": row["task_id"],
            "context_route": row["context_route"],
            "context_required": row["context_required"],
            "analysis_intent_type": row["analysis_intent_type"],
            "analysis_evidence_requirement": row["analysis_evidence_requirement"],
            "repair_evidence_mode": row["repair_evidence_mode"],
            "retrieved_method_count": row["retrieved_method_count"],
            "repository_evidence_count": row["repository_evidence_count"],
            "exact_match": row["exact_match"],
        }
        for row in rows
    ]


def _failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not row["exact_match"] and (row["repository_evidence_count"] > 0 or row["repair_evidence_mode"] != "snippet_only")
    ]


def _write_recommendation(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Agent Utility Study Recommendation",
        "",
        "This report is generated from real agent runs, not oracle-only explanation labels.",
        "",
        "## Variant ranking",
    ]
    for phase in ("dev", "holdout", "full"):
        phase_rows = [row for row in summaries if row["phase"] == phase]
        if not phase_rows:
            continue
        lines.append("")
        lines.append(f"### {phase}")
        for row in sorted(
            phase_rows,
            key=lambda item: (int(item["delta_exact_match_vs_baseline"]), -int(item["baseline_success_regressions"])),
            reverse=True,
        ):
            lines.append(
                f"- {row['variant']}: exact={row['exact_match_count']}/{row['sample_count']}, "
                f"delta_vs_baseline={row['delta_exact_match_vs_baseline']}, "
                f"rescues={row['failure_rescues_vs_baseline']}, regressions={row['baseline_success_regressions']}"
            )
    lines.extend(
        [
            "",
            "## Reading rule",
            "",
            "- Prefer variants that improve exact match on holdout while keeping baseline-success regressions near zero.",
            "- Treat repository evidence as useful only when it increases agent EM, not merely when it explains the human patch.",
        ]
    )
    (output_dir / "agent_strategy_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _as_int(value: Any) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0

