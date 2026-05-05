from __future__ import annotations

import argparse
import csv
from pathlib import Path

from satd_langgraph.llm_judge import LLMRepairJudge
from satd_langgraph.openai_client import OpenAICompatClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill LLM-as-judge scores for baseline repair metrics.")
    parser.add_argument("--input", type=Path, default=Path("baseline_code_metrics.csv"))
    parser.add_argument("--existing", type=Path, default=Path("baseline_code_metrics_llm_judge_500_manual_semantic.csv"))
    parser.add_argument("--output", type=Path, default=Path("baseline_code_metrics_llm_judge_1000_manual_semantic.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("baseline_code_metrics_llm_judge_1000_manual_semantic_summary.csv"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--candidate-column", default="gpt_clean")
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, str]) -> str:
    return str(row.get("index") or row.get("task_id") or "").strip()


def to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def average(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def write_summary(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    judged = [to_float(row.get("LLM_as_judge")) for row in rows]
    judged_values = [value for value in judged if value is not None]
    pass_count = sum(1 for value in judged_values if value >= 0.5)
    em_count = sum(1 for row in rows if str(row.get("EM") or "").strip().upper() == "YES")
    calculated_em_count = sum(
        1
        for row in rows
        if str(row.get("calculated_exact_match") or "").strip().lower() in {"true", "1", "yes"}
        or str(row.get("calculated_EM") or "").strip().upper() == "YES"
    )
    non_empty_count = sum(1 for row in rows if str(row.get(args.candidate_column) or "").strip())
    numeric_columns = ["BLEU_diff", "CrystalBLEU_diff", "LEMOD"]
    summary: dict[str, object] = {
        "baseline_name": "gpt_clean",
        "input_path": str(args.input),
        "rows": len(rows),
        "EM_count": em_count,
        "EM": round(em_count / len(rows), 6) if rows else 0.0,
        "avg_LLM_as_judge": average(judged_values),
        "llm_judge_count": len(judged_values),
        "llm_judge_pass_count": pass_count,
        "llm_judge_fail_count": len(judged_values) - pass_count,
        "llm_judge_pass_rate": round(pass_count / len(judged_values), 6) if judged_values else 0.0,
        "llm_judge_enabled": True,
        "judge_model": args.model,
        "calculated_EM_count": calculated_em_count,
        "calculated_EM": round(calculated_em_count / len(rows), 6) if rows else 0.0,
        "non_empty_model_output_count": non_empty_count,
    }
    for column in numeric_columns:
        values = [value for value in (to_float(row.get(column)) for row in rows) if value is not None]
        summary[f"avg_{column}"] = average(values)

    fieldnames = [
        "baseline_name",
        "input_path",
        "rows",
        "EM_count",
        "EM",
        "avg_BLEU_diff",
        "avg_CrystalBLEU_diff",
        "avg_LEMOD",
        "avg_LLM_as_judge",
        "llm_judge_count",
        "llm_judge_pass_count",
        "llm_judge_fail_count",
        "llm_judge_pass_rate",
        "llm_judge_enabled",
        "judge_model",
        "calculated_EM_count",
        "calculated_EM",
        "non_empty_model_output_count",
    ]
    write_rows(path, [summary], fieldnames)


def main() -> None:
    args = parse_args()
    base_rows = read_rows(args.input)
    existing_rows = {row_key(row): row for row in read_rows(args.existing) if row_key(row)}
    output_rows = {row_key(row): row for row in read_rows(args.output) if row_key(row)}
    cached_rows = {**existing_rows, **output_rows}

    fieldnames = list(base_rows[0].keys()) if base_rows else []
    if "LLM_as_judge" not in fieldnames:
        fieldnames.append("LLM_as_judge")

    merged_rows: list[dict[str, object]] = []
    for row in base_rows:
        merged = dict(row)
        cached = cached_rows.get(row_key(row))
        if cached and cached.get("LLM_as_judge") not in (None, ""):
            merged["LLM_as_judge"] = cached.get("LLM_as_judge")
        else:
            merged["LLM_as_judge"] = ""
        merged_rows.append(merged)

    client = OpenAICompatClient(model=args.model, verbose=args.verbose)
    judge = LLMRepairJudge(client)

    pending_indexes = [i for i, row in enumerate(merged_rows) if row.get("LLM_as_judge") in (None, "")]
    for offset, row_index in enumerate(pending_indexes, start=1):
        row = merged_rows[row_index]
        exact_match = str(row.get("calculated_exact_match") or "").strip().lower() in {"true", "1", "yes"}
        if exact_match:
            score = 1.0
        else:
            result = judge.judge(
                original_code=str(row.get("original_code") or ""),
                manual_code=str(row.get("manual_code") or ""),
                candidate_code=str(row.get(args.candidate_column) or ""),
                satd_comment=str(row.get("SATD_comment") or ""),
            )
            score = result.llm_as_judge
        row["LLM_as_judge"] = score
        print(f"[judge] {offset}/{len(pending_indexes)} index={row.get('index')} score={score}", flush=True)
        if offset % args.flush_every == 0:
            write_rows(args.output, merged_rows, fieldnames)
            write_summary(args.summary_output, merged_rows, args)

    write_rows(args.output, merged_rows, fieldnames)
    write_summary(args.summary_output, merged_rows, args)
    judged_count = sum(1 for row in merged_rows if row.get("LLM_as_judge") not in (None, ""))
    pass_count = sum(1 for row in merged_rows if (to_float(row.get("LLM_as_judge")) or 0.0) >= 0.5)
    print(f"[done] rows={len(merged_rows)} judged={judged_count} pass={pass_count} rate={pass_count / judged_count:.6f}")


if __name__ == "__main__":
    main()
