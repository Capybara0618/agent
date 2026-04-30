from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from satd_langgraph.repair_metrics import (
    build_metric_context,
    calculate_repair_metrics,
    metric_result_to_row,
)
from satd_langgraph.schema import preprocess_python_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate baseline repair metrics for a code CSV file.")
    parser.add_argument("--input", "-i", type=Path, default=Path("code.csv"), help="Input CSV path.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("baseline_code_metrics.csv"),
        help="Per-row metric output CSV path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("baseline_code_metrics_summary.csv"),
        help="Summary output CSV path.",
    )
    parser.add_argument("--source-col", default="original_code", help="Original/source code column.")
    parser.add_argument("--manual-col", default="manual_code", help="Manual repair/reference code column.")
    parser.add_argument("--model-col", default="gpt_clean", help="Model repair/baseline code column.")
    parser.add_argument("--em-col", default="EM", help="Existing EM label column.")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding.")
    return parser.parse_args()


def text_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def mean(series: pd.Series) -> float:
    return round(float(series.fillna(0).mean()), 6) if len(series) else 0.0


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, encoding=args.encoding)
    for col in [args.source_col, args.manual_col, args.model_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    metric_context = build_metric_context(text_value(value) for value in df[args.manual_col])

    metric_rows: list[dict] = []
    exact_matches: list[bool] = []
    for _, row in df.iterrows():
        source_code = text_value(row[args.source_col])
        manual_code = text_value(row[args.manual_col])
        model_code = text_value(row[args.model_col])
        metrics = calculate_repair_metrics(source_code, manual_code, model_code, metric_context)
        metric_rows.append(metric_result_to_row(metrics))
        exact_matches.append(preprocess_python_code(model_code) == preprocess_python_code(manual_code))

    metric_df = pd.DataFrame(metric_rows)
    result = df.copy()
    result["baseline_name"] = args.model_col
    result["calculated_exact_match"] = exact_matches
    result["calculated_EM"] = ["YES" if item else "NO" for item in exact_matches]
    for col in metric_df.columns:
        result[col] = metric_df[col]

    em_label = result[args.em_col].astype(str).str.upper().eq("YES") if args.em_col in result.columns else result["calculated_exact_match"]
    calculated_em = result["calculated_exact_match"].astype(bool)
    accepted = result[args.model_col].fillna("").astype(str).str.strip().ne("")

    summary_rows = [
        {
            "baseline_name": args.model_col,
            "input_path": str(args.input),
            "rows": len(result),
            "EM_count": int(em_label.sum()),
            "EM": round(float(em_label.mean()), 6) if len(result) else 0.0,
            "avg_BLEU_diff": mean(result["BLEU_diff"]),
            "avg_CrystalBLEU_diff": mean(result["CrystalBLEU_diff"]),
            "avg_LEMOD": mean(result["LEMOD"]),
            "calculated_EM_count": int(calculated_em.sum()),
            "calculated_EM": round(float(calculated_em.mean()), 6) if len(result) else 0.0,
            "non_empty_model_output_count": int(accepted.sum()),
        }
    ]
    summary = pd.DataFrame(summary_rows)

    result.to_csv(args.output, index=False, encoding=args.encoding)
    summary.to_csv(args.summary_output, index=False, encoding=args.encoding)

    print(f"Wrote per-row baseline metrics: {args.output}")
    print(f"Wrote baseline summary: {args.summary_output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
