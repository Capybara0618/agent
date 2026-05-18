from __future__ import annotations

import argparse
import json
from pathlib import Path

from .lightweight_repair import LightweightRepairExperiment, summarize_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lightweight SATD repair context utility experiments.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=[
            "baseline",
            "baseline_guarded",
            "evidence",
            "evidence_guarded",
            "evidence_guarded_v2",
            "guided",
            "guided_complete",
        ],
        required=True,
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-evidence", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment = LightweightRepairExperiment(
        model=args.model,
        max_evidence=args.max_evidence,
        candidate_pool=args.candidate_pool,
        verbose=args.verbose,
    )
    rows = experiment.run_csv(
        args.input,
        args.output,
        variant=args.variant,
        resume=args.resume,
        flush_every=args.flush_every,
    )
    summary = summarize_rows(rows)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
