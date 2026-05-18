from __future__ import annotations

import argparse
from pathlib import Path

from .reporting import build_reports
from .runner import prepare_study, run_phase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the agent-facing SATD context utility study.")
    parser.add_argument("--input", type=Path, default=Path("code.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("satd_agent_utility_study_outputs"))
    parser.add_argument("--stage", choices=["prepare", "dev", "holdout", "full", "report", "all"], default="all")
    parser.add_argument("--oracle-inventory", type=Path, default=None)
    parser.add_argument("--dev-size", type=int, default=150)
    parser.add_argument("--holdout-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--select-top-k", type=int, default=0)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--write-batch-size", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--llm-judge", dest="enable_llm_judge", action="store_true", default=True)
    parser.add_argument("--no-llm-judge", dest="enable_llm_judge", action="store_false")
    parser.add_argument("--judge-model", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage in {"prepare", "all"}:
        prepare_study(
            args.input,
            args.output_dir,
            dev_size=args.dev_size,
            holdout_size=args.holdout_size,
            seed=args.seed,
            oracle_inventory=args.oracle_inventory,
        )
    if args.stage in {"dev", "all"}:
        run_phase(
            args.output_dir,
            phase="dev",
            variants=args.variants,
            model=args.model,
            max_rounds=args.max_rounds,
            verbose=args.verbose,
            resume=args.resume,
            write_batch_size=args.write_batch_size,
            enable_llm_judge=args.enable_llm_judge,
            judge_model=args.judge_model,
        )
    if args.stage == "holdout":
        run_phase(
            args.output_dir,
            phase="holdout",
            variants=args.variants,
            model=args.model,
            max_rounds=args.max_rounds,
            verbose=args.verbose,
            resume=args.resume,
            write_batch_size=args.write_batch_size,
            enable_llm_judge=args.enable_llm_judge,
            judge_model=args.judge_model,
            select_top_k=args.select_top_k,
        )
    if args.stage == "full":
        run_phase(
            args.output_dir,
            phase="full",
            variants=args.variants,
            model=args.model,
            max_rounds=args.max_rounds,
            verbose=args.verbose,
            resume=args.resume,
            write_batch_size=args.write_batch_size,
            enable_llm_judge=args.enable_llm_judge,
            judge_model=args.judge_model,
            select_top_k=args.select_top_k,
        )
    if args.stage in {"report", "all"}:
        build_reports(args.output_dir)


if __name__ == "__main__":
    main()

