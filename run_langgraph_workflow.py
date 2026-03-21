from __future__ import annotations

import argparse
from pathlib import Path

from satd_langgraph import LangGraphSATDWorkflow


# python run_langgraph_workflow.py --input code.csv --output-dir outputs_langgraph_smoke5 --limit 5 --model gpt-4o-mini --verbose --write-batch-size 10 --resume


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LangGraph SATD workflow on code.csv.")
    parser.add_argument("--input", type=Path, default=Path("code.csv"), help="Path to the SATD CSV file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_langgraph"),
        help="Directory for workflow outputs.",
    )
    parser.add_argument("--max-rounds", type=int, default=2, help="Maximum repair-review iterations.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick experiments.")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-task and per-stage progress logs while running.",
    )
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=10,
        help="Flush results to disk every N tasks.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output directory by skipping task_ids already present in results.csv.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    workflow = LangGraphSATDWorkflow(
        max_rounds=args.max_rounds,
        model=args.model,
        verbose=args.verbose,
        write_batch_size=args.write_batch_size,
    )
    summary = workflow.run_csv(args.input, args.output_dir, limit=args.limit, resume=args.resume)

    print("LangGraph SATD workflow finished.")
    print(f"Agent mode: {summary['agent_mode']}")
    print(f"Model: {summary['model']}")
    print(f"Max rounds: {summary['max_rounds']}")
    print(f"Input SATD count: {summary['input_satd_count']}")
    print(f"Analyze filtered count: {summary['analyze_filtered_count']}")
    print(f"Review rejected count: {summary['review_rejected_count']}")
    print(f"Workflow output count: {summary['workflow_output_count']}")
    print(f"Successful repair count: {summary['successful_repair_count']}")
    print(f"Precision: {summary['precision']}")
    print(f"Recall: {summary['recall']}")
    print(f"Written tasks: {summary['written_tasks']}")
    print(f"Write batch size: {summary['write_batch_size']}")
    print(f"Resume mode: {args.resume}")
    print(f"Main trajectory file: {args.output_dir / 'trajectory_overview.csv'}")
    print(f"Context cache index: {args.output_dir / 'context_cache.csv'}")
    print(f"Context cache dir: {args.output_dir / 'context_cache'}")
    print(f"Summary file: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

