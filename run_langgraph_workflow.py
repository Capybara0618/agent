from __future__ import annotations

import argparse
import os
from pathlib import Path

from satd_langgraph import LangGraphSATDWorkflow


def _default_repo_cache_dir(input_path: Path) -> Path | None:
    if input_path.stem == "random_code":
        return input_path.resolve().parent / ".repo_cache_random_code"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LangGraph SATD workflow on code.csv.")
    parser.add_argument("--input", type=Path, default=Path("random_code.csv"), help="Path to the SATD CSV file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_langgraph_random1000"),
        help="Directory for workflow outputs.",
    )
    parser.add_argument("--max-rounds", type=int, default=2, help="Maximum repair-review iterations.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick experiments.")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name for the OpenAI-compatible endpoint.")
    parser.add_argument("--verbose", action="store_true", help="Print per-task and per-stage progress logs while running.")
    parser.add_argument("--write-batch-size", type=int, default=10, help="Flush results to disk every N tasks.")
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Run context routing and analyzer only, then stop before fixer/reviewer.",
    )
    parser.add_argument(
        "--fixer-only",
        action="store_true",
        help="Skip analyzer and reviewer; run one fixer attempt and accept its output for metric inspection.",
    )
    parser.add_argument(
        "--force-route",
        choices=["auto", "no_context", "context_required"],
        default="context_required",
        help="Force the context route and skip the LLM context router. Default is context_required; use auto to enable the router.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output directory by skipping task_ids already present in results.csv.",
    )
    parser.add_argument(
        "--repair-context-mode",
        choices=["clone_treesitter", "method_query"],
        default="clone_treesitter",
        help="Repair context construction mode. Default clones the repo locally and uses Tree-sitter to find relevant methods.",
    )
    parser.add_argument(
        "--max-method-contexts",
        type=int,
        default=2,
        help="Maximum number of model-identified methods to retrieve as repair context. Default is 2.",
    )
    parser.add_argument(
        "--git-remote-base",
        default=None,
        help="Optional git remote base for repo clone/fetch, e.g. https://mirrors.tuna.tsinghua.edu.cn/git/github.com",
    )
    parser.add_argument(
        "--git-remote-template",
        default=None,
        help="Optional git remote template with {owner} and {repo}; overrides --git-remote-base.",
    )
    parser.add_argument(
        "--repo-cache-dir",
        type=Path,
        default=None,
        help="Optional repo cache directory. By default, random_code.csv uses .repo_cache_random_code.",
    )
    parser.add_argument(
        "--llm-judge",
        dest="enable_llm_judge",
        action="store_true",
        default=True,
        help="Add offline LLM-as-judge metric using the manual repair as evaluation reference. Enabled by default.",
    )
    parser.add_argument(
        "--no-llm-judge",
        dest="enable_llm_judge",
        action="store_false",
        help="Disable the per-row offline LLM-as-judge metric.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional model name for LLM-as-judge. Defaults to --model.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.git_remote_base:
        os.environ["SATD_GIT_REMOTE_BASE"] = args.git_remote_base
    if args.git_remote_template:
        os.environ["SATD_GIT_REMOTE_TEMPLATE"] = args.git_remote_template
    repo_cache_dir = args.repo_cache_dir
    if repo_cache_dir is None:
        repo_cache_dir = _default_repo_cache_dir(args.input)
    if repo_cache_dir is not None:
        resolved_repo_cache_dir = repo_cache_dir if repo_cache_dir.is_absolute() else (Path.cwd() / repo_cache_dir)
        resolved_repo_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["SATD_REPO_CACHE_DIR"] = str(resolved_repo_cache_dir)

    workflow = LangGraphSATDWorkflow(
        max_rounds=args.max_rounds,
        model=args.model,
        verbose=args.verbose,
        write_batch_size=args.write_batch_size,
        repair_context_mode=args.repair_context_mode,
        max_method_contexts=args.max_method_contexts,
        analysis_only=args.analysis_only,
        fixer_only=args.fixer_only,
        force_route=None if args.force_route == "auto" else args.force_route,
        enable_llm_judge=args.enable_llm_judge,
        judge_model=args.judge_model,
    )
    summary = workflow.run_csv(args.input, args.output_dir, limit=args.limit, resume=args.resume)

    print("LangGraph SATD workflow finished.")
    print(f"Input path: {summary.get('input_path') or args.input}")
    print(f"Input limit: {summary.get('input_limit')}")
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
    print(f"Avg BLEU-diff: {summary.get('avg_BLEU_diff')}")
    print(f"Avg CrystalBLEU-diff: {summary.get('avg_CrystalBLEU_diff')}")
    print(f"Avg LEMOD: {summary.get('avg_LEMOD')}")
    print(f"Avg LLM-as-judge: {summary.get('avg_LLM_as_judge')}")
    print(f"Written tasks: {summary['written_tasks']}")
    print(f"Write batch size: {summary['write_batch_size']}")
    print(f"Analysis only: {summary.get('analysis_only')}")
    print(f"Fixer only: {summary.get('fixer_only')}")
    print(f"Repair context mode: {summary.get('repair_context_mode')}")
    print(f"Method inquiry enabled: {summary.get('method_inquiry_enabled')}")
    print(f"Context router enabled: {summary.get('context_router_enabled')}")
    print(f"Force route: {summary.get('force_route') or 'none'}")
    print(f"Max method contexts: {summary.get('max_method_contexts')}")
    print(f"LLM judge enabled: {summary.get('llm_judge_enabled')}")
    print(f"Judge model: {summary.get('judge_model') or 'none'}")
    print(f"Git remote base: {os.environ.get('SATD_GIT_REMOTE_BASE') or 'https://mirrors.tuna.tsinghua.edu.cn/git/github.com'}")
    print(f"Git remote template: {os.environ.get('SATD_GIT_REMOTE_TEMPLATE') or 'none'}")
    print(f"Repo cache dir: {os.environ.get('SATD_REPO_CACHE_DIR') or (Path.cwd() / '.repo_cache')}")
    print(f"Resume mode: {args.resume}")
    print(f"Main trajectory file: {args.output_dir / 'trajectory_overview.csv'}")
    print(f"Task progress file: {args.output_dir / 'task_progress.csv'}")
    print(f"Context cache index: {args.output_dir / 'context_cache.csv'}")
    print(f"Context cache dir: {args.output_dir / 'context_cache'}")
    print(f"Repair debug dir: {args.output_dir / 'repair_debug'}")
    print(f"Summary file: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
