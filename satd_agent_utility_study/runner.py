from __future__ import annotations

from pathlib import Path

from satd_langgraph import LangGraphSATDWorkflow
from satd_langgraph.csv_loader import load_satd_csv

from .io_utils import ensure_dir, read_csv_rows, write_csv_rows, write_json
from .models import VariantConfig
from .reporting import build_reports, select_top_variants
from .splitting import build_split_assignments, load_evidence_buckets
from .validation import validate_records
from .variants import DEFAULT_VARIANTS, default_variant_map


def prepare_study(
    input_csv: Path,
    output_dir: Path,
    *,
    dev_size: int,
    holdout_size: int,
    seed: int,
    oracle_inventory: Path | None = None,
) -> None:
    records = load_satd_csv(input_csv)
    write_csv_rows(output_dir / "dataset_validation.csv", validate_records(records))
    evidence_buckets = load_evidence_buckets(oracle_inventory)
    assignments = build_split_assignments(
        records,
        dev_size=dev_size,
        holdout_size=holdout_size,
        seed=seed,
        evidence_buckets=evidence_buckets,
    )
    ensure_dir(output_dir / "splits")
    write_csv_rows(output_dir / "split_assignments.csv", [item.to_row() for item in assignments])
    rows = read_csv_rows(input_csv)
    by_split = {
        "dev": {item.task_id for item in assignments if item.split == "dev"},
        "holdout": {item.task_id for item in assignments if item.split == "holdout"},
    }
    for split, task_ids in by_split.items():
        split_rows = [row for row in rows if str(row.get("index") or "") in task_ids]
        write_csv_rows(output_dir / "splits" / f"{split}.csv", split_rows, fieldnames=list(rows[0].keys()))
    write_csv_rows(output_dir / "splits" / "full.csv", rows, fieldnames=list(rows[0].keys()))
    write_json(output_dir / "variant_matrix.json", [item.to_row() for item in DEFAULT_VARIANTS])


def run_phase(
    output_dir: Path,
    *,
    phase: str,
    variants: list[str] | None,
    model: str,
    max_rounds: int,
    verbose: bool,
    resume: bool,
    write_batch_size: int,
    enable_llm_judge: bool,
    judge_model: str | None,
    select_top_k: int = 0,
) -> list[str]:
    variant_map = default_variant_map()
    if not variants:
        if select_top_k > 0:
            source_phase = "dev" if phase == "holdout" else "holdout"
            variants = select_top_variants(
                output_dir,
                source_phase,
                baseline_variant="current_agent",
                top_k=select_top_k,
            )
        else:
            variants = list(variant_map)
    selected = [name for name in variants if name in variant_map]
    if "current_agent" not in selected:
        selected = ["current_agent", *selected]
    split_csv = output_dir / "splits" / f"{phase}.csv"
    if phase == "full":
        split_csv = output_dir / "splits" / "full.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing prepared split file: {split_csv}")
    for name in selected:
        _run_variant(
            split_csv,
            output_dir / "runs" / phase / name,
            variant_map[name],
            model=model,
            max_rounds=max_rounds,
            verbose=verbose,
            resume=resume,
            write_batch_size=write_batch_size,
            enable_llm_judge=enable_llm_judge,
            judge_model=judge_model,
        )
    build_reports(output_dir)
    return selected


def _run_variant(
    input_csv: Path,
    variant_dir: Path,
    variant: VariantConfig,
    *,
    model: str,
    max_rounds: int,
    verbose: bool,
    resume: bool,
    write_batch_size: int,
    enable_llm_judge: bool,
    judge_model: str | None,
) -> None:
    workflow = LangGraphSATDWorkflow(
        max_rounds=max_rounds,
        model=model,
        verbose=verbose,
        write_batch_size=write_batch_size,
        repair_context_mode=variant.repair_context_mode,
        max_method_contexts=variant.max_method_contexts,
        repository_evidence_mode=variant.repository_evidence_mode,
        max_repository_evidence=variant.max_repository_evidence,
        repository_evidence_prompt_mode=variant.repository_evidence_prompt_mode,
        repository_evidence_guidance_mode=variant.repository_evidence_guidance_mode,
        repository_evidence_rerank_mode=variant.repository_evidence_rerank_mode,
        force_route="context_required",
        enable_llm_judge=enable_llm_judge,
        judge_model=judge_model,
    )
    workflow.run_csv(input_csv, variant_dir, resume=resume)
