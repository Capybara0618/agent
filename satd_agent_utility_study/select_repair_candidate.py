from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from satd_langgraph.csv_loader import load_satd_csv
from satd_langgraph.openai_client import OpenAICompatClient
from satd_langgraph.schema import SATDRecord, preprocess_python_code


SYSTEM_PROMPT = "You are an expert software engineer specialized in technical debt refactoring."


@dataclass(frozen=True)
class CandidateSelectionRow:
    task_id: str
    project: str
    file_path: str
    commit: str
    selected_variant: str
    exact_match: bool
    candidate_variants: str
    candidate_exact_matches: str
    rationale: str
    repaired_code: str
    processed_repaired_code: str
    error: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


class RepairCandidateSelector:
    def __init__(self, *, model: str = "gpt-4o-mini", verbose: bool = False) -> None:
        self.client = OpenAICompatClient(model=model, verbose=verbose)
        self.verbose = bool(verbose)

    def run(
        self,
        input_csv: Path,
        output_csv: Path,
        candidate_csvs: list[Path],
        *,
        resume: bool = False,
        flush_every: int = 1,
    ) -> list[CandidateSelectionRow]:
        records = load_satd_csv(input_csv)
        existing = _read_rows(output_csv) if resume and output_csv.exists() else []
        completed = {row.get("task_id", "") for row in existing}
        candidate_tables = [_read_candidate_table(path) for path in candidate_csvs]
        new_rows: list[CandidateSelectionRow] = []
        pending: list[CandidateSelectionRow] = []
        if self.verbose:
            print(
                f"[select] resume_completed={len(completed)} total={len(records)} "
                f"candidates={len(candidate_tables)} output={output_csv}",
                flush=True,
            )
        for index, record in enumerate(records, start=1):
            if record.task_id in completed:
                continue
            if self.verbose:
                print(f"[select] {index}/{len(records)} task_id={record.task_id} start project={record.project}", flush=True)
            row = self.run_record(record, candidate_tables)
            new_rows.append(row)
            pending.append(row)
            if self.verbose:
                print(
                    f"[select] {index}/{len(records)} task_id={record.task_id} "
                    f"selected={row.selected_variant} exact={row.exact_match} error={bool(row.error)}",
                    flush=True,
                )
            if len(pending) >= max(1, flush_every):
                _write_rows(output_csv, existing, new_rows)
                pending = []
        if pending or not output_csv.exists():
            _write_rows(output_csv, existing, new_rows)
        return [_row_from_existing(row) for row in existing] + new_rows

    def run_record(
        self,
        record: SATDRecord,
        candidate_tables: list[dict[str, dict[str, str]]],
    ) -> CandidateSelectionRow:
        try:
            candidates = _collect_candidates(record.task_id, candidate_tables)
            if not candidates:
                raise ValueError("no candidates found for task")
            selected_variant, rationale = self._select(record, candidates)
            if selected_variant not in candidates:
                selected_variant = _fallback_variant(candidates)
                rationale = f"selector returned unavailable variant; fallback to {selected_variant}"
            repaired = candidates[selected_variant].get("repaired_code", "")
            processed = preprocess_python_code(repaired)
            exact = processed == preprocess_python_code(record.manual_code)
            error = ""
        except Exception as exc:
            candidates = {}
            selected_variant = ""
            rationale = ""
            repaired = ""
            processed = ""
            exact = False
            error = f"{type(exc).__name__}: {exc}"
        return CandidateSelectionRow(
            task_id=record.task_id,
            project=record.project,
            file_path=record.file_path,
            commit=record.commit,
            selected_variant=selected_variant,
            exact_match=exact,
            candidate_variants=" | ".join(candidates.keys()),
            candidate_exact_matches=" | ".join(
                f"{name}:{str(row.get('exact_match', '')).lower()}" for name, row in candidates.items()
            ),
            rationale=rationale,
            repaired_code=repaired,
            processed_repaired_code=processed,
            error=error,
        )

    def _select(self, record: SATDRecord, candidates: dict[str, dict[str, str]]) -> tuple[str, str]:
        candidate_blocks = []
        for name, row in candidates.items():
            code = row.get("repaired_code", "")
            candidate_blocks.append(
                f"### Candidate {name}\n"
                f"metadata: lines={len(code.splitlines())}, chars={len(code)}, evidence_count={row.get('evidence_count', '')}\n"
                f"```python\n{code[:6000]}\n```"
            )
        payload = self.client.generate_json(
            SYSTEM_PROMPT,
            "Choose the best repair candidate for this SATD. Do not use any hidden answer key. "
            "Judge only from the SATD comment, the original code, and the candidate outputs.\n\n"
            "Selection criteria, in priority order:\n"
            "1. The candidate resolves the SATD comment with a concrete code change.\n"
            "2. The candidate is a complete updated version of the original code block, not only changed lines.\n"
            "3. The candidate preserves unrelated behavior and avoids rewrites not required by the SATD.\n"
            "4. If two candidates are both plausible, prefer the smaller, more local edit.\n\n"
            f"SATD comment:\n{record.satd_comment}\n\n"
            f"Original code:\n```python\n{record.original_code[:8000]}\n```\n\n"
            f"Candidates:\n{chr(10).join(candidate_blocks)}\n\n"
            'Return JSON with keys "selected_variant" and "rationale".',
            temperature=0.0,
            request_label=f"candidate_selector:task_{record.task_id}",
            max_tokens=350,
        )
        return str(payload.get("selected_variant") or "").strip(), " ".join(str(payload.get("rationale") or "").split())


def _collect_candidates(
    task_id: str,
    candidate_tables: list[dict[str, dict[str, str]]],
) -> dict[str, dict[str, str]]:
    candidates: dict[str, dict[str, str]] = {}
    seen_code: set[str] = set()
    for table in candidate_tables:
        row = table.get(task_id)
        if not row:
            continue
        variant = row.get("variant", "").strip()
        code = row.get("repaired_code", "")
        if not variant or not code:
            continue
        signature = preprocess_python_code(code)
        if signature in seen_code:
            continue
        seen_code.add(signature)
        candidates[variant] = row
    return candidates


def _fallback_variant(candidates: dict[str, dict[str, str]]) -> str:
    for preferred in ("evidence", "guided", "guided_complete", "baseline"):
        if preferred in candidates:
            return preferred
    return next(iter(candidates))


def _read_candidate_table(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_rows(path)
    table: dict[str, dict[str, str]] = {}
    for row in rows:
        task_id = row.get("task_id", "")
        if task_id:
            table[task_id] = row
    return table


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, existing: list[dict[str, str]], rows: list[CandidateSelectionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(CandidateSelectionRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in existing:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
        for row in rows:
            writer.writerow(row.to_row())


def _row_from_existing(row: dict[str, str]) -> CandidateSelectionRow:
    return CandidateSelectionRow(
        task_id=row.get("task_id", ""),
        project=row.get("project", ""),
        file_path=row.get("file_path", ""),
        commit=row.get("commit", ""),
        selected_variant=row.get("selected_variant", ""),
        exact_match=str(row.get("exact_match", "")).lower() == "true",
        candidate_variants=row.get("candidate_variants", ""),
        candidate_exact_matches=row.get("candidate_exact_matches", ""),
        rationale=row.get("rationale", ""),
        repaired_code=row.get("repaired_code", ""),
        processed_repaired_code=row.get("processed_repaired_code", ""),
        error=row.get("error", ""),
    )


def summarize_rows(rows: list[CandidateSelectionRow]) -> dict[str, Any]:
    total = len(rows)
    exact = sum(1 for row in rows if row.exact_match)
    errors = sum(1 for row in rows if row.error)
    by_variant: dict[str, int] = {}
    for row in rows:
        by_variant[row.selected_variant] = by_variant.get(row.selected_variant, 0) + 1
    return {
        "sample_count": total,
        "exact_match_count": exact,
        "exact_match_rate": round(exact / total, 4) if total else 0.0,
        "error_count": errors,
        "selected_variant_counts": by_variant,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select among SATD repair candidates without using manual answers.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selector = RepairCandidateSelector(model=args.model, verbose=args.verbose)
    rows = selector.run(
        args.input,
        args.output,
        args.candidate,
        resume=args.resume,
        flush_every=args.flush_every,
    )
    summary = summarize_rows(rows)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
