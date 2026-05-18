from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from satd_langgraph.csv_loader import load_satd_csv


REMOVE_OR_DEFAULT_PATTERN = re.compile(
    r"\b(remove|legacy|default|work\s*around|workaround|compatib|upgrade issues|temporary workaround)\b",
    re.IGNORECASE,
)

CLEANUP_UNCERTAINTY_PATTERN = re.compile(
    r"\b("
    r"remove|removed|removing|legacy|default|work\s*around|workaround|compatib|upgrade issues|temporary workaround|"
    r"drop|eliminate|opposite|string copy|lie|busted|temporary hack|once it exists|maybe a different"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ContextPolicyRow:
    task_id: str
    project: str
    file_path: str
    commit: str
    policy: str
    selected_variant: str
    exact_match: bool
    baseline_guarded_exact: bool
    evidence_guarded_exact: bool
    policy_reason: str
    evidence_count: int
    evidence_types: str
    repaired_code: str
    processed_repaired_code: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def apply_policy(
    input_csv: Path,
    baseline_csv: Path,
    evidence_csv: Path,
    output_csv: Path,
    *,
    policy: str = "remove_default_gate",
) -> list[ContextPolicyRow]:
    records = load_satd_csv(input_csv)
    baseline = _read_table(baseline_csv)
    evidence = _read_table(evidence_csv)
    rows: list[ContextPolicyRow] = []
    for record in records:
        baseline_row = baseline.get(record.task_id)
        evidence_row = evidence.get(record.task_id)
        if baseline_row is None or evidence_row is None:
            continue
        use_baseline, reason = _choose_baseline(record.satd_comment, policy=policy)
        selected_variant = "baseline_guarded" if use_baseline else "evidence_guarded"
        selected = baseline_row if use_baseline else evidence_row
        rows.append(
            ContextPolicyRow(
                task_id=record.task_id,
                project=record.project,
                file_path=record.file_path,
                commit=record.commit,
                policy=policy,
                selected_variant=selected_variant,
                exact_match=_as_bool(selected.get("exact_match", "")),
                baseline_guarded_exact=_as_bool(baseline_row.get("exact_match", "")),
                evidence_guarded_exact=_as_bool(evidence_row.get("exact_match", "")),
                policy_reason=reason,
                evidence_count=int(float(evidence_row.get("evidence_count") or 0)),
                evidence_types=evidence_row.get("evidence_types", ""),
                repaired_code=selected.get("repaired_code", ""),
                processed_repaired_code=selected.get("processed_repaired_code", ""),
            )
        )
    _write_rows(output_csv, rows)
    return rows


def _choose_baseline(satd_comment: str, *, policy: str) -> tuple[bool, str]:
    if policy == "remove_default_gate":
        matched = REMOVE_OR_DEFAULT_PATTERN.search(satd_comment or "")
    elif policy == "cleanup_uncertainty_gate":
        matched = CLEANUP_UNCERTAINTY_PATTERN.search(satd_comment or "")
    else:
        raise ValueError(f"unknown policy: {policy}")
    if matched:
        return True, f"baseline_guarded selected because SATD matched {policy}: {matched.group(0)}"
    return False, f"evidence_guarded selected because SATD did not match {policy}"


def _read_table(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row.get("task_id", ""): row for row in csv.DictReader(handle) if row.get("task_id")}


def _write_rows(path: Path, rows: list[ContextPolicyRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ContextPolicyRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def summarize(rows: list[ContextPolicyRow]) -> dict[str, Any]:
    total = len(rows)
    exact = sum(1 for row in rows if row.exact_match)
    baseline_exact = sum(1 for row in rows if row.baseline_guarded_exact)
    evidence_exact = sum(1 for row in rows if row.evidence_guarded_exact)
    selected_counts: dict[str, int] = {}
    for row in rows:
        selected_counts[row.selected_variant] = selected_counts.get(row.selected_variant, 0) + 1
    baseline_wins = {row.task_id for row in rows if row.baseline_guarded_exact}
    policy_wins = {row.task_id for row in rows if row.exact_match}
    evidence_wins = {row.task_id for row in rows if row.evidence_guarded_exact}
    return {
        "sample_count": total,
        "policy_exact_match_count": exact,
        "policy_exact_match_rate": round(exact / total, 4) if total else 0.0,
        "baseline_guarded_exact_match_count": baseline_exact,
        "evidence_guarded_exact_match_count": evidence_exact,
        "policy_rescues_vs_baseline_guarded": len(policy_wins - baseline_wins),
        "policy_regressions_vs_baseline_guarded": len(baseline_wins - policy_wins),
        "evidence_rescues_vs_baseline_guarded": len(evidence_wins - baseline_wins),
        "evidence_regressions_vs_baseline_guarded": len(baseline_wins - evidence_wins),
        "selected_variant_counts": selected_counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply a deployable SATD context-injection policy to repair outputs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="remove_default_gate")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = apply_policy(args.input, args.baseline, args.evidence, args.output, policy=args.policy)
    summary = summarize(rows)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
