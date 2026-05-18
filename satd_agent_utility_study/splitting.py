from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from satd_langgraph.schema import SATDRecord

from .io_utils import read_csv_rows
from .models import SplitAssignment


def load_evidence_buckets(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    useful_by_task: dict[str, bool] = defaultdict(bool)
    seen_by_task: dict[str, bool] = defaultdict(bool)
    for row in read_csv_rows(path):
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        seen_by_task[task_id] = True
        status = str(row.get("retrieval_status") or "")
        support = str(row.get("top_support_level") or row.get("support_level") or "")
        if status == "retrieved_direct_supporting" or support in {"direct", "supporting"}:
            useful_by_task[task_id] = True
    buckets: dict[str, str] = {}
    for task_id in seen_by_task:
        buckets[task_id] = "useful" if useful_by_task[task_id] else "weak_or_absent"
    return buckets


def build_split_assignments(
    records: list[SATDRecord],
    *,
    dev_size: int = 150,
    holdout_size: int = 150,
    seed: int = 20260517,
    evidence_buckets: dict[str, str] | None = None,
) -> list[SplitAssignment]:
    evidence_buckets = evidence_buckets or {}
    failure_records = [record for record in records if record.em_label == "NO"]
    grouped: dict[str, list[SATDRecord]] = defaultdict(list)
    for record in failure_records:
        intent = infer_intent_bucket(record.satd_comment)
        evidence = evidence_buckets.get(record.task_id, "unknown")
        edit_shape = infer_edit_shape_bucket(record.satd_comment, record.original_code)
        grouped[f"{intent}|{evidence}|{edit_shape}"].append(record)

    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)

    dev_targets = _allocate_targets(grouped, dev_size)
    holdout_targets = _allocate_targets(grouped, holdout_size, offsets=dev_targets)
    assignments: list[SplitAssignment] = []
    selected_dev: set[str] = set()
    selected_holdout: set[str] = set()

    for stratum, values in grouped.items():
        intent, evidence, edit_shape = stratum.split("|")
        for record in values[: dev_targets.get(stratum, 0)]:
            selected_dev.add(record.task_id)
            assignments.append(
                SplitAssignment(record.task_id, "dev", record.em_label, record.project, intent, evidence, edit_shape, stratum)
            )
        start = dev_targets.get(stratum, 0)
        end = start + holdout_targets.get(stratum, 0)
        for record in values[start:end]:
            selected_holdout.add(record.task_id)
            assignments.append(
                SplitAssignment(record.task_id, "holdout", record.em_label, record.project, intent, evidence, edit_shape, stratum)
            )

    for record in records:
        if record.task_id in selected_dev or record.task_id in selected_holdout:
            continue
        intent = infer_intent_bucket(record.satd_comment)
        evidence = evidence_buckets.get(record.task_id, "unknown")
        edit_shape = infer_edit_shape_bucket(record.satd_comment, record.original_code)
        assignments.append(
            SplitAssignment(
                record.task_id,
                "full_only",
                record.em_label,
                record.project,
                intent,
                evidence,
                edit_shape,
                f"{intent}|{evidence}|{edit_shape}",
            )
        )
    return assignments


def infer_intent_bucket(comment: str) -> str:
    lowered = str(comment or "").lower()
    if any(token in lowered for token in ("remove", "delete", "drop", "obsolete", "deprecated")):
        return "remove_or_cleanup"
    if any(token in lowered for token in ("doc", "comment", "readme", "describe")):
        return "documentation"
    if any(token in lowered for token in ("rename", "replace", "instead", "use ")):
        return "replacement"
    if any(token in lowered for token in ("handle", "exception", "error", "check", "validate")):
        return "guard_or_error"
    if any(token in lowered for token in ("optimi", "faster", "performance", "cache")):
        return "optimization"
    return "unclear_or_general"


def infer_edit_shape_bucket(comment: str, code: str) -> str:
    lowered = str(comment or "").lower()
    if any(token in lowered for token in ("remove", "delete", "drop")):
        return "delete"
    if any(token in lowered for token in ("rename", "replace", "instead")):
        return "replace"
    if any(token in lowered for token in ("check", "handle", "exception", "validate")):
        return "guard"
    line_count = len(str(code or "").splitlines())
    return "small_local" if line_count <= 20 else "broader_local"


def _allocate_targets(
    grouped: dict[str, list[SATDRecord]],
    total: int,
    offsets: dict[str, int] | None = None,
) -> dict[str, int]:
    offsets = offsets or {}
    available = {key: max(0, len(values) - offsets.get(key, 0)) for key, values in grouped.items()}
    total_available = sum(available.values())
    if total_available <= 0 or total <= 0:
        return {key: 0 for key in grouped}
    raw = {key: total * count / total_available for key, count in available.items()}
    allocation = {key: min(available[key], int(value)) for key, value in raw.items()}
    remainder = total - sum(allocation.values())
    ranked = sorted(
        grouped,
        key=lambda key: (raw.get(key, 0.0) - allocation.get(key, 0), available.get(key, 0), key),
        reverse=True,
    )
    for key in ranked:
        if remainder <= 0:
            break
        if allocation[key] >= available[key]:
            continue
        allocation[key] += 1
        remainder -= 1
    return allocation

