from __future__ import annotations

from collections import Counter

from satd_langgraph.schema import SATDRecord


def validate_records(records: list[SATDRecord]) -> list[dict[str, object]]:
    ids = [record.task_id for record in records]
    em_counts = Counter(record.em_label for record in records)
    duplicate_ids = len(ids) - len(set(ids))
    missing_locator_count = sum(
        1 for record in records if not (record.user and record.project and record.file_path and record.commit)
    )
    return [
        {"check": "record_count", "value": len(records), "status": "ok" if len(records) == 1000 else "fail"},
        {"check": "em_yes_count", "value": em_counts.get("YES", 0), "status": "ok" if em_counts.get("YES", 0) == 97 else "fail"},
        {"check": "em_no_count", "value": em_counts.get("NO", 0), "status": "ok" if em_counts.get("NO", 0) == 903 else "fail"},
        {"check": "duplicate_task_id_count", "value": duplicate_ids, "status": "ok" if duplicate_ids == 0 else "fail"},
        {
            "check": "missing_repo_locator_count",
            "value": missing_locator_count,
            "status": "ok" if missing_locator_count == 0 else "fail",
        },
    ]
