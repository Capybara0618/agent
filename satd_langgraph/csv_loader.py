from __future__ import annotations

import csv
from pathlib import Path

from .schema import SATDRecord


def load_satd_csv(path: Path, limit: int | None = None) -> list[SATDRecord]:
    records: list[SATDRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                SATDRecord(
                    task_id=str(row.get("index", "")).strip(),
                    satd_comment=str(row.get("SATD_comment", "")).strip(),
                    original_code=str(row.get("original_code", "")).rstrip(),
                    manual_code=str(row.get("manual_code", "")).rstrip(),
                    user=str(row.get("user", "")).strip(),
                    project=str(row.get("project", "")).strip(),
                    file_path=str(row.get("file_path", "")).strip(),
                    em_label=str(row.get("EM", "")).strip().upper(),
                )
            )
            if limit is not None and len(records) >= limit:
                break
    return records
