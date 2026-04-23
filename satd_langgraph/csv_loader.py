from __future__ import annotations

import csv
from pathlib import Path

from .schema import SATDRecord


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    encodings = ("utf-8-sig", "utf-8", "gb18030", "cp936", "latin-1")
    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return []


def _first_non_empty(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def load_satd_csv(path: Path, limit: int | None = None) -> list[SATDRecord]:
    records: list[SATDRecord] = []
    for row in _read_csv_rows(path):
        records.append(
            SATDRecord(
                task_id=_first_non_empty(row, "index"),
                satd_comment=_first_non_empty(row, "SATD_comment"),
                original_code=str(row.get("original_code", "")).rstrip(),
                manual_code=str(row.get("manual_code") or row.get("manual_clean") or "").rstrip(),
                user=_first_non_empty(row, "user"),
                project=_first_non_empty(row, "project"),
                file_path=_first_non_empty(row, "file_path", "created_in_file"),
                commit=_first_non_empty(row, "commit", "created_in_commit"),
                em_label=_first_non_empty(row, "EM").upper(),
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records
