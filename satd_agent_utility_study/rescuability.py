from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from satd_empirical_study.repair_signals import extract_repair_signals
from satd_langgraph.csv_loader import load_satd_csv
from satd_langgraph.tools.repository_evidence_retriever import RepositoryEvidenceRetriever

from .io_utils import read_csv_rows, write_csv_rows


@dataclass(frozen=True)
class RescuabilityAuditRow:
    task_id: str
    project: str
    actionable_query_count: int
    actionable_queries: str
    introduced_query_count: int
    introduced_queries: str
    repo_supported_introduced_queries: str
    repo_unsupported_introduced_queries: str
    repo_support_scan_status: str
    retrieved_evidence_count: int
    direct_or_supporting_count: int
    covered_actionable_query_count: int
    covered_actionable_queries: str
    uncovered_actionable_queries: str
    coverage_ratio: float
    proxy_label: str
    rationale: str

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def run_rescuability_audit(
    input_csv: Path,
    output_path: Path,
    *,
    max_evidence: int = 10,
    resume: bool = False,
    verbose: bool = False,
    flush_every: int = 1,
) -> list[RescuabilityAuditRow]:
    records = load_satd_csv(input_csv)
    retriever = RepositoryEvidenceRetriever()
    existing_rows = read_csv_rows(output_path) if resume and output_path.exists() else []
    completed = {row.get("task_id", "") for row in existing_rows}
    rows: list[RescuabilityAuditRow] = []
    pending_rows: list[RescuabilityAuditRow] = []
    if verbose and completed:
        print(f"[audit] resume completed={len(completed)} total={len(records)} output={output_path}")
    for index, record in enumerate(records, start=1):
        if record.task_id in completed:
            continue
        if verbose:
            print(f"[audit] {index}/{len(records)} task_id={record.task_id} start project={record.project}", flush=True)
        signals = extract_repair_signals(record)
        actionable = signals.actionable_queries()
        introduced = _introduced_queries(record, signals)
        supported_introduced, unsupported_introduced, scan_status = _split_repo_supported_introduced_queries(
            retriever,
            record,
            introduced,
        )
        evidence = retriever.retrieve(
            owner=record.user,
            repo=record.project,
            current_path=record.file_path,
            satd_comment=record.satd_comment,
            original_code=record.original_code,
            ref=record.commit,
            max_items=max_evidence,
        )
        haystack = "\n".join(
            f"{item.query}\n{item.content}\n{item.why_relevant}"
            for item in evidence
        ).lower()
        covered = [query for query in actionable if _query_is_covered(query, haystack)]
        uncovered = [query for query in actionable if query not in covered]
        direct_or_supporting = sum(1 for item in evidence if item.support_level in {"direct", "supporting"})
        coverage_ratio = round(len(covered) / len(actionable), 4) if actionable else 0.0
        proxy_label, rationale = _label(
            actionable_count=len(actionable),
            unsupported_introduced_count=len(unsupported_introduced),
            evidence_count=len(evidence),
            direct_or_supporting_count=direct_or_supporting,
            coverage_ratio=coverage_ratio,
        )
        row = RescuabilityAuditRow(
                task_id=record.task_id,
                project=record.project,
                actionable_query_count=len(actionable),
                actionable_queries=" | ".join(actionable),
                introduced_query_count=len(introduced),
                introduced_queries=" | ".join(introduced),
                repo_supported_introduced_queries=" | ".join(supported_introduced),
                repo_unsupported_introduced_queries=" | ".join(unsupported_introduced),
                repo_support_scan_status=scan_status,
                retrieved_evidence_count=len(evidence),
                direct_or_supporting_count=direct_or_supporting,
                covered_actionable_query_count=len(covered),
                covered_actionable_queries=" | ".join(covered),
                uncovered_actionable_queries=" | ".join(uncovered),
                coverage_ratio=coverage_ratio,
                proxy_label=proxy_label,
                rationale=rationale,
            )
        rows.append(row)
        pending_rows.append(row)
        if verbose:
            print(
                f"[audit] {index}/{len(records)} task_id={record.task_id} "
                f"label={proxy_label} coverage={coverage_ratio} evidence={len(evidence)}",
                flush=True,
            )
        if len(pending_rows) >= max(1, flush_every):
            _write_audit_rows(output_path, existing_rows, rows)
            pending_rows = []
    if pending_rows or not output_path.exists():
        _write_audit_rows(output_path, existing_rows, rows)
    return [
        *[_row_from_existing(item) for item in existing_rows],
        *rows,
    ]


def _query_is_covered(query: str, haystack: str) -> bool:
    cleaned = str(query or "").strip().lower()
    if not cleaned:
        return False
    tail = cleaned.split(".")[-1]
    return cleaned in haystack or tail in haystack


def _write_audit_rows(
    output_path: Path,
    existing_rows: list[dict[str, str]],
    new_rows: list[RescuabilityAuditRow],
) -> None:
    write_csv_rows(output_path, [*existing_rows, *[row.to_row() for row in new_rows]])


def _row_from_existing(row: dict[str, str]) -> RescuabilityAuditRow:
    return RescuabilityAuditRow(
        task_id=row.get("task_id", ""),
        project=row.get("project", ""),
        actionable_query_count=int(row.get("actionable_query_count") or 0),
        actionable_queries=row.get("actionable_queries", ""),
        introduced_query_count=int(row.get("introduced_query_count") or 0),
        introduced_queries=row.get("introduced_queries", ""),
        repo_supported_introduced_queries=row.get("repo_supported_introduced_queries", ""),
        repo_unsupported_introduced_queries=row.get("repo_unsupported_introduced_queries", ""),
        repo_support_scan_status=row.get("repo_support_scan_status", ""),
        retrieved_evidence_count=int(row.get("retrieved_evidence_count") or 0),
        direct_or_supporting_count=int(row.get("direct_or_supporting_count") or 0),
        covered_actionable_query_count=int(row.get("covered_actionable_query_count") or 0),
        covered_actionable_queries=row.get("covered_actionable_queries", ""),
        uncovered_actionable_queries=row.get("uncovered_actionable_queries", ""),
        coverage_ratio=float(row.get("coverage_ratio") or 0.0),
        proxy_label=row.get("proxy_label", ""),
        rationale=row.get("rationale", ""),
    )


def _label(
    *,
    actionable_count: int,
    unsupported_introduced_count: int,
    evidence_count: int,
    direct_or_supporting_count: int,
    coverage_ratio: float,
) -> tuple[str, str]:
    if actionable_count == 0:
        return "low_signal", "The manual repair exposes no actionable symbol-level queries for repository evidence."
    if unsupported_introduced_count:
        return "requires_absent_symbols", "The manual repair introduces actionable symbols not found elsewhere in the target commit."
    if evidence_count == 0 or direct_or_supporting_count == 0:
        return "low_rescuability", "The current retriever found no direct/supporting repository evidence."
    if coverage_ratio >= 0.5:
        return "likely_rescuable", "Direct/supporting evidence covers at least half of the actionable patch signals."
    if coverage_ratio > 0.0:
        return "partial_rescuability", "Some patch-driving signals appear in retrieved evidence, but coverage is incomplete."
    return "low_rescuability", "Retrieved evidence does not cover the actionable manual repair signals."


def _introduced_queries(record, signals) -> list[str]:
    values: list[str] = []
    manual_only_identifiers = [
        value
        for value in _identifiers(record.manual_code)
        if value not in set(_identifiers(_strip_comments(record.original_code)))
    ]
    for group in (manual_only_identifiers, signals.added_calls, signals.changed_parameters):
        for value in group:
            cleaned = str(value or "").strip()
            if not cleaned or not signals._is_actionable(cleaned):
                continue
            if cleaned not in values:
                values.append(cleaned)
    return values


def _identifiers(code: str) -> list[str]:
    values: list[str] = []
    for value in re.findall(r"[A-Za-z_][A-Za-z0-9_\.]*", code or ""):
        if value not in values:
            values.append(value)
    return values


def _strip_comments(code: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in (code or "").splitlines())


def _split_repo_supported_introduced_queries(
    retriever: RepositoryEvidenceRetriever,
    record,
    introduced: list[str],
) -> tuple[list[str], list[str], str]:
    if not introduced:
        return [], [], "not_needed"
    tree = retriever.downloader._fetch_repo_tree_for_ref(record.user, record.project, "", record.commit)
    entries = list(tree.get("entries") or []) if tree.get("ok") else []
    source_paths = [str(item.get("path") or "") for item in entries if item.get("type") == "file" and retriever._looks_like_source(str(item.get("path") or ""))]
    source_paths = _rank_scan_paths(source_paths, record.file_path)[:250]
    scan_status = "complete" if len(source_paths) < 250 else "repo_support_scan_limited"
    supported: list[str] = []
    unsupported: list[str] = []
    current_content = retriever._read(record.user, record.project, record.commit, record.file_path)
    current_without_target = current_content.replace(record.original_code, "")
    for query in introduced:
        tail = query.split(".")[-1]
        found = tail in current_without_target
        if not found:
            for path in source_paths:
                if path == record.file_path:
                    continue
                if tail in retriever._read(record.user, record.project, record.commit, path):
                    found = True
                    break
        if found:
            supported.append(query)
        else:
            unsupported.append(query)
    return supported, unsupported, scan_status


def _rank_scan_paths(paths: list[str], current_path: str) -> list[str]:
    current_parts = [part for part in current_path.replace("\\", "/").split("/") if part]
    current_dir = "/".join(current_parts[:-1])

    def score(path: str) -> tuple[int, int, str]:
        normalized = path.replace("\\", "/")
        path_parts = [part for part in normalized.split("/") if part]
        shared = 0
        for left, right in zip(current_parts, path_parts):
            if left != right:
                break
            shared += 1
        same_dir = 0 if normalized.startswith(current_dir) else 1
        return (same_dir, -shared, normalized)

    return sorted(paths, key=score)
