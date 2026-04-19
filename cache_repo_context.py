from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

from satd_langgraph.csv_loader import load_satd_csv
from satd_langgraph.github_tools import GitHubToolbox


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Warm the shared SATD repo cache in code.csv task order without running repair."
    )
    parser.add_argument("--input", type=Path, default=Path("code.csv"), help="Path to the SATD CSV file.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_repo_cache_only"),
        help="Directory for cache-only progress files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from task progress and skip completed task_ids.",
    )
    parser.add_argument(
        "--resume-from-progress",
        type=Path,
        default=None,
        help="Optional task_progress.csv to align cache warming with an existing repair run.",
    )
    parser.add_argument(
        "--git-remote-base",
        default=None,
        help="Optional git remote base, e.g. https://mirrors.tuna.tsinghua.edu.cn/git/github.com",
    )
    parser.add_argument(
        "--git-remote-template",
        default=None,
        help="Optional git remote template with {owner} and {repo}; overrides --git-remote-base.",
    )
    return parser


def _progress_path(output_dir: Path) -> Path:
    return output_dir / "task_progress.csv"


def _load_completed_task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("status") or "").strip().lower() != "done":
                continue
            task_id = str(row.get("task_id") or "").strip()
            if task_id:
                completed.add(task_id)
    return completed


def _load_last_completed_index(path: Path) -> int | None:
    if not path.exists():
        return None
    last_index: int | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            status = str(row.get("status") or "").strip().lower()
            if status not in {"done", "accepted"}:
                continue
            try:
                task_id = int(str(row.get("task_id") or "").strip())
            except Exception:
                continue
            if last_index is None or task_id > last_index:
                last_index = task_id
    return last_index


def _append_progress_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fieldnames = [
        "task_id",
        "user",
        "project",
        "commit",
        "file_path",
        "tree_entries",
        "source_files",
        "symbols",
        "status",
        "elapsed_seconds",
        "error",
    ]
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _has_any_cache_artifact(toolbox: GitHubToolbox, record) -> bool:
    owner = record.user
    repo = record.project
    commit = record.commit
    normalized_path = record.file_path.replace("\\", "/").strip("/")
    candidates = [
        toolbox._repo_tree_cache_path(owner, repo, commit, ""),
        toolbox._repo_symbol_index_cache_path(owner, repo, commit),
        toolbox._repo_symbol_index_progress_path(owner, repo, commit),
    ]
    if normalized_path:
        candidates.append(toolbox._repo_file_cache_path(owner, repo, commit, normalized_path))
    return any(path.exists() for path in candidates)


def _find_repair_progress_start(root: Path, explicit_progress: Path | None) -> tuple[int | None, str | None]:
    candidates: list[Path] = []
    if explicit_progress is not None:
        candidates.append(explicit_progress)
    else:
        candidates.extend(
            sorted(
                (
                    path
                    for path in root.glob("outputs_*/task_progress.csv")
                    if path.parent.name != "outputs_repo_cache_only"
                ),
                key=lambda item: item.parent.name,
            )
        )

    best_index: int | None = None
    best_source: str | None = None
    for candidate in candidates:
        last_index = _load_last_completed_index(candidate)
        if last_index is None:
            continue
        if best_index is None or last_index > best_index:
            best_index = last_index
            best_source = candidate.as_posix()
    if best_index is None:
        return None, None
    return best_index + 1, best_source


def _resolve_start_index(
    records: list,
    toolbox: GitHubToolbox,
    completed: set[str],
    use_resume: bool,
    root: Path,
    explicit_progress: Path | None,
) -> tuple[int, str]:
    repair_start_index, repair_source = _find_repair_progress_start(root, explicit_progress)
    if repair_start_index is not None:
        return repair_start_index, f"repair_progress:{repair_source}"

    if use_resume and completed:
        last_done_index = -1
        for index, record in enumerate(records):
            if str(record.task_id) in completed:
                last_done_index = index
        if last_done_index >= 0:
            return last_done_index + 1, "progress"

    for index in range(len(records) - 1, -1, -1):
        if _has_any_cache_artifact(toolbox, records[index]):
            return index, "cache_tail"
    return 0, "csv_head"


def main() -> None:
    args = build_parser().parse_args()
    if args.git_remote_base:
        os.environ["SATD_GIT_REMOTE_BASE"] = args.git_remote_base
    if args.git_remote_template:
        os.environ["SATD_GIT_REMOTE_TEMPLATE"] = args.git_remote_template

    records = load_satd_csv(args.input, limit=args.limit)
    total = len(records)
    progress_path = _progress_path(args.output_dir)
    completed = _load_completed_task_ids(progress_path) if args.resume else set()

    toolbox = GitHubToolbox()
    toolbox.logger = print
    explicit_progress = None
    if args.resume_from_progress is not None:
        explicit_progress = args.resume_from_progress if args.resume_from_progress.is_absolute() else (Path.cwd() / args.resume_from_progress)
    start_index, start_source = _resolve_start_index(records, toolbox, completed, args.resume, Path.cwd(), explicit_progress)
    pending_records = records[start_index:]
    pending = [record for record in pending_records if str(record.task_id) not in completed]
    print(
        f"[cache] tasks={total} pending={len(pending)} start_index={start_index} "
        f"start_source={start_source} repo_cache_dir={toolbox.repo_cache_dir}"
    )

    for absolute_index, record in enumerate(pending_records, start=start_index + 1):
        task_id = str(record.task_id)
        if task_id in completed:
            continue

        log_prefix = f"[task {task_id} {absolute_index}/{total}]"
        started = time.time()
        tree_entries = 0
        source_files = 0
        symbol_count = 0
        print(f"[progress] {absolute_index}/{total} task_id={task_id} begin")
        print(
            f"{log_prefix} cache start project={record.project} file={record.file_path} commit={record.commit}"
        )
        try:
            tree_payload = toolbox._fetch_repo_tree_for_ref(record.user, record.project, "", record.commit)
            if not tree_payload.get("ok"):
                raise RuntimeError(str(tree_payload.get("error") or "repo tree fetch failed"))
            tree_entries = len(tree_payload.get("entries") or [])

            file_payload = toolbox.fetch_repo_file(record.user, record.project, record.file_path, ref=record.commit)
            if not file_payload.get("ok"):
                raise RuntimeError(str(file_payload.get("error") or "repo file fetch failed"))

            repo_paths = [
                str(item.get("path") or "")
                for item in tree_payload.get("entries", [])
                if item.get("type") == "file"
                and toolbox._looks_like_source_file(str(item.get("path") or ""))
            ]
            source_files = len(repo_paths)
            symbols = toolbox._load_symbol_index(
                record.user,
                record.project,
                record.commit,
                repo_paths,
                log_prefix=log_prefix,
            )
            symbol_count = sum(len(items) for items in symbols.values())
            elapsed = time.time() - started
            _append_progress_row(
                progress_path,
                {
                    "task_id": task_id,
                    "user": record.user,
                    "project": record.project,
                    "commit": record.commit,
                    "file_path": record.file_path,
                    "tree_entries": str(tree_entries),
                    "source_files": str(source_files),
                    "symbols": str(symbol_count),
                    "status": "done",
                    "elapsed_seconds": f"{elapsed:.2f}",
                    "error": "",
                },
            )
            print(
                f"{log_prefix} cache done tree_entries={tree_entries} "
                f"source_files={source_files} symbols={symbol_count} elapsed={elapsed:.1f}s"
            )
            completed.add(task_id)
        except Exception as exc:
            elapsed = time.time() - started
            _append_progress_row(
                progress_path,
                {
                    "task_id": task_id,
                    "user": record.user,
                    "project": record.project,
                    "commit": record.commit,
                    "file_path": record.file_path,
                    "tree_entries": str(tree_entries),
                    "source_files": str(source_files),
                    "symbols": str(symbol_count),
                    "status": "error",
                    "elapsed_seconds": f"{elapsed:.2f}",
                    "error": str(exc),
                },
            )
            print(f"{log_prefix} cache error elapsed={elapsed:.1f}s error={exc}")

    print(f"[cache] progress file={progress_path}")


if __name__ == "__main__":
    main()
