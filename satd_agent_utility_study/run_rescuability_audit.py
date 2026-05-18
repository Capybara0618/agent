from __future__ import annotations

import argparse
from pathlib import Path

from .rescuability import run_rescuability_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a lightweight agent-facing SATD rescuability audit.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-evidence", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run_rescuability_audit(
        args.input,
        args.output,
        max_evidence=args.max_evidence,
        resume=args.resume,
        verbose=args.verbose,
        flush_every=args.flush_every,
    )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.proxy_label] = counts.get(row.proxy_label, 0) + 1
    print(f"wrote {len(rows)} rows to {args.output}")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")


if __name__ == "__main__":
    main()
