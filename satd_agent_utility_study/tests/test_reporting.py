from pathlib import Path

from satd_agent_utility_study.io_utils import write_csv_rows
from satd_agent_utility_study.reporting import build_reports, select_top_variants


def test_reporting_ranks_variants_by_real_em_gain(tmp_path: Path) -> None:
    write_csv_rows(
        tmp_path / "runs" / "dev" / "current_agent" / "results.csv",
        [
            {"task_id": "1", "em_label": "NO", "exact_match": False, "status": "accepted"},
            {"task_id": "2", "em_label": "YES", "exact_match": True, "status": "accepted"},
        ],
    )
    write_csv_rows(
        tmp_path / "runs" / "dev" / "two_stage_top5" / "results.csv",
        [
            {"task_id": "1", "em_label": "NO", "exact_match": True, "status": "accepted", "repository_evidence_count": 3},
            {"task_id": "2", "em_label": "YES", "exact_match": True, "status": "accepted", "repository_evidence_count": 2},
        ],
    )
    build_reports(tmp_path)

    ranked = select_top_variants(tmp_path, "dev", baseline_variant="current_agent", top_k=1)
    summary = (tmp_path / "context_utility_results.csv").read_text(encoding="utf-8")

    assert ranked == ["two_stage_top5"]
    assert "two_stage_top5" in summary
    assert "1" in summary
