from satd_agent_utility_study.splitting import build_split_assignments
from satd_langgraph.schema import SATDRecord


def test_split_assignments_are_disjoint_and_failure_only() -> None:
    records = [
        SATDRecord(str(index), "remove old code" if index % 2 else "handle error", "def f():\n    pass", "", "u", f"p{index % 3}", "x.py", "c", "NO")
        for index in range(12)
    ] + [
        SATDRecord("success", "remove old code", "def f():\n    pass", "", "u", "p", "x.py", "c", "YES")
    ]

    rows = build_split_assignments(records, dev_size=4, holdout_size=4, seed=7)
    dev = {row.task_id for row in rows if row.split == "dev"}
    holdout = {row.task_id for row in rows if row.split == "holdout"}

    assert len(dev) == 4
    assert len(holdout) == 4
    assert not dev & holdout
    assert "success" not in dev
    assert "success" not in holdout

