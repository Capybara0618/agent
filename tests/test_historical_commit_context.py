from __future__ import annotations

import base64
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from satd_langgraph.agents import OpenAICompatClient
from satd_langgraph.csv_loader import load_satd_csv
from satd_langgraph.github_tools import GitHubToolbox
from satd_langgraph.schema import SATDRecord, record_to_graph_input


class HistoricalCommitContextTests(unittest.TestCase):
    def test_csv_loader_reads_commit_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["index", "SATD_comment", "original_code", "manual_code", "user", "project", "file_path", "commit", "EM"]
                )
                writer.writerow(["1", "# TODO", "def a():\n    pass", "def a():\n    return 1", "u", "p", "pkg/a.py", "abc123", "NO"])

            records = load_satd_csv(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].commit, "abc123")

    def test_record_to_graph_input_carries_commit(self) -> None:
        record = SATDRecord(
            task_id="1",
            satd_comment="# TODO",
            original_code="def a():\n    pass",
            manual_code="def a():\n    return 1",
            user="u",
            project="p",
            file_path="pkg/a.py",
            commit="deadbeef",
            em_label="NO",
        )

        state = record_to_graph_input(record, max_rounds=2)

        self.assertEqual(state["commit"], "deadbeef")

    def test_fetch_repo_file_uses_ref_query_parameter(self) -> None:
        toolbox = GitHubToolbox()
        seen_urls: list[str] = []
        encoded = base64.b64encode(b"print('hello')\n").decode("ascii")

        def fake_github_json(url: str):
            seen_urls.append(url)
            return {"path": "pkg/a.py", "sha": "1", "download_url": "https://example.invalid", "content": encoded}

        with patch.object(toolbox, "_github_json", side_effect=fake_github_json):
            payload = toolbox.fetch_repo_file("u", "p", "pkg/a.py", ref="abc123")

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["ref"], "abc123")
        self.assertIn("ref=abc123", seen_urls[0])

    def test_fetch_repo_tree_uses_git_tree_for_ref(self) -> None:
        toolbox = GitHubToolbox()
        seen_urls: list[str] = []

        def fake_github_json(url: str):
            seen_urls.append(url)
            return {
                "tree": [
                    {"path": "pkg/a.py", "type": "blob"},
                    {"path": "pkg/tests/test_a.py", "type": "blob"},
                ]
            }

        with patch.object(toolbox, "_github_json", side_effect=fake_github_json):
            payload = toolbox.fetch_repo_tree("u", "p", "pkg", ref="abc123")

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["ref"], "abc123")
        self.assertIn("/git/trees/abc123?recursive=1", seen_urls[0])
        self.assertEqual([item["path"] for item in payload["entries"]], ["pkg/a.py", "pkg/tests/test_a.py"])

    def test_augment_bundle_metadata_records_historical_status(self) -> None:
        client = OpenAICompatClient.__new__(OpenAICompatClient)
        bundle = {
            "commit": "abc123",
            "metadata": {"context_commit": "abc123"},
            "base_context": {
                "target_file": {"ok": False, "error": "historical_file_missing: 404", "path": "pkg/a.py", "ref": "abc123"},
                "satd_window": {"found": False},
                "enclosing_symbol": {"found": False},
            },
            "repair_context": {},
            "review_context": {},
        }

        updated = client._augment_bundle_metadata(bundle)

        self.assertEqual(updated["metadata"]["context_commit"], "abc123")
        self.assertEqual(updated["metadata"]["snapshot_alignment_status"], "historical_file_missing")
        self.assertTrue(updated["metadata"]["historical_snapshot_mismatch"])

if __name__ == "__main__":
    unittest.main()
