from __future__ import annotations

import base64
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from satd_langgraph.agents import OpenAICompatClient, OpenAIFixer
from satd_langgraph.csv_loader import load_satd_csv
from satd_langgraph.github_tools import GitHubToolbox
from satd_langgraph.schema import AnalysisResult, ReviewResult, SATDRecord, record_to_graph_input
from satd_langgraph.workflow import LangGraphSATDWorkflow


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

    def test_same_file_helpers_prefers_comment_relevant_candidates(self) -> None:
        client = OpenAICompatClient.__new__(OpenAICompatClient)
        client.toolbox = GitHubToolbox()
        file_content = """
class RESTClient:
    def perform(self):
        value = close_request()
        return value

    def close_request(self):
        return "ok"

    def failed_request_handler(self):
        return "failed"
"""
        target_function = {
            "found": True,
            "symbol_name": "perform",
            "class_name": "RESTClient",
            "calls": ["close_request"],
            "self_calls": [],
        }

        payload = client._extract_same_file_helpers(file_content, target_function, "# TODO: handle failed requests")

        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["items"][0]["symbol_name"], "close_request")
        self.assertEqual(payload["items"][1]["symbol_name"], "failed_request_handler")

    def test_weak_targeted_test_item_is_filtered(self) -> None:
        client = OpenAICompatClient.__new__(OpenAICompatClient)
        client.toolbox = GitHubToolbox()

        weak_item = {
            "path": "pkg/tests/test_atomic.py",
            "snippet_ok": True,
            "match_reason": "assert_window",
            "excerpt": "def test_atomic_functions():\n    assert True",
        }

        strong = client._strong_targeted_item(
            weak_item,
            symbol_name="test_bug_1333982",
            satd_comment="# XXX: re-enable this test! # fails with -O",
            current_path="pkg/tests/test_dis.py",
            expected_kind="test",
        )

        self.assertFalse(strong)

    def test_lightweight_prompt_ignores_analyzer_but_uses_reviewer_feedback(self) -> None:
        fixer = OpenAIFixer(client=object(), prompt_mode="lightweight")
        state = {
            "task_id": "7",
            "user": "u",
            "project": "p",
            "file_path": "pkg/a.py",
            "satd_comment": "# TODO: restore the old branch",
            "original_code": "def a():\n    pass",
            "repair_feedback": {
                "reject_type": "overwritten",
                "revision_advice": "Keep the original API shape.",
                "key_issues": ["Repair rewrote the control flow."],
                "has_api_shape_drift": True,
                "has_noop_change": False,
                "has_over_edit": True,
                "has_comment_only_problem": False,
                "can_retry": True,
            },
        }
        analysis = AnalysisResult(
            decision="repairable",
            repairable=True,
            repairability_score=0.9,
            intent_clarity=0.9,
            change_locality=0.9,
            semantic_risk=0.2,
            context_sufficiency=0.8,
            verifiability=0.8,
            analyze_score=0.85,
            confidence=0.8,
            satd_type="temporary",
            reason="",
            evidence_summary="",
            risk_level="low",
            context_score=0.8,
            clarity_score=0.9,
            scope_radius="function",
        )

        _, prompt = fixer._build_repair_prompts(
            state=state,
            round_id=2,
            github_context='{"repair_context": {}}',
            repair_evidence_mode="weak",
            snapshot_alignment_status="aligned",
            contextual_repair_hint="Prefer the smallest edit.",
        )

        self.assertIn("### Reviewer feedback from previous attempt:", prompt)
        self.assertIn("Reject type: overwritten", prompt)
        self.assertIn("Revision advice: Keep the original API shape.", prompt)
        self.assertNotIn("Analyzer decision:", prompt)
        self.assertNotIn("Analyzer type:", prompt)

    def test_compact_shared_context_excludes_review_context(self) -> None:
        client = OpenAICompatClient.__new__(OpenAICompatClient)
        bundle = {
            "metadata": {},
            "base_context": {"file_path": "pkg/a.py", "target_function": {"found": True}},
            "repair_context": {"same_file_helpers": {"count": 1, "items": []}},
            "review_context": {"risk_indicators": ["api_shape_drift"]},
        }

        compact = client.compact_shared_context(bundle)

        self.assertIn("repair_context", compact)
        self.assertNotIn("review_context", compact)

    def test_reviewer_retry_is_limited_to_one_round(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)
        first_review = ReviewResult(
            round_id=1,
            approved=False,
            review_score=0.6,
            problem_alignment=0.6,
            minimality=0.4,
            semantic_preservation=0.7,
            internal_consistency=0.7,
            issues=["Repair rewrote too much."],
            revision_advice="Keep the original API shape.",
            reject_type="overwritten",
            rationale="too broad",
        )
        feedback = workflow._build_repair_feedback(first_review)
        self.assertTrue(feedback["can_retry"])
        self.assertTrue(
            workflow._should_retry_after_review(
                {
                    "round_id": 1,
                    "max_rounds": 2,
                    "latest_review": first_review,
                    "repair_feedback": feedback,
                }
            )
        )
        self.assertFalse(
            workflow._should_retry_after_review(
                {
                    "round_id": 2,
                    "max_rounds": 2,
                    "latest_review": first_review,
                    "repair_feedback": feedback,
                }
            )
        )

if __name__ == "__main__":
    unittest.main()
