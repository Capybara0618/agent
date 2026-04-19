from __future__ import annotations

import base64
import csv
import tempfile
import unittest
from pathlib import Path
import os
from unittest.mock import patch

from satd_langgraph.agents import OpenAICompatClient, OpenAIFixer, OpenAISelector
from satd_langgraph.csv_loader import load_satd_csv
from satd_langgraph.github_tools import GitHubToolbox
from satd_langgraph.schema import (
    AnalysisResult,
    EditConstraint,
    MethodInquiryResult,
    RepairAttempt,
    RetrievedMethodContext,
    ReviewResult,
    SATDRecord,
    SelectorDecision,
    UncertaintyItem,
    record_to_graph_input,
    trace_from_state,
)
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
        with tempfile.TemporaryDirectory() as tmpdir:
            toolbox = GitHubToolbox()
            toolbox.repo_cache_dir = Path(tmpdir)
            toolbox.repo_cache_dir.mkdir(parents=True, exist_ok=True)
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
        with tempfile.TemporaryDirectory() as tmpdir:
            toolbox = GitHubToolbox()
            toolbox.repo_cache_dir = Path(tmpdir)
            toolbox.repo_cache_dir.mkdir(parents=True, exist_ok=True)
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

        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["items"][0]["symbol_name"], "close_request")

    def test_module_symbols_use_direct_global_references_without_keyword_match(self) -> None:
        client = OpenAICompatClient.__new__(OpenAICompatClient)
        client.toolbox = GitHubToolbox()
        file_content = """
VALUE_ERROR = ValueError
DEFAULT_LIMIT = 2

def perform():
    raise VALUE_ERROR("boom")
"""
        target_function = {
            "found": True,
            "global_uses": {"VALUE_ERROR"},
        }

        payload = client._extract_module_symbols(file_content, target_function, "# TODO: change it to ValueError")

        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["items"][0]["symbol_name"], "VALUE_ERROR")
        self.assertEqual(payload["items"][0]["relevance"], ["direct_global_use"])

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
        fixer = OpenAIFixer(client=object())
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
            method_inquiry=MethodInquiryResult(required_methods=["helper"], reason="needed for repair"),
            retrieved_method_contexts=[
                RetrievedMethodContext(
                    method_name="helper",
                    path="pkg/a.py",
                    class_name=None,
                    start_line=5,
                    end_line=7,
                    source="def helper():\n    return 1",
                    found=True,
                )
            ],
            missing_method_names=[],
            edit_constraints=[],
        )

        self.assertIn("### Reviewer feedback from previous attempt:", prompt)
        self.assertIn("Reject type: overwritten", prompt)
        self.assertIn("Revision advice: Keep the original API shape.", prompt)
        self.assertIn("### Supporting method/symbol evidence:", prompt)
        self.assertNotIn("Analyzer decision:", prompt)
        self.assertNotIn("Analyzer type:", prompt)

    def test_no_context_satd_uses_minimal_prompt(self) -> None:
        fixer = OpenAIFixer(client=object())
        state = {
            "task_id": "9",
            "user": "u",
            "project": "p",
            "file_path": "pkg/a.py",
            "satd_comment": "# TODO: temporary workaround, remove this log branch later",
            "original_code": "if debug:\n    logger.info('tmp')\nreturn run()",
            "repair_feedback": {},
            "satd_route_type": "remove_temporary",
            "candidate_mode": "baseline_context",
        }

        system_prompt, user_prompt = fixer._build_repair_prompts(
            state=state,
            round_id=1,
            method_inquiry=MethodInquiryResult(required_methods=[], reason="no_context"),
            retrieved_method_contexts=[],
            missing_method_names=[],
            edit_constraints=[],
        )

        self.assertEqual(system_prompt, "")
        self.assertIn("Repair the code according to the SATD comment. Output code only.", user_prompt)
        self.assertIn("Code:", user_prompt)
        self.assertIn("SATD:", user_prompt)
        self.assertNotIn("### Supporting method/symbol evidence:", user_prompt)
        self.assertNotIn("### Hard rules:", user_prompt)

    def test_generic_prompt_uses_supporting_method_symbol_evidence_block(self) -> None:
        fixer = OpenAIFixer(client=object())
        full_method_source = (
            "def close_request(self, request):\n"
            "    if request.ok:\n"
            "        self.mark_closed(request)\n"
            "        return True\n"
            "    self.log_failure(request)\n"
            "    self.retry_queue.append(request)\n"
            "    return False"
        )
        state = {
            "task_id": "9",
            "user": "u",
            "project": "p",
            "file_path": "pkg/a.py",
            "satd_comment": "# TODO handle failed requests",
            "original_code": "def close_requests(self, failed):\n    self.close_request(failed[0])",
            "repair_feedback": {},
            "satd_route_type": "generic",
            "candidate_mode": "baseline_context",
        }

        system_prompt, user_prompt = fixer._build_repair_prompts(
            state=state,
            round_id=1,
            method_inquiry=MethodInquiryResult(
                required_methods=["close_request"],
                reason="failure path semantics",
                method_notes=[
                    {
                        "method_name": "close_request",
                        "why_blocking": "This method controls the failed-request path that the SATD wants to repair.",
                    }
                ],
            ),
            retrieved_method_contexts=[
                RetrievedMethodContext(
                    method_name="close_request",
                    path="pkg/client.py",
                    class_name=None,
                    start_line=10,
                    end_line=14,
                    source=full_method_source,
                    found=True,
                )
            ],
            missing_method_names=[],
            edit_constraints=[EditConstraint(focus_point="close_request", must_do="Keep the repair local to the failed path.")],
        )

        self.assertIn("### SATD comment:", user_prompt)
        self.assertIn("### Original code:", user_prompt)
        self.assertIn("### Supporting method/symbol evidence:", user_prompt)
        self.assertIn("### Hard rules:", user_prompt)
        self.assertIn("Repair the SATD in the code block below, using supporting evidence only as auxiliary context.", user_prompt)
        self.assertIn("Output only the repaired code.", user_prompt)
        self.assertIn("Return valid JSON only with keys: repair_plan, repaired_code, changed_scope, confidence, notes.", system_prompt)
        self.assertIn("Make the smallest local repair and return the full repaired code block.", system_prompt)
        self.assertNotIn("### Contextual repair hint", user_prompt)
        self.assertNotIn("### Repair core evidence", user_prompt)
        self.assertNotIn("Missing methods", user_prompt)
        self.assertNotIn("Low-confidence methods skipped", user_prompt)
        self.assertNotIn("why_blocking", user_prompt)
        self.assertNotIn("what_to_learn", user_prompt)
        self.assertNotIn("Must do", user_prompt)
        self.assertNotIn("Do not add helper functions.", user_prompt)
        self.assertNotIn("Do not rewrite unrelated lines.", user_prompt)
        self.assertNotIn("signature:", user_prompt)
        self.assertNotIn("evidence_slice:", user_prompt)
        self.assertNotIn("pkg/client.py:10", user_prompt)
        self.assertIn("- method: close_request", user_prompt)
        self.assertIn("local call:", user_prompt)
        self.assertIn("self.close_request(failed[0])", user_prompt)
        self.assertIn("method behavior:", user_prompt)
        self.assertIn("def close_request(self, request):", user_prompt)
        self.assertIn("self.mark_closed(request)", user_prompt)
        self.assertIn("self.retry_queue.append(request)", user_prompt)
        self.assertIn("selection reason:", user_prompt)
        self.assertIn("failed-request path", user_prompt)
        self.assertIn("- Edit only the smallest local block nearest to the SATD comment.", user_prompt)
        self.assertIn("- Preserve the original signature.", user_prompt)
        self.assertIn("- Do not broaden control flow unless the retrieved evidence directly requires it.", user_prompt)

    def test_long_method_behavior_is_condensed_to_signature_and_relevant_block(self) -> None:
        fixer = OpenAIFixer(client=object())
        long_method_source = (
            "def close_request(self, request):\n"
            "    metrics = self.metrics\n"
            "    audit = self.audit_log\n"
            "    audit.append(request.id)\n"
            "    if request.ok:\n"
            "        self.mark_closed(request)\n"
            "        metrics['closed'] += 1\n"
            "        return True\n"
            "    self.log_failure(request)\n"
            "    self.retry_queue.append(request)\n"
            "    self.metrics['failed'] += 1\n"
            "    return False\n"
        )
        state = {
            "task_id": "10",
            "user": "u",
            "project": "p",
            "file_path": "pkg/a.py",
            "satd_comment": "# TODO handle failed requests",
            "original_code": "def close_requests(self, failed):\n    self.close_request(failed[0])",
            "repair_feedback": {},
            "satd_route_type": "generic",
            "candidate_mode": "baseline_context",
        }

        _, user_prompt = fixer._build_repair_prompts(
            state=state,
            round_id=1,
            method_inquiry=MethodInquiryResult(
                required_methods=["close_request"],
                reason="failure path semantics",
                method_notes=[
                    {
                        "method_name": "close_request",
                        "why_blocking": "This method decides the local failed-request behavior.",
                    }
                ],
            ),
            retrieved_method_contexts=[
                RetrievedMethodContext(
                    method_name="close_request",
                    path="pkg/client.py",
                    class_name=None,
                    start_line=10,
                    end_line=21,
                    source=long_method_source,
                    found=True,
                )
            ],
            missing_method_names=[],
            edit_constraints=[],
        )

        self.assertIn("def close_request(self, request):", user_prompt)
        self.assertIn("if request.ok:", user_prompt)
        self.assertIn("self.retry_queue.append(request)", user_prompt)
        self.assertIn("return False", user_prompt)
        self.assertNotIn("metrics = self.metrics", user_prompt)
        self.assertNotIn("audit = self.audit_log", user_prompt)

    def test_generic_context_gate_limits_to_two_items(self) -> None:
        fixer = OpenAIFixer(client=object(), max_method_contexts=2)
        state = {
            "satd_route_type": "generic",
            "satd_comment": "# TODO use close_failed_request for failed items",
            "original_code": "def f():\n    close_request(x)\n    close_failed_request(x)\n",
        }
        inquiry = MethodInquiryResult(required_methods=["close_request", "close_failed_request"], reason="needed")
        gated = fixer._apply_method_context_gate(
            state=state,
            method_inquiry=inquiry,
            retrieved_method_contexts=[
                RetrievedMethodContext("close_request", "a.py", None, 1, 3, "def close_request(x):\n    return x", True),
                RetrievedMethodContext("close_failed_request", "a.py", None, 5, 9, "def close_failed_request(x):\n    if x:\n        return x", True),
                RetrievedMethodContext("helper", "a.py", None, 10, 12, "def helper():\n    return 1", True),
            ],
        )

        self.assertEqual(len(gated), 2)
        self.assertEqual(gated[0].method_name, "close_failed_request")

    def test_extract_uncertainty_candidates_keeps_callable_comment_symbols_and_demotes_values(self) -> None:
        fixer = OpenAIFixer(client=object(), max_method_contexts=2)

        items = fixer._extract_uncertainty_candidates_from_satd_code(
            satd_comment=(
                "# TODO call bot.send_message_to_user and close_failed_request when backend is None; "
                "do not rely on config.default_backend"
            ),
            original_code=(
                "def repair(bot, backend, failed):\n"
                "    if backend is None:\n"
                "        return close_failed_request(failed)\n"
                "    return failed\n"
            ),
        )

        retrieval_names = {item.name for item in items if item.retrieval_eligible}
        item_kinds = {item.name: item.kind for item in items}

        self.assertIn("bot.send_message_to_user", retrieval_names)
        self.assertIn("close_failed_request", retrieval_names)
        self.assertNotIn("config.default_backend", retrieval_names)
        self.assertEqual(item_kinds["config.default_backend"], "value")
        self.assertEqual(item_kinds["backend is None"], "condition")
        backend_item = next(item for item in items if item.name == "config.default_backend")
        self.assertEqual(backend_item.raw_name, "config.default_backend")
        self.assertEqual(backend_item.normalized_name, "config.default_backend")
        self.assertIn("do not rely on config.default_backend", backend_item.source_excerpt)

    def test_identify_required_methods_uses_single_llm_selection_route(self) -> None:
        class _Client:
            def generate_json(self, *args, **kwargs):
                return {
                    "required_methods": [
                        {
                            "method_name": "bot.send_message_to_user",
                            "selection_reason": "This API is the expected notification action in the fallback branch.",
                        },
                        {
                            "method_name": "close_failed_request",
                            "selection_reason": "This method controls the failed-request branch that the SATD wants to repair.",
                        },
                        {
                            "method_name": "config.default_backend",
                            "selection_reason": "This should be ignored because it is not retrieval-eligible here.",
                        },
                    ],
                    "reason": "Need the notification and failed-request behavior before repairing.",
                }

        fixer = OpenAIFixer(client=_Client(), max_method_contexts=2)
        state = {
            "task_id": "1",
            "satd_route_type": "generic",
            "satd_comment": (
                "# TODO call bot.send_message_to_user and close_failed_request when backend is None; "
                "do not rely on config.default_backend"
            ),
            "original_code": (
                "def repair(bot, backend, failed):\n"
                "    if backend is None:\n"
                "        bot.send_message_to_user('fallback')\n"
                "        return close_failed_request(failed)\n"
                "    return failed\n"
            ),
        }

        inquiry = fixer.identify_required_methods(state)

        self.assertIn("bot.send_message_to_user", inquiry.required_methods)
        self.assertIn("close_failed_request", inquiry.required_methods)
        self.assertNotIn("config.default_backend", inquiry.required_methods)
        self.assertEqual(
            inquiry.reason,
            "Need the notification and failed-request behavior before repairing.",
        )
        self.assertTrue(inquiry.method_notes)

    def test_generic_context_gate_prefers_high_confidence_semantic_evidence(self) -> None:
        fixer = OpenAIFixer(client=object(), max_method_contexts=2)
        state = {
            "satd_route_type": "generic",
            "satd_comment": "# TODO: gradgradcheck does not work correctly yet for complex",
            "original_code": (
                "def test_cholesky():\n"
                "    gradcheck(func, [root, upper])\n"
                "    gradgradcheck(func, [root, upper])\n"
                "    root = torch.matmul(root, root.transpose(-1, -2).conj())\n"
                "    root.requires_grad_()\n"
            ),
        }
        inquiry = MethodInquiryResult(
            required_methods=["gradcheck", "torch.matmul", "root.requires_grad_"],
            reason="llm_required_method_selection",
        )
        contexts = fixer._enrich_retrieved_method_contexts(
            state=state,
            method_inquiry=inquiry,
            retrieved_method_contexts=[
                RetrievedMethodContext(
                    "gradcheck",
                    "torch/autograd/gradcheck.py",
                    None,
                    10,
                    14,
                    "def gradcheck(*args, **kwargs):\n    return True",
                    True,
                ),
                RetrievedMethodContext(
                    "matmul",
                    "torch/utils/_benchmark/examples/compare.py",
                    "FauxTorch",
                    43,
                    44,
                    "def matmul(self, *args, **kwargs):\n    return self.extra_overhead(self._real_torch.matmul(*args, **kwargs))",
                    True,
                ),
                RetrievedMethodContext(
                    "requires_grad_",
                    "torch/distributed/nn/api/remote_module.py",
                    "_RemoteModule",
                    300,
                    301,
                    "def requires_grad_(self, requires_grad=True):\n    _raise_not_supported(self.requires_grad_.__name__)",
                    True,
                ),
            ],
        )

        gated = fixer._apply_method_context_gate(
            state=state,
            method_inquiry=inquiry,
            retrieved_method_contexts=contexts,
        )

        self.assertEqual([item.method_name for item in gated], ["matmul", "requires_grad_"])

    def test_workflow_repair_context_used_tracks_selected_candidate_prompt(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)

        no_context = RepairAttempt(1, "baseline", "def a():\n    return 1", "function", 0.8, "", "baseline_no_context")
        with_context = RepairAttempt(1, "context", "def a():\n    return helper()", "function", 0.7, "", "baseline_context")
        retrieved = [
            RetrievedMethodContext(
                method_name="helper",
                path="pkg/a.py",
                class_name=None,
                start_line=3,
                end_line=4,
                source="def helper():\n    return 1",
                found=True,
            )
        ]

        self.assertFalse(workflow._repair_attempt_uses_method_evidence("generic", no_context, retrieved))
        self.assertTrue(workflow._repair_attempt_uses_method_evidence("generic", with_context, retrieved))
        self.assertTrue(workflow._repair_attempt_uses_method_evidence("remove_temporary", with_context, retrieved))

    def test_compact_shared_context_excludes_review_context(self) -> None:
        client = OpenAICompatClient.__new__(OpenAICompatClient)
        bundle = {
            "metadata": {},
            "base_context": {"file_path": "pkg/a.py", "target_function": {"found": True}},
            "repair_context": {
                "same_file_helpers": {
                    "count": 1,
                    "items": [
                        {
                            "symbol_name": "close_failed_request",
                            "start_line": 10,
                            "end_line": 18,
                            "relevance": ["direct_call_reference"],
                            "source": "def close_failed_request(self, fail):\n    return fail",
                        }
                    ],
                }
            },
            "review_context": {"risk_indicators": ["api_shape_drift"]},
        }

        compact = client.compact_shared_context(bundle)

        self.assertIn("repair_context", compact)
        self.assertIn("evidence_cards", compact)
        self.assertIn("close_failed_request", compact)
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

    def test_dual_candidate_without_selector_prefers_no_context_on_tie(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)
        no_context = RepairAttempt(
            round_id=1,
            repair_plan="baseline",
            repaired_code="def a():\n    return 1",
            changed_scope="function",
            confidence=0.8,
            notes="",
            candidate_mode="no_context",
        )
        with_context = RepairAttempt(
            round_id=1,
            repair_plan="context",
            repaired_code="def a():\n    return 2",
            changed_scope="function",
            confidence=0.8,
            notes="",
            candidate_mode="context",
        )

        selected = workflow._select_candidate_without_selector([with_context, no_context])

        self.assertEqual(selected.candidate_mode, "no_context")

    def test_selector_fallback_chooses_best_candidate_when_disabled(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)
        workflow.use_selector = False
        no_context = RepairAttempt(
            round_id=1,
            repair_plan="baseline",
            repaired_code="def a():\n    return 1",
            changed_scope="function",
            confidence=0.7,
            notes="",
            candidate_mode="no_context",
        )
        with_context = RepairAttempt(
            round_id=1,
            repair_plan="context",
            repaired_code="def a():\n    return 2",
            changed_scope="function",
            confidence=0.6,
            notes="",
            candidate_mode="context",
        )
        decision = workflow._run_selector(
            {"round_id": 0, "task_id": "1", "satd_route_type": "generic"},
            [with_context, no_context],
        )

        self.assertEqual(decision.selected_candidate_mode, "no_context")
        self.assertEqual(decision.selected_index, 1)
        self.assertGreaterEqual(decision.confidence, 0.45)

    def test_selector_decision_picks_indexed_candidate(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)
        candidates = [
            RepairAttempt(1, "baseline", "def a():\n    return 1", "function", 0.6, "", "baseline_no_context"),
            RepairAttempt(1, "typed", "def a():\n    return 2", "function", 0.5, "", "typed_no_context"),
        ]
        decision = SelectorDecision(
            round_id=1,
            satd_route_type="type_annotation",
            selected_candidate_mode="typed_no_context",
            selected_index=1,
            confidence=0.8,
            rationale="typed candidate is closer",
            candidate_scores=[],
        )

        selected = workflow._select_repair_from_decision(candidates, decision)

        self.assertEqual(selected.candidate_mode, "typed_no_context")

    def test_selector_bias_prefers_no_context_for_generic_near_tie(self) -> None:
        selector = OpenAISelector(client=object())
        candidates = [
            RepairAttempt(1, "context", "def a():\n    return 2", "function", 0.7, "", "baseline_context"),
            RepairAttempt(1, "baseline", "def a():\n    return 1", "function", 0.7, "", "baseline_no_context"),
        ]

        selected_index, selected_mode, _, rationale, scores = selector._apply_route_bias(
            satd_route_type="generic",
            candidates=candidates,
            selected_index=0,
            selected_candidate_mode="baseline_context",
            confidence=0.8,
            rationale="LLM preferred context candidate.",
            candidate_scores=[
                {"index": 0, "candidate_mode": "baseline_context", "score": 0.84},
                {"index": 1, "candidate_mode": "baseline_no_context", "score": 0.82},
            ],
        )

        self.assertEqual(selected_index, 1)
        self.assertEqual(selected_mode, "baseline_no_context")
        self.assertIn("Route bias applied for generic", rationale)
        self.assertEqual(len(scores), 2)

    def test_selector_bias_prefers_typed_no_context_for_remove_temporary_near_tie(self) -> None:
        selector = OpenAISelector(client=object())
        candidates = [
            RepairAttempt(1, "baseline", "def a():\n    return 1", "function", 0.7, "", "baseline_no_context"),
            RepairAttempt(1, "typed", "def a():\n    return 1", "function", 0.7, "", "typed_no_context"),
        ]

        selected_index, selected_mode, _, rationale, _ = selector._apply_route_bias(
            satd_route_type="remove_temporary",
            candidates=candidates,
            selected_index=0,
            selected_candidate_mode="baseline_no_context",
            confidence=0.75,
            rationale="LLM weakly preferred baseline.",
            candidate_scores=[
                {"index": 0, "candidate_mode": "baseline_no_context", "score": 0.81},
                {"index": 1, "candidate_mode": "typed_no_context", "score": 0.80},
            ],
        )

        self.assertEqual(selected_index, 1)
        self.assertEqual(selected_mode, "typed_no_context")
        self.assertIn("Route bias applied for remove_temporary", rationale)

    def test_selector_bias_does_not_override_clear_gap(self) -> None:
        selector = OpenAISelector(client=object())
        candidates = [
            RepairAttempt(1, "context", "def a():\n    return 2", "function", 0.9, "", "baseline_context"),
            RepairAttempt(1, "baseline", "def a():\n    return 1", "function", 0.4, "", "baseline_no_context"),
        ]

        selected_index, selected_mode, _, rationale, _ = selector._apply_route_bias(
            satd_route_type="generic",
            candidates=candidates,
            selected_index=0,
            selected_candidate_mode="baseline_context",
            confidence=0.9,
            rationale="LLM strongly preferred context candidate.",
            candidate_scores=[
                {"index": 0, "candidate_mode": "baseline_context", "score": 0.90},
                {"index": 1, "candidate_mode": "baseline_no_context", "score": 0.70},
            ],
        )

        self.assertEqual(selected_index, 0)
        self.assertEqual(selected_mode, "baseline_context")
        self.assertNotIn("Route bias applied", rationale)

    def test_infer_satd_route_type_uses_generic_rules(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)

        self.assertEqual(
            workflow._infer_satd_route_type({"satd_comment": "# pyre-fixme[3]: Return type must be annotated."}),
            "type_annotation",
        )
        self.assertEqual(
            workflow._infer_satd_route_type({"satd_comment": "# TODO: remove this temporary workaround once fixed"}),
            "remove_temporary",
        )
        self.assertEqual(
            workflow._infer_satd_route_type({"satd_comment": "# TODO: switch to full_path below"}),
            "replace_symbol",
        )

    def test_trace_preserves_candidate_histories(self) -> None:
        state = record_to_graph_input(
            SATDRecord(
                task_id="1",
                satd_comment="# TODO",
                original_code="def a():\n    pass",
                manual_code="def a():\n    return 1",
                user="u",
                project="p",
                file_path="pkg/a.py",
                commit="abc123",
                em_label="NO",
            ),
            max_rounds=2,
        )
        state["status"] = "accepted"
        state["round_id"] = 1
        state["final_repaired_code"] = "def a():\n    return 1"
        state["candidate_repairs"] = [
            RepairAttempt(1, "baseline", "def a():\n    return 1", "function", 0.8, "", "no_context"),
            RepairAttempt(1, "context", "def a():\n    return 2", "function", 0.7, "", "context"),
        ]
        state["candidate_reviews"] = [
            ReviewResult(1, True, 0.9, 0.9, 0.8, 0.9, 0.9, [], "", None, "ok", False, "no_context"),
            ReviewResult(1, False, 0.4, 0.5, 0.4, 0.5, 0.5, ["too broad"], "smaller", "overwritten", "no", False, "context"),
        ]

        trace = trace_from_state(state, "NO")

        self.assertEqual(len(trace.candidate_repairs), 2)
        self.assertEqual(len(trace.candidate_reviews), 2)
        self.assertEqual(trace.candidate_repairs[0]["candidate_mode"], "no_context")
        self.assertEqual(trace.candidate_reviews[1]["candidate_mode"], "context")

    def test_method_inquiry_coerces_method_names_without_limit(self) -> None:
        fixer = OpenAIFixer(client=object(), max_method_contexts=2)

        inquiry = fixer._coerce_method_inquiry(
            {
                "required_methods": ["self.helper", "helper()", "  other_method  ", "third_method"],
                "reason": "Need helper behavior.",
            }
        )

        self.assertEqual(inquiry.required_methods, ["helper", "other_method", "third_method"])
        self.assertEqual(inquiry.reason, "Need helper behavior.")


    def test_method_inquiry_filters_low_quality_generic_names(self) -> None:
        fixer = OpenAIFixer(client=object(), max_method_contexts=5)

        inquiry = fixer._coerce_method_inquiry(
            {
                "required_methods": ["time", "get", "logger.info", "values", "RolloutSamplerForSBI.get_dim_data", "self.helper"],
            }
        )

        self.assertEqual(inquiry.required_methods, ["RolloutSamplerForSBI.get_dim_data", "helper"])

    def test_method_inquiry_keeps_per_method_selection_reasons(self) -> None:
        fixer = OpenAIFixer(client=object(), max_method_contexts=5)

        inquiry = fixer._coerce_method_inquiry(
            {
                "required_methods": [
                    {
                        "method_name": "close_failed_request",
                        "selection_reason": "This method controls the failed-request branch that the SATD wants to repair.",
                    },
                    {
                        "method_name": "config.default_backend",
                        "selection_reason": "Should be ignored because it is not an allowed retrieval method in this test.",
                    },
                ],
                "reason": "Need the failed-request behavior before repair.",
            },
            allowed_methods=["close_failed_request"],
        )

        self.assertEqual(inquiry.required_methods, ["close_failed_request"])
        self.assertEqual(inquiry.reason, "Need the failed-request behavior before repair.")
        self.assertEqual(len(inquiry.method_notes), 1)
        self.assertEqual(inquiry.method_notes[0]["method_name"], "close_failed_request")
        self.assertIn("failed-request branch", inquiry.method_notes[0]["why_blocking"])

    def test_identify_required_methods_does_not_special_case_remove_temporary(self) -> None:
        class _Client:
            def generate_json(self, *args, **kwargs):
                raise AssertionError("LLM should not be called for remove_temporary after heuristic filtering.")

        fixer = OpenAIFixer(client=_Client(), max_method_contexts=5)
        state = record_to_graph_input(
            SATDRecord(
                task_id="1",
                satd_comment="# TODO remove this temporary workaround",
                original_code="def f():\n    return helper()",
                manual_code="def f():\n    return 1",
                user="u",
                project="p",
                file_path="pkg/a.py",
                commit="abc123",
                em_label="NO",
            ),
            max_rounds=1,
        )
        state["satd_route_type"] = "remove_temporary"

        inquiry = fixer.identify_required_methods(state)

        self.assertEqual(inquiry.required_methods, [])
        self.assertEqual(inquiry.reason, "rule_without_method_context:remove_temporary")

    def test_identify_required_methods_keeps_llm_selected_methods_without_postfilter(self) -> None:
        class _Client:
            def generate_json(self, *args, **kwargs):
                return {
                    "required_methods": [
                        {"method_name": "os.path.join", "selection_reason": "Path construction affects the repair behavior."},
                        {"method_name": "service.resolve", "selection_reason": "Resolver semantics are central to the SATD."},
                        {"method_name": "json.loads", "selection_reason": "Parsed data shape affects the downstream resolver call."},
                        {"method_name": "assertEqual", "selection_reason": "Assertion semantics constrain the expected local behavior."},
                    ],
                    "reason": "Selected indispensable methods.",
                }

        fixer = OpenAIFixer(client=_Client(), max_method_contexts=5)
        state = record_to_graph_input(
            SATDRecord(
                task_id="1",
                satd_comment="# TODO fix the resolver behavior",
                original_code=(
                    "def f(service, raw, left, right):\n"
                    "    path = os.path.join(left, right)\n"
                    "    data = json.loads(raw)\n"
                    "    service.resolve(data)\n"
                    "    self.assertEqual(path, left)\n"
                ),
                manual_code="",
                user="u",
                project="p",
                file_path="pkg/a.py",
                commit="abc123",
                em_label="NO",
            ),
            max_rounds=1,
        )
        state["satd_route_type"] = "generic"

        inquiry = fixer.identify_required_methods(state)

        self.assertEqual(
            inquiry.required_methods,
            ["os.path.join", "service.resolve", "json.loads", "assertEqual"],
        )
        self.assertEqual(inquiry.reason, "Selected indispensable methods.")

    def test_identify_required_methods_skips_documentation_comments_by_rule(self) -> None:
        class _Client:
            def generate_json(self, *args, **kwargs):
                raise AssertionError("LLM should not be called for documentation-related SATD comments.")

        fixer = OpenAIFixer(client=_Client(), max_method_contexts=5)
        state = {
            "task_id": "1",
            "satd_route_type": "generic",
            "satd_comment": "# TODO document why helper() is needed here",
            "original_code": "def f():\n    return helper() if flag else False\n",
        }

        inquiry = fixer.identify_required_methods(state)

        self.assertEqual(inquiry.required_methods, [])
        self.assertEqual(inquiry.reason, "rule_without_method_context:generic")

    def test_effective_candidate_mode_downgrades_context_when_no_method_contexts_found(self) -> None:
        fixer = OpenAIFixer(client=object(), max_method_contexts=5)

        self.assertEqual(
            fixer._effective_candidate_mode("baseline_context", []),
            "baseline_no_context",
        )
        self.assertEqual(
            fixer._effective_candidate_mode(
                "baseline_context",
                [RetrievedMethodContext("helper", "a.py", None, 1, 2, "def helper():\n    return 1", True)],
            ),
            "baseline_context",
        )

    def test_identify_required_methods_skips_context_for_small_local_value_fix(self) -> None:
        class _Client:
            def generate_json(self, *args, **kwargs):
                raise AssertionError("LLM should not be called for routes that skip method context.")

        fixer = OpenAIFixer(client=_Client(), max_method_contexts=5)
        state = {
            "task_id": "1",
            "satd_route_type": "small_local_value_fix",
            "satd_comment": "# TODO: this line is quite clearly wrong, change it to 644",
            "original_code": "def f():\n    os.chmod(path, 755)\n",
        }

        inquiry = fixer.identify_required_methods(state)

        self.assertEqual(inquiry.required_methods, [])
        self.assertEqual(inquiry.reason, "rule_without_method_context:small_local_value_fix")

    def test_method_inquiry_prompt_only_requests_methods_and_reasons(self) -> None:
        class _Client:
            def __init__(self) -> None:
                self.system_prompt = ""
                self.user_prompt = ""

            def generate_json(self, system_prompt, user_prompt, *args, **kwargs):
                self.system_prompt = system_prompt
                self.user_prompt = user_prompt
                return {
                    "required_methods": [{"method_name": "BayesSimEmbedding", "selection_reason": "Embedding behavior affects the repair."}],
                    "reason": "Need embedding behavior before repair.",
                }

        client = _Client()
        fixer = OpenAIFixer(client=client, max_method_contexts=5)
        state = {
            "task_id": "1",
            "satd_route_type": "generic",
            "satd_comment": "# TODO replace by SNPE-A when available",
            "original_code": "embedding = BayesSimEmbedding(spec=spec)\n",
        }

        fixer.identify_required_methods(state)

        self.assertIn("select only the methods that are strongly relevant", client.system_prompt)
        self.assertNotIn("needs_method_context", client.system_prompt)
        self.assertIn("Output the strongly relevant methods and their selection reasons.", client.user_prompt)

    def test_fetch_method_contexts_prefers_current_file_then_historical_tree(self) -> None:
        toolbox = GitHubToolbox()
        current_content = """
def target():
    return helper()

def helper():
    return 1
"""
        other_content = """
class Service:
    def other_method(self):
        return 2
"""

        def fake_fetch_repo_file(owner, repo, path, ref):
            if path == "pkg/a.py":
                return {"ok": True, "full_content": current_content}
            if path == "pkg/b.py":
                return {"ok": True, "full_content": other_content}
            return {"ok": False, "error": "missing", "path": path, "ref": ref}

        with patch.object(
            toolbox,
            "_fetch_repo_file",
            side_effect=fake_fetch_repo_file,
        ), patch.object(
            toolbox,
            "_fetch_repo_tree_for_ref",
            return_value={"ok": True, "entries": [{"path": "pkg/a.py", "type": "file"}, {"path": "pkg/b.py", "type": "file"}]},
        ):
            contexts = toolbox.fetch_method_contexts("u", "p", "pkg/a.py", ["helper", "other_method"], ref="abc123")

        self.assertEqual(len(contexts), 2)
        self.assertTrue(contexts[0]["found"])
        self.assertEqual(contexts[0]["path"], "pkg/a.py")
        self.assertTrue(contexts[1]["found"])
        self.assertEqual(contexts[1]["path"], "pkg/b.py")

    def test_fetch_method_contexts_matches_call_expression_to_definition(self) -> None:
        toolbox = GitHubToolbox()
        current_content = """
class Client:
    def helper(self):
        return 1
"""

        with patch.object(
            toolbox,
            "_fetch_repo_file",
            return_value={"ok": True, "full_content": current_content},
        ):
            contexts = toolbox.fetch_method_contexts("u", "p", "pkg/a.py", ["self.client.helper()"], ref="abc123")

        self.assertEqual(len(contexts), 1)
        self.assertTrue(contexts[0]["found"])
        self.assertEqual(contexts[0]["method_name"], "helper")
        self.assertEqual(contexts[0]["path"], "pkg/a.py")

    def test_fetch_method_contexts_uses_import_guided_candidate_file(self) -> None:
        toolbox = GitHubToolbox()
        current_content = """
from pkg.helpers import helper

def target():
    return helper()
"""
        helper_content = """
def helper():
    return 1
"""

        def fake_fetch_repo_file(owner, repo, path, ref):
            if path == "pkg/a.py":
                return {"ok": True, "full_content": current_content}
            if path == "pkg/helpers.py":
                return {"ok": True, "full_content": helper_content}
            return {"ok": False, "error": "missing", "path": path, "ref": ref}

        with patch.object(toolbox, "_fetch_repo_file", side_effect=fake_fetch_repo_file), patch.object(
            toolbox,
            "_fetch_repo_tree_for_ref",
            return_value={
                "ok": True,
                "entries": [
                    {"path": "pkg/a.py", "type": "file"},
                    {"path": "pkg/helpers.py", "type": "file"},
                    {"path": "pkg/other.py", "type": "file"},
                ],
            },
        ):
            contexts = toolbox.fetch_method_contexts("u", "p", "pkg/a.py", ["helper"], ref="abc123")

        self.assertEqual(len(contexts), 1)
        self.assertTrue(contexts[0]["found"])
        self.assertEqual(contexts[0]["path"], "pkg/helpers.py")

    def test_fetch_method_contexts_uses_import_alias_for_candidate_file(self) -> None:
        toolbox = GitHubToolbox()
        current_content = """
import pkg.helpers as helpers

def target():
    return helpers.helper()
"""
        helper_content = """
def helper():
    return 1
"""

        def fake_fetch_repo_file(owner, repo, path, ref):
            if path == "pkg/a.py":
                return {"ok": True, "full_content": current_content}
            if path == "pkg/helpers.py":
                return {"ok": True, "full_content": helper_content}
            return {"ok": False, "error": "missing", "path": path, "ref": ref}

        with patch.object(toolbox, "_fetch_repo_file", side_effect=fake_fetch_repo_file), patch.object(
            toolbox,
            "_fetch_repo_tree_for_ref",
            return_value={
                "ok": True,
                "entries": [
                    {"path": "pkg/a.py", "type": "file"},
                    {"path": "pkg/helpers.py", "type": "file"},
                ],
            },
        ):
            contexts = toolbox.fetch_method_contexts("u", "p", "pkg/a.py", ["helpers.helper()"], ref="abc123")

        self.assertEqual(len(contexts), 1)
        self.assertTrue(contexts[0]["found"])
        self.assertEqual(contexts[0]["path"], "pkg/helpers.py")

    def test_fetch_method_contexts_rejects_similar_but_not_exact_definition(self) -> None:
        toolbox = GitHubToolbox()
        current_content = """
def target(storagedriver_config):
    return storagedriver_config.load()
"""
        wrong_content = """
def get_mds_load():
    return 1
"""

        def fake_fetch_repo_file(owner, repo, path, ref):
            if path == "pkg/a.py":
                return {"ok": True, "full_content": current_content}
            if path == "pkg/config.py":
                return {"ok": True, "full_content": wrong_content}
            return {"ok": False, "error": "missing", "path": path, "ref": ref}

        with patch.object(toolbox, "_fetch_repo_file", side_effect=fake_fetch_repo_file), patch.object(
            toolbox,
            "_fetch_repo_tree_for_ref",
            return_value={
                "ok": True,
                "entries": [
                    {"path": "pkg/a.py", "type": "file"},
                    {"path": "pkg/config.py", "type": "file"},
                ],
            },
        ):
            contexts = toolbox.fetch_method_contexts("u", "p", "pkg/a.py", ["storagedriver_config.load"], ref="abc123")

        self.assertEqual(len(contexts), 1)
        self.assertFalse(contexts[0]["found"])

    def test_fetch_method_contexts_rejects_wrong_class_for_explicit_class_method(self) -> None:
        toolbox = GitHubToolbox()
        current_content = """
def target(spec):
    return RolloutSamplerForSBI.get_dim_data(spec)
"""
        wrong_content = """
class OtherSampler:
    @staticmethod
    def get_dim_data(spec):
        return 1
"""

        def fake_fetch_repo_file(owner, repo, path, ref):
            if path == "pkg/a.py":
                return {"ok": True, "full_content": current_content}
            if path == "pkg/sampler.py":
                return {"ok": True, "full_content": wrong_content}
            return {"ok": False, "error": "missing", "path": path, "ref": ref}

        with patch.object(toolbox, "_fetch_repo_file", side_effect=fake_fetch_repo_file), patch.object(
            toolbox,
            "_fetch_repo_tree_for_ref",
            return_value={
                "ok": True,
                "entries": [
                    {"path": "pkg/a.py", "type": "file"},
                    {"path": "pkg/sampler.py", "type": "file"},
                ],
            },
        ):
            contexts = toolbox.fetch_method_contexts("u", "p", "pkg/a.py", ["RolloutSamplerForSBI.get_dim_data"], ref="abc123")

        self.assertEqual(len(contexts), 1)
        self.assertFalse(contexts[0]["found"])

    def test_fetch_method_contexts_uses_symbol_index_when_path_hints_are_weak(self) -> None:
        toolbox = GitHubToolbox()
        current_content = """
def target(service):
    return service.resolve()
"""
        impl_content = """
class Resolver:
    def resolve(self):
        return 1
"""

        def fake_fetch_repo_file(owner, repo, path, ref):
            if path == "pkg/a.py":
                return {"ok": True, "full_content": current_content}
            if path == "pkg/impl/core.py":
                return {"ok": True, "full_content": impl_content}
            return {"ok": False, "error": "missing", "path": path, "ref": ref}

        with patch.object(toolbox, "_fetch_repo_file", side_effect=fake_fetch_repo_file), patch.object(
            toolbox,
            "_fetch_repo_tree_for_ref",
            return_value={
                "ok": True,
                "entries": [
                    {"path": "pkg/a.py", "type": "file"},
                    {"path": "pkg/impl/core.py", "type": "file"},
                ],
            },
        ):
            contexts = toolbox.fetch_method_contexts("u", "p", "pkg/a.py", ["service.resolve"], ref="abc123")

        self.assertEqual(len(contexts), 1)
        self.assertTrue(contexts[0]["found"])
        self.assertEqual(contexts[0]["path"], "pkg/impl/core.py")

    def test_symbol_index_cache_reuses_saved_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            toolbox = GitHubToolbox()
            toolbox.repo_cache_dir = Path(tmpdir)
            toolbox.repo_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = toolbox._repo_symbol_index_cache_path("u", "p", "abc123")
            toolbox._write_json_file(
                cache_path,
                {
                    "ok": True,
                    "ref": "abc123",
                    "resolved_ref": "abc123",
                    "symbol_count": 1,
                    "symbols": {
                        "helper": [
                            {
                                "path": "pkg/helpers.py",
                                "symbol_name": "helper",
                                "qualified_name": "helper",
                                "class_name": None,
                                "start_line": 1,
                                "end_line": 2,
                            }
                        ]
                    },
                },
            )

            with patch.object(toolbox, "_build_symbol_index", side_effect=AssertionError("should use cache")):
                symbols = toolbox._load_symbol_index("u", "p", "abc123", ["pkg/helpers.py"])

        self.assertIn("helper", symbols)
        self.assertEqual(symbols["helper"][0]["path"], "pkg/helpers.py")

    def test_symbol_index_cache_resumes_from_partial_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            toolbox = GitHubToolbox()
            toolbox.repo_cache_dir = Path(tmpdir)
            toolbox.repo_cache_dir.mkdir(parents=True, exist_ok=True)
            partial_path = toolbox._repo_symbol_index_partial_cache_path("u", "p", "abc123")
            toolbox._write_json_file(
                partial_path,
                {
                    "ok": True,
                    "partial": True,
                    "ref": "abc123",
                    "resolved_ref": "abc123",
                    "scanned_file_count": 1,
                    "total_files": 2,
                    "symbol_count": 1,
                    "scanned_paths": ["pkg/helpers.py"],
                    "symbols": {
                        "helper": [
                            {
                                "path": "pkg/helpers.py",
                                "symbol_name": "helper",
                                "qualified_name": "helper",
                                "class_name": None,
                                "start_line": 1,
                                "end_line": 2,
                            }
                        ]
                    },
                },
            )
            fetched_paths: list[str] = []

            def fake_fetch_repo_file(owner, repo, path, ref):
                fetched_paths.append(path)
                if path == "pkg/helpers.py":
                    raise AssertionError("partial checkpoint should skip scanned paths")
                if path == "pkg/extra.py":
                    return {"ok": True, "full_content": "def extra():\n    return 1\n", "sha": "sha-extra"}
                return {"ok": False, "error": "missing"}

            with patch.object(toolbox, "_fetch_repo_file", side_effect=fake_fetch_repo_file), patch.object(
                toolbox,
                "_extract_tree_sitter_method_matches",
                return_value=[
                    {
                        "symbol_name": "extra",
                        "qualified_name": "extra",
                        "class_name": None,
                        "start_line": 1,
                        "end_line": 2,
                    }
                ],
            ):
                symbols = toolbox._load_symbol_index("u", "p", "abc123", ["pkg/helpers.py", "pkg/extra.py"])

        self.assertEqual(fetched_paths, ["pkg/extra.py"])
        self.assertIn("helper", symbols)
        self.assertIn("extra", symbols)
        self.assertFalse(partial_path.exists())

    def test_build_symbol_index_prefers_snapshot_files_for_historical_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            toolbox = GitHubToolbox()
            toolbox.repo_cache_dir = Path(tmpdir)
            toolbox.repo_cache_dir.mkdir(parents=True, exist_ok=True)
            snapshot_root = Path(tmpdir) / "snapshot"
            (snapshot_root / "pkg").mkdir(parents=True, exist_ok=True)
            (snapshot_root / "pkg" / "helpers.py").write_text("def helper():\n    return 1\n", encoding="utf-8")

            with patch.object(toolbox, "_ensure_snapshot_for_ref", return_value=(snapshot_root, "abc123")), patch.object(
                toolbox,
                "_fetch_repo_file",
                side_effect=AssertionError("symbol index should not use GitHub contents API when snapshot is available"),
            ):
                symbols = toolbox._build_symbol_index("u", "p", "abc123", ["pkg/helpers.py"])

        self.assertIn("helper", symbols)
        self.assertEqual(symbols["helper"][0]["path"], "pkg/helpers.py")

    def test_fetch_method_contexts_uses_repo_prefixed_import_and_class_definition(self) -> None:
        toolbox = GitHubToolbox()
        current_content = """
from pyrado.sampling.sbi_embeddings import BayesSimEmbedding

def target():
    embedding = BayesSimEmbedding()
    return embedding
"""
        embedding_content = """
class BayesSimEmbedding:
    def __init__(self):
        pass
"""

        def fake_fetch_repo_file(owner, repo, path, ref):
            if path == "Pyrado/pyrado/algorithms/meta/bayessim.py":
                return {"ok": True, "full_content": current_content}
            if path == "Pyrado/pyrado/sampling/sbi_embeddings.py":
                return {"ok": True, "full_content": embedding_content}
            return {"ok": False, "error": "missing", "path": path, "ref": ref}

        with patch.object(toolbox, "_fetch_repo_file", side_effect=fake_fetch_repo_file), patch.object(
            toolbox,
            "_fetch_repo_tree_for_ref",
            return_value={
                "ok": True,
                "entries": [
                    {"path": "Pyrado/pyrado/algorithms/meta/bayessim.py", "type": "file"},
                    {"path": "Pyrado/pyrado/sampling/sbi_embeddings.py", "type": "file"},
                ],
            },
        ):
            contexts = toolbox.fetch_method_contexts(
                "u",
                "p",
                "Pyrado/pyrado/algorithms/meta/bayessim.py",
                ["BayesSimEmbedding"],
                ref="abc123",
            )

        self.assertEqual(len(contexts), 1)
        self.assertTrue(contexts[0]["found"])
        self.assertEqual(contexts[0]["path"], "Pyrado/pyrado/sampling/sbi_embeddings.py")
        self.assertEqual(contexts[0]["method_name"], "BayesSimEmbedding")

    def test_fetch_method_contexts_refines_bare_method_name_from_usage_chain(self) -> None:
        toolbox = GitHubToolbox()
        current_content = """
from pyrado.sampling.sbi_rollout_sampler import RolloutSamplerForSBI

def target(env_sim):
    return RolloutSamplerForSBI.get_dim_data(env_sim.spec)
"""
        sampler_content = """
class RolloutSamplerForSBI:
    @staticmethod
    def get_dim_data(spec):
        return 1
"""

        def fake_fetch_repo_file(owner, repo, path, ref):
            if path == "Pyrado/pyrado/algorithms/meta/bayessim.py":
                return {"ok": True, "full_content": current_content}
            if path == "Pyrado/pyrado/sampling/sbi_rollout_sampler.py":
                return {"ok": True, "full_content": sampler_content}
            return {"ok": False, "error": "missing", "path": path, "ref": ref}

        with patch.object(toolbox, "_fetch_repo_file", side_effect=fake_fetch_repo_file), patch.object(
            toolbox,
            "_fetch_repo_tree_for_ref",
            return_value={
                "ok": True,
                "entries": [
                    {"path": "Pyrado/pyrado/algorithms/meta/bayessim.py", "type": "file"},
                    {"path": "Pyrado/pyrado/sampling/sbi_rollout_sampler.py", "type": "file"},
                ],
            },
        ):
            contexts = toolbox.fetch_method_contexts(
                "u",
                "p",
                "Pyrado/pyrado/algorithms/meta/bayessim.py",
                ["get_dim_data"],
                ref="abc123",
            )

        self.assertEqual(len(contexts), 1)
        self.assertTrue(contexts[0]["found"])
        self.assertEqual(contexts[0]["path"], "Pyrado/pyrado/sampling/sbi_rollout_sampler.py")
        self.assertEqual(contexts[0]["method_name"], "RolloutSamplerForSBI.get_dim_data")

    def test_fetch_method_contexts_marks_missing_methods(self) -> None:
        toolbox = GitHubToolbox()

        with patch.object(
            toolbox,
            "_fetch_repo_file",
            return_value={"ok": True, "full_content": "def target():\n    return 1\n"},
        ), patch.object(
            toolbox,
            "_fetch_repo_tree_for_ref",
            return_value={"ok": True, "entries": [{"path": "pkg/a.py", "type": "file"}]},
        ):
            contexts = toolbox.fetch_method_contexts("u", "p", "pkg/a.py", ["missing_method"], ref="abc123")

        self.assertEqual(contexts[0]["method_name"], "missing_method")
        self.assertFalse(contexts[0]["found"])

    def test_fetch_method_contexts_prefers_model_guided_path_hint(self) -> None:
        toolbox = GitHubToolbox()
        current_content = "def target():\n    return get_dim_data()\n"
        hinted_content = """
class RolloutSamplerForSBI:
    @staticmethod
    def get_dim_data(spec=None):
        return 1
"""

        def fake_fetch_repo_file(owner, repo, path, ref):
            if path == "Pyrado/pyrado/algorithms/meta/bayessim.py":
                return {"ok": True, "full_content": current_content}
            if path == "Pyrado/pyrado/sampling/sbi_rollout_sampler.py":
                return {"ok": True, "full_content": hinted_content}
            return {"ok": False, "error": "missing", "path": path, "ref": ref}

        with patch.object(toolbox, "_fetch_repo_file", side_effect=fake_fetch_repo_file), patch.object(
            toolbox,
            "_fetch_repo_tree_for_ref",
            return_value={
                "ok": True,
                "entries": [
                    {"path": "Pyrado/pyrado/algorithms/meta/bayessim.py", "type": "file"},
                    {"path": "Pyrado/pyrado/sampling/sbi_rollout_sampler.py", "type": "file"},
                ],
            },
        ):
            contexts = toolbox.fetch_method_contexts(
                "u",
                "p",
                "Pyrado/pyrado/algorithms/meta/bayessim.py",
                ["get_dim_data"],
                ref="abc123",
                path_hints_by_method={"get_dim_data": ["Pyrado/pyrado/sampling/sbi_rollout_sampler.py"]},
            )

        self.assertEqual(len(contexts), 1)
        self.assertTrue(contexts[0]["found"])
        self.assertEqual(contexts[0]["path"], "Pyrado/pyrado/sampling/sbi_rollout_sampler.py")

    def test_fetch_method_contexts_falls_back_after_explicit_hint_miss(self) -> None:
        toolbox = GitHubToolbox()

        def fake_fetch_repo_file(owner, repo, path, ref):
            if path == "Pyrado/pyrado/sampling/sbi_rollout_sampler.py":
                return {"ok": True, "full_content": "def other_name():\n    return 1\n"}
            if path == "Pyrado/pyrado/algorithms/meta/bayessim.py":
                return {"ok": True, "full_content": "def get_dim_data(spec=None):\n    return 1\n"}
            return {"ok": False, "error": "missing", "path": path, "ref": ref}

        with patch.object(toolbox, "_fetch_repo_file", side_effect=fake_fetch_repo_file):
            contexts = toolbox.fetch_method_contexts(
                "u",
                "p",
                "Pyrado/pyrado/algorithms/meta/bayessim.py",
                ["get_dim_data"],
                ref="abc123",
                path_hints_by_method={"get_dim_data": ["Pyrado/pyrado/sampling/sbi_rollout_sampler.py"]},
            )

        self.assertEqual(len(contexts), 1)
        self.assertTrue(contexts[0]["found"])
        self.assertEqual(contexts[0]["path"], "Pyrado/pyrado/algorithms/meta/bayessim.py")

    def test_fetch_method_contexts_uses_regex_fallback_when_ast_parse_fails(self) -> None:
        toolbox = GitHubToolbox()
        hinted_content = """
class RHBugzilla:
    def okay(self):
        return True

    def pre_translation(self, query):
        raise ValueError, "python2 syntax fallback"
"""

        def fake_fetch_repo_file(owner, repo, path, ref):
            if path == "bugzilla/rhbugzilla.py":
                return {"ok": True, "full_content": hinted_content}
            raise AssertionError(f"unexpected path fetch: {path}")

        with patch.object(toolbox, "_fetch_repo_file", side_effect=fake_fetch_repo_file):
            contexts = toolbox.fetch_method_contexts(
                "u",
                "p",
                "bugzilla/rhbugzilla.py",
                ["pre_translation"],
                ref="abc123",
                path_hints_by_method={"pre_translation": ["bugzilla/rhbugzilla.py"]},
            )

        self.assertEqual(len(contexts), 1)
        self.assertTrue(contexts[0]["found"])
        self.assertEqual(contexts[0]["path"], "bugzilla/rhbugzilla.py")
        self.assertEqual(contexts[0]["method_name"], "pre_translation")
        self.assertIn("raise ValueError, \"python2 syntax fallback\"", contexts[0]["source"])

    def test_fetch_method_contexts_without_explicit_hints_still_searches(self) -> None:
        toolbox = GitHubToolbox()

        def fake_fetch_repo_file(owner, repo, path, ref):
            if path == "pkg/a.py":
                return {"ok": True, "full_content": "def target():\n    return helper()\n\ndef helper():\n    return 1\n"}
            return {"ok": False, "error": "missing", "path": path, "ref": ref}

        with patch.object(toolbox, "_fetch_repo_file", side_effect=fake_fetch_repo_file):
            contexts = toolbox.fetch_method_contexts(
                "u",
                "p",
                "pkg/a.py",
                ["helper"],
                ref="abc123",
                path_hints_by_method={"helper": []},
            )

        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0]["method_name"], "helper")
        self.assertTrue(contexts[0]["found"])

    def test_fetch_method_contexts_uses_local_receiver_type_inference(self) -> None:
        toolbox = GitHubToolbox()
        current_content = """
from pkg.impl import Resolver

def target():
    service = Resolver()
    return service.resolve()
"""
        impl_content = """
class Resolver:
    def resolve(self):
        return 1
"""

        def fake_fetch_repo_file(owner, repo, path, ref):
            if path == "pkg/a.py":
                return {"ok": True, "full_content": current_content}
            if path == "pkg/impl.py":
                return {"ok": True, "full_content": impl_content}
            return {"ok": False, "error": "missing", "path": path, "ref": ref}

        with patch.object(toolbox, "_fetch_repo_file", side_effect=fake_fetch_repo_file), patch.object(
            toolbox,
            "_fetch_repo_tree_for_ref",
            return_value={
                "ok": True,
                "entries": [
                    {"path": "pkg/a.py", "type": "file"},
                    {"path": "pkg/impl.py", "type": "file"},
                ],
            },
        ):
            contexts = toolbox.fetch_method_contexts("u", "p", "pkg/a.py", ["service.resolve"], ref="abc123")

        self.assertEqual(len(contexts), 1)
        self.assertTrue(contexts[0]["found"])
        self.assertEqual(contexts[0]["path"], "pkg/impl.py")
        self.assertEqual(contexts[0]["method_name"], "Resolver.resolve")

    def test_fetch_method_contexts_resolves_relative_imports(self) -> None:
        toolbox = GitHubToolbox()
        current_content = """
from .impl import Resolver

def target():
    service = Resolver()
    return service.resolve()
"""
        impl_content = """
class Resolver:
    def resolve(self):
        return 1
"""

        def fake_fetch_repo_file(owner, repo, path, ref):
            if path == "pkg/sub/a.py":
                return {"ok": True, "full_content": current_content}
            if path == "pkg/sub/impl.py":
                return {"ok": True, "full_content": impl_content}
            return {"ok": False, "error": "missing", "path": path, "ref": ref}

        with patch.object(toolbox, "_fetch_repo_file", side_effect=fake_fetch_repo_file), patch.object(
            toolbox,
            "_fetch_repo_tree_for_ref",
            return_value={
                "ok": True,
                "entries": [
                    {"path": "pkg/sub/a.py", "type": "file"},
                    {"path": "pkg/sub/impl.py", "type": "file"},
                ],
            },
        ):
            contexts = toolbox.fetch_method_contexts("u", "p", "pkg/sub/a.py", ["service.resolve"], ref="abc123")

        self.assertEqual(len(contexts), 1)
        self.assertTrue(contexts[0]["found"])
        self.assertEqual(contexts[0]["path"], "pkg/sub/impl.py")
        self.assertEqual(contexts[0]["method_name"], "Resolver.resolve")

    def test_fetch_method_contexts_tracks_package_reexports(self) -> None:
        toolbox = GitHubToolbox()
        current_content = """
from pkg import Resolver

def target():
    service = Resolver()
    return service.resolve()
"""
        init_content = "from .impl import Resolver\n"
        impl_content = """
class Resolver:
    def resolve(self):
        return 1
"""

        def fake_fetch_repo_file(owner, repo, path, ref):
            if path == "pkg/a.py":
                return {"ok": True, "full_content": current_content}
            if path == "pkg/__init__.py":
                return {"ok": True, "full_content": init_content}
            if path == "pkg/impl.py":
                return {"ok": True, "full_content": impl_content}
            return {"ok": False, "error": "missing", "path": path, "ref": ref}

        with patch.object(toolbox, "_fetch_repo_file", side_effect=fake_fetch_repo_file), patch.object(
            toolbox,
            "_fetch_repo_tree_for_ref",
            return_value={
                "ok": True,
                "entries": [
                    {"path": "pkg/a.py", "type": "file"},
                    {"path": "pkg/__init__.py", "type": "file"},
                    {"path": "pkg/impl.py", "type": "file"},
                ],
            },
        ):
            contexts = toolbox.fetch_method_contexts("u", "p", "pkg/a.py", ["service.resolve"], ref="abc123")

        self.assertEqual(len(contexts), 1)
        self.assertTrue(contexts[0]["found"])
        self.assertEqual(contexts[0]["path"], "pkg/impl.py")
        self.assertEqual(contexts[0]["method_name"], "Resolver.resolve")

    def test_method_query_workflow_uses_single_candidate_mode(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)
        workflow.context_client = OpenAICompatClient.__new__(OpenAICompatClient)
        workflow.dual_repair_candidates = False
        workflow.use_selector = False
        workflow.fixer = OpenAIFixer.__new__(OpenAIFixer)
        workflow.fixer.run = lambda state, candidate_mode="baseline_context": (
            RepairAttempt(1, "plan", "def a():\n    return helper()", "function", 0.8, "ok", candidate_mode),
            MethodInquiryResult(required_methods=["helper"], reason="needed", uncertainty_items=[UncertaintyItem(kind="method", name="helper", retrieval_eligible=True)]),
            [
                RetrievedMethodContext(
                    method_name="helper",
                    path="pkg/a.py",
                    class_name=None,
                    start_line=3,
                    end_line=4,
                    source="def helper():\n    return 1",
                    found=True,
                )
            ],
            [],
            [UncertaintyItem(kind="method", name="helper", retrieval_eligible=True)],
            [EditConstraint(focus_point="helper", required_fact="Use helper semantics carefully.")],
        )

        candidates, inquiry, retrieved, missing, uncertainty_items, edit_constraints = workflow._run_repair_candidates(
            {
                "task_id": "1",
                "round_id": 0,
                "user": "u",
                "project": "p",
                "file_path": "pkg/a.py",
                "commit": "abc123",
                "github_context": {},
            },
            {},
            1,
            "generic",
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].candidate_mode, "baseline_context")
        self.assertEqual(inquiry.required_methods, ["helper"])
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(missing, [])
        self.assertEqual(len(uncertainty_items), 1)
        self.assertEqual(len(edit_constraints), 1)

    def test_trace_preserves_method_query_fields(self) -> None:
        state = record_to_graph_input(
            SATDRecord(
                task_id="1",
                satd_comment="# TODO",
                original_code="def a():\n    pass",
                manual_code="def a():\n    return helper()",
                user="u",
                project="p",
                file_path="pkg/a.py",
                commit="abc123",
                em_label="NO",
            ),
            max_rounds=2,
        )
        state["status"] = "accepted"
        state["round_id"] = 1
        state["final_repaired_code"] = "def a():\n    return helper()"
        state["method_inquiry"] = MethodInquiryResult(required_methods=["helper"], reason="needed")
        state["retrieved_method_contexts"] = [
            RetrievedMethodContext(
                method_name="helper",
                path="pkg/a.py",
                class_name=None,
                start_line=3,
                end_line=4,
                source="def helper():\n    return 1",
                found=True,
            )
        ]
        state["missing_method_names"] = ["missing_helper"]

        trace = trace_from_state(state, "NO")

        self.assertEqual(trace.method_inquiry["required_methods"], ["helper"])
        self.assertEqual(trace.retrieved_method_contexts[0]["method_name"], "helper")
        self.assertEqual(trace.missing_method_names, ["missing_helper"])

if __name__ == "__main__":
    unittest.main()

