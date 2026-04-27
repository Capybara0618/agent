from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from satd_langgraph.agents import OpenAIAnalyzer, OpenAICompatClient, OpenAIFixer, OpenAIReviewer
from satd_langgraph.github_tools import GitHubToolbox
from satd_langgraph.schema import EditConstraint, MethodInquiryResult, RepairAttempt, RetrievedMethodContext, ReviewResult
from satd_langgraph.workflow import LangGraphSATDWorkflow


class _FakeReviewClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.system_prompt = ""
        self.user_prompt = ""
        self.kwargs = {}

    def generate_json(self, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.kwargs = dict(kwargs)
        return self.payload


class _FailingFixer:
    def __init__(self) -> None:
        self.checkpoints = []

    def run(self, state: dict, candidate_mode: str):
        raise RuntimeError("simulated api failure")

    def _checkpoint(self, state: dict, stage: str, payload: dict) -> None:
        self.checkpoints.append((stage, payload))


class _NoContextRecoveryFixer(_FailingFixer):
    def run(self, state: dict, candidate_mode: str):
        if candidate_mode == "baseline_no_context":
            return (
                RepairAttempt(
                    round_id=int(state.get("round_id", 0)) + 1,
                    repair_plan="fallback retry plan",
                    repaired_code="def f():\n    return 2\n",
                    changed_scope="line",
                    confidence=0.4,
                    notes="recovered",
                    candidate_mode=candidate_mode,
                ),
                MethodInquiryResult(reason="candidate_mode_without_method_context"),
                [],
                [],
                [],
                [],
            )
        raise RuntimeError("simulated api failure")


class ContextPromptSafetyTests(unittest.TestCase):
    def test_prompt_coercion_flattens_sequences(self) -> None:
        client = OpenAICompatClient.__new__(OpenAICompatClient)
        self.assertEqual(client._coerce_prompt_text(("a", "b", ("c", None, 1))), "abc1")

    def test_openai_client_defaults_to_three_attempts(self) -> None:
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"OPENAI_TIMEOUT_SECONDS", "OPENAI_MAX_ATTEMPTS", "OPENAI_SDK_MAX_RETRIES", "OPENAI_API_BASE", "OPENAI_BASE_URL"}
        }
        env["OPENAI_API_KEY"] = "test-key"
        with patch.dict(os.environ, env, clear=True):
            client = OpenAICompatClient(verbose=False)
        self.assertEqual(client.max_attempts, 3)
        self.assertEqual(client.request_timeout, 60)
        self.assertEqual(client.client.max_retries, 0)

    def test_tree_sitter_deep_call_ast_does_not_recurse(self) -> None:
        toolbox = GitHubToolbox()
        nested_call = "value"
        for _ in range(1400):
            nested_call = f"wrap({nested_call})"
        source = f"def outer():\n    return {nested_call}\n"
        method_matches = toolbox._extract_tree_sitter_method_matches(source)
        called_references = toolbox._extract_called_references(source)
        self.assertTrue(any(item.get("symbol_name") == "outer" for item in method_matches))
        self.assertTrue(any(item.get("qualified_name") == "wrap" for item in called_references))

    def test_generic_context_prompt_is_a_string(self) -> None:
        fixer = OpenAIFixer.__new__(OpenAIFixer)
        system_prompt, user_prompt = OpenAIFixer._build_generic_repair_prompts(
            fixer,
            state={
                "satd_comment": "TODO: fix docs",
                "original_code": "def f():\n    return x",
            },
            method_context_block="supporting method context",
            edit_constraints=[],
            reviewer_feedback_block="",
        )
        self.assertIsInstance(system_prompt, str)
        self.assertIsInstance(user_prompt, str)
        self.assertIn("### Supporting evidence:", user_prompt)
        self.assertIn("supporting method context", user_prompt)

    def test_no_context_repair_prompt_includes_constraint_feedback(self) -> None:
        fixer = OpenAIFixer.__new__(OpenAIFixer)
        system_prompt, user_prompt = OpenAIFixer._build_no_context_repair_prompts(
            fixer,
            state={
                "satd_comment": "TODO: fix behavior",
                "original_code": "def f():\n    return 1",
            },
            reviewer_feedback_block="### Reviewer feedback from previous attempt:\nRepair constraints: preserve_unrelated_lines\nRetry hint: Make a smaller local edit.\n",
        )
        self.assertIsInstance(system_prompt, str)
        self.assertIn("Reviewer", user_prompt)
        self.assertNotIn("replacement code", user_prompt.lower())

    def test_repair_prompt_expands_tabs_before_llm(self) -> None:
        fixer = OpenAIFixer.__new__(OpenAIFixer)
        _system_prompt, user_prompt = OpenAIFixer._build_no_context_repair_prompts(
            fixer,
            state={
                "satd_comment": "TODO: fix behavior",
                "original_code": "def f():\n\treturn call(\\\n\t\t1)\n",
            },
        )
        self.assertIn("    return call", user_prompt)
        self.assertNotIn("\t", user_prompt)
        self.assertIn("excessive whitespace", user_prompt)

    def test_fixer_generation_uses_repair_token_cap(self) -> None:
        client = _FakeReviewClient({"repaired_code": "def f():\n    return 2\n"})
        fixer = OpenAIFixer(client)
        state = {
            "task_id": "unit",
            "satd_comment": "TODO: fix behavior",
            "original_code": "def f():\n    return 1\n",
            "candidate_mode": "baseline_no_context",
            "round_id": 0,
            "github_context": {},
            "repair_feedback": {},
        }
        repair, *_ = fixer.run(state, candidate_mode="baseline_no_context")
        self.assertEqual(repair.repaired_code, "def f():\n    return 2\n")
        self.assertEqual(client.kwargs.get("max_tokens"), 4096)

    def test_analyzer_prompt_uses_method_context_sections(self) -> None:
        client = _FakeReviewClient(
            {
                "decision": "pass",
                "confidence": 0.82,
                "operation_concrete": "high",
                "localizable": "high",
                "local_scope": "high",
                "end_state_clear": "high",
                "context_sufficiency": "high",
                "method_context_used": True,
                "drop_reason": "",
                "comment_evidence": "handle missing config",
                "code_evidence": "read_json supports default",
                "notes": "The repair target is local.",
            }
        )
        analyzer = OpenAIAnalyzer(client)
        state = {
            "task_id": "unit",
            "satd_comment": "TODO: handle missing config file gracefully",
            "original_code": "def load_config(path):\n    return read_json(path)\n",
            "satd_route_type": "generic",
            "method_inquiry": MethodInquiryResult(required_methods=["read_json"], reason="needed for behavior"),
            "retrieved_method_contexts": [
                RetrievedMethodContext(
                    method_name="read_json",
                    path="pkg/config.py",
                    class_name=None,
                    start_line=1,
                    end_line=2,
                    source="def read_json(path, default=None):\n    return default\n",
                    found=True,
                    signature="read_json(path, default=None)",
                )
            ],
            "missing_method_names": [],
            "edit_constraints": [EditConstraint(focus_point="read_json call", must_do="use existing default argument")],
        }
        result = analyzer.run(
            state,
            method_context_block="- method: read_json\n  method behavior:\n    def read_json(path, default=None): ...",
        )

        self.assertTrue(result.repairable)
        self.assertIn("### Required methods:", client.user_prompt)
        self.assertIn("read_json", client.user_prompt)
        self.assertIn("### Supporting evidence:", client.user_prompt)
        self.assertIn("def read_json(path, default=None)", client.user_prompt)
        self.assertIn("### Context quality:", client.user_prompt)
        self.assertIn("repair_evidence_mode: strong", client.user_prompt)
        self.assertIn("method context alone is insufficient", client.user_prompt)
        self.assertIn("supporting evidence only proves that related methods exist", client.user_prompt)
        self.assertIn("Do not output repaired_code", client.system_prompt)
        self.assertNotIn('"repaired_code"', client.user_prompt)

    def test_fixer_run_reuses_analyzer_method_context_without_inquiry(self) -> None:
        client = _FakeReviewClient({"repaired_code": "def load_config(path):\n    return read_json(path, default={})\n"})
        fixer = OpenAIFixer(client)

        def fail_identify_required_methods(_state: dict) -> MethodInquiryResult:
            raise AssertionError("Fixer should reuse Analyzer method context instead of asking again.")

        fixer.identify_required_methods = fail_identify_required_methods
        state = {
            "task_id": "unit",
            "round_id": 0,
            "satd_comment": "TODO: handle missing config file gracefully",
            "original_code": "def load_config(path):\n    return read_json(path)\n",
            "satd_route_type": "generic",
            "candidate_mode": "baseline_context",
            "method_inquiry": MethodInquiryResult(required_methods=["read_json"], reason="needed for behavior"),
            "retrieved_method_contexts": [
                RetrievedMethodContext(
                    method_name="read_json",
                    path="pkg/config.py",
                    class_name=None,
                    start_line=1,
                    end_line=2,
                    source="def read_json(path, default=None):\n    return default\n",
                    found=True,
                    signature="read_json(path, default=None)",
                )
            ],
            "missing_method_names": [],
            "uncertainty_items": [],
            "edit_constraints": [EditConstraint(focus_point="read_json call", must_do="use existing default argument")],
            "repair_feedback": {},
        }

        repair, method_inquiry, contexts, missing, uncertainty, constraints = fixer.run(
            state,
            candidate_mode="baseline_context",
        )

        self.assertEqual(repair.repaired_code, "def load_config(path):\n    return read_json(path, default={})\n")
        self.assertEqual(method_inquiry.required_methods, ["read_json"])
        self.assertEqual([item.method_name for item in contexts], ["read_json"])
        self.assertEqual(missing, [])
        self.assertEqual(uncertainty, [])
        self.assertEqual(constraints[0].focus_point, "read_json call")
        self.assertIn("### Supporting evidence:", client.user_prompt)
        self.assertIn("def read_json(path, default=None)", client.user_prompt)
        self.assertEqual(client.kwargs.get("max_tokens"), 4096)

    def test_special_routes_skip_method_context_again(self) -> None:
        fixer = OpenAIFixer.__new__(OpenAIFixer)
        for route in ("remove_temporary", "type_annotation", "replace_symbol", "document"):
            self.assertTrue(OpenAIFixer._skip_method_context_by_rule(fixer, route, "TODO: sample"))
        self.assertFalse(OpenAIFixer._skip_method_context_by_rule(fixer, "generic", "TODO: sample"))

    def test_simple_analyzer_easy_route_passes_directly(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        result = OpenAIAnalyzer.build_easy_route_analysis(analyzer, "document")
        self.assertTrue(result.repairable)
        self.assertEqual(result.decision, "pass")
        self.assertEqual(result.satd_type, "document")
        self.assertEqual(result.operation_concrete, "high")
        self.assertEqual(result.localizable, "high")
        self.assertEqual(result.local_scope, "high")
        self.assertEqual(result.end_state_clear, "high")

    def test_simple_analyzer_rule_drop_marks_all_three_checks_false(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        result = OpenAIAnalyzer.build_rule_drop_analysis(
            analyzer,
            "Code snippet is empty, so the repair target cannot be located.",
        )
        self.assertFalse(result.repairable)
        self.assertEqual(result.decision, "drop")
        self.assertEqual(result.operation_concrete, "low")
        self.assertEqual(result.localizable, "low")
        self.assertEqual(result.local_scope, "low")
        self.assertEqual(result.end_state_clear, "low")

    def test_simple_analyzer_uncertain_keeps_item_repairable(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        payload = {
            "decision": "uncertain",
            "confidence": 0.45,
            "operation_concrete": "high",
            "localizable": "partial",
            "local_scope": "high",
            "end_state_clear": "partial",
            "comment_evidence": "should return a Rad",
            "code_evidence": "return target is not obvious",
            "notes": "The target is not fully explicit.",
        }
        state = {
            "satd_route_type": "generic",
            "original_code": "def f(x):\n    return x",
        }
        result = OpenAIAnalyzer._coerce_analysis(analyzer, payload, state, source="llm")
        self.assertTrue(result.repairable)
        self.assertEqual(result.decision, "uncertain")
        self.assertEqual(result.operation_concrete, "high")
        self.assertEqual(result.localizable, "partial")
        self.assertEqual(result.end_state_clear, "partial")

    def test_simple_analyzer_high_confidence_drop_stays_drop(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        payload = {
            "decision": "uncertain",
            "confidence": 0.82,
            "operation_concrete": "low",
            "localizable": "low",
            "local_scope": "low",
            "end_state_clear": "low",
            "comment_evidence": "refactor this workflow",
            "code_evidence": "no single local target",
            "notes": "The comment asks for a refactor-like change.",
        }
        state = {
            "satd_route_type": "generic",
            "original_code": "def f(x):\n    return x",
        }
        result = OpenAIAnalyzer._coerce_analysis(analyzer, payload, state, source="llm")
        self.assertFalse(result.repairable)
        self.assertEqual(result.decision, "drop")
        self.assertEqual(result.operation_concrete, "low")
        self.assertEqual(result.localizable, "low")
        self.assertEqual(result.local_scope, "low")
        self.assertEqual(result.end_state_clear, "low")

    def test_simple_analyzer_drops_open_ended_decision_even_with_context(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        payload = {
            "decision": "pass",
            "confidence": 0.78,
            "operation_concrete": "partial",
            "localizable": "high",
            "local_scope": "high",
            "end_state_clear": "partial",
            "context_sufficiency": "high",
            "method_context_used": True,
            "comment_evidence": "Decide whether we want 1:1 or 1:many",
            "code_evidence": "related method context was retrieved",
            "notes": "The supporting method exists.",
        }
        state = {
            "satd_route_type": "generic",
            "satd_comment": "TODO: Decide whether we want 1:1 or 1:many",
            "original_code": "def map_items(items):\n    return build_mapping(items)\n",
            "method_inquiry": MethodInquiryResult(required_methods=["build_mapping"]),
            "retrieved_method_contexts": [
                RetrievedMethodContext(
                    method_name="build_mapping",
                    path="pkg/mapping.py",
                    class_name=None,
                    start_line=1,
                    end_line=2,
                    source="def build_mapping(items):\n    return dict(items)\n",
                    found=True,
                )
            ],
        }
        result = OpenAIAnalyzer._coerce_analysis(analyzer, payload, state, source="llm")
        self.assertFalse(result.repairable)
        self.assertEqual(result.decision, "drop")
        self.assertIn("open_ended_without_specific_local_edit", result.evidence_summary)

    def test_reviewer_rejects_no_effective_change(self) -> None:
        reviewer = OpenAIReviewer.__new__(OpenAIReviewer)
        state = {
            "satd_comment": "TODO: fix this",
            "original_code": "def f():\n    return 1\n",
        }
        repair = RepairAttempt(1, "plan", "def f():\n    return 1\n", "line", 0.5, "notes")
        self.assertEqual(
            OpenAIReviewer._local_failed_checks(reviewer, state, repair),
            ["no_effective_change"],
        )

    def test_reviewer_rejects_syntax_error(self) -> None:
        reviewer = OpenAIReviewer.__new__(OpenAIReviewer)
        state = {
            "satd_comment": "TODO: fix this",
            "original_code": "def f():\n    return 1\n",
        }
        repair = RepairAttempt(1, "plan", "def f(:\n    return 1\n", "line", 0.5, "notes")
        self.assertEqual(
            OpenAIReviewer._local_failed_checks(reviewer, state, repair),
            ["syntax_error"],
        )

    def test_reviewer_rejects_unsupported_signature_change(self) -> None:
        reviewer = OpenAIReviewer.__new__(OpenAIReviewer)
        state = {
            "satd_comment": "TODO: fix behavior",
            "original_code": "def f(x):\n    return x\n",
        }
        repair = RepairAttempt(1, "plan", "def f(x, y):\n    return x\n", "line", 0.5, "notes")
        evidence = OpenAIReviewer._structural_evidence_payload(reviewer, state, repair)
        self.assertIn(
            "unsupported_signature_change",
            OpenAIReviewer._structural_risk_checks(reviewer, state, repair, evidence),
        )

    def test_reviewer_does_not_hard_reject_new_helper(self) -> None:
        reviewer = OpenAIReviewer.__new__(OpenAIReviewer)
        state = {
            "satd_comment": "TODO: fix behavior",
            "original_code": "def f(x):\n    return x\n",
        }
        repair = RepairAttempt(
            1,
            "plan",
            "def f(x):\n    return helper(x)\n\n\ndef helper(x):\n    return x\n",
            "function",
            0.5,
            "notes",
        )
        self.assertEqual(OpenAIReviewer._local_failed_checks(reviewer, state, repair), [])

    def test_reviewer_allows_explicit_obsolete_todo_comment_deletion(self) -> None:
        reviewer = OpenAIReviewer.__new__(OpenAIReviewer)
        state = {
            "satd_comment": "Remove obsolete TODO comment.",
            "original_code": "def f():\n    # TODO: obsolete\n    return 1\n",
        }
        repair = RepairAttempt(1, "plan", "def f():\n    return 1\n", "line", 0.5, "notes")
        self.assertEqual(OpenAIReviewer._local_failed_checks(reviewer, state, repair), [])

    def test_reviewer_rejects_raw_noop_even_for_doc_comment(self) -> None:
        reviewer = OpenAIReviewer.__new__(OpenAIReviewer)
        state = {
            "satd_comment": "TODO: Missing doc",
            "original_code": 'def f():\n    """\n    # TODO: Missing doc\n    """\n    return 1\n',
        }
        repair = RepairAttempt(1, "plan", state["original_code"], "line", 0.5, "notes")
        self.assertEqual(OpenAIReviewer._local_failed_checks(reviewer, state, repair), ["no_effective_change"])

    def test_reviewer_rejects_fallback_noop_repair(self) -> None:
        reviewer = OpenAIReviewer.__new__(OpenAIReviewer)
        state = {
            "satd_comment": "TODO: fix this",
            "original_code": "def f():\n    return 1\n",
        }
        repair = RepairAttempt(
            1,
            "Fallback no-op repair because the fixer request failed.",
            "def f():\n    return 1\n",
            "none",
            0.0,
            "fixer_exception:baseline_context:RecursionError",
        )
        self.assertEqual(OpenAIReviewer._local_failed_checks(reviewer, state, repair), ["no_effective_change"])

    def test_reviewer_structural_evidence_detects_satd_block_removal(self) -> None:
        reviewer = OpenAIReviewer.__new__(OpenAIReviewer)
        original = (
            "def f():\n"
            "    x = 1\n"
            "    if False:  # FIXME would we still need this?\n"
            "        x = old_fix()\n"
            "    return x\n"
        )
        repaired = "def f():\n    x = 1\n    return x\n"
        state = {
            "satd_comment": "FIXME would we still need this?",
            "original_code": original,
        }
        repair = RepairAttempt(1, "plan", repaired, "function", 0.6, "notes")
        evidence = OpenAIReviewer._structural_evidence_payload(reviewer, state, repair)
        self.assertTrue(evidence["satd_anchor_removed"])
        self.assertTrue(evidence["satd_block_removed"])
        self.assertTrue(evidence["dead_or_temporary_block_removed"])

    def test_reviewer_rejects_anchor_untouched_unrelated_repair(self) -> None:
        client = _FakeReviewClient({"approved": True, "failed_checks": [], "repair_constraints": [], "failure_anchor": ""})
        reviewer = OpenAIReviewer(client)
        original = (
            "def f(x):\n"
            "    # TODO: validate input\n"
            "    a = 1\n"
            "    b = 2\n"
            "    c = 3\n"
            "    return x\n"
        )
        repaired = original.replace("return x", "return x + 1")
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "TODO: validate input",
                "original_code": original,
                "latest_repair": RepairAttempt(2, "plan", repaired, "line", 0.5, "notes"),
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertFalse(result.approved)
        self.assertIn("anchor_not_modified", result.failed_checks)

    def test_reviewer_rejects_comment_only_without_satd_support(self) -> None:
        client = _FakeReviewClient({"approved": True, "failed_checks": [], "repair_constraints": [], "failure_anchor": ""})
        reviewer = OpenAIReviewer(client)
        original = "def f():\n    return 1\n"
        repaired = "def f():\n    # fixed\n    return 1\n"
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "TODO: fix behavior",
                "original_code": original,
                "latest_repair": RepairAttempt(2, "plan", repaired, "line", 0.5, "notes"),
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertFalse(result.approved)
        self.assertIn("comment_only_without_satd_support", result.failed_checks)

    def test_reviewer_rejects_unsupported_control_flow_change(self) -> None:
        client = _FakeReviewClient({"approved": True, "failed_checks": [], "repair_constraints": [], "failure_anchor": ""})
        reviewer = OpenAIReviewer(client)
        original = "def f(x):\n    return x\n"
        repaired = "def f(x):\n    if x is None:\n        return 0\n    return x\n"
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "TODO: improve speed",
                "original_code": original,
                "latest_repair": RepairAttempt(2, "plan", repaired, "function", 0.5, "notes"),
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertFalse(result.approved)
        self.assertIn("unsupported_control_flow_change", result.failed_checks)

    def test_reviewer_rejects_over_expanded_change(self) -> None:
        client = _FakeReviewClient({"approved": True, "failed_checks": [], "repair_constraints": [], "failure_anchor": ""})
        reviewer = OpenAIReviewer(client)
        original = "def f(x):\n    # TODO: normalize value\n    return x\n"
        repaired = (
            "def f(x):\n"
            "    # TODO: normalize value\n"
            "    if x is None:\n"
            "        return None\n"
            "    value = str(x)\n"
            "    value = value.strip()\n"
            "    value = value.lower()\n"
            "    value = value.replace('-', '_')\n"
            "    value = value.replace(' ', '_')\n"
            "    if not value:\n"
            "        return None\n"
            "    return value\n"
        )
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "TODO: normalize value",
                "original_code": original,
                "latest_repair": RepairAttempt(2, "plan", repaired, "function", 0.5, "notes"),
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertFalse(result.approved)
        self.assertIn("over_expanded_change", result.failed_checks)

    def test_reviewer_does_not_treat_docstring_addition_as_over_expanded(self) -> None:
        client = _FakeReviewClient({"approved": True, "failed_checks": [], "repair_constraints": [], "failure_anchor": ""})
        reviewer = OpenAIReviewer(client)
        original = "def f(x):\n    return x\n"
        repaired = (
            "def f(x):\n"
            "    \"\"\"\n"
            "    Normalize and return the provided value.\n"
            "    This function preserves the existing object.\n"
            "    It intentionally performs no conversion.\n"
            "    \"\"\"\n"
            "    return x\n"
        )
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "TODO: add documentation for f",
                "original_code": original,
                "latest_repair": RepairAttempt(2, "plan", repaired, "function", 0.5, "notes"),
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertTrue(result.approved)
        self.assertNotIn("over_expanded_change", result.failed_checks)

    def test_reviewer_does_not_treat_large_deletion_as_over_expanded(self) -> None:
        client = _FakeReviewClient({"approved": False, "failed_checks": ["over_expanded_change"], "repair_constraints": [], "failure_anchor": "scope"})
        reviewer = OpenAIReviewer(client)
        original = (
            "def f(x):\n"
            "    # TODO remove deprecated workaround\n"
            "    if x is None:\n"
            "        x = legacy_default()\n"
            "    if hasattr(x, 'old'):\n"
            "        x.old.cleanup()\n"
            "    return x.value\n"
        )
        repaired = "def f(x):\n    return x.value\n"
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "TODO remove deprecated workaround",
                "original_code": original,
                "latest_repair": RepairAttempt(2, "plan", repaired, "function", 0.5, "notes"),
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertTrue(result.approved)
        self.assertNotIn("over_expanded_change", result.failed_checks)

    def test_reviewer_retry_hint_prefers_comment_only_template(self) -> None:
        client = _FakeReviewClient(
            {
                "approved": False,
                "failed_checks": ["comment_only_without_satd_support"],
                "repair_constraints": ["avoid_comment_only_change"],
                "failure_anchor": "comment_only",
                "retry_hint": "Ensure the repair directly addresses the SATD comment.",
            }
        )
        reviewer = OpenAIReviewer(client)
        original = "def f():\n    # TODO: fix behavior\n    return 1\n"
        repaired = "def f():\n    # fixed behavior\n    return 1\n"
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "TODO: fix behavior",
                "original_code": original,
                "latest_repair": RepairAttempt(1, "plan", repaired, "line", 0.5, "notes"),
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertFalse(result.approved)
        self.assertIn("do not only rewrite the comment", result.retry_hint)

    def test_reviewer_retry_hint_prefers_anchor_template(self) -> None:
        client = _FakeReviewClient(
            {
                "approved": False,
                "failed_checks": ["anchor_not_modified"],
                "repair_constraints": ["modify_satd_anchor_region"],
                "failure_anchor": "alignment",
                "retry_hint": "Ensure the repair directly addresses the SATD comment.",
            }
        )
        reviewer = OpenAIReviewer(client)
        original = (
            "def f(x):\n"
            "    # TODO: validate input\n"
            "    a = 1\n"
            "    b = 2\n"
            "    c = 3\n"
            "    return x\n"
        )
        repaired = original.replace("return x", "return x + 1")
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "TODO: validate input",
                "original_code": original,
                "latest_repair": RepairAttempt(1, "plan", repaired, "line", 0.5, "notes"),
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertFalse(result.approved)
        self.assertIn("SATD anchor region", result.retry_hint)

    def test_reviewer_retry_hint_prefers_control_flow_template(self) -> None:
        client = _FakeReviewClient(
            {
                "approved": False,
                "failed_checks": ["unsupported_control_flow_change"],
                "repair_constraints": ["avoid_new_control_flow_paths"],
                "failure_anchor": "control_flow",
                "retry_hint": "Ensure the repair directly addresses the SATD comment.",
            }
        )
        reviewer = OpenAIReviewer(client)
        original = "def f(x):\n    return x\n"
        repaired = "def f(x):\n    if x is None:\n        return 0\n    return x\n"
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "TODO: improve speed",
                "original_code": original,
                "latest_repair": RepairAttempt(1, "plan", repaired, "function", 0.5, "notes"),
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertFalse(result.approved)
        self.assertIn("Avoid new branches", result.retry_hint)

    def test_reviewer_rejects_second_round_not_addressing_satd(self) -> None:
        client = _FakeReviewClient(
            {
                "approved": False,
                "failed_checks": ["not_addressing_satd"],
                "repair_constraints": ["make_one_direct_change_matching_satd"],
                "failure_anchor": "alignment",
            }
        )
        reviewer = OpenAIReviewer(client)
        repair = RepairAttempt(2, "plan", "def f():\n    return 2\n", "line", 0.5, "notes")
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "TODO: return the correct value",
                "original_code": "def f():\n    return 1\n",
                "latest_repair": repair,
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertFalse(result.approved)
        self.assertEqual(result.failed_checks, ["not_addressing_satd"])

    def test_reviewer_protects_satd_anchor_block_removal_from_alignment_reject(self) -> None:
        client = _FakeReviewClient(
            {
                "approved": False,
                "failed_checks": ["not_addressing_satd", "over_scoped_change", "unrelated_change"],
                "repair_constraints": ["make_one_direct_change_matching_satd"],
                "failure_anchor": "alignment",
            }
        )
        reviewer = OpenAIReviewer(client)
        original = (
            "def f():\n"
            "    x = 1\n"
            "    if False:  # FIXME would we still need this?\n"
            "        x = old_fix()\n"
            "    return x\n"
        )
        repaired = "def f():\n    x = 1\n    return x\n"
        repair = RepairAttempt(2, "plan", repaired, "function", 0.6, "notes")
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "FIXME would we still need this?",
                "original_code": original,
                "latest_repair": repair,
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertTrue(result.approved)
        self.assertEqual(result.failed_checks, [])

    def test_reviewer_protects_documentation_anchor_deletion_from_unrelated_change(self) -> None:
        client = _FakeReviewClient(
            {
                "approved": False,
                "failed_checks": ["unrelated_change"],
                "repair_constraints": ["avoid_unrelated_rewrite"],
                "failure_anchor": "scope",
            }
        )
        reviewer = OpenAIReviewer(client)
        original = (
            "def f():\n"
            "    \"\"\"\n"
            "    # TODO: Missing doc\n"
            "    \"\"\"\n"
            "    return 1\n"
        )
        repaired = "def f():\n    return 1\n"
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "TODO: Missing doc",
                "original_code": original,
                "latest_repair": RepairAttempt(2, "plan", repaired, "function", 0.6, "notes"),
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertTrue(result.approved)
        self.assertEqual(result.failed_checks, [])

    def test_reviewer_protects_replace_instruction_from_unrelated_reject(self) -> None:
        client = _FakeReviewClient(
            {
                "approved": False,
                "failed_checks": ["unrelated_change", "not_addressing_satd"],
                "repair_constraints": ["avoid_unrelated_rewrite"],
                "failure_anchor": "scope",
            }
        )
        reviewer = OpenAIReviewer(client)
        original = (
            "def _matmul(self, rhs):\n"
            "    # TODO: replace with `self.covar_mat @ rhs` on next release.\n"
            "    return old_matmul(self.covar_mat, rhs)\n"
        )
        repaired = "def _matmul(self, rhs):\n    return self.covar_mat @ rhs.contiguous()\n"
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "TODO: replace with `self.covar_mat @ rhs` on next release.",
                "original_code": original,
                "latest_repair": RepairAttempt(1, "plan", repaired, "function", 0.6, "notes"),
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertTrue(result.approved)
        self.assertEqual(result.failed_checks, [])

    def test_reviewer_protects_exception_change_instruction(self) -> None:
        client = _FakeReviewClient(
            {
                "approved": False,
                "failed_checks": ["unsupported_control_flow_change"],
                "repair_constraints": ["avoid_new_control_flow_paths"],
                "failure_anchor": "control_flow",
            }
        )
        reviewer = OpenAIReviewer(client)
        original = (
            "def test_add():\n"
            "    # TODO change this to ValueError once validation is fixed\n"
            "    with pytest.raises(TypeError):\n"
            "        obj.add(obj)\n"
        )
        repaired = "def test_add():\n    with pytest.raises(ValueError):\n        obj.add(obj)\n"
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "TODO change this to ValueError once validation is fixed",
                "original_code": original,
                "latest_repair": RepairAttempt(1, "plan", repaired, "line", 0.6, "notes"),
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertTrue(result.approved)
        self.assertEqual(result.failed_checks, [])

    def test_reviewer_protects_hack_cleanup_signature_formatting(self) -> None:
        client = _FakeReviewClient(
            {
                "approved": False,
                "failed_checks": ["unsupported_signature_change", "unrelated_change"],
                "repair_constraints": ["preserve_original_signature"],
                "failure_anchor": "signature",
            }
        )
        reviewer = OpenAIReviewer(client)
        original = (
            "def _setObject(self,id,object,roles=None,user=None):\n"
            "    self._setOb(id,object)\n"
            "    # This is a nasty hack that provides a workaround for old data\n"
            "    if self.__dict__.has_key('__allow_groups__'):\n"
            "        delattr(self, '__allow_groups__')\n"
            "    return id\n"
        )
        repaired = "def _setObject(self, id, object, roles=None, user=None):\n    self._setOb(id, object)\n    return id\n"
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "This is a nasty hack that provides a workaround for old data",
                "original_code": original,
                "latest_repair": RepairAttempt(1, "plan", repaired, "function", 0.6, "notes"),
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertTrue(result.approved)
        self.assertEqual(result.failed_checks, [])

    def test_reviewer_protects_rename_signature_instruction(self) -> None:
        client = _FakeReviewClient(
            {
                "approved": False,
                "failed_checks": ["unsupported_signature_change"],
                "repair_constraints": ["preserve_original_signature"],
                "failure_anchor": "signature",
            }
        )
        reviewer = OpenAIReviewer(client)
        original = (
            "# TODO: change name to self._set_root(root_identifier)\n"
            "def _set_root(self, root):\n"
            "    self._anchor = root\n"
        )
        repaired = "def _set_root(self, root_identifier):\n    self._anchor = root_identifier\n"
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "TODO: change name to self._set_root(root_identifier)",
                "original_code": original,
                "latest_repair": RepairAttempt(1, "plan", repaired, "line", 0.6, "notes"),
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertTrue(result.approved)
        self.assertEqual(result.failed_checks, [])

    def test_reviewer_accepts_dedented_valid_python_snippet(self) -> None:
        client = _FakeReviewClient({"approved": True, "failed_checks": [], "repair_constraints": [], "failure_anchor": ""})
        reviewer = OpenAIReviewer(client)
        original = "    def f(old=None):\n        return old\n"
        repaired = "    def f():\n        return None\n"
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "TODO remove old parameter",
                "original_code": original,
                "latest_repair": RepairAttempt(1, "plan", repaired, "line", 0.6, "notes"),
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertTrue(result.approved)
        self.assertNotIn("syntax_error", result.failed_checks)

    def test_reviewer_rejects_first_round_uncertain_for_retry(self) -> None:
        client = _FakeReviewClient(
            {
                "approved": False,
                "failed_checks": ["reviewer_uncertain"],
                "repair_constraints": ["make_smallest_local_edit"],
                "failure_anchor": "uncertain",
            }
        )
        reviewer = OpenAIReviewer(client)
        repair = RepairAttempt(1, "plan", "def f():\n    return 2\n", "line", 0.5, "notes")
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "TODO: return the correct value",
                "original_code": "def f():\n    return 1\n",
                "latest_repair": repair,
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertFalse(result.approved)
        self.assertEqual(result.failed_checks, ["reviewer_uncertain"])

    def test_reviewer_accepts_second_round_uncertain_only(self) -> None:
        client = _FakeReviewClient(
            {
                "approved": False,
                "failed_checks": ["reviewer_uncertain"],
                "repair_constraints": ["make_smallest_local_edit"],
                "failure_anchor": "uncertain",
            }
        )
        reviewer = OpenAIReviewer(client)
        repair = RepairAttempt(2, "plan", "def f():\n    return 2\n", "line", 0.5, "notes")
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "TODO: return the correct value",
                "original_code": "def f():\n    return 1\n",
                "latest_repair": repair,
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertTrue(result.approved)
        self.assertEqual(result.failed_checks, [])

    def test_reviewer_feedback_is_constraint_only(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)
        review = ReviewResult(
            round_id=1,
            approved=False,
            review_score=0.0,
            problem_alignment=0.0,
            minimality=0.0,
            semantic_preservation=0.0,
            internal_consistency=0.0,
            issues=["long natural language issue should not drive feedback"],
            revision_advice="replace the code with something else",
            reject_type="scope",
            rationale="long rationale",
            failed_checks=["over_scoped_change", "unsupported_signature_change", "new_helper_without_evidence", "semantic_drift_risk"],
            repair_constraints=["preserve_unrelated_lines", "preserve_original_signature", "avoid_new_helpers", "preserve_existing_behavior", "extra"],
            failure_anchor="scope",
            retry_hint="Make a smaller local edit and preserve unrelated lines.",
        )
        feedback = LangGraphSATDWorkflow._build_repair_feedback(workflow, review)
        self.assertEqual(
            set(feedback.keys()),
            {"repair_constraints", "retry_hint", "can_retry"},
        )
        self.assertLessEqual(len(feedback["repair_constraints"]), 4)
        self.assertIn("smaller local edit", feedback["retry_hint"])
        self.assertNotIn("failed_checks", feedback)
        self.assertNotIn("failure_anchor", feedback)
        self.assertNotIn("revision_advice", feedback)

    def test_workflow_retries_first_round_review_reject(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)
        review = ReviewResult(
            round_id=1,
            approved=False,
            review_score=0.0,
            problem_alignment=0.0,
            minimality=0.0,
            semantic_preservation=0.0,
            internal_consistency=0.0,
            issues=["anchor_not_modified"],
            revision_advice="constraints:modify_satd_anchor_region",
            reject_type="alignment",
            rationale="failed_checks:anchor_not_modified",
            failed_checks=["anchor_not_modified"],
            repair_constraints=["modify_satd_anchor_region"],
            failure_anchor="alignment",
        )
        state = {
            "latest_review": review,
            "round_id": 1,
            "max_rounds": 2,
            "repair_feedback": LangGraphSATDWorkflow._build_repair_feedback(workflow, review),
        }
        self.assertEqual(LangGraphSATDWorkflow._route_after_review(workflow, state), "repair")

    def test_workflow_retries_first_round_review_reject_even_without_feedback(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)
        review = ReviewResult(
            round_id=1,
            approved=False,
            review_score=0.0,
            problem_alignment=0.0,
            minimality=0.0,
            semantic_preservation=0.0,
            internal_consistency=0.0,
            issues=["reviewer fallback"],
            revision_advice="",
            reject_type="review_error",
            rationale="reviewer fallback",
            failed_checks=[],
            repair_constraints=[],
            failure_anchor="",
            retry_hint="",
        )
        feedback = LangGraphSATDWorkflow._build_repair_feedback(workflow, review)
        state = {
            "latest_review": review,
            "round_id": 1,
            "max_rounds": 2,
            "repair_feedback": None,
        }
        self.assertEqual(LangGraphSATDWorkflow._route_after_review(workflow, state), "repair")
        self.assertEqual(feedback["repair_constraints"], ["make_smallest_local_edit"])
        self.assertIn("SATD comment", feedback["retry_hint"])

    def test_workflow_does_not_repeat_second_round_after_api_timeout_fallback(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)
        repair = RepairAttempt(
            round_id=1,
            repair_plan="Fallback no-op repair because the fixer request failed.",
            repaired_code="def f():\n    return 1\n",
            changed_scope="none",
            confidence=0.0,
            notes="fixer_exception:baseline_context:APITimeoutError",
            candidate_mode="baseline_context",
        )
        review = ReviewResult(
            round_id=1,
            approved=False,
            review_score=0.0,
            problem_alignment=0.0,
            minimality=0.0,
            semantic_preservation=0.0,
            internal_consistency=0.0,
            issues=["no_effective_change"],
            revision_advice="",
            reject_type="alignment",
            rationale="failed_checks:no_effective_change",
            failed_checks=["no_effective_change"],
            repair_constraints=["modify_code_to_address_satd"],
            failure_anchor="alignment",
            retry_hint="Make one concrete local edit.",
        )
        state = {
            "latest_review": review,
            "latest_repair": repair,
            "round_id": 1,
            "max_rounds": 2,
            "repair_feedback": LangGraphSATDWorkflow._build_repair_feedback(workflow, review),
        }
        self.assertEqual(LangGraphSATDWorkflow._route_after_review(workflow, state), "drop")

    def test_workflow_drops_second_round_high_risk_review_reject(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)
        review = ReviewResult(
            round_id=2,
            approved=False,
            review_score=0.0,
            problem_alignment=0.0,
            minimality=0.0,
            semantic_preservation=0.0,
            internal_consistency=0.0,
            issues=["unrelated_change"],
            revision_advice="constraints:avoid_unrelated_rewrite",
            reject_type="scope",
            rationale="failed_checks:unrelated_change",
            failed_checks=["unrelated_change"],
            repair_constraints=["avoid_unrelated_rewrite"],
            failure_anchor="scope",
        )
        state = {
            "latest_review": review,
            "round_id": 2,
            "max_rounds": 2,
            "repair_feedback": LangGraphSATDWorkflow._build_repair_feedback(workflow, review),
        }
        self.assertEqual(LangGraphSATDWorkflow._route_after_review(workflow, state), "drop")

    def test_workflow_context_repair_error_recovers_with_no_context_candidate(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)
        workflow.fixer = _NoContextRecoveryFixer()
        workflow.context_client = OpenAICompatClient.__new__(OpenAICompatClient)
        workflow.context_client._is_content_filter_error = lambda exc: False
        workflow._log = lambda message: None
        workflow._task_label = lambda state: "[task]"
        state = {
            "task_id": "1",
            "round_id": 0,
            "original_code": "def f():\n    return 1\n",
            "candidate_repairs": [],
        }
        candidates, inquiry, contexts, missing, uncertainty, constraints = LangGraphSATDWorkflow._run_repair_candidates(
            workflow,
            state,
            {},
            1,
            "generic",
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].candidate_mode, "baseline_no_context")
        self.assertIn("recovered_from_context_candidate_error", candidates[0].notes)
        self.assertEqual(inquiry.reason, "candidate_mode_without_method_context")
        self.assertEqual(contexts, [])
        self.assertEqual(missing, [])
        self.assertTrue(any(stage == "generation_error" for stage, _ in workflow.fixer.checkpoints))

    def test_workflow_context_and_no_context_errors_still_return_fallback_candidate(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)
        workflow.fixer = _FailingFixer()
        workflow.context_client = OpenAICompatClient.__new__(OpenAICompatClient)
        workflow.context_client._is_content_filter_error = lambda exc: False
        workflow._log = lambda message: None
        workflow._task_label = lambda state: "[task]"
        state = {
            "task_id": "1",
            "round_id": 0,
            "original_code": "def f():\n    return 1\n",
            "candidate_repairs": [],
        }
        candidates, *_ = LangGraphSATDWorkflow._run_repair_candidates(
            workflow,
            state,
            {},
            1,
            "generic",
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].candidate_mode, "baseline_context")
        self.assertEqual(candidates[0].repaired_code, state["original_code"])
        self.assertIn("fixer_exception:baseline_context:RuntimeError", candidates[0].notes)
        self.assertTrue(any(stage == "generation_retry_error" for stage, _ in workflow.fixer.checkpoints))

    def test_workflow_run_lock_blocks_second_process_for_same_output_dir(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            lock_path = LangGraphSATDWorkflow._acquire_run_lock(workflow, output_dir)
            try:
                with self.assertRaises(RuntimeError):
                    LangGraphSATDWorkflow._acquire_run_lock(workflow, output_dir)
            finally:
                LangGraphSATDWorkflow._release_run_lock(workflow, lock_path)
            self.assertFalse(lock_path.exists())

    def test_reviewer_prompt_does_not_depend_on_analyzer_fields(self) -> None:
        client = _FakeReviewClient({"approved": True, "failed_checks": [], "repair_constraints": [], "failure_anchor": ""})
        reviewer = OpenAIReviewer(client)
        repair = RepairAttempt(1, "plan", "def f():\n    return 2\n", "line", 0.6, "notes")
        result = reviewer.run(
            {
                "task_id": "1",
                "user": "owner",
                "project": "repo",
                "file_path": "x.py",
                "satd_comment": "TODO: return the correct value",
                "original_code": "def f():\n    return 1\n",
                "latest_repair": repair,
                "method_inquiry": None,
                "retrieved_method_contexts": [],
                "missing_method_names": [],
                "github_context": None,
                "analysis": None,
            }
        )
        self.assertTrue(result.approved)
        self.assertNotIn("Analysis decision", client.user_prompt)
        self.assertNotIn("analysis score", client.user_prompt.lower())

    def test_simple_analyzer_two_false_checks_force_drop(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        payload = {
            "decision": "pass",
            "confidence": 0.61,
            "operation_concrete": "high",
            "localizable": "low",
            "local_scope": "low",
            "end_state_clear": "low",
            "comment_evidence": "implement this once endpoint is available",
            "code_evidence": "no local target is visible",
            "notes": "The request is not a local repair candidate.",
        }
        state = {
            "satd_route_type": "generic",
            "original_code": "def f(x):\n    return x",
        }
        result = OpenAIAnalyzer._coerce_analysis(analyzer, payload, state, source="llm")
        self.assertFalse(result.repairable)
        self.assertEqual(result.decision, "drop")

    def test_simple_analyzer_non_concrete_non_localizable_and_unclear_drop(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        payload = {
            "decision": "uncertain",
            "confidence": 0.57,
            "operation_concrete": "low",
            "localizable": "low",
            "local_scope": "partial",
            "end_state_clear": "low",
            "comment_evidence": "implement filters",
            "code_evidence": "no visible filter edit target",
            "notes": "The request is open-ended and lacks a concrete local target.",
        }
        state = {
            "satd_route_type": "generic",
            "original_code": "def f(x):\n    return x",
        }
        result = OpenAIAnalyzer._coerce_analysis(analyzer, payload, state, source="llm")
        self.assertFalse(result.repairable)
        self.assertEqual(result.decision, "drop")

    def test_simple_analyzer_vague_tone_but_visible_object_stays_uncertain(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        payload = {
            "decision": "drop",
            "confidence": 0.64,
            "operation_concrete": "low",
            "localizable": "high",
            "local_scope": "high",
            "end_state_clear": "partial",
            "comment_evidence": "why are these keys even here",
            "code_evidence": "keys 'pp_files' and 'pp_test_index' are visible",
            "notes": "The tone is vague, but the object under discussion is local and visible.",
        }
        state = {
            "satd_route_type": "generic",
            "original_code": "config = {'pp_files': [], 'pp_test_index': 0}",
        }
        result = OpenAIAnalyzer._coerce_analysis(analyzer, payload, state, source="llm")
        self.assertTrue(result.repairable)
        self.assertEqual(result.decision, "uncertain")
        self.assertEqual(result.localizable, "high")

    def test_grounded_target_hint_preserves_deprecate_as_uncertain(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        payload = {
            "decision": "drop",
            "confidence": 0.77,
            "operation_concrete": "low",
            "localizable": "low",
            "local_scope": "low",
            "end_state_clear": "low",
            "comment_evidence": "fully deprecate index name",
            "code_evidence": "index name path is visible",
            "notes": "The request looks broad.",
        }
        state = {
            "satd_route_type": "generic",
            "satd_comment": "TODO at some point fully deprecate index name - it could be buggy",
            "original_code": "if index_name is not None:\n    kwargs['hint'] = index_name",
        }
        result = OpenAIAnalyzer._coerce_analysis(analyzer, payload, state, source="llm")
        self.assertTrue(result.repairable)
        self.assertEqual(result.decision, "uncertain")
        self.assertEqual(result.operation_concrete, "low")
        self.assertEqual(result.localizable, "partial")
        self.assertEqual(result.local_scope, "partial")
        self.assertEqual(result.end_state_clear, "low")
        self.assertIn("grounded_local_target", result.evidence_summary)

    def test_open_ended_task_does_not_get_existing_target_floor(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        payload = {
            "decision": "drop",
            "confidence": 0.82,
            "operation_concrete": "low",
            "localizable": "low",
            "local_scope": "low",
            "end_state_clear": "low",
            "comment_evidence": "implement API support here",
            "code_evidence": "current function is visible",
            "notes": "This remains open-ended.",
        }
        state = {
            "satd_route_type": "generic",
            "satd_comment": "TODO: implement API support here",
            "original_code": "def handle_request(x):\n    return x",
        }
        result = OpenAIAnalyzer._coerce_analysis(analyzer, payload, state, source="llm")
        self.assertFalse(result.repairable)
        self.assertEqual(result.decision, "drop")

    def test_simple_analyzer_future_wording_can_still_pass(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        payload = {
            "decision": "pass",
            "confidence": 0.79,
            "operation_concrete": "high",
            "localizable": "high",
            "local_scope": "high",
            "end_state_clear": "high",
            "comment_evidence": "remove default from condition after airflow update",
            "code_evidence": "default value branch is visible in the snippet",
            "notes": "The desired local edit is still clear despite future timing wording.",
        }
        state = {
            "satd_route_type": "generic",
            "satd_comment": "TODO remove default from condition after airflow update",
            "original_code": "def f(x):\n    return x",
        }
        result = OpenAIAnalyzer._coerce_analysis(analyzer, payload, state, source="llm")
        self.assertTrue(result.repairable)
        self.assertEqual(result.decision, "pass")

    def test_simple_analyzer_ambiguous_open_check_drops(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        payload = {
            "decision": "uncertain",
            "confidence": 0.81,
            "operation_concrete": "partial",
            "localizable": "partial",
            "local_scope": "partial",
            "end_state_clear": "partial",
            "comment_evidence": "consider doing additional check",
            "code_evidence": "the subnet check location is visible but underspecified",
            "notes": "The local area is visible, but the concrete edit operation is still underspecified.",
        }
        state = {
            "satd_route_type": "generic",
            "satd_comment": "TODO: consider doing additional check for subnet such as",
            "original_code": "def f(x):\n    return x",
        }
        result = OpenAIAnalyzer._coerce_analysis(analyzer, payload, state, source="llm")
        self.assertFalse(result.repairable)
        self.assertEqual(result.decision, "drop")
        self.assertIn("open_ended_without_specific_local_edit", result.evidence_summary)

    def test_simple_analyzer_partial_partial_high_partial_stays_uncertain_in_stage1(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        payload = {
            "decision": "uncertain",
            "confidence": 0.58,
            "operation_concrete": "partial",
            "localizable": "partial",
            "local_scope": "high",
            "end_state_clear": "partial",
            "comment_evidence": "add more validation here",
            "code_evidence": "local block is visible but target and end state are not pinned down",
            "notes": "The task is local in scope, but still underspecified in operation, localization, and outcome.",
        }
        state = {
            "satd_route_type": "generic",
            "satd_comment": "TODO: add more validation here.",
            "original_code": "def f(x):\n    return x",
        }
        result = OpenAIAnalyzer._coerce_analysis(analyzer, payload, state, source="llm")
        self.assertTrue(result.repairable)
        self.assertEqual(result.decision, "uncertain")

    def test_second_stage_candidate_requires_any_context_signal(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        base = OpenAIAnalyzer._build_result(
            analyzer,
            decision="uncertain",
            confidence=0.58,
            operation_concrete="partial",
            localizable="partial",
            local_scope="high",
            end_state_clear="partial",
            notes="Borderline local task.",
            comment_evidence="add more validation",
            code_evidence="local block is visible",
            satd_type="generic",
            source="llm",
            scope_radius="function",
        )
        state = {
            "satd_route_type": "generic",
            "github_context": {
                "base_context": {"target_function": {"found": False}},
                "repair_context": {
                    "module_symbols": {"count": 0},
                    "targeted_callsite_snippet": {"used": False},
                    "same_file_pattern": {"count": 1},
                },
                "metadata": {"call_sites_count": 0},
            },
        }
        self.assertTrue(OpenAIAnalyzer._is_second_stage_uncertain_candidate(analyzer, state, base))
        no_context_state = {"satd_route_type": "generic", "github_context": {"base_context": {}, "repair_context": {}}}
        self.assertFalse(OpenAIAnalyzer._is_second_stage_uncertain_candidate(analyzer, no_context_state, base))

    def test_second_stage_candidate_allows_weak_local_shape_without_context(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        base = OpenAIAnalyzer._build_result(
            analyzer,
            decision="uncertain",
            confidence=0.55,
            operation_concrete="partial",
            localizable="high",
            local_scope="high",
            end_state_clear="partial",
            notes="Local slot is visible but the repair is still weakly specified.",
            comment_evidence="use settings",
            code_evidence='repos_dir = os.path.expanduser("~/.rez/formulae-repos")',
            satd_type="generic",
            source="llm",
            scope_radius="function",
        )
        state = {"satd_route_type": "generic", "github_context": {"base_context": {}, "repair_context": {}}}
        self.assertTrue(OpenAIAnalyzer._is_second_stage_uncertain_candidate(analyzer, state, base))

    def test_second_stage_can_drop_contextually_open_ended_uncertain_candidate(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        base = OpenAIAnalyzer._build_result(
            analyzer,
            decision="uncertain",
            confidence=0.58,
            operation_concrete="partial",
            localizable="partial",
            local_scope="high",
            end_state_clear="partial",
            notes="Borderline local task.",
            comment_evidence="implement add margins when support lands",
            code_evidence="local function is visible",
            satd_type="generic",
            source="llm",
            scope_radius="function",
        )
        payload = {
            "decision": "drop",
            "confidence": 0.82,
            "target_existence": "low",
            "usage_locality": "low",
            "replacement_evidence": "low",
            "task_closedness": "low",
            "notes": "This is fundamentally a capability implementation request.",
        }
        result = OpenAIAnalyzer._coerce_uncertain_compression(
            analyzer,
            payload,
            base,
            {"satd_comment": "TODO: implement add margins when support lands"},
        )
        self.assertFalse(result.repairable)
        self.assertEqual(result.decision, "drop")

    def test_second_stage_keeps_borderline_uncertain_when_local_edit_remains_plausible(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        base = OpenAIAnalyzer._build_result(
            analyzer,
            decision="uncertain",
            confidence=0.58,
            operation_concrete="partial",
            localizable="partial",
            local_scope="high",
            end_state_clear="partial",
            notes="Borderline local task.",
            comment_evidence="deprecated index name path is visible",
            code_evidence="current parameter path is visible",
            satd_type="generic",
            source="llm",
            scope_radius="function",
        )
        payload = {
            "decision": "drop",
            "confidence": 0.82,
            "target_existence": "high",
            "usage_locality": "partial",
            "replacement_evidence": "partial",
            "task_closedness": "partial",
            "notes": "This may still be a local edit despite ambiguity.",
        }
        result = OpenAIAnalyzer._coerce_uncertain_compression(
            analyzer,
            payload,
            base,
            {"satd_comment": "TODO at some point fully deprecate index name - it could be buggy"},
        )
        self.assertTrue(result.repairable)
        self.assertEqual(result.decision, "uncertain")

    def test_second_stage_drops_when_usage_and_closedness_are_low(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        base = OpenAIAnalyzer._build_result(
            analyzer,
            decision="uncertain",
            confidence=0.52,
            operation_concrete="partial",
            localizable="high",
            local_scope="partial",
            end_state_clear="partial",
            notes="Borderline context-backed task.",
            comment_evidence="tie this into global output controls",
            code_evidence="current helper exists but task remains broad",
            satd_type="generic",
            source="llm",
            scope_radius="function",
        )
        payload = {
            "decision": "drop",
            "confidence": 0.72,
            "target_existence": "partial",
            "usage_locality": "low",
            "replacement_evidence": "partial",
            "task_closedness": "low",
            "notes": "The context still shows this as a broad open-ended integration task.",
        }
        result = OpenAIAnalyzer._coerce_uncertain_compression(
            analyzer,
            payload,
            base,
            {"satd_comment": "TODO: tie this into global output controls"},
        )
        self.assertFalse(result.repairable)
        self.assertEqual(result.decision, "drop")

    def test_second_stage_drops_partial_target_when_context_still_nonlocal_and_open(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        base = OpenAIAnalyzer._build_result(
            analyzer,
            decision="uncertain",
            confidence=0.50,
            operation_concrete="partial",
            localizable="partial",
            local_scope="high",
            end_state_clear="partial",
            notes="Borderline context-backed task.",
            comment_evidence="tie this into system-wide controls",
            code_evidence="current function exists but broader control path is unclear",
            satd_type="generic",
            source="llm",
            scope_radius="function",
        )
        payload = {
            "decision": "drop",
            "confidence": 0.67,
            "target_existence": "partial",
            "usage_locality": "low",
            "replacement_evidence": "partial",
            "task_closedness": "low",
            "notes": "The context still points to a broad open-ended integration task.",
        }
        result = OpenAIAnalyzer._coerce_uncertain_compression(
            analyzer,
            payload,
            base,
            {"satd_comment": "TODO: tie this into system-wide controls"},
        )
        self.assertFalse(result.repairable)
        self.assertEqual(result.decision, "drop")

    def test_second_stage_keeps_when_replacement_evidence_is_high(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        base = OpenAIAnalyzer._build_result(
            analyzer,
            decision="uncertain",
            confidence=0.52,
            operation_concrete="partial",
            localizable="partial",
            local_scope="high",
            end_state_clear="partial",
            notes="Borderline context-backed task.",
            comment_evidence="migrate this path carefully",
            code_evidence="existing path is visible",
            satd_type="generic",
            source="llm",
            scope_radius="function",
        )
        payload = {
            "decision": "drop",
            "confidence": 0.80,
            "target_existence": "partial",
            "usage_locality": "low",
            "replacement_evidence": "high",
            "task_closedness": "low",
            "notes": "There is still a strong replacement path in context.",
        }
        result = OpenAIAnalyzer._coerce_uncertain_compression(
            analyzer,
            payload,
            base,
            {"satd_comment": "TODO: migrate this path carefully"},
        )
        self.assertTrue(result.repairable)
        self.assertEqual(result.decision, "uncertain")

    def test_second_stage_drops_open_ended_task_even_when_local_slot_is_visible(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        base = OpenAIAnalyzer._build_result(
            analyzer,
            decision="uncertain",
            confidence=0.51,
            operation_concrete="low",
            localizable="high",
            local_scope="high",
            end_state_clear="low",
            notes="The local function is visible but the requested task remains open-ended.",
            comment_evidence="implement this function once endpoint is available",
            code_evidence="function body exists but no closed local patch is specified",
            satd_type="generic",
            source="llm",
            scope_radius="function",
        )
        payload = {
            "decision": "drop",
            "confidence": 0.72,
            "target_existence": "partial",
            "usage_locality": "partial",
            "replacement_evidence": "low",
            "task_closedness": "low",
            "notes": "The context still indicates an open-ended implementation task.",
        }
        result = OpenAIAnalyzer._coerce_uncertain_compression(
            analyzer,
            payload,
            base,
            {"satd_comment": "TODO: implement this function once endpoint is available"},
        )
        self.assertFalse(result.repairable)
        self.assertEqual(result.decision, "drop")

    def test_second_stage_keeps_local_description_adjustment_visible_in_code(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        base = OpenAIAnalyzer._build_result(
            analyzer,
            decision="uncertain",
            confidence=0.50,
            operation_concrete="low",
            localizable="high",
            local_scope="high",
            end_state_clear="low",
            notes="The local target is visible but the wording is terse.",
            comment_evidence="update description",
            code_evidence="description field is visible in the current object",
            satd_type="generic",
            source="llm",
            scope_radius="function",
        )
        payload = {
            "decision": "drop",
            "confidence": 0.74,
            "target_existence": "partial",
            "usage_locality": "partial",
            "replacement_evidence": "partial",
            "task_closedness": "partial",
            "notes": "This could still be a local documentation adjustment.",
        }
        result = OpenAIAnalyzer._coerce_uncertain_compression(
            analyzer,
            payload,
            base,
            {"satd_comment": "TODO : Update description"},
        )
        self.assertTrue(result.repairable)
        self.assertEqual(result.decision, "uncertain")

    def test_analysis_only_accept_node_marks_analyzer_pass(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)
        workflow.analysis_only = True
        workflow.verbose = False
        workflow._log = lambda message: None
        result = LangGraphSATDWorkflow._accept_node(workflow, {"task_id": "1", "task_index": 0, "task_total": 0})
        self.assertEqual(result["status"], "passed_by_analyzer")
        self.assertIsNone(result["final_repaired_code"])

    def test_remove_temporary_route_matches_future_removal_phrasing(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)
        state = {"satd_comment": "FIXME: remove it on HG1710, when this", "original_code": "x = 1"}
        route = LangGraphSATDWorkflow._infer_satd_route_type(workflow, state)
        self.assertEqual(route, "remove_temporary")

    def test_document_route_matches_update_description_phrasing(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)
        state = {"satd_comment": "TODO : Update description", "original_code": "x = 1"}
        route = LangGraphSATDWorkflow._infer_satd_route_type(workflow, state)
        self.assertEqual(route, "document")

    def test_analyzer_context_summary_contains_definition_and_usage(self) -> None:
        analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
        bundle = {
            "base_context": {
                "target_function": {
                    "found": True,
                    "symbol_name": "find_items",
                    "class_name": "RepoClient",
                    "matched_by": "signature",
                }
            },
            "repair_context": {
                "module_symbols": {
                    "count": 2,
                    "items": [
                        {"symbol_name": "find_items", "kind": "function"},
                        {"symbol_name": "index_name", "kind": "assign"},
                    ],
                },
                "targeted_callsite_snippet": {
                    "used": True,
                    "item": {
                        "path": "pkg/example.py",
                        "excerpt": "result = find_items(index_name=index_name)",
                    },
                },
            },
        }
        text = OpenAIAnalyzer._build_analyzer_context_summary(analyzer, bundle)
        payload = json.loads(text)
        self.assertEqual(payload["definition_context"]["symbol_name"], "find_items")
        self.assertEqual(payload["usage_context"]["targeted_callsite_path"], "pkg/example.py")

    def test_analyze_node_builds_shared_context_only_for_generic(self) -> None:
        workflow = LangGraphSATDWorkflow.__new__(LangGraphSATDWorkflow)
        workflow.use_analyzer = False
        workflow._log = lambda message: None
        workflow._task_label = lambda state: "[task]"
        workflow._rule_based_analyzer_drop = lambda state: None
        workflow._fallback_analysis = lambda reason: None
        workflow._bypass_analysis = lambda state, route: OpenAIAnalyzer.build_easy_route_analysis(OpenAIAnalyzer.__new__(OpenAIAnalyzer), route)
        calls = []
        workflow._load_or_build_shared_context = lambda state: calls.append(state["task_id"]) or {"metadata": {"shared_context_mode": True}}

        generic_state = {"task_id": "g1", "satd_comment": "TODO: investigate this", "original_code": "def f():\n    pass"}
        workflow._infer_satd_route_type = lambda state: "generic"
        generic_result = LangGraphSATDWorkflow._analyze_node(workflow, generic_state)
        self.assertEqual(calls, ["g1"])
        self.assertEqual((generic_result.get("github_context") or {}).get("metadata", {}).get("shared_context_mode"), True)

        calls.clear()
        easy_state = {"task_id": "e1", "satd_comment": "TODO: document method", "original_code": "def f():\n    pass"}
        workflow._infer_satd_route_type = lambda state: "document"
        easy_result = LangGraphSATDWorkflow._analyze_node(workflow, easy_state)
        self.assertEqual(calls, [])
        self.assertIsNone(easy_result.get("github_context"))


if __name__ == "__main__":
    unittest.main()
