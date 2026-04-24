from __future__ import annotations

import unittest
import json

from satd_langgraph.agents import OpenAIAnalyzer, OpenAICompatClient, OpenAIFixer
from satd_langgraph.workflow import LangGraphSATDWorkflow


class ContextPromptSafetyTests(unittest.TestCase):
    def test_prompt_coercion_flattens_sequences(self) -> None:
        client = OpenAICompatClient.__new__(OpenAICompatClient)
        self.assertEqual(client._coerce_prompt_text(("a", "b", ("c", None, 1))), "abc1")

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

    def test_existing_target_hint_lifts_all_low_to_partial_when_not_open_ended(self) -> None:
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
        self.assertEqual(result.operation_concrete, "partial")
        self.assertEqual(result.localizable, "partial")
        self.assertEqual(result.end_state_clear, "partial")

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

    def test_simple_analyzer_ambiguous_end_state_stays_uncertain(self) -> None:
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
        self.assertTrue(result.repairable)
        self.assertEqual(result.decision, "uncertain")

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
