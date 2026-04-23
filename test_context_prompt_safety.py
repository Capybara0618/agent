from __future__ import annotations

import unittest

from satd_langgraph.agents import OpenAICompatClient, OpenAIFixer


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


if __name__ == "__main__":
    unittest.main()
