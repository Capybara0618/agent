from __future__ import annotations

import ast
import difflib
import re
import textwrap
from typing import Any

from ..schema import GraphState, RepairAttempt, preprocess_python_code


class ReviewSummaryTool:
    """Compute compact structural evidence for reviewer prompts."""

    def structural_evidence(self, state: GraphState, repair: RepairAttempt) -> dict[str, Any]:
        original_code = state["original_code"] or ""
        repaired_code = repair.repaired_code or ""
        original_profile = self._structure_profile(original_code)
        repaired_profile = self._structure_profile(repaired_code)
        normalized_original = preprocess_python_code(original_code)
        normalized_repaired = preprocess_python_code(repaired_code)
        changed_line_count = self._changed_line_count(original_code, repaired_code)
        added_line_count, deleted_line_count = self._line_delta_counts(original_code, repaired_code)
        return {
            "raw_code_changed": original_code.strip() != repaired_code.strip(),
            "normalized_code_changed": normalized_original != normalized_repaired,
            "changed_line_count": changed_line_count,
            "added_line_count": added_line_count,
            "deleted_line_count": deleted_line_count,
            "signature_changed": self._signature_changed(original_code, repaired_code),
            "structure_delta": {
                "definitions": repaired_profile["definitions"] - original_profile["definitions"],
                "returns": repaired_profile["returns"] - original_profile["returns"],
                "raises": repaired_profile["raises"] - original_profile["raises"],
                "branches": repaired_profile["branches"] - original_profile["branches"],
                "imports": repaired_profile["imports"] - original_profile["imports"],
            },
        }

    def syntax_valid(self, code: str) -> bool:
        try:
            ast.parse(code or "")
            return True
        except SyntaxError:
            try:
                ast.parse(textwrap.dedent(code or ""))
                return True
            except SyntaxError:
                return False

    def _structure_profile(self, code: str) -> dict[str, int]:
        source = textwrap.dedent(code or "")
        try:
            tree = ast.parse(source)
            return {
                "definitions": sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree)),
                "returns": sum(isinstance(node, ast.Return) for node in ast.walk(tree)),
                "raises": sum(isinstance(node, ast.Raise) for node in ast.walk(tree)),
                "branches": sum(isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)) for node in ast.walk(tree)),
                "imports": sum(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)),
            }
        except SyntaxError:
            return {
                "definitions": len(re.findall(r"^\s*(?:async\s+def|def|class)\s+", source, flags=re.MULTILINE)),
                "returns": len(re.findall(r"\breturn\b", source)),
                "raises": len(re.findall(r"\braise\b", source)),
                "branches": len(re.findall(r"\b(if|for|while|try|with)\b", source)),
                "imports": len(re.findall(r"^\s*(?:import|from)\s+", source, flags=re.MULTILINE)),
            }

    def _first_signature_line(self, code: str) -> str:
        for line in (code or "").splitlines():
            stripped = line.strip()
            if stripped.startswith(("def ", "async def ", "class ")):
                return stripped
        return ""

    def _signature_changed(self, original_code: str, repaired_code: str) -> bool:
        original_signature = self._first_signature_line(original_code)
        repaired_signature = self._first_signature_line(repaired_code)
        return bool(original_signature and repaired_signature and original_signature != repaired_signature)

    def _changed_line_count(self, original_code: str, repaired_code: str) -> int:
        original_lines = (original_code or "").splitlines()
        repaired_lines = (repaired_code or "").splitlines()
        changed = abs(len(original_lines) - len(repaired_lines))
        for before, after in zip(original_lines, repaired_lines):
            if before != after:
                changed += 1
        return changed

    def _line_delta_counts(self, original_code: str, repaired_code: str) -> tuple[int, int]:
        original_lines = (original_code or "").splitlines()
        repaired_lines = (repaired_code or "").splitlines()
        added = 0
        deleted = 0
        matcher = difflib.SequenceMatcher(a=original_lines, b=repaired_lines)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag in {"insert", "replace"}:
                added += max(0, j2 - j1)
            if tag in {"delete", "replace"}:
                deleted += max(0, i2 - i1)
        return added, deleted
