from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from typing import Any

from .openai_client import OpenAICompatClient
from .schema import preprocess_python_code


@dataclass(frozen=True)
class LLMJudgeResult:
    llm_as_judge: float
    reason: str = ""


@dataclass(frozen=True)
class _StaticCodeFacts:
    names: set[str]
    loaded_names: set[str]
    defined_names: set[str]
    imports: set[str]
    function_defs: set[str]
    class_defs: set[str]
    call_names: set[str]
    call_attrs: set[str]
    self_attrs: set[str]
    keyword_names: set[str]


ZERO_LLM_JUDGE = LLMJudgeResult(llm_as_judge=0.0, reason="")


class LLMRepairJudge:
    """Offline semantic repair evaluator.

    The manual repair is used only as evaluation reference. It must not be
    used by the repair pipeline before candidate generation.
    """

    def __init__(self, client: OpenAICompatClient, max_code_chars: int = 7000) -> None:
        self.client = client
        self.max_code_chars = max(1200, int(max_code_chars))

    def judge(
        self,
        original_code: str | None,
        manual_code: str | None,
        candidate_code: str | None,
        satd_comment: str | None,
    ) -> LLMJudgeResult:
        if not candidate_code or not str(candidate_code).strip():
            return ZERO_LLM_JUDGE
        if not manual_code or not str(manual_code).strip():
            return ZERO_LLM_JUDGE
        if not self._syntax_valid(candidate_code):
            return LLMJudgeResult(llm_as_judge=0.0, reason="candidate_syntax_invalid")

        if preprocess_python_code(candidate_code) == preprocess_python_code(manual_code):
            return LLMJudgeResult(llm_as_judge=1.0, reason="exact_match")

        fabrication_reason = self._static_fabrication_reason(original_code, manual_code, candidate_code)
        if fabrication_reason:
            return LLMJudgeResult(llm_as_judge=0.0, reason=fabrication_reason)

        payload = self.client.generate_json(
            self._system_prompt(),
            self._user_prompt(original_code, manual_code, candidate_code, satd_comment),
            temperature=0.0,
            request_label="llm_as_judge",
            max_tokens=256,
        )
        score = self._score_from_payload(payload)
        reason = str(payload.get("reason") or "").strip()
        return LLMJudgeResult(llm_as_judge=score, reason=reason)

    def _system_prompt(self) -> str:
        return (
            "You are a strict offline evaluator for SATD code repair. "
            "Your job is to decide whether a candidate repair is semantically equivalent to the human repair. "
            "Compare the repair effect from original code to human repair with the repair effect from original "
            "code to candidate repair. The candidate does not need textual or structural exact match, but it must "
            "preserve the same key repair meaning and external contract as the human repair. "
            "Do not reward a merely plausible alternative repair if it differs semantically from the human repair. "
            "Return only JSON."
        )

    def _user_prompt(
        self,
        original_code: str | None,
        manual_code: str | None,
        candidate_code: str | None,
        satd_comment: str | None,
    ) -> str:
        return (
            "Judge whether the candidate repair should be counted as successful.\n\n"
            "Use this decision order:\n"
            "1. Identify the key repair effect made by the human repair relative to the original code.\n"
            "2. Identify the key repair effect made by the candidate repair relative to the original code.\n"
            "3. Decide whether those two repair effects are semantically equivalent for the SATD item.\n"
            "4. Check that the candidate is logically coherent and valid Python.\n"
            "5. Check that the candidate does not rely on fabricated program elements or break external contracts.\n\n"
            "Count as successful when all of these are true:\n"
            "- The candidate resolves the same SATD repair intent as the human repair.\n"
            "- The candidate covers the human repair's key behavior changes. It may use a different implementation "
            "form, but the observable repair effect must be equivalent.\n"
            "- The candidate logic is coherent: control flow, data flow, return behavior, exception behavior, "
            "and state updates make sense.\n"
            "- The candidate does not invent unsupported APIs, helper methods, attributes, parameters, imports, "
            "or variables. Normal local temporary variables are acceptable only when fully defined and "
            "behavior-preserving.\n"
            "- The candidate does not break key external contracts preserved by the human repair, such as "
            "function signature, return structure, exception type, required keyword parameters, or project API usage, "
            "unless the human repair makes the same kind of contract change.\n\n"
            "Implementation form may differ. Accept harmless reordering of independent statements, keyword argument "
            "order changes, equivalent variable names, formatting differences, equivalent local rewrites, and minor "
            "detail differences when the repair effect remains equivalent to the human repair.\n\n"
            "Return llm_as_judge = 0 if the candidate only removes the SATD comment, misses a key behavior from "
            "the human repair, implements a different repair goal, is logically broken, uses fabricated program "
            "elements, changes unrelated behavior, or changes an external contract that the human repair preserves.\n\n"
            'Return JSON with exactly: {"llm_as_judge": 0 or 1, "reason": "short reason"}.\n\n'
            f"SATD comment:\n{satd_comment or ''}\n\n"
            f"Original code:\n{self._compact_code(original_code)}\n\n"
            f"Human repair reference:\n{self._compact_code(manual_code)}\n\n"
            f"Candidate repair:\n{self._compact_code(candidate_code)}"
        )

    def _compact_code(self, code: str | None) -> str:
        text = (code or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if len(text) <= self.max_code_chars:
            return text
        head_len = self.max_code_chars // 2
        tail_len = self.max_code_chars - head_len
        return text[:head_len] + "\n\n# ... code truncated for evaluation ...\n\n" + text[-tail_len:]

    def _score_from_payload(self, payload: dict[str, Any]) -> float:
        value = payload.get("llm_as_judge")
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return 1.0 if float(value) >= 0.5 else 0.0
        cleaned = str(value or "").strip().lower()
        return 1.0 if cleaned in {"1", "true", "yes", "success", "successful", "pass"} else 0.0

    def _syntax_valid(self, code: str | None) -> bool:
        try:
            ast.parse(code or "")
            return True
        except SyntaxError:
            return False

    def _static_fabrication_reason(
        self,
        original_code: str | None,
        manual_code: str | None,
        candidate_code: str | None,
    ) -> str:
        try:
            original = self._static_facts(original_code)
            manual = self._static_facts(manual_code)
            candidate = self._static_facts(candidate_code)
        except SyntaxError:
            return "candidate_syntax_invalid"

        reference_names = original.names | manual.names
        reference_calls = original.call_names | manual.call_names
        reference_call_attrs = original.call_attrs | manual.call_attrs
        reference_self_attrs = original.self_attrs | manual.self_attrs
        reference_keywords = original.keyword_names | manual.keyword_names
        reference_imports = original.imports | manual.imports
        reference_defs = original.function_defs | manual.function_defs | original.class_defs | manual.class_defs
        reference_names = reference_names | reference_imports | reference_defs

        new_imports = sorted(candidate.imports - reference_imports)
        if new_imports:
            return f"fabricated_import:{new_imports[0]}"

        new_nested_defs = sorted((candidate.function_defs | candidate.class_defs) - reference_defs)
        if new_nested_defs:
            return f"fabricated_helper_definition:{new_nested_defs[0]}"

        allowed_names = set(dir(builtins)) | {"self", "cls", "None", "True", "False"}
        undefined_names = sorted(
            name
            for name in candidate.loaded_names
            if name not in candidate.defined_names
            and name not in reference_names
            and name not in allowed_names
            and not name.startswith("__")
        )
        if undefined_names:
            return f"fabricated_or_undefined_variable:{undefined_names[0]}"

        new_self_attrs = sorted(candidate.self_attrs - reference_self_attrs)
        if new_self_attrs:
            return f"fabricated_self_attribute_or_method:{new_self_attrs[0]}"

        suspicious_calls = sorted(
            name
            for name in candidate.call_names
            if name not in reference_calls
            and name not in reference_names
            and name not in candidate.defined_names
            and name not in allowed_names
        )
        if suspicious_calls:
            return f"fabricated_function_call:{suspicious_calls[0]}"

        suspicious_attr_calls = sorted(
            attr
            for attr in candidate.call_attrs
            if attr not in reference_call_attrs
            and attr not in reference_self_attrs
            and not self._common_builtin_attr(attr)
        )
        if suspicious_attr_calls:
            return f"fabricated_attribute_call:{suspicious_attr_calls[0]}"

        new_keywords = sorted(
            name
            for name in candidate.keyword_names
            if name not in reference_keywords and name not in {"key", "reverse"}
        )
        if new_keywords:
            return f"fabricated_or_unjustified_keyword:{new_keywords[0]}"

        return ""

    def _static_facts(self, code: str | None) -> _StaticCodeFacts:
        tree = ast.parse(code or "")
        facts = _StaticFactVisitor()
        facts.visit(tree)
        return _StaticCodeFacts(
            names=facts.names,
            loaded_names=facts.loaded_names,
            defined_names=facts.defined_names,
            imports=facts.imports,
            function_defs=facts.function_defs,
            class_defs=facts.class_defs,
            call_names=facts.call_names,
            call_attrs=facts.call_attrs,
            self_attrs=facts.self_attrs,
            keyword_names=facts.keyword_names,
        )

    def _common_builtin_attr(self, attr: str) -> bool:
        return attr in {
            "append",
            "clear",
            "copy",
            "decode",
            "encode",
            "endswith",
            "extend",
            "format",
            "get",
            "items",
            "join",
            "keys",
            "lower",
            "pop",
            "remove",
            "replace",
            "rstrip",
            "setdefault",
            "split",
            "startswith",
            "strip",
            "update",
            "upper",
            "values",
        }


class _StaticFactVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()
        self.loaded_names: set[str] = set()
        self.defined_names: set[str] = set()
        self.imports: set[str] = set()
        self.function_defs: set[str] = set()
        self.class_defs: set[str] = set()
        self.call_names: set[str] = set()
        self.call_attrs: set[str] = set()
        self.self_attrs: set[str] = set()
        self.keyword_names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.function_defs.add(node.name)
        self.defined_names.add(node.name)
        self._visit_arguments(node.args)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.class_defs.add(node.name)
        self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            imported = alias.asname or alias.name.split(".")[0]
            self.imports.add(imported)
            self.defined_names.add(imported)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        for alias in node.names:
            imported = alias.asname or alias.name
            self.imports.add(imported)
            self.defined_names.add(imported)

    def visit_Name(self, node: ast.Name) -> Any:
        self.names.add(node.id)
        if isinstance(node.ctx, ast.Load):
            self.loaded_names.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Param)):
            self.defined_names.add(node.id)

    def visit_arg(self, node: ast.arg) -> Any:
        self.defined_names.add(node.arg)

    def visit_Assign(self, node: ast.Assign) -> Any:
        for target in node.targets:
            self._collect_target(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        self._collect_target(node.target)
        if node.value:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        self._collect_target(node.target)
        self.visit(node.target)
        self.visit(node.value)

    def visit_For(self, node: ast.For) -> Any:
        self._collect_target(node.target)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:
        self.visit_For(node)

    def visit_With(self, node: ast.With) -> Any:
        for item in node.items:
            if item.optional_vars is not None:
                self._collect_target(item.optional_vars)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> Any:
        self.visit_With(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> Any:
        if node.name:
            self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name):
            self.call_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.call_attrs.add(node.func.attr)
            root = self._attribute_root(node.func)
            if root in {"self", "cls"}:
                self.self_attrs.add(node.func.attr)
        for keyword in node.keywords:
            if keyword.arg:
                self.keyword_names.add(keyword.arg)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        root = self._attribute_root(node)
        if root in {"self", "cls"}:
            self.self_attrs.add(node.attr)
        self.generic_visit(node)

    def _visit_arguments(self, args: ast.arguments) -> None:
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            self.defined_names.add(arg.arg)
        if args.vararg:
            self.defined_names.add(args.vararg.arg)
        if args.kwarg:
            self.defined_names.add(args.kwarg.arg)

    def _collect_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.defined_names.add(target.id)
        elif isinstance(target, ast.Attribute):
            root = self._attribute_root(target)
            if root in {"self", "cls"}:
                self.self_attrs.add(target.attr)
            self.visit(target.value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._collect_target(item)
        elif isinstance(target, ast.Starred):
            self._collect_target(target.value)
        else:
            self.visit(target)

    def _attribute_root(self, node: ast.AST) -> str:
        current = node
        while isinstance(current, ast.Attribute):
            current = current.value
        if isinstance(current, ast.Name):
            return current.id
        return ""


def llm_judge_result_to_row(result: LLMJudgeResult | None) -> dict[str, float | str]:
    if result is None:
        return {"LLM_as_judge": ""}
    return {"LLM_as_judge": result.llm_as_judge}
