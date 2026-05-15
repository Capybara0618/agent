from __future__ import annotations

import ast
import io
import re
import textwrap
import tokenize
from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict


@dataclass
class SATDRecord:
    task_id: str
    satd_comment: str
    original_code: str
    manual_code: str
    user: str
    project: str
    file_path: str
    commit: str
    em_label: str


@dataclass
class AnalysisResult:
    decision: str
    repairable: bool
    reason: str
    repair_plan: str = ""
    target_summary: str = ""
    context_summary: str = ""


@dataclass
class RepairAttempt:
    round_id: int
    repair_plan: str
    repaired_code: str
    changed_scope: str
    confidence: float
    notes: str
    candidate_mode: str = "single"


@dataclass
class MethodInquiryResult:
    required_methods: list[str] = field(default_factory=list)
    reason: str = ""
    method_notes: list[dict[str, Any]] = field(default_factory=list)
    uncertainty_items: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ContextNeedDecision:
    route: str
    context_required: bool
    confidence: float
    reason: str
    blocking_unknowns: list[str] = field(default_factory=list)


@dataclass
class UncertaintyItem:
    kind: str
    name: str
    line: int | None = None
    retrieval_eligible: bool = False
    why_this_matters: str = ""
    raw_name: str = ""
    normalized_name: str = ""
    source_excerpt: str = ""


@dataclass
class EditConstraint:
    focus_point: str
    required_fact: str = ""
    must_do: str = ""
    must_not_do: str = ""
    supporting_point_kind: str = ""
    target_kind: str = ""
    target_hint: str = ""
    allowed_edit_radius: str = "local_block"
    must_preserve_signature: bool = True
    must_not_add_helper: bool = True
    must_not_expand_control_flow: bool = True
    must_not_rewrite_unrelated_lines: bool = True
    supporting_symbol: str = ""
    supporting_evidence: str = ""
    confidence: float = 0.45


@dataclass
class RetrievedMethodContext:
    method_name: str
    path: str
    class_name: str | None
    start_line: int | None
    end_line: int | None
    source: str
    found: bool
    signature: str = ""
    callsite_slice: str = ""
    evidence_slice: str = ""
    match_score: int = 0
    confidence: float = 0.0
    confidence_label: str = ""


@dataclass
class ReviewResult:
    round_id: int
    approved: bool
    issues: list[str]
    candidate_mode: str = "single"
    gate_decision: str = ""
    failure_modes: list[str] = field(default_factory=list)
    review_score: float = 0.0
    problem_alignment: float = 0.0
    minimality: float = 0.0
    semantic_preservation: float = 0.0
    internal_consistency: float = 0.0
    revision_advice: str = ""
    reject_type: str | None = None
    rationale: str = ""
    softened_gate_used: bool = False
    failed_checks: list[str] = field(default_factory=list)
    repair_constraints: list[str] = field(default_factory=list)
    failure_anchor: str = ""
    retry_hint: str = ""


@dataclass
class WorkflowTrace:
    task_id: str
    project: str
    file_path: str
    commit: str
    satd_comment: str
    original_code: str
    processed_manual_code: str
    status: str
    rounds_used: int
    github_context: dict[str, Any] | None
    repair_context_used: bool
    repair_feedback: dict[str, Any] | None
    review_strict_gate_result: str | None
    analysis: dict[str, Any] | None
    satd_route_type: str | None = None
    context_route: str | None = None
    context_required: bool | None = None
    context_confidence: float | None = None
    context_reason: str = ""
    context_blocking_unknowns: list[str] = field(default_factory=list)
    method_inquiry: dict[str, Any] | None = None
    uncertainty_items: list[dict[str, Any]] = field(default_factory=list)
    edit_constraints: list[dict[str, Any]] = field(default_factory=list)
    retrieved_method_contexts: list[dict[str, Any]] = field(default_factory=list)
    missing_method_names: list[str] = field(default_factory=list)
    repairs: list[dict[str, Any]] = field(default_factory=list)
    reviews: list[dict[str, Any]] = field(default_factory=list)
    processed_final_repaired_code: str | None = None
    em_label: str | None = None
    exact_match: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GraphState(TypedDict):
    task_id: str
    task_index: int
    task_total: int
    satd_comment: str
    original_code: str
    manual_code: str
    user: str
    project: str
    file_path: str
    commit: str
    status: str
    round_id: int
    max_rounds: int
    github_context: dict[str, Any] | None
    repair_context_used: bool
    repair_feedback: dict[str, Any] | None
    review_strict_gate_result: str | None
    analysis: AnalysisResult | None
    satd_route_type: str | None
    context_decision: ContextNeedDecision | None
    method_inquiry: MethodInquiryResult | None
    uncertainty_items: list[UncertaintyItem]
    edit_constraints: list[EditConstraint]
    retrieved_method_contexts: list[RetrievedMethodContext]
    missing_method_names: list[str]
    repairs: list[RepairAttempt]
    reviews: list[ReviewResult]
    latest_repair: RepairAttempt | None
    latest_review: ReviewResult | None
    final_repaired_code: str | None


class _DocstringStripper(ast.NodeTransformer):
    def _strip_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        if body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return body[1:]
        return body

    def visit_Module(self, node: ast.Module) -> ast.AST:
        node.body = self._strip_body(node.body)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = self._strip_body(node.body)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.body = self._strip_body(node.body)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.body = self._strip_body(node.body)
        self.generic_visit(node)
        return node


def record_to_graph_input(record: SATDRecord, max_rounds: int) -> GraphState:
    return GraphState(
        task_id=record.task_id,
        task_index=0,
        task_total=0,
        satd_comment=record.satd_comment,
        original_code=record.original_code,
        manual_code=record.manual_code,
        user=record.user,
        project=record.project,
        file_path=record.file_path,
        commit=record.commit,
        status="pending",
        round_id=0,
        max_rounds=max_rounds,
        github_context=None,
        repair_context_used=False,
        repair_feedback=None,
        review_strict_gate_result=None,
        analysis=None,
        satd_route_type=None,
        context_decision=None,
        method_inquiry=None,
        uncertainty_items=[],
        edit_constraints=[],
        retrieved_method_contexts=[],
        missing_method_names=[],
        repairs=[],
        reviews=[],
        latest_repair=None,
        latest_review=None,
        final_repaired_code=None,
    )


def _normalize_source(code: str | None) -> str:
    if not code:
        return ""
    normalized = code.replace("\r\n", "\n").replace("\r", "\n")
    return textwrap.dedent(normalized).strip()


def _strip_python_comments(code: str | None) -> str:
    source = _normalize_source(code)
    if not source:
        return ""

    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        kept_tokens = [token for token in tokens if token.type != tokenize.COMMENT]
        return tokenize.untokenize(kept_tokens)
    except (tokenize.TokenError, IndentationError):
        return source


def _strip_standalone_triple_quoted_blocks(code: str) -> str:
    pattern = re.compile(
        r"(?ms)^(?P<indent>[ \t]*)(?P<prefix>[rRuUbBfF]{0,2})?(?:'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\")[ \t]*\n?"
    )
    return re.sub(pattern, "", code)


def _canonicalize_python(code: str | None) -> str:
    uncommented = _strip_python_comments(code)
    if not uncommented.strip():
        return ""

    try:
        tree = ast.parse(uncommented)
        tree = _DocstringStripper().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree).strip()
    except SyntaxError:
        fallback = _strip_standalone_triple_quoted_blocks(uncommented)
        lines = [line.rstrip() for line in fallback.split("\n")]
        compact = "\n".join(line for line in lines if line.strip())
        return compact.strip()


def preprocess_python_code(code: str | None) -> str:
    return _canonicalize_python(code)


def trace_from_state(state: GraphState, em_label: str) -> WorkflowTrace:
    processed_manual_code = preprocess_python_code(state["manual_code"])
    processed_final_repaired_code = (
        preprocess_python_code(state["final_repaired_code"]) if state["final_repaired_code"] is not None else None
    )

    exact_match = None
    if processed_final_repaired_code is not None:
        exact_match = processed_final_repaired_code == processed_manual_code
    context_decision = state.get("context_decision")

    return WorkflowTrace(
        task_id=state["task_id"],
        project=state["project"],
        file_path=state["file_path"],
        commit=state["commit"],
        satd_comment=state["satd_comment"],
        original_code=state["original_code"],
        processed_manual_code=processed_manual_code,
        status=state["status"],
        rounds_used=state["round_id"],
        github_context=state["github_context"],
        repair_context_used=state["repair_context_used"],
        repair_feedback=state["repair_feedback"],
        review_strict_gate_result=state["review_strict_gate_result"],
        analysis=asdict(state["analysis"]) if state["analysis"] else None,
        satd_route_type=state.get("satd_route_type"),
        context_route=context_decision.route if context_decision else None,
        context_required=context_decision.context_required if context_decision else None,
        context_confidence=context_decision.confidence if context_decision else None,
        context_reason=context_decision.reason if context_decision else "",
        context_blocking_unknowns=list(context_decision.blocking_unknowns) if context_decision else [],
        method_inquiry=asdict(state["method_inquiry"]) if state.get("method_inquiry") else None,
        uncertainty_items=[asdict(item) for item in state.get("uncertainty_items", [])],
        edit_constraints=[asdict(item) for item in state.get("edit_constraints", [])],
        retrieved_method_contexts=[asdict(item) for item in state["retrieved_method_contexts"]],
        missing_method_names=list(state["missing_method_names"]),
        repairs=[asdict(item) for item in state["repairs"]],
        reviews=[asdict(item) for item in state["reviews"]],
        processed_final_repaired_code=processed_final_repaired_code,
        em_label=em_label,
        exact_match=exact_match,
    )

