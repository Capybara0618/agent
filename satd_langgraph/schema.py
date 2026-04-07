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
    repairability_score: float
    intent_clarity: float
    change_locality: float
    semantic_risk: float
    context_sufficiency: float
    verifiability: float
    analyze_score: float
    confidence: float
    satd_type: str
    reason: str
    evidence_summary: str
    risk_level: str
    context_score: float
    clarity_score: float
    scope_radius: str
    validation_signals: list[str] = field(default_factory=list)
    context_gaps: list[str] = field(default_factory=list)
    followup_context_requests: list[str] = field(default_factory=list)
    repair_strategy: str = ""
    drop_reason: str | None = None
    historical_snapshot_mismatch: bool = False
    github_evidence_strength: str = "low"


@dataclass
class RepairAttempt:
    round_id: int
    repair_plan: str
    repaired_code: str
    changed_scope: str
    confidence: float
    notes: str


@dataclass
class ReviewResult:
    round_id: int
    approved: bool
    review_score: float
    problem_alignment: float
    minimality: float
    semantic_preservation: float
    internal_consistency: float
    issues: list[str]
    revision_advice: str
    reject_type: str | None
    rationale: str
    softened_gate_used: bool = False


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
    repairs: list[dict[str, Any]] = field(default_factory=list)
    reviews: list[dict[str, Any]] = field(default_factory=list)
    processed_final_repaired_code: str | None = None
    em_label: str | None = None
    exact_match: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GraphState(TypedDict):
    task_id: str
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
        repairs=[asdict(item) for item in state["repairs"]],
        reviews=[asdict(item) for item in state["reviews"]],
        processed_final_repaired_code=processed_final_repaired_code,
        em_label=em_label,
        exact_match=exact_match,
    )

