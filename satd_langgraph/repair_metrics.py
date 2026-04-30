from __future__ import annotations

import difflib
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .schema import preprocess_python_code


@dataclass(frozen=True)
class RepairMetricContext:
    ignored_ngrams: set[tuple[str, ...]]


@dataclass(frozen=True)
class RepairMetricResult:
    bleu_diff: float
    crystalbleu_diff: float
    lemod_line_p: float
    lemod_line_r: float
    lemod_line_f: float
    reference_diff_line_count: int
    candidate_diff_line_count: int
    overlap_diff_line_count: int


ZERO_REPAIR_METRICS = RepairMetricResult(
    bleu_diff=0.0,
    crystalbleu_diff=0.0,
    lemod_line_p=0.0,
    lemod_line_r=0.0,
    lemod_line_f=0.0,
    reference_diff_line_count=0,
    candidate_diff_line_count=0,
    overlap_diff_line_count=0,
)


def build_metric_context(manual_codes: Iterable[str | None], crystal_k: int = 500) -> RepairMetricContext:
    return RepairMetricContext(ignored_ngrams=_build_ignored_ngrams(manual_codes, crystal_k))


def calculate_repair_metrics(
    original_code: str | None,
    manual_code: str | None,
    repaired_code: str | None,
    context: RepairMetricContext | None = None,
) -> RepairMetricResult:
    if repaired_code is None or not str(repaired_code).strip():
        return ZERO_REPAIR_METRICS

    original = preprocess_python_code(original_code)
    manual = preprocess_python_code(manual_code)
    repaired = preprocess_python_code(repaired_code)
    if not manual or not repaired:
        return ZERO_REPAIR_METRICS

    reference_diff = "\n".join(_diff_lines(original, manual))
    candidate_diff = "\n".join(_diff_lines(original, repaired))
    line_p, line_r, line_f, ref_count, cand_count, overlap_count = _lemod(reference_diff, candidate_diff)
    ignored = context.ignored_ngrams if context is not None else set()
    return RepairMetricResult(
        bleu_diff=_bleu_score_method4(reference_diff, candidate_diff),
        crystalbleu_diff=_crystal_bleu_score(reference_diff, candidate_diff, ignored),
        lemod_line_p=line_p,
        lemod_line_r=line_r,
        lemod_line_f=line_f,
        reference_diff_line_count=ref_count,
        candidate_diff_line_count=cand_count,
        overlap_diff_line_count=overlap_count,
    )


def average_metric_rows(rows: Sequence[dict], accepted_only: bool = False) -> dict[str, float]:
    selected = []
    for row in rows:
        if accepted_only and row.get("status") != "accepted":
            continue
        selected.append(row)
    return {
        "avg_bleu_diff": _avg(_row_float(row, "BLEU_diff") for row in selected),
        "avg_crystalbleu_diff": _avg(_row_float(row, "CrystalBLEU_diff") for row in selected),
        "avg_lemod": _avg(_row_float(row, "LEMOD") or _row_float(row, "LEMOD_LineF") for row in selected),
    }


def metric_result_to_row(metrics: RepairMetricResult) -> dict[str, float | int]:
    return {
        "BLEU_diff": metrics.bleu_diff,
        "CrystalBLEU_diff": metrics.crystalbleu_diff,
        "LEMOD": metrics.lemod_line_f,
    }


def _build_ignored_ngrams(codes: Iterable[str | None], crystal_k: int) -> set[tuple[str, ...]]:
    if crystal_k <= 0:
        return set()
    counts: Counter[tuple[str, ...]] = Counter()
    for code in codes:
        tokens = _tokens(preprocess_python_code(code))
        for n in range(1, 5):
            counts.update(_ngrams(tokens, n))
    return {ngram for ngram, _ in counts.most_common(crystal_k)}


def _diff_lines(old_code: str, new_code: str) -> list[str]:
    diff = difflib.ndiff(old_code.splitlines(), new_code.splitlines())
    return [line[2:] for line in diff if line.startswith(("+ ", "- "))]


def _lemod(reference_diff: str, candidate_diff: str) -> tuple[float, float, float, int, int, int]:
    ref_lines = {_normalize_line(line) for line in reference_diff.splitlines() if line.strip()}
    cand_lines = {_normalize_line(line) for line in candidate_diff.splitlines() if line.strip()}
    overlap = ref_lines & cand_lines
    if not ref_lines or not cand_lines:
        return 0.0, 0.0, 0.0, len(ref_lines), len(cand_lines), len(overlap)
    precision = len(overlap) / len(cand_lines)
    recall = len(overlap) / len(ref_lines)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1, len(ref_lines), len(cand_lines), len(overlap)


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", "", line.strip())


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text or "", re.UNICODE)


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _modified_precision(
    reference_tokens: Sequence[str],
    candidate_tokens: Sequence[str],
    n: int,
    ignored: set[tuple[str, ...]],
) -> tuple[int, int]:
    ref_counts = Counter(ngram for ngram in _ngrams(reference_tokens, n) if ngram not in ignored)
    cand_counts = Counter(ngram for ngram in _ngrams(candidate_tokens, n) if ngram not in ignored)
    numerator = sum(min(count, ref_counts[ngram]) for ngram, count in cand_counts.items())
    denominator = sum(cand_counts.values())
    return numerator, denominator


def _bleu_score_method4(reference: str, candidate: str, max_order: int = 4) -> float:
    reference_tokens = _tokens(reference)
    candidate_tokens = _tokens(candidate)
    if not candidate_tokens:
        return 0.0
    brevity_penalty = _brevity_penalty(len(reference_tokens), len(candidate_tokens))
    numerators = []
    denominators = []
    for n in range(1, max_order + 1):
        numerator, denominator = _modified_precision(reference_tokens, candidate_tokens, n, set())
        numerators.append(numerator)
        denominators.append(max(1, denominator))
    if numerators[0] == 0:
        return 0.0

    smooth_count = 1
    log_precision_sum = 0.0
    for numerator, denominator in zip(numerators, denominators):
        if numerator == 0:
            if len(candidate_tokens) > 1:
                smoothed_numerator = 1 / (2**smooth_count * 5 / math.log(len(candidate_tokens)))
                precision = smoothed_numerator / denominator
                smooth_count += 1
            else:
                precision = sys.float_info.min
        else:
            precision = numerator / denominator
        log_precision_sum += math.log(precision) / max_order
    return brevity_penalty * math.exp(log_precision_sum)


def _crystal_bleu_score(reference: str, candidate: str, ignored: set[tuple[str, ...]], max_order: int = 4) -> float:
    # Match the replication notebook's CrystalBLEU call: it passes each diff string
    # as a sequence, so the scorer operates over character n-grams.
    reference_tokens = list(reference)
    candidate_tokens = list(candidate)
    if not candidate_tokens:
        return 0.0
    brevity_penalty = _brevity_penalty(len(reference_tokens), len(candidate_tokens))
    numerators = []
    denominators = []
    for n in range(1, max_order + 1):
        numerator, denominator = _modified_precision(reference_tokens, candidate_tokens, n, ignored)
        numerators.append(numerator)
        denominators.append(max(1, denominator))
    if numerators[0] == 0:
        return 0.0
    log_precision_sum = 0.0
    for numerator, denominator in zip(numerators, denominators):
        precision = numerator / denominator if numerator else sys.float_info.min
        log_precision_sum += math.log(precision) / max_order
    return brevity_penalty * math.exp(log_precision_sum)


def _brevity_penalty(reference_length: int, candidate_length: int) -> float:
    if candidate_length == 0:
        return 0.0
    if candidate_length > reference_length:
        return 1.0
    return math.exp(1.0 - reference_length / candidate_length)


def _row_float(row: dict, field: str) -> float:
    try:
        return float(row.get(field) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _avg(values: Iterable[float]) -> float:
    items = list(values)
    return round(sum(items) / len(items), 6) if items else 0.0
