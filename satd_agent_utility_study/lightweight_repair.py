from __future__ import annotations

import csv
import hashlib
import json
import queue
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from satd_langgraph.csv_loader import load_satd_csv
from satd_langgraph.openai_client import OpenAICompatClient
from satd_langgraph.schema import RepositoryEvidenceContext, SATDRecord, preprocess_python_code
from satd_langgraph.tools.repository_evidence_retriever import RepositoryEvidenceRetriever


SYSTEM_PROMPT = "You are an expert software engineer specialized in technical debt refactoring."

BASELINE_PROMPT = """
How to update the following code to resolve the SATD?

### Code:
{original_code}

### SATD comment:
{satd_comment}

### Consider the following questions in your answer:
Shortly explain how to resolve the SATD.
Provide the updated code."""

BASELINE_GUARDED_PROMPT = """
How to update the following code to resolve the SATD?

### Code:
{original_code}

### SATD comment:
{satd_comment}

### Consider the following questions in your answer:
Shortly explain how to resolve the SATD.
Provide the updated code.

Repair constraints:
- Return the complete updated version of the entire `### Code` block, not only changed lines.
- Make the smallest local edit that resolves the SATD.
- Do not add a nested helper function or new abstraction unless the original code already requires that shape.
- Preserve unrelated behavior and control flow.
- If the SATD asks to remove a temporary workaround, remove only that workaround and keep the surrounding logic intact."""

EVIDENCE_PROMPT = """
How to update the following code to resolve the SATD?

### Code:
{original_code}

### SATD comment:
{satd_comment}

### Repository-local evidence from the target commit:
{evidence_block}

### Consider the following questions in your answer:
Shortly explain how to resolve the SATD.
Provide the updated code."""

EVIDENCE_GUARDED_PROMPT = """
How to update the following code to resolve the SATD?

### Code:
{original_code}

### SATD comment:
{satd_comment}

### Repository-local evidence from the target commit:
{evidence_block}

### Consider the following questions in your answer:
Shortly explain how to resolve the SATD.
Provide the updated code.

Repair constraints:
- Return the complete updated version of the entire `### Code` block, not only changed lines.
- Make the smallest local edit that resolves the SATD.
- Use repository evidence only when it directly supports that local edit.
- Prefer direct sibling implementations, replacement APIs, and tests over broad definitions or generic examples.
- Do not add a nested helper function or new abstraction unless a direct sibling implementation shows that exact pattern.
- Preserve unrelated behavior and control flow.
- If the SATD asks to remove a temporary workaround, remove only that workaround and keep the surrounding logic intact."""

EVIDENCE_GUARDED_V2_PROMPT = """
How to update the following code to resolve the SATD?

### Code:
{original_code}

### SATD comment:
{satd_comment}

### Repository-local evidence from the target commit:
{evidence_block}

### Consider the following questions in your answer:
Shortly explain how to resolve the SATD.
Provide the updated code.

Repair constraints:
- Return the complete updated version of the entire `### Code` block, not only changed lines.
- Make the smallest local edit that resolves the SATD.
- Use repository evidence only when it directly supports that local edit.
- Prefer direct sibling implementations, replacement APIs, and tests over broad definitions or generic examples.
- Do not add a nested helper function or new abstraction unless a direct sibling implementation shows that exact pattern.
- Preserve unrelated behavior, control flow, names, string literals, and log messages exactly.
- If the SATD asks to remove a workaround, legacy compatibility path, temporary configuration, deprecated argument, or default override, prefer deleting that specific argument/config/block rather than replacing it with an explicit default value.
- If a parameter's documented or project-local default already matches the desired behavior, remove the redundant argument instead of setting it explicitly."""

GUIDED_PROMPT = """
How to update the following code to resolve the SATD?

### Code:
{original_code}

### SATD comment:
{satd_comment}

### Repository-local evidence from the target commit:
{evidence_block}

### Evidence-backed repair guidance:
{guidance}

### Consider the following questions in your answer:
Shortly explain how to resolve the SATD.
Provide the updated code.

Use the repository evidence only when it directly supports a concrete edit. Prefer sibling implementations, replacement APIs, tests, and project usage examples over generic matches."""

GUIDED_COMPLETE_PROMPT = """
How to update the following code to resolve the SATD?

### Code:
{original_code}

### SATD comment:
{satd_comment}

### Repository-local evidence from the target commit:
{evidence_block}

### Evidence-backed repair guidance:
{guidance}

### Consider the following questions in your answer:
Shortly explain how to resolve the SATD.
Provide the updated code.

Output requirements:
- The updated code must be the complete updated version of the entire code block shown in `### Code`, not only the changed lines.
- Preserve all unrelated code, signatures, decorators, imports, docstrings, comments, formatting structure, and control flow unless the SATD directly requires changing them.
- Use the repository evidence only when it directly supports a concrete edit.
- Prefer sibling implementations, replacement APIs, tests, and project usage examples over generic matches.
- Do not delete a surrounding block merely because a TODO says something should be removed; first infer the smallest edit that resolves the SATD."""


@dataclass(frozen=True)
class LightweightRepairRow:
    task_id: str
    project: str
    file_path: str
    commit: str
    variant: str
    exact_match: bool
    evidence_count: int
    evidence_types: str
    guidance: str
    repaired_code: str
    processed_repaired_code: str
    prompt_hash: str
    error: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


class LightweightRepairExperiment:
    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        max_evidence: int = 5,
        candidate_pool: int = 10,
        verbose: bool = False,
    ) -> None:
        self.client = OpenAICompatClient(model=model, verbose=verbose)
        self.retriever = RepositoryEvidenceRetriever()
        self.max_evidence = max(0, int(max_evidence))
        self.candidate_pool = max(self.max_evidence, int(candidate_pool))
        self.verbose = bool(verbose)

    def run_csv(
        self,
        input_csv: Path,
        output_csv: Path,
        *,
        variant: str,
        resume: bool = False,
        flush_every: int = 1,
    ) -> list[LightweightRepairRow]:
        variant = _normalize_variant(variant)
        records = load_satd_csv(input_csv)
        existing = _read_rows(output_csv) if resume and output_csv.exists() else []
        completed = {row.get("task_id", "") for row in existing if row.get("variant", variant) == variant}
        new_rows: list[LightweightRepairRow] = []
        pending: list[LightweightRepairRow] = []
        if self.verbose:
            print(f"[light] variant={variant} resume_completed={len(completed)} total={len(records)} output={output_csv}", flush=True)
        for index, record in enumerate(records, start=1):
            if record.task_id in completed:
                continue
            if self.verbose:
                print(f"[light] {index}/{len(records)} task_id={record.task_id} start project={record.project}", flush=True)
            row = self.run_record(record, variant=variant)
            new_rows.append(row)
            pending.append(row)
            if self.verbose:
                print(
                    f"[light] {index}/{len(records)} task_id={record.task_id} "
                    f"exact={row.exact_match} evidence={row.evidence_count} error={bool(row.error)}",
                    flush=True,
                )
            if len(pending) >= max(1, flush_every):
                _write_rows(output_csv, existing, new_rows)
                pending = []
        if pending or not output_csv.exists():
            _write_rows(output_csv, existing, new_rows)
        return [_row_from_existing(row) for row in existing] + new_rows

    def run_record(self, record: SATDRecord, *, variant: str) -> LightweightRepairRow:
        evidence: list[RepositoryEvidenceContext] = []
        guidance = ""
        try:
            if variant in {"evidence", "evidence_guarded", "evidence_guarded_v2", "guided", "guided_complete"}:
                evidence = self._retrieve_evidence(record, variant=variant)
                if variant in {"guided", "guided_complete"}:
                    guidance = self._build_guidance(record, evidence)
            user_prompt = self._build_user_prompt(record, variant=variant, evidence=evidence, guidance=guidance)
            prompt_hash = hashlib.sha1(f"{SYSTEM_PROMPT}\n---\n{user_prompt}".encode("utf-8", errors="replace")).hexdigest()
            raw = self._generate_plain_text(user_prompt, request_label=f"lightweight_repair:{variant}:task_{record.task_id}")
            repaired = _extract_code(raw)
            processed = preprocess_python_code(repaired)
            exact = processed == preprocess_python_code(record.manual_code)
            error = ""
        except Exception as exc:
            repaired = ""
            processed = ""
            exact = False
            prompt_hash = ""
            error = f"{type(exc).__name__}: {exc}"
        return LightweightRepairRow(
            task_id=record.task_id,
            project=record.project,
            file_path=record.file_path,
            commit=record.commit,
            variant=variant,
            exact_match=exact,
            evidence_count=len(evidence),
            evidence_types=" | ".join(f"{item.evidence_type}/{item.evidence_subtype}/{item.support_level}" for item in evidence),
            guidance=guidance,
            repaired_code=repaired,
            processed_repaired_code=processed,
            prompt_hash=prompt_hash,
            error=error,
        )

    def _retrieve_evidence(self, record: SATDRecord, *, variant: str) -> list[RepositoryEvidenceContext]:
        items = self.retriever.retrieve(
            owner=record.user,
            repo=record.project,
            current_path=record.file_path,
            satd_comment=record.satd_comment,
            original_code=record.original_code,
            ref=record.commit,
            max_items=self.candidate_pool if variant == "guided" else self.max_evidence,
        )
        if variant == "evidence":
            return items[: self.max_evidence]
        return self._rerank_evidence(record, items)[: self.max_evidence]

    def _rerank_evidence(
        self,
        record: SATDRecord,
        items: list[RepositoryEvidenceContext],
    ) -> list[RepositoryEvidenceContext]:
        if len(items) <= self.max_evidence:
            return items
        snippets = []
        for index, item in enumerate(items, start=1):
            snippets.append(
                f"[{index}] {item.evidence_type}/{item.evidence_subtype}/{item.support_level} "
                f"{item.source_path}:{item.span}\n{str(item.content or '').strip()[:900]}"
            )
        payload = self.client.generate_json(
            "You select repository evidence for a SATD repair prompt. Return JSON only.",
            "Select the snippets most likely to directly determine the code edit. Prefer sibling implementations, tests, "
            "replacement APIs, definitions, and concrete project usage. Avoid generic or duplicate snippets.\n\n"
            f"SATD:\n{record.satd_comment}\n\nCode:\n```python\n{record.original_code}\n```\n\n"
            f"Candidates:\n{chr(10).join(snippets)}\n\n"
            f"Return JSON with selected_indices: a list of up to {self.max_evidence} 1-based indices.",
            temperature=0.0,
            request_label=f"lightweight_rerank:task_{record.task_id}",
            max_tokens=250,
        )
        selected: list[RepositoryEvidenceContext] = []
        seen: set[int] = set()
        for value in payload.get("selected_indices") or []:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if 1 <= index <= len(items) and index not in seen:
                selected.append(items[index - 1])
                seen.add(index)
            if len(selected) >= self.max_evidence:
                break
        return selected or items[: self.max_evidence]

    def _build_guidance(self, record: SATDRecord, evidence: list[RepositoryEvidenceContext]) -> str:
        if not evidence:
            return "[no repository-local evidence retrieved]"
        snippets = []
        for index, item in enumerate(evidence, start=1):
            snippets.append(
                f"[{index}] {item.evidence_type}/{item.evidence_subtype}/{item.support_level} "
                f"{item.source_path}:{item.span}\n{str(item.content or '').strip()[:1200]}"
            )
        payload = self.client.generate_json(
            "You convert repository evidence into concise SATD repair guidance. Return JSON only.",
            "Use only the evidence shown. Extract concrete edit instructions that the repair model should follow. "
            "Do not write code.\n\n"
            f"SATD:\n{record.satd_comment}\n\nCode:\n```python\n{record.original_code}\n```\n\n"
            f"Evidence:\n{chr(10).join(snippets)}\n\n"
            'Return JSON with keys "summary", "edit_steps", "must_use", "avoid".',
            temperature=0.0,
            request_label=f"lightweight_guidance:task_{record.task_id}",
            max_tokens=500,
        )
        lines = []
        summary = " ".join(str(payload.get("summary") or "").split())
        if summary:
            lines.append(f"- summary: {summary}")
        for key, label in (("edit_steps", "edit steps"), ("must_use", "must use"), ("avoid", "avoid")):
            values = payload.get(key)
            if isinstance(values, list):
                cleaned = [" ".join(str(item).split()) for item in values if " ".join(str(item).split())]
                if cleaned:
                    lines.append(f"- {label}: " + " | ".join(cleaned[:5]))
        return "\n".join(lines) or "[no concrete guidance extracted]"

    def _build_user_prompt(
        self,
        record: SATDRecord,
        *,
        variant: str,
        evidence: list[RepositoryEvidenceContext],
        guidance: str,
    ) -> str:
        if variant == "baseline":
            return BASELINE_PROMPT.format(original_code=record.original_code, satd_comment=record.satd_comment)
        if variant == "baseline_guarded":
            return BASELINE_GUARDED_PROMPT.format(original_code=record.original_code, satd_comment=record.satd_comment)
        evidence_block = _format_evidence(evidence)
        if variant == "evidence":
            return EVIDENCE_PROMPT.format(
                original_code=record.original_code,
                satd_comment=record.satd_comment,
                evidence_block=evidence_block,
            )
        if variant in {"evidence_guarded", "evidence_guarded_v2"}:
            prompt_template = EVIDENCE_GUARDED_V2_PROMPT if variant == "evidence_guarded_v2" else EVIDENCE_GUARDED_PROMPT
            return prompt_template.format(
                original_code=record.original_code,
                satd_comment=record.satd_comment,
                evidence_block=evidence_block,
            )
        prompt_template = GUIDED_COMPLETE_PROMPT if variant == "guided_complete" else GUIDED_PROMPT
        return prompt_template.format(
            original_code=record.original_code,
            satd_comment=record.satd_comment,
            evidence_block=evidence_block,
            guidance=guidance,
        )

    def _generate_plain_text(self, user_prompt: str, request_label: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.client.max_attempts):
            started = time.time()
            self.client._emit_log(
                f"[llm] start label={request_label} attempt={attempt + 1}/{self.client.max_attempts} "
                f"model={self.client.model} timeout={self.client.request_timeout:.0f}s"
            )
            try:
                result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

                def invoke() -> None:
                    try:
                        result_queue.put(
                            (
                                "ok",
                                self.client.client.chat.completions.create(
                                    model=self.client.model,
                                    messages=[
                                        {"role": "system", "content": SYSTEM_PROMPT},
                                        {"role": "user", "content": user_prompt},
                                    ],
                                    temperature=0.0,
                                ),
                            )
                        )
                    except Exception as exc:
                        result_queue.put(("error", exc))

                worker = threading.Thread(target=invoke, daemon=True)
                worker.start()
                status, payload = result_queue.get(timeout=self.client.request_timeout + 5)
                if status == "error":
                    raise payload
                elapsed = time.time() - started
                self.client._emit_log(f"[llm] success label={request_label} elapsed={elapsed:.2f}s")
                return payload.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                self.client._emit_log(
                    f"[llm] error label={request_label} attempt={attempt + 1}/{self.client.max_attempts} "
                    f"type={type(exc).__name__} message={self.client._short_error(exc)}"
                )
                if attempt < self.client.max_attempts - 1:
                    time.sleep(2 * (attempt + 1))
        if last_error:
            raise last_error
        return ""


def _normalize_variant(value: str) -> str:
    variant = str(value or "").strip().lower()
    if variant not in {
        "baseline",
        "baseline_guarded",
        "evidence",
        "evidence_guarded",
        "evidence_guarded_v2",
        "guided",
        "guided_complete",
    }:
        raise ValueError(
            "variant must be one of: baseline, baseline_guarded, evidence, evidence_guarded, "
            "evidence_guarded_v2, guided, guided_complete"
        )
    return variant


def _format_evidence(items: list[RepositoryEvidenceContext]) -> str:
    if not items:
        return "[no repository-local evidence retrieved]"
    chunks = []
    for index, item in enumerate(items, start=1):
        chunks.append(
            f"[{index}] {item.evidence_type}/{item.evidence_subtype}/{item.support_level} "
            f"score={item.score:.2f}\n"
            f"location: {item.source_path}:{item.span}\n"
            f"why: {item.why_relevant}\n"
            f"```python\n{str(item.content or '').strip()[:2400]}\n```"
        )
    return "\n\n".join(chunks)


def _extract_code(raw_response: str) -> str:
    python_fence = re.search(r"```python\s*(.*?)```", raw_response or "", flags=re.IGNORECASE | re.DOTALL)
    if python_fence:
        return python_fence.group(1).strip()
    any_fence = re.search(r"```\s*(.*?)```", raw_response or "", flags=re.DOTALL)
    if any_fence:
        return any_fence.group(1).strip()
    lines = (raw_response or "").splitlines()
    for index, line in enumerate(lines):
        if line.lstrip().startswith(("def ", "async def ", "class ")):
            return "\n".join(lines[index:]).strip()
    return (raw_response or "").strip()


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, existing: list[dict[str, str]], rows: list[LightweightRepairRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(LightweightRepairRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in existing:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
        for row in rows:
            writer.writerow(row.to_row())


def _row_from_existing(row: dict[str, str]) -> LightweightRepairRow:
    return LightweightRepairRow(
        task_id=row.get("task_id", ""),
        project=row.get("project", ""),
        file_path=row.get("file_path", ""),
        commit=row.get("commit", ""),
        variant=row.get("variant", ""),
        exact_match=str(row.get("exact_match", "")).lower() == "true",
        evidence_count=int(float(row.get("evidence_count") or 0)),
        evidence_types=row.get("evidence_types", ""),
        guidance=row.get("guidance", ""),
        repaired_code=row.get("repaired_code", ""),
        processed_repaired_code=row.get("processed_repaired_code", ""),
        prompt_hash=row.get("prompt_hash", ""),
        error=row.get("error", ""),
    )


def summarize_rows(rows: list[LightweightRepairRow]) -> dict[str, Any]:
    total = len(rows)
    exact = sum(1 for row in rows if row.exact_match)
    errors = sum(1 for row in rows if row.error)
    with_evidence = sum(1 for row in rows if row.evidence_count > 0)
    return {
        "sample_count": total,
        "exact_match_count": exact,
        "exact_match_rate": round(exact / total, 4) if total else 0.0,
        "error_count": errors,
        "with_evidence_count": with_evidence,
    }
