from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class VariantConfig:
    name: str
    description: str
    repository_evidence_mode: str
    max_repository_evidence: int
    repository_evidence_prompt_mode: str
    repository_evidence_guidance_mode: str = "none"
    repository_evidence_rerank_mode: str = "none"
    repair_context_mode: str = "clone_treesitter"
    max_method_contexts: int = 2

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitAssignment:
    task_id: str
    split: str
    em_label: str
    project: str
    intent_bucket: str
    evidence_bucket: str
    edit_shape_bucket: str
    stratum: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)
