from __future__ import annotations

from .models import VariantConfig


DEFAULT_VARIANTS = [
    VariantConfig(
        name="current_agent",
        description="Existing LangGraph agent with method context only.",
        repository_evidence_mode="none",
        max_repository_evidence=0,
        repository_evidence_prompt_mode="append",
    ),
    VariantConfig(
        name="fixed_top3",
        description="Always inject the top three repository evidence snippets.",
        repository_evidence_mode="fixed",
        max_repository_evidence=3,
        repository_evidence_prompt_mode="append",
    ),
    VariantConfig(
        name="adaptive_top3",
        description="Inject repository evidence only when the context-required path still needs non-local evidence.",
        repository_evidence_mode="adaptive",
        max_repository_evidence=3,
        repository_evidence_prompt_mode="append",
    ),
    VariantConfig(
        name="two_stage_top5",
        description="Use a small first pass and expand only when the first pass is weak or uncertainty remains.",
        repository_evidence_mode="two_stage",
        max_repository_evidence=5,
        repository_evidence_prompt_mode="append",
    ),
    VariantConfig(
        name="two_stage_constrained_top5",
        description="Two-stage retrieval plus a fixer prompt that constrains repairs to grounded evidence.",
        repository_evidence_mode="two_stage",
        max_repository_evidence=5,
        repository_evidence_prompt_mode="constrained",
    ),
    VariantConfig(
        name="two_stage_guided_top5",
        description="Two-stage retrieval plus evidence-backed repair guidance synthesized before fixing.",
        repository_evidence_mode="two_stage",
        max_repository_evidence=5,
        repository_evidence_prompt_mode="constrained",
        repository_evidence_guidance_mode="summarize",
    ),
    VariantConfig(
        name="two_stage_guided_reranked_top5",
        description="Two-stage retrieval plus LLM reranking and evidence-backed repair guidance.",
        repository_evidence_mode="two_stage",
        max_repository_evidence=5,
        repository_evidence_prompt_mode="constrained",
        repository_evidence_guidance_mode="summarize",
        repository_evidence_rerank_mode="llm",
    ),
]


def default_variant_map() -> dict[str, VariantConfig]:
    return {item.name: item for item in DEFAULT_VARIANTS}
