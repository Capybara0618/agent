# SATD Agent Utility Study

This directory contains the agent-facing empirical study. Its question is not whether repository evidence can explain the human repair, but whether a deployable agent policy can raise exact match.

## Relationship to the oracle study

- `satd_empirical_study/` remains an offline oracle helper that can suggest candidate evidence using `manual_code`.
- `satd_agent_utility_study/` is the deployment-facing study. Formal agent runs never expose `manual_code` to the repair workflow.

## Default variants

1. `current_agent`: existing LangGraph workflow with method context only
2. `fixed_top3`: always append three repository evidence snippets
3. `adaptive_top3`: append evidence only when the context-required path still needs non-local evidence
4. `two_stage_top5`: start small and expand retrieval when uncertainty remains
5. `two_stage_constrained_top5`: two-stage retrieval plus evidence-grounded repair rules
6. `two_stage_guided_top5`: two-stage retrieval plus synthesized evidence-backed repair guidance
7. `two_stage_guided_reranked_top5`: two-stage retrieval plus LLM reranking before guided repair

## Typical workflow

Prepare deterministic development and holdout splits:

```powershell
python -m satd_agent_utility_study.run_agent_utility_study `
  --stage prepare `
  --input code.csv `
  --output-dir satd_agent_utility_study_outputs `
  --oracle-inventory satd_oracle_study_v2_outputs/oracle_evidence_inventory.csv
```

Run all development variants:

```powershell
python -m satd_agent_utility_study.run_agent_utility_study `
  --stage dev `
  --output-dir satd_agent_utility_study_outputs `
  --verbose `
  --resume
```

Run the best two development variants on holdout:

```powershell
python -m satd_agent_utility_study.run_agent_utility_study `
  --stage holdout `
  --output-dir satd_agent_utility_study_outputs `
  --select-top-k 2 `
  --verbose `
  --resume
```

Run the best holdout variant on all 1000 samples:

```powershell
python -m satd_agent_utility_study.run_agent_utility_study `
  --stage full `
  --output-dir satd_agent_utility_study_outputs `
  --select-top-k 1 `
  --verbose `
  --resume
```

## Outputs

- `split_assignments.csv`
- `splits/dev.csv`
- `splits/holdout.csv`
- `variant_matrix.json`
- `agent_variant_results.csv`
- `context_utility_results.csv`
- `dev_holdout_summary.csv`
- `per_sample_decision_trace.csv`
- `failure_after_context.csv`
- `agent_strategy_recommendation.md`
- optional `rescuability_audit.csv` from the lightweight agent-facing audit

Each variant run keeps its original LangGraph outputs under `runs/<phase>/<variant>/`, so interrupted runs can resume from the existing checkpoints.

## Lightweight rescuability audit

Use this before spending heavily on full agent variants when you want to know whether the current retriever even covers the patch-driving symbols exposed by the manual repair:

```powershell
python -m satd_agent_utility_study.run_rescuability_audit `
  --input satd_agent_utility_study_outputs/splits/dev.csv `
  --output satd_agent_utility_study_outputs/rescuability_audit.csv
```

This audit uses `manual_code` only offline. It is a deployment-oriented proxy, not a formal causal result.

## Lightweight repair utility result

The fastest validated path so far is independent of the LangGraph agent:

```powershell
python -m satd_agent_utility_study.run_lightweight_repair `
  --input code.csv `
  --output satd_agent_utility_study_outputs/lightweight/full1000_baseline_guarded.csv `
  --variant baseline_guarded `
  --verbose `
  --resume

python -m satd_agent_utility_study.run_lightweight_repair `
  --input code.csv `
  --output satd_agent_utility_study_outputs/lightweight/full1000_evidence_guarded.csv `
  --variant evidence_guarded `
  --verbose `
  --resume

python -m satd_agent_utility_study.apply_context_policy `
  --input code.csv `
  --baseline satd_agent_utility_study_outputs/lightweight/full1000_baseline_guarded.csv `
  --evidence satd_agent_utility_study_outputs/lightweight/full1000_evidence_guarded.csv `
  --output satd_agent_utility_study_outputs/lightweight/full1000_context_policy_cleanup_v2.csv `
  --policy cleanup_uncertainty_gate
```

Current full-data result:

- Historical one-shot baseline in `code.csv`: `97/1000`
- Previous heavy agent result reported by the experiment: `104/1000`
- `baseline_guarded`: `122/1000`
- `evidence_guarded`: `121/1000`
- `cleanup_uncertainty_gate`: `135/1000`

The useful policy is not “always add repository evidence.” The observed rule is:

- Use `evidence_guarded` by default: complete-code output, smallest local edit, and repository evidence restricted to directly supporting the edit.
- Fall back to `baseline_guarded` for cleanup/uncertainty SATD comments matching remove/default/legacy/workaround/drop/eliminate/opposite/string-copy/temporary-hack/LIE/busted/once-it-exists/maybe-a-different patterns.

Interpretation for the fixer:

- Valuable context: direct sibling implementations, concrete tests, replacement APIs, project-local usage examples.
- Risky context: broad definitions or generic examples for vague cleanup comments, because they often induce semantically plausible but strict-EM-different edits.
- Prompt constraints matter as much as retrieval: forcing complete output and smallest local edits raised the no-evidence baseline to `122/1000`.
