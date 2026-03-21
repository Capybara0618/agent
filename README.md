# SATD Multi-Agent Workflow

This workspace contains a LangGraph-based SATD workflow for `code.csv`.

The workflow now uses a layered GitHub context cache and three specialized LLM agents:

- analyzer node: reads only the shared `base_context` and filters out obviously poor repair candidates
- fixer node: enriches `repair_context` and generates repaired code with richer project evidence
- reviewer node: reads `base_context + repair_context + review_context` and acts as a strict final gate
- loop limit: 2 repair-review rounds by default
- evaluation: compares the final repaired code with `manual_code` only after the workflow finishes
- all three agents use the same OpenAI-compatible `gpt-4o-mini` interface

The graph never sees the ground-truth answer while making decisions.
The final comparison is done offline after preprocessing both sides.

## Layered Context Cache

Each SATD now gets a persistent per-task context bundle under `context_cache/`.

Layers:

- `base_context`
  - file path
  - original code
  - SATD comment
  - target file summary
  - SATD window
  - enclosing symbol
  - imports
  - issue refs
  - module docs
- `repair_context`
  - related tests
  - call sites
  - repo tree
  - neighbor files
  - commits for path
  - similar history
  - issue comments
  - related PR files
- `review_context`
  - validation signal summary
  - risk indicators
  - change-scope evidence

Workflow behavior:

- every SATD builds or loads `base_context` before analyze
- repair adds `repair_context` only when the task passes analyzer
- review derives `review_context` without re-fetching large GitHub payloads
- all layers are saved to disk and reused on reruns

## Analyzer Design

The analyzer is now intentionally narrower:

- primary evidence: `SATD comment + original_code + file_path`
- auxiliary evidence: current GitHub repository state
- GitHub mismatch is treated as `historical_snapshot_mismatch`, not automatic failure
- clear, local, non-high-risk SATD items are biased toward `repairable`

Analyzer output includes:

- `decision`
- `repairability_score`
- `scope_radius`
- `validation_signals`
- `context_gaps`
- `evidence_summary`
- `drop_reason`
- `historical_snapshot_mismatch`
- `github_evidence_strength`

## Preprocessed Evaluation

Final exact match is computed on preprocessed Python code:

- remove `#` comments
- remove standalone triple-quoted docstring/comment blocks
- normalize formatting through Python parsing/unparsing when possible
- compare the processed final repaired code with the processed `manual_code`

Display rules are now:

- `original_code` stays raw, with comments preserved
- `manual_code` is shown only as processed code
- model repaired code is shown only as processed code

## Project Layout

- `satd_langgraph/schema.py`: dataset records, preprocessing, graph state, and output models
- `satd_langgraph/csv_loader.py`: `code.csv` reader and normalizer
- `satd_langgraph/github_tools.py`: GitHub retrieval and local context extraction tools
- `satd_langgraph/agents.py`: analyzer, fixer, reviewer, and layered context manager
- `satd_langgraph/workflow.py`: LangGraph state graph, cache persistence, and CSV writers
- `run_langgraph_workflow.py`: command-line entry point for the CSV experiment
- `code.csv`: your SATD dataset
- `.vendor/`: local dependencies installed into the workspace

## Quick Start

Fill `satd_langgraph/local_settings.py` or set environment variables first:

```powershell
$env:OPENAI_API_KEY="your-proxy-key"
$env:OPENAI_BASE_URL="https://your-proxy-host/v1"
$env:GITHUB_TOKEN="your-github-token"
```

Then run:

```powershell
python run_langgraph_workflow.py --input code.csv --output-dir outputs_langgraph --model gpt-4o-mini --verbose
```

If you want a small smoke test first:

```powershell
python run_langgraph_workflow.py --input code.csv --output-dir outputs_langgraph_smoke --limit 20 --model gpt-4o-mini --verbose --write-batch-size 10
```

## Dataset Assumptions

The current CSV loader expects these columns:

- `index`
- `SATD_comment`
- `original_code`
- `manual_code`
- `user`
- `project`
- `file_path`
- `EM`

Unnamed empty columns in `code.csv` are ignored.

## Main Outputs

The main file to inspect is:

- `trajectory_overview.csv`

It gives one row per SATD and includes:

- analyzer decision, score, scope, validation signals, context gaps, and drop reason
- whether repair context was actually used
- the strict review gate result
- processed repaired code for round 1 and round 2
- review result for each round
- where the task was dropped
- whether the workflow finally output the task
- raw original code
- processed final repaired code
- processed manual code
- final `exact_match` result after preprocessing

Other files:

- `summary.csv`: dataset-level metrics
- `results.csv`: task-level compact result table
- `repairs.csv`: per-repair detail table
- `reviews.csv`: per-review detail table
- `github_context.csv`: flattened context content used during execution
- `context_cache.csv`: context cache index with cache status/source/timestamps
- `context_cache/`: per-task JSON context bundles
