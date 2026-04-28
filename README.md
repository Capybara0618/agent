# SATD Multi-Agent Workflow

This workspace contains a LangGraph-based SATD workflow for SATD CSV datasets such as `code.csv` and `random_code.csv`.

The current retained implementation targets the final three-agent experiment (`outputs_three_agent`), which produced 633 workflow outputs and 120 exact-match repairs. Older compatibility code has been trimmed so the active path is:

- analyzer node: disabled by default and bypassed with a heuristic pass-through result
- fixer node: first asks the model which methods must be understood, then strictly retrieves those methods from the target commit, then performs a single repair
- reviewer node: disabled by default and bypassed to accept the fixer output directly
- selector: removed from the runtime; candidate choice is deterministic because only a single repair candidate is produced
- loop limit: 2 rounds remain available for compatibility, but the default experiment is a single repair path
- evaluation: compares the final repaired code with `manual_code` only after the workflow finishes
- all stages use the same OpenAI-compatible `gpt-4o-mini` interface

The graph never sees the ground-truth answer while making decisions.
The final comparison is done offline after preprocessing both sides.

## Repair Pipeline

Each SATD now follows three repair steps:

1. Method inquiry
   - send `SATD comment + original_code` to the model
   - ask which called methods/functions must be understood before repair
2. Strict method retrieval
   - retrieve only the named methods from the target `commit`
   - search current file first, then the historical repo tree
   - if a named method is missing, record it explicitly and do not broaden retrieval
3. Context-injected repair
   - inject only the retrieved method implementations plus the missing-method list
   - generate one repair candidate

Each SATD still gets a persistent per-task context bundle under `context_cache/`.

Layers:

- `base_context`
  - file path
  - original code
  - target file summary
- `repair_context`
  - method inquiry result
  - retrieved method implementations
  - missing method names
- `review_context`
  - compatibility shell for the bypassed reviewer stage

Workflow behavior:

- every SATD stores only lightweight local metadata before analyze
- repair writes method-query retrieval results into `repair_context`
- review reuses the existing context bundle when reviewer bypass is active
- all layers are saved to disk and reused on reruns

## Default Experiment Mode

The default CLI configuration is now:

- `analyzer` disabled
- `reviewer` disabled
- `selector` disabled
- single repair candidate
- `repair_context_mode=clone_treesitter`
- `max_method_contexts=5`

Clone/fetch acceleration options:

- default remote now points to TUNA mirror: `https://mirrors.tuna.tsinghua.edu.cn/git/github.com`
- use `--git-remote-base` to rewrite `https://github.com/<owner>/<repo>.git`
- for TUNA mirror, pass `--git-remote-base https://mirrors.tuna.tsinghua.edu.cn/git/github.com`
- use `--git-remote-template` if you need a fully custom pattern with `{owner}` and `{repo}`
- clone/fetch progress emits heartbeat logs while running under `--verbose`
- repo cache reuse is now strict: only repositories with a `clone_complete.json` marker and passing git self-checks are reused; invalid caches are rebuilt automatically

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
- `satd_langgraph/agents.py`: OpenAI-compatible client plus analyzer, fixer, and reviewer agents
- `satd_langgraph/workflow.py`: LangGraph state graph, cache persistence, and CSV writers
- `run_langgraph_workflow.py`: command-line entry point for the CSV experiment
- `code.csv`: original SATD dataset
- `random_code.csv`: 1000-row SATD dataset with `manual_clean` / `created_in_*` columns
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
python run_langgraph_workflow.py --input random_code.csv --output-dir outputs_langgraph_random1000 --model gpt-4o-mini --verbose
```

If you want a small smoke test first:

```powershell
python run_langgraph_workflow.py --input random_code.csv --output-dir outputs_langgraph_random_smoke --limit 20 --model gpt-4o-mini --verbose --write-batch-size 10
```

For `random_code.csv`, the CLI now defaults to a dedicated repo cache directory:

- `.repo_cache_random_code`

You can override it in either workflow or cache-warming runs with:

```powershell
--repo-cache-dir .repo_cache_any_name
```

## Dataset Assumptions

The current CSV loader supports these dataset shapes:

- `index`
- `SATD_comment`
- `original_code`
- `manual_code` or `manual_clean`
- `user`
- `project`
- `file_path` or `created_in_file`
- `commit` or `created_in_commit`
- optional `EM`

Unnamed empty columns in the CSV are ignored.

## Main Outputs

The main file to inspect is:

- `trajectory_overview.csv`

It gives one row per SATD and includes:

- analyzer decision, score, scope, validation signals, context gaps, and drop reason
- identified method names, retrieved method names, missing method names, and retrieved method count
- whether repair method context was actually used
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
- `repair_candidates.csv`: repair candidate table; default experiment writes one candidate per task
- `reviews.csv`: per-review detail table
- `github_context.csv`: flattened context content including `method_context_json`
- `context_cache.csv`: context cache index with cache status/source/timestamps
- `context_cache/`: per-task JSON context bundles
