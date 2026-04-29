# SATD System Flow

```mermaid
flowchart TD
    A["Input SATD dataset<br/>random_code.csv / code.csv"] --> B["Load and normalize CSV<br/>load_satd_csv"]
    B --> C["Build GraphState<br/>SATD comment, original code, manual repair, repo info, commit"]
    C --> D["Context router<br/>LLM sees only SATD comment + original_code"]

    D --> E{"Need repository context?"}
    E -->|no_context| F["Concise repair<br/>skip analyzer and method retrieval"]
    E -->|context_required| G["Analyzer context preparation<br/>method inquiry + strict retrieval"]

    G --> H["Analyzer gate<br/>decide repairable / drop"]
    H -->|drop| X["Drop task<br/>dropped_by_analyzer"]
    H -->|repairable| I["Context-injected fixer<br/>SATD + original_code + method evidence"]

    F --> J["Generate repaired_code"]
    I --> J
    J --> L["Reviewer"]

    L --> M["LLM review<br/>alignment, locality, semantics"]
    M --> M2{"Approved?"}
    M2 -->|yes| N["Accept repair<br/>final_repaired_code"]
    M2 -->|no in round 1| R["Reviewer feedback<br/>constraints + retry hint"]
    R --> I
    M2 -->|no in round 2| Y["Drop task<br/>dropped_after_review"]

    N --> O["Offline evaluation<br/>preprocess repaired_code and manual_code"]
    O --> P["Exact Match comparison<br/>remove comments/docstrings, normalize AST"]
    P --> Q["Output experiment files<br/>trajectory_overview.csv<br/>results.csv / repairs.csv / reviews.csv<br/>github_context.csv / context_cache.csv / summary.csv"]

    X --> Q
    Y --> Q
```

## Reporting Notes

- The system no longer hardcodes special SATD classes.
- The LLM context router is the first decision point.
- `no_context` tasks skip analyzer and go directly to concise repair.
- `context_required` tasks enter analyzer context retrieval and filtering before fixer/reviewer.
- Ground truth `manual_code` is used only for offline Exact Match evaluation after the workflow finishes.
