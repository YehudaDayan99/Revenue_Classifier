# Dev Review: Revenue Classifier Pipeline

## Review Objective
Critically evaluate the LLM agent architecture and general approach for extracting revenue segment data from SEC 10-K filings.

## What to Review

### 1. Agent Architecture (High Priority)
The pipeline uses multiple LLM agents for different tasks. Evaluate:

- **Agent separation**: Are the responsibilities well-divided?
- **Prompt design**: Are the system/user prompts clear, specific, and constrained?
- **JSON output contracts**: Are output schemas well-defined to minimize parsing errors?
- **Error handling**: How robust is the pipeline when LLM returns unexpected output?

Key agents in `react_agents.py`:
1. `discover_primary_business_lines()` - Infers business segments from text snippets
2. `select_revenue_disaggregation_table()` - Selects the best table from candidates
3. `infer_disaggregation_layout()` - Identifies column/row structure in selected table
4. `describe_revenue_lines()` - Extracts product/service descriptions

### 2. Table Selection Strategy (High Priority)
Review the table selection approach in `react_agents.py` and `table_kind.py`:

- **Candidate ranking**: Is the scoring heuristic effective?
- **Negative pattern filtering**: Are the regex patterns in `table_kind.py` sufficient?
- **Dimension classification**: How well does `_classify_table_dimension()` work?

### 3. Deterministic vs LLM Logic (Medium Priority)
The pipeline combines LLM inference with deterministic extraction:

- **LLM**: Table selection, layout inference, description extraction
- **Deterministic**: Column scoring (`choose_item_col`), value extraction, validation

Questions to evaluate:
- Is the LLM/deterministic balance appropriate?
- Could more logic be made deterministic to reduce cost/latency?
- Are there places where LLM would improve accuracy?

### 4. Validation Strategy (Medium Priority)
Review the validation approach in `pipeline.py` and `extraction/validation.py`:

- **Internal validation**: Does extracted sum match table total?
- **External validation**: Does total match SEC CompanyFacts API?
- **Fail-fast behavior**: Pipeline rejects mismatches >10%

Is this validation strategy sufficient? Too strict? Too lenient?

### 5. Known Failure Cases
Review the 2 failure samples (TSLA, WMT):

**TSLA (17.9% mismatch)**:
- Extracted: ~$80B
- SEC external: ~$98B
- Root cause to investigate: Missing revenue items?

**WMT (32.1% mismatch)**:
- Extracted: ~$462B
- SEC external: ~$681B
- Root cause to investigate: Fiscal year mismatch? Wrong table?

### 6. Success Cases
Review the 2 success samples (NVDA, AAPL) to understand what works well.

## Files Included

```
dev_review_pack/
├── REVIEW_PROMPT.md          # This file
├── LLM_PROMPTS_SUMMARY.md    # Summary of all LLM prompts
├── PIPELINE_FLOW.md          # Technical documentation
├── PROJECT_OVERVIEW.md       # High-level overview
├── csv1_segment_revenue.csv  # Output for 15 tickers
├── code/
│   ├── react_agents.py       # Main agent logic (2762 lines)
│   ├── pipeline.py           # Orchestration (1475 lines)
│   ├── llm_client.py         # OpenAI client
│   ├── table_kind.py         # Table rejection patterns
│   ├── table_candidates.py   # HTML table extraction
│   ├── mappings.py           # Company-specific mappings
│   ├── core.py               # Extraction logic
│   └── validation.py         # Validation logic
└── 10k_samples/
    ├── NVDA_success/         # Working example
    ├── AAPL_success/         # Working example
    ├── TSLA_failure/         # 17.9% mismatch
    └── WMT_failure/          # 32.1% mismatch
```

## Specific Questions for Reviewer

1. **Cost efficiency**: Each ticker currently uses 4-6 LLM calls. Can this be reduced?

2. **Robustness**: The pipeline succeeds on 75% of tickers. What's needed to reach 90%?

3. **Architecture**: Is the multi-agent approach appropriate, or would a single-shot extraction work better?

4. **Failure modes**: For TSLA and WMT failures, what specific changes would fix them?

5. **Scalability**: What changes are needed to process 100+ tickers reliably?

## Expected Output

Please provide:
1. Critical assessment of the architecture
2. Specific recommendations for improvements (prioritized)
3. Root cause analysis for TSLA and WMT failures
4. Estimated effort for suggested changes
