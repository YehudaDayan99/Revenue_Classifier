# Revenue Segmentation Pipeline

## Objective

> **Extract, for a given company's latest 10-K fiscal year, the complete set of revenue line items that are explicitly quantified in the filing and that represent products and/or services, and map each line item to the company's reported operating/business segments, producing a dataset that (a) is traceable to evidence in the filing and (b) reconciles to total revenue under a defined reconciliation policy.**

## Overview

This pipeline extracts revenue segmentation data from SEC 10-K filings using a combination of **LLM agents** and **deterministic extraction**. The goal is to produce structured output showing how a company's revenue breaks down by business segment and product/service line.

### Design Principles

1. **Evidence-based**: All extracted items must be traceable to the filing text
2. **Products/Services only**: Adjustments (hedging, corporate) are excluded from primary output
3. **Reconciliation**: Internal validation ensures extracted items sum to total revenue
4. **No hallucination**: CSV3 items require evidence spans validated against source text

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   10-K Filing (HTML)                                                         │
│         │                                                                    │
│         ▼                                                                    │
│   ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐              │
│   │   Scout     │────▶│   Discover   │────▶│  Table Select   │              │
│   │  (extract   │     │  (identify   │     │  (find best     │              │
│   │  headings,  │     │  segments)   │     │  revenue table) │              │
│   │  snippets)  │     │              │     │                 │              │
│   └─────────────┘     └──────────────┘     └────────┬────────┘              │
│                                                      │                       │
│                                                      ▼                       │
│                              ┌─────────────────────────────────────┐        │
│                              │         Layout Inference            │        │
│                              │   (identify columns, rows, units)   │        │
│                              └────────────────┬────────────────────┘        │
│                                               │                              │
│                                               ▼                              │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                  Deterministic Extraction                        │       │
│   │   • Parse table grid                                             │       │
│   │   • Map items to segments (using mappings.py)                    │       │
│   │   • Extract revenue values                                       │       │
│   │   • Validate against table total                                 │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                               │                              │
│                                               ▼                              │
│   ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐              │
│   │    CSV1     │     │    CSV2      │     │     CSV3        │              │
│   │  (revenue   │     │  (segment    │     │  (detailed      │              │
│   │   by item)  │     │  descrip.)   │     │   items)        │              │
│   └─────────────┘     └──────────────┘     └─────────────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: How It Works (Intuitive)

### The Problem
10-K filings contain revenue data in HTML tables, but these tables vary widely:
- Different structures (AAPL uses product categories, MSFT uses reportable segments)
- Different locations (Item 1, Item 7, Item 8 / Notes)
- Different formats (iXBRL with split cells, nested tables)

### The Solution: Agent-Based Approach

**1. Scout Agent** — Scans the document to extract:
- Section headings (to understand document structure)
- Text snippets containing revenue/segment keywords
- All table candidates with metadata (location, preview, numeric density)

**2. Discovery Agent** — Identifies business segments:
- Prompt asks: *"What are the primary business segments or product categories?"*
- Returns segment names (e.g., "Intelligent Cloud", "iPhone", "Google Services")
- Flags optional adjustment lines (e.g., "Corporate", "Hedging")

**3. Table Selection Agent** — Finds the best revenue table:
- Receives ranked table candidates with previews
- Prompt asks: *"Select the table that disaggregates revenue by these segments"*
- Prefers granular tables (product/service level) over segment totals
- Prefers Item 8 / Notes over Item 7 narrative

**4. Layout Inference Agent** — Understands table structure:
- Prompt asks: *"Which column has labels? Which columns have year data? What are the units?"*
- Returns: `item_col=0, year_cols={2024: 15}, units_multiplier=1000000`

**5. Deterministic Extraction** — No LLM, pure code:
- Parses the HTML table into a grid
- Uses `mappings.py` to assign items to segments (e.g., "LinkedIn" → "Productivity and Business Processes")
- Extracts values, handles accounting negatives `(500)`, validates totals

**6. Description Agents** — Enrich with context (evidence-based):
- CSV2: Summarizes each segment from bounded 10-K text (segment note section)
  - Uses revenue_items from CSV1 as grounding keywords
  - Excludes "Other" residual segments
- CSV3: **Extracts** (not expands) items with evidence validation:
  - Each item requires an `evidence_span` verbatim quote
  - Post-validation rejects items not found in source text
  - De-duplication prevents cross-segment duplicates

---

## Part 2: Technical Details

### Key Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Main orchestration, loops over tickers |
| `react_agents.py` | All LLM agent functions |
| `extraction/core.py` | Deterministic extraction logic |
| `extraction/matching.py` | Fuzzy segment name matching |
| `extraction/validation.py` | Revenue sum validation |
| `mappings.py` | Item-to-segment mappings per company |
| `table_candidates.py` | HTML parsing, table extraction |
| `table_kind.py` | Deterministic gates (reject unearned revenue, etc.) |

### LLM Configuration

- **Model**: `gpt-4.1-mini` (all agents)
- **Output format**: Strict JSON with defined schemas
- **Token limits**: 700-2000 depending on task

### Table Selection Logic

```
1. Extract all <table> elements from HTML
2. Score by: numeric_ratio, keyword_hits, location (Item 8 preferred)
3. Apply negative gates (reject: unearned revenue, leases, derivatives)
4. LLM selects best match from top 80 candidates
5. If validation fails, retry with next-best candidate (max 3 iterations)
```

### Validation Rules

**Revenue Extraction (internal)**:
```
|segment_sum + adjustment_sum - table_total| / table_total < 2%
```
- Adjustments (hedging, corporate) are included for validation only
- If no table total found, falls back to SEC CompanyFacts API

**CSV3 Evidence Validation**:
- Each item must have `evidence_span` found in source text (70%+ word match)
- OR item name must appear in source text
- Items failing validation are rejected and logged

### Output Schema

**CSV1** (primary output — products/services only):
```
Year, Company, Ticker, Segment, Item, Income $, Income %, Row type, Primary source, Link
```
- **Excludes**: Adjustments (hedging), "Other" residual segments
- **Includes**: Only explicitly quantified product/service revenue lines

**CSV2** (segment descriptions):
```
Company, Ticker, Segment, Segment description, Key products/services, Primary source, Link
```

**CSV3** (detailed items — evidence-based):
```
Company, Ticker, Segment, Business item, Short description, Long description, Evidence span, Link
```
- Items must be found in source text (not LLM-invented)
- Excludes "Other" and "Corporate" segments

### Adding New Companies

For companies where item-to-segment mapping isn't automatic:

1. Add mapping to `mappings.py`:
```python
NEWCO_ITEM_TO_SEGMENT = {
    "Product A": "Segment 1",
    "Product B": "Segment 2",
}
```

2. Update `get_segment_for_item()` to use it.

For most companies, the LLM agents handle mapping automatically.

---

## Running the Pipeline

```bash
# Single ticker
python -m revseg.pipeline --tickers MSFT --out-dir data/outputs

# Multiple tickers
python -m revseg.pipeline --tickers AAPL,MSFT,GOOGL --out-dir data/outputs

# Generate Excel
python -m revseg.export_xlsx --csv-dir data/outputs --out data/outputs/results.xlsx
```

### Artifacts

Each run produces artifacts in `data/artifacts/{TICKER}/`:
- `scout.json` — Document structure analysis
- `disagg_layout.json` — Inferred table layout
- `disagg_extracted.json` — Raw extraction results
- `csv2_llm.json`, `csv3_llm.json` — LLM responses for descriptions
- `trace.jsonl` — Full execution trace for debugging
