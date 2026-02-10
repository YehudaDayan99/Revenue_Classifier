# LLM Prompts Summary

This document summarizes all LLM prompts used in the Revenue Classifier pipeline.

## Overview

| Agent | Model | Purpose | Tokens (approx) |
|-------|-------|---------|-----------------|
| discover_primary_business_lines | gpt-4.1-mini | Identify business segments | ~2K input, 700 output |
| select_revenue_disaggregation_table | gpt-4.1-mini | Select best table | ~4K input, 700 output |
| infer_disaggregation_layout | gpt-4.1-mini | Parse table structure | ~3K input, 900 output |
| describe_revenue_lines | gpt-4.1 | Extract descriptions | ~6K input, 2K output |

**Total per ticker**: 4-6 calls, ~15-20K input tokens, ~4K output tokens

---

## 1. discover_primary_business_lines

**Purpose**: Infer the primary business segments/lines from filing text snippets.

**Model**: gpt-4.1-mini

### System Prompt
```
You are a financial filings analyst. Determine the primary business-line dimension for CSV1.
Rules:
- For AAPL, treat business lines as product categories (iPhone, Mac, iPad, Wearables/Home/Accessories, Services).
- For MSFT and GOOGL, treat business lines as reportable segments (e.g., Intelligent Cloud).
- If the filing includes corporate adjustments (e.g., hedging gains/losses) that are included in Total Revenues, 
  put that under include_segments_optional=['Corporate'].
Output STRICT JSON ONLY.
```

### User Prompt (JSON)
```json
{
  "ticker": "NVDA",
  "company_name": "NVIDIA Corporation",
  "snippets": ["<text snippets from filing>"],
  "few_shot_examples": [
    {"ticker": "AAPL", "dimension": "product_category", "segments": ["iPhone", "Mac", ...]},
    {"ticker": "MSFT", "dimension": "reportable_segments", "segments": [...]}
  ],
  "output_schema": {
    "dimension": "product_category | reportable_segments",
    "segments": "list[string]",
    "include_segments_optional": "list[string]",
    "notes": "short string"
  }
}
```

### Expected Output
```json
{
  "dimension": "reportable_segments",
  "segments": ["Compute & Networking", "Graphics"],
  "include_segments_optional": [],
  "notes": "NVIDIA reports by two segments"
}
```

---

## 2. select_revenue_disaggregation_table

**Purpose**: Select the single best table that disaggregates revenue by business lines.

**Model**: gpt-4.1-mini

### System Prompt
```
You are a financial filings analyst. Select the single best table that DISAGGREGATES revenue 
by business lines (segments or product categories) and includes a Total Revenue/Net Sales row.
Constraints:
- **CRITICAL**: When multiple tables exist, PREFER tables with inferred_dimension='product_service' 
  over tables with inferred_dimension='segment'. Product/service tables provide the most granular 
  revenue breakdown.
- Ignore geography-only tables (inferred_dimension='geography').
- Prefer Item 8 / Notes (Note 17 or Note 18 often has the most granular breakdown).
- Prefer tables whose year columns are recent fiscal years (>= 2018).
Output STRICT JSON ONLY.
```

### User Prompt (JSON)
```json
{
  "ticker": "NVDA",
  "company_name": "NVIDIA Corporation",
  "business_lines": ["Compute & Networking", "Graphics"],
  "keyword_hints": ["revenue", "segment"],
  "headings": ["<document headings>"],
  "retrieved_snippets": ["<text snippets>"],
  "table_candidates": [
    {
      "table_id": "t0042",
      "n_rows": 15,
      "n_cols": 5,
      "detected_years": [2024, 2025],
      "keyword_hits": ["revenue", "segment"],
      "item8_score": 3.0,
      "inferred_dimension": "product_service",
      "row_label_preview": ["Compute", "Networking", "Gaming", "..."],
      "caption_text": "Revenue by Reportable Segments"
    }
  ],
  "output_schema": {
    "table_id": "tXXXX",
    "confidence": "0..1",
    "selected_dimension": "product_service|segment|other",
    "rationale": "short string"
  }
}
```

### Expected Output
```json
{
  "table_id": "t0042",
  "confidence": 0.95,
  "selected_dimension": "product_service",
  "rationale": "Table contains Compute, Networking, Gaming, Automotive revenue lines with Total"
}
```

---

## 3. infer_disaggregation_layout

**Purpose**: Identify column structure and row patterns in the selected table.

**Model**: gpt-4.1-mini

### System Prompt
```
You analyze a revenue disaggregation table from a 10-K. 
Identify which columns correspond to Segment (optional), Item/Product (required), and years, 
and how to identify the Total row.
Important: year columns should be recent fiscal years (>= 2018) and usually appear as FY2025/FY2024 or 2025/2024.
Output STRICT JSON ONLY.
```

### User Prompt (JSON)
```json
{
  "ticker": "NVDA",
  "company_name": "NVIDIA Corporation",
  "table_id": "t0042",
  "business_lines": ["Compute & Networking", "Graphics"],
  "candidate_summary": {"<table metadata>"},
  "table_grid_preview": [
    ["", "2025", "2024", "2023"],
    ["Compute", "$47,524", "$23,869", "$14,026"],
    ["Networking", "$12,957", "$11,331", "$7,339"],
    ["Gaming", "$11,357", "$10,447", "$9,067"],
    ["Total revenue", "$130,497", "$60,922", "$26,974"]
  ],
  "output_schema": {
    "segment_col": "int|null",
    "item_col": "int",
    "year_cols": {"YYYY": "int column index"},
    "header_rows": "list[int]",
    "total_row_regex": "string regex",
    "exclude_row_regex": "string regex for rows to exclude",
    "units_multiplier": "int (1, 1000, 1000000)",
    "notes": "short string"
  }
}
```

### Expected Output
```json
{
  "segment_col": null,
  "item_col": 0,
  "year_cols": {"2025": 1, "2024": 2, "2023": 3},
  "header_rows": [0],
  "total_row_regex": "Total revenue|Total net revenue",
  "exclude_row_regex": "$^",
  "units_multiplier": 1000000,
  "notes": "Single column table with product lines as rows, values in millions"
}
```

---

## 4. describe_revenue_lines

**Purpose**: Extract textual descriptions for each revenue line item.

**Model**: gpt-4.1 (higher quality model for text extraction)

### Approach
This agent uses a multi-phase deterministic-first strategy:

1. **Phase 1**: Try heading-based extraction (AAPL-style)
2. **Phase 2**: Try Note 2 paragraph extraction (META-style)
3. **Phase 3**: Try segment enumeration extraction (NVDA-style)
4. **Phase 4**: Try footnote extraction (AMZN-style)
5. **Phase 5**: Fall back to LLM with targeted evidence windows

### LLM System Prompt (Phase 5 fallback only)
```
You are an EXTRACTIVE information retrieval system. You ONLY output items that are 
EXPLICITLY NAMED as products, services, or brands in the provided text.

STRICT RULES - VIOLATIONS WILL BE REJECTED:
1. ONLY output items whose EXACT NAME appears VERBATIM in the text.
2. 'evidence_span' MUST be a WORD-FOR-WORD quote from the text (15-40 words) that includes the item name.
3. DO NOT paraphrase, summarize, or infer. Quote EXACTLY.
4. If you cannot find an explicit mention, return empty items list.
5. Maximum 8 items per segment.
```

### LLM User Prompt (Phase 5 fallback only)
```json
{
  "ticker": "NVDA",
  "company_name": "NVIDIA Corporation",
  "segment": "Compute & Networking",
  "source_text": "<extracted text window around the segment>",
  "output_schema": {
    "items": [
      {
        "business_item": "string (exact product/service name from text)",
        "evidence_span": "string (15-40 word VERBATIM quote)"
      }
    ]
  }
}
```

---

## Key Design Decisions

### 1. Deterministic-First Description Extraction
The `describe_revenue_lines` function prioritizes deterministic extraction:
- Heading-based (looks for bold/strong tags with label names)
- Note 2 parsing (regex patterns for "X revenue includes...")
- Segment enumeration (for "Segment includes X; Y; Z" patterns)
- Footnote extraction (DOM-based parsing)

LLM is only used as a fallback, reducing cost and improving consistency.

### 2. Few-Shot Examples
The `discover_primary_business_lines` prompt includes few-shot examples for AAPL, MSFT, GOOGL to guide the model toward correct segment identification.

### 3. Strict JSON Output
All prompts use `"Output STRICT JSON ONLY"` and enforce JSON mode via OpenAI's `response_format={"type": "json_object"}`. Output schemas are included in the user prompt.

### 4. Dimension Pre-Classification
Tables are pre-classified into dimensions (product_service, segment, geography) using regex patterns before LLM selection. This helps guide the LLM to prefer granular tables.

### 5. Negative Pattern Filtering
`table_kind.py` contains extensive regex patterns to reject non-target tables (stock charts, earnings tables, volume metrics) before they reach the LLM.

---

## Cost Optimization Notes

1. **Model selection**: Using gpt-4.1-mini for table selection (cheaper) and gpt-4.1 only for description extraction (requires higher quality).

2. **Caching**: BeautifulSoup objects and LLM call results are cached within a run.

3. **Early rejection**: Tables are filtered by deterministic patterns before LLM calls.

4. **Token limits**: Input is truncated and only relevant snippets/previews are sent.

5. **Rate limiting**: Built-in 60 RPM throttle to avoid hitting rate limits.
