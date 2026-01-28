# CSV1 Revenue Segmentation Pipeline

## Objective

> **Extract, for a given company's latest 10-K fiscal year, the complete set of revenue line items that are explicitly quantified in the filing and that represent products and/or services, and map each line item to the company's reported operating/business segments, producing a dataset that (a) is traceable to evidence in the filing and (b) reconciles to total revenue under a defined reconciliation policy.**

---

## Pipeline Flow Schema

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                          CSV1 REVENUE EXTRACTION PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│   INPUT: Ticker Symbol (e.g., "MSFT")                                                       │
│         │                                                                                    │
│         ▼                                                                                    │
│   ┌───────────────────────────────────────────────────────────────────────────────────────┐ │
│   │  PHASE 1: DOCUMENT ACQUISITION                                                         │ │
│   │  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐               │ │
│   │  │  SEC EDGAR API  │─────▶│  Download 10-K  │─────▶│  Parse HTML     │               │ │
│   │  │  (REST)         │      │  (cache to disk)│      │  (BeautifulSoup)│               │ │
│   │  │                 │      │                 │      │                 │               │ │
│   │  │  Rate: 10 req/s │      │  Files:         │      │  Output:        │               │ │
│   │  │  User-Agent req │      │  - primary.htm  │      │  - Text (400k)  │               │ │
│   │  │                 │      │  - filing_ref   │      │  - Tables []    │               │ │
│   │  └─────────────────┘      └─────────────────┘      └────────┬────────┘               │ │
│   └──────────────────────────────────────────────────────────────┼───────────────────────┘ │
│                                                                   │                         │
│   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
│                                                                   │                         │
│   ┌───────────────────────────────────────────────────────────────▼───────────────────────┐ │
│   │  PHASE 2: DOCUMENT UNDERSTANDING (LLM: gpt-4.1-mini)                                  │ │
│   │                                                                                        │ │
│   │  ┌─────────────────────────────────────────────────────────────────────────────────┐  │ │
│   │  │  2A. SCOUT AGENT                                                                 │  │ │
│   │  │  ┌─────────────┐     ┌─────────────────────┐     ┌─────────────────────┐        │  │ │
│   │  │  │ Heading     │     │ Keyword Windows     │     │ Table Candidates    │        │  │ │
│   │  │  │ Extraction  │     │ (±2500 chars)       │     │ (~80-150 tables)    │        │  │ │
│   │  │  │             │     │                     │     │                     │        │  │ │
│   │  │  │ Regex:      │     │ Keywords:           │     │ Each table has:     │        │  │ │
│   │  │  │ ITEM\s*\d+  │     │ "segment"           │     │ - preview (15x10)   │        │  │ │
│   │  │  │ NOTE\s*\d+  │     │ "disaggregation"    │     │ - numeric_ratio     │        │  │ │
│   │  │  │             │     │ "revenue by"        │     │ - caption_text      │        │  │ │
│   │  │  │             │     │ "net sales"         │     │ - row_label_preview │        │  │ │
│   │  │  └─────────────┘     └─────────────────────┘     └─────────────────────┘        │  │ │
│   │  └─────────────────────────────────────────────────────────────────────────────────┘  │ │
│   │                                          │                                             │ │
│   │                                          ▼                                             │ │
│   │  ┌─────────────────────────────────────────────────────────────────────────────────┐  │ │
│   │  │  2B. DISCOVERY AGENT (LLM Call #1)                                               │  │ │
│   │  │                                                                                   │  │ │
│   │  │  Prompt: "What are the primary business segments/product categories for {ticker}?"│  │ │
│   │  │                                                                                   │  │ │
│   │  │  Output JSON:                                                                     │  │ │
│   │  │  {                                                                                │  │ │
│   │  │    "segments": ["Intelligent Cloud", "Productivity and Business Processes", ...], │  │ │
│   │  │    "include_segments_optional": ["Other", "Corporate"]                            │  │ │
│   │  │  }                                                                                │  │ │
│   │  │                                                                                   │  │ │
│   │  │  Function: discover_primary_business_lines()                                      │  │ │
│   │  └─────────────────────────────────────────────────────────────────────────────────┘  │ │
│   │                                          │                                             │ │
│   │                                          ▼                                             │ │
│   │  ┌─────────────────────────────────────────────────────────────────────────────────┐  │ │
│   │  │  2C. TABLE KIND GATE (Deterministic - No LLM)                                    │  │ │
│   │  │                                                                                   │  │ │
│   │  │  Reject tables matching NEGATIVE patterns:                                        │  │ │
│   │  │  • r"(?:unearned|deferred)\s+revenue"  → Liability tables                        │  │ │
│   │  │  • r"remaining\s+performance\s+obligation"  → RPO tables                         │  │ │
│   │  │  • r"lease\s+(?:liability|payment|income)"  → Lease tables                       │  │ │
│   │  │  • r"derivative|hedge|fair\s+value"  → Financial instrument tables               │  │ │
│   │  │                                                                                   │  │ │
│   │  │  Function: tablekind_gate() in table_kind.py                                      │  │ │
│   │  └─────────────────────────────────────────────────────────────────────────────────┘  │ │
│   │                                          │                                             │ │
│   │                                          ▼                                             │ │
│   │  ┌─────────────────────────────────────────────────────────────────────────────────┐  │ │
│   │  │  2D. TABLE SELECTION AGENT (LLM Call #2)                                         │  │ │
│   │  │                                                                                   │  │ │
│   │  │  Prompt: "Select the table that best disaggregates revenue by {segments}"        │  │ │
│   │  │                                                                                   │  │ │
│   │  │  Input: Top 80 gated candidates with previews                                     │  │ │
│   │  │  Output: { "table_id": "t0073", "confidence": 0.92, "reason": "..." }            │  │ │
│   │  │                                                                                   │  │ │
│   │  │  Function: select_revenue_disaggregation_table()                                  │  │ │
│   │  └─────────────────────────────────────────────────────────────────────────────────┘  │ │
│   └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                   │                         │
│   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
│                                                                   │                         │
│   ┌───────────────────────────────────────────────────────────────▼───────────────────────┐ │
│   │  PHASE 3: TABLE EXTRACTION                                                            │ │
│   │                                                                                        │ │
│   │  ┌─────────────────────────────────────────────────────────────────────────────────┐  │ │
│   │  │  3A. LAYOUT INFERENCE AGENT (LLM Call #3)                                        │  │ │
│   │  │                                                                                   │  │ │
│   │  │  Prompt: "For table {id}, identify label column, year columns, units"            │  │ │
│   │  │                                                                                   │  │ │
│   │  │  Output JSON:                                                                     │  │ │
│   │  │  {                                                                                │  │ │
│   │  │    "label_col": 0,                                                                │  │ │
│   │  │    "year_cols": { "2024": 15, "2023": 17, "2022": 19 },                           │  │ │
│   │  │    "header_rows": [0, 1],                                                         │  │ │
│   │  │    "units_multiplier": 1000000,                                                   │  │ │
│   │  │    "total_row_regex": "^total\\s+(?:net\\s+)?(?:revenue|sales)"                   │  │ │
│   │  │  }                                                                                │  │ │
│   │  │                                                                                   │  │ │
│   │  │  Function: infer_disaggregation_layout()                                          │  │ │
│   │  └─────────────────────────────────────────────────────────────────────────────────┘  │ │
│   │                                          │                                             │ │
│   │                                          ▼                                             │ │
│   │  ┌─────────────────────────────────────────────────────────────────────────────────┐  │ │
│   │  │  3B. DETERMINISTIC EXTRACTION (No LLM)                                           │  │ │
│   │  │                                                                                   │  │ │
│   │  │  ┌─────────────────────────────────────────────────────────────────────────────┐│  │ │
│   │  │  │ Step 1: Parse HTML Table to Grid                                            ││  │ │
│   │  │  │ Function: extract_table_grid_normalized()                                   ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ Handles iXBRL quirks:                                                       ││  │ │
│   │  │  │ • Split currency symbols: "$" in one cell, "123" in next                    ││  │ │
│   │  │  │ • Hidden/collapsed rows (visibility: collapse)                              ││  │ │
│   │  │  │ • Spacer columns with no data                                               ││  │ │
│   │  │  │ • Nested tables (flattens to single grid)                                   ││  │ │
│   │  │  └─────────────────────────────────────────────────────────────────────────────┘│  │ │
│   │  │                                          │                                       │  │ │
│   │  │                                          ▼                                       │  │ │
│   │  │  ┌─────────────────────────────────────────────────────────────────────────────┐│  │ │
│   │  │  │ Step 2: Dimension Detection                                                 ││  │ │
│   │  │  │ Function: detect_dimension()                                                ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ Classifies table as:                                                        ││  │ │
│   │  │  │ • "product_service" - AAPL style (iPhone, Mac, iPad, Services)              ││  │ │
│   │  │  │ • "segment" - MSFT style (Intelligent Cloud, PBP, MPC)                      ││  │ │
│   │  │  │ • "end_market" - NVDA style (Compute, Gaming, Automotive)                   ││  │ │
│   │  │  │ • "revenue_source" - META style (Advertising, Other revenue)                ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ Regex patterns:                                                             ││  │ │
│   │  │  │ • r"revenue\s+by\s+(?:product|service)" → product_service                   ││  │ │
│   │  │  │ • r"(?:reportable\s+)?segment" → segment                                    ││  │ │
│   │  │  │ • r"end\s+market" → end_market                                              ││  │ │
│   │  │  └─────────────────────────────────────────────────────────────────────────────┘│  │ │
│   │  │                                          │                                       │  │ │
│   │  │                                          ▼                                       │  │ │
│   │  │  ┌─────────────────────────────────────────────────────────────────────────────┐│  │ │
│   │  │  │ Step 3: Row Classification & Value Extraction                               ││  │ │
│   │  │  │ Function: extract_revenue_unified()                                         ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ For each row:                                                               ││  │ │
│   │  │  │ ┌─────────────────────────────────────────────────────────────────────────┐││  │ │
│   │  │  │ │ 1. Is it a TOTAL row?                                                   │││  │ │
│   │  │  │ │    Regex: r"^total\s+(?:net\s+)?(?:revenue|sales)"                      │││  │ │
│   │  │  │ │    → row_type = "total", skip from output                               │││  │ │
│   │  │  │ │                                                                         │││  │ │
│   │  │  │ │ 2. Is it a SUBTOTAL row?                                                │││  │ │
│   │  │  │ │    Check is_subtotal_row() in mappings.py                               │││  │ │
│   │  │  │ │    Examples: "Google Services total", "Google advertising"              │││  │ │
│   │  │  │ │    → row_type = "subtotal", skip to avoid double-counting               │││  │ │
│   │  │  │ │                                                                         │││  │ │
│   │  │  │ │ 3. Is it an ADJUSTMENT row?                                             │││  │ │
│   │  │  │ │    Regex: r"hedge|corporate|intersegment|elimination"                   │││  │ │
│   │  │  │ │    → row_type = "adjustment", used for validation only                  │││  │ │
│   │  │  │ │                                                                         │││  │ │
│   │  │  │ │ 4. Otherwise → row_type = "revenue_item"                                │││  │ │
│   │  │  │ │    Extract value: parse "$1,234" or "(500)" → int                       │││  │ │
│   │  │  │ │    Apply units_multiplier: value * 1_000_000                            │││  │ │
│   │  │  │ └─────────────────────────────────────────────────────────────────────────┘││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ Output: List[ExtractedRow] with:                                            ││  │ │
│   │  │  │ • item: "iPhone", segment: "Product", value: 209586000000                   ││  │ │
│   │  │  │ • row_type: "revenue_item", dimension: "product_service"                    ││  │ │
│   │  │  └─────────────────────────────────────────────────────────────────────────────┘│  │ │
│   │  │                                          │                                       │  │ │
│   │  │                                          ▼                                       │  │ │
│   │  │  ┌─────────────────────────────────────────────────────────────────────────────┐│  │ │
│   │  │  │ Step 4: Segment Mapping                                                     ││  │ │
│   │  │  │ Function: get_segment_for_item() in mappings.py                             ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ Maps items to Revenue Groups using company-specific dictionaries:           ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ MSFT_ITEM_TO_SEGMENT = {                                                    ││  │ │
│   │  │  │   "LinkedIn": "Productivity and Business Processes",                        ││  │ │
│   │  │  │   "Server products": "Intelligent Cloud",                                   ││  │ │
│   │  │  │   "Gaming": "More Personal Computing",                                      ││  │ │
│   │  │  │ }                                                                           ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ NVDA_ITEM_TO_SEGMENT = {                                                    ││  │ │
│   │  │  │   "Compute": "Compute & Networking",                                        ││  │ │
│   │  │  │   "Gaming": "Graphics",                                                     ││  │ │
│   │  │  │ }                                                                           ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ Fallback: If no mapping, use "Product/Service disclosure"                   ││  │ │
│   │  │  └─────────────────────────────────────────────────────────────────────────────┘│  │ │
│   │  │                                          │                                       │  │ │
│   │  │                                          ▼                                       │  │ │
│   │  │  ┌─────────────────────────────────────────────────────────────────────────────┐│  │ │
│   │  │  │ Step 5: Validation                                                          ││  │ │
│   │  │  │ Function: validate_extraction() in extraction/validation.py                 ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ Rule: |segment_sum + adjustment_sum - reference_total| / reference_total < 2%│  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ Reference priority:                                                         ││  │ │
│   │  │  │ 1. Table's own "Total revenue" row (self-consistent)                        ││  │ │
│   │  │  │ 2. SEC CompanyFacts API (external validation)                               ││  │ │
│   │  │  │    API: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json            ││  │ │
│   │  │  │    Field: facts.us-gaap.Revenues.units.USD[fiscal_year].val                 ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ If validation fails → retry with next-best table (max 3 attempts)           ││  │ │
│   │  │  └─────────────────────────────────────────────────────────────────────────────┘│  │ │
│   │  └─────────────────────────────────────────────────────────────────────────────────┘  │ │
│   └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                   │                         │
│   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
│                                                                   │                         │
│   ┌───────────────────────────────────────────────────────────────▼───────────────────────┐ │
│   │  PHASE 4: DESCRIPTION EXTRACTION (LLM: gpt-4.1)                                       │ │
│   │                                                                                        │ │
│   │  ┌─────────────────────────────────────────────────────────────────────────────────┐  │ │
│   │  │  OPTION A: LEGACY KEYWORD-BASED (Default)                                        │  │ │
│   │  │  Function: describe_revenue_lines() in react_agents.py                           │  │ │
│   │  │                                                                                   │  │ │
│   │  │  Process:                                                                         │  │ │
│   │  │  1. DOM-BASED FOOTNOTE ID RECOVERY (new - handles iXBRL)                          │  │ │
│   │  │     • Function: extract_footnote_ids_from_table()                                │  │ │
│   │  │     • Parse <sup>, <a> tags in table cells to recover footnote markers           │  │ │
│   │  │     • Maps: {"Online stores": ["1"], "AWS": ["3"]}                               │  │ │
│   │  │     • Solves: AMZN footnotes lost during text extraction                         │  │ │
│   │  │                                                                                   │  │ │
│   │  │  2. FOOTNOTE EXTRACTION (first priority)                                          │  │ │
│   │  │     • Use DOM-recovered IDs OR regex marker: "Online stores (1)"                 │  │ │
│   │  │     • Search for separator: r"_{5,}" (5+ underscores)                            │  │ │
│   │  │     • Extract: r"\(1\)\s+(Includes|Consists|Represents.*?)(?=\(\d+\)|$)"         │  │ │
│   │  │     • Apply: strip_accounting_sentences() to remove accounting language          │  │ │
│   │  │                                                                                   │  │ │
│   │  │  3. SECTION-AWARE SEARCH (fallback)                                               │  │ │
│   │  │     • Priority: Item 1 (Business) → Item 8 (Notes) → Full text                   │  │ │
│   │  │     • NOTE: Item 7 (MD&A) EXCLUDED - contains performance drivers not definitions│  │ │
│   │  │     • Search for revenue line label, extract ±2500 char window                   │  │ │
│   │  │                                                                                   │  │ │
│   │  │  4. ACCOUNTING SENTENCE FILTER (new)                                              │  │ │
│   │  │     • Function: strip_accounting_sentences()                                      │  │ │
│   │  │     • Removes sentences with: "recognized", "deferred", "amortization",          │  │ │
│   │  │       "performance obligation", "increased due to", "driven by", etc.            │  │ │
│   │  │     • Applied to: footnote text + LLM output                                     │  │ │
│   │  │                                                                                   │  │ │
│   │  │  5. LLM SUMMARIZATION (LLM Call #4)                                               │  │ │
│   │  │     • Model: gpt-4.1                                                              │  │ │
│   │  │     • Prompt: "Extract PRODUCT/SERVICE DEFINITION. Describe WHAT IT IS, not      │  │ │
│   │  │       how it performed. EXCLUDE: accounting language, performance drivers."       │  │ │
│   │  │     • Output: { "rows": [{ "revenue_line": "...", "description": "..." }] }      │  │ │
│   │  └─────────────────────────────────────────────────────────────────────────────────┘  │ │
│   │                                                                                        │ │
│   │                               ─── OR ───                                               │ │
│   │                                                                                        │ │
│   │  ┌─────────────────────────────────────────────────────────────────────────────────┐  │ │
│   │  │  OPTION B: RAG-BASED (--use-rag flag)                                            │  │ │
│   │  │  Module: revseg/rag/                                                              │  │ │
│   │  │                                                                                   │  │ │
│   │  │  ┌─────────────────────────────────────────────────────────────────────────────┐│  │ │
│   │  │  │ 4B.1 PREPROCESSING (one-time per filing)                                    ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ a) TOC Detection (non-destructive)                                          ││  │ │
│   │  │  │    Function: detect_toc_regions()                                           ││  │ │
│   │  │  │    Regex: r"Item\s+\d+[A-Z]?\s*[\.\s]{2,}" (5+ in 2000 chars = TOC)         ││  │ │
│   │  │  │    Output: chunks marked is_toc=True, excluded at retrieval                 ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ b) Section Identification                                                   ││  │ │
│   │  │  │    Function: _identify_sections()                                           ││  │ │
│   │  │  │    Patterns:                                                                ││  │ │
│   │  │  │    • item1: r"ITEM\s*1[^0-9A-Z].*?BUSINESS"                                 ││  │ │
│   │  │  │    • item7: r"ITEM\s*7[^A-Z].*?MANAGEMENT.{0,30}DISCUSSION"                 ││  │ │
│   │  │  │    • note_segment: r"NOTE\s*\d+\s*[-–—]?\s*(SEGMENT|OPERATING)"             ││  │ │
│   │  │  │    • note_revenue: r"NOTE\s*\d+\s*[-–—]?\s*REVENUE"                         ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ c) Structure-Aware Chunking                                                 ││  │ │
│   │  │  │    Function: chunk_10k_structured()                                         ││  │ │
│   │  │  │    • 800 chars per chunk, 100 char overlap                                  ││  │ │
│   │  │  │    • Each chunk has: section, heading, char_range, is_toc                   ││  │ │
│   │  │  │    • ~400-600 chunks per 10-K                                               ││  │ │
│   │  │  │    • P1: note_revenue chunks sub-classified via classify_note_revenue_chunk()│  │ │
│   │  │  │      → note_revenue_sources (definitions) OR note_revenue_recognition       ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ d) Embedding Generation                                                     ││  │ │
│   │  │  │    API: OpenAI text-embedding-3-small (1536 dims)                           ││  │ │
│   │  │  │    Cost: ~$0.002 per 10-K (100k tokens)                                     ││  │ │
│   │  │  │    Function: embed_chunks() batches 100 chunks/request                      ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ e) FAISS Index Build                                                        ││  │ │
│   │  │  │    Class: TwoTierIndex                                                      ││  │ │
│   │  │  │    • Tier 1: table-local chunks (DOM siblings, footnotes)                   ││  │ │
│   │  │  │    • Tier 2: full-filing chunks (~500 chunks)                               ││  │ │
│   │  │  │    Storage: data/embeddings/{ticker}/*.faiss + metadata.json                ││  │ │
│   │  │  └─────────────────────────────────────────────────────────────────────────────┘│  │ │
│   │  │                                          │                                       │  │ │
│   │  │                                          ▼                                       │  │ │
│   │  │  ┌─────────────────────────────────────────────────────────────────────────────┐│  │ │
│   │  │  │ 4B.2 RETRIEVAL (per revenue line)                                           ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ a) Rich Query Construction                                                  ││  │ │
│   │  │  │    Function: build_rag_query()                                              ││  │ │
│   │  │  │    Example: "NVIDIA (NVDA) FY2025 revenue line 'Compute' in segment         ││  │ │
│   │  │  │             'Compute & Networking'. Products and services included.          ││  │ │
│   │  │  │             Use definitions from revenue/segment note."                     ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ b) Two-Tier Retrieval                                                       ││  │ │
│   │  │  │    Function: TwoTierIndex.retrieve()                                        ││  │ │
│   │  │  │    • Tier 1: table-local, threshold=0.55, if score≥threshold → use         ││  │ │
│   │  │  │    • Tier 2: full-filing, threshold=0.45, section boosting                  ││  │ │
│   │  │  │      - Boost 1.15x: note_revenue_sources, note_segment, item1, table_footnote│  │ │
│   │  │  │      - BLOCKED: note_revenue_recognition (P1: accounting mechanics)        ││  │ │
│   │  │  │      - Reduce 0.75x: item1a (risk factors), liquidity                       ││  │ │
│   │  │  │    • MMR deduplication (remove chunks with >85% text similarity)            ││  │ │
│   │  │  │    Output: top 10 chunks (before boost), then top 5 after                   ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ c) P1: Definitional Chunk Boost                                            ││  │ │
│   │  │  │    Function: _boost_definitional_chunks()                                   ││  │ │
│   │  │  │    • After retrieval, re-rank by boosting chunks with BOTH:                 ││  │ │
│   │  │  │      - The revenue line label (e.g., "Other revenue")                       ││  │ │
│   │  │  │      - Definition patterns: "consists of", "includes", "comprises"          ││  │ │
│   │  │  │    • Boost factor: 1.25x                                                    ││  │ │
│   │  │  │    • Fixes: META "Other revenue" (generic mentions → actual definition)     ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ d) Evidence Gate                                                            ││  │ │
│   │  │  │    Function: check_evidence_gate()                                          ││  │ │
│   │  │  │    Pass if: tier1_local OR ≥1 chunk from preferred_section OR max_score≥0.55│  │ │
│   │  │  │    Fail → return empty description (don't hallucinate)                      ││  │ │
│   │  │  └─────────────────────────────────────────────────────────────────────────────┘│  │ │
│   │  │                                          │                                       │  │ │
│   │  │                                          ▼                                       │  │ │
│   │  │  ┌─────────────────────────────────────────────────────────────────────────────┐│  │ │
│   │  │  │ 4B.3 GENERATION (LLM Call #4)                                               ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ a) Extractive-First Products                                                ││  │ │
│   │  │  │    Function: extract_candidate_products()                                   ││  │ │
│   │  │  │    Deterministic patterns:                                                  ││  │ │
│   │  │  │    • Capitalized phrases: r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b"       ││  │ │
│   │  │  │    • Trademarks: r"(\b\w+[®™])"                                             ││  │ │
│   │  │  │    • Model numbers: r"\b([A-Z]{1,4}\d{2,4}[A-Z]?)\b"                        ││  │ │
│   │  │  │    • "including X, Y" patterns                                              ││  │ │
│   │  │  │    Output: candidate set for LLM to filter                                  ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ b) LLM Generation                                                           ││  │ │
│   │  │  │    Function: generate_description_with_evidence()                           ││  │ │
│   │  │  │    Model: gpt-4.1                                                           ││  │ │
│   │  │  │    Prompt: "Extract description from chunks. Filter candidate_products      ││  │ │
│   │  │  │             to only those EXPLICITLY mentioned for this revenue line."      ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │    Output JSON:                                                             ││  │ │
│   │  │  │    {                                                                        ││  │ │
│   │  │  │      "description": "1-2 sentences in company language",                    ││  │ │
│   │  │  │      "products_services_list": ["Azure", "DGX", ...],                       ││  │ │
│   │  │  │      "evidence_chunk_ids": ["chunk_0142", "chunk_0143"],                    ││  │ │
│   │  │  │      "evidence_quotes": ["exact text from filing"]                          ││  │ │
│   │  │  │    }                                                                        ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ c) Post-Validation                                                          ││  │ │
│   │  │  │    • Verify evidence_chunk_ids exist in retrieved set                       ││  │ │
│   │  │  │    • Verify evidence_quotes[:50] found in chunk text                        ││  │ │
│   │  │  │    • If validation fails → return empty (don't use hallucinated text)       ││  │ │
│   │  │  └─────────────────────────────────────────────────────────────────────────────┘│  │ │
│   │  │                                          │                                       │  │ │
│   │  │                                          ▼                                       │  │ │
│   │  │  ┌─────────────────────────────────────────────────────────────────────────────┐│  │ │
│   │  │  │ 4B.4 QA ARTIFACT                                                            ││  │ │
│   │  │  │ Function: write_csv1_qa_artifact()                                          ││  │ │
│   │  │  │                                                                             ││  │ │
│   │  │  │ Output: data/artifacts/{ticker}/{ticker}_csv1_desc_coverage.json            ││  │ │
│   │  │  │ {                                                                           ││  │ │
│   │  │  │   "ticker": "NVDA",                                                         ││  │ │
│   │  │  │   "coverage_pct": 83.3,                                                     ││  │ │
│   │  │  │   "missing_labels": ["OEM and Other"],                                      ││  │ │
│   │  │  │   "tier1_count": 2, "tier2_count": 3,                                       ││  │ │
│   │  │  │   "line_details": [{ "revenue_line": "Compute", ... }]                      ││  │ │
│   │  │  │ }                                                                           ││  │ │
│   │  │  └─────────────────────────────────────────────────────────────────────────────┘│  │ │
│   │  └─────────────────────────────────────────────────────────────────────────────────┘  │ │
│   └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                   │                         │
│   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
│                                                                   │                         │
│   ┌───────────────────────────────────────────────────────────────▼───────────────────────┐ │
│   │  PHASE 5: OUTPUT GENERATION                                                           │ │
│   │                                                                                        │ │
│   │  ┌─────────────────────────────────────────────────────────────────────────────────┐  │ │
│   │  │  5A. FOOTNOTE STRIPPING                                                          │  │ │
│   │  │  Function: _clean_revenue_line()                                                  │  │ │
│   │  │  Regex: r"\s*\(\d+\)\s*$"                                                         │  │ │
│   │  │  "Online stores (1)" → "Online stores"                                            │  │ │
│   │  └─────────────────────────────────────────────────────────────────────────────────┘  │ │
│   │                                          │                                             │ │
│   │                                          ▼                                             │ │
│   │  ┌─────────────────────────────────────────────────────────────────────────────────┐  │ │
│   │  │  5B. REVENUE GROUP ASSIGNMENT                                                    │  │ │
│   │  │  Function: _get_revenue_group()                                                   │  │ │
│   │  │                                                                                   │  │ │
│   │  │  Priority:                                                                        │  │ │
│   │  │  1. Explicit mapping from mappings.py (company-specific)                         │  │ │
│   │  │  2. If dimension="segment" → use row's segment name                              │  │ │
│   │  │  3. Fallback → "Product/Service disclosure"                                      │  │ │
│   │  └─────────────────────────────────────────────────────────────────────────────────┘  │ │
│   │                                          │                                             │ │
│   │                                          ▼                                             │ │
│   │  ┌─────────────────────────────────────────────────────────────────────────────────┐  │ │
│   │  │  5C. UNIT CONVERSION                                                             │  │ │
│   │  │  Function: _to_millions()                                                         │  │ │
│   │  │  209586000000 → 209586.0 ($ millions)                                             │  │ │
│   │  └─────────────────────────────────────────────────────────────────────────────────┘  │ │
│   │                                          │                                             │ │
│   │                                          ▼                                             │ │
│   │  ┌─────────────────────────────────────────────────────────────────────────────────┐  │ │
│   │  │  5D. CSV1 OUTPUT                                                                 │  │ │
│   │  │  File: data/outputs/csv1_segment_revenue.csv                                      │  │ │
│   │  │                                                                                   │  │ │
│   │  │  Schema:                                                                          │  │ │
│   │  │  ┌────────────────────────────────────────────────────────────────────────────┐  │  │ │
│   │  │  │ Company Name | Ticker | Fiscal Year | Revenue Group (Reportable Segment) | │  │  │ │
│   │  │  │ Revenue Line | Line Item description (company language) | Revenue ($m)    │  │  │ │
│   │  │  └────────────────────────────────────────────────────────────────────────────┘  │  │ │
│   │  │                                                                                   │  │ │
│   │  │  Example:                                                                         │  │ │
│   │  │  NVIDIA CORP,NVDA,2025,Compute & Networking,Compute,"The Compute revenue line    │  │ │
│   │  │  includes data center compute platforms for accelerated computing and AI...",    │  │ │
│   │  │  102196.0                                                                         │  │ │
│   │  └─────────────────────────────────────────────────────────────────────────────────┘  │ │
│   └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Intuitive Description

### The Challenge

SEC 10-K filings contain detailed revenue breakdowns, but extracting them is hard because:

1. **Varied Structures**: AAPL lists products (iPhone, Mac), MSFT lists segments (Intelligent Cloud), GOOGL lists both plus sub-categories
2. **Hidden in Tables**: Revenue tables are buried among 80-150 tables (balance sheets, expenses, leases, derivatives)
3. **iXBRL Complexity**: Currency symbols split across cells, hidden rows, nested tables
4. **Descriptions in Footnotes**: "Online stores (1)" where `(1)` refers to a footnote 3000 characters away

### The Solution: Multi-Phase Agent Pipeline

**Phase 1: Document Acquisition**
- Download 10-K from SEC EDGAR API (respecting 10 req/sec limit)
- Parse HTML with BeautifulSoup, extract up to 400k chars of text

**Phase 2: Document Understanding**
- **Scout Agent**: Identifies document structure (headings, tables, keywords)
- **Discovery Agent (LLM)**: Asks "What are this company's business segments?"
- **Table Kind Gate**: Rejects wrong tables (unearned revenue, leases) using regex patterns
- **Table Selection Agent (LLM)**: Picks the best revenue disaggregation table

**Phase 3: Table Extraction**
- **Layout Agent (LLM)**: Determines which column has labels, which has 2024 values
- **Deterministic Extraction**: Parses table without LLM, applies mappings, validates totals
- No LLM guessing at numbers—pure code extraction with fallback strategies

**Phase 4: Description Extraction**
Two options:
- **Legacy (default)**: Search for footnote markers, extract from `_____` separator patterns
- **RAG (--use-rag)**: Embed entire 10-K, semantic search for relevant chunks, LLM summarizes

**Phase 5: Output Generation**
- Clean footnote markers ("Online stores (1)" → "Online stores")
- Assign Revenue Groups using company mappings
- Convert to millions, write CSV1

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Deterministic extraction** | LLMs can't reliably extract numbers; use code instead |
| **Self-consistent validation** | Compare to table's own "Total" row, not just external API |
| **Tiered LLM models** | gpt-4.1-mini for volume (fast), gpt-4.1 for descriptions (quality) |
| **Footnote priority** | Company's own footnote text is more accurate than LLM paraphrase |
| **RAG optional** | Semantic search helps when footnotes aren't structured (NVDA) |

---

## Part 2: Technical Reference

### Key Functions by Phase

#### Phase 1: Document Acquisition

| Function | File | Description |
|----------|------|-------------|
| `download_latest_10k()` | `sec_edgar.py` | Downloads 10-K from SEC EDGAR |
| `find_primary_document_html()` | `table_candidates.py` | Locates main HTML file |
| `_html_text_for_llm()` | `pipeline.py` | Extracts text, caches result |

#### Phase 2: Document Understanding

| Function | File | Description |
|----------|------|-------------|
| `document_scout()` | `react_agents.py` | Extracts headings, snippets |
| `extract_keyword_windows()` | `react_agents.py` | Finds text around keywords |
| `discover_primary_business_lines()` | `react_agents.py` | LLM identifies segments |
| `tablekind_gate()` | `table_kind.py` | Rejects non-revenue tables |
| `select_revenue_disaggregation_table()` | `react_agents.py` | LLM picks best table |

#### Phase 3: Table Extraction

| Function | File | Description |
|----------|------|-------------|
| `extract_table_grid_normalized()` | `react_agents.py` | Parses HTML table to grid |
| `infer_disaggregation_layout()` | `react_agents.py` | LLM infers table structure |
| `detect_dimension()` | `extraction/core.py` | Classifies table type |
| `extract_revenue_unified()` | `extraction/core.py` | Extracts all rows with types |
| `get_segment_for_item()` | `mappings.py` | Maps items to segments |
| `validate_extraction()` | `extraction/validation.py` | Validates sum vs total |

#### Phase 4: Description Extraction

**Legacy (default):**

| Function | File | Description |
|----------|------|-------------|
| `describe_revenue_lines()` | `react_agents.py` | Main orchestrator: DOM + keyword + LLM |
| `extract_footnote_ids_from_table()` | `react_agents.py` | **NEW**: Recovers footnote IDs from DOM |
| `_extract_footnote_for_label()` | `react_agents.py` | Finds footnote definitions |
| `_extract_footnotes_from_text()` | `react_agents.py` | Batch extracts all footnotes |
| `strip_accounting_sentences()` | `react_agents.py` | **NEW**: Removes accounting/driver text |
| `_extract_section()` | `react_agents.py` | Extracts Item 1/8 sections (Item 7 excluded) |
| `extract_footnotes_from_dom_context()` | `react_agents.py` | **P2**: DOM-based footnote extraction |
| `_extract_heading_based_definition()` | `react_agents.py` | **P3**: Finds definitions under headings |

**RAG (--use-rag):**

| Function | File | Description |
|----------|------|-------------|
| `detect_toc_regions()` | `rag/chunking.py` | Identifies TOC areas |
| `chunk_10k_structured()` | `rag/chunking.py` | Creates chunks with metadata |
| `classify_note_revenue_chunk()` | `rag/chunking.py` | **P1**: Separates sources from recognition |
| `embed_chunks()` | `rag/index.py` | Calls OpenAI embeddings API |
| `TwoTierIndex.build()` | `rag/index.py` | Builds FAISS indexes |
| `TwoTierIndex.retrieve()` | `rag/index.py` | Semantic search with boosting |
| `_boost_definitional_chunks()` | `rag/generation.py` | **P1**: Re-ranks by definition patterns |
| `check_evidence_gate()` | `rag/generation.py` | Validates chunk quality |
| `extract_candidate_products()` | `rag/generation.py` | Deterministic product extraction |
| `generate_description_with_evidence()` | `rag/generation.py` | LLM with evidence validation |
| `write_csv1_qa_artifact()` | `rag/qa.py` | Writes coverage metrics |

#### Phase 5: Output Generation

| Function | File | Description |
|----------|------|-------------|
| `_clean_revenue_line()` | `pipeline.py` | Strips footnote markers |
| `_get_revenue_group()` | `pipeline.py` | Assigns Revenue Group |
| `_to_millions()` | `pipeline.py` | Converts to $M |
| `_write_csv()` | `pipeline.py` | Writes CSV file |

---

### API Usage

| API | Endpoint | Purpose | Rate Limit |
|-----|----------|---------|------------|
| SEC EDGAR | `sec.gov/cgi-bin/browse-edgar` | Download 10-K filings | 10 req/sec |
| SEC CompanyFacts | `data.sec.gov/api/xbrl/companyfacts/` | External revenue validation | 10 req/sec |
| OpenAI Chat | `api.openai.com/v1/chat/completions` | LLM agents | 60 req/min |
| OpenAI Embeddings | `api.openai.com/v1/embeddings` | RAG embeddings | 3000 req/min |

---

### LLM Configuration

| Agent | Model | Tokens (in/out) | Purpose |
|-------|-------|-----------------|---------|
| Discovery | `gpt-4.1-mini` | ~3000/500 | Identify segments |
| Table Selection | `gpt-4.1-mini` | ~8000/500 | Pick revenue table |
| Layout Inference | `gpt-4.1-mini` | ~2000/500 | Understand table structure |
| Description | `gpt-4.1` | ~4000/600 | Generate company-language descriptions |
| RAG Generation | `gpt-4.1` | ~3000/500 | Summarize from retrieved chunks |

---

### Key Regex Patterns

#### Table Kind Gating (`table_kind.py`)

```python
STRICT_NEGATIVE_PATTERNS = [
    re.compile(r"(?:unearned|deferred)\s+revenue", re.IGNORECASE),
    re.compile(r"remaining\s+performance\s+obligation", re.IGNORECASE),
    re.compile(r"contract\s+(?:liability|liabilities)", re.IGNORECASE),
]

NEGATIVE_PATTERNS = [
    re.compile(r"lease\s+(?:liability|payment|income|asset)", re.IGNORECASE),
    re.compile(r"derivative|hedge|fair\s+value\s+measurement", re.IGNORECASE),
    re.compile(r"(?:accounts|notes)\s+(?:payable|receivable)", re.IGNORECASE),
]
```

#### Dimension Detection (`extraction/core.py`)

```python
DIMENSION_PATTERNS = {
    "product_service": [
        re.compile(r"revenue\s+by\s+(?:product|service)", re.IGNORECASE),
        re.compile(r"net\s+sales\s+by\s+(?:product|category)", re.IGNORECASE),
    ],
    "segment": [
        re.compile(r"(?:reportable\s+)?segment", re.IGNORECASE),
        re.compile(r"operating\s+segment", re.IGNORECASE),
    ],
    "end_market": [
        re.compile(r"end\s+market", re.IGNORECASE),
        re.compile(r"revenue\s+by\s+market", re.IGNORECASE),
    ],
}
```

#### Row Classification (`extraction/core.py`)

```python
TOTAL_ROW_PATTERNS = [
    re.compile(r"^total\s+(?:net\s+)?(?:revenue|sales)", re.IGNORECASE),
    re.compile(r"^total\s+net\s+sales", re.IGNORECASE),
]

ADJUSTMENT_PATTERNS = [
    re.compile(r"hedge|hedging", re.IGNORECASE),
    re.compile(r"corporate", re.IGNORECASE),
    re.compile(r"intersegment|elimination", re.IGNORECASE),
]
```

#### Footnote Extraction (`react_agents.py`)

```python
# Footnote marker in label
FOOTNOTE_MARKER_RE = re.compile(r"\((\d+)\)\s*$")

# Separator line (5+ underscores)
SEPARATOR_RE = re.compile(r"_{5,}")

# Footnote definition after separator (prioritizes "Includes" pattern)
FOOTNOTE_DEF_RE = re.compile(r"\((\d+)\)\s+(Includes|Consists|Represents\s+.*?)(?=\(\d+\)|$)", re.DOTALL)
```

#### Accounting Sentence Filter (`react_agents.py`)

```python
# Deny patterns - sentences containing these are removed from descriptions
ACCOUNTING_DENY_PATTERNS = [
    # Revenue recognition / accounting
    r"\bperformance obligation\b", r"\brecognized\b", r"\bdeferred\b",
    r"\bamortization\b", r"\bcontract liabilit", r"\bASC\s+\d", r"\bGAAP\b",
    # Performance drivers (MD&A-style)
    r"\bincreased due to\b", r"\bdecreased due to\b", r"\bprimarily driven\b",
    r"\bhigher sales of\b", r"\blower sales of\b", r"\bcompared to\b.*\bprior\b",
]
```

#### TOC Detection (`rag/chunking.py`)

```python
# Dense Item listings indicate TOC
TOC_ITEM_RE = re.compile(r"Item\s+\d+[A-Z]?\s*[\.\s]{2,}", re.IGNORECASE)

# Section headers
SECTION_PATTERNS = {
    'item1': re.compile(r'ITEM\s*1[^0-9A-Z].*?BUSINESS', re.IGNORECASE),
    'item7': re.compile(r'ITEM\s*7[^A-Z].*?MANAGEMENT.{0,30}DISCUSSION', re.IGNORECASE),
    'note_segment': re.compile(r'NOTE\s*\d+\s*[-–—]?\s*(SEGMENT|OPERATING)', re.IGNORECASE),
    'note_revenue': re.compile(r'NOTE\s*\d+\s*[-–—]?\s*REVENUE', re.IGNORECASE),
}
```

#### P1: Note Revenue Classification (`rag/chunking.py`)

```python
# Definition patterns → note_revenue_sources (keep for retrieval)
DEFINITION_PATTERNS = [
    re.compile(r'\bconsists?\s+of\b', re.IGNORECASE),
    re.compile(r'\bincludes?\b.*\b(?:products?|services?)\b', re.IGNORECASE),
    re.compile(r'\bgenerat(?:es?|ed)\s+from\b', re.IGNORECASE),
]

# Accounting patterns → note_revenue_recognition (block from retrieval)
ACCOUNTING_PATTERNS = [
    re.compile(r'\bperformance\s+obligat', re.IGNORECASE),
    re.compile(r'\brecogniz(?:es?|ed|ing)\s+(?:revenue|when|upon)\b', re.IGNORECASE),
    re.compile(r'\bSSP\b|\bstand-alone\s+selling\s+price\b', re.IGNORECASE),
    re.compile(r'\bASC\s+\d{3}\b', re.IGNORECASE),
]
```

#### Product Extraction (`rag/generation.py`)

```python
# Capitalized multi-word phrases
CAP_PHRASE_RE = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b')

# Trademarks
TRADEMARK_RE = re.compile(r'(\b\w+[®™])')

# Model numbers
MODEL_RE = re.compile(r'\b([A-Z]{1,4}\d{2,4}[A-Z]?)\b')

# "including X, Y, Z" patterns
INCLUDING_RE = re.compile(r'includ(?:es?|ing)\s+([A-Z][^,\.]{2,40})', re.IGNORECASE)
```

---

### Data Structures

#### ExtractedRow (`extraction/core.py`)

```python
@dataclass
class ExtractedRow:
    item: str           # "iPhone", "Azure"
    segment: str        # "Product", "Intelligent Cloud"
    value: int          # 209586000000 (base units)
    row_type: str       # "revenue_item" | "total" | "subtotal" | "adjustment"
    dimension: str      # "product_service" | "segment" | "end_market"
```

#### ExtractionResult (`extraction/core.py`)

```python
@dataclass
class ExtractionResult:
    year: int                              # 2024
    dimension: str                         # "product_service"
    rows: List[ExtractedRow]               # All extracted rows
    table_total: Optional[int]             # Total from table's own row
    segment_revenues: Dict[str, int]       # {"iPhone": 209586000000}
    adjustment_revenues: Dict[str, int]    # {"Hedging": -1500000000}
```

#### Chunk (`rag/chunking.py`)

```python
@dataclass
class Chunk:
    chunk_id: str           # "chunk_0042"
    text: str               # 800 chars of 10-K text
    section: str            # "item1" | "note_segment" | "other"
    heading: Optional[str]  # "Segment Information"
    char_range: tuple       # (12000, 12800)
    is_toc: bool           # True if in Table of Contents
```

#### DescriptionResult (`rag/generation.py`)

```python
@dataclass
class DescriptionResult:
    revenue_line: str              # "Compute"
    description: str               # "1-2 sentences"
    products_services_list: list   # ["DGX", "H100"]
    evidence_chunk_ids: list       # ["chunk_0142"]
    evidence_quotes: list          # ["exact text..."]
    retrieval_tier: str            # "tier1_local" | "tier2_full"
    validated: bool                # True if quotes verified
    evidence_gate_passed: bool     # True if quality gate passed
```

---

## Running the Pipeline

### Basic Usage

```bash
# CSV1 only (fastest)
python -m revseg.pipeline --tickers MSFT,AAPL,GOOGL --csv1-only

# With RAG-based descriptions (better for NVDA-style filings)
python -m revseg.pipeline --tickers MSFT,AAPL,GOOGL --csv1-only --use-rag

# Full output (CSV1 + CSV2 + CSV3)
python -m revseg.pipeline --tickers MSFT,AAPL,GOOGL
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--tickers` | (required) | Comma-separated tickers |
| `--out-dir` | `data/outputs` | Output directory |
| `--csv1-only` | `false` | Skip CSV2/CSV3 |
| `--use-rag` | `false` | Use RAG for descriptions |
| `--model-fast` | `gpt-4.1-mini` | Model for volume tasks |
| `--model-quality` | `gpt-4.1` | Model for descriptions |

### Output Files

```
data/
├── outputs/
│   ├── csv1_segment_revenue.csv      # Primary output
│   ├── csv2_segment_descriptions.csv # (if not --csv1-only)
│   ├── csv3_segment_items.csv        # (if not --csv1-only)
│   └── run_report.json               # Execution summary
├── artifacts/{ticker}/
│   ├── scout.json                    # Document structure
│   ├── disagg_layout.json            # Table layout
│   ├── disagg_extracted.json         # Raw extraction
│   ├── csv1_line_descriptions.json   # Descriptions
│   ├── csv1_desc_provenance.json     # Provenance tracking (source, evidence)
│   ├── {ticker}_csv1_desc_coverage.json  # (if --use-rag)
│   └── trace.jsonl                   # Debug trace
└── embeddings/{ticker}/              # (if --use-rag)
    ├── full.faiss                    # FAISS index
    └── metadata.json                 # Chunk metadata
```

### Provenance Artifact (`csv1_desc_provenance.json`)

**Phase 4 addition**: Each ticker now produces a provenance artifact tracking the source of each description:

```json
{
  "ticker": "AMZN",
  "company_name": "AMAZON COM INC",
  "fiscal_year": 2024,
  "line_provenance": [
    {
      "revenue_line": "Online stores (1)",
      "description": "We leverage our retail infrastructure...",
      "source_section": "table_footnote_regex",
      "evidence_snippet": "Includes product sales and digital media content...",
      "footnote_id": "1",
      "table_id": "t0069"
    },
    {
      "revenue_line": "AWS",
      "description": "AWS offers a broad set of on-demand technology services...",
      "source_section": "llm_table_context",
      "evidence_snippet": "[TABLE CONTEXT] ...",
      "footnote_id": null,
      "table_id": "t0069"
    }
  ]
}
```

**Source sections** (priority order):
1. `table_footnote_dom` - DOM-extracted footnotes (most reliable)
2. `table_footnote_regex` - Regex-extracted footnotes
3. `heading_based_item1` - Label found as heading in Item 1
4. `heading_based_html_heading` - Label found as heading elsewhere
5. `llm_table_context` - LLM extracted from table context
6. `llm_item1` - LLM extracted from Item 1 (Business)
7. `llm_item8` - LLM extracted from Item 8 (Notes)
8. `tier1_local` / `tier2_full` - RAG retrieval tiers (if `--use-rag`)

---

## Performance

| Mode | 1 Ticker | 6 Tickers | LLM Calls |
|------|----------|-----------|-----------|
| `--csv1-only` | ~25 sec | ~2.5 min | ~4-5/ticker |
| `--csv1-only --use-rag` | ~45 sec | ~4 min | ~5-6/ticker + embeddings |
| Full | ~50 sec | ~5 min | ~8-10/ticker |

### Latest Results (v18 - Post P0/P1 with RAG)

| Ticker | Coverage | Lines | Mode | Notes |
|--------|----------|-------|------|-------|
| AAPL | **100%** | 5/5 | RAG | Services description incomplete (missing sub-categories) |
| MSFT | **100%** | 10/10 | RAG | Full coverage, high quality descriptions |
| GOOGL | 83.3% | 5/6 | RAG | "Google Network" missing (needs AdMob/AdSense query) |
| AMZN | 71.4% | 5/7 | RAG | Subscription services, Other missing (footnote extraction) |
| META | **100%** | **3/3** | RAG | **Fixed with P1**: Other revenue now has description |
| NVDA | **0%** | 0/6 | RAG | **TABLE EXTRACTION BUG**: Labels showing as $ amounts |

### P1 Fixes Implemented

1. **Note 2 Classification** (`rag/chunking.py`):
   - `classify_note_revenue_chunk()` separates `note_revenue_sources` from `note_revenue_recognition`
   - Blocks accounting mechanics from retrieval

2. **Definitional Chunk Boost** (`rag/generation.py`):
   - `_boost_definitional_chunks()` re-ranks chunks with BOTH label AND definition patterns
   - Fixed META "Other revenue" (was generic, now finds "consists of WhatsApp Business Platform...")

3. **Blocked Sections** (`rag/index.py`):
   - `BLOCKED_SECTIONS = {'note_revenue_recognition'}`
   - `PREFERRED_SECTIONS` removed `item7` (MD&A has performance drivers, not definitions)

### Known Gaps (Require Future Development)

| Issue | Affected | Root Cause | Proposed Fix | Effort |
|-------|----------|------------|--------------|--------|
| **NVDA label extraction** | NVDA | Layout inference picks wrong column as label | Validate label_col is not numeric | Low |
| **Subscription/Other footnotes** | AMZN | Footnotes (5) and (6) not retrieved | Enhance DOM footnote parsing | Medium |
| **Google Network** | GOOGL | Query too generic for AdMob/AdSense | Add domain terms to query | Low |
| **Services richness** | AAPL | Only partial capture of 6 sub-categories | Sub-heading aggregation | Medium |

See `docs/DEV_PROPOSAL.md` for detailed fix plan.

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Wrong table selected | Gate patterns too permissive | Add patterns to `table_kind.py` |
| Validation fails | Missing adjustment rows | Add to `is_adjustment_item()` in mappings |
| Double-counting | Subtotal included | Add to `is_subtotal_row()` in mappings |
| Empty descriptions | Footnotes not found | Enable `--use-rag` |
| Low RAG coverage | Thresholds too high | Adjust in `rag/index.py` |
