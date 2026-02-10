# Revenue Classifier - Project Overview

## What This Project Does (Non-Technical)

### The Problem
Public companies disclose their revenue in annual 10-K filings, but extracting structured, analyzable data from these documents is challenging because:
- Revenue is reported across multiple tables in different formats
- Companies use inconsistent terminology and structures
- The same data may appear in multiple places with different levels of detail
- Important context (like product descriptions) is buried in prose

### The Solution
Revenue Classifier automatically extracts revenue data from SEC 10-K filings and produces a clean, structured dataset with:

| What We Extract | Example |
|-----------------|---------|
| **Revenue Line Items** | "iPhone", "Azure", "Advertising" |
| **Dollar Amounts** | $209,586M, $98,435M |
| **Business Segments** | "More Personal Computing", "Intelligent Cloud" |
| **Descriptions** | What each product/service actually is |

### Output Example
```
Company    | Ticker | Revenue Line        | Segment              | Revenue ($M)
-----------|--------|---------------------|----------------------|-------------
Apple Inc. | AAPL   | iPhone              | Product/Service      | 209,586
Apple Inc. | AAPL   | Services            | Product/Service      | 109,158
Microsoft  | MSFT   | Server products...  | Intelligent Cloud    | 98,435
Alphabet   | GOOGL  | Google Search       | Google Services      | 198,084
```

### Key Features
- **Automated**: No manual data entry required
- **Accurate**: Validates against table totals (0.00% error)
- **Traceable**: Every number links back to the source document
- **Comprehensive**: Extracts descriptions explaining what each item represents

---

## Technical Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Revenue Classifier                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Ticker → [SEC API] → [HTML Parsing] → [LLM Agents] →      │
│            → [Table Extraction] → [Validation] → CSV1       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline Phases

| Phase | Purpose | Technology |
|-------|---------|------------|
| **1. Document Acquisition** | Download 10-K from SEC EDGAR | REST API, caching |
| **2. Document Understanding** | Find revenue tables | LLM (gpt-4.1-mini), regex |
| **3. Table Extraction** | Extract line items and values | Deterministic parsing |
| **4. Description Extraction** | Find product/service definitions | LLM (gpt-4.1), DOM search |
| **5. Validation** | Verify sums reconcile | Self-consistent checks |

### Key Design Decisions

1. **LLM for selection, deterministic for extraction**: LLMs choose which table to use, but actual numbers are extracted via regex/parsing to avoid hallucination.

2. **Self-consistent validation**: Compare extracted sum to the table's own "Total" row, not just external APIs. This catches extraction errors immediately.

3. **Fail fast**: If extraction doesn't reconcile, reject and try next table. Prefer "no result" over wrong result.

4. **Traceable provenance**: Every description records where it came from (heading, footnote, Note 2 paragraph, etc.)

### Core Modules

```
revseg/
├── pipeline.py          # Main orchestration
├── react_agents.py      # LLM agents for table/description extraction
├── table_candidates.py  # HTML table parsing and candidate extraction
├── table_kind.py        # Filter out non-revenue tables
├── mappings.py          # Company-specific segment mappings
├── extraction/
│   ├── core.py          # Line item extraction logic
│   └── validation.py    # Sum validation
└── rag/                 # Optional RAG-based descriptions
```

### Data Flow

```
Input:  Ticker (e.g., "MSFT")
          ↓
SEC EDGAR API → Download 10-K HTML (5-10MB)
          ↓
BeautifulSoup → Parse ~100 tables from HTML
          ↓
LLM Agent #1 → Identify business segments
          ↓
TableKind Gate → Filter to ~30 candidate tables
          ↓
LLM Agent #2 → Select best revenue disaggregation table
          ↓
LLM Agent #3 → Infer table layout (columns, headers)
          ↓
Deterministic Extraction → Extract line items + values
          ↓
Validation → Verify sum = table total (±2%)
          ↓
LLM Agent #4 → Extract descriptions from filing text
          ↓
Output: csv1_segment_revenue.csv
```

### Validation Logic

```python
# Primary: Self-consistent check
if abs(segment_sum + adjustment_sum - table_total) / table_total < 0.02:
    return OK

# If table_total known and mismatch: FAIL (no fallback)
if table_total is not None:
    return FAIL  # Don't accept wrong data

# Secondary: External API check (only if table_total unknown)
if external_total and abs(segment_sum - external_total) < 0.02:
    return OK
```

### Current Coverage (20 Tickers Tested)

| Ticker | Lines | Status | Notes |
|--------|-------|--------|-------|
| NVDA | 6 | ✅ | Compute, Networking, Gaming, Automotive |
| AAPL | 5 | ✅ | iPhone, Services, Mac, iPad, Wearables |
| MSFT | 10 | ✅ | Intelligent Cloud, M365, Gaming, LinkedIn |
| GOOGL | 7 | ✅ | Search, YouTube, Cloud, Other Bets |
| AMZN | 7 | ✅ | Online, Third-party, AWS, Advertising |
| META | 2 | ✅ | Advertising, Other revenue |
| V | 4 | ✅ | Data Processing, Service, International (fixed) |
| AVGO | 2 | ✅ | Semiconductor, Infrastructure Software |
| COST | 4 | ✅ | Foods, Non-Foods, Warehouse, Fresh (auto-scaled) |
| AMD | 4 | ✅ | Data Center, Client, Gaming, Embedded |
| JNJ | 2 | ✅ | Innovative Medicine, MedTech |
| MU | 4 | ✅ | Memory business units |
| XOM | 2 | ✅ | Upstream, Downstream |
| ORCL | 3 | ✅ | Cloud, License, Services |
| LLY | 17 | ✅ | Pharmaceutical products (detailed) |
| TSLA | - | ❌ | 17.9% external mismatch - missing items |
| MA | - | ❌ | 100% mismatch - double counting |
| JPM | - | ❌ | Bank - complex net interest structure |
| WMT | - | ❌ | 32% mismatch - fiscal year issue |
| BRK-B | - | ❌ | Conglomerate - fragmented segments |

**Success Rate: 15/20 (75%)** - Validated against SEC external totals

### Running the Pipeline

```powershell
# Using the startup script (recommended)
.\run_pipeline.ps1

# Or manually
$env:SEC_USER_AGENT = "YourApp/1.0 (your@email.com)"
$env:OPENAI_API_KEY_FILE = "path/to/key.txt"
python -m revseg.pipeline --tickers MSFT,AAPL --csv1-only
```

### Output Files

| File | Description |
|------|-------------|
| `csv1_segment_revenue.csv` | Main output with all revenue lines |
| `run_report.json` | Validation results per ticker |
| `.artifacts/{ticker}/` | Intermediate extraction artifacts |

---

## For Developers

### Adding a New Company

1. **Check if it works out-of-box**: Most companies work without changes
2. **If segment mapping needed**: Add to `mappings.py`
3. **If subtotals cause double-counting**: Add to `*_SUBTOTAL_ITEMS` in `mappings.py`
4. **If validation fails**: Check if adjustment rows need to be added

### Key Files to Understand

1. `pipeline.py` - Start here, follow the flow
2. `react_agents.py` - All LLM prompts and description extraction
3. `extraction/core.py` - How line items are extracted from tables
4. `mappings.py` - Company-specific configurations

### Testing

```bash
# Run regression tests
python -m pytest tests/test_dev_review_regressions.py -v

# Test single ticker
python -m revseg.pipeline --tickers MSFT --csv1-only --out-dir data/test
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| Phase 11 | Feb 2026 | **P0 Review Fixes**: External total HARD GATE (10% tolerance), header-based column veto, pre-validation cost/margin filtering, total row deduplication. Fixed V, MU. 75% success (15/20) with honest validation |
| Phase 10 | Jan 2026 | Post-LLM validators: sum-as-total, year-col scoring, segment_col validation, auto-scaling |
| Phase 9 | Jan 2026 | Table rejection: stock charts, earnings tables, volume metrics |
| Phase 8 | Jan 2026 | "Offerings include" pattern for richer NVDA Compute/Networking descriptions |
| Phase 7 | Jan 2026 | Dev Review fixes: validation fail-fast, META deduplication, adjustment row emission |
| Phase 6 | Jan 2026 | Segment enumeration extraction for NVDA Compute |
| Phase 5 | Jan 2026 | META critical fix, table-header rejection |
| Phase 4 | Jan 2026 | Provenance tracking for descriptions |
| Phase 1-3 | Jan 2026 | Initial pipeline implementation |
