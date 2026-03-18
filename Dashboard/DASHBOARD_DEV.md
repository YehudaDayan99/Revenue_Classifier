# Revenue Validator Dashboard

## Purpose
Manual QA interface for revenue extraction pipeline output. Reviewer cycles through tickers, compares extracted data vs original SEC 10-K filings, approves/rejects with notes.

## Current State (v0.2)
- **Streamlit app**: `dashboard.py` (~300 lines)
- **Data prep**: `prepare_dashboard_data.py`
- **Persistence**: JSON files (no database)

### Working Features
- Ticker dropdown with status icons (✅/❌/⏳)
- Side-by-side: extracted table vs original 10-K
- Metrics cards (delta %, totals, pipeline status)
- Approve/Reject/Notes with persistent state
- Summary view with color-coded table
- Prev/Next navigation
- Export review state to JSON

## Data Schema

### dashboard_data.json
```json
{
  "tickers": {
    "NVDA": {
      "lines": [
        {
          "line": "Compute",
          "revenue": 102196,
          "segment": "Product/Service disclosure",
          "description": "Our Compute & Networking segment includes..."
        }
      ],
      "html_table": "<table>...</table>",
      "edgar_url": "https://www.sec.gov/Archives/edgar/data/1045810/...",
      "filing_html_path": "data/filings/NVDA_10K.html",
      "metrics": {
        "status": "PASS",
        "delta_pct": 0.0,
        "calculated_sum": 130497,
        "expected_total": 130497000000,
        "table_id": "t0009"
      }
    }
  },
  "summary": {
    "total": 20,
    "passed": 18,
    "failed": 2
  }
}
```

### review_state.json
```json
{
  "NVDA": {
    "status": "approved",
    "notes": "Clean extraction, all segments match",
    "reviewed_at": "2024-01-15T10:30:00"
  }
}
```

## Tech Stack
- Python 3.11
- Streamlit 1.30+
- Pandas
- No database — JSON files for state
- Runs locally on Windows

## File Locations
```
Revenue Classifier/
├── Dashboard/
│   ├── dashboard.py                  # Main app (v0.1)
│   ├── dashboard_v2.py               # v0.2 app (full 10-K viewer, clickable lines)
│   ├── prepare_dashboard_data.py     # Data prep (v0.1)
│   ├── prepare_dashboard_data_v2.py  # Data prep v0.2 (EDGAR URLs, etc.)
│   ├── cache_filings_for_dashboard.py # Option B: populate data/filings/
│   └── DASHBOARD_DEV.md              # This file
├── dashboard_data.json               # Generated from pipeline output
├── review_state.json                 # Persistent review state
└── data/
    ├── regression_90pct/             # Pipeline output
    │   ├── csv1_segment_revenue.csv
    │   ├── run_report.json
    │   └── trace.jsonl
    ├── 10k/                          # Pipeline-downloaded 10-Ks (per ticker/date)
    └── filings/                      # Option B: flat 10-K HTML for dashboard
        ├── AAPL_10K.html
        ├── NVDA_10K.html
        └── ...
```

## Backlog (prioritized)

### P0 — Current Sprint
- [x] Basic side-by-side view
- [x] Approve/Reject/Notes
- [ ] **Full 10-K viewer** — scrollable HTML/PDF of actual filing, not just extracted table
- [ ] **Clickable line items** — click revenue line to show its description

### P1 — Next Sprint
- [ ] EDGAR URL link — "Open in SEC" button per ticker
- [ ] Keyboard shortcuts (j/k navigate, a/r approve/reject)
- [ ] Filter by review status (pending/approved/rejected)
- [ ] Highlight extracted table in full 10-K context

### P2 — Future
- [ ] Diff view for re-runs (compare two extractions)
- [ ] Batch approve/reject
- [ ] Description quality score (length, keyword coverage)
- [ ] Export to Excel for stakeholder review
- [ ] Section navigation (jump to Item 8, Note 2, etc.)
- [ ] Side-by-side filing comparison (YoY)

## Integration Points

### Pipeline Output Required
The dashboard reads from pipeline output directory:
- `csv1_segment_revenue.csv` — extracted line items
- `run_report.json` — validation metrics per ticker
- `trace.jsonl` — (optional) detailed extraction trace

### EDGAR Integration
To enable full 10-K viewing:
1. Pipeline saves `edgar_url` per ticker in run_report.json
2. Or: **Option B** — cache filing HTML to `data/filings/{ticker}_10K.html` (see below)
3. Dashboard embeds local HTML or shows "Open in SEC EDGAR" button

---

## Full 10-K Viewing: Option B (Local cache)

**Option A: EDGAR link (zero setup)**  
In the dashboard, click "Open in SEC EDGAR" to open the filing in your browser.

**Option B: Local cache (best UX)**  
Cache 10-K HTML so the dashboard can embed the full filing inline (scrollable, no leave-to-browser).

### 1. Create filings directory (if missing)
```powershell
mkdir data\filings
```

### 2. Populate cache

From **project root** (`Revenue Classifier/`):

```powershell
# Use tickers from dashboard_data.json; copy from existing data/10k only (no download)
python Dashboard/cache_filings_for_dashboard.py

# Same, but also download from EDGAR for any ticker not already in data/10k
python Dashboard/cache_filings_for_dashboard.py --tickers NVDA,AAPL,MSFT,GOOGL

# Only copy from data/10k (never call EDGAR)
python Dashboard/cache_filings_for_dashboard.py --from-10k-only

# Overwrite existing cached files
python Dashboard/cache_filings_for_dashboard.py --force
```

If the script downloads from EDGAR, set `SEC_USER_AGENT` first (required by SEC):
```powershell
$env:SEC_USER_AGENT = "RevenueClassifier/1.0 (your.email@example.com)"
python Dashboard/cache_filings_for_dashboard.py --tickers AAPL,NVDA
```

Saved files: `data/filings/AAPL_10K.html`, `data/filings/NVDA_10K.html`, etc.  
The v0.2 dashboard looks for these and embeds them in the "Full 10-K Filing" tab when present.

---

## Usage

```powershell
# 1. Prep data from pipeline output
python Dashboard/prepare_dashboard_data_v2.py --input data/regression_90pct --output dashboard_data.json

# 2. (Optional) Populate local 10-K cache for full filing view
python Dashboard/cache_filings_for_dashboard.py

# 3. Launch dashboard (v0.2)
python -m streamlit run Dashboard/dashboard_v2.py
```

## Design Decisions

1. **JSON over SQLite** — Simpler for single-user local workflow, easy to inspect/edit
2. **No React/JS** — Streamlit-only for fast iteration, Claude can modify easily
3. **Artifact-based** — Reads pipeline artifacts directly, no separate data pipeline
4. **Stateless prep** — `prepare_dashboard_data.py` is idempotent, re-run anytime

## Known Issues

1. Original table HTML not loading — artifact path patterns may not match
2. Expected total showing "—" — run_report.json schema mismatch
3. Status showing "UNKNOWN" — metrics extraction needs fixing
