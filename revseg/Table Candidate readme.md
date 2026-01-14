# Stage 3 (Part 1): Table Candidate Extraction from 10-K HTML

## Objective

This stage prepares for automated extraction of reported business lines / revenue lines
(e.g., Microsoft segments, Apple net sales by product, Google revenues by type).

Because 10-K table layouts and headings vary significantly across issuers, we first build
a robust, auditable pipeline that:

1. Extracts all HTML tables from the downloaded 10-K primary document.
2. Captures surrounding context (nearby headings and prose) for each table.
3. Serializes the resulting “candidate table set” into a compact JSON file.

Downstream (next step), an LLM will review the JSON candidates and select:
- which table contains the desired revenue lines
- which year column is the target year
- which rows are the line items vs subtotals
- which row is the total (for computing percentages)

Numeric extraction and CSV creation will remain deterministic to preserve fidelity.

---

## Files

### `table_candidates.py`
Provides functions to:

- Locate the latest downloaded filing folder for a ticker:
  - `find_latest_downloaded_filing_dir(base_dir, ticker)`

- Locate the primary HTML document:
  - `find_primary_document_html(filing_dir)`

- Extract table candidates (Step 1):
  - `extract_table_candidates_from_html(html_path, preview_rows, preview_cols, ...)`

Each candidate includes:
- a stable table id (`t0000`, `t0001`, ...)
- dimensions (#rows, #cols)
- a top-left preview grid of cell text
- detected years (e.g., 2023, 2024)
- keyword hits (e.g., “revenue”, “net sales”, “year ended”)
- heading context and nearby text context
- the source HTML path

- Serialize candidates to JSON (Step 2):
  - `write_candidates_json(candidates, out_path)`

### Notebook usage
Run the notebook cells to create:
- `data/table_candidates/MSFT_table_candidates.json`

---

## Requirements

Install:
- `beautifulsoup4`
- `lxml`

Optional but recommended:
- `pandas` (not required for this stage as implemented)

Example:
```bash
pip install beautifulsoup4 lxml
