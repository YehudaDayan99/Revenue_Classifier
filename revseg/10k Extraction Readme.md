# 10k Extraction Description

## Purpose

This module implements **Stage 1** of the Revenue Segmentation product: **programmatically collecting the latest 10-K filing** for a list of US-traded public companies from **SEC EDGAR**.

It performs three core tasks:

1. **Resolve** each input ticker (e.g., MSFT) to its SEC **CIK** identifier.
2. **Locate** the most recent 10-K filing (optionally including amendments 10-K/A).
3. **Download** the filing’s primary document (typically HTML) and associated metadata into a deterministic folder structure.

The output of Stage 1 is a set of downloaded filings that become the input to Stage 2 (segment extraction).

---

## SEC Compliance Requirements

SEC requests that automated clients provide a descriptive **User-Agent** including contact information.

This code enforces that via the environment variable:

- `SEC_USER_AGENT="YourApp/0.1 (your.email@domain.com)"`

If it is missing or does not contain an “@”, the module raises an error.

---

## Data Sources (SEC Endpoints)

1. **Ticker → CIK mapping**
   - `https://www.sec.gov/files/company_tickers.json`

2. **Company submissions JSON (filings index)**
   - `https://data.sec.gov/submissions/CIK##########.json`

3. **Filing primary document**
   - `https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodashes}/{primaryDocument}`

4. **Filing folder index (optional but useful)**
   - `https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodashes}/index.json`

---

## Output Layout

For each ticker, the module creates a directory:

`<out_dir>/<TICKER>/<filingDate>_<accessionNoDashes>/`

and writes:
- `primary_document.html` (or `primary_document.<ext>` depending on source)
- `filing_ref.json` (normalized metadata for the selected filing)
- `submission.json` (SEC submissions JSON, cached for provenance)
- `filing_index.json` (listing of all files in the submission folder; may fail without stopping the run)

Example:
- `data/10k/MSFT/2024-07-30_000119312524186446/primary_document.html`

---

## Module: revseg.sec_edgar

### Data Structures

#### `FilingRef`
A dataclass that captures a specific filing selection:
- `ticker`: input ticker
- `cik`: numeric CIK
- `form`: “10-K” or “10-K/A”
- `accession_number`: dashed accession number
- `filing_date`: YYYY-MM-DD
- `primary_document`: filename of the primary document

It also exposes computed properties:
- `cik10`: zero-padded 10-digit CIK
- `accession_no_dashes`
- `sec_doc_url`: fully-qualified URL to download the primary document
- `sec_index_url`: fully-qualified URL for `index.json` within the filing folder

---

### Functions

#### `_sec_user_agent() -> str`
Reads and validates `SEC_USER_AGENT`.
Raises `SecEdgarError` if missing or not compliant.

#### `_session() -> requests.Session`
Creates a `requests.Session` with SEC-compliant headers including `User-Agent`.

#### `_sleep_rate_limit(min_interval_s, last_call_ts)`
A small rate limiter that ensures at least `min_interval_s` seconds between SEC requests.

#### `fetch_ticker_cik_map(cache_path=None, min_interval_s=0.2) -> dict[str,int]`
Downloads `company_tickers.json` and returns a mapping of `TICKER -> CIK`.
If `cache_path` exists, loads it instead and avoids an external call.

#### `get_company_submissions(cik, min_interval_s=0.2, cache_dir=None) -> dict`
Downloads the company submissions JSON for a given CIK.
If cached at `<cache_dir>/CIK##########.json`, reads from disk instead.

#### `select_latest_10k(ticker, cik, submissions, include_amendments=False) -> FilingRef`
Scans the “recent filings” block in the submissions JSON and selects the first match:
- default: “10-K”
- if `include_amendments=True`: “10-K” or “10-K/A”

#### `download_latest_10k(ticker, out_dir, include_amendments=False, cache_dir=None, min_interval_s=0.2) -> Path`
End-to-end operation for a single ticker:
1. resolve CIK
2. fetch submissions
3. select latest 10-K
4. create output folder
5. write metadata files
6. download `index.json` (non-fatal on failure)
7. download primary document (fatal on failure)

Returns the folder path for the downloaded filing.

#### `download_many_latest_10k(tickers, out_dir, include_amendments=False, cache_dir=None, min_interval_s=0.2) -> dict[str,(bool,str)]`
Runs `download_latest_10k()` across multiple tickers.
Returns a status dictionary where each ticker maps to:
- `(True, <folder-path>)` on success
- `(False, <error message>)` on failure

---

## Operational Notes / Extensions

- Add retry/backoff for transient 429/5xx responses.
- Persist HTTP status and response snippets for auditability.
- Add “as-of date” controls (e.g., select filing by filing_date cutoff).
- Optionally download all documents listed in `filing_index.json` (Exhibits, XBRL, etc.).
