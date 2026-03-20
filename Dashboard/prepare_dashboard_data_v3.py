#!/usr/bin/env python3
"""
Prepare dashboard data from pipeline output. v0.3

Changes in v0.3:
- Extracts source table HTML from filing using table_id, with row-level highlight injection
- Loads description provenance (source section + evidence snippet) from csv1_desc_provenance.json
- Creates annotated filing HTML with evidence anchor spans for reactive scroll
- New dashboard_data.json fields: table_id, provenance per line, evidence_anchor_ids

Usage:
    python prepare_dashboard_data_v3.py --input data/regression_90pct --output dashboard_data.json

Requires:
    pip install beautifulsoup4 lxml
"""

import argparse
import json
import os
import re
import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    from bs4 import BeautifulSoup, Tag
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("Warning: beautifulsoup4 not installed. Table extraction will be limited.")


# CIK mappings (same as v2, trimmed for brevity — extend as needed)
CIK_MAP = {
    "AAPL": "0000320193", "MSFT": "0000789019", "GOOGL": "0001652044",
    "AMZN": "0001018724", "META": "0001326801", "NVDA": "0001045810",
    "TSLA": "0001318605", "JPM": "0000019617", "V": "0001403161",
    "MA": "0001141391", "JNJ": "0000200406", "WMT": "0000104169",
    "PG": "0000080424", "XOM": "0000034088", "HD": "0000354950",
    "BAC": "0000070858", "CVX": "0000093410", "ABBV": "0001551152",
    "KO": "0000021344", "PFE": "0000078003", "COST": "0000909832",
    "MRK": "0000310158", "AVGO": "0001730168", "LLY": "0000059478",
    "PEP": "0000077476", "MCD": "0000063908", "TMO": "0000097745",
    "CSCO": "0000858877", "AMD": "0000002488", "ABT": "0000001800",
    "ORCL": "0001341439", "ACN": "0001467373", "DHR": "0000313616",
    "MU": "0000723125",
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILINGS_DIR = _PROJECT_ROOT / "data" / "filings"


def get_edgar_url(ticker: str) -> str:
    cik = CIK_MAP.get(ticker.upper(), ticker)
    return f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=include&count=10"


def _anchor_id(line_name: str) -> str:
    """Deterministic anchor ID from a line name."""
    slug = re.sub(r"[^a-z0-9]+", "_", line_name.lower()).strip("_")
    return f"ev_{slug}"


# ═══════════════════════════════════════════════════════════════════════
# Artifact Loaders
# ═══════════════════════════════════════════════════════════════════════

def get_table_id_from_artifacts(artifacts_dir: Path, ticker: str) -> Optional[str]:
    """Read accepted table_id from pipeline artifacts."""
    t_art = artifacts_dir / ticker

    # Primary: disagg_choice.json
    choice_path = t_art / "disagg_choice.json"
    if choice_path.exists():
        try:
            data = json.loads(choice_path.read_text(encoding="utf-8"))
            tid = data.get("table_id")
            if tid:
                return str(tid)
        except Exception:
            pass

    # Fallback: trace.jsonl
    trace_path = t_art / "trace.jsonl"
    if trace_path.exists():
        try:
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                if entry.get("stage") == "table_selected" or entry.get("stage") == "disagg_select":
                    tid = entry.get("table_id") or (entry.get("choice", {}).get("table_id"))
                    if tid:
                        return str(tid)
        except Exception:
            pass

    return None


def load_provenance(artifacts_dir: Path, ticker: str) -> Dict[str, Dict[str, Any]]:
    """Load description provenance per revenue line.

    Returns: {line_name: {source, evidence_snippet, footnote_id, evidence_anchor_id}}
    """
    t_art = artifacts_dir / ticker
    prov_path = t_art / "csv1_desc_provenance.json"
    if not prov_path.exists():
        return {}

    try:
        data = json.loads(prov_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for entry in data.get("line_provenance", []):
        line = entry.get("revenue_line", "")
        if not line:
            continue
        result[line] = {
            "source": entry.get("source_section") or entry.get("source", ""),
            "evidence_snippet": entry.get("evidence_snippet", ""),
            "footnote_id": entry.get("footnote_id"),
            "evidence_anchor_id": _anchor_id(line),
        }
    return result


# ═══════════════════════════════════════════════════════════════════════
# Source Table Extraction + Row Highlight Injection
# ═══════════════════════════════════════════════════════════════════════

def _normalize_label(text: str) -> str:
    """Normalize a label for fuzzy matching: lowercase, strip whitespace/parens/footnotes."""
    t = re.sub(r"\s+", " ", text).strip().lower()
    t = re.sub(r"\(\d+\)", "", t).strip()  # Remove footnote markers like (1)
    t = re.sub(r"[^\w\s]", "", t).strip()  # Remove punctuation
    return t


def extract_source_table_html(
    filing_html: str,
    table_id: str,
    extracted_lines: List[str],
) -> str:
    """Extract the source table from filing HTML and inject row highlights.

    Args:
        filing_html: Full 10-K HTML content
        table_id: The accepted table ID (e.g., "t0042")
        extracted_lines: List of revenue line names to highlight

    Returns:
        HTML string of the table with `data-line` attributes and highlight class on matched rows.
        Empty string if table not found.
    """
    if not BS4_AVAILABLE or not filing_html or not table_id:
        return ""

    soup = BeautifulSoup(filing_html, "lxml")

    # Pipeline table IDs are positional (e.g. "t0009" = 10th table, 0-indexed)
    table = soup.find("table", id=table_id)
    if not table:
        idx_match = re.match(r"t(\d+)", table_id)
        if idx_match:
            idx = int(idx_match.group(1))
            all_tables = soup.find_all("table")
            if idx < len(all_tables):
                table = all_tables[idx]
                table["id"] = table_id
    if not table:
        return ""

    # Build normalized lookup of extracted lines
    norm_lines = {_normalize_label(l): l for l in extracted_lines if l}

    # Walk rows, inject data-line + highlight class on matches
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue

        # Get text of first non-empty cell (label column)
        label_text = ""
        for cell in cells:
            txt = cell.get_text(strip=True)
            if txt and len(txt) > 1:
                label_text = txt
                break

        if not label_text:
            continue

        norm = _normalize_label(label_text)

        # Try exact match first, then substring match
        matched_line = norm_lines.get(norm)
        if not matched_line:
            for norm_key, orig in norm_lines.items():
                if norm_key in norm or norm in norm_key:
                    matched_line = orig
                    break

        if matched_line:
            tr["data-line"] = matched_line
            tr["class"] = tr.get("class", []) + ["hl-extracted-row"]

    # Return just the table HTML
    return str(table)


# ═══════════════════════════════════════════════════════════════════════
# Table ID Injection (pipeline IDs → HTML id attributes)
# ═══════════════════════════════════════════════════════════════════════

def _inject_table_id(filing_html: str, table_id: str) -> str:
    """Inject the pipeline table_id as an HTML id attribute on the target table.

    Pipeline IDs like 't0009' correspond to the 10th <table> (0-indexed).
    """
    if not BS4_AVAILABLE or not table_id:
        return filing_html

    idx_match = re.match(r"t(\d+)", table_id)
    if not idx_match:
        return filing_html

    idx = int(idx_match.group(1))
    soup = BeautifulSoup(filing_html, "lxml")
    all_tables = soup.find_all("table")
    if idx >= len(all_tables):
        return filing_html

    all_tables[idx]["id"] = table_id
    return str(soup)


# ═══════════════════════════════════════════════════════════════════════
# Evidence Anchor Injection into Filing HTML
# ═══════════════════════════════════════════════════════════════════════

def inject_evidence_anchors(
    filing_html: str,
    provenance: Dict[str, Dict[str, Any]],
) -> Tuple[str, Dict[str, bool]]:
    """Inject <span id="ev_xxx"> anchors around evidence passages in the filing HTML.

    Uses a simple text search approach: find the first ~80 chars of each evidence
    snippet in the HTML (plain text extracted) and wrap the match in a span.

    Args:
        filing_html: Full 10-K HTML
        provenance: {line_name: {evidence_snippet, evidence_anchor_id, ...}}

    Returns:
        (annotated_html, {anchor_id: found_bool})
    """
    if not filing_html or not provenance:
        return filing_html, {}

    anchor_status: Dict[str, bool] = {}
    annotated = filing_html

    for line_name, prov in provenance.items():
        snippet = prov.get("evidence_snippet", "")
        anchor_id = prov.get("evidence_anchor_id", "")
        if not snippet or not anchor_id or len(snippet) < 30:
            anchor_status[anchor_id] = False
            continue

        # Clean the snippet for search: take first ~100 chars, strip tags/markers
        search_text = re.sub(r"\[ITEM\d+\]|\[TABLE CONTEXT\]|\[ITEM8\]", "", snippet)
        search_text = re.sub(r"<[^>]+>", "", search_text).strip()

        # Use first 80 chars as search needle (more reliable than full snippet)
        needle = search_text[:80].strip()
        if len(needle) < 25:
            anchor_status[anchor_id] = False
            continue

        # Escape for regex but allow flexible whitespace
        escaped = re.escape(needle)
        # Allow flexible whitespace (HTML may have different spacing)
        pattern = re.sub(r"\\ ", r"\\s+", escaped)

        try:
            match = re.search(pattern, annotated, re.IGNORECASE)
            if match:
                # Wrap the match in an anchor span
                original = match.group(0)
                replacement = f'<span id="{anchor_id}" class="evidence-anchor">{original}</span>'
                # Only replace first occurrence
                annotated = annotated[:match.start()] + replacement + annotated[match.end():]
                anchor_status[anchor_id] = True
            else:
                anchor_status[anchor_id] = False
        except re.error:
            anchor_status[anchor_id] = False

    return annotated, anchor_status


# ═══════════════════════════════════════════════════════════════════════
# CSV / Run Report Loaders (carried from v2)
# ═══════════════════════════════════════════════════════════════════════

def load_run_report(input_dir: Path) -> Dict[str, Any]:
    report_path = input_dir / "run_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def parse_revenue(val: str) -> Optional[float]:
    if not val:
        return None
    try:
        cleaned = re.sub(r"[,$\s]", "", str(val))
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def load_csv1_data(input_dir: Path) -> Dict[str, List[Dict]]:
    csv_path = input_dir / "csv1_segment_revenue.csv"
    if not csv_path.exists():
        return {}
    ticker_data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("Ticker", "").strip()
            if not ticker:
                continue
            if ticker not in ticker_data:
                ticker_data[ticker] = []
            revenue_val = (
                row.get("Revenue (FY2024, $m)") or row.get("Revenue (FY2025, $m)")
                or row.get("Revenue ($m)") or row.get("Revenue") or ""
            )
            ticker_data[ticker].append({
                "segment": row.get("Revenue Group (Reportable Segment)", "").strip(),
                "line": row.get("Revenue Line", "").strip(),
                "description": row.get("Line Item description (company language)", "").strip(),
                "revenue": parse_revenue(revenue_val),
                "fiscal_year": row.get("Fiscal Year", ""),
            })
    return ticker_data


def load_trace_data(input_dir: Path) -> Dict[str, Dict[str, Any]]:
    trace_path = input_dir / "trace.jsonl"
    if not trace_path.exists():
        return {}
    ticker_info = {}
    current_ticker = None
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("stage") == "start":
                    current_ticker = entry.get("ticker")
                    ticker_info[current_ticker] = {}
                if current_ticker and entry.get("stage") == "table_selected":
                    ticker_info[current_ticker]["table_id"] = entry.get("table_id")
                    ticker_info[current_ticker]["confidence"] = entry.get("confidence")
                if current_ticker and entry.get("stage") == "validation":
                    ticker_info[current_ticker]["delta_pct"] = entry.get("delta_pct")
                    ticker_info[current_ticker]["status"] = entry.get("status")
            except json.JSONDecodeError:
                continue
    return ticker_info


def extract_validation_metrics(
    run_report: Dict, trace_data: Dict, ticker: str, csv_lines: List[Dict],
) -> Dict[str, Any]:
    report_data = run_report.get("tickers", {}).get(ticker, {})
    trace_info = trace_data.get(ticker, {})
    validation = report_data.get("validation", {})
    calculated_sum = sum(l["revenue"] for l in csv_lines if l.get("revenue") is not None)

    ok = report_data.get("ok")
    status = "PASS" if ok is True else ("FAIL" if ok is False else
             (report_data.get("status") or trace_info.get("status") or "UNKNOWN"))

    table_id = (report_data.get("income_statement_table_id_guess")
                or report_data.get("table_id")
                or trace_info.get("table_id"))

    expected = (validation.get("external_total")
                or report_data.get("expected_total"))
    extracted = (validation.get("segment_sum")
                 or report_data.get("extracted_total"))

    delta_pct = report_data.get("delta_pct") or trace_info.get("delta_pct")
    if delta_pct is None and expected and extracted:
        try:
            delta_pct = round((extracted - expected) / expected * 100, 2)
        except ZeroDivisionError:
            pass

    return {
        "status": status,
        "extracted_total": extracted,
        "expected_total": expected,
        "delta_pct": delta_pct,
        "n_segments": len(csv_lines),
        "table_id": table_id,
        "confidence": trace_info.get("confidence"),
        "validation_notes": report_data.get("validation_notes", ""),
        "calculated_sum": calculated_sum,
    }


# ═══════════════════════════════════════════════════════════════════════
# Filing HTML Loader
# ═══════════════════════════════════════════════════════════════════════

def find_filing_html(ticker: str) -> Optional[Path]:
    """Find local filing HTML for a ticker."""
    for suffix in ["_10K.html", "_10-K.html", ".html"]:
        path = FILINGS_DIR / f"{ticker}{suffix}"
        if path.exists():
            return path
    return None


# ═══════════════════════════════════════════════════════════════════════
# Main Build
# ═══════════════════════════════════════════════════════════════════════

def build_dashboard_data(input_dir: Path) -> Dict[str, Any]:
    input_dir = Path(input_dir)
    print(f"Loading from: {input_dir}")

    run_report = load_run_report(input_dir)
    csv_data = load_csv1_data(input_dir)
    trace_data = load_trace_data(input_dir)

    tickers = list(csv_data.keys())
    if not tickers and run_report:
        tickers = list(run_report.get("tickers", {}).keys())

    print(f"Found {len(tickers)} tickers: {', '.join(sorted(tickers))}")

    # Detect artifacts directory
    artifacts_dirs = [
        input_dir.parent / ".artifacts",
        input_dir / ".artifacts_v2",
        input_dir / "artifacts",
        input_dir.parent / ".artifacts_v2",
    ]
    artifacts_dir = None
    for ad in artifacts_dirs:
        if ad.exists():
            artifacts_dir = ad
            print(f"Artifacts dir: {artifacts_dir}")
            break

    if not artifacts_dir:
        print("Warning: No artifacts directory found. Table extraction + provenance unavailable.")

    dashboard_data = {
        "generated_at": str(input_dir.name),
        "input_dir": str(input_dir),
        "tickers": {},
        "summary": {"total": len(tickers), "passed": 0, "failed": 0},
    }

    annotated_count = 0
    table_found_count = 0
    provenance_count = 0

    for ticker in sorted(tickers):
        csv_lines = csv_data.get(ticker, [])
        metrics = extract_validation_metrics(run_report, trace_data, ticker, csv_lines)
        line_names = [l.get("line", "") for l in csv_lines if l.get("line")]

        # ── Get table_id ──
        table_id = None
        if artifacts_dir:
            table_id = get_table_id_from_artifacts(artifacts_dir, ticker)
        if not table_id:
            table_id = metrics.get("table_id")
        metrics["table_id"] = table_id

        # ── Load provenance ──
        provenance = {}
        if artifacts_dir:
            provenance = load_provenance(artifacts_dir, ticker)
            if provenance:
                provenance_count += 1

        # ── Merge provenance into line items ──
        for line_item in csv_lines:
            ln = line_item.get("line", "")
            prov = provenance.get(ln, {})
            line_item["provenance"] = {
                "source": prov.get("source", ""),
                "evidence_snippet": prov.get("evidence_snippet", ""),
                "evidence_anchor_id": prov.get("evidence_anchor_id", _anchor_id(ln)),
                "footnote_id": prov.get("footnote_id"),
            }

        # ── Extract source table + create annotated filing ──
        source_table_html = ""
        filing_path = find_filing_html(ticker)

        if filing_path and table_id:
            filing_html = filing_path.read_text(encoding="utf-8", errors="ignore")

            # Extract and highlight the source table
            source_table_html = extract_source_table_html(filing_html, table_id, line_names)
            if source_table_html:
                table_found_count += 1
                print(f"  [{ticker}] Source table {table_id} extracted ({len(source_table_html):,} chars)")

            # Build annotated filing: inject table ID + evidence anchors
            annotated_html = _inject_table_id(filing_html, table_id)
            save_annotated = annotated_html != filing_html

            if provenance:
                annotated_html, anchor_status = inject_evidence_anchors(annotated_html, provenance)
                anchors_found = sum(1 for v in anchor_status.values() if v)
                total_anchors = len(anchor_status)
                if anchors_found > 0:
                    save_annotated = True
                    print(f"  [{ticker}] Evidence anchors: {anchors_found}/{total_anchors} injected")
                else:
                    print(f"  [{ticker}] Evidence anchors: 0/{total_anchors} matched")
            else:
                print(f"  [{ticker}] No provenance data")

            if save_annotated:
                annotated_path = FILINGS_DIR / f"{ticker}_10K_annotated.html"
                annotated_path.write_text(annotated_html, encoding="utf-8")
                annotated_count += 1
                print(f"  [{ticker}] Annotated filing saved -> {annotated_path.name}")
        elif not filing_path:
            print(f"  [{ticker}] No local filing HTML found")
        elif not table_id:
            print(f"  [{ticker}] No table_id in artifacts")

        dashboard_data["tickers"][ticker] = {
            "lines": csv_lines,
            "table_id": table_id,
            "source_table_html": source_table_html,
            "edgar_url": get_edgar_url(ticker),
            "metrics": metrics,
        }

        if metrics.get("status") == "PASS":
            dashboard_data["summary"]["passed"] += 1
        elif metrics.get("status") == "FAIL":
            dashboard_data["summary"]["failed"] += 1

    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Tickers: {len(tickers)}")
    print(f"  Source tables found: {table_found_count}/{len(tickers)}")
    print(f"  Provenance loaded: {provenance_count}/{len(tickers)}")
    print(f"  Annotated filings: {annotated_count}/{len(tickers)}")
    print(f"  Pipeline pass: {dashboard_data['summary']['passed']}")
    print(f"  Pipeline fail: {dashboard_data['summary']['failed']}")

    return dashboard_data


def main():
    global FILINGS_DIR
    parser = argparse.ArgumentParser(description="Prepare dashboard data v0.3")
    parser.add_argument("--input", "-i", required=True, help="Pipeline output directory")
    parser.add_argument("--output", "-o", default=str(_PROJECT_ROOT / "dashboard_data.json"), help="Output JSON file")
    parser.add_argument("--filings-dir", default=str(FILINGS_DIR), help="Local filings cache dir")

    args = parser.parse_args()
    FILINGS_DIR = Path(args.filings_dir)

    print("=" * 50)
    print("Revenue Dashboard Data Prep v0.3")
    print("=" * 50)

    dashboard_data = build_dashboard_data(args.input)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)

    print(f"\nDashboard data written to: {args.output}")
    print(f"\nNext: streamlit run dashboard_v5.py")


if __name__ == "__main__":
    main()
