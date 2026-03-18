#!/usr/bin/env python3
"""
Prepare dashboard data from pipeline output. v0.2

Changes in v0.2:
- Adds EDGAR URL per ticker
- Better artifact path detection
- Loads from trace.jsonl for table_id

Usage:
    python prepare_dashboard_data.py --input data/regression_90pct --output dashboard_data.json
"""

import argparse
import json
import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional


# CIK mappings for common tickers (extend as needed)
CIK_MAP = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "META": "0001326801",
    "NVDA": "0001045810",
    "TSLA": "0001318605",
    "JPM": "0000019617",
    "V": "0001403161",
    "MA": "0001141391",
    "JNJ": "0000200406",
    "WMT": "0000104169",
    "PG": "0000080424",
    "XOM": "0000034088",
    "HD": "0000354950",
    "BAC": "0000070858",
    "CVX": "0000093410",
    "ABBV": "0001551152",
    "KO": "0000021344",
    "PFE": "0000078003",
    "COST": "0000909832",
    "MRK": "0000310158",
    "AVGO": "0001730168",
    "LLY": "0000059478",
    "PEP": "0000077476",
    "MCD": "0000063908",
    "TMO": "0000097745",
    "CSCO": "0000858877",
    "AMD": "0000002488",
    "ABT": "0000001800",
    "ORCL": "0001341439",
    "ACN": "0001467373",
    "DHR": "0000313616",
    "MU": "0000723125",
    "BRK-B": "0001067983",
    "BRK.B": "0001067983",
}


def get_edgar_url(ticker: str) -> str:
    """Generate EDGAR 10-K search URL for ticker."""
    cik = CIK_MAP.get(ticker.upper(), ticker)
    return f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=include&count=10"


def load_run_report(input_dir: Path) -> Dict[str, Any]:
    """Load run_report.json from pipeline output."""
    report_path = input_dir / "run_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_csv1_data(input_dir: Path) -> Dict[str, List[Dict]]:
    """Load extracted data from csv1_segment_revenue.csv."""
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
            
            # Try multiple possible column names for revenue
            revenue_val = (
                row.get("Revenue (FY2024, $m)") or 
                row.get("Revenue ($m)") or 
                row.get("Revenue") or
                ""
            )
            
            ticker_data[ticker].append({
                "segment": row.get("Revenue Group (Reportable Segment)", "").strip(),
                "line": row.get("Revenue Line", "").strip(),
                "description": row.get("Line Item description (company language)", "").strip(),
                "revenue": parse_revenue(revenue_val),
                "fiscal_year": row.get("Fiscal Year", "2024"),
            })
    return ticker_data


def parse_revenue(val: str) -> Optional[float]:
    """Parse revenue value from string."""
    if not val:
        return None
    try:
        # Remove commas, $, spaces
        cleaned = re.sub(r"[,$\s]", "", str(val))
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def load_trace_data(input_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load trace.jsonl for table_id and other metadata."""
    trace_path = input_dir / "trace.jsonl"
    if not trace_path.exists():
        return {}
    
    ticker_info = {}
    current_ticker = None
    
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                
                # Track which ticker we're processing
                if entry.get("stage") == "start":
                    current_ticker = entry.get("ticker")
                    ticker_info[current_ticker] = {}
                
                # Capture table selection
                if current_ticker and entry.get("stage") == "table_selected":
                    ticker_info[current_ticker]["table_id"] = entry.get("table_id")
                    ticker_info[current_ticker]["confidence"] = entry.get("confidence")
                
                # Capture validation result
                if current_ticker and entry.get("stage") == "validation":
                    ticker_info[current_ticker]["delta_pct"] = entry.get("delta_pct")
                    ticker_info[current_ticker]["status"] = entry.get("status")
                    
            except json.JSONDecodeError:
                continue
    
    return ticker_info


def find_html_tables(input_dir: Path, tickers: List[str]) -> Dict[str, str]:
    """Find and load HTML tables from artifacts."""
    tables = {}
    
    # Check multiple possible artifact directories
    artifact_dirs = [
        input_dir / ".artifacts_v2",
        input_dir / "artifacts",
        input_dir / "tables",
        input_dir,  # Tables might be in root
    ]
    
    for ticker in tickers:
        # Try each directory
        for artifacts_dir in artifact_dirs:
            if not artifacts_dir.exists():
                continue
            
            # Try multiple naming patterns
            patterns = [
                f"{ticker}_selected_table.html",
                f"{ticker}_table.html",
                f"{ticker}_t*.html",
                f"{ticker.lower()}_*.html",
                f"*{ticker}*.html",
            ]
            
            for pattern in patterns:
                matches = list(artifacts_dir.glob(pattern))
                if matches:
                    # Take the first match
                    try:
                        with open(matches[0], "r", encoding="utf-8", errors="ignore") as f:
                            tables[ticker] = f.read()
                        break
                    except Exception as e:
                        print(f"Warning: Could not read {matches[0]}: {e}")
            
            if ticker in tables:
                break
        
        if ticker not in tables:
            tables[ticker] = ""
    
    return tables


def extract_validation_metrics(
    run_report: Dict, 
    trace_data: Dict,
    ticker: str,
    csv_lines: List[Dict]
) -> Dict[str, Any]:
    """Extract validation metrics for a ticker."""
    
    # From run_report
    report_data = run_report.get("tickers", {}).get(ticker, {})
    
    # From trace
    trace_info = trace_data.get(ticker, {})
    
    # Calculate from CSV
    calculated_sum = sum(
        line["revenue"] for line in csv_lines 
        if line.get("revenue") is not None
    )
    
    return {
        "status": report_data.get("status") or trace_info.get("status") or "UNKNOWN",
        "extracted_total": report_data.get("extracted_total"),
        "expected_total": report_data.get("expected_total"),
        "delta_pct": report_data.get("delta_pct") or trace_info.get("delta_pct"),
        "n_segments": len(csv_lines),
        "table_id": report_data.get("table_id") or trace_info.get("table_id"),
        "confidence": trace_info.get("confidence"),
        "validation_notes": report_data.get("validation_notes", ""),
        "calculated_sum": calculated_sum,
    }


def build_dashboard_data(input_dir: Path) -> Dict[str, Any]:
    """Build complete dashboard data structure."""
    input_dir = Path(input_dir)
    
    print(f"Loading from: {input_dir}")
    
    # Load all data sources
    run_report = load_run_report(input_dir)
    csv_data = load_csv1_data(input_dir)
    trace_data = load_trace_data(input_dir)
    
    # Get ticker list from CSV (primary) or run_report (fallback)
    tickers = list(csv_data.keys())
    if not tickers and run_report:
        tickers = list(run_report.get("tickers", {}).keys())
    
    print(f"Found {len(tickers)} tickers: {', '.join(tickers)}")
    
    # Load HTML tables
    html_tables = find_html_tables(input_dir, tickers)
    tables_found = sum(1 for t in html_tables.values() if t)
    print(f"Found HTML tables for {tables_found}/{len(tickers)} tickers")
    
    # Build per-ticker data
    dashboard_data = {
        "generated_at": str(input_dir.name),
        "input_dir": str(input_dir),
        "tickers": {},
        "summary": {
            "total": len(tickers),
            "passed": 0,
            "failed": 0,
            "pending_review": len(tickers),
        },
    }
    
    for ticker in sorted(tickers):
        csv_lines = csv_data.get(ticker, [])
        metrics = extract_validation_metrics(run_report, trace_data, ticker, csv_lines)
        
        dashboard_data["tickers"][ticker] = {
            "lines": csv_lines,
            "html_table": html_tables.get(ticker, ""),
            "edgar_url": get_edgar_url(ticker),
            "metrics": metrics,
        }
        
        if metrics.get("status") == "PASS":
            dashboard_data["summary"]["passed"] += 1
        elif metrics.get("status") == "FAIL":
            dashboard_data["summary"]["failed"] += 1
    
    return dashboard_data


def main():
    parser = argparse.ArgumentParser(description="Prepare dashboard data from pipeline output")
    parser.add_argument("--input", "-i", required=True, help="Pipeline output directory")
    parser.add_argument("--output", "-o", default="dashboard_data.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    print(f"=" * 50)
    print("Revenue Dashboard Data Prep v0.2")
    print(f"=" * 50)
    
    dashboard_data = build_dashboard_data(args.input)
    
    print(f"\nSummary:")
    print(f"  Total tickers: {dashboard_data['summary']['total']}")
    print(f"  Passed: {dashboard_data['summary']['passed']}")
    print(f"  Failed: {dashboard_data['summary']['failed']}")
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDashboard data written to: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. streamlit run dashboard.py")
    print(f"  2. (Optional) Cache 10-K filings to data/filings/{{TICKER}}_10K.html")


if __name__ == "__main__":
    main()
