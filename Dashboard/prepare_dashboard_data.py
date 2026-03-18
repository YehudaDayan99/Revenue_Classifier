#!/usr/bin/env python3
"""
Prepare dashboard data from pipeline output.

Usage:
    python prepare_dashboard_data.py --input data/regression_90pct --output dashboard_data.json
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_run_report(input_dir: Path) -> Dict[str, Any]:
    """Load run_report.json from pipeline output."""
    report_path = input_dir / "run_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_csv1_data(input_dir: Path) -> Dict[str, List[Dict]]:
    """Load extracted data from csv1_segment_revenue.csv."""
    import csv
    
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
            ticker_data[ticker].append({
                "segment": row.get("Revenue Group (Reportable Segment)", "").strip(),
                "line": row.get("Revenue Line", "").strip(),
                "description": row.get("Line Item description (company language)", "").strip(),
                "revenue": parse_revenue(row.get("Revenue (FY2024, $m)", row.get("Revenue ($m)", ""))),
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


def load_trace_data(input_dir: Path) -> Dict[str, List[Dict]]:
    """Load trace.jsonl for validation details."""
    trace_path = input_dir / "trace.jsonl"
    if not trace_path.exists():
        return {}
    
    ticker_traces = {}
    current_ticker = None
    
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                # Track which ticker we're processing
                if entry.get("stage") == "start":
                    current_ticker = entry.get("ticker")
                if current_ticker:
                    if current_ticker not in ticker_traces:
                        ticker_traces[current_ticker] = []
                    ticker_traces[current_ticker].append(entry)
            except json.JSONDecodeError:
                continue
    
    return ticker_traces


def load_html_tables(input_dir: Path, tickers: List[str]) -> Dict[str, str]:
    """Load original HTML tables from artifacts."""
    tables = {}
    artifacts_dir = input_dir / ".artifacts_v2"
    
    if not artifacts_dir.exists():
        artifacts_dir = input_dir / "artifacts"
    
    for ticker in tickers:
        # Look for table HTML file
        patterns = [
            f"{ticker}_selected_table.html",
            f"{ticker}_t*.html",
            f"{ticker}_table.html",
        ]
        for pattern in patterns:
            matches = list(artifacts_dir.glob(pattern)) if artifacts_dir.exists() else []
            if matches:
                with open(matches[0], "r", encoding="utf-8", errors="ignore") as f:
                    tables[ticker] = f.read()
                break
        
        if ticker not in tables:
            tables[ticker] = "<p><em>Original table HTML not found in artifacts</em></p>"
    
    return tables


def extract_validation_metrics(run_report: Dict, ticker: str) -> Dict[str, Any]:
    """Extract validation metrics for a ticker from run report."""
    ticker_data = run_report.get("tickers", {}).get(ticker, {})
    
    return {
        "status": ticker_data.get("status", "UNKNOWN"),
        "extracted_total": ticker_data.get("extracted_total"),
        "expected_total": ticker_data.get("expected_total"),
        "delta_pct": ticker_data.get("delta_pct"),
        "n_segments": ticker_data.get("n_segments", 0),
        "table_id": ticker_data.get("table_id"),
        "validation_notes": ticker_data.get("validation_notes", ""),
    }


def build_dashboard_data(input_dir: Path) -> Dict[str, Any]:
    """Build complete dashboard data structure."""
    input_dir = Path(input_dir)
    
    # Load all data sources
    run_report = load_run_report(input_dir)
    csv_data = load_csv1_data(input_dir)
    
    # Get ticker list
    tickers = list(csv_data.keys())
    if not tickers and run_report:
        tickers = list(run_report.get("tickers", {}).keys())
    
    # Load HTML tables
    html_tables = load_html_tables(input_dir, tickers)
    
    # Build per-ticker data
    dashboard_data = {
        "generated_at": str(Path(input_dir).name),
        "input_dir": str(input_dir),
        "tickers": {},
        "summary": {
            "total": len(tickers),
            "passed": 0,
            "failed": 0,
            "pending_review": len(tickers),
        },
        "review_state": {},  # Will be populated by dashboard
    }
    
    for ticker in sorted(tickers):
        metrics = extract_validation_metrics(run_report, ticker)
        
        # Calculate totals from extracted data
        extracted_lines = csv_data.get(ticker, [])
        extracted_sum = sum(
            line["revenue"] for line in extracted_lines 
            if line["revenue"] is not None
        )
        
        dashboard_data["tickers"][ticker] = {
            "lines": extracted_lines,
            "html_table": html_tables.get(ticker, ""),
            "metrics": {
                **metrics,
                "calculated_sum": extracted_sum,
            },
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
    
    print(f"Loading data from: {args.input}")
    dashboard_data = build_dashboard_data(args.input)
    
    print(f"Found {len(dashboard_data['tickers'])} tickers")
    print(f"  Passed: {dashboard_data['summary']['passed']}")
    print(f"  Failed: {dashboard_data['summary']['failed']}")
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
    
    print(f"Dashboard data written to: {args.output}")
    print(f"\nNext: streamlit run dashboard.py")


if __name__ == "__main__":
    main()
