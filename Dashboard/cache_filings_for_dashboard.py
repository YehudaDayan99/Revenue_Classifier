#!/usr/bin/env python3
"""
Option B: Local cache — populate data/filings/ for the dashboard Full 10-K viewer.

1. Creates data/filings/ if missing.
2. For each ticker: copy from existing data/10k/{TICKER}/.../primary_document.html
   if present; otherwise download 10-K from EDGAR and save as data/filings/{TICKER}_10K.html.

Usage (from project root):
    # Cache all tickers from dashboard_data.json (copy from 10k only; no download)
    python Dashboard/cache_filings_for_dashboard.py

    # Cache specific tickers, download from EDGAR if not in data/10k
    python Dashboard/cache_filings_for_dashboard.py --tickers AAPL,NVDA,TSLA

    # Force re-copy / re-download
    python Dashboard/cache_filings_for_dashboard.py --force

    # Only copy from existing data/10k (never call EDGAR)
    python Dashboard/cache_filings_for_dashboard.py --from-10k-only

Requires SEC_USER_AGENT if downloading (e.g. RevenueClassifier/1.0 (you@email.com)).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def _project_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root


def _tickers_from_dashboard_data(data_path: Path) -> list[str]:
    if not data_path.exists():
        return []
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("tickers", {}).keys())


def _latest_10k_folder(ticker: str, base_10k: Path) -> Path | None:
    """Return path to the latest 10-K folder under data/10k/{TICKER}/."""
    ticker_dir = base_10k / ticker
    if not ticker_dir.is_dir():
        return None
    subdirs = [d for d in ticker_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    # Sort by folder name (date_accession, e.g. 2025-02-26_000104581025000023) descending
    subdirs.sort(key=lambda p: p.name, reverse=True)
    primary = subdirs[0] / "primary_document.html"
    return subdirs[0] if primary.exists() else None


def copy_from_10k(ticker: str, base_10k: Path, filings_dir: Path) -> bool:
    """Copy primary_document.html from data/10k to data/filings/{TICKER}_10K.html. Return True if copied."""
    folder = _latest_10k_folder(ticker, base_10k)
    if not folder:
        return False
    src = folder / "primary_document.html"
    dest = filings_dir / f"{ticker}_10K.html"
    shutil.copy2(src, dest)
    return True


def download_and_cache(
    ticker: str,
    filings_dir: Path,
    base_10k: Path,
    cache_dir: Path,
) -> bool:
    """Download 10-K via revseg.sec_edgar and save to data/filings/{TICKER}_10K.html. Return True if ok."""
    try:
        from revseg.sec_edgar import download_latest_10k
    except ImportError:
        print("  revseg.sec_edgar not available; run from project root with revseg on PYTHONPATH", file=sys.stderr)
        return False

    try:
        folder = download_latest_10k(
            ticker,
            base_10k,
            cache_dir=cache_dir,
            min_interval_s=0.2,
        )
        src = folder / "primary_document.html"
        dest = filings_dir / f"{ticker}_10K.html"
        if src.exists():
            shutil.copy2(src, dest)
            return True
    except Exception as e:
        print(f"  Download failed: {e}", file=sys.stderr)
        return False
    return False


def main():
    ap = argparse.ArgumentParser(description="Populate data/filings/ for dashboard Option B (local 10-K cache)")
    ap.add_argument("--input", default="dashboard_data.json", help="Path to dashboard_data.json (for ticker list)")
    ap.add_argument("--tickers", type=str, help="Comma-separated tickers (overrides --input tickers)")
    ap.add_argument("--from-10k-only", action="store_true", help="Only copy from data/10k; do not download from EDGAR")
    ap.add_argument("--force", action="store_true", help="Overwrite existing data/filings/{TICKER}_10K.html")
    ap.add_argument("--out-dir", default=None, help="Filings output dir (default: data/filings)")
    ap.add_argument("--10k-dir", dest="tenk_dir", default=None, help="Existing 10-K root (default: data/10k)")
    args = ap.parse_args()

    root = _project_root()
    filings_dir = Path(args.out_dir) if args.out_dir else root / "data" / "filings"
    base_10k = Path(args.tenk_dir) if args.tenk_dir else root / "data" / "10k"
    cache_dir = root / ".cache" / "sec"
    data_file = root / args.input if not Path(args.input).is_absolute() else Path(args.input)

    filings_dir.mkdir(parents=True, exist_ok=True)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = _tickers_from_dashboard_data(data_file)

    if not tickers:
        print("No tickers to process. Use --tickers AAPL,NVDA,... or ensure dashboard_data.json has tickers.", file=sys.stderr)
        sys.exit(1)

    print(f"Cache target: {filings_dir}")
    print(f"10-K source: {base_10k}")
    print(f"Tickers: {len(tickers)}")
    if args.from_10k_only:
        print("Mode: from-10k-only (no EDGAR download)")
    print()

    ok = 0
    skipped = 0
    failed = []

    for ticker in tickers:
        dest = filings_dir / f"{ticker}_10K.html"
        if dest.exists() and not args.force:
            print(f"  {ticker}: already cached (use --force to overwrite)")
            skipped += 1
            continue

        if copy_from_10k(ticker, base_10k, filings_dir):
            print(f"  {ticker}: copied from data/10k -> {dest.name}")
            ok += 1
            continue

        if args.from_10k_only:
            print(f"  {ticker}: not found in data/10k (skip; remove --from-10k-only to download)")
            failed.append(ticker)
            continue

        if download_and_cache(ticker, filings_dir, base_10k, cache_dir):
            print(f"  {ticker}: downloaded and cached -> {dest.name}")
            ok += 1
        else:
            failed.append(ticker)
            print(f"  {ticker}: failed")

    print()
    print(f"Done: {ok} cached, {skipped} skipped, {len(failed)} failed")
    if failed:
        print("Failed:", ", ".join(failed))
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
