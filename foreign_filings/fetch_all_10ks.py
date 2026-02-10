#!/usr/bin/env python3
"""Fetch latest 10-K filings for list of companies, skipping existing ones."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sec_edgar_client import download_latest_filing

# Mapping of company names to ticker symbols
COMPANY_TICKERS = {
    "NVIDIA": "NVDA",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Alphabet (Google)": "GOOGL",
    "Amazon": "AMZN",
    "Meta Platforms": "META",
    "Berkshire Hathaway": "BRK-B",
    "Eli Lilly": "LLY",
    "Broadcom": "AVGO",
    "Tesla": "TSLA",
    "JPMorgan Chase": "JPM",
    "Visa": "V",
    "Walmart": "WMT",
    "ExxonMobil": "XOM",
    "UnitedHealth": "UNH",
    "Mastercard": "MA",
    "Procter & Gamble": "PG",
    "Costco": "COST",
    "Oracle": "ORCL",
    "Johnson & Johnson": "JNJ",
    "Home Depot": "HD",
    "Bank of America": "BAC",
    "AbbVie": "ABBV",
    "Netflix": "NFLX",
    "Chevron": "CVX",
    "Coca-Cola": "KO",
    "AMD": "AMD",
    "Adobe": "ADBE",
    "Reliance Industries": "RIL",
    "Salesforce": "CRM",
    "PepsiCo": "PEP",
    "Merck & Co.": "MRK",
    "Accenture": "ACN",
    "Danaher": "DHR",
    "McDonald's": "MCD",
    "Abbott Labs": "ABT",
    "Thermo Fisher": "TMO",
    "Cisco Systems": "CSCO",
    "Intel": "INTC",
    "Wells Fargo": "WFC",
    "Walt Disney": "DIS",
    "Qualcomm": "QCOM",
    "Amgen": "AMGN",
    "IBM": "IBM",
    "Union Pacific": "UNP",
    "Honeywell": "HON",
    "General Electric": "GE",
    "Morgan Stanley": "MS",
    "Caterpillar": "CAT",
    "Comcast": "CMCSA",
    "Pinduoduo (PDD)": "PDD",
    "Goldman Sachs": "GS",
    "Intuitive Surgical": "ISRG",
    "Texas Instruments": "TXN",
    "Booking Holdings": "BKNG",
    "ServiceNow": "NOW",
    "American Express": "AXP",
    "Lowe's": "LOW",
    "UPS": "UPS",
    "ConocoPhillips": "COP",
    "Philip Morris": "PM",
    "RTX (Raytheon)": "RTX",
    "Uber": "UBER",
    "Nike": "NKE",
    "BlackRock": "BLK",
    "HDFC Bank": "HDB",
    "T-Mobile US": "TMUS",
    "Bristol-Myers": "BMY",
    "Medtronic": "MDT",
    "Prologis": "PLD",
    "Starbucks": "SBUX",
    "Eaton": "ETN",
    "Applied Materials": "AMAT",
    "Lockheed Martin": "LMT",
    "Stryker": "SYK",
    "Lam Research": "LRCX",
}

def main():
    output_dir = Path(__file__).parent.parent / "data" / "10k"
    cache_dir = Path(__file__).parent / "cache"
    
    # Check which companies already have filings
    already_downloaded = set()
    for ticker_dir in output_dir.glob("*"):
        if ticker_dir.is_dir():
            already_downloaded.add(ticker_dir.name)
    
    # Get list of tickers to download
    tickers_to_download = []
    skipped = []
    
    for company_name, ticker in sorted(COMPANY_TICKERS.items()):
        if ticker in already_downloaded:
            skipped.append(f"{company_name} ({ticker})")
        else:
            tickers_to_download.append((company_name, ticker))
    
    print(f"Companies already downloaded ({len(skipped)}):")
    for name in skipped:
        print(f"  ✓ {name}")
    
    print(f"\nCompanies to download ({len(tickers_to_download)}):")
    for name, ticker in tickers_to_download:
        print(f"  - {name} ({ticker})")
    
    if not tickers_to_download:
        print("\n✓ All companies already have 10-K filings!")
        return 0
    
    print(f"\n{'='*60}")
    print(f"Downloading {len(tickers_to_download)} 10-K filings...")
    print(f"{'='*60}\n")
    
    results = {}
    for idx, (company_name, ticker) in enumerate(tickers_to_download, 1):
        print(f"[{idx}/{len(tickers_to_download)}] Processing {ticker} ({company_name})...")
        try:
            filing_path = download_latest_filing(
                ticker=ticker,
                out_dir=output_dir,
                form_type="10-K",
                cache_dir=cache_dir,
                min_interval_s=0.5,
            )
            print(f"  ✓ Success: {filing_path.name}\n")
            results[ticker] = (True, str(filing_path))
        except Exception as e:
            print(f"  ✗ Failed: {type(e).__name__}: {e}\n")
            results[ticker] = (False, str(e))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful = [t for t, (ok, _) in results.items() if ok]
    failed = [t for t, (ok, _) in results.items() if not ok]
    
    print(f"\nSuccessfully downloaded: {len(successful)}")
    for ticker in sorted(successful):
        print(f"  ✓ {ticker}")
    
    if failed:
        print(f"\nFailed to download: {len(failed)}")
        for ticker in sorted(failed):
            print(f"  ✗ {ticker}")
    
    print(f"\nTotal already available: {len(already_downloaded)}")
    
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())
