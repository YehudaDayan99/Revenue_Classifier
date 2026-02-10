#!/usr/bin/env python3
"""Check what forms Saudi Aramco has filed."""

import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from sec_edgar_client import fetch_ticker_cik_map, get_company_submissions

def main():
    ticker = "SAR"
    
    print(f"Checking filings for {ticker}...\n")
    
    try:
        ticker_cik = fetch_ticker_cik_map()
        cik = ticker_cik[ticker]
        print(f"CIK for {ticker}: {cik}\n")
        
        subs = get_company_submissions(cik)
        recent = subs.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        
        form_counts = Counter(forms)
        print("Forms filed (most recent first):")
        for form, count in form_counts.most_common(10):
            print(f"  {form}: {count} filings")
        
        print("\nMost recent 10 filings:")
        for i, form in enumerate(forms[:10], 1):
            print(f"  {i}. {form}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
