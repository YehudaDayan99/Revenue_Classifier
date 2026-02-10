#!/usr/bin/env python3
"""Script to download Saudi Aramco's latest 20-F filing."""

import sys
from pathlib import Path

# Add parent directory to path so we can import the module
sys.path.insert(0, str(Path(__file__).parent))

from sec_edgar_client import download_latest_filing

def main():
    """Download Saudi Aramco's latest 20-F filing."""
    output_dir = Path(__file__).parent / "data" / "20f"
    cache_dir = Path(__file__).parent / "cache"
    
    print("Downloading Saudi Aramco (SAR) latest 20-F filing...\n")
    
    try:
        filing_path = download_latest_filing(
            ticker="SAR",
            out_dir=output_dir,
            form_type="20-F",
            cache_dir=cache_dir,
            min_interval_s=0.5,  # Respectful rate limiting for SEC API
        )
        
        print(f"\n✓ Successfully downloaded Saudi Aramco 20-F filing to:")
        print(f"  {filing_path}")
        print(f"\nFiles saved:")
        for file in sorted(filing_path.glob("*")):
            if file.is_file():
                size_kb = file.stat().st_size / 1024
                print(f"  - {file.name} ({size_kb:.1f} KB)")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Failed to download Saudi Aramco 20-F filing:")
        print(f"  {type(e).__name__}: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
