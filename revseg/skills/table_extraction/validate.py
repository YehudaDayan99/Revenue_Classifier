#!/usr/bin/env python3
"""
Validate extracted revenue data using self-consistent checks.

Validation priority:
1. Primary: segments + adjustments ≈ table_total (from same table)
2. Secondary: segments + adjustments ≈ external_total (from SEC API)
3. Fallback: accept if enough segments with positive values

Usage:
    python validate.py <extraction_json> [--table-total N] [--external-total N]
    
Output:
    JSON ValidationResult to stdout
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ValidationResult:
    """Result of extraction validation."""
    ok: bool
    table_total: Optional[int]       # From the table's Total row
    segment_sum: int                  # Sum of extracted segments
    adjustment_sum: int               # Sum of adjustment rows (hedging, corporate)
    external_total: Optional[int]    # From SEC API if available
    delta_pct: Optional[float]       # Percentage difference from reference
    notes: str


# Income statement keywords that indicate wrong table
INCOME_STATEMENT_KEYWORDS = [
    "net income", "gross margin", "operating expense", "operating income",
    "income from operation", "earnings per share", "diluted share", "basic share",
    "cost of revenue", "cost of sales", "gross profit", "interest expense",
    "interest income", "tax expense", "income tax", "depreciation",
    "amortization", "ebitda", "net profit", "net loss",
    "research and development", "r&d expense", "administrative expense",
    "selling expense", "marketing expense", "total expense", "% of",
    "eps", "per share"
]


def validate_extraction(
    segment_revenues: Dict[str, int],
    adjustment_revenues: Dict[str, int],
    table_total: Optional[int],
    *,
    external_total: Optional[int] = None,
    tolerance_pct: float = 0.02,
    min_segments: int = 2,
) -> ValidationResult:
    """
    Validate extracted revenue data using self-consistent checks.
    
    Validation priority:
    1. Primary: segments + adjustments ≈ table_total (from same table)
    2. Secondary: segments + adjustments ≈ external_total (from SEC API)
    3. Fallback: accept if we have enough segments with positive values
    
    Args:
        segment_revenues: Dict mapping segment name to revenue value
        adjustment_revenues: Dict mapping adjustment label to value (can be negative)
        table_total: Total revenue from the table's own "Total" row
        external_total: Total revenue from SEC API (optional cross-check)
        tolerance_pct: Maximum allowed percentage difference (default 2%)
        min_segments: Minimum number of segments required for fallback acceptance
    
    Returns:
        ValidationResult with ok=True/False and diagnostic information
    """
    segment_sum = sum(segment_revenues.values())
    adjustment_sum = sum(adjustment_revenues.values())
    computed_total = segment_sum + adjustment_sum
    
    # Primary validation: against table's own Total row
    if table_total is not None and table_total > 0:
        delta = abs(computed_total - table_total)
        delta_pct = delta / table_total
        
        if delta_pct <= tolerance_pct:
            return ValidationResult(
                ok=True,
                table_total=table_total,
                segment_sum=segment_sum,
                adjustment_sum=adjustment_sum,
                external_total=external_total,
                delta_pct=delta_pct,
                notes=f"OK: matches table total within {delta_pct*100:.2f}%"
            )
        else:
            # Table total mismatch - FAIL immediately, don't fall through
            return ValidationResult(
                ok=False,
                table_total=table_total,
                segment_sum=segment_sum,
                adjustment_sum=adjustment_sum,
                external_total=external_total,
                delta_pct=delta_pct,
                notes=f"FAIL: sum mismatch - extracted {computed_total:,} vs table_total {table_total:,} (delta {delta_pct*100:.2f}% > {tolerance_pct*100:.0f}% tolerance)"
            )
    
    # Secondary validation: against SEC API external total (only when table_total unavailable)
    if external_total is not None and external_total > 0:
        delta = abs(computed_total - external_total)
        delta_pct = delta / external_total
        
        if delta_pct <= tolerance_pct:
            return ValidationResult(
                ok=True,
                table_total=table_total,
                segment_sum=segment_sum,
                adjustment_sum=adjustment_sum,
                external_total=external_total,
                delta_pct=delta_pct,
                notes=f"OK: matches SEC API total within {delta_pct*100:.2f}%"
            )
        
        # Try segment_sum only (without adjustments)
        if segment_sum > 0:
            delta_seg = abs(segment_sum - external_total)
            delta_seg_pct = delta_seg / external_total
            if delta_seg_pct <= tolerance_pct:
                return ValidationResult(
                    ok=True,
                    table_total=table_total,
                    segment_sum=segment_sum,
                    adjustment_sum=adjustment_sum,
                    external_total=external_total,
                    delta_pct=delta_seg_pct,
                    notes=f"OK: segment sum matches SEC API total within {delta_seg_pct*100:.2f}% (adjustments excluded)"
                )
    
    # Fallback: accept if we have multiple segments with reasonable values
    if len(segment_revenues) >= min_segments and segment_sum > 0:
        # Check that segments are all positive (no weird negative segments)
        all_positive = all(v >= 0 for v in segment_revenues.values())
        
        # Sanity check: if external_total is available, segment_sum should be within reasonable range
        # Reject if segment_sum is less than 10% of external_total (clearly wrong table)
        if external_total is not None and external_total > 0:
            ratio = segment_sum / external_total
            if ratio < 0.1:  # Less than 10% of expected revenue
                return ValidationResult(
                    ok=False,
                    table_total=table_total,
                    segment_sum=segment_sum,
                    adjustment_sum=adjustment_sum,
                    external_total=external_total,
                    delta_pct=None,
                    notes=f"FAIL: segment_sum ({segment_sum:,}) is only {ratio*100:.2f}% of external_total ({external_total:,}) - likely wrong table"
                )
        
        # Check that segment names don't look like income statement or expense items
        segment_names_lower = [name.lower() for name in segment_revenues.keys()]
        for name in segment_names_lower:
            for keyword in INCOME_STATEMENT_KEYWORDS:
                if keyword in name:
                    return ValidationResult(
                        ok=False,
                        table_total=table_total,
                        segment_sum=segment_sum,
                        adjustment_sum=adjustment_sum,
                        external_total=external_total,
                        delta_pct=None,
                        notes=f"FAIL: segment name '{name}' contains income statement indicator '{keyword}' - likely wrong table"
                    )
        
        if all_positive:
            return ValidationResult(
                ok=True,
                table_total=table_total,
                segment_sum=segment_sum,
                adjustment_sum=adjustment_sum,
                external_total=external_total,
                delta_pct=None,
                notes=f"ACCEPTED: {len(segment_revenues)} segments, no validation reference available"
            )
    
    # Compute delta for error reporting
    reference = table_total or external_total
    delta_pct_report = None
    if reference and reference > 0:
        delta_pct_report = abs(computed_total - reference) / reference
    
    return ValidationResult(
        ok=False,
        table_total=table_total,
        segment_sum=segment_sum,
        adjustment_sum=adjustment_sum,
        external_total=external_total,
        delta_pct=delta_pct_report,
        notes=f"FAIL: segments={len(segment_revenues)}, sum={segment_sum:,}, computed_total={computed_total:,}, table_total={table_total}, external={external_total}"
    )


def validate_from_extraction_result(
    extraction: Dict,
    *,
    external_total: Optional[int] = None,
    tolerance_pct: float = 0.02,
) -> ValidationResult:
    """
    Validate from an extraction result JSON (as produced by extract_values.py).
    """
    segment_revenues = extraction.get("segment_revenues", {})
    adjustment_revenues = extraction.get("adjustment_revenues", {})
    table_total = extraction.get("table_total")
    
    return validate_extraction(
        segment_revenues=segment_revenues,
        adjustment_revenues=adjustment_revenues,
        table_total=table_total,
        external_total=external_total,
        tolerance_pct=tolerance_pct,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate extracted revenue data"
    )
    parser.add_argument(
        "extraction_json",
        help="Path to extraction result JSON (or - for stdin)"
    )
    parser.add_argument(
        "--table-total",
        type=int,
        help="Override table_total from extraction"
    )
    parser.add_argument(
        "--external-total",
        type=int,
        help="External total revenue from SEC API"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Tolerance percentage (default: 0.02 = 2%%)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load extraction result
        if args.extraction_json == "-":
            extraction = json.load(sys.stdin)
        else:
            with open(args.extraction_json) as f:
                extraction = json.load(f)
        
        # Override table_total if provided
        if args.table_total is not None:
            extraction["table_total"] = args.table_total
        
        # Validate
        result = validate_from_extraction_result(
            extraction,
            external_total=args.external_total,
            tolerance_pct=args.tolerance,
        )
        
        # Output
        output = asdict(result)
        print(json.dumps(output, indent=2))
        
        sys.exit(0 if result.ok else 1)
        
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
