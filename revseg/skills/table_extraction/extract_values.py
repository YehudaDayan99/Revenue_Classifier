#!/usr/bin/env python3
"""
Extract revenue values from a parsed grid using layout inference.

Handles both AAPL-style (row-based) and MSFT-style (header-based) formats.
Classifies rows as: item, segment, subtotal, adjustment, total.

Usage:
    python extract_values.py <grid_json> <layout_json>
    
Output:
    JSON extraction result to stdout
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ExtractedRow:
    """A single extracted row from a revenue table."""
    segment: str           # Canonical segment/group name (or empty for adjustments)
    item: str              # Original row label
    value: int             # Revenue value in base units (USD)
    row_type: str          # "item" | "segment" | "subtotal" | "adjustment" | "total"
    year: int
    dimension: str = "segment"


@dataclass
class ExtractionResult:
    """Complete extraction result from a revenue table."""
    year: int
    rows: List[ExtractedRow]
    table_total: Optional[int]
    dimension: str = "segment"
    segment_revenues: Dict[str, int] = None
    adjustment_revenues: Dict[str, int] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.segment_revenues is None:
            self.segment_revenues = {}
        if self.adjustment_revenues is None:
            self.adjustment_revenues = {}


# ============================================================================
# CLASSIFICATION PATTERNS
# ============================================================================

# Adjustment rows (hedging, corporate, eliminations)
ADJUSTMENT_PATTERNS = [
    re.compile(r"\bhedging\s+(gains?|losses?)\b", re.IGNORECASE),
    re.compile(r"^\s*hedging\b", re.IGNORECASE),
    re.compile(r"\bcorporate\s+(costs?|expenses?|overhead)\b", re.IGNORECASE),
    re.compile(r"^\s*corporate\s*$", re.IGNORECASE),
    re.compile(r"\belimination", re.IGNORECASE),
    re.compile(r"\bintercompany\b", re.IGNORECASE),
    re.compile(r"\breconcil", re.IGNORECASE),
    re.compile(r"\bunallocated\b", re.IGNORECASE),
]

# Total row patterns
TOTAL_PATTERNS = [
    re.compile(r"^\s*total\s+net\s+sales\s*$", re.IGNORECASE),
    re.compile(r"^\s*total\s+revenues?\s*$", re.IGNORECASE),
    re.compile(r"^\s*total\s*$", re.IGNORECASE),
    re.compile(r"^\s*consolidated\s+total", re.IGNORECASE),
    re.compile(r"^\s*net\s+revenues?\s*$", re.IGNORECASE),
    re.compile(r"^\s*total\s+net\s+revenues?\s*$", re.IGNORECASE),
]

# Skip patterns (not revenue items)
SKIP_PATTERNS = [
    re.compile(r"\bdeferred\s+revenue\b", re.IGNORECASE),
    re.compile(r"\bcontract\s+liabil", re.IGNORECASE),
    re.compile(r"\bportion\s+of\s+total\b", re.IGNORECASE),
    re.compile(r"\bincluded\s+in\s+deferred\b", re.IGNORECASE),
    re.compile(r"\bunearned\b", re.IGNORECASE),
    re.compile(r"^\s*$"),
]

# Cost/expense patterns (not revenue)
COST_PATTERNS = [
    re.compile(r"\bcost\s+of\b", re.IGNORECASE),
    re.compile(r"\bgross\s+margin\b", re.IGNORECASE),
    re.compile(r"\boperating\s+(income|expense|loss)\b", re.IGNORECASE),
    re.compile(r"\bresearch\s+and\s+development\b", re.IGNORECASE),
    re.compile(r"\bselling.*general\b", re.IGNORECASE),
    re.compile(r"\bamortization\b", re.IGNORECASE),
    re.compile(r"\bdepreciation\b", re.IGNORECASE),
    re.compile(r"\bnet\s+income\b", re.IGNORECASE),
    re.compile(r"\bearnings?\s+per\b", re.IGNORECASE),
]

# Subtotal patterns
SUBTOTAL_PATTERNS = [
    re.compile(r"^\s*subtotal\b", re.IGNORECASE),
    re.compile(r"^\s*total\s+\w+\s+segment\b", re.IGNORECASE),
    # "Total automotive revenues", "Total product revenues"
    re.compile(r"^\s*total\s+\w+.*\brevenues?\s*$", re.IGNORECASE),
    # "Total automotive & services and other"
    re.compile(r"^\s*total\s+\w+.*(?:&|and)\s+\w+", re.IGNORECASE),
    # Any "Total X" that is NOT the final "Total revenues" / "Total net sales"
    re.compile(r"^\s*total\s+(?!revenues?\s*$|net\s+sales\s*$|net\s+revenue\s*$)\w+", re.IGNORECASE),
]


# ============================================================================
# PARSING UTILITIES
# ============================================================================

_WS_RE = re.compile(r"\s+")
_MONEY_CLEAN_RE = re.compile(r"[^0-9.\-]")


def _clean(s: str) -> str:
    """Normalize whitespace and strip."""
    return _WS_RE.sub(" ", (s or "").strip())


def _parse_money_to_int(val: str) -> Optional[int]:
    """
    Parse a money string to integer.
    
    Handles:
    - Currency symbols: $, €, £
    - Thousands separators: 1,234,567
    - Parentheses for negatives: (1,234)
    - Decimal points: 1234.56 → 1234
    - Empty/dash: —, -, blank
    """
    if not val:
        return None
    
    val = val.strip()
    
    # Handle dash/em-dash as zero or None
    if val in ("—", "-", "–", ""):
        return None
    
    # Handle parentheses for negatives: (1,234) → -1234
    is_negative = False
    if "(" in val and ")" in val:
        is_negative = True
        val = val.replace("(", "").replace(")", "")
    
    # Remove currency symbols and commas
    cleaned = _MONEY_CLEAN_RE.sub("", val)
    
    if not cleaned:
        return None
    
    try:
        # Parse as float first to handle decimals
        num = float(cleaned)
        if is_negative:
            num = -num
        return int(num)
    except ValueError:
        return None


def _strip_footnotes(label: str) -> str:
    """Remove footnote markers like (1), (4) from labels."""
    return re.sub(r'\s*\(\d+\)\s*$', '', label).strip()


def _is_subtotal_of_existing(label: str, all_labels: List[str]) -> bool:
    """
    Detect if a "Total X" label is a subtotal of other extracted items.

    E.g. "Total automotive revenues" is a subtotal when "Automotive sales",
    "Automotive regulatory credits", and "Automotive leasing" also exist.
    """
    label_lower = _clean(label).lower()
    if not label_lower.startswith("total "):
        return False

    # Don't flag the final total row
    if re.match(r"^total\s+(revenues?|net\s+sales|net\s+revenue)\s*$", label_lower):
        return False

    # Extract category word(s) after "Total "
    category = re.sub(r"^total\s+", "", label_lower)
    category = re.sub(r"\s*(revenues?|sales|income)\s*$", "", category).strip()
    if not category:
        return False

    # Split into words for prefix matching
    cat_words = category.split()
    first_word = cat_words[0] if cat_words else ""

    for other in all_labels:
        other_lower = _clean(other).lower()
        if other_lower == label_lower:
            continue
        # If another row starts with the same category word, this is a subtotal
        if first_word and other_lower.startswith(first_word) and other_lower != label_lower:
            return True

    return False


def deduplicate_year_columns(
    year_cols: Dict[int, int],
    grid: List[List[str]],
    header_rows: Optional[set] = None,
) -> Dict[int, int]:
    """
    Validate year column assignments by checking for the $ symbol in separate
    cells and collapsed/spacer columns.

    SEC iXBRL tables often put "$" in one cell and the number in the next,
    creating 40+ columns where only a few carry data.  If the selected column
    yields mostly "$" or empty, shift right to the actual numeric column.
    """
    if not grid or not year_cols:
        return year_cols
    
    header_set = header_rows or set()
    fixed: Dict[int, int] = {}
    n_cols = max(len(r) for r in grid)

    for year, col in year_cols.items():
        if col >= n_cols:
            fixed[year] = col
            continue

        data_cells = []
        for ri, row in enumerate(grid):
            if ri in header_set or col >= len(row):
                continue
            data_cells.append(row[col].strip())

        non_empty = [c for c in data_cells if c]
        if not non_empty:
            fixed[year] = col
            continue

        # Count how many cells are just "$" or currency symbols
        dollar_only = sum(1 for c in non_empty if c in ("$", "€", "£"))
        dollar_ratio = dollar_only / len(non_empty) if non_empty else 0

        if dollar_ratio > 0.4:
            # Likely a currency-symbol column; shift right
            for offset in range(1, 4):
                alt = col + offset
                if alt >= n_cols:
                    break
                alt_cells = []
                for ri, row in enumerate(grid):
                    if ri in header_set or alt >= len(row):
                        continue
                    alt_cells.append(row[alt].strip())
                alt_non_empty = [c for c in alt_cells if c]
                alt_dollar = sum(1 for c in alt_non_empty if c in ("$", "€", "£"))
                if alt_non_empty and alt_dollar / len(alt_non_empty) < 0.3:
                    fixed[year] = alt
                    break
            else:
                fixed[year] = col
        else:
            fixed[year] = col

    return fixed


# ============================================================================
# ROW CLASSIFICATION
# ============================================================================

def classify_row(label: str, expected_segments: Optional[List[str]] = None) -> str:
    """
    Classify a row label into: item, segment, subtotal, adjustment, total, skip.
    
    Args:
        label: The row label text
        expected_segments: Optional list of expected segment names for matching
    
    Returns:
        Row type string
    """
    label_clean = _clean(label)
    label_lower = label_clean.lower()
    
    # Check skip patterns first
    for pat in SKIP_PATTERNS:
        if pat.search(label_clean):
            return "skip"
    
    # Check cost/expense patterns
    for pat in COST_PATTERNS:
        if pat.search(label_clean):
            return "skip"
    
    # Check total patterns
    for pat in TOTAL_PATTERNS:
        if pat.match(label_clean):
            return "total"
    
    # Check adjustment patterns
    for pat in ADJUSTMENT_PATTERNS:
        if pat.search(label_clean):
            return "adjustment"
    
    # Check subtotal patterns
    for pat in SUBTOTAL_PATTERNS:
        if pat.search(label_clean):
            return "subtotal"
    
    # Check if it matches an expected segment
    if expected_segments:
        for seg in expected_segments:
            if label_lower == seg.lower():
                return "segment"
    
    # Default to item (granular revenue line)
    return "item"


def detect_segment_header_mode(grid: List[List[str]], expected_segments: List[str]) -> bool:
    """
    Detect MSFT-style header-based format.
    
    Returns True if table has:
    - Row with first cell matching a segment name
    - Next row with first cell being "Revenue"
    """
    if not expected_segments:
        return False
    
    seg_set = {s.lower() for s in expected_segments}
    
    for i in range(len(grid) - 1):
        if not grid[i]:
            continue
        
        first = _clean(grid[i][0]).lower()
        
        if first in seg_set or first == "total":
            # Check next row for "Revenue"
            if i + 1 < len(grid) and grid[i + 1]:
                next_first = _clean(grid[i + 1][0]).lower()
                if next_first == "revenue":
                    return True
    
    return False


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_row_based(
    grid: List[List[str]],
    layout: Dict[str, Any],
    expected_segments: Optional[List[str]] = None,
) -> ExtractionResult:
    """
    Extract from AAPL-style row-based table.
    
    Revenue items are row labels in item_col, values in year columns.
    """
    # Parse layout parameters
    item_col = int(layout.get("item_col", layout.get("label_col", 0)))
    segment_col = layout.get("segment_col")
    segment_col = int(segment_col) if segment_col is not None else None
    
    year_cols = {}
    for y, c in (layout.get("year_cols") or {}).items():
        try:
            year_cols[int(y)] = int(c)
        except (ValueError, TypeError):
            continue
    
    if not year_cols:
        raise ValueError("No year_cols in layout")
    
    header_rows = set()
    for i in (layout.get("header_rows") or []):
        try:
            header_rows.add(int(i))
        except (ValueError, TypeError):
            continue
    
    # Fix A: deduplicate year columns (handles $ in separate cell)
    year_cols = deduplicate_year_columns(year_cols, grid, header_rows)
    
    year = max(year_cols.keys())
    val_col = year_cols[year]
    
    units_mult = int(layout.get("units_multiplier") or 1)
    if units_mult <= 0:
        units_mult = 1
    
    # Total row regex
    total_regex = layout.get("total_row_regex") or r"total"
    total_re = re.compile(total_regex, re.IGNORECASE)
    
    # Exclude row regex
    exclude_regex = layout.get("exclude_row_regex") or r"$^"
    exclude_re = re.compile(exclude_regex, re.IGNORECASE)
    
    rows: List[ExtractedRow] = []
    table_total: Optional[int] = None
    last_segment: str = ""
    
    for row_idx, row in enumerate(grid):
        if row_idx in header_rows:
            continue
        
        if item_col >= len(row) or val_col >= len(row):
            continue
        
        # Get item label
        item_label = _clean(row[item_col])
        item_label = _strip_footnotes(item_label)
        
        if not item_label:
            continue
        
        # Get segment (if separate column)
        segment = ""
        if segment_col is not None and segment_col < len(row):
            segment = _clean(row[segment_col])
            if segment:
                last_segment = segment
            else:
                segment = last_segment  # Fill-down for rowspan
        
        # Check exclude pattern
        if exclude_re.search(item_label) or (segment and exclude_re.search(segment)):
            continue
        
        # Parse value
        raw_val = _parse_money_to_int(row[val_col])
        
        # Try adjacent columns if $ is in separate column
        if raw_val is None and val_col + 1 < len(row):
            raw_val = _parse_money_to_int(row[val_col + 1])
        if raw_val is None and val_col + 2 < len(row):
            raw_val = _parse_money_to_int(row[val_col + 2])
        
        if raw_val is None:
            continue
        
        value = raw_val * units_mult
        
        # Check for total row
        if total_re.search(item_label) or (segment and total_re.search(segment)):
            table_total = value
            rows.append(ExtractedRow(
                segment=segment,
                item=item_label,
                value=value,
                row_type="total",
                year=year,
            ))
            continue
        
        # Classify row type
        row_type = classify_row(item_label, expected_segments)
        
        if row_type == "skip":
            continue
        
        rows.append(ExtractedRow(
            segment=segment or item_label,  # Use item as segment if no segment column
            item=item_label,
            value=value,
            row_type=row_type,
            year=year,
        ))
    
    return _aggregate_result(rows, table_total, year)


def extract_header_based(
    grid: List[List[str]],
    layout: Dict[str, Any],
    expected_segments: List[str],
) -> ExtractionResult:
    """
    Extract from MSFT-style header-based table.
    
    Segment names appear as header rows, "Revenue" is a metric row beneath.
    """
    year_cols = {}
    for y, c in (layout.get("year_cols") or {}).items():
        try:
            year_cols[int(y)] = int(c)
        except (ValueError, TypeError):
            continue
    
    if not year_cols:
        raise ValueError("No year_cols in layout")
    
    year = max(year_cols.keys())
    val_col = year_cols[year]
    
    units_mult = int(layout.get("units_multiplier") or 1)
    if units_mult <= 0:
        units_mult = 1
    
    header_rows = set()
    for i in (layout.get("header_rows") or []):
        try:
            header_rows.add(int(i))
        except (ValueError, TypeError):
            continue
    
    seg_set = {s.lower(): s for s in expected_segments}
    
    rows: List[ExtractedRow] = []
    table_total: Optional[int] = None
    current_segment: str = ""
    
    for row_idx, row in enumerate(grid):
        if row_idx in header_rows:
            continue
        
        if not row:
            continue
        
        first_cell = _clean(row[0]).lower()
        
        # Check if this is a segment header row
        if first_cell in seg_set:
            current_segment = seg_set[first_cell]
            continue
        
        # Check if this is a "Total" header
        if first_cell == "total":
            current_segment = "Total"
            continue
        
        # Check if this is a "Revenue" metric row
        if first_cell == "revenue" and current_segment:
            if val_col >= len(row):
                continue
            
            raw_val = _parse_money_to_int(row[val_col])
            if raw_val is None and val_col + 1 < len(row):
                raw_val = _parse_money_to_int(row[val_col + 1])
            
            if raw_val is not None:
                value = raw_val * units_mult
                
                if current_segment == "Total":
                    table_total = value
                    rows.append(ExtractedRow(
                        segment=current_segment,
                        item="Total Revenue",
                        value=value,
                        row_type="total",
                        year=year,
                    ))
                else:
                    rows.append(ExtractedRow(
                        segment=current_segment,
                        item=current_segment,
                        value=value,
                        row_type="segment",
                        year=year,
                    ))
    
    return _aggregate_result(rows, table_total, year)


def _aggregate_result(
    rows: List[ExtractedRow],
    table_total: Optional[int],
    year: int,
) -> ExtractionResult:
    """
    Aggregate extracted rows into segment_revenues and adjustment_revenues.
    Applies contextual subtotal detection before aggregation.
    """
    # Build label list for contextual subtotal detection
    all_labels = [r.item for r in rows if r.row_type not in ("total",)]

    # Re-classify subtotals using contextual awareness
    for row in rows:
        if row.row_type in ("item", "segment"):
            if _is_subtotal_of_existing(row.item, all_labels):
                row.row_type = "subtotal"

    segment_revenues: Dict[str, int] = {}
    adjustment_revenues: Dict[str, int] = {}
    
    for row in rows:
        if row.row_type in ("total", "subtotal"):
            continue
        elif row.row_type == "adjustment":
            adjustment_revenues[row.item] = adjustment_revenues.get(row.item, 0) + row.value
        elif row.row_type in ("item", "segment"):
            key = row.segment or row.item
            segment_revenues[key] = segment_revenues.get(key, 0) + row.value
    
    extracted_sum = sum(segment_revenues.values()) + sum(adjustment_revenues.values())
    warnings: List[str] = []

    if table_total and table_total > 0 and extracted_sum > table_total * 1.5:
        warnings.append(
            f"SANITY: extracted_sum ({extracted_sum:,}) is "
            f"{extracted_sum / table_total:.1f}x table_total ({table_total:,}); "
            f"likely multi-year column or subtotal leak"
        )

    return ExtractionResult(
        year=year,
        rows=rows,
        table_total=table_total,
        segment_revenues=segment_revenues,
        adjustment_revenues=adjustment_revenues,
        warnings=warnings,
    )


def extract_values(
    grid: List[List[str]],
    layout: Dict[str, Any],
    expected_segments: Optional[List[str]] = None,
) -> ExtractionResult:
    """
    Main entry point: extract revenue values from grid using layout.
    
    Auto-detects AAPL-style vs MSFT-style based on grid structure.
    """
    # Normalize grid (pad short rows)
    if grid:
        max_len = max(len(row) for row in grid)
        grid = [list(row) + [""] * (max_len - len(row)) for row in grid]
    
    # Detect extraction mode
    segments = expected_segments or []
    is_header_mode = detect_segment_header_mode(grid, segments)
    
    if is_header_mode and segments:
        return extract_header_based(grid, layout, segments)
    else:
        return extract_row_based(grid, layout, segments)


# ============================================================================
# CLI
# ============================================================================

def _to_dict(result: ExtractionResult) -> Dict[str, Any]:
    """Convert ExtractionResult to JSON-serializable dict."""
    return {
        "year": result.year,
        "rows": [asdict(r) for r in result.rows],
        "table_total": result.table_total,
        "dimension": result.dimension,
        "segment_revenues": result.segment_revenues,
        "adjustment_revenues": result.adjustment_revenues,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_values.py <grid_json> <layout_json> [segments_json]", file=sys.stderr)
        sys.exit(1)
    
    grid_path = sys.argv[1]
    layout_path = sys.argv[2]
    segments_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        # Load inputs
        if grid_path == "-":
            grid = json.load(sys.stdin)
        else:
            with open(grid_path) as f:
                grid = json.load(f)
        
        with open(layout_path) as f:
            layout = json.load(f)
        
        segments = None
        if segments_path:
            with open(segments_path) as f:
                segments = json.load(f)
        
        # Extract
        result = extract_values(grid, layout, segments)
        
        # Output
        print(json.dumps(_to_dict(result), indent=2))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
