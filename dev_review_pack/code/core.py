"""Unified revenue extraction that handles multiple table formats."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from revseg.extraction.matching import fuzzy_match_segment, tokenize_label, normalize_segment_name
from revseg.mappings import get_segment_for_item, is_adjustment_item, is_subtotal_row


@dataclass
class ExtractedRow:
    """A single extracted row from a revenue table."""
    segment: str           # Canonical segment/group name (or empty for adjustments)
    item: str              # Original row label
    value: int             # Revenue value in base units (USD)
    row_type: str          # "segment" | "adjustment" | "total" | "unknown" | "item"
    year: int
    dimension: str = "segment"  # "segment" | "product_service" | "end_market" | "revenue_source" | "geography"


@dataclass
class ExtractionResult:
    """Complete extraction result from a revenue table."""
    year: int
    rows: List[ExtractedRow]
    table_total: Optional[int]
    dimension: str = "segment"  # Primary dimension of this extraction
    segment_revenues: Dict[str, int] = field(default_factory=dict)
    adjustment_revenues: Dict[str, int] = field(default_factory=dict)


# Patterns for adjustment rows (hedging, corporate, eliminations)
# These are legitimate reconciling items that may be negative
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

# Patterns for the Total row
TOTAL_PATTERNS = [
    re.compile(r"^\s*total\s+net\s+sales\s*$", re.IGNORECASE),
    re.compile(r"^\s*total\s+revenues?\s*$", re.IGNORECASE),
    re.compile(r"^\s*total\s*$", re.IGNORECASE),
    re.compile(r"^\s*consolidated\s+total", re.IGNORECASE),
]

# Patterns for rows to skip entirely
SKIP_PATTERNS = [
    re.compile(r"\bdeferred\s+revenue\b", re.IGNORECASE),
    re.compile(r"\bcontract\s+liabil", re.IGNORECASE),
    re.compile(r"\bportion\s+of\s+total\b", re.IGNORECASE),  # AAPL's deferred row
    re.compile(r"\bincluded\s+in\s+deferred\b", re.IGNORECASE),
    re.compile(r"\bunearned\b", re.IGNORECASE),
    re.compile(r"^\s*$"),  # Empty
]

# ============================================================================
# DIMENSION DETECTION - Classify revenue table by disclosure dimension
# ============================================================================

# Patterns indicating product/service disaggregation (most granular)
PRODUCT_SERVICE_PATTERNS = [
    re.compile(r"groups?\s+of\s+similar\s+products?\s+(and|&)\s+services?", re.IGNORECASE),
    re.compile(r"disaggregat(ed|ion)\s+(of\s+)?revenue", re.IGNORECASE),
    re.compile(r"revenue\s+by\s+(product|service|category|type)", re.IGNORECASE),
    re.compile(r"net\s+sales\s+by\s+(product|category)", re.IGNORECASE),
    re.compile(r"by\s+(product|service)\s+(line|category|type)", re.IGNORECASE),
]

# Patterns indicating end-market breakdown
END_MARKET_PATTERNS = [
    re.compile(r"revenue\s+by\s+end\s*[-\s]?market", re.IGNORECASE),
    re.compile(r"by\s+end\s*[-\s]?market", re.IGNORECASE),
    re.compile(r"end\s*[-\s]?market\s+revenue", re.IGNORECASE),
]

# Patterns indicating geography breakdown
GEOGRAPHY_PATTERNS = [
    re.compile(r"revenue\s+by\s+geograph", re.IGNORECASE),
    re.compile(r"by\s+geograph", re.IGNORECASE),
    re.compile(r"geographic\s+(area|region)", re.IGNORECASE),
    re.compile(r"revenue\s+by\s+region", re.IGNORECASE),
]

# Patterns indicating segment breakdown (ASC 280 reportable segments)
SEGMENT_PATTERNS = [
    re.compile(r"reportable\s+segment", re.IGNORECASE),
    re.compile(r"operating\s+segment", re.IGNORECASE),
    re.compile(r"segment\s+(revenue|result|information)", re.IGNORECASE),
    re.compile(r"revenue\s+by\s+segment", re.IGNORECASE),
]


def detect_dimension(
    caption: str = "",
    heading: str = "",
    row_labels: Optional[List[str]] = None,
    ticker: str = "",
) -> str:
    """
    Detect the disclosure dimension of a revenue table.
    
    Returns: 'product_service', 'end_market', 'geography', 'segment', or 'unknown'
    
    Priority order (most specific first):
    1. product_service - most granular, preferred for objective
    2. end_market - company-specific breakdown (e.g., NVDA)
    3. geography - regional breakdown
    4. segment - ASC 280 reportable segments
    """
    # Combine text sources for pattern matching
    text = f"{caption} {heading}".lower()
    
    # Check patterns in priority order
    for pattern in PRODUCT_SERVICE_PATTERNS:
        if pattern.search(text):
            return "product_service"
    
    for pattern in END_MARKET_PATTERNS:
        if pattern.search(text):
            return "end_market"
    
    for pattern in GEOGRAPHY_PATTERNS:
        if pattern.search(text):
            return "geography"
    
    for pattern in SEGMENT_PATTERNS:
        if pattern.search(text):
            return "segment"
    
    # Ticker-specific overrides based on known filing structures
    ticker_upper = ticker.upper() if ticker else ""
    if ticker_upper == "NVDA" and row_labels:
        # NVDA's "Data Center/Gaming/etc." table is end_market
        labels_text = " ".join(row_labels).lower()
        if "data center" in labels_text and "gaming" in labels_text:
            return "end_market"
    
    if ticker_upper == "AAPL" and row_labels:
        # AAPL's iPhone/Mac/iPad breakdown is product_service
        labels_text = " ".join(row_labels).lower()
        if "iphone" in labels_text and "mac" in labels_text:
            return "product_service"
    
    if ticker_upper == "AMZN" and row_labels:
        # AMZN has two tables - detect which one
        labels_text = " ".join(row_labels).lower()
        if "online stores" in labels_text or "third-party" in labels_text:
            return "product_service"
        if "north america" in labels_text and "international" in labels_text and "aws" in labels_text:
            return "segment"
    
    # Default to segment if can't determine
    return "segment"


# Patterns for classifying individual row labels as revenue source vs segment
REVENUE_SOURCE_LABEL_PATTERNS = [
    re.compile(r"^\s*advertising\s*$", re.IGNORECASE),
    re.compile(r"^\s*other\s+revenue\s*$", re.IGNORECASE),
    re.compile(r"^\s*subscription\s*(revenue|services?)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*licensing\s*(revenue)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*service\s+fees?\s*$", re.IGNORECASE),
    re.compile(r"^\s*product\s+sales?\s*$", re.IGNORECASE),
]

SEGMENT_LABEL_PATTERNS = [
    re.compile(r"family\s+of\s+apps", re.IGNORECASE),
    re.compile(r"reality\s+labs", re.IGNORECASE),
    re.compile(r"intelligent\s+cloud", re.IGNORECASE),
    re.compile(r"productivity\s+and\s+business", re.IGNORECASE),
    re.compile(r"personal\s+computing", re.IGNORECASE),
    re.compile(r"google\s+(services|cloud)", re.IGNORECASE),
    re.compile(r"other\s+bets", re.IGNORECASE),
    re.compile(r"north\s+america", re.IGNORECASE),
    re.compile(r"international", re.IGNORECASE),
    # NOTE: Removed "aws" pattern - AWS should inherit table dimension (product_service for AMZN)
]


def classify_row_dimension(label: str, table_dimension: str = "segment") -> str:
    """
    Classify an individual row label's dimension.
    
    For tables with mixed dimensions (like META), this helps assign the correct
    dimension to each row.
    
    Args:
        label: The row label to classify
        table_dimension: The overall table dimension (used as default)
    
    Returns: 'segment', 'revenue_source', or the table_dimension as fallback
    """
    label_clean = label.strip()
    
    # Check revenue source patterns
    for pattern in REVENUE_SOURCE_LABEL_PATTERNS:
        if pattern.search(label_clean):
            return "revenue_source"
    
    # Check segment patterns
    for pattern in SEGMENT_LABEL_PATTERNS:
        if pattern.search(label_clean):
            return "segment"
    
    # Default to table dimension
    return table_dimension


def _is_adjustment_row(label: str) -> bool:
    """Check if a row label indicates an adjustment/reconciling item."""
    for pat in ADJUSTMENT_PATTERNS:
        if pat.search(label):
            return True
    return False


def _is_total_row(label: str) -> bool:
    """Check if a row label indicates a Total row."""
    for pat in TOTAL_PATTERNS:
        if pat.search(label):
            return True
    return False


def _is_skip_row(label: str) -> bool:
    """Check if a row should be skipped entirely."""
    for pat in SKIP_PATTERNS:
        if pat.search(label):
            return True
    return False


def classify_row_label(label: str, expected_segments: List[str]) -> str:
    """
    Classify a row label into categories.
    
    Returns: 'segment', 'adjustment', 'total', or 'skip'
    """
    if not label or not label.strip():
        return "skip"
    
    label = label.strip()
    
    # Check skip patterns first
    if _is_skip_row(label):
        return "skip"
    
    # Check total patterns
    if _is_total_row(label):
        return "total"
    
    # Check if it matches an expected segment (fuzzy)
    matched = fuzzy_match_segment(label, expected_segments, threshold=0.6)
    if matched:
        return "segment"
    
    # Check adjustment patterns
    if _is_adjustment_row(label):
        return "adjustment"
    
    # Default: treat as potential segment (will be filtered later)
    return "unknown"


def _parse_money(s: str) -> Optional[int]:
    """
    Parse a money string to integer.
    
    Handles:
    - '$1,234' -> 1234
    - '(500)' -> -500 (accounting negative)
    - '1,234.56' -> 1235 (rounded)
    """
    if not s:
        return None
    
    t = str(s).strip()
    if t in {"", "-", "—", "–", "$", "—", "−"}:
        return None
    
    # Handle accounting-style negatives: (500)
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg = True
        t = t[1:-1]
    
    # Remove currency symbols and thousands separators
    t = t.replace("$", "").replace(",", "").replace(" ", "").strip()
    
    # Handle explicit minus sign
    if t.startswith("-") or t.startswith("−"):
        neg = True
        t = t[1:]
    
    try:
        v = float(t)
        return int(round(-v if neg else v))
    except ValueError:
        return None


def _try_parse_value_with_fallback(row: List[str], start_col: int, max_offset: int = 3) -> Optional[int]:
    """
    Try parsing value from start_col, then +1, +2, etc.
    
    This handles iXBRL tables where '$' is in a separate column from the number.
    """
    for offset in range(max_offset):
        col = start_col + offset
        if col < len(row):
            val = _parse_money(row[col])
            if val is not None:
                return val
    return None


def _auto_detect_year_cols(grid: List[List[str]], max_header_rows: int = 12) -> Dict[int, int]:
    """
    Scan header rows for year patterns like '2024', 'FY2024', 'Year Ended 2024'.
    
    Returns: Dict mapping year (int) to column index
    """
    year_re = re.compile(r"\b(20\d{2})\b")
    year_cols: Dict[int, int] = {}
    
    for row in grid[:max_header_rows]:
        for col_i, cell in enumerate(row):
            m = year_re.search(str(cell or ""))
            if m:
                y = int(m.group(1))
                if 2018 <= y <= 2030:
                    # Prefer first occurrence of each year
                    year_cols.setdefault(y, col_i)
    
    return year_cols


def _detect_segment_header_mode(grid: List[List[str]], expected_segments: List[str]) -> bool:
    """
    Detect if table uses segment names as header rows with 'Revenue' as a metric row.
    
    This is common in MSFT's "segment results of operations" tables:
    - Row: "Productivity and Business Processes" (header)
    - Row: "Revenue  $XX,XXX  $XX,XXX" (metric)
    - Row: "Cost of revenue ..."
    - Row: "Intelligent Cloud" (next header)
    - ...
    """
    seg_set = {s.lower() for s in expected_segments}
    found_seg_header = False
    found_revenue_after = False
    
    for i, row in enumerate(grid[:50]):
        if not row:
            continue
        
        first = (row[0] or "").strip().lower()
        
        # Check if this row looks like a segment header
        if first in seg_set or first == "total":
            found_seg_header = True
            continue
        
        # After finding a segment header, look for "Revenue" row
        if found_seg_header and first == "revenue":
            found_revenue_after = True
            break
    
    return found_seg_header and found_revenue_after


def extract_revenue_unified(
    grid: List[List[str]],
    *,
    expected_segments: List[str],
    layout: Dict[str, Any],
    include_unknown_as_segments: bool = False,
) -> ExtractionResult:
    """
    Unified extraction that handles multiple table formats:
    
    1. Row-based (AAPL, GOOGL disaggregation):
       - Product/segment names as row labels in column 0
       - Values in year columns
    
    2. Header-based (MSFT segment results):
       - Segment names as standalone header rows
       - "Revenue" as a metric row under each segment
    
    Args:
        grid: 2D list of cell strings from the table
        expected_segments: List of segment names to look for
        layout: Layout dict from LLM with item_col, year_cols, etc.
        include_unknown_as_segments: If True, include unmatched rows as segments
    
    Returns:
        ExtractionResult with rows, segment_revenues, adjustment_revenues, table_total
    """
    # Normalize grid (pad short rows to same length)
    max_len = max((len(r) for r in grid), default=0)
    if max_len > 0:
        grid = [list(r) + [""] * (max_len - len(r)) for r in grid]
    
    # Extract layout params with fallbacks
    item_col = int(layout.get("item_col", layout.get("label_col", 0)))
    segment_col = layout.get("segment_col")
    segment_col = int(segment_col) if segment_col is not None else None
    
    # Year columns
    year_cols_raw = layout.get("year_cols") or {}
    year_cols: Dict[int, int] = {}
    for y, c in year_cols_raw.items():
        try:
            year_cols[int(y)] = int(c)
        except (ValueError, TypeError):
            continue
    
    # Auto-detect year columns if not provided or invalid
    if not year_cols:
        year_cols = _auto_detect_year_cols(grid)
    
    if not year_cols:
        raise ValueError("No year columns detected in table")
    
    # Use the most recent year
    year = max(year_cols.keys())
    val_col = year_cols[year]
    
    # Header rows to skip
    header_rows = set()
    for i in (layout.get("header_rows") or []):
        try:
            header_rows.add(int(i))
        except (ValueError, TypeError):
            continue
    
    # Units multiplier
    units_mult = int(layout.get("units_multiplier") or 1)
    if units_mult <= 0:
        units_mult = 1
    
    # Detect extraction mode
    is_segment_header_mode = _detect_segment_header_mode(grid, expected_segments)
    
    rows: List[ExtractedRow] = []
    table_total: Optional[int] = None
    current_segment_header = ""
    last_segment_from_col = ""
    
    for r_i, row in enumerate(grid):
        if r_i in header_rows:
            continue
        if not row or all(not (c or "").strip() for c in row):
            continue
        
        # Get labels
        item_label = (row[item_col] if item_col < len(row) else "").strip()
        seg_label = (row[segment_col] if segment_col is not None and segment_col < len(row) else "").strip()
        
        # Track segment from dedicated column (with fill-down for iXBRL)
        if segment_col is not None and seg_label:
            last_segment_from_col = seg_label
        elif segment_col is not None and not seg_label:
            seg_label = last_segment_from_col
        
        # Combined label for classification
        combined_label = seg_label if seg_label else item_label
        if not combined_label:
            continue
        
        # === Segment Header Mode (MSFT-style) ===
        if is_segment_header_mode:
            # Check if this row is a segment header
            matched_header = fuzzy_match_segment(item_label, expected_segments, threshold=0.8)
            if matched_header and item_label.lower() != "revenue":
                current_segment_header = matched_header
                continue
            
            # Check for "Total" header
            if item_label.lower() == "total":
                current_segment_header = "Total"
                continue
            
            # Look for "Revenue" metric row under current segment
            if item_label.lower() == "revenue" and current_segment_header:
                val = _try_parse_value_with_fallback(row, val_col)
                if val is not None:
                    scaled_val = val * units_mult
                    if current_segment_header == "Total":
                        table_total = scaled_val
                    else:
                        rows.append(ExtractedRow(
                            segment=current_segment_header,
                            item=f"{current_segment_header}",
                            value=scaled_val,
                            row_type="segment",
                            year=year,
                        ))
                continue
        
        # === Standard Row-Based Mode (AAPL, GOOGL disaggregation) ===
        row_type = classify_row_label(combined_label, expected_segments)
        
        if row_type == "skip":
            continue
        
        # Parse value
        val = _try_parse_value_with_fallback(row, val_col)
        if val is None:
            continue
        
        scaled_val = val * units_mult
        
        # Handle Total row
        if row_type == "total":
            table_total = scaled_val
            continue
        
        # For segments, try to match to expected
        matched_seg = ""
        if row_type == "segment":
            matched_seg = fuzzy_match_segment(combined_label, expected_segments, threshold=0.6)
            if matched_seg:
                matched_seg = normalize_segment_name(matched_seg)
        
        # For adjustments, keep the original label
        if row_type == "adjustment":
            rows.append(ExtractedRow(
                segment="",
                item=normalize_segment_name(combined_label),
                value=scaled_val,
                row_type="adjustment",
                year=year,
            ))
            continue
        
        # For segments and unknowns
        if matched_seg:
            rows.append(ExtractedRow(
                segment=matched_seg,
                item=normalize_segment_name(combined_label),
                value=scaled_val,
                row_type="segment",
                year=year,
            ))
        elif include_unknown_as_segments and row_type == "unknown":
            rows.append(ExtractedRow(
                segment=normalize_segment_name(combined_label),
                item=normalize_segment_name(combined_label),
                value=scaled_val,
                row_type="unknown",
                year=year,
            ))
    
    # Aggregate into segment_revenues and adjustment_revenues
    segment_revenues: Dict[str, int] = {}
    adjustment_revenues: Dict[str, int] = {}
    
    for r in rows:
        if r.row_type == "segment" and r.segment:
            segment_revenues[r.segment] = segment_revenues.get(r.segment, 0) + r.value
        elif r.row_type == "adjustment":
            adjustment_revenues[r.item] = adjustment_revenues.get(r.item, 0) + r.value
    
    return ExtractionResult(
        year=year,
        rows=rows,
        table_total=table_total,
        segment_revenues=segment_revenues,
        adjustment_revenues=adjustment_revenues,
    )


def extract_line_items_granular(
    grid: List[List[str]],
    *,
    ticker: str,
    layout: Dict[str, Any],
    expected_segments: Optional[List[str]] = None,
    dimension: str = "segment",
    caption: str = "",
    heading: str = "",
) -> ExtractionResult:
    """
    Extract all line items from a 'Revenue by Products and Services' table.
    
    This is the granular extraction mode that:
    1. Extracts each line item with its revenue figure
    2. Assigns segment using mapping dictionary (for MSFT-style tables)
    3. Includes adjustment lines (hedging, etc.) for 100% reconciliation
    4. Skips segment subtotal rows if granular items are present
    
    Args:
        grid: 2D list of cell strings from the table
        ticker: Company ticker for segment mapping lookup
        layout: Layout dict from LLM with item_col, year_cols, etc.
        expected_segments: Optional list of expected segment names
        dimension: Disclosure dimension ('segment', 'product_service', 'end_market', etc.)
        caption: Table caption (for dimension detection)
        heading: Table heading context (for dimension detection)
    
    Returns:
        ExtractionResult with individual line items
    """
    # Normalize grid
    max_len = max((len(r) for r in grid), default=0)
    if max_len > 0:
        grid = [list(r) + [""] * (max_len - len(r)) for r in grid]
    
    # Extract layout params
    item_col = int(layout.get("item_col", layout.get("label_col", 0)))
    
    # Year columns
    year_cols_raw = layout.get("year_cols") or {}
    year_cols: Dict[int, int] = {}
    for y, c in year_cols_raw.items():
        try:
            year_cols[int(y)] = int(c)
        except (ValueError, TypeError):
            continue
    
    if not year_cols:
        year_cols = _auto_detect_year_cols(grid)
    
    if not year_cols:
        raise ValueError("No year columns detected in table")
    
    year = max(year_cols.keys())
    val_col = year_cols[year]
    
    # Header rows to skip
    header_rows = set()
    for i in (layout.get("header_rows") or []):
        try:
            header_rows.add(int(i))
        except (ValueError, TypeError):
            continue
    
    # Units multiplier
    units_mult = int(layout.get("units_multiplier") or 1)
    if units_mult <= 0:
        units_mult = 1
    
    rows: List[ExtractedRow] = []
    table_total: Optional[int] = None
    collected_items: Set[str] = set()
    
    for r_i, row in enumerate(grid):
        if r_i in header_rows:
            continue
        if not row or all(not (c or "").strip() for c in row):
            continue
        
        # Get item label
        item_label = (row[item_col] if item_col < len(row) else "").strip()
        if not item_label:
            continue
        
        # Skip rows we've already seen (deduplication)
        item_key = item_label.lower()
        if item_key in collected_items:
            continue
        
        # Parse value
        val = _try_parse_value_with_fallback(row, val_col)
        if val is None:
            continue
        
        scaled_val = val * units_mult
        
        # Check if this is the Total row
        if _is_total_row(item_label):
            table_total = scaled_val
            continue
        
        # Check if this should be skipped
        if _is_skip_row(item_label):
            continue
        
        # Check if this is a segment subtotal row (skip if we have granular items)
        if is_subtotal_row(item_label, ticker):
            continue
        
        # Determine row type and segment assignment
        row_type = "item"
        segment = ""
        
        # Check if it's an adjustment
        if is_adjustment_item(ticker, item_label) or _is_adjustment_row(item_label):
            row_type = "adjustment"
            segment = "Other"
        else:
            # Try to get segment from mapping
            mapped_segment = get_segment_for_item(ticker, item_label)
            if mapped_segment:
                segment = mapped_segment
            elif expected_segments:
                # Try fuzzy match to expected segments
                matched = fuzzy_match_segment(item_label, expected_segments, threshold=0.7)
                if matched:
                    segment = matched
                else:
                    # Item itself might be the segment (AAPL case)
                    segment = normalize_segment_name(item_label)
            else:
                segment = normalize_segment_name(item_label)
            
            row_type = "item"
        
        collected_items.add(item_key)
        
        # Classify this specific row's dimension (handles mixed tables like META)
        row_dimension = classify_row_dimension(item_label, dimension)
        
        rows.append(ExtractedRow(
            segment=segment,
            item=item_label,  # Keep original label for transparency
            value=scaled_val,
            row_type=row_type,
            year=year,
            dimension=row_dimension,
        ))
    
    # P0.2 FIX: Deduplicate segment-totals when granular items exist
    # Track which segments have granular (non-segment-level) items
    granular_dims = {"product_service", "revenue_source", "end_market", "geography"}
    segments_with_granular: Set[str] = set()
    
    for r in rows:
        if r.segment and r.dimension in granular_dims:
            segments_with_granular.add(r.segment)
    
    # Aggregate by segment, excluding segment-level totals when granular exists
    segment_revenues: Dict[str, int] = {}
    adjustment_revenues: Dict[str, int] = {}
    
    for r in rows:
        if r.row_type == "adjustment":
            adjustment_revenues[r.item] = adjustment_revenues.get(r.item, 0) + r.value
        else:
            if r.segment:
                # Skip segment-level rows if we have granular items for that segment
                if r.dimension == "segment" and r.segment in segments_with_granular:
                    continue  # Exclude to avoid double-counting
                segment_revenues[r.segment] = segment_revenues.get(r.segment, 0) + r.value
    
    return ExtractionResult(
        year=year,
        rows=rows,
        table_total=table_total,
        dimension=dimension,
        segment_revenues=segment_revenues,
        adjustment_revenues=adjustment_revenues,
    )


def extract_with_layout_fallback(
    grid: List[List[str]],
    *,
    expected_segments: List[str],
    llm_layout: Dict[str, Any],
    ticker: str = "",
    prefer_granular: bool = True,
    dimension: str = "segment",
    caption: str = "",
    heading: str = "",
) -> Optional[ExtractionResult]:
    """
    Try extraction with multiple layout variations if the first attempt fails.
    
    Strategies:
    1. Granular line-item extraction (if prefer_granular=True)
    2. LLM layout exactly  
    3. LLM layout with year_col shifted ±1 (handles $ in separate column)
    4. Auto-detect layout ignoring LLM hints
    """
    # Strategy 0: Granular line-item extraction (preferred for CSV1)
    if prefer_granular and ticker:
        try:
            result = extract_line_items_granular(
                grid,
                ticker=ticker,
                layout=llm_layout,
                expected_segments=expected_segments,
                dimension=dimension,
                caption=caption,
                heading=heading,
            )
            if result.rows:
                return result
        except Exception:
            pass
    
    # Strategy 1: Try LLM layout exactly
    try:
        result = extract_revenue_unified(
            grid,
            expected_segments=expected_segments,
            layout=llm_layout,
        )
        if result.segment_revenues:
            return result
    except Exception:
        pass
    
    # Strategy 2: Try with shifted year columns
    year_cols = llm_layout.get("year_cols") or {}
    for shift in [1, -1, 2]:
        try:
            shifted_layout = dict(llm_layout)
            shifted_year_cols = {y: c + shift for y, c in year_cols.items()}
            shifted_layout["year_cols"] = shifted_year_cols
            
            result = extract_revenue_unified(
                grid,
                expected_segments=expected_segments,
                layout=shifted_layout,
            )
            if result.segment_revenues:
                return result
        except Exception:
            continue
    
    # Strategy 3: Auto-detect layout (ignore LLM hints for year_cols)
    try:
        auto_layout = {
            "item_col": 0,
            "year_cols": _auto_detect_year_cols(grid),
            "header_rows": llm_layout.get("header_rows", [0, 1]),
            "units_multiplier": llm_layout.get("units_multiplier", 1),
        }
        result = extract_revenue_unified(
            grid,
            expected_segments=expected_segments,
            layout=auto_layout,
        )
        if result.segment_revenues:
            return result
    except Exception:
        pass
    
    return None
