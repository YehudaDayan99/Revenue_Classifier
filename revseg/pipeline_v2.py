"""
Revenue Classifier Pipeline v2

This pipeline implements the revised architecture with:
- Phase 0: Income Statement Anchor (ground truth)
- Phase 1: Document Scout (find ALL revenue tables)
- Phase 2: Prioritized Table Selection (A→B→C→D hierarchy)
- Phase 3: Reconciliation Agent (verify completeness)
- Multi-table aggregation for complex companies

Per Financial_Analyst_Prompt.md:
"You must extract the most granular explicit revenue information available."
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from revseg.llm_client import OpenAIChatClient
from revseg.income_statement import (
    IncomeStatementAnchor,
    extract_income_statement_anchor,
)
from revseg.document_scout import (
    DocumentMap,
    RevenueTableInfo,
    scout_document,
    select_tables_for_extraction,
)
from revseg.reconciliation import (
    ReconciliationResult,
    reconcile_extraction,
    suggest_missing_components,
)
from revseg.table_candidates import (
    TableCandidate,
    extract_table_candidates_from_html,
    extract_table_grid_normalized,
    find_latest_downloaded_filing_dir,
    find_primary_document_html,
)
from revseg.sec_edgar import download_latest_10k
from revseg.react_agents import (
    infer_disaggregation_layout,
    choose_item_col,
    validate_extracted_labels,
)
from revseg.extraction import (
    extract_with_layout_fallback,
    detect_dimension,
    ExtractionResult,
)
from revseg.extraction.validation import validate_extraction, ValidationResult
from revseg.validate import fetch_companyfacts_total_revenue_usd
from revseg.mappings import get_segment_for_item, is_subtotal_row, is_total_row


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_company_name_from_submission(filing_dir: Path) -> str:
    sub = filing_dir / "submission.json"
    if not sub.exists():
        return ""
    data = _load_json(sub)
    return str(data.get("name") or "").strip()


def _read_filing_ref(filing_dir: Path) -> Dict[str, Any]:
    ref = filing_dir / "filing_ref.json"
    if not ref.exists():
        raise FileNotFoundError(f"Missing filing_ref.json in {filing_dir}")
    return _load_json(ref)


def _ensure_filing_dir(ticker: str, *, base_dir: Path, cache_dir: Optional[Path]) -> Path:
    try:
        return find_latest_downloaded_filing_dir(base_dir, ticker)
    except Exception:
        return download_latest_10k(
            ticker,
            base_dir,
            cache_dir=cache_dir,
            include_amendments=False,
            min_interval_s=0.2,
        )


def _clean_revenue_line(label: str) -> str:
    """Remove footnote markers like (1), (4) from revenue line labels."""
    return re.sub(r'\s*\(\d+\)\s*$', '', label).strip()


def _get_revenue_group(ticker: str, item: str, dimension: str, row_segment: str = "") -> str:
    """Determine Revenue Group for output."""
    segment = get_segment_for_item(ticker, item)
    if segment:
        return segment
    if dimension == "segment" and row_segment:
        return row_segment
    return "Product/Service disclosure"


def _to_millions(value_usd: int) -> float:
    """Convert base units (USD) to millions."""
    return round(value_usd / 1_000_000, 2)


def _estimate_table_revenue(
    table_id: str,
    candidates: List[TableCandidate],
    html_path: Path,
) -> Optional[int]:
    """
    Quick estimate of total revenue in a table by finding the largest value
    (typically a total row) in the most recent year's column.
    Returns value in base units (not millions) or None if can't estimate.
    """
    try:
        grid = extract_table_grid_normalized(html_path, table_id)
        if not grid:
            return None
        
        from revseg.extraction.core import _parse_money
        import re
        
        # Try to detect year columns from header rows
        year_re = re.compile(r"\b(20\d{2})\b")
        year_cols: Dict[int, int] = {}  # year -> column index
        
        for row in grid[:5]:  # Check first 5 rows for year headers
            for col_i, cell in enumerate(row):
                match = year_re.search(str(cell or ""))
                if match:
                    y = int(match.group(1))
                    if 2020 <= y <= 2030:
                        year_cols.setdefault(y, col_i)
        
        # Find max value per column (typically the total row)
        col_max: Dict[int, int] = {}
        col_counts: Dict[int, int] = {}
        
        for row in grid:
            for col_i, cell in enumerate(row):
                val = _parse_money(cell)
                if val is not None and val > 0:
                    col_max[col_i] = max(col_max.get(col_i, 0), val)
                    col_counts[col_i] = col_counts.get(col_i, 0) + 1
        
        if not col_max:
            return None
        
        # Find the most recent year column and try nearby columns
        if year_cols:
            most_recent_year = max(year_cols.keys())
            target_col = year_cols[most_recent_year]
            
            # Try target_col, target_col+1, target_col-1 to handle header misalignment
            best_max = 0
            for offset in [0, 1, -1]:
                col = target_col + offset
                if col in col_max and col_counts.get(col, 0) >= 2:
                    if col_max[col] > best_max:
                        best_max = col_max[col]
            
            if best_max > 0:
                return best_max
        
        # Otherwise, return the largest max value from any column with at least 2 values
        best_max = 0
        for col_i, mx in col_max.items():
            if col_counts.get(col_i, 0) >= 2 and mx > best_max:
                best_max = mx
        
        return best_max if best_max > 0 else None
    except Exception:
        return None


def _prioritize_tables_with_revenue(
    selected_tables: List[RevenueTableInfo],
    candidates: List[TableCandidate],
    html_path: Path,
    income_anchor: Optional[Any],  # Can be IncomeStatementAnchor or None
    ticker: str,
) -> List[RevenueTableInfo]:
    """
    Re-prioritize tables based on their actual revenue content.
    
    Tables with values in the expected revenue range (relative to income statement)
    are prioritized over tables with very small or percentage values.
    """
    if not income_anchor:
        return selected_tables[:3]  # No anchor to compare against
    
    # Handle both dataclass and dict access
    target_revenue = getattr(income_anchor, "total_revenue", 0) if hasattr(income_anchor, "total_revenue") else income_anchor.get("total_revenue", 0) if isinstance(income_anchor, dict) else 0
    if target_revenue <= 0:
        return selected_tables[:3]
    
    # Estimate revenue for each table
    table_estimates: List[Tuple[RevenueTableInfo, int]] = []
    for table in selected_tables:
        estimate = _estimate_table_revenue(table.table_id, candidates, html_path)
        if estimate is not None:
            # Check if estimate is in the right ballpark (allow 10x for units multiplier)
            table_estimates.append((table, estimate))
    
    if not table_estimates:
        return selected_tables[:3]
    
    # Sort by how close the estimate is to target (allowing for units multiplier)
    def score_table(item: Tuple[RevenueTableInfo, int]) -> float:
        table, estimate = item
        if estimate == 0:
            return float('inf')
        
        # Check multiple possible unit multipliers
        for mult in [1, 1000, 1000000]:
            scaled = estimate * mult
            if target_revenue > 0:
                ratio = scaled / target_revenue
                # Good if ratio is between 0.5 and 2.0 (accounting for partial tables)
                if 0.5 <= ratio <= 2.0:
                    return abs(1 - ratio)  # Lower is better
                elif 0.1 <= ratio <= 10:
                    return abs(1 - ratio) + 1  # Acceptable but not ideal
        
        return float('inf')  # Way off
    
    table_estimates.sort(key=score_table)
    
    # Return top 5 tables sorted by score (instead of just 3)
    # This gives more chances to find the right table
    result = [item[0] for item in table_estimates[:5]]
    
    # If we still have fewer than 3 tables, add from original selection
    original_ids = {t.table_id for t in result}
    for table in selected_tables:
        if table.table_id not in original_ids and len(result) < 5:
            result.append(table)
    
    return result


# Known subtotal patterns that should be removed when granular items exist
_SUBTOTAL_PATTERNS = [
    "family of apps",
    "foa",
    "total family of apps",
]


_NON_REVENUE_PATTERNS = re.compile(
    r'(members?|cardholders?|households?|employees?|stores?(?!\s+ancillary)|'
    r'square\s+feet|locations?|countries?|markets?|percentage|margin\s+percentage|'
    r'total\s+paid|gold\s+star|business.*affiliates|'
    r'merchandise\s+costs?|less\s+merchandise|cost\s+of\s+sales|'
    r'income\s+from\s+operations|operating\s+income|net\s+income|'
    r'total\s+income\s+from|income\s+\(loss\)|'
    # Expense items
    r'personnel|salaries|wages|compensation|'
    r'network\s+and\s+processing|data\s+center|infrastructure|'
    r'professional\s+fees|legal\s+fees|consulting|'
    r'general\s+and\s+administrative|g\s*&\s*a|administrative|'
    r'litigation\s+provision|legal\s+provision|settlement|'
    r'investment\s+(income|expense)|interest\s+(income|expense)|'
    r'income\s+tax\s+provision|tax\s+expense|provision\s+for\s+tax|'
    r'stock|common\s+stock|preferred\s+stock|treasury|'
    r'depreciation|amortization|impairment|'
    r'earnings?\s+per\s+share|diluted\s+earnings|basic\s+earnings|'
    r'non.?gaap|adjusted|'
    r'total\s+operating\s+expenses?|operating\s+expenses?)',
    re.IGNORECASE
)


def _filter_non_revenue_items(revenues: Dict[str, int], ticker: str) -> Dict[str, int]:
    """
    Filter out items that are clearly not revenue (membership counts, percentages, etc.).
    Also filter out items that are very small relative to the largest items (likely % comparisons).
    """
    filtered = {}
    
    # First pass: filter by pattern
    for label, value in revenues.items():
        if _NON_REVENUE_PATTERNS.search(label):
            print(f"[{ticker}] Filtering non-revenue item: '{label}'", flush=True)
            continue
        filtered[label] = value
    
    if not filtered:
        return filtered
    
    # Second pass: filter out very small values (< 0.1% of max) that are likely percentages
    max_value = max(filtered.values())
    if max_value > 0:
        threshold = max_value * 0.001  # 0.1% of max
        to_remove = []
        for label, value in filtered.items():
            if value > 0 and value < threshold:
                # Very small relative value - likely a percentage or count
                print(f"[{ticker}] Filtering tiny value: '{label}' = {value} (threshold: {threshold:.0f})", flush=True)
                to_remove.append(label)
        for label in to_remove:
            del filtered[label]
    
    return filtered


def _auto_scale_units(
    revenues: Dict[str, int],
    adjustments: Dict[str, int],
    target_total: int,
    ticker: str,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Auto-detect and apply units scaling if extracted values are off by a large factor.
    
    For example, if extracted values are 109,564 and target is 254,453,000,000,
    then the extracted values are likely in millions and need to be multiplied by 1,000,000.
    
    Args:
        revenues: Dict of {item_name: value} in extracted units
        adjustments: Dict of {item_name: value} for adjustment items
        target_total: Target total revenue from income statement (in base units)
        ticker: Company ticker for logging
    
    Returns:
        Tuple of (scaled_revenues, scaled_adjustments)
    """
    if not revenues or target_total <= 0:
        return revenues, adjustments
    
    # Calculate current sum
    current_sum = sum(v for v in revenues.values() if v > 0)
    if current_sum <= 0:
        return revenues, adjustments
    
    # Calculate ratio
    ratio = target_total / current_sum
    
    # Detect if we need to scale (using wider range to account for missing/extra items)
    scale_factor = 1
    if 0.3e6 <= ratio <= 3e6:
        # Likely off by 1 million (values in millions, target in units)
        scale_factor = 1_000_000
    elif 0.3e3 <= ratio <= 3e3:
        # Likely off by 1 thousand (values in thousands, target in units)
        scale_factor = 1_000
    elif 0.3e9 <= ratio <= 3e9:
        # Likely off by 1 billion (rare, but possible)
        scale_factor = 1_000_000_000
    
    if scale_factor > 1:
        print(f"[{ticker}] Auto-scaling units: extracted sum={current_sum:,}, target={target_total:,}, ratio={ratio:.0f}", flush=True)
        print(f"[{ticker}]   Applying scale_factor={scale_factor:,}", flush=True)
        
        scaled_revenues = {k: v * scale_factor for k, v in revenues.items()}
        scaled_adjustments = {k: v * scale_factor for k, v in adjustments.items()}
        return scaled_revenues, scaled_adjustments
    
    return revenues, adjustments


def _deduplicate_alternate_views(
    revenues: Dict[str, int],
    income_anchor: Optional[Any],
    ticker: str,
) -> Dict[str, int]:
    """
    Remove duplicate revenue breakdowns that represent different views of the same total.
    
    For example, if we have:
      - Segment view: FoA + Reality Labs = $164B
      - Geography view: US + Europe + Asia + Rest = $164B
    
    We should keep only one view (prefer the more granular or semantically richer one).
    """
    if len(revenues) < 3 or not income_anchor:
        print(f"[{ticker}] Dedup: skipping (revenues={len(revenues)}, anchor={bool(income_anchor)})", flush=True)
        return revenues
    
    target_total = getattr(income_anchor, "total_revenue", 0)
    if target_total <= 0:
        print(f"[{ticker}] Dedup: skipping (target_total={target_total})", flush=True)
        return revenues
    
    # FIRST, filter out year/date labels that aren't revenue items
    date_pattern = re.compile(
        r'^(december|january|february|march|april|may|june|july|august|september|october|november)\s+\d|'
        r'^\d{4}$|'  # Just a year like "2023"
        r'^(fiscal|fy|year)\s*(20)?\d{2}',  # FY2024, Fiscal 2024
        re.IGNORECASE
    )
    revenues = {k: v for k, v in revenues.items() if not date_pattern.search(k)}
    
    # Identify potential "views" (groups of items that sum to approximately the target)
    items = list(revenues.items())
    total_sum = sum(v for _, v in items if v > 0)
    
    # If total sum is approximately 2x target, we likely have duplicate views
    # Widen the range slightly to account for small additional items
    if 1.5 * target_total <= total_sum <= 3.0 * target_total:
        # Try to identify geographic items (these are typically secondary to segment/product views)
        # Geographic patterns - pure geographic names (end-of-string anchored)
        # Must not match revenue types like "International transaction revenue"
        geo_patterns = re.compile(
            r'^(united\s+states(\s+and\s+canada)?|u\.s\.\s*|americas?\s*|europe\s*|asia\s*|apac\s*|asia.?pacific\s*|'
            r'emea\s*|latam\s*|rest\s+of\s+(the\s+)?world|international\s*|north\s+america\s*|'
            r'canada\s*|united\s+kingdom\s*|china\s*|japan\s*|germany\s*|france\s*)$',
            re.IGNORECASE
        )
        
        geo_items = {k: v for k, v in revenues.items() if geo_patterns.search(k)}
        non_geo_items = {k: v for k, v in revenues.items() if not geo_patterns.search(k)}
        
        # If we have both geo and non-geo items, prefer non-geo (segment/product)
        if geo_items and non_geo_items:
            geo_sum = sum(v for v in geo_items.values() if v > 0)
            non_geo_sum = sum(v for v in non_geo_items.values() if v > 0)
            
            # If non-geo sum is close to target, use non-geo items
            if 0.8 * target_total <= non_geo_sum <= 1.2 * target_total:
                print(f"[{ticker}] Removing geographic breakdown (keeping segment view)", flush=True)
                for k in geo_items:
                    print(f"[{ticker}]   Removed: '{k}'", flush=True)
                revenues = non_geo_items  # Continue to date filter below
            
            # If geo sum is close to target and non-geo is way off, use geo items
            elif 0.8 * target_total <= geo_sum <= 1.2 * target_total:
                if non_geo_sum > 1.5 * target_total:
                    print(f"[{ticker}] Keeping geographic breakdown (non-geo items doubled)", flush=True)
                    revenues = geo_items  # Continue to date filter below
    
    return revenues


def _deduplicate_subtotals(revenues: Dict[str, int], tolerance: float = 0.01) -> None:
    """
    Remove subtotal entries when their component items exist.
    
    For example, if we have:
      - Advertising: $160,633M
      - Other revenue: $1,722M  
      - Family of Apps: $162,355M  (= Advertising + Other)
    
    Then "Family of Apps" is a subtotal and should be removed.
    
    This function modifies the dict in-place.
    """
    if len(revenues) < 3:
        return  # Need at least 3 items to have a subtotal scenario
    
    labels = list(revenues.keys())
    values = list(revenues.values())
    to_remove = set()
    
    # Check each item to see if it equals the sum of other items
    for i, (label, value) in enumerate(zip(labels, values)):
        label_lower = label.lower()
        
        # Skip known granular items
        if any(granular in label_lower for granular in ["advertising", "subscription", "other revenue", "cloud", "aws"]):
            continue
        
        # Check if this is a known subtotal pattern
        is_known_subtotal = any(pattern in label_lower for pattern in _SUBTOTAL_PATTERNS)
        
        # Calculate sum of other items (excluding this one and any already marked for removal)
        other_values = [v for j, (l, v) in enumerate(zip(labels, values)) 
                       if j != i and l not in to_remove and v > 0]
        
        if len(other_values) >= 2:
            other_sum = sum(other_values)
            
            # Check if this value equals the sum of others (within tolerance)
            if other_sum > 0 and abs(value - other_sum) / other_sum <= tolerance:
                # This appears to be a subtotal
                print(f"[dedup] Removing subtotal '{label}' (${value:,} = sum of ${other_sum:,})", flush=True)
                to_remove.add(label)
            elif is_known_subtotal:
                # Known subtotal pattern - check if value matches any subset sum
                for subset_size in range(2, len(other_values)):
                    from itertools import combinations
                    for subset in combinations(other_values, subset_size):
                        subset_sum = sum(subset)
                        if subset_sum > 0 and abs(value - subset_sum) / subset_sum <= tolerance:
                            print(f"[dedup] Removing known subtotal '{label}' (${value:,})", flush=True)
                            to_remove.add(label)
                            break
                    if label in to_remove:
                        break
    
    # Remove identified subtotals
    for label in to_remove:
        del revenues[label]


def _trace_append(t_art: Path, event: Dict[str, Any]) -> None:
    """Append a trace event (JSONL) for audit/debug."""
    trace_path = t_art / "trace.jsonl"
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def extract_from_table(
    *,
    html_path: Path,
    table_info: RevenueTableInfo,
    candidates: List[TableCandidate],
    ticker: str,
    company_name: str,
    expected_segments: List[str],
    llm: OpenAIChatClient,
    t_art: Path,
) -> Optional[Tuple[ExtractionResult, ValidationResult, str]]:
    """
    Extract revenue data from a single table.
    
    Returns: (extraction_result, validation_result, table_id) or None
    """
    table_id = table_info.table_id
    cand = next((c for c in candidates if c.table_id == table_id), None)
    if cand is None:
        return None
    
    try:
        grid = extract_table_grid_normalized(html_path, table_id)
        if not grid:
            return None
        
        # Get table metadata
        table_caption = str(getattr(cand, "caption_text", "") or "")
        table_heading = str(getattr(cand, "heading_context", "") or "")
        row_labels = [r[0] if r else "" for r in (cand.preview or [])]
        
        # Detect dimension
        dimension = detect_dimension(
            caption=table_caption,
            heading=table_heading,
            row_labels=row_labels,
            ticker=ticker,
        )
        
        # Infer layout using LLM
        layout = infer_disaggregation_layout(
            llm,
            ticker=ticker,
            company_name=company_name,
            table_id=table_id,
            candidate=cand,
            grid=grid,
            business_lines=expected_segments,
        )
        
        # Validate item column choice
        llm_item_col = layout.get("item_col")
        header_rows_list = layout.get("header_rows", [])
        validated_col, col_reason = choose_item_col(
            grid,
            header_rows=header_rows_list,
            llm_proposed_col=llm_item_col,
        )
        
        if validated_col != llm_item_col:
            print(f"[{ticker}] item_col override: {col_reason}", flush=True)
            layout["item_col"] = validated_col
        
        # Extract revenue data
        result = extract_with_layout_fallback(
            grid,
            expected_segments=expected_segments,
            llm_layout=layout,
            ticker=ticker,
            prefer_granular=True,
            dimension=dimension,
            caption=table_caption,
            heading=table_heading,
        )
        
        if result is None or not result.segment_revenues:
            _trace_append(t_art, {"stage": "extract_fail", "table_id": table_id, "reason": "no_segments"})
            return None
        
        # QA: Validate extracted labels
        extracted_labels = [r.item for r in result.rows if r.row_type in ("item", "segment", "unknown")]
        labels_valid, labels_reason = validate_extracted_labels(extracted_labels, threshold=0.5)
        if not labels_valid:
            print(f"[{ticker}] REJECT table {table_id}: {labels_reason}", flush=True)
            return None
        
        # Simple validation (no external total check yet - that happens in reconciliation)
        validation = validate_extraction(
            segment_revenues=result.segment_revenues,
            adjustment_revenues=result.adjustment_revenues,
            table_total=result.table_total,
            external_total=None,  # Skip external check here
            tolerance_pct=0.02,
        )
        
        return (result, validation, table_id)
        
    except Exception as e:
        _trace_append(t_art, {"stage": "extract_error", "table_id": table_id, "error": str(e)})
        return None


def run_pipeline_v2(
    *,
    tickers: List[str],
    out_dir: Path | str = Path("data/outputs_v2"),
    filings_base_dir: Path | str = Path("data/10k"),
    cache_dir: Path | str = Path(".cache/sec"),
    model: str = "gpt-4.1-mini",
    reconciliation_tolerance: float = 0.05,
) -> Dict[str, Any]:
    """
    Run the v2 pipeline with the new architecture.
    
    Key differences from v1:
    1. Extracts income statement anchor FIRST
    2. Uses document scout to find ALL revenue tables
    3. Follows priority hierarchy (A→B→C→D)
    4. Reconciles against income statement total
    5. Can aggregate from multiple tables if needed
    
    Args:
        tickers: List of stock tickers to process
        out_dir: Output directory
        filings_base_dir: Directory with 10-K filings
        cache_dir: Cache directory for SEC downloads
        model: OpenAI model to use
        reconciliation_tolerance: Acceptable gap vs income statement (default 5%)
    
    Returns:
        Report dict with results per ticker
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    filings_base_dir = Path(filings_base_dir).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    
    llm = OpenAIChatClient(model=model, rate_limit_rpm=60.0)
    
    artifacts_dir = out_dir.parent / ".artifacts_v2"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    csv1_rows: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {"tickers": {}, "outputs_dir": str(out_dir)}
    
    for t in tickers:
        ticker = str(t).upper().strip()
        if not ticker:
            continue
        
        per = {
            "ok": False,
            "errors": [],
            "phases": {},
            "artifacts_dir": str(artifacts_dir / ticker),
        }
        report["tickers"][ticker] = per
        t_art = artifacts_dir / ticker
        t_art.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"\n{'='*60}", flush=True)
            print(f"[{ticker}] Starting v2 pipeline", flush=True)
            print(f"{'='*60}", flush=True)
            
            # Get filing
            filing_dir = _ensure_filing_dir(ticker, base_dir=filings_base_dir, cache_dir=cache_dir)
            html_path = find_primary_document_html(filing_dir)
            company_name = _read_company_name_from_submission(filing_dir) or ticker
            filing_ref = _read_filing_ref(filing_dir)
            cik = int(filing_ref["cik"])
            
            # Extract all table candidates
            candidates = extract_table_candidates_from_html(html_path)
            print(f"[{ticker}] Found {len(candidates)} table candidates", flush=True)
            
            # ================================================================
            # PHASE 0: Income Statement Anchor
            # ================================================================
            print(f"[{ticker}] Phase 0: Extracting income statement anchor...", flush=True)
            
            income_anchor = extract_income_statement_anchor(html_path, candidates=candidates)
            
            if income_anchor:
                print(f"[{ticker}] Income statement total: ${income_anchor.total_revenue:,} ({income_anchor.units_hint})", flush=True)
                print(f"[{ticker}]   Label: '{income_anchor.total_revenue_label}'", flush=True)
                print(f"[{ticker}]   Year: {income_anchor.fiscal_year}", flush=True)
                per["phases"]["income_statement"] = {
                    "total_revenue": income_anchor.total_revenue,
                    "label": income_anchor.total_revenue_label,
                    "fiscal_year": income_anchor.fiscal_year,
                    "confidence": income_anchor.confidence,
                }
                _trace_append(t_art, {"stage": "phase0_income_anchor", "total": income_anchor.total_revenue, "label": income_anchor.total_revenue_label})
            else:
                # Try SEC API as fallback
                print(f"[{ticker}] Could not extract income statement anchor, trying SEC API...", flush=True)
                external_total = fetch_companyfacts_total_revenue_usd(cik, fiscal_year=2024)
                if external_total:
                    print(f"[{ticker}] SEC API total revenue: ${external_total:,}", flush=True)
                    # Create a synthetic anchor
                    from revseg.income_statement import IncomeStatementAnchor
                    income_anchor = IncomeStatementAnchor(
                        total_revenue=external_total,
                        total_revenue_label="(from SEC API)",
                        fiscal_year=2024,  # Approximate
                        source_table_id="",
                        confidence=0.7,
                        units_hint="millions",
                    )
                    per["phases"]["income_statement"] = {
                        "total_revenue": external_total,
                        "source": "SEC_API",
                    }
                else:
                    print(f"[{ticker}] WARNING: No income statement anchor available", flush=True)
                    per["phases"]["income_statement"] = {"error": "not_found"}
            
            # ================================================================
            # PHASE 1: Document Scout
            # ================================================================
            print(f"[{ticker}] Phase 1: Scouting document structure...", flush=True)
            
            doc_map = scout_document(html_path, ticker=ticker, candidates=candidates)
            
            print(f"[{ticker}] Document sections found:", flush=True)
            print(f"[{ticker}]   Item 1 (Business): {doc_map.has_item1_business}", flush=True)
            print(f"[{ticker}]   Item 7 (MD&A): {doc_map.has_item7_mda}", flush=True)
            print(f"[{ticker}]   Item 8 (Notes): {doc_map.has_item8_notes}", flush=True)
            print(f"[{ticker}]   Note 2 (Revenue): {doc_map.has_note2_revenue}", flush=True)
            print(f"[{ticker}]   Segment Note: {doc_map.has_segment_note}", flush=True)
            
            print(f"[{ticker}] Revenue tables found:", flush=True)
            print(f"[{ticker}]   Priority A (Disaggregation): {len(doc_map.priority_a_tables)}", flush=True)
            print(f"[{ticker}]   Priority B (Segment): {len(doc_map.priority_b_tables)}", flush=True)
            print(f"[{ticker}]   Priority C (MD&A): {len(doc_map.priority_c_tables)}", flush=True)
            
            per["phases"]["document_scout"] = {
                "priority_a_count": len(doc_map.priority_a_tables),
                "priority_b_count": len(doc_map.priority_b_tables),
                "priority_c_count": len(doc_map.priority_c_tables),
                "total_tables": len(doc_map.all_revenue_tables),
            }
            
            # Save document map
            doc_map_dict = {
                "ticker": ticker,
                "sections": {
                    "item1_business": doc_map.has_item1_business,
                    "item7_mda": doc_map.has_item7_mda,
                    "item8_notes": doc_map.has_item8_notes,
                    "note2_revenue": doc_map.has_note2_revenue,
                    "segment_note": doc_map.has_segment_note,
                },
                "priority_a_tables": [
                    {"table_id": t.table_id, "type": t.table_type, "confidence": t.confidence, "caption": t.caption[:100]}
                    for t in doc_map.priority_a_tables
                ],
                "priority_b_tables": [
                    {"table_id": t.table_id, "type": t.table_type, "confidence": t.confidence, "caption": t.caption[:100]}
                    for t in doc_map.priority_b_tables
                ],
                "priority_c_tables": [
                    {"table_id": t.table_id, "type": t.table_type, "confidence": t.confidence, "caption": t.caption[:100]}
                    for t in doc_map.priority_c_tables
                ],
            }
            (t_art / "document_map.json").write_text(json.dumps(doc_map_dict, indent=2), encoding="utf-8")
            _trace_append(t_art, {"stage": "phase1_document_scout", "tables_found": len(doc_map.all_revenue_tables)})
            
            # ================================================================
            # PHASE 2: Table Selection (Priority A → B → C)
            # ================================================================
            print(f"[{ticker}] Phase 2: Selecting tables for extraction...", flush=True)
            
            income_total = income_anchor.total_revenue if income_anchor else None
            selected_tables = select_tables_for_extraction(
                doc_map,
                income_statement_total=income_total,
                max_tables=10,  # Get more tables, prioritization will filter
            )
            
            if not selected_tables:
                per["errors"].append("No revenue tables found in document")
                continue
            
            # Filter tables to prioritize those with reasonable revenue values
            selected_tables = _prioritize_tables_with_revenue(
                selected_tables, candidates, html_path, income_anchor, ticker
            )
            
            print(f"[{ticker}] Selected {len(selected_tables)} tables:", flush=True)
            for i, table in enumerate(selected_tables):
                print(f"[{ticker}]   {i+1}. {table.table_id} (Priority {table.priority}, {table.table_type})", flush=True)
            
            # Infer expected segments from table labels
            expected_segments: List[str] = []
            for table in selected_tables:
                expected_segments.extend(table.row_labels_preview)
            expected_segments = list(set(s for s in expected_segments if s and len(s) > 2))
            
            # ================================================================
            # PHASE 2b: Extract from selected tables
            # ================================================================
            all_extractions: List[Tuple[ExtractionResult, str]] = []
            combined_revenues: Dict[str, int] = {}
            combined_adjustments: Dict[str, int] = {}
            
            for table_info in selected_tables:
                print(f"[{ticker}] Extracting from table {table_info.table_id}...", flush=True)
                
                result = extract_from_table(
                    html_path=html_path,
                    table_info=table_info,
                    candidates=candidates,
                    ticker=ticker,
                    company_name=company_name,
                    expected_segments=expected_segments,
                    llm=llm,
                    t_art=t_art,
                )
                
                if result:
                    extraction, validation, table_id = result
                    all_extractions.append((extraction, table_id))
                    
                    # Use granular rows instead of aggregated segment_revenues
                    # This gives us Advertising + Other revenue instead of Family of Apps
                    for row in extraction.rows:
                        if row.row_type == "total":
                            continue
                        clean_label = _clean_revenue_line(row.item)
                        if clean_label not in combined_revenues:
                            if row.row_type == "adjustment":
                                combined_adjustments[clean_label] = row.value
                            else:
                                combined_revenues[clean_label] = row.value
                    
                    print(f"[{ticker}]   Extracted {len(extraction.rows)} items from {table_id}", flush=True)
            
            # Smart deduplication: Remove subtotals when granular items exist
            # E.g., if we have "Advertising" + "Other revenue" = $162B, and also "Family of Apps" = $162B,
            # then "Family of Apps" is a subtotal and should be removed
            _deduplicate_subtotals(combined_revenues)
            
            # Filter out non-revenue items (membership counts, percentages, etc.)
            combined_revenues = _filter_non_revenue_items(combined_revenues, ticker)
            
            # Units auto-detection: Check if extracted values need scaling
            if income_anchor and combined_revenues:
                combined_revenues, combined_adjustments = _auto_scale_units(
                    combined_revenues,
                    combined_adjustments,
                    income_anchor.total_revenue,
                    ticker,
                )
            
            # Remove duplicate revenue breakdowns (e.g., segment + geography views of same total)
            combined_revenues = _deduplicate_alternate_views(combined_revenues, income_anchor, ticker)
            
            if not all_extractions:
                per["errors"].append("No successful extractions from any table")
                continue
            
            # ================================================================
            # PHASE 3: Reconciliation
            # ================================================================
            print(f"[{ticker}] Phase 3: Reconciling against income statement...", flush=True)
            
            # Filter combined_revenues for reconciliation (same logic as CSV output)
            _COST_MARGIN_PATTERN_RECON = re.compile(
                r'(cost\s+of|costs?\s+and\s+expenses?|gross\s+margin|operating\s+(income|expense|loss|margin)|'
                r'research\s+and\s+development|selling.*general|amortization|'
                r'restructuring|gross\s+profit|total\s+cost|percent|'
                r'total\s+operating|income\s+(from\s+operations|\(loss\)\s+from|before)|net\s+income|depreciation|'
                r'provision\s+for|earnings?\s+per|interest\s+(and\s+other|expense|income)|'
                r'income\s+\(loss\)\s+from|^\s*income\s+from\s+operations)',
                re.IGNORECASE
            )
            filtered_revenues_for_recon = {
                label: value for label, value in combined_revenues.items()
                if not _COST_MARGIN_PATTERN_RECON.search(label)
                and label.lower() not in ("revenue", "revenues", "total revenue", "net revenue")
            }
            
            if income_anchor:
                reconciliation = reconcile_extraction(
                    income_statement_anchor=income_anchor,
                    extracted_items=filtered_revenues_for_recon,
                    adjustment_items=combined_adjustments,
                    ticker=ticker,
                    doc_map=doc_map,
                    tolerance=reconciliation_tolerance,
                )
                
                print(f"[{ticker}] Reconciliation status: {reconciliation.status}", flush=True)
                print(f"[{ticker}]   Income statement: ${reconciliation.income_statement_total:,}", flush=True)
                print(f"[{ticker}]   Extracted total:  ${reconciliation.extracted_total:,}", flush=True)
                print(f"[{ticker}]   Gap: ${reconciliation.gap_amount:,} ({reconciliation.gap_percent*100:.1f}%)", flush=True)
                
                if reconciliation.status != "PASS":
                    print(f"[{ticker}]   Notes: {reconciliation.notes}", flush=True)
                    
                    # Get suggestions for missing components
                    html_text = html_path.read_text(encoding="utf-8", errors="ignore")
                    suggestions = suggest_missing_components(reconciliation, doc_map=doc_map, html_text=html_text[:100000])
                    
                    if suggestions:
                        print(f"[{ticker}]   Suggestions for missing components:", flush=True)
                        for s in suggestions[:3]:
                            print(f"[{ticker}]     - {s.get('description', '')[:100]}", flush=True)
                
                per["phases"]["reconciliation"] = {
                    "status": reconciliation.status,
                    "income_total": reconciliation.income_statement_total,
                    "extracted_total": reconciliation.extracted_total,
                    "gap_amount": reconciliation.gap_amount,
                    "gap_percent": reconciliation.gap_percent,
                    "confidence": reconciliation.confidence,
                }
                
                # Save reconciliation
                recon_dict = {
                    "status": reconciliation.status,
                    "income_statement_total": reconciliation.income_statement_total,
                    "extracted_total": reconciliation.extracted_total,
                    "gap_amount": reconciliation.gap_amount,
                    "gap_percent": reconciliation.gap_percent,
                    "extracted_items": reconciliation.extracted_items,
                    "adjustment_items": reconciliation.adjustment_items,
                    "notes": reconciliation.notes,
                }
                (t_art / "reconciliation.json").write_text(json.dumps(recon_dict, indent=2), encoding="utf-8")
            else:
                reconciliation = None
                print(f"[{ticker}] Skipping reconciliation (no income statement anchor)", flush=True)
            
            # ================================================================
            # Generate CSV1 output
            # ================================================================
            primary_extraction, primary_table_id = all_extractions[0]
            year = primary_extraction.year
            
            # Filter patterns for non-revenue items
            _COST_MARGIN_PATTERN = re.compile(
                r'(cost\s+of|costs?\s+and\s+expenses?|gross\s+margin|operating\s+(income|expense|loss|margin)|'
                r'research\s+and\s+development|selling.*general|amortization|'
                r'restructuring|gross\s+profit|total\s+cost|percent|'
                r'total\s+operating|income\s+(from\s+operations|\(loss\)\s+from|before)|net\s+income|depreciation|'
                r'provision\s+for|earnings?\s+per|interest\s+(and\s+other|expense|income)|'
                r'income\s+\(loss\)\s+from|^\s*income\s+from\s+operations)',
                re.IGNORECASE
            )
            
            # Build CSV1 rows from filtered_revenues_for_recon (already filtered, scaled, deduplicated)
            # This ensures CSV output matches reconciliation totals
            for label, value in filtered_revenues_for_recon.items():
                # Get dimension/segment info from original extraction if available
                dimension = "segment"  # Default
                segment = label
                for extraction, _ in all_extractions:
                    for row in extraction.rows:
                        if _clean_revenue_line(row.item).lower() == label.lower():
                            dimension = row.dimension
                            segment = row.segment
                            break
                
                revenue_group = _get_revenue_group(
                    ticker,
                    label,
                    dimension,
                    segment,
                )
                
                csv1_rows.append({
                    "Company Name": company_name,
                    "Ticker": ticker,
                    "Revenue Group": revenue_group,
                    "Revenue Line": label,
                    "Line Item description (company language)": "",  # Skip descriptions for now
                    f"Revenue (FY{year}, $m)": _to_millions(value),
                })
            
            # Also add adjustment items
            for label, value in combined_adjustments.items():
                csv1_rows.append({
                    "Company Name": company_name,
                    "Ticker": ticker,
                    "Revenue Group": "Adjustments",
                    "Revenue Line": label,
                    "Line Item description (company language)": "",
                    f"Revenue (FY{year}, $m)": _to_millions(value),
                })
            
            per["ok"] = True
            per["year"] = year
            per["n_rows"] = len([r for r in csv1_rows if r["Ticker"] == ticker])
            per["tables_used"] = [t_id for _, t_id in all_extractions]
            
            print(f"[{ticker}] Complete: {per['n_rows']} revenue lines extracted", flush=True)
            
        except Exception as e:
            import traceback
            per["errors"].append(str(e))
            per["traceback"] = traceback.format_exc()
            print(f"[{ticker}] ERROR: {e}", flush=True)
    
    # Write output CSV
    if csv1_rows:
        year_col = next((k for k in csv1_rows[0].keys() if k.startswith("Revenue (FY")), "Revenue ($m)")
        csv1_path = out_dir / "csv1_segment_revenue.csv"
        _write_csv(
            csv1_path,
            ["Company Name", "Ticker", "Revenue Group", "Revenue Line", "Line Item description (company language)", year_col],
            csv1_rows,
        )
        print(f"\nWrote {len(csv1_rows)} rows to {csv1_path}", flush=True)
    
    # Write report
    report_path = out_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    
    # Summary
    print(f"\n{'='*60}", flush=True)
    print("PIPELINE V2 SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    
    passed = sum(1 for t in report["tickers"].values() if t.get("ok"))
    total = len(report["tickers"])
    print(f"Success rate: {passed}/{total} ({passed/total*100:.0f}%)", flush=True)
    
    for ticker, result in report["tickers"].items():
        recon = result.get("phases", {}).get("reconciliation", {})
        status = recon.get("status", "N/A")
        gap = recon.get("gap_percent", 0) * 100
        icon = "PASS" if result.get("ok") else "FAIL"
        print(f"  [{icon}] {ticker}: {status} (gap: {gap:.1f}%)", flush=True)
    
    return report


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Revenue Classifier Pipeline v2")
    parser.add_argument("--tickers", required=True, help="Comma-separated list of tickers")
    parser.add_argument("--out-dir", default="data/outputs_v2", help="Output directory")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Reconciliation tolerance")
    
    args = parser.parse_args()
    
    tickers = [t.strip() for t in args.tickers.split(",")]
    
    run_pipeline_v2(
        tickers=tickers,
        out_dir=args.out_dir,
        model=args.model,
        reconciliation_tolerance=args.tolerance,
    )
