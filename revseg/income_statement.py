"""
Phase 0: Income Statement Anchor Extraction

This module extracts the consolidated total revenue from the Income Statement
(Consolidated Statements of Operations/Income) to serve as the ground truth
for reconciliation.

Per Financial_Analyst_Prompt.md:
"Identify the table reconciling segment revenue to total consolidated revenue."
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

from revseg.table_candidates import TableCandidate, extract_table_candidates_from_html, _get_cached_soup


@dataclass
class IncomeStatementAnchor:
    """Ground truth revenue from the income statement."""
    total_revenue: int              # In base units (USD)
    total_revenue_label: str        # Original label (e.g., "Net sales", "Total revenues")
    fiscal_year: int
    source_table_id: str
    confidence: float               # 0.0 - 1.0
    units_hint: str                 # "millions", "thousands", etc.


# Patterns to identify income statement tables
INCOME_STATEMENT_PATTERNS = [
    re.compile(r"consolidated\s+statements?\s+of\s+(operations|income|earnings)", re.IGNORECASE),
    re.compile(r"statements?\s+of\s+consolidated\s+(operations|income|earnings)", re.IGNORECASE),
    re.compile(r"consolidated\s+(income|operations)\s+statements?", re.IGNORECASE),
]

# Patterns to identify the total revenue row
TOTAL_REVENUE_PATTERNS = [
    # Most specific first
    re.compile(r"^\s*total\s+net\s+(revenue|sales)\s*$", re.IGNORECASE),
    re.compile(r"^\s*net\s+(revenue|sales)\s*$", re.IGNORECASE),
    re.compile(r"^\s*total\s+(revenue|revenues|net\s+sales)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(revenue|revenues)\s*$", re.IGNORECASE),
    re.compile(r"^\s*sales\s+and\s+other\s+operating\s+revenue", re.IGNORECASE),  # XOM
    re.compile(r"^\s*net\s+sales\s+and\s+operating\s+revenues?", re.IGNORECASE),
]

# Patterns for rows that confirm we're in an income statement (used for validation)
INCOME_STATEMENT_MARKER_PATTERNS = [
    re.compile(r"cost\s+of\s+(revenue|sales|goods)", re.IGNORECASE),
    re.compile(r"gross\s+(profit|margin)", re.IGNORECASE),
    re.compile(r"operating\s+(income|expenses?)", re.IGNORECASE),
    re.compile(r"net\s+income", re.IGNORECASE),
    re.compile(r"income\s+before\s+(income\s+)?tax", re.IGNORECASE),
    re.compile(r"earnings?\s+per\s+share", re.IGNORECASE),
]

# Year patterns
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")


def _parse_money(s: str) -> Optional[int]:
    """Parse a money string to integer."""
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


def _detect_units_multiplier(html_text: str, table_context: str) -> int:
    """Detect units multiplier from context (millions, thousands, etc.)."""
    context = f"{html_text[:10000]} {table_context}".lower()
    
    # Look for explicit unit markers near the table
    if "in millions" in context or "(in millions)" in context or "$ in millions" in context:
        return 1_000_000
    if "in thousands" in context or "(in thousands)" in context:
        return 1_000
    if "in billions" in context or "(in billions)" in context:
        return 1_000_000_000
    
    # For income statements, default to millions (most common for large-cap 10-Ks)
    # but check for "except per share" which often accompanies unit disclosure
    if "except per share" in context or "except share" in context:
        # This confirms it's a financial statement with units
        return 1_000_000
    
    # Default to millions (most common for large-cap 10-Ks)
    return 1_000_000


def _is_income_statement_table(candidate: TableCandidate) -> Tuple[bool, float]:
    """
    Check if a table candidate is likely an income statement.
    
    Returns: (is_income_statement, confidence_score)
    """
    # Combine text sources
    context_text = " ".join([
        str(getattr(candidate, "caption_text", "") or ""),
        str(getattr(candidate, "heading_context", "") or ""),
        str(getattr(candidate, "nearby_text_context", "") or ""),
    ]).lower()
    
    # Get row labels preview
    row_labels = getattr(candidate, "row_label_preview", []) or []
    labels_text = " ".join([str(l).lower() for l in row_labels])
    
    # Check caption/heading for income statement patterns
    caption_match = False
    for pattern in INCOME_STATEMENT_PATTERNS:
        if pattern.search(context_text):
            caption_match = True
            break
    
    # Check row labels for income statement markers
    marker_count = 0
    for pattern in INCOME_STATEMENT_MARKER_PATTERNS:
        if pattern.search(labels_text):
            marker_count += 1
    
    # Also check for revenue row specifically
    has_revenue = bool(re.search(r'\b(net\s+)?revenue[s]?\b|\bnet\s+sales\b', labels_text))
    has_net_income = bool(re.search(r'\bnet\s+income\b|\bnet\s+earnings\b', labels_text))
    has_expenses = bool(re.search(r'\boperating\s+expenses?\b|\bcost\s+of\b', labels_text))
    
    # Method 1: Caption match + at least 2 markers
    if caption_match and marker_count >= 2:
        confidence = min(0.5 + (marker_count * 0.15), 0.95)
        return (True, confidence)
    
    # Method 2: No caption, but strong row label indicators
    # Must have revenue + net income + expenses to be confident
    if has_revenue and has_net_income and has_expenses:
        return (True, 0.85)
    
    # Method 3: Caption match only (lower confidence)
    if caption_match:
        return (True, 0.6)
    
    return (False, 0.0)


def _extract_revenue_from_grid(
    grid: List[List[str]],
    *,
    target_year: Optional[int] = None,
    units_mult: int = 1_000_000,
) -> Optional[Tuple[int, str, int]]:
    """
    Extract total revenue value from an income statement grid.
    
    Returns: (revenue_value, label, year) or None
    """
    # Detect year columns
    year_cols: Dict[int, int] = {}
    for r_i, row in enumerate(grid[:10]):
        for c_i, cell in enumerate(row):
            m = YEAR_PATTERN.search(str(cell or ""))
            if m:
                y = int(m.group(1))
                if 2018 <= y <= 2030:
                    year_cols.setdefault(y, c_i)
    
    if not year_cols:
        return None
    
    # Use target year or most recent
    year = target_year if target_year and target_year in year_cols else max(year_cols.keys())
    val_col = year_cols[year]
    
    # Find the revenue row
    for r_i, row in enumerate(grid):
        if not row:
            continue
        
        # First cell is usually the label
        label = str(row[0] or "").strip()
        
        for pattern in TOTAL_REVENUE_PATTERNS:
            if pattern.search(label):
                # Found revenue row - extract value
                if val_col < len(row):
                    val = _parse_money(row[val_col])
                    if val is None and val_col + 1 < len(row):
                        val = _parse_money(row[val_col + 1])
                    if val is None and val_col + 2 < len(row):
                        val = _parse_money(row[val_col + 2])
                    
                    if val is not None and val > 0:
                        # Skip percentage tables (values < 200 are likely percentages)
                        # Real revenue in millions would be at least 200M for public companies
                        if val <= 200:
                            # This looks like a percentage table, skip it
                            break
                        return (val * units_mult, label, year)
                break
    
    return None


def extract_income_statement_anchor(
    html_path: Path,
    *,
    candidates: Optional[List[TableCandidate]] = None,
    target_year: Optional[int] = None,
) -> Optional[IncomeStatementAnchor]:
    """
    Extract the consolidated total revenue from the income statement.
    
    This is Phase 0 of the pipeline - establishing the ground truth
    that all subsequent extractions must reconcile to.
    
    Args:
        html_path: Path to the 10-K HTML file
        candidates: Pre-extracted table candidates (optional, will extract if not provided)
        target_year: Specific fiscal year to target (optional, defaults to most recent)
    
    Returns:
        IncomeStatementAnchor with total revenue, or None if not found
    """
    # Get table candidates
    if candidates is None:
        candidates = extract_table_candidates_from_html(html_path)
    
    # Read HTML for units detection
    html_text = html_path.read_text(encoding="utf-8", errors="ignore")
    
    # Find income statement tables (may be multiple, some could be percentage tables)
    income_stmt_candidates: List[Tuple[TableCandidate, float]] = []
    
    for c in candidates:
        is_income_stmt, confidence = _is_income_statement_table(c)
        if is_income_stmt:
            income_stmt_candidates.append((c, confidence))
    
    # Sort by confidence (highest first)
    income_stmt_candidates.sort(key=lambda x: x[1], reverse=True)
    
    if not income_stmt_candidates:
        return None
    
    from revseg.table_candidates import extract_table_grid_normalized
    
    # Try each candidate until we find one with valid revenue
    for candidate, confidence in income_stmt_candidates:
        grid = extract_table_grid_normalized(html_path, candidate.table_id)
        
        if not grid:
            continue
        
        # Detect units
        table_context = " ".join([
            str(getattr(candidate, "caption_text", "") or ""),
            str(getattr(candidate, "heading_context", "") or ""),
        ])
        units_mult = _detect_units_multiplier(html_text, table_context)
        units_hint = {
            1_000: "thousands",
            1_000_000: "millions",
            1_000_000_000: "billions",
        }.get(units_mult, "units")
        
        # Extract revenue
        result = _extract_revenue_from_grid(grid, target_year=target_year, units_mult=units_mult)
        
        if result is not None:
            revenue_value, revenue_label, year = result
            
            return IncomeStatementAnchor(
                total_revenue=revenue_value,
                total_revenue_label=revenue_label,
                fiscal_year=year,
                source_table_id=candidate.table_id,
                confidence=confidence,
                units_hint=units_hint,
            )
    
    return None


def find_all_income_statement_tables(
    html_path: Path,
    *,
    candidates: Optional[List[TableCandidate]] = None,
) -> List[Tuple[TableCandidate, float]]:
    """
    Find all tables that look like income statements (for debugging/review).
    
    Returns: List of (candidate, confidence) tuples sorted by confidence
    """
    if candidates is None:
        candidates = extract_table_candidates_from_html(html_path)
    
    results = []
    for c in candidates:
        is_income_stmt, confidence = _is_income_statement_table(c)
        if is_income_stmt:
            results.append((c, confidence))
    
    return sorted(results, key=lambda x: x[1], reverse=True)
