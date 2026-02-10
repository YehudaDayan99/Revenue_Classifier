"""
Phase 3: Reconciliation Agent

This module verifies that extracted revenue data reconciles to the
income statement total. If there's a gap, it identifies missing components
and triggers re-search.

Per Financial_Analyst_Prompt.md:
"Identify the table reconciling segment revenue to total consolidated revenue."
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from revseg.income_statement import IncomeStatementAnchor
from revseg.document_scout import DocumentMap, RevenueTableInfo


@dataclass
class ReconciliationGap:
    """Identified gap in the reconciliation."""
    gap_amount: int              # Positive = missing revenue, negative = over-counted
    gap_percent: float           # Gap as percentage of income statement total
    likely_cause: str            # "missing_segment", "missing_product", "contra_revenue", "units_error", "unknown"
    suggested_action: str        # What to do about it
    candidate_tables: List[str]  # Table IDs that might fill the gap


@dataclass
class ReconciliationResult:
    """Result of the reconciliation check."""
    status: str                  # "PASS", "FAIL", "WARNING"
    
    # Totals
    income_statement_total: int
    extracted_total: int
    
    # Gap analysis
    gap_amount: int              # income_statement_total - extracted_total
    gap_percent: float           # Gap as percentage of income statement total
    
    # Components
    extracted_items: Dict[str, int]   # Label -> value
    adjustment_items: Dict[str, int]  # Contra-revenue / adjustments
    
    # If FAIL, what's missing
    gaps: List[ReconciliationGap] = field(default_factory=list)
    
    # Confidence in the reconciliation
    confidence: float = 0.0
    
    notes: str = ""


# Known contra-revenue patterns (these reduce gross to net)
CONTRA_REVENUE_PATTERNS = [
    re.compile(r"\bclient\s+incentive", re.IGNORECASE),           # V/Visa
    re.compile(r"\bcustomer\s+incentive", re.IGNORECASE),
    re.compile(r"\brebate", re.IGNORECASE),
    re.compile(r"\bdiscount", re.IGNORECASE),
    re.compile(r"\ballowance", re.IGNORECASE),
    re.compile(r"\bpromotion", re.IGNORECASE),
    re.compile(r"\bcoupon", re.IGNORECASE),
]

# Known segment patterns for multi-segment companies
KNOWN_SEGMENT_PATTERNS = {
    "META": [
        re.compile(r"\bfamily\s+of\s+apps\b", re.IGNORECASE),
        re.compile(r"\breality\s+labs\b", re.IGNORECASE),
    ],
    "COST": [
        re.compile(r"\bmembership\s+fee", re.IGNORECASE),
        re.compile(r"\bmerchandise\s+sales", re.IGNORECASE),
    ],
    "XOM": [
        re.compile(r"\bupstream\b", re.IGNORECASE),
        re.compile(r"\bdownstream\b", re.IGNORECASE),
        re.compile(r"\bchemical\b", re.IGNORECASE),
    ],
}


def _detect_likely_cause(
    gap_amount: int,
    gap_percent: float,
    ticker: str,
    extracted_labels: List[str],
    doc_map: Optional[DocumentMap] = None,
) -> Tuple[str, str]:
    """
    Analyze the gap and determine likely cause and suggested action.
    
    Returns: (likely_cause, suggested_action)
    """
    # Very large gap might indicate units error
    if abs(gap_percent) > 0.9:
        return ("units_error", "Check units multiplier (thousands vs millions vs billions)")
    
    # Large positive gap = missing revenue
    if gap_percent > 0.05:
        # Check known patterns for this ticker
        if ticker.upper() in KNOWN_SEGMENT_PATTERNS:
            patterns = KNOWN_SEGMENT_PATTERNS[ticker.upper()]
            missing_segments = []
            for pattern in patterns:
                found = any(pattern.search(label) for label in extracted_labels)
                if not found:
                    missing_segments.append(pattern.pattern)
            
            if missing_segments:
                return ("missing_segment", f"Search for segments matching: {', '.join(missing_segments)}")
        
        # Generic suggestion
        return ("missing_segment", "Search for additional revenue segments or product lines")
    
    # Negative gap might be contra-revenue
    if gap_percent < -0.05:
        return ("contra_revenue", "Search for contra-revenue items (incentives, rebates, allowances)")
    
    # Small gap is acceptable
    if abs(gap_percent) <= 0.05:
        return ("rounding", "Gap within acceptable tolerance (5%)")
    
    return ("unknown", "Manual review recommended")


def reconcile_extraction(
    *,
    income_statement_anchor: IncomeStatementAnchor,
    extracted_items: Dict[str, int],
    adjustment_items: Optional[Dict[str, int]] = None,
    ticker: str = "",
    doc_map: Optional[DocumentMap] = None,
    tolerance: float = 0.05,
) -> ReconciliationResult:
    """
    Reconcile extracted revenue items against the income statement total.
    
    This is Phase 3 of the pipeline - verifying extraction completeness.
    
    Args:
        income_statement_anchor: Ground truth from Phase 0
        extracted_items: Dict of label -> revenue value from extraction
        adjustment_items: Dict of label -> adjustment value (can be negative)
        ticker: Company ticker for context-aware analysis
        doc_map: Document map for finding missing tables
        tolerance: Acceptable gap as fraction (default 5%)
    
    Returns:
        ReconciliationResult with status and gap analysis
    """
    adjustment_items = adjustment_items or {}
    
    income_total = income_statement_anchor.total_revenue
    extracted_total = sum(extracted_items.values())
    adjustment_total = sum(adjustment_items.values())
    
    # Net total including adjustments
    net_total = extracted_total + adjustment_total
    
    gap_amount = income_total - net_total
    gap_percent = gap_amount / income_total if income_total else 0.0
    
    # Determine status
    if abs(gap_percent) <= tolerance:
        status = "PASS"
    elif abs(gap_percent) <= 0.15:
        status = "WARNING"
    else:
        status = "FAIL"
    
    # Calculate confidence
    if status == "PASS":
        confidence = 1.0 - abs(gap_percent)
    elif status == "WARNING":
        confidence = max(0.5, 1.0 - abs(gap_percent) * 2)
    else:
        confidence = max(0.1, 0.5 - abs(gap_percent))
    
    # Analyze gap if significant
    gaps: List[ReconciliationGap] = []
    notes = ""
    
    if abs(gap_percent) > tolerance:
        extracted_labels = list(extracted_items.keys()) + list(adjustment_items.keys())
        likely_cause, suggested_action = _detect_likely_cause(
            gap_amount, gap_percent, ticker, extracted_labels, doc_map
        )
        
        # Find candidate tables that might fill the gap
        candidate_tables: List[str] = []
        if doc_map:
            # Look for tables we haven't used yet
            # This would require tracking which tables were used in extraction
            for table in doc_map.all_revenue_tables:
                if table.estimated_total and abs(table.estimated_total - abs(gap_amount)) < abs(gap_amount) * 0.3:
                    candidate_tables.append(table.table_id)
        
        gap_info = ReconciliationGap(
            gap_amount=gap_amount,
            gap_percent=gap_percent,
            likely_cause=likely_cause,
            suggested_action=suggested_action,
            candidate_tables=candidate_tables,
        )
        gaps.append(gap_info)
        
        notes = f"Gap of ${gap_amount:,} ({gap_percent*100:.1f}%): {likely_cause} - {suggested_action}"
    else:
        notes = f"Reconciled within tolerance ({gap_percent*100:.1f}% gap)"
    
    return ReconciliationResult(
        status=status,
        income_statement_total=income_total,
        extracted_total=net_total,
        gap_amount=gap_amount,
        gap_percent=gap_percent,
        extracted_items=extracted_items,
        adjustment_items=adjustment_items,
        gaps=gaps,
        confidence=confidence,
        notes=notes,
    )


def suggest_missing_components(
    reconciliation: ReconciliationResult,
    *,
    doc_map: Optional[DocumentMap] = None,
    html_text: str = "",
) -> List[Dict[str, Any]]:
    """
    Suggest what might be missing based on the reconciliation gap.
    
    Returns: List of suggestions with table_ids or search patterns
    """
    suggestions: List[Dict[str, Any]] = []
    
    if reconciliation.status == "PASS":
        return suggestions
    
    gap = reconciliation.gap_amount
    gap_pct = reconciliation.gap_percent
    
    # Check for likely contra-revenue scenario (over-counted)
    if gap < 0:
        suggestions.append({
            "type": "contra_revenue",
            "description": "Extracted gross revenue exceeds net revenue - look for contra-revenue items",
            "search_patterns": [p.pattern for p in CONTRA_REVENUE_PATTERNS],
        })
    
    # Check for missing segment scenario (under-counted)
    if gap > 0 and doc_map:
        # Find tables we haven't extracted from
        for table in doc_map.all_revenue_tables:
            # If this table has an estimated total close to our gap
            if table.estimated_total:
                match_pct = abs(table.estimated_total - gap) / gap if gap else 0
                if match_pct < 0.5:
                    suggestions.append({
                        "type": "missing_table",
                        "description": f"Table {table.table_id} has estimated total ${table.estimated_total:,} - close to gap",
                        "table_id": table.table_id,
                        "table_type": table.table_type,
                        "match_quality": 1.0 - match_pct,
                    })
    
    # Narrative search suggestions
    if gap > 0 and html_text:
        # Look for revenue amounts in narrative that match our gap
        money_pattern = re.compile(r'\$?([\d,]+(?:\.\d+)?)\s*(?:million|billion)', re.IGNORECASE)
        for match in money_pattern.finditer(html_text):
            try:
                amount_str = match.group(1).replace(",", "")
                amount = float(amount_str)
                
                # Scale based on unit
                unit = match.group(0).lower()
                if "billion" in unit:
                    amount *= 1_000_000_000
                elif "million" in unit:
                    amount *= 1_000_000
                
                # Check if this amount is close to our gap
                if abs(amount - gap) / gap < 0.2 if gap else False:
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(html_text), match.end() + 100)
                    context = html_text[context_start:context_end]
                    
                    suggestions.append({
                        "type": "narrative_amount",
                        "description": f"Found ${amount:,.0f} in narrative near: {context[:100]}...",
                        "amount": amount,
                        "context": context,
                    })
            except (ValueError, TypeError):
                continue
    
    return suggestions


def iterative_reconciliation(
    *,
    income_statement_anchor: IncomeStatementAnchor,
    initial_extraction: Dict[str, int],
    doc_map: DocumentMap,
    html_path: Path,
    ticker: str,
    max_iterations: int = 3,
) -> Tuple[ReconciliationResult, List[Dict[str, int]]]:
    """
    Iteratively try to reconcile by extracting from additional tables.
    
    Args:
        income_statement_anchor: Ground truth
        initial_extraction: First extraction attempt
        doc_map: Document map with all revenue tables
        html_path: Path to filing
        ticker: Company ticker
        max_iterations: Maximum extraction attempts
    
    Returns:
        (final_reconciliation, list_of_extractions)
    """
    all_extractions: List[Dict[str, int]] = [initial_extraction]
    combined_items = dict(initial_extraction)
    used_tables: set = set()
    
    for iteration in range(max_iterations):
        # Reconcile current state
        result = reconcile_extraction(
            income_statement_anchor=income_statement_anchor,
            extracted_items=combined_items,
            ticker=ticker,
            doc_map=doc_map,
        )
        
        if result.status == "PASS":
            return (result, all_extractions)
        
        # If FAIL, try to find missing components
        suggestions = suggest_missing_components(result, doc_map=doc_map)
        
        # Find a table we haven't used yet
        next_table = None
        for suggestion in suggestions:
            if suggestion.get("type") == "missing_table":
                table_id = suggestion.get("table_id")
                if table_id and table_id not in used_tables:
                    next_table = table_id
                    break
        
        if not next_table:
            # No more tables to try
            break
        
        # Extract from the next table (placeholder - would call actual extraction)
        # In practice, this would call the extraction logic for the new table
        used_tables.add(next_table)
        
        # For now, just return what we have
        break
    
    # Final reconciliation
    final_result = reconcile_extraction(
        income_statement_anchor=income_statement_anchor,
        extracted_items=combined_items,
        ticker=ticker,
        doc_map=doc_map,
    )
    
    return (final_result, all_extractions)
