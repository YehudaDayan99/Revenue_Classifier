"""
Phase 1: Document Scout Agent

This module maps the document structure and identifies ALL revenue-related
tables, following the priority hierarchy from Financial_Analyst_Prompt.md:

Priority A: "Disaggregation of Revenue" Note
Priority B: Segment Note Detail  
Priority C: MD&A / Operating Review Tables
Priority D: Narrative Text

The scout returns a structured map of the document with all revenue sources.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

from revseg.table_candidates import (
    TableCandidate,
    extract_table_candidates_from_html,
    extract_table_grid_normalized,
    _get_cached_soup,
)
from revseg.table_kind import tablekind_gate


@dataclass
class RevenueTableInfo:
    """Information about a revenue-related table."""
    table_id: str
    priority: str           # "A", "B", "C", "D"
    location: str           # "Note 2 - Revenue", "Segment Information", "MD&A", etc.
    table_type: str         # "disaggregation", "segment", "product_service", "geography"
    estimated_total: Optional[int]  # Estimated total from the table
    row_count: int
    confidence: float
    caption: str
    row_labels_preview: List[str]


@dataclass
class DocumentMap:
    """Complete map of document structure and revenue sources."""
    ticker: str
    filing_path: str
    
    # Document sections found
    has_item1_business: bool = False
    has_item7_mda: bool = False
    has_item8_notes: bool = False
    has_note2_revenue: bool = False
    has_segment_note: bool = False
    
    # Revenue tables by priority
    priority_a_tables: List[RevenueTableInfo] = field(default_factory=list)  # Disaggregation
    priority_b_tables: List[RevenueTableInfo] = field(default_factory=list)  # Segment Note
    priority_c_tables: List[RevenueTableInfo] = field(default_factory=list)  # MD&A
    priority_d_narratives: List[str] = field(default_factory=list)           # Text extracts
    
    # All revenue tables (combined and sorted)
    all_revenue_tables: List[RevenueTableInfo] = field(default_factory=list)


# Patterns for document sections
ITEM1_PATTERNS = [
    re.compile(r"\bitem\s*1[.\s]+business\b", re.IGNORECASE),
    re.compile(r"\bitem\s*1\b(?![\dA])", re.IGNORECASE),
]

ITEM7_PATTERNS = [
    re.compile(r"\bitem\s*7[.\s]+management", re.IGNORECASE),
    re.compile(r"\bmanagement['']?s?\s+discussion\s+and\s+analysis\b", re.IGNORECASE),
    re.compile(r"\bmd&a\b", re.IGNORECASE),
]

ITEM8_PATTERNS = [
    re.compile(r"\bitem\s*8[.\s]+financial\s+statements\b", re.IGNORECASE),
    re.compile(r"\bnotes\s+to\s+(?:the\s+)?(?:consolidated\s+)?financial\s+statements\b", re.IGNORECASE),
]

NOTE2_REVENUE_PATTERNS = [
    re.compile(r"\bnote\s*2[.\s:—–-]+\s*revenue\b", re.IGNORECASE),
    re.compile(r"\b(?:note|notes)\s+\d+[.\s:—–-]+\s*revenue\s+(?:recognition|from\s+contracts)\b", re.IGNORECASE),
    re.compile(r"\brevenue\s+recognition\b", re.IGNORECASE),
    re.compile(r"\bcontracts?\s+with\s+customers?\b", re.IGNORECASE),
]

SEGMENT_NOTE_PATTERNS = [
    re.compile(r"\bnote\s*\d+[.\s:—–-]+\s*segment\b", re.IGNORECASE),
    re.compile(r"\bsegment\s+(?:information|reporting|results)\b", re.IGNORECASE),
    re.compile(r"\breportable\s+segments?\b", re.IGNORECASE),
]

# Priority A: Disaggregation of Revenue patterns
DISAGGREGATION_PATTERNS = [
    re.compile(r"disaggregat(?:ed|ion)\s+(?:of\s+)?revenue", re.IGNORECASE),
    re.compile(r"revenue\s+(?:from\s+external\s+customers\s+)?by\s+(?:major\s+)?(?:products?\s+(?:and|&)\s+services?|category)", re.IGNORECASE),
    re.compile(r"groups?\s+of\s+similar\s+products?\s+(?:and|&)\s+services?", re.IGNORECASE),
    re.compile(r"net\s+sales\s+by\s+(?:category|product|reportable\s+segment)", re.IGNORECASE),
    re.compile(r"revenue\s+by\s+source", re.IGNORECASE),
    re.compile(r"sources?\s+of\s+revenue", re.IGNORECASE),
    # META-specific: revenue by segment showing FoA vs RL breakdown
    re.compile(r"family\s+of\s+apps.*reality\s+labs", re.IGNORECASE),
    re.compile(r"revenue\s+by\s+user\s+geography", re.IGNORECASE),  # May be geography but useful
]

# Priority B: Segment detail patterns
SEGMENT_DETAIL_PATTERNS = [
    re.compile(r"revenue\s+by\s+(?:reportable\s+)?segment", re.IGNORECASE),
    re.compile(r"segment\s+revenue", re.IGNORECASE),
    re.compile(r"operating\s+segment", re.IGNORECASE),
    re.compile(r"segment\s+information", re.IGNORECASE),
    re.compile(r"reportable\s+segment", re.IGNORECASE),
]

# Priority C: MD&A table patterns
MDA_TABLE_PATTERNS = [
    re.compile(r"results\s+of\s+operations", re.IGNORECASE),
    re.compile(r"revenue\s+(?:analysis|summary|breakdown)", re.IGNORECASE),
    re.compile(r"net\s+sales\s+(?:and|by)", re.IGNORECASE),
]

# Geography patterns (to deprioritize)
GEOGRAPHY_PATTERNS = [
    re.compile(r"revenue\s+by\s+geograph", re.IGNORECASE),
    re.compile(r"geographic\s+(?:area|region|breakdown)", re.IGNORECASE),
    re.compile(r"by\s+(?:country|region)", re.IGNORECASE),
]


def _detect_document_sections(html_text: str) -> Dict[str, bool]:
    """Detect which major sections are present in the document."""
    text_lower = html_text.lower()
    
    return {
        "has_item1_business": any(p.search(text_lower) for p in ITEM1_PATTERNS),
        "has_item7_mda": any(p.search(text_lower) for p in ITEM7_PATTERNS),
        "has_item8_notes": any(p.search(text_lower) for p in ITEM8_PATTERNS),
        "has_note2_revenue": any(p.search(text_lower) for p in NOTE2_REVENUE_PATTERNS),
        "has_segment_note": any(p.search(text_lower) for p in SEGMENT_NOTE_PATTERNS),
    }


def _classify_revenue_table(candidate: TableCandidate) -> Tuple[str, str, float]:
    """
    Classify a table candidate into priority and type.
    
    Returns: (priority, table_type, confidence)
    """
    # Combine context text
    context = " ".join([
        str(getattr(candidate, "caption_text", "") or ""),
        str(getattr(candidate, "heading_context", "") or ""),
        str(getattr(candidate, "nearby_text_context", "") or ""),
    ]).lower()
    
    # Also check row labels
    row_labels = getattr(candidate, "row_label_preview", []) or []
    labels_text = " ".join(str(l) for l in row_labels).lower()
    combined = f"{context} {labels_text}"
    
    # REJECT: Income statement / P&L tables (we want revenue BREAKDOWN, not the P&L)
    income_statement_patterns = [
        re.compile(r"consolidated\s+statements?\s+of\s+(operations|income|earnings)", re.IGNORECASE),
        re.compile(r"statements?\s+of\s+consolidated\s+(operations|income)", re.IGNORECASE),
        re.compile(r"\bnet\s+income\b.*\bearnings\s+per\s+share\b", re.IGNORECASE),
    ]
    # Check for income statement / segment results markers in row labels
    # These indicate P&L-style tables, not revenue breakdowns
    income_stmt_row_markers = ["cost of revenue", "gross profit", "operating income", 
                                "net income", "earnings per share", "provision for income tax",
                                "costs and expenses", "income from operations", "income (loss) from operations",
                                "operating margin", "operating loss"]
    income_marker_count = sum(1 for m in income_stmt_row_markers if m in labels_text)
    
    for pattern in income_statement_patterns:
        if pattern.search(context):
            return ("", "", 0.0)  # Reject
    
    # Reject if too many P&L markers (segment results table, not revenue breakdown)
    if income_marker_count >= 2:
        return ("", "", 0.0)  # Reject - this is an income statement or segment results table
    
    # Check for geography (to deprioritize)
    is_geography = any(p.search(combined) for p in GEOGRAPHY_PATTERNS)
    
    # Priority A: Disaggregation of Revenue
    for pattern in DISAGGREGATION_PATTERNS:
        if pattern.search(context):
            table_type = "geography" if is_geography else "disaggregation"
            confidence = 0.7 if is_geography else 0.95
            return ("A", table_type, confidence)
    
    # Priority B: Segment Note Detail
    for pattern in SEGMENT_DETAIL_PATTERNS:
        if pattern.search(context):
            table_type = "geography" if is_geography else "segment"
            confidence = 0.6 if is_geography else 0.85
            return ("B", table_type, confidence)
    
    # Priority C: MD&A Tables
    for pattern in MDA_TABLE_PATTERNS:
        if pattern.search(context):
            table_type = "geography" if is_geography else "product_service"
            confidence = 0.5 if is_geography else 0.75
            return ("C", table_type, confidence)
    
    # Check for revenue keywords in general
    if "revenue" in combined or "net sales" in combined:
        table_type = "geography" if is_geography else "unknown"
        return ("C", table_type, 0.5)
    
    return ("", "", 0.0)


def _estimate_table_total(
    html_path: Path,
    table_id: str,
    units_mult: int = 1_000_000,
) -> Optional[int]:
    """
    Estimate the total revenue from a table by looking for a Total row.
    """
    try:
        grid = extract_table_grid_normalized(html_path, table_id)
        if not grid:
            return None
        
        # Look for Total row
        total_patterns = [
            re.compile(r"^\s*total\s+(net\s+)?(revenue|sales)", re.IGNORECASE),
            re.compile(r"^\s*total\s*$", re.IGNORECASE),
        ]
        
        # Find year columns
        year_re = re.compile(r"\b(20\d{2})\b")
        year_cols: Dict[int, int] = {}
        for row in grid[:10]:
            for c_i, cell in enumerate(row):
                m = year_re.search(str(cell or ""))
                if m:
                    y = int(m.group(1))
                    if 2018 <= y <= 2030:
                        year_cols.setdefault(y, c_i)
        
        if not year_cols:
            return None
        
        year = max(year_cols.keys())
        val_col = year_cols[year]
        
        for row in grid:
            if not row:
                continue
            label = str(row[0] or "").strip()
            for pattern in total_patterns:
                if pattern.search(label):
                    # Parse value
                    for offset in range(3):
                        col = val_col + offset
                        if col < len(row):
                            t = str(row[col] or "").strip()
                            t = t.replace("$", "").replace(",", "").strip()
                            if t and t not in {"-", "—", "–"}:
                                try:
                                    return int(float(t) * units_mult)
                                except ValueError:
                                    continue
                    break
        
        return None
    except Exception:
        return None


def scout_document(
    html_path: Path,
    *,
    ticker: str = "",
    candidates: Optional[List[TableCandidate]] = None,
) -> DocumentMap:
    """
    Scout the document structure and identify all revenue-related tables.
    
    This is Phase 1 of the pipeline - building a complete map of revenue
    sources before selection.
    
    Args:
        html_path: Path to the 10-K HTML file
        ticker: Company ticker
        candidates: Pre-extracted table candidates (optional)
    
    Returns:
        DocumentMap with all identified revenue tables and document structure
    """
    # Read HTML
    html_text = html_path.read_text(encoding="utf-8", errors="ignore")
    
    # Detect document sections
    sections = _detect_document_sections(html_text)
    
    # Get table candidates
    if candidates is None:
        candidates = extract_table_candidates_from_html(html_path)
    
    # Classify each candidate
    priority_a: List[RevenueTableInfo] = []
    priority_b: List[RevenueTableInfo] = []
    priority_c: List[RevenueTableInfo] = []
    
    for c in candidates:
        # Apply deterministic gate first
        if not tablekind_gate(c):
            continue
        
        priority, table_type, confidence = _classify_revenue_table(c)
        
        if not priority:
            continue
        
        # Estimate total if possible
        estimated_total = _estimate_table_total(html_path, c.table_id)
        
        info = RevenueTableInfo(
            table_id=c.table_id,
            priority=priority,
            location=str(getattr(c, "heading_context", "") or "")[:100],
            table_type=table_type,
            estimated_total=estimated_total,
            row_count=c.n_rows,
            confidence=confidence,
            caption=str(getattr(c, "caption_text", "") or "")[:200],
            row_labels_preview=list(getattr(c, "row_label_preview", []) or [])[:10],
        )
        
        if priority == "A":
            priority_a.append(info)
        elif priority == "B":
            priority_b.append(info)
        elif priority == "C":
            priority_c.append(info)
    
    # Sort each priority list by confidence
    priority_a.sort(key=lambda x: x.confidence, reverse=True)
    priority_b.sort(key=lambda x: x.confidence, reverse=True)
    priority_c.sort(key=lambda x: x.confidence, reverse=True)
    
    # Combine all tables (maintaining priority order)
    all_tables = priority_a + priority_b + priority_c
    
    return DocumentMap(
        ticker=ticker,
        filing_path=str(html_path),
        has_item1_business=sections["has_item1_business"],
        has_item7_mda=sections["has_item7_mda"],
        has_item8_notes=sections["has_item8_notes"],
        has_note2_revenue=sections["has_note2_revenue"],
        has_segment_note=sections["has_segment_note"],
        priority_a_tables=priority_a,
        priority_b_tables=priority_b,
        priority_c_tables=priority_c,
        priority_d_narratives=[],  # TODO: Implement narrative extraction
        all_revenue_tables=all_tables,
    )


def select_tables_for_extraction(
    doc_map: DocumentMap,
    *,
    income_statement_total: Optional[int] = None,
    max_tables: int = 3,
) -> List[RevenueTableInfo]:
    """
    Select the best tables for extraction based on the document map.
    
    Strategy:
    1. If Priority A (Disaggregation) tables exist, prefer them
    2. If their totals reconcile to income statement, use them
    3. Otherwise, combine with Priority B (Segment) tables
    4. Fall back to Priority C (MD&A) if needed
    
    Args:
        doc_map: Document map from scout_document()
        income_statement_total: Ground truth total from income statement
        max_tables: Maximum number of tables to return
    
    Returns:
        List of selected tables in extraction order
    """
    selected: List[RevenueTableInfo] = []
    total_covered = 0
    
    # Filter out geography tables unless nothing else available
    non_geo_a = [t for t in doc_map.priority_a_tables if t.table_type != "geography"]
    non_geo_b = [t for t in doc_map.priority_b_tables if t.table_type != "geography"]
    non_geo_c = [t for t in doc_map.priority_c_tables if t.table_type != "geography"]
    
    # Prefer non-geography tables
    priority_a = non_geo_a if non_geo_a else doc_map.priority_a_tables
    priority_b = non_geo_b if non_geo_b else doc_map.priority_b_tables
    priority_c = non_geo_c if non_geo_c else doc_map.priority_c_tables
    
    # Try Priority A first
    for table in priority_a:
        if len(selected) >= max_tables:
            break
        selected.append(table)
        if table.estimated_total:
            total_covered += table.estimated_total
    
    # If we have income statement total and haven't covered it, add more tables
    if income_statement_total and total_covered < income_statement_total * 0.95:
        for table in priority_b:
            if len(selected) >= max_tables:
                break
            if table.table_id not in {t.table_id for t in selected}:
                selected.append(table)
                if table.estimated_total:
                    total_covered += table.estimated_total
    
    # If still not covered, try MD&A
    if income_statement_total and total_covered < income_statement_total * 0.95:
        for table in priority_c:
            if len(selected) >= max_tables:
                break
            if table.table_id not in {t.table_id for t in selected}:
                selected.append(table)
    
    # If no tables selected, take best from any priority
    if not selected:
        all_tables = priority_a + priority_b + priority_c
        if all_tables:
            selected = all_tables[:max_tables]
    
    return selected
