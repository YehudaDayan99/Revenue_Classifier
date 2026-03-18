#!/usr/bin/env python3
"""
Parse HTML table to normalized 2D grid.

Extracts table cells from SEC 10-K HTML, handling:
- Cell padding (extends short rows to max width)
- Whitespace normalization
- Currency symbol preservation
- Rowspan/colspan handling (basic)

Usage:
    python parse_grid.py <html_path> <table_id>
    
Output:
    JSON array of arrays (the grid) to stdout
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Optional

try:
    from bs4 import BeautifulSoup, Tag
except ImportError:
    print("Error: beautifulsoup4 required. Install with: pip install beautifulsoup4", file=sys.stderr)
    sys.exit(1)


# Whitespace normalization regex
_WS_RE = re.compile(r"\s+")


def _clean_cell(text: str) -> str:
    """Normalize whitespace and strip cell text."""
    return _WS_RE.sub(" ", (text or "").strip())


def _extract_table_by_id(soup: BeautifulSoup, table_id: str) -> Optional[Tag]:
    """
    Find table by ID pattern (e.g., 't0042').
    
    Searches for tables and assigns sequential IDs starting from t0000.
    Returns the table matching the requested ID.
    """
    tables = soup.find_all("table")
    for i, table in enumerate(tables):
        tid = f"t{i:04d}"
        if tid == table_id:
            return table
    return None


def _parse_table_to_grid(table: Tag) -> List[List[str]]:
    """
    Convert HTML table to 2D grid of strings.
    
    Handles basic rowspan/colspan by repeating values.
    Returns list of rows, each row is list of cell strings.
    """
    rows = table.find_all("tr")
    if not rows:
        return []
    
    # First pass: determine grid dimensions
    max_cols = 0
    for row in rows:
        cells = row.find_all(["td", "th"])
        col_count = sum(int(cell.get("colspan", 1)) for cell in cells)
        max_cols = max(max_cols, col_count)
    
    # Initialize grid with empty strings
    grid: List[List[Optional[str]]] = []
    for _ in range(len(rows)):
        grid.append([None] * max_cols)
    
    # Second pass: fill grid handling rowspan/colspan
    for row_idx, row in enumerate(rows):
        cells = row.find_all(["td", "th"])
        col_idx = 0
        
        for cell in cells:
            # Find next available column (skip cells filled by rowspan)
            while col_idx < max_cols and grid[row_idx][col_idx] is not None:
                col_idx += 1
            
            if col_idx >= max_cols:
                break
            
            # Get cell text
            text = _clean_cell(cell.get_text())
            
            # Get span attributes
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            
            # Fill grid cells covered by this cell
            for r in range(row_idx, min(row_idx + rowspan, len(rows))):
                for c in range(col_idx, min(col_idx + colspan, max_cols)):
                    if grid[r][c] is None:
                        # Only first cell gets the value; others get empty (for span indication)
                        if r == row_idx and c == col_idx:
                            grid[r][c] = text
                        else:
                            grid[r][c] = ""  # Placeholder for spanned cell
            
            col_idx += colspan
    
    # Convert None to empty string
    result: List[List[str]] = []
    for row in grid:
        result.append([cell if cell is not None else "" for cell in row])
    
    return result


def normalize_grid(grid: List[List[str]]) -> List[List[str]]:
    """
    Normalize grid for consistent processing:
    - Pad short rows to max width
    - Strip whitespace from all cells
    """
    if not grid:
        return []
    
    max_len = max(len(row) for row in grid)
    normalized = []
    
    for row in grid:
        # Extend row to max length
        padded = list(row) + [""] * (max_len - len(row))
        # Strip each cell
        normalized.append([_clean_cell(cell) for cell in padded])
    
    return normalized


def parse_grid_from_html(html_path: str, table_id: str) -> List[List[str]]:
    """
    Main entry point: parse HTML file and extract grid for specified table.
    
    Args:
        html_path: Path to HTML file
        table_id: Table ID in format "tXXXX" (e.g., "t0042")
    
    Returns:
        Normalized 2D grid of strings
    """
    path = Path(html_path)
    if not path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")
    
    html_content = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html_content, "html.parser")
    
    table = _extract_table_by_id(soup, table_id)
    if table is None:
        raise ValueError(f"Table {table_id} not found in {html_path}")
    
    grid = _parse_table_to_grid(table)
    return normalize_grid(grid)


def extract_table_metadata(html_path: str, table_id: str) -> dict:
    """
    Extract metadata about a table: caption, nearby headings, units hints.
    
    Useful for layout inference context.
    """
    path = Path(html_path)
    html_content = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html_content, "html.parser")
    
    table = _extract_table_by_id(soup, table_id)
    if table is None:
        return {"error": f"Table {table_id} not found"}
    
    # Look for caption
    caption = table.find("caption")
    caption_text = _clean_cell(caption.get_text()) if caption else ""
    
    # Look for preceding heading (h1-h4 or strong/b within 500 chars)
    heading_text = ""
    prev = table.find_previous(["h1", "h2", "h3", "h4"])
    if prev:
        heading_text = _clean_cell(prev.get_text())
    
    # Look for units hint in first few rows
    units_hint = ""
    units_patterns = [
        r"\(in\s+millions?\)",
        r"\(in\s+thousands?\)",
        r"\(in\s+billions?\)",
        r"dollars?\s+in\s+millions?",
        r"amounts?\s+in\s+millions?",
    ]
    
    grid = _parse_table_to_grid(table)
    first_rows_text = " ".join(" ".join(row) for row in grid[:5]).lower()
    
    for pattern in units_patterns:
        if re.search(pattern, first_rows_text, re.IGNORECASE):
            if "million" in first_rows_text:
                units_hint = "millions"
            elif "thousand" in first_rows_text:
                units_hint = "thousands"
            elif "billion" in first_rows_text:
                units_hint = "billions"
            break
    
    return {
        "table_id": table_id,
        "caption_text": caption_text,
        "heading_context": heading_text,
        "units_hint": units_hint,
        "row_count": len(grid),
        "col_count": max((len(r) for r in grid), default=0),
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python parse_grid.py <html_path> <table_id>", file=sys.stderr)
        print("       python parse_grid.py <html_path> <table_id> --metadata", file=sys.stderr)
        sys.exit(1)
    
    html_path = sys.argv[1]
    table_id = sys.argv[2]
    
    if len(sys.argv) > 3 and sys.argv[3] == "--metadata":
        metadata = extract_table_metadata(html_path, table_id)
        print(json.dumps(metadata, indent=2))
    else:
        try:
            grid = parse_grid_from_html(html_path, table_id)
            print(json.dumps(grid, indent=2))
        except Exception as e:
            print(json.dumps({"error": str(e)}), file=sys.stderr)
            sys.exit(1)
