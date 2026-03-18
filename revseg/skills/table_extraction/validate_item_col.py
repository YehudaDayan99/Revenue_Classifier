#!/usr/bin/env python3
"""
Validate LLM's item_col choice using deterministic heuristics.

The LLM infers which column contains row labels (item_col).
This script validates that choice and overrides if it's clearly wrong
(e.g., LLM picked a numeric column).

Usage:
    python validate_item_col.py <grid_json> <llm_item_col> [header_rows_json]
    
Output:
    JSON with validated_col, override_reason (or null if accepted)
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


# Pattern to detect currency/numeric cells
_CURRENCY_NUM_RE = re.compile(r"^[\s$€£¥(),.0-9\-—–]+$")

# Pattern to detect cells with meaningful text
_ALPHA_RE = re.compile(r"[a-zA-Z]")

# Minimum alpha characters for a "text" cell
_MIN_ALPHA_CHARS = 3


def _clean(s: str) -> str:
    """Normalize whitespace and strip."""
    return re.sub(r"\s+", " ", (s or "").strip())


def _is_numeric_cell(text: str) -> bool:
    """Check if cell appears to be numeric/currency."""
    text = _clean(text)
    if not text:
        return False
    return bool(_CURRENCY_NUM_RE.match(text))


def _has_alpha(text: str) -> bool:
    """Check if cell contains meaningful alphabetic text."""
    text = _clean(text)
    alpha_chars = sum(1 for c in text if c.isalpha())
    return alpha_chars >= _MIN_ALPHA_CHARS


def _count_alpha_chars(text: str) -> int:
    """Count alphabetic characters in text."""
    return sum(1 for c in text if c.isalpha())


def compute_column_scores(
    grid: List[List[str]],
    header_rows: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute scoring metrics for each column to identify the best label column.
    
    For each column, computes:
    - numeric_ratio: fraction of non-empty cells that are numeric/currency
    - alpha_ratio: fraction of non-empty cells that contain text
    - avg_alpha_chars: average alphabetic characters per cell
    - score: combined heuristic score (higher = more likely label column)
    """
    if not grid:
        return []
    
    header_set = set(header_rows or [])
    
    # Determine max columns
    max_cols = max((len(row) for row in grid), default=0)
    
    col_scores: List[Dict[str, Any]] = []
    
    for col_idx in range(max_cols):
        # Gather non-header cells from this column
        cells = []
        for row_idx, row in enumerate(grid):
            if row_idx in header_set:
                continue
            if col_idx < len(row):
                cell = _clean(row[col_idx])
                if cell:  # Only count non-empty cells
                    cells.append(cell)
        
        if not cells:
            col_scores.append({
                "col": col_idx,
                "numeric_ratio": 1.0,
                "alpha_ratio": 0.0,
                "avg_alpha_chars": 0,
                "score": 0,
            })
            continue
        
        # Compute ratios
        numeric_count = sum(1 for c in cells if _is_numeric_cell(c))
        alpha_count = sum(1 for c in cells if _has_alpha(c))
        total_alpha_chars = sum(_count_alpha_chars(c) for c in cells)
        
        numeric_ratio = numeric_count / len(cells)
        alpha_ratio = alpha_count / len(cells)
        avg_alpha_chars = total_alpha_chars / len(cells)
        
        # Score: prefer columns with high text content, low numeric content
        # Higher alpha_ratio and avg_alpha_chars = better
        # Lower numeric_ratio = better
        score = (alpha_ratio * 40) + (avg_alpha_chars * 0.5) - (numeric_ratio * 30)
        
        col_scores.append({
            "col": col_idx,
            "numeric_ratio": round(numeric_ratio, 3),
            "alpha_ratio": round(alpha_ratio, 3),
            "avg_alpha_chars": round(avg_alpha_chars, 1),
            "score": round(score, 2),
        })
    
    return col_scores


def choose_item_col(
    grid: List[List[str]],
    header_rows: Optional[List[int]] = None,
    llm_proposed_col: Optional[int] = None,
) -> Tuple[int, str]:
    """
    Deterministically select the best label/item column in a table grid.
    
    Validates LLM's proposed column and overrides if it fails heuristic checks.
    
    Args:
        grid: 2D list of cell strings
        header_rows: Row indices to exclude from analysis
        llm_proposed_col: Column index proposed by LLM layout inference
    
    Returns:
        (validated_col, reason_string)
        - If LLM choice accepted: (llm_proposed_col, "LLM choice validated (...)")
        - If LLM choice overridden: (heuristic_best, "LLM col X OVERRIDDEN (...)")
    """
    col_scores = compute_column_scores(grid, header_rows)
    
    if not col_scores:
        return (0, "no columns to analyze")
    
    # Sort by score descending
    col_scores.sort(key=lambda x: x["score"], reverse=True)
    heuristic_best = col_scores[0]
    
    # Validate LLM's proposed column
    if llm_proposed_col is not None and 0 <= llm_proposed_col < len(col_scores):
        llm_col_data = next((c for c in col_scores if c["col"] == llm_proposed_col), None)
        
        if llm_col_data:
            # Accept LLM choice if:
            # 1. numeric_ratio < 0.5 (not mostly numbers)
            # 2. alpha_ratio > 0.3 (has some text)
            if llm_col_data["numeric_ratio"] < 0.5 and llm_col_data["alpha_ratio"] > 0.3:
                return (
                    llm_proposed_col,
                    f"LLM choice validated (num={llm_col_data['numeric_ratio']:.2f}, alpha={llm_col_data['alpha_ratio']:.2f})"
                )
            else:
                # LLM choice failed validation, override with heuristic best
                return (
                    heuristic_best["col"],
                    f"LLM col {llm_proposed_col} OVERRIDDEN (num={llm_col_data['numeric_ratio']:.2f}, alpha={llm_col_data['alpha_ratio']:.2f}) → col {heuristic_best['col']} (score={heuristic_best['score']:.2f})"
                )
    
    # No LLM proposal or invalid index, use heuristic
    return (
        heuristic_best["col"],
        f"heuristic best (score={heuristic_best['score']:.2f})"
    )


def validate_extracted_labels(
    labels: List[str],
    threshold: float = 0.5,
) -> Tuple[bool, str]:
    """
    Validate that extracted revenue line labels are not mostly numeric/currency.
    
    If >50% of labels are numeric, the extraction likely used a value column
    as the label column.
    
    Args:
        labels: List of extracted revenue line labels
        threshold: Maximum allowed ratio of numeric labels (default 50%)
    
    Returns:
        (is_valid, reason)
    """
    if not labels:
        return (False, "no labels extracted")
    
    numeric_count = sum(1 for label in labels if _is_numeric_cell(label))
    numeric_ratio = numeric_count / len(labels)
    
    if numeric_ratio > threshold:
        return (
            False,
            f"FAIL: {numeric_ratio*100:.0f}% of labels are numeric/currency (threshold: {threshold*100:.0f}%)"
        )
    
    return (
        True,
        f"OK: {numeric_ratio*100:.0f}% numeric labels"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python validate_item_col.py <grid_json> <llm_item_col> [header_rows_json]", file=sys.stderr)
        print("       python validate_item_col.py --validate-labels <labels_json>", file=sys.stderr)
        sys.exit(1)
    
    # Handle label validation mode
    if sys.argv[1] == "--validate-labels":
        if len(sys.argv) < 3:
            print("Usage: python validate_item_col.py --validate-labels <labels_json>", file=sys.stderr)
            sys.exit(1)
        
        with open(sys.argv[2]) as f:
            labels = json.load(f)
        
        is_valid, reason = validate_extracted_labels(labels)
        print(json.dumps({
            "valid": is_valid,
            "reason": reason,
        }, indent=2))
        sys.exit(0 if is_valid else 1)
    
    # Normal column validation mode
    grid_path = sys.argv[1]
    llm_item_col = int(sys.argv[2])
    header_rows = None
    
    if len(sys.argv) > 3:
        with open(sys.argv[3]) as f:
            header_rows = json.load(f)
    
    try:
        # Load grid
        if grid_path == "-":
            grid = json.load(sys.stdin)
        else:
            with open(grid_path) as f:
                grid = json.load(f)
        
        # Validate
        validated_col, reason = choose_item_col(grid, header_rows, llm_item_col)
        
        # Compute all column scores for debugging
        all_scores = compute_column_scores(grid, header_rows)
        
        result = {
            "llm_proposed_col": llm_item_col,
            "validated_col": validated_col,
            "was_overridden": validated_col != llm_item_col,
            "reason": reason,
            "column_scores": all_scores,
        }
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
