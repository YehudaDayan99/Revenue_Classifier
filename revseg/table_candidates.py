from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, Tag


YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
WS_RE = re.compile(r"\s+")

DEFAULT_KEYWORDS = [
    "revenue",
    "revenues",
    "net sales",
    "net sale",
    "sales",
    "segment",
    "segments",
    "disaggregated",
    "by product",
    "by type",
    "by geography",
    "cloud",
    "advertising",
    "subscription",
    "subscriptions",
    "services",
    "total",
    "year ended",
]


@dataclass
class TableCandidate:
    table_id: str
    n_rows: int
    n_cols: int
    preview: List[List[str]]           # top-left preview_rows x preview_cols
    detected_years: List[int]
    keyword_hits: List[str]
    heading_context: str               # nearest preceding heading-ish text
    nearby_text_context: str           # nearby prose (limited chars)
    source_html_path: str              # where the HTML came from


# ----------------------------
# Utilities: locating filings
# ----------------------------

def find_latest_downloaded_filing_dir(base_dir: Path, ticker: str) -> Path:
    """
    Given base_dir like data/10k and a ticker, return the most recent downloaded filing folder.

    Expected structure:
      base_dir/<TICKER>/<filingDate>_<accessionNoDashes>/
    """
    ticker = ticker.upper().strip()
    tdir = (base_dir / ticker)
    if not tdir.exists():
        raise FileNotFoundError(f"No directory found for ticker {ticker}: {tdir}")

    # Sort by folder name; filingDate prefix sorts lexicographically (YYYY-MM-DD).
    candidates = [p for p in tdir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No filing subfolders found under: {tdir}")

    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def find_primary_document_html(filing_dir: Path) -> Path:
    """
    Prefer the standardized name created in stage 1: primary_document.html
    Fallback: find the largest .htm/.html in the folder.
    """
    preferred = filing_dir / "primary_document.html"
    if preferred.exists():
        return preferred

    htmls = list(filing_dir.glob("*.htm")) + list(filing_dir.glob("*.html"))
    if not htmls:
        raise FileNotFoundError(f"No HTML document found in: {filing_dir}")

    # Choose the largest as a heuristic.
    htmls.sort(key=lambda p: p.stat().st_size, reverse=True)
    return htmls[0]


# ----------------------------
# HTML parsing + extraction
# ----------------------------

def _clean_text(s: str) -> str:
    s = WS_RE.sub(" ", s or "").strip()
    return s


def _extract_table_grid(table: Tag, max_rows: int, max_cols: int) -> Tuple[List[List[str]], int, int]:
    """
    Extract table text content (simple grid). This is intentionally conservative:
    - ignores complex rowspan/colspan normalization at this stage
    - produces a preview matrix for LLM selection / heuristics
    """
    rows = table.find_all("tr")
    grid: List[List[str]] = []
    max_col_count = 0

    for r in rows[:max_rows]:
        cells = r.find_all(["th", "td"])
        row_vals: List[str] = []
        for c in cells[:max_cols]:
            txt = _clean_text(c.get_text(" ", strip=True))
            row_vals.append(txt)
        max_col_count = max(max_col_count, len(cells))
        grid.append(row_vals)

    n_rows = len(rows)
    n_cols = max_col_count
    return grid, n_rows, n_cols


def _collect_nearby_text(table: Tag, max_chars: int = 800) -> str:
    """
    Collect nearby text by walking backwards from the table in document order.
    This provides context such as:
      - "The following table presents revenues by type..."
      - "Revenues" headings
    """
    parts: List[str] = []
    chars = 0

    # Walk previous elements in the DOM
    node = table
    for _ in range(250):  # hard stop to avoid pathological docs
        prev = node.find_previous()
        if prev is None:
            break
        if isinstance(prev, Tag):
            if prev.name in {"script", "style", "table"}:
                node = prev
                continue

            txt = _clean_text(prev.get_text(" ", strip=True))
            if not txt:
                node = prev
                continue

            # stop if we hit an earlier major section boundary and already have some context
            if prev.name in {"h1", "h2", "h3"} and parts:
                parts.append(txt)
                break

            # accumulate
            if txt not in parts:  # de-dup
                parts.append(txt)
                chars += len(txt) + 1
                if chars >= max_chars:
                    break

        node = prev

    # Reverse to get natural reading order; keep only the tail end (closest context first)
    parts = list(reversed(parts))
    context = _clean_text(" ".join(parts))
    if len(context) > max_chars:
        context = context[-max_chars:]
    return context


def _nearest_heading_text(table: Tag, max_chars: int = 250) -> str:
    """
    Find nearest preceding heading-ish tag (h1-h6) and return its text.
    """
    for prev in table.find_all_previous(["h1", "h2", "h3", "h4", "h5", "h6"]):
        txt = _clean_text(prev.get_text(" ", strip=True))
        if txt:
            return txt[:max_chars]
    return ""


def _detect_years(text_blob: str) -> List[int]:
    years = set()
    for m in YEAR_RE.finditer(text_blob or ""):
        y = int(m.group(0))
        # Practical filter
        if 1990 <= y <= 2100:
            years.add(y)
    return sorted(years)


def _keyword_hits(text_blob: str, keywords: List[str]) -> List[str]:
    t = (text_blob or "").lower()
    hits = [kw for kw in keywords if kw in t]
    return hits


def extract_table_candidates_from_html(
    html_path: Path,
    *,
    preview_rows: int = 15,
    preview_cols: int = 8,
    max_tables: Optional[int] = None,
    keywords: Optional[List[str]] = None,
) -> List[TableCandidate]:
    """
    Step 1: extract all tables + context from a 10-K HTML.

    Output is a list of TableCandidate objects (Step 2 will serialize to JSON).
    """
    html_path = html_path.expanduser().resolve()
    if not html_path.exists():
        raise FileNotFoundError(f"HTML path not found: {html_path}")

    kw = keywords or DEFAULT_KEYWORDS

    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    tables = soup.find_all("table")

    candidates: List[TableCandidate] = []
    for i, table in enumerate(tables):
        grid, n_rows, n_cols = _extract_table_grid(table, preview_rows, preview_cols)
        heading = _nearest_heading_text(table)
        nearby = _collect_nearby_text(table)

        # Use combined context for year/keyword detection
        combined = " ".join(
            [
                heading,
                nearby,
                " ".join([" ".join(r) for r in grid[: min(5, len(grid))]]),
            ]
        )
        years = _detect_years(combined)
        hits = _keyword_hits(combined, kw)

        cand = TableCandidate(
            table_id=f"t{i:04d}",
            n_rows=n_rows,
            n_cols=n_cols,
            preview=grid,
            detected_years=years,
            keyword_hits=hits,
            heading_context=heading,
            nearby_text_context=nearby,
            source_html_path=str(html_path),
        )
        candidates.append(cand)

        if max_tables is not None and len(candidates) >= max_tables:
            break

    return candidates


# ----------------------------
# Step 2: JSON serialization
# ----------------------------

def candidates_to_json_dict(candidates: List[TableCandidate]) -> Dict[str, Any]:
    return {
        "schema": "revseg.table_candidates.v1",
        "n_candidates": len(candidates),
        "candidates": [asdict(c) for c in candidates],
    }


def write_candidates_json(candidates: List[TableCandidate], out_path: Path) -> Path:
    """
    Step 2: serialize the candidate tables into a compact JSON artifact
    intended for downstream LLM selection.
    """
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = candidates_to_json_dict(candidates)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
