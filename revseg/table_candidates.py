"""
Stage 3 (Steps 1–2): Extract candidate tables from a 10-K HTML filing and serialize to JSON.

This module:
1) Extracts all <table> elements from a filing HTML and builds "TableCandidate" objects
   containing a compact preview plus rich context and structural signals.
2) Writes the candidates to a JSON artifact for downstream LLM-based table selection.

Dependencies:
  pip install beautifulsoup4 lxml
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, Tag

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
WS_RE = re.compile(r"\s+")

# Numeric / money detection (preview-level signals)
MONEY_RE = re.compile(
    r"^\(?\$?\s*[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?$|^\(?\$?\s*[-+]?\d+(?:\.\d+)?\)?$"
)
PCT_RE = re.compile(r"^\(?\s*[-+]?\d+(?:\.\d+)?\s*%?\s*\)?$")
UNITS_RE = re.compile(r"\(\s*in\s+(millions|billions|thousands)\s*\)", re.IGNORECASE)
CURRENCY_SYMBOL_RE = re.compile(r"[$€£]")

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
    "in millions",
    "in billions",
    "in thousands",
]


@dataclass
class TableCandidate:
    table_id: str
    n_rows: int
    n_cols: int

    # Content previews
    preview: List[List[str]]                 # top-left preview_rows x preview_cols
    header_preview: List[List[str]]          # first 1-2 rows of preview
    row_label_preview: List[str]             # first-column labels for first N rows

    # Signals
    detected_years: List[int]
    keyword_hits: List[str]

    numeric_cell_ratio: float
    money_cell_ratio: float
    has_currency_symbol: bool

    has_units_marker: bool
    units_hint: str

    has_year_header: bool
    year_header_text: str

    # Context
    caption_text: str
    heading_context: str
    nearby_text_context: str

    # Provenance
    source_html_path: str


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
    tdir = base_dir / ticker
    if not tdir.exists():
        raise FileNotFoundError(f"No directory found for ticker {ticker}: {tdir}")

    subdirs = [p for p in tdir.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No filing subfolders found under: {tdir}")

    # filingDate prefix sorts lexicographically (YYYY-MM-DD)
    subdirs.sort(key=lambda p: p.name, reverse=True)
    return subdirs[0]


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

    htmls.sort(key=lambda p: p.stat().st_size, reverse=True)
    return htmls[0]


# ----------------------------
# HTML parsing + extraction
# ----------------------------

def _clean_text(s: str) -> str:
    return WS_RE.sub(" ", s or "").strip()


def _looks_numeric(cell: str) -> bool:
    if not cell:
        return False
    c = cell.strip()
    if c.lower() in {"—", "-", "–", "n/a", "na"}:
        return False

    # Percentages or plain numbers
    compact = c.replace(" ", "")
    if PCT_RE.match(compact):
        return any(ch.isdigit() for ch in c)

    # Simple numeric patterns (with or without commas / decimals / parens)
    if re.match(r"^\(?[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?$", c):
        return True
    if re.match(r"^\(?[-+]?\d+(?:\.\d+)?\)?$", c):
        return True
    return False


def _looks_money(cell: str) -> bool:
    if not cell:
        return False
    c = cell.strip()
    compact = c.replace(" ", "")
    if MONEY_RE.match(compact):
        return any(ch.isdigit() for ch in c)
    return False


def _extract_caption_text(table: Tag) -> str:
    cap = table.find("caption")
    if cap:
        return _clean_text(cap.get_text(" ", strip=True))
    return ""


def _detect_units_hint(text_blob: str) -> Tuple[bool, str]:
    if not text_blob:
        return False, ""
    m = UNITS_RE.search(text_blob)
    if m:
        return True, m.group(0)

    lower = text_blob.lower()
    for u in ["in millions", "in billions", "in thousands"]:
        if u in lower:
            return True, u
    return False, ""


def _detect_years(text_blob: str) -> List[int]:
    years = set()
    for m in YEAR_RE.finditer(text_blob or ""):
        y = int(m.group(0))
        if 1990 <= y <= 2100:
            years.add(y)
    return sorted(years)


def _keyword_hits(text_blob: str, keywords: List[str]) -> List[str]:
    t = (text_blob or "").lower()
    return [kw for kw in keywords if kw in t]


def _extract_table_grid(
    table: Tag, max_rows: int, max_cols: int
) -> Tuple[List[List[str]], int, int, Dict[str, Any]]:
    """
    Extract a preview grid AND compute simple structural stats over the preview.

    We intentionally avoid complex rowspan/colspan normalization at this stage.
    The preview is designed for downstream selection, not for numeric extraction.
    """
    rows = table.find_all("tr")
    grid: List[List[str]] = []
    max_col_count = 0

    preview_cell_count = 0
    numeric_count = 0
    money_count = 0
    currency_symbol_found = False

    for r in rows[:max_rows]:
        cells = r.find_all(["th", "td"])
        row_vals: List[str] = []
        for c in cells[:max_cols]:
            txt = _clean_text(c.get_text(" ", strip=True))
            row_vals.append(txt)

            preview_cell_count += 1
            if CURRENCY_SYMBOL_RE.search(txt or ""):
                currency_symbol_found = True
            if _looks_numeric(txt):
                numeric_count += 1
            if _looks_money(txt):
                money_count += 1

        max_col_count = max(max_col_count, len(cells))
        grid.append(row_vals)

    n_rows = len(rows)
    n_cols = max_col_count

    numeric_ratio = (numeric_count / preview_cell_count) if preview_cell_count else 0.0
    money_ratio = (money_count / preview_cell_count) if preview_cell_count else 0.0

    stats = {
        "numeric_cell_ratio": float(numeric_ratio),
        "money_cell_ratio": float(money_ratio),
        "has_currency_symbol": bool(currency_symbol_found),
    }
    return grid, n_rows, n_cols, stats


def _collect_nearby_text(table: Tag, max_chars: int = 800) -> str:
    """
    Collect nearby text by walking backwards from the table in document order.
    This provides context such as:
      - "The following table presents revenues by type..."
      - "Revenues" headings / narrative
    """
    parts: List[str] = []
    chars = 0

    node = table
    for _ in range(300):  # hard stop to avoid pathological docs
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

            # Stop if we hit a major section boundary and already have context
            if prev.name in {"h1", "h2", "h3"} and parts:
                parts.append(txt)
                break

            # Deduplicate exact repeats
            if txt not in parts:
                parts.append(txt)
                chars += len(txt) + 1
                if chars >= max_chars:
                    break

        node = prev

    parts = list(reversed(parts))
    context = _clean_text(" ".join(parts))
    if len(context) > max_chars:
        context = context[-max_chars:]
    return context


def _nearest_heading_text(table: Tag, max_chars: int = 250) -> str:
    """
    Find nearest preceding heading-like text. Many filings do not use h1-h6 consistently.
    We search in this order:
      1) table <caption>
      2) nearest previous h1-h6
      3) nearest previous element containing <b>/<strong> that looks like a title line
    """
    cap = _extract_caption_text(table)
    if cap:
        return cap[:max_chars]

    for prev in table.find_all_previous(["h1", "h2", "h3", "h4", "h5", "h6"]):
        txt = _clean_text(prev.get_text(" ", strip=True))
        if txt:
            return txt[:max_chars]

    for prev in table.find_all_previous(["p", "div", "span"]):
        if prev.find(["b", "strong"]):
            txt = _clean_text(prev.get_text(" ", strip=True))
            if 3 <= len(txt) <= 200:
                return txt[:max_chars]

    return ""


def extract_table_candidates_from_html(
    html_path: Path,
    *,
    preview_rows: int = 15,
    preview_cols: int = 8,
    max_tables: Optional[int] = None,
    keywords: Optional[List[str]] = None,
) -> List[TableCandidate]:
    """
    Step 1: extract all tables + context + structural signals from a 10-K HTML.

    Output is a list of TableCandidate objects (Step 2 will serialize to JSON).
    """
    html_path = html_path.expanduser().resolve()
    if not html_path.exists():
        raise FileNotFoundError(f"HTML path not found: {html_path}")

    kw = keywords or DEFAULT_KEYWORDS

    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")

    candidates: List[TableCandidate] = []

    for i, table in enumerate(tables):
        grid, n_rows, n_cols, stats = _extract_table_grid(table, preview_rows, preview_cols)

        caption = _extract_caption_text(table)
        heading = _nearest_heading_text(table)
        nearby = _collect_nearby_text(table)

        # Header preview: first 6 rows of preview grid (to catch year headers that may be in rows 3-5 due to spacer rows)
        header_preview = grid[:6]

        # Row label preview: first column from first 25 preview rows, keep up to 20 non-empty
        row_labels: List[str] = []
        for r in grid[: min(len(grid), 25)]:
            if not r:
                continue
            lab = (r[0] or "").strip()
            if lab:
                row_labels.append(lab)
        row_label_preview = row_labels[:20]

        # Year header detection: check year presence in the header preview
        year_header_text = " ".join([" ".join(r) for r in header_preview]).strip()
        header_years = _detect_years(year_header_text)
        has_year_header = len(set(header_years)) >= 2

        # Combine context for year/keyword/units detection
        combined = " ".join(
            [
                caption,
                heading,
                nearby,
                year_header_text,
                " ".join([" ".join(r) for r in grid[: min(5, len(grid))]]),
            ]
        )

        years = _detect_years(combined)
        hits = _keyword_hits(combined, kw)
        has_units_marker, units_hint = _detect_units_hint(combined)

        cand = TableCandidate(
            table_id=f"t{i:04d}",
            n_rows=n_rows,
            n_cols=n_cols,
            preview=grid,
            header_preview=header_preview,
            row_label_preview=row_label_preview,
            detected_years=years,
            keyword_hits=hits,
            numeric_cell_ratio=float(stats["numeric_cell_ratio"]),
            money_cell_ratio=float(stats["money_cell_ratio"]),
            has_currency_symbol=bool(stats["has_currency_symbol"]),
            has_units_marker=bool(has_units_marker),
            units_hint=units_hint,
            has_year_header=bool(has_year_header),
            year_header_text=year_header_text,
            caption_text=caption,
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
        "schema": "revseg.table_candidates.v2",
        "generated_utc": int(time.time()),
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
