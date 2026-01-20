from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup

from revseg.llm_client import OpenAIChatClient
from revseg.table_candidates import TableCandidate, extract_table_grid_normalized


_WS_RE = re.compile(r"\s+")
_MONEY_CLEAN_RE = re.compile(r"[^0-9.\-]")
_ITEM8_RE = re.compile(
    r"\bitem\s*8\b|\bfinancial statements\b|\bnotes to (?:the )?financial statements\b",
    re.IGNORECASE,
)
_ITEM7_RE = re.compile(r"\bitem\s*7\b|\bmanagement['’]s discussion\b|\bmd&a\b", re.IGNORECASE)
_SEGMENT_NOTE_RE = re.compile(r"\bsegment(s)?\b|\breportable segment(s)?\b", re.IGNORECASE)


def _clean(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip())


def _parse_number(s: str) -> Optional[float]:
    if s is None:
        return None
    t = _clean(s)
    if t in {"", "-", "—", "–"}:
        return None
    # Handle parentheses negatives
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg = True
        t = t[1:-1]
    t = t.replace("$", "").replace(",", "").strip()
    try:
        v = float(t)
        return -v if neg else v
    except Exception:
        return None


def _parse_money_to_int(s: str) -> Optional[int]:
    v = _parse_number(s)
    if v is None:
        return None
    return int(round(v))


def rank_candidates_for_financial_tables(candidates: List[TableCandidate]) -> List[TableCandidate]:
    return sorted(
        candidates,
        key=lambda c: (
            float(guess_item8_score(c)),
            bool(getattr(c, "has_year_header", False)),
            bool(getattr(c, "has_units_marker", False)),
            float(getattr(c, "money_cell_ratio", 0.0)),
            float(getattr(c, "numeric_cell_ratio", 0.0)),
            len(getattr(c, "keyword_hits", []) or []),
            int(getattr(c, "n_rows", 0)) * int(getattr(c, "n_cols", 0)),
        ),
        reverse=True,
    )


def guess_item8_score(c: TableCandidate) -> float:
    """Soft signal: does the local context look like Item 8 / Notes / Segment Note?"""
    blob = " ".join(
        [
            str(getattr(c, "heading_context", "") or ""),
            str(getattr(c, "caption_text", "") or ""),
            str(getattr(c, "nearby_text_context", "") or ""),
        ]
    )
    blob = _clean(blob)
    score = 0.0
    if _ITEM8_RE.search(blob):
        score += 3.0
    if _SEGMENT_NOTE_RE.search(blob):
        score += 1.0
    # If it looks like Item 7/MD&A, slightly downweight (soft preference, not exclusion)
    if _ITEM7_RE.search(blob):
        score -= 1.0
    return score


def extract_keyword_windows(
    html_path: Path,
    *,
    keywords: List[str],
    window_chars: int = 2500,
    max_windows: int = 12,
) -> List[str]:
    """Deterministically extract short text windows around keywords for LLM context."""
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    text = _clean(text)
    low = text.lower()

    windows: List[str] = []
    for kw in keywords:
        k = kw.lower()
        start = 0
        while True:
            i = low.find(k, start)
            if i == -1:
                break
            a = max(0, i - window_chars // 3)
            b = min(len(text), i + window_chars)
            snippet = _clean(text[a:b])
            if snippet and snippet not in windows:
                windows.append(snippet)
            start = i + max(1, len(k))
            if len(windows) >= max_windows:
                return windows
    return windows


def document_scout(html_path: Path, *, max_headings: int = 80) -> Dict[str, Any]:
    """Lightweight scan of headings to help the LLM orient itself."""
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    headings: List[str] = []
    for tag in soup.find_all(["h1", "h2", "h3", "b", "strong"]):
        txt = _clean(tag.get_text(" ", strip=True))
        if 5 <= len(txt) <= 180 and txt not in headings:
            headings.append(txt)
        if len(headings) >= max_headings:
            break
    return {"headings": headings}


def _candidate_summary(c: TableCandidate) -> Dict[str, Any]:
    return {
        "table_id": c.table_id,
        "n_rows": c.n_rows,
        "n_cols": c.n_cols,
        "detected_years": c.detected_years,
        "keyword_hits": c.keyword_hits,
        "item8_score": guess_item8_score(c),
        "has_year_header": getattr(c, "has_year_header", False),
        "has_units_marker": getattr(c, "has_units_marker", False),
        "units_hint": getattr(c, "units_hint", ""),
        "money_cell_ratio": getattr(c, "money_cell_ratio", 0.0),
        "numeric_cell_ratio": getattr(c, "numeric_cell_ratio", 0.0),
        "row_label_preview": getattr(c, "row_label_preview", [])[:12],
        "caption_text": getattr(c, "caption_text", "")[:200],
        "heading_context": getattr(c, "heading_context", "")[:200],
        "nearby_text_context": getattr(c, "nearby_text_context", "")[:280],
    }


def select_segment_revenue_table(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    candidates: List[TableCandidate],
    scout: Dict[str, Any],
    snippets: List[str],
    max_candidates: int = 80,
) -> Dict[str, Any]:
    ranked = rank_candidates_for_financial_tables(candidates)[:max_candidates]
    payload = [_candidate_summary(c) for c in ranked]

    system = (
        "You are a financial filings analyst. You select the single best HTML table candidate "
        "that represents REVENUE BY REPORTABLE SEGMENT (or equivalent business segments) for the latest fiscal year. "
        "Prefer tables from Item 8 / Notes to Financial Statements when possible, but you may select other sections if they clearly match and are consistent. "
        "Output must be STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "objective": "Find the reportable segment revenue table (e.g., segments with revenue totals).",
            "headings": scout.get("headings", [])[:40],
            "retrieved_snippets": snippets[:10],
            "table_candidates": payload,
            "output_schema": {
                "table_id": "string like t0071",
                "confidence": "number 0..1",
                "kind": "string, use 'segment_revenue' or 'not_found'",
                "rationale": "short string",
            },
        },
        ensure_ascii=False,
    )
    out = llm.json_call(system=system, user=user, max_output_tokens=700)
    return out


TABLE_KINDS = [
    "segment_revenue",
    "product_service_revenue",
    "segment_results_of_operations",
    "other",
]


def discover_primary_business_lines(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    snippets: List[str],
) -> Dict[str, Any]:
    """Text-first agent: infer primary business lines for CSV1 (Option 1).

    Output contracts:
      - dimension: product_category | reportable_segments
      - segments: list[str] (primary business lines)
      - include_segments_optional: list[str] (e.g., Corporate adjustments) if needed for reconciliation
    """
    system = (
        "You are a financial filings analyst. Determine the primary business-line dimension for CSV1.\n"
        "Rules:\n"
        "- For AAPL, treat business lines as product categories (iPhone, Mac, iPad, Wearables/Home/Accessories, Services).\n"
        "- For MSFT and GOOGL, treat business lines as reportable segments (e.g., Intelligent Cloud).\n"
        "- If the filing includes corporate adjustments (e.g., hedging gains/losses) that are included in Total Revenues, "
        "put that under include_segments_optional=['Corporate'].\n"
        "Output STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "snippets": snippets[:10],
            "few_shot_examples": [
                {
                    "ticker": "AAPL",
                    "dimension": "product_category",
                    "segments": ["iPhone", "Mac", "iPad", "Wearables, Home and Accessories", "Services"],
                    "include_segments_optional": [],
                },
                {
                    "ticker": "MSFT",
                    "dimension": "reportable_segments",
                    "segments": [
                        "Productivity and Business Processes",
                        "Intelligent Cloud",
                        "More Personal Computing",
                    ],
                    "include_segments_optional": [],
                },
                {
                    "ticker": "GOOGL",
                    "dimension": "reportable_segments",
                    "segments": ["Google Services", "Google Cloud", "Other Bets"],
                    "include_segments_optional": ["Corporate"],
                },
            ],
            "output_schema": {
                "dimension": "product_category | reportable_segments",
                "segments": "list[string]",
                "include_segments_optional": "list[string]",
                "notes": "short string",
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=700)


def select_revenue_disaggregation_table(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    candidates: List[TableCandidate],
    scout: Dict[str, Any],
    snippets: List[str],
    segments: List[str],
    keyword_hints: Optional[List[str]] = None,
    max_candidates: int = 80,
    prefer_granular: bool = True,
) -> Dict[str, Any]:
    """Select the most granular revenue disaggregation table that includes a Total row.
    
    When prefer_granular=True, prioritize tables with product/service line items
    (e.g., 'Revenue by Products and Services') over segment-level totals.
    """
    ranked = rank_candidates_for_financial_tables(candidates)[:max_candidates]
    payload = [_candidate_summary(c) for c in ranked]
    
    granular_guidance = ""
    if prefer_granular:
        granular_guidance = (
            "- STRONGLY PREFER tables titled 'Revenue from External Customers by Products and Services' "
            "or 'Disaggregation of Revenue' that show individual product/service line items "
            "(e.g., 'Server products and cloud services', 'LinkedIn', 'Gaming', 'YouTube ads').\n"
            "- These granular tables are better than segment-level totals.\n"
        )
    
    system = (
        "You are a financial filings analyst. Select the single best table that DISAGGREGATES revenue "
        "by business lines (segments or product categories) and includes a Total Revenue/Net Sales row.\n"
        "Constraints:\n"
        f"{granular_guidance}"
        "- Ignore geography-only tables.\n"
        "- Prefer Item 8 / Notes (Note 17 or Note 18 often has the most granular breakdown).\n"
        "- Prefer tables whose year columns are recent fiscal years (>= 2018).\n"
        "- Prefer tables where row/column labels overlap the provided business lines or known products.\n"
        "Output STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "business_lines": segments,
            "keyword_hints": keyword_hints or [],
            "headings": scout.get("headings", [])[:40],
            "retrieved_snippets": snippets[:10],
            "table_candidates": payload,
            "output_schema": {
                "table_id": "tXXXX",
                "confidence": "0..1",
                "rationale": "short string",
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=700)


def infer_disaggregation_layout(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    table_id: str,
    candidate: TableCandidate,
    grid: List[List[str]],
    business_lines: List[str],
    max_rows_for_llm: int = 40,
) -> Dict[str, Any]:
    """Infer layout for tables like:
    - AAPL: Category | Product/Service | FY2025 | FY2024 | ...
    - MSFT/GOOGL: Segment | Product/Service | FY... | ...
    """
    preview = grid[:max_rows_for_llm]
    system = (
        "You analyze a revenue disaggregation table from a 10-K. "
        "Identify which columns correspond to Segment (optional), Item/Product (required), and years, "
        "and how to identify the Total row.\n"
        "Important: year columns should be recent fiscal years (>= 2018) and usually appear as FY2025/FY2024 or 2025/2024.\n"
        "Output STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "table_id": table_id,
            "business_lines": business_lines,
            "candidate_summary": _candidate_summary(candidate),
            "table_grid_preview": preview,
            "output_schema": {
                "segment_col": "int|null (e.g., 0 for Segment; null if no segment column)",
                "item_col": "int (e.g., Product / Service column)",
                "year_cols": {"YYYY": "int column index"},
                "header_rows": "list[int]",
                "total_row_regex": "string regex matching the Total row label (e.g., Total Revenues|Total Net Sales)",
                "exclude_row_regex": "string regex for rows to exclude (e.g., Hedging gains)",
                "units_multiplier": "int (1, 1000, 1000000, 1000000000)",
                "notes": "short string",
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=900)


def extract_disaggregation_rows_from_grid(
    grid: List[List[str]],
    *,
    layout: Dict[str, Any],
    target_year: Optional[int] = None,
    business_lines: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Deterministically extract (segment,item,value) rows and a table total."""
    # Pad rows so column indices inferred from a wide header row work across short rows.
    max_len = max((len(r) for r in grid), default=0)
    if max_len > 0:
        grid = [list(r) + [""] * (max_len - len(r)) for r in grid]

    seg_col = layout.get("segment_col")
    seg_col = int(seg_col) if seg_col is not None else None
    item_col = int(layout["item_col"])
    year_cols_raw = layout.get("year_cols") or {}
    year_cols: Dict[int, int] = {int(y): int(ci) for y, ci in year_cols_raw.items()}
    if not year_cols:
        raise ValueError("No year_cols detected")
    year = target_year or max(year_cols.keys())
    if year not in year_cols:
        year = max(year_cols.keys())
    val_col = year_cols[year]

    header_rows = set(int(i) for i in (layout.get("header_rows") or []))
    total_re = re.compile(layout.get("total_row_regex") or r"total", re.IGNORECASE)
    exclude_re = re.compile(layout.get("exclude_row_regex") or r"$^", re.IGNORECASE)
    mult = int(layout.get("units_multiplier") or 1)
    if mult <= 0:
        mult = 1

    bl_norm = {b.lower(): b for b in (business_lines or [])}
    def _is_business_line(s: str) -> bool:
        if not bl_norm:
            return True
        return s.lower() in bl_norm

    rows: List[Dict[str, Any]] = []
    total_val: Optional[int] = None
    last_seg: str = ""

    for r_i, row in enumerate(grid):
        if r_i in header_rows:
            continue
        if item_col >= len(row) or val_col >= len(row):
            continue
        seg = _clean(row[seg_col]) if seg_col is not None and seg_col < len(row) else ""
        if seg_col is not None:
            if seg:
                last_seg = seg
            else:
                # iXBRL often blanks repeated segment labels; fill down.
                seg = last_seg
        item = _clean(row[item_col])
        if not item:
            continue
        if exclude_re.search(item) or exclude_re.search(seg):
            continue

        # Some tables put a currency symbol column before the number (e.g., '$', '209,586').
        raw_val = _parse_money_to_int(row[val_col])
        if raw_val is None and (val_col + 1) < len(row):
            raw_val = _parse_money_to_int(row[val_col + 1])
        if raw_val is None and (val_col + 2) < len(row):
            raw_val = _parse_money_to_int(row[val_col + 2])
        if raw_val is None:
            continue
        val = int(raw_val) * mult

        # Total row detection: match across the row, not just item/segment cell.
        if total_re.search(item) or total_re.search(seg) or any(total_re.search(_clean(c)) for c in row if c):
            total_val = val
            continue

        if seg and not _is_business_line(seg) and seg.lower() != "corporate":
            # keep corporate as optional; otherwise require match if business lines provided
            continue

        rows.append({"segment": seg, "item": item, "value": val, "year": year})

    return {"year": year, "rows": rows, "total_value": total_val}


def extract_segment_revenue_from_segment_results_grid(
    grid: List[List[str]],
    *,
    segments: List[str],
    target_year: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract segment revenues from a 'segment results of operations' style table.

    Shape example (MSFT t0071):
      - segment header rows: 'Productivity and Business Processes'
      - metric rows under each segment: 'Revenue', 'Cost of revenue', ...
      - final 'Total' section with 'Revenue'
    """
    import re

    # Pad rows to a common width
    max_len = max((len(r) for r in grid), default=0)
    if max_len > 0:
        grid = [list(r) + [""] * (max_len - len(r)) for r in grid]

    year_re = re.compile(r"\b(20\d{2})\b")
    year_cols: dict[int, int] = {}
    for r in grid[:15]:
        for ci, cell in enumerate(r):
            m = year_re.search(str(cell or ""))
            if not m:
                continue
            y = int(m.group(1))
            if 2015 <= y <= 2100:
                year_cols.setdefault(y, ci)
    if not year_cols:
        raise ValueError("No year columns detected in segment results grid")

    year = target_year or max(year_cols.keys())
    if year not in year_cols:
        year = max(year_cols.keys())
    val_col = year_cols[year]

    seg_norm = {s.lower(): s for s in segments}
    current_seg = ""
    out: dict[str, int] = {}
    total_value: Optional[int] = None

    for row in grid:
        if not row:
            continue
        first = _clean(row[0] or "")
        if not first:
            continue

        # Segment header row
        if first.lower() in seg_norm or first.lower() == "total":
            current_seg = seg_norm.get(first.lower(), "Total")
            continue

        # Metric row under current segment
        if first.lower() == "revenue" and current_seg:
            raw = _parse_money_to_int(row[val_col])
            if raw is None and (val_col + 1) < len(row):
                raw = _parse_money_to_int(row[val_col + 1])
            if raw is None and (val_col + 2) < len(row):
                raw = _parse_money_to_int(row[val_col + 2])
            if raw is None:
                continue
            if current_seg == "Total":
                total_value = int(raw)
            else:
                out[current_seg] = int(raw)

    if not out:
        raise ValueError("No segment revenues extracted from segment results grid")
    return {"year": year, "segment_totals": out, "total_value": total_value}


def classify_table_candidates(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    candidates: List[TableCandidate],
    scout: Dict[str, Any],
    snippets: List[str],
    max_candidates: int = 60,
) -> Dict[str, Any]:
    """Classify top candidates into a strict table_kind enum for routing."""
    ranked = rank_candidates_for_financial_tables(candidates)[:max_candidates]
    payload = [_candidate_summary(c) for c in ranked]

    system = (
        "You are a financial filings analyst. Classify each table candidate into a strict table_kind enum.\n"
        "Definitions:\n"
        "- segment_revenue: revenue by reportable segment/business segment\n"
        "- product_service_revenue: revenue by product/service offerings or disaggregation\n"
        "- segment_results_of_operations: segment operating income/costs/expenses (NOT revenue)\n"
        "- other: anything else\n"
        "Output STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "retrieved_snippets": snippets[:8],
            "headings": scout.get("headings", [])[:30],
            "table_candidates": payload,
            "table_kind_enum": TABLE_KINDS,
            "output_schema": {
                "tables": [
                    {
                        "table_id": "tXXXX",
                        "table_kind": "one of table_kind_enum",
                        "confidence": "0..1",
                        "rationale": "short string",
                    }
                ]
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=1200)


def select_other_revenue_tables(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    candidates: List[TableCandidate],
    scout: Dict[str, Any],
    snippets: List[str],
    exclude_table_ids: Iterable[str],
    max_tables: int = 3,
    max_candidates: int = 120,
) -> Dict[str, Any]:
    ranked = rank_candidates_for_financial_tables(candidates)[:max_candidates]
    payload = [_candidate_summary(c) for c in ranked if c.table_id not in set(exclude_table_ids)]

    system = (
        "You are a financial filings analyst. Identify up to N additional REVENUE tables (not the main segments table), "
        "such as revenue by product/service offering, geography, customer type, or disaggregation. "
        "Prefer Item 8 / Notes sources when available; otherwise select the best matching revenue disclosures. "
        "Output must be STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "objective": "Find other revenue tables (product/service offerings etc.)",
            "N": max_tables,
            "headings": scout.get("headings", [])[:40],
            "retrieved_snippets": snippets[:10],
            "table_candidates": payload,
            "output_schema": {
                "tables": [
                    {
                        "table_id": "tXXXX",
                        "kind": "revenue_by_product_service | revenue_by_geography | other_revenue",
                        "confidence": "0..1",
                        "rationale": "short string",
                    }
                ]
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=900)


def extract_table_grid_normalized_with_fallback(
    html_path: Path, table_id: str, *, max_rows: int = 250
) -> List[List[str]]:
    # Wrapper in case we want to add fallbacks later (e.g., pandas.read_html)
    return extract_table_grid_normalized(html_path, table_id, max_rows=max_rows)


def infer_table_layout(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    table_id: str,
    candidate: TableCandidate,
    grid: List[List[str]],
    max_rows_for_llm: int = 30,
) -> Dict[str, Any]:
    """Ask the LLM to identify label/year columns and which rows are data."""
    preview = grid[:max_rows_for_llm]
    system = (
        "You analyze HTML tables from SEC 10-K filings. "
        "Your job: identify which column contains row labels and which columns correspond to fiscal years. "
        "Output STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "table_id": table_id,
            "candidate_summary": _candidate_summary(candidate),
            "table_grid_preview": preview,
            "output_schema": {
                "label_col": "int",
                "year_cols": {"YYYY": "int column index"},
                "header_rows": "list[int] (rows to ignore as header, from the preview)",
                "skip_row_regex": "string regex for rows to skip (e.g., totals, separators) or empty",
                "units_multiplier": "int (1, 1000, 1000000, 1000000000) inferred from units_hint if possible",
                "notes": "short string",
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=900)


def extract_revenue_rows_from_grid(
    grid: List[List[str]],
    *,
    layout: Dict[str, Any],
    target_year: Optional[int] = None,
) -> Tuple[int, Dict[str, int]]:
    """Return (year, {label -> revenue_usd_scaled}). Values are scaled by units_multiplier."""
    label_col = int(layout["label_col"])
    year_cols_raw = layout.get("year_cols") or {}
    year_cols: Dict[int, int] = {int(y): int(ci) for y, ci in year_cols_raw.items()}
    if not year_cols:
        raise ValueError("No year_cols detected")

    year = target_year or max(year_cols.keys())
    if year not in year_cols:
        year = max(year_cols.keys())
    value_col = year_cols[year]

    header_rows = set(int(i) for i in (layout.get("header_rows") or []))
    skip_row_re = layout.get("skip_row_regex") or ""
    skip_pat = re.compile(skip_row_re, re.IGNORECASE) if skip_row_re else None
    mult = int(layout.get("units_multiplier") or 1)
    if mult <= 0:
        mult = 1

    out: Dict[str, int] = {}
    for r_i, row in enumerate(grid):
        if r_i in header_rows:
            continue
        if label_col >= len(row) or value_col >= len(row):
            continue
        label = _clean(row[label_col])
        if not label:
            continue
        if skip_pat and skip_pat.search(label):
            continue
        if label.lower() in {"total", "total revenue", "revenues", "net sales"}:
            continue

        val = _parse_money_to_int(row[value_col])
        if val is None:
            continue
        out[label] = int(val) * mult

    return year, out


def summarize_segment_descriptions(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    sec_doc_url: str,
    html_text: str,
    segment_names: List[str],
    max_chars_per_segment: int = 6000,
) -> Dict[str, Any]:
    """Produce CSV2-style rows via LLM from extracted filing text snippets.
    
    Enhanced to find segment descriptions in Notes sections (Note 18 - Segment Information)
    which contain detailed product listings, falling back to Item 1 if needed.
    """
    snippets: Dict[str, str] = {}
    t = html_text
    low = t.lower()
    
    # Find key sections in the document
    # Priority 1: Note 18 / Segment Information section (has detailed product lists)
    segment_info_idx = low.find("segment information")
    if segment_info_idx == -1:
        segment_info_idx = low.find("note 18")
    
    # Also look for "segment primarily comprises" pattern (where bullet lists usually are)
    segment_comprises_idx = low.find("segment primarily comprises")
    
    # Notes section as fallback
    notes_idx = low.find("notes to consolidated financial statements")
    if notes_idx == -1:
        notes_idx = low.find("notes to financial statements")
    
    # Item 1 Business section as final fallback
    item1_idx = low.find("item 1")
    item1_business_idx = low.find("item 1.", item1_idx) if item1_idx >= 0 else -1
    
    for seg in segment_names:
        key = seg
        seg_low = seg.lower()
        
        # Strategy: Find the best occurrence of segment name with detailed product list
        # Priority 1: In Segment Information section (Note 18, has detailed products like Azure)
        # Priority 2: Near "segment primarily comprises" pattern (has bullet lists)
        # Priority 3: In Notes section (Item 8)
        # Priority 4: In Item 1 Business section
        # Priority 5: First occurrence anywhere
        
        best_idx = -1
        
        # Priority 1: Segment Information section
        if segment_info_idx >= 0:
            seg_in_segment_info = low.find(seg_low, segment_info_idx)
            if seg_in_segment_info >= 0:
                best_idx = seg_in_segment_info
        
        # Priority 2: Near "segment primarily comprises" (detailed product bullets)
        if best_idx == -1 and segment_comprises_idx >= 0:
            # Search backwards to find segment name before "primarily comprises"
            search_start = max(0, segment_comprises_idx - 500)
            seg_near_comprises = low.find(seg_low, search_start)
            if seg_near_comprises >= 0 and seg_near_comprises < segment_comprises_idx + 5000:
                best_idx = seg_near_comprises
        
        # Priority 3: Notes section (Item 8)
        if best_idx == -1 and notes_idx >= 0:
            seg_in_notes = low.find(seg_low, notes_idx)
            if seg_in_notes >= 0:
                best_idx = seg_in_notes
        
        # Priority 4: Item 1 Business section
        if best_idx == -1 and item1_business_idx >= 0:
            seg_in_item1 = low.find(seg_low, item1_business_idx)
            if seg_in_item1 >= 0:
                best_idx = seg_in_item1
        
        # Priority 5: Fallback to first occurrence
        if best_idx == -1:
            best_idx = low.find(seg_low)
        
        if best_idx == -1:
            snippets[key] = ""
            continue
        
        # Extract more context after the segment name (where product details usually are)
        start = max(0, best_idx - 300)
        end = min(len(t), best_idx + max_chars_per_segment)
        snippets[key] = _clean(t[start:end])

    system = (
        "You summarize company business segments from SEC 10-K text. "
        "For each segment, write a comprehensive description and list SPECIFIC product/brand names "
        "(e.g., 'Azure', 'Office 365', 'Microsoft 365 Commercial', 'Dynamics 365', 'LinkedIn', 'GitHub', "
        "'iPhone', 'iPad', 'Google Cloud Platform', 'YouTube'). "
        "Do NOT use generic terms when specific product names are mentioned in the text. "
        "Output STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "sec_doc_url": sec_doc_url,
            "segments": [{"segment": s, "text_snippet": snippets.get(s, "")} for s in segment_names],
            "output_schema": {
                "rows": [
                    {
                        "segment": "string",
                        "segment_description": "string (comprehensive, 2-3 sentences)",
                        "key_products_services": "list[string] (specific brand/product names)",
                        "primary_source": "string short",
                    }
                ]
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=2000)


def expand_key_items_per_segment(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    sec_doc_url: str,
    segment_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Produce CSV3 rows: key items per segment with short + long description.
    
    Process each segment individually to prevent token truncation.
    """
    all_rows: List[Dict[str, Any]] = []
    
    for seg_row in segment_rows:
        segment_name = seg_row.get("segment", "Unknown")
        system = (
            "You expand a single business segment description into 5-10 key product/service items. "
            "Use SPECIFIC brand names and product names when mentioned in the source "
            "(e.g., 'Azure', 'Microsoft 365 Commercial', 'Dynamics 365', 'LinkedIn', 'GitHub', 'Office 365'). "
            "Do NOT use generic terms when specific product names are available. "
            "Output STRICT JSON ONLY."
        )
        user = json.dumps(
            {
                "ticker": ticker,
                "company_name": company_name,
                "sec_doc_url": sec_doc_url,
                "segment": seg_row,
                "output_schema": {
                    "rows": [
                        {
                            "segment": "string (must match input segment name)",
                            "business_item": "string (specific product/brand name)",
                            "business_item_short_description": "string (1 sentence)",
                            "business_item_long_description": "string (2-3 sentences)",
                            "primary_source": "string short",
                        }
                    ]
                },
            },
            ensure_ascii=False,
        )
        try:
            result = llm.json_call(system=system, user=user, max_output_tokens=1800)
            rows = result.get("rows", [])
            # Ensure segment name is consistent
            for row in rows:
                row["segment"] = segment_name
            all_rows.extend(rows)
        except Exception as e:
            print(f"[{ticker}] Warning: expand_key_items failed for segment '{segment_name}': {e}", flush=True)
            continue
    
    return {"rows": all_rows}

