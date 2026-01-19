from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup

from revseg.llm_client import OpenAIChatClient
from revseg.react_agents import (
    document_scout,
    extract_table_grid_normalized,
    rank_candidates_for_financial_tables,
    extract_keyword_windows,
    discover_primary_business_lines,
    select_segment_revenue_table,
    select_revenue_disaggregation_table,
    infer_disaggregation_layout,
    extract_disaggregation_rows_from_grid,
    extract_segment_revenue_from_segment_results_grid,
    summarize_segment_descriptions,
    expand_key_items_per_segment,
)
from revseg.sec_edgar import SEC_ARCHIVES_BASE, download_latest_10k
from revseg.table_candidates import (
    TableCandidate,
    extract_table_candidates_from_html,
    find_latest_downloaded_filing_dir,
    find_primary_document_html,
    write_candidates_json,
)
from revseg.validate import fetch_companyfacts_total_revenue_usd, validate_segment_table
from revseg.table_kind import tablekind_gate


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


def _sec_doc_url_from_filing_ref(ref: Dict[str, Any]) -> str:
    cik = int(ref["cik"])
    acc = str(ref["accession_number"])
    primary = str(ref["primary_document"])
    return f"{SEC_ARCHIVES_BASE}/{cik}/{acc.replace('-', '')}/{primary}"


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


def _html_text_for_llm(html_path: Path, *, max_chars: int = 250_000) -> str:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _trace_append(t_art: Path, event: Dict[str, Any]) -> None:
    """Append a trace event (JSONL) for audit/debug."""
    trace_path = t_art / "trace.jsonl"
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _override_layout_with_heuristics(
    *,
    grid: List[List[str]],
    layout: Dict[str, Any],
    segments: List[str],
) -> Dict[str, Any]:
    """Fix common LLM layout mistakes deterministically.

    Supports:
    - AAPL-style: one label col (product categories)
    - MSFT-style: segment_col + item_col (product/service lines under segments)
    """
    import re

    # pad
    max_len = max((len(r) for r in grid), default=0)
    if max_len > 0:
        grid = [list(r) + [""] * (max_len - len(r)) for r in grid]

    seg_set = {s.lower() for s in segments}

    def score_col_for_values(col: int) -> int:
        hits = 0
        for r in grid[:50]:
            if col < len(r):
                v = str(r[col] or "").strip().lower()
                if v in seg_set:
                    hits += 1
        return hits

    # Identify a likely segment column if segment names exist in the grid.
    cand_cols = list(range(0, min(6, max_len)))
    seg_col = None
    if cand_cols:
        best_seg = max(cand_cols, key=score_col_for_values)
        if score_col_for_values(best_seg) >= 2:
            seg_col = int(best_seg)
            layout["segment_col"] = seg_col

    # Fix item_col:
    # - If we found a segment_col, item_col should be a different nearby text column.
    # - Otherwise fall back to AAPL heuristic (label column contains segment names).
    def _col_text_stats(col: int) -> tuple[int, int]:
        nonempty = 0
        uniq = set()
        for r in grid[:80]:
            if col >= len(r):
                continue
            v = str(r[col] or "").strip()
            if not v:
                continue
            nonempty += 1
            if len(v) <= 80:
                uniq.add(v.lower())
        return nonempty, len(uniq)

    if seg_col is not None:
        # choose a neighbor col with many non-empty and many unique values
        item_candidates = [c for c in cand_cols if c != seg_col]
        if item_candidates:
            item_best = max(item_candidates, key=lambda c: _col_text_stats(c))
            layout["item_col"] = int(item_best)
    else:
        # AAPL-like fallback: choose col with max segment-name hits (product label col)
        cand_cols = list(range(0, min(20, max_len)))
        if cand_cols:
            best = max(cand_cols, key=score_col_for_values)
            if score_col_for_values(best) > 0:
                layout["item_col"] = int(best)
                if layout.get("segment_col") is not None and int(layout["segment_col"]) == int(best):
                    layout["segment_col"] = None

    # Fix year_cols: detect recent years in first 8 rows
    year_re = re.compile(r"\b(20\d{2})\b")
    year_cols: Dict[str, int] = {}
    for r in grid[:8]:
        for ci, cell in enumerate(r):
            m = year_re.search(str(cell or ""))
            if not m:
                continue
            y = int(m.group(1))
            if y < 2018 or y > 2100:
                continue
            # prefer first seen
            year_cols.setdefault(str(y), int(ci))
    if len(year_cols) >= 2:
        layout["year_cols"] = year_cols
    return layout


def _keyword_hints_for_ticker(ticker: str) -> List[str]:
    t = ticker.upper()
    if t == "AAPL":
        return ["net sales", "total net sales", "category", "product / service"]
    if t == "MSFT":
        return [
            "significant product and service offerings",
            "server products and cloud services",
            "microsoft 365",
            "gaming",
            "linkedin",
            "windows and devices",
            "search and news advertising",
            "dynamics products and cloud services",
        ]
    if t == "GOOGL":
        return [
            "disaggregation of revenue",
            "google search & other",
            "youtube ads",
            "google network",
            "subscriptions, platforms, and devices",
            "hedging gains",
            "total revenues",
        ]
    return []


def _canonicalize_business_lines(ticker: str, segments: List[str]) -> List[str]:
    t = ticker.upper()
    out: List[str] = []
    for s in segments:
        ss = str(s or "").strip()
        if not ss:
            continue
        out.append(ss)

    if t == "AAPL":
        # Normalize to filing's canonical labels
        canon = []
        for s in out:
            low = s.lower()
            if "iphone" == low:
                canon.append("iPhone")
            elif "mac" == low:
                canon.append("Mac")
            elif "ipad" == low:
                canon.append("iPad")
            elif "wear" in low:
                canon.append("Wearables, Home and Accessories")
            elif "service" in low:
                canon.append("Services")
            else:
                canon.append(s)
        # de-dupe preserving order
        seen = set()
        dedup = []
        for s in canon:
            if s.lower() in seen:
                continue
            seen.add(s.lower())
            dedup.append(s)
        return dedup

    return out


def _normalize_label_for_match(label: str) -> str:
    s = str(label or "").strip()
    if not s:
        return ""
    # strip common footnotes like "Services (1)" or "EMEA (1)"
    s = s.split(" (")[0].strip()
    return s


def _infer_segment_from_item(item: str, segments: List[str]) -> str:
    it = _normalize_label_for_match(item).lower()
    for seg in segments:
        s = seg.lower()
        if it.startswith(s):
            return seg
    return ""


def _looks_like_geography_table(rows: List[Dict[str, Any]]) -> bool:
    labs = " ".join([str(r.get("item") or "") for r in rows]).lower()
    geo_hits = 0
    for kw in ["united states", "emea", "apac", "other americas", "china", "japan", "europe"]:
        if kw in labs:
            geo_hits += 1
    return geo_hits >= 2


def _business_line_overlap_score(
    *,
    ticker: str,
    dimension: str,
    rows: List[Dict[str, Any]],
    segments: List[str],
) -> int:
    """Count how many business lines we can match in extracted rows (used to reject wrong tables)."""
    seg_set = {s.lower() for s in segments}
    matched = set()
    if dimension == "product_category":
        for r in rows:
            item = _normalize_label_for_match(r.get("item") or "")
            if item.lower() in seg_set:
                matched.add(item.lower())
        return len(matched)

    # reportable_segments
    for r in rows:
        seg = str(r.get("segment") or "").strip()
        item = str(r.get("item") or "").strip()
        if seg:
            if seg.lower() in seg_set:
                matched.add(seg.lower())
        else:
            inf = _infer_segment_from_item(item, segments)
            if inf:
                matched.add(inf.lower())
    return len(matched)

def _is_corporate_adjustment_item(item: str) -> bool:
    t = str(item or "").lower()
    return ("hedging" in t) or ("corporate" in t)


def _pick_income_statement_candidate(candidates: List[TableCandidate]) -> Optional[TableCandidate]:
    """Heuristic pick: table whose preview mentions Revenue + Net income/Operating income."""
    best: Optional[TableCandidate] = None
    best_score = -1.0
    for c in candidates:
        preview_text = " ".join([" ".join(r) for r in (c.preview or [])]).lower()
        if "revenue" not in preview_text:
            continue
        score = 0.0
        if "net income" in preview_text:
            score += 3.0
        if "operating income" in preview_text:
            score += 2.0
        if "cost of revenue" in preview_text or "cost of sales" in preview_text:
            score += 1.5
        score += float(getattr(c, "money_cell_ratio", 0.0)) * 3.0
        score += float(getattr(c, "numeric_cell_ratio", 0.0)) * 1.0
        if getattr(c, "has_year_header", False):
            score += 1.0
        if score > best_score:
            best_score = score
            best = c
    return best


def _has_negative(values: Dict[str, int]) -> bool:
    return any((v is not None and int(v) < 0) for v in values.values())

def _is_nonempty_revenue_set(values: Dict[str, int], *, min_rows: int = 2) -> bool:
    if not values:
        return False
    if len(values) < min_rows:
        return False
    try:
        s = sum(int(v) for v in values.values())
    except Exception:
        return False
    return s > 0


def _pick_best_by_kind(
    classifications: Dict[str, Any],
    *,
    kind: str,
    fallback_ranked: List[TableCandidate],
    exclude: set[str] | None = None,
) -> List[str]:
    """Return ordered table_ids for a given kind based on LLM classification + fallback rank."""
    exclude = exclude or set()
    tables = (classifications.get("tables") or []) if isinstance(classifications, dict) else []
    scored: List[Tuple[float, str]] = []
    for t in tables:
        try:
            tid = str(t.get("table_id") or "")
            tk = str(t.get("table_kind") or "")
            conf = float(t.get("confidence") or 0.0)
        except Exception:
            continue
        if not tid or tid in exclude:
            continue
        if tk != kind:
            continue
        scored.append((conf, tid))
    scored.sort(reverse=True)
    ordered = [tid for _, tid in scored]

    # IMPORTANT: Do NOT backfill with non-classified tables. If classifier returns none for a kind,
    # the caller should fall back to a dedicated selector, not guess via generic ranking.
    return ordered

def _extract_row_value_for_year(
    grid: List[List[str]],
    *,
    layout: Dict[str, Any],
    row_label_regex: str,
    year: int,
) -> Optional[int]:
    import re

    label_col = int(layout["label_col"])
    year_cols_raw = layout.get("year_cols") or {}
    year_cols: Dict[int, int] = {int(y): int(ci) for y, ci in year_cols_raw.items()}
    if year not in year_cols:
        return None
    value_col = year_cols[year]
    header_rows = set(int(i) for i in (layout.get("header_rows") or []))
    pat = re.compile(row_label_regex, re.IGNORECASE)
    mult = int(layout.get("units_multiplier") or 1)
    if mult <= 0:
        mult = 1

    for r_i, row in enumerate(grid):
        if r_i in header_rows:
            continue
        if label_col >= len(row) or value_col >= len(row):
            continue
        lab = str(row[label_col] or "").strip()
        if not lab:
            continue
        if not pat.search(lab):
            continue
        v = row[value_col]
        # parse simple money-ish values
        try:
            s = str(v).replace("$", "").replace(",", "").strip()
            if s.startswith("(") and s.endswith(")"):
                s = "-" + s[1:-1]
            return int(round(float(s))) * mult
        except Exception:
            continue
    return None


def run_pipeline(
    *,
    tickers: List[str],
    out_dir: Path | str = Path("data/outputs"),
    filings_base_dir: Path | str = Path("data/10k"),
    cache_dir: Path | str = Path(".cache/sec"),
    model: str = "gpt-4.1-mini",
    max_react_iters: int = 3,
    validation_tolerance_pct: float = 0.02,
) -> Dict[str, Any]:
    """End-to-end run for multiple tickers (latest 10-K per ticker)."""
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    filings_base_dir = Path(filings_base_dir).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()

    llm = OpenAIChatClient(model=model)

    artifacts_dir = out_dir.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    csv1_rows: List[Dict[str, Any]] = []
    csv2_rows: List[Dict[str, Any]] = []
    csv3_rows: List[Dict[str, Any]] = []
    csv4_rows: List[Dict[str, Any]] = []

    report: Dict[str, Any] = {"tickers": {}, "outputs_dir": str(out_dir)}

    for t in tickers:
        ticker = str(t).upper().strip()
        if not ticker:
            continue

        per = {"ok": False, "errors": [], "artifacts_dir": str(artifacts_dir / ticker)}
        report["tickers"][ticker] = per
        t_art = artifacts_dir / ticker
        t_art.mkdir(parents=True, exist_ok=True)

        try:
            print(f"[{ticker}] start", flush=True)
            filing_dir = _ensure_filing_dir(ticker, base_dir=filings_base_dir, cache_dir=cache_dir)
            html_path = find_primary_document_html(filing_dir)
            company_name = _read_company_name_from_submission(filing_dir) or ticker
            filing_ref = _read_filing_ref(filing_dir)
            sec_doc_url = _sec_doc_url_from_filing_ref(filing_ref)
            cik = int(filing_ref["cik"])

            # Stage: candidates
            candidates = extract_table_candidates_from_html(html_path, preview_rows=15, preview_cols=10)
            write_candidates_json(candidates, t_art / f"{ticker}_table_candidates.json")
            per["html_path"] = str(html_path)
            per["sec_doc_url"] = sec_doc_url
            income_cand = _pick_income_statement_candidate(candidates)
            per["income_statement_table_id_guess"] = income_cand.table_id if income_cand else None
            _trace_append(t_art, {"stage": "candidates", "n_candidates": len(candidates), "income_guess": per["income_statement_table_id_guess"]})

            scout = document_scout(html_path)
            (t_art / "scout.json").write_text(json.dumps(scout, indent=2), encoding="utf-8")
            # Deterministic snippet retrieval (soft preference toward Item 8 / Notes)
            snippets = extract_keyword_windows(
                html_path,
                keywords=[
                    "Item 8",
                    "Financial Statements",
                    "Notes to Financial Statements",
                    "Segment",
                    "Reportable segment",
                    "disaggregation",
                    "revenue by",
                    "net sales",
                ],
                window_chars=2500,
                max_windows=12,
            )
            (t_art / "retrieved_snippets.json").write_text(json.dumps(snippets, indent=2, ensure_ascii=False), encoding="utf-8")
            _trace_append(t_art, {"stage": "scout", "n_headings": len(scout.get("headings", [])), "n_snippets": len(snippets)})

            # NEW POC FLOW: text-first discovery -> revenue disaggregation table -> deterministic extraction
            discovery = discover_primary_business_lines(
                llm, ticker=ticker, company_name=company_name, snippets=snippets
            )
            (t_art / "business_lines.json").write_text(json.dumps(discovery, indent=2, ensure_ascii=False), encoding="utf-8")
            _trace_append(t_art, {"stage": "discover", "discovery": discovery})

            segments: List[str] = _canonicalize_business_lines(ticker, list(discovery.get("segments") or []))
            include_optional: List[str] = list(discovery.get("include_segments_optional") or [])
            print(f"[{ticker}] business lines: {segments} (optional={include_optional})", flush=True)

            # Priority-0 deterministic gate: remove common non-target tables (unearned/deferred/RPO/etc.)
            gated: list[TableCandidate] = []
            gated_out: list[dict[str, Any]] = []
            for c in candidates:
                dec = tablekind_gate(c)
                if dec.ok:
                    gated.append(c)
                else:
                    gated_out.append(
                        {
                            "table_id": c.table_id,
                            "reason": dec.reason,
                            "negative_hit": dec.negative_hit,
                            "caption": getattr(c, "caption_text", "")[:160],
                            "heading": getattr(c, "heading_context", "")[:160],
                            "row_labels": (getattr(c, "row_label_preview", []) or [])[:8],
                        }
                    )
            (t_art / "tablekind_gate_rejects.json").write_text(
                json.dumps({"rejected": gated_out[:200]}, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            # If gate removes everything (rare), fall back to all candidates.
            if gated:
                candidates_for_select = gated
            else:
                candidates_for_select = candidates

            # If the primary dimension is reportable segments, prefer a segment-results table
            # and extract the 'Revenue' line per segment (MSFT-style).
            if str(discovery.get("dimension") or "") == "reportable_segments":
                seg_choice = select_segment_revenue_table(
                    llm,
                    ticker=ticker,
                    company_name=company_name,
                    candidates=candidates_for_select,
                    scout=scout,
                    snippets=snippets,
                )
                seg_table_id = str(seg_choice.get("table_id") or "")
                if seg_table_id:
                    print(f"[{ticker}] preferred segment table: {seg_table_id}", flush=True)
                    try:
                        seg_grid = extract_table_grid_normalized(html_path, seg_table_id)
                        seg_extracted = extract_segment_revenue_from_segment_results_grid(
                            seg_grid, segments=segments
                        )
                        year = int(seg_extracted["year"])
                        seg_totals = dict(seg_extracted["segment_totals"])
                        total_rev = seg_extracted.get("total_value")
                        if total_rev is None:
                            total_rev = fetch_companyfacts_total_revenue_usd(cik, year)
                        validation = validate_segment_table(
                            segment_revenues_usd=seg_totals,
                            total_revenue_usd=total_rev,
                            tolerance_pct=validation_tolerance_pct,
                        )
                        if validation.ok:
                            # Write trace artifacts for this path
                            (t_art / "segment_choice.json").write_text(
                                json.dumps(seg_choice, indent=2, ensure_ascii=False), encoding="utf-8"
                            )
                            (t_art / "segment_extracted.json").write_text(
                                json.dumps(seg_extracted, indent=2, ensure_ascii=False), encoding="utf-8"
                            )
                            (t_art / "csv1_validation.json").write_text(
                                json.dumps(asdict(validation), indent=2), encoding="utf-8"
                            )
                            _trace_append(t_art, {"stage": "csv1_validate", "validation": asdict(validation), "table_id": seg_table_id})

                            total_for_pct = validation.total_revenue_usd or sum(seg_totals.values())
                            for seg, rev in sorted(seg_totals.items(), key=lambda kv: kv[1], reverse=True):
                                pct = (rev / total_for_pct * 100.0) if total_for_pct else 0.0
                                csv1_rows.append(
                                    {
                                        "Year": year,
                                        "Company": company_name,
                                        "Ticker": ticker,
                                        "Segment": seg,
                                        "Income $": int(rev) * 1_000_000,  # MSFT grid values are in millions
                                        "Income %": round(pct, 4),
                                        "Primary source": f"10-K segment results table ({seg_table_id})",
                                        "Link": sec_doc_url,
                                    }
                                )

                            # Continue to CSV2/CSV3 using these segments
                            html_text = _html_text_for_llm(html_path)
                            seg_names = sorted(seg_totals.keys())
                            print(f"[{ticker}] summarizing segment descriptions (LLM)...", flush=True)
                            seg_desc = summarize_segment_descriptions(
                                llm,
                                ticker=ticker,
                                company_name=company_name,
                                sec_doc_url=sec_doc_url,
                                html_text=html_text,
                                segment_names=seg_names,
                            )
                            (t_art / "csv2_llm.json").write_text(json.dumps(seg_desc, indent=2, ensure_ascii=False), encoding="utf-8")
                            for r in (seg_desc.get("rows") or []):
                                csv2_rows.append(
                                    {
                                        "Company": company_name,
                                        "Ticker": ticker,
                                        "Segment": r.get("segment", ""),
                                        "Segment description": r.get("segment_description", ""),
                                        "Key products / services (keywords)": "; ".join(r.get("key_products_services", []) or []),
                                        "Primary source": r.get("primary_source", "10-K segment/business description"),
                                        "Link": sec_doc_url,
                                    }
                                )

                            print(f"[{ticker}] expanding key items per segment (LLM)...", flush=True)
                            csv3_payload = expand_key_items_per_segment(
                                llm,
                                ticker=ticker,
                                company_name=company_name,
                                sec_doc_url=sec_doc_url,
                                segment_rows=(seg_desc.get("rows") or []),
                            )
                            (t_art / "csv3_llm.json").write_text(
                                json.dumps(csv3_payload, indent=2, ensure_ascii=False), encoding="utf-8"
                            )
                            for r in (csv3_payload.get("rows") or []):
                                csv3_rows.append(
                                    {
                                        "Company Name": company_name,
                                        "Business segment": r.get("segment", ""),
                                        "Business item": r.get("business_item", ""),
                                        "Description of Business item": r.get("business_item_short_description", ""),
                                        "Textual description of the business item- Long form description": r.get(
                                            "business_item_long_description", ""
                                        ),
                                        "Primary source": r.get("primary_source", "10-K segment/business description"),
                                        "Link": sec_doc_url,
                                    }
                                )

                            per["ok"] = True
                            per["segment_year"] = year
                            per["n_segments"] = len(seg_totals)
                            per["validation"] = asdict(validation)
                            print(f"[{ticker}] done", flush=True)
                            continue
                    except Exception:
                        # Fall back to disaggregation selection below
                        pass

            # Select the disaggregation table (most granular revenue table with Total row)
            choice = select_revenue_disaggregation_table(
                llm,
                ticker=ticker,
                company_name=company_name,
                candidates=candidates_for_select,
                scout=scout,
                snippets=snippets,
                segments=segments,
                keyword_hints=_keyword_hints_for_ticker(ticker),
            )
            (t_art / "disagg_choice.json").write_text(json.dumps(choice, indent=2, ensure_ascii=False), encoding="utf-8")
            _trace_append(t_art, {"stage": "disagg_select", "choice": choice})
            preferred_table_id = str(choice.get("table_id") or "")
            if not preferred_table_id:
                per["errors"].append("No disaggregation table selected")
                continue
            print(f"[{ticker}] preferred table: {preferred_table_id}", flush=True)

            # Build a ranked fallback list by keyword hits in candidate context
            hints = [h.lower() for h in _keyword_hints_for_ticker(ticker)]
            def _hint_score(c: TableCandidate) -> int:
                blob = " ".join(
                    [
                        " ".join([" ".join(r) for r in (c.preview or [])]),
                        str(getattr(c, "caption_text", "") or ""),
                        str(getattr(c, "heading_context", "") or ""),
                        str(getattr(c, "nearby_text_context", "") or ""),
                    ]
                ).lower()
                return sum(1 for h in hints if h and h in blob)

            ranked_by_hints = sorted(candidates_for_select, key=_hint_score, reverse=True)
            candidate_table_ids = [preferred_table_id] + [
                c.table_id for c in ranked_by_hints[:15] if c.table_id != preferred_table_id
            ]

            year = None
            rows: List[Dict[str, Any]] = []
            seg_totals: Dict[str, int] = {}
            validation = None
            table_id = ""

            for attempt_id in candidate_table_ids:
                print(f"[{ticker}] try table {attempt_id} ...", flush=True)
                cand = next((c for c in candidates if c.table_id == attempt_id), None)
                if cand is None:
                    continue
                try:
                    grid = extract_table_grid_normalized(html_path, attempt_id)
                    layout = infer_disaggregation_layout(
                        llm,
                        ticker=ticker,
                        company_name=company_name,
                        table_id=attempt_id,
                        candidate=cand,
                        grid=grid,
                        business_lines=segments,
                    )
                    layout = _override_layout_with_heuristics(grid=grid, layout=layout, segments=segments + include_optional)

                    extracted = extract_disaggregation_rows_from_grid(
                        grid,
                        layout=layout,
                        business_lines=segments + include_optional,
                    )
                except Exception:
                    continue
                attempt_rows = extracted.get("rows") or []
                attempt_total = extracted.get("total_value")
                attempt_year = int(extracted["year"])

                if not attempt_rows:
                    continue
                # Reject geography-only tables for tickers where we want business lines
                if _looks_like_geography_table(attempt_rows) and ticker.upper() in {"GOOGL", "MSFT"}:
                    continue

                # Require overlap with intended business lines (prevents index tables and wrong disclosures)
                dim = str(discovery.get("dimension") or "")
                overlap = _business_line_overlap_score(
                    ticker=ticker, dimension=dim, rows=attempt_rows, segments=segments
                )
                if dim == "product_category":
                    # AAPL needs at least 4/5 categories present
                    if overlap < max(3, min(4, len(segments) - 1)):
                        continue
                else:
                    # MSFT/GOOGL: need at least 2 segments matched
                    if overlap < 2:
                        continue
                # Allow negatives only for optional/corporate adjustments; otherwise reject.
                bad = False
                for r in attempt_rows:
                    segv = str(r.get("segment") or "").strip()
                    if int(r.get("value") or 0) < 0 and (segv not in include_optional and segv.lower() != "corporate"):
                        bad = True
                        break
                if bad:
                    continue

                # Aggregate segment totals
                attempt_seg_totals: Dict[str, int] = {}
                if str(discovery.get("dimension")) == "product_category":
                    seg_norm = {s.lower(): s for s in segments}
                    for r in attempt_rows:
                        raw_item = str(r.get("item") or "").strip()
                        if not raw_item:
                            continue
                        # normalize footnotes like "Services (1)" -> "Services"
                        item = _normalize_label_for_match(raw_item)
                        key = item.lower()
                        if key not in seg_norm:
                            continue
                        canon = seg_norm[key]
                        attempt_seg_totals[canon] = attempt_seg_totals.get(canon, 0) + int(r.get("value") or 0)
                else:
                    # Prefer explicit segment totals (e.g., "Google Services total") to avoid double counting.
                    totals_by_seg: Dict[str, int] = {}
                    for r in attempt_rows:
                        seg = str(r.get("segment") or "").strip()
                        item = str(r.get("item") or "").strip()
                        if not seg:
                            seg = _infer_segment_from_item(item, segments) or ""
                        if not seg:
                            continue
                        norm_item = (item or "").lower()
                        # Accept either explicit "Segment total" rows OR direct "Segment" rows (e.g., "Google Cloud")
                        if (("total" in norm_item and seg.lower() in norm_item) or (_normalize_label_for_match(item).lower() == seg.lower())):
                            totals_by_seg[seg] = int(r.get("value") or 0)
                        # Allow corporate adjustments (hedging gains/losses) to contribute to Corporate if requested
                        if include_optional and "Corporate" in include_optional and _is_corporate_adjustment_item(item):
                            totals_by_seg["Corporate"] = totals_by_seg.get("Corporate", 0) + int(r.get("value") or 0)

                    if totals_by_seg:
                        attempt_seg_totals = totals_by_seg
                    else:
                        for r in attempt_rows:
                            seg = str(r.get("segment") or "").strip()
                            item = str(r.get("item") or "").strip()
                            if not seg:
                                seg = _infer_segment_from_item(item, segments) or "Other"
                            # skip subtotal/total lines when summing by segment
                            if "total" in (item or "").lower():
                                continue
                            if include_optional and "Corporate" in include_optional and _is_corporate_adjustment_item(item):
                                seg = "Corporate"
                            attempt_seg_totals[seg] = attempt_seg_totals.get(seg, 0) + int(r.get("value") or 0)

                total_rev = int(attempt_total) if attempt_total is not None else None
                if total_rev is None:
                    total_rev = fetch_companyfacts_total_revenue_usd(cik, attempt_year)
                attempt_validation = validate_segment_table(
                    segment_revenues_usd=attempt_seg_totals,
                    total_revenue_usd=total_rev,
                    tolerance_pct=validation_tolerance_pct,
                )
                if not attempt_validation.ok:
                    continue

                # Accept this table
                table_id = attempt_id
                year = attempt_year
                rows = attempt_rows
                seg_totals = attempt_seg_totals
                validation = attempt_validation
                (t_art / "disagg_layout.json").write_text(json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8")
                (t_art / "disagg_extracted.json").write_text(json.dumps(extracted, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[{ticker}] accepted table {table_id} year={year} segments={len(seg_totals)}", flush=True)
                break

            if not rows or year is None or validation is None:
                per["errors"].append("No rows extracted from disaggregation table")
                continue

            (t_art / "csv1_validation.json").write_text(json.dumps(asdict(validation), indent=2), encoding="utf-8")
            _trace_append(t_art, {"stage": "csv1_validate", "validation": asdict(validation), "table_id": table_id})

            total_for_pct = validation.total_revenue_usd or sum(seg_totals.values())
            for seg, rev in sorted(seg_totals.items(), key=lambda kv: kv[1], reverse=True):
                pct = (rev / total_for_pct * 100.0) if total_for_pct else 0.0
                csv1_rows.append(
                    {
                        "Year": year,
                        "Company": company_name,
                        "Ticker": ticker,
                        "Segment": seg,
                        "Income $": rev,
                        "Income %": round(pct, 4),
                        "Primary source": f"10-K revenue disaggregation table ({table_id})",
                        "Link": sec_doc_url,
                    }
                )

            # CSV2 + CSV3 via LLM
            html_text = _html_text_for_llm(html_path)
            seg_names = sorted(seg_totals.keys())
            print(f"[{ticker}] summarizing segment descriptions (LLM)...", flush=True)
            seg_desc = summarize_segment_descriptions(
                llm,
                ticker=ticker,
                company_name=company_name,
                sec_doc_url=sec_doc_url,
                html_text=html_text,
                segment_names=seg_names,
            )
            (t_art / "csv2_llm.json").write_text(json.dumps(seg_desc, indent=2, ensure_ascii=False), encoding="utf-8")

            for r in (seg_desc.get("rows") or []):
                csv2_rows.append(
                    {
                        "Company": company_name,
                        "Ticker": ticker,
                        "Segment": r.get("segment", ""),
                        "Segment description": r.get("segment_description", ""),
                        "Key products / services (keywords)": "; ".join(r.get("key_products_services", []) or []),
                        "Primary source": r.get("primary_source", "10-K segment/business description"),
                        "Link": sec_doc_url,
                    }
                )

            print(f"[{ticker}] expanding key items per segment (LLM)...", flush=True)
            csv3_payload = expand_key_items_per_segment(
                llm,
                ticker=ticker,
                company_name=company_name,
                sec_doc_url=sec_doc_url,
                segment_rows=(seg_desc.get("rows") or []),
            )
            (t_art / "csv3_llm.json").write_text(
                json.dumps(csv3_payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            for r in (csv3_payload.get("rows") or []):
                csv3_rows.append(
                    {
                        "Company Name": company_name,
                        "Business segment": r.get("segment", ""),
                        "Business item": r.get("business_item", ""),
                        "Description of Business item": r.get("business_item_short_description", ""),
                        "Textual description of the business item- Long form description": r.get(
                            "business_item_long_description", ""
                        ),
                        "Primary source": r.get("primary_source", "10-K segment/business description"),
                        "Link": sec_doc_url,
                    }
                )

            # CSV4: item-level disaggregation rows (for inspection)
            for r in rows:
                seg = str(r.get("segment") or "").strip()
                item = str(r.get("item") or "").strip()
                val = int(r.get("value") or 0)
                csv4_rows.append(
                    {
                        "Year": year,
                        "Company": company_name,
                        "Ticker": ticker,
                        "Associated segment": seg,
                        "Item": item,
                        "Revenue $": val,
                        "Table kind": "revenue_disaggregation",
                        "Table id": table_id,
                        "Primary source": f"10-K revenue disaggregation table ({table_id})",
                        "Link": sec_doc_url,
                    }
                )

            per["ok"] = True
            per["segment_year"] = year
            per["n_segments"] = len(seg_totals)
            per["validation"] = asdict(validation) if validation else None
            print(f"[{ticker}] done", flush=True)

        except Exception as e:
            per["errors"].append(f"{type(e).__name__}: {e}")

    # Write CSVs
    _write_csv(
        out_dir / "csv1_segment_revenue.csv",
        ["Year", "Company", "Ticker", "Segment", "Income $", "Income %", "Primary source", "Link"],
        csv1_rows,
    )
    _write_csv(
        out_dir / "csv2_segment_descriptions.csv",
        [
            "Company",
            "Ticker",
            "Segment",
            "Segment description",
            "Key products / services (keywords)",
            "Primary source",
            "Link",
        ],
        csv2_rows,
    )
    _write_csv(
        out_dir / "csv3_segment_items.csv",
        [
            "Company Name",
            "Business segment",
            "Business item",
            "Description of Business item",
            "Textual description of the business item- Long form description",
            "Primary source",
            "Link",
        ],
        csv3_rows,
    )
    _write_csv(
        out_dir / "csv4_other_revenue_tables.csv",
        [
            "Year",
            "Company",
            "Ticker",
            "Associated segment",
            "Item",
            "Revenue $",
            "Table kind",
            "Table id",
            "Primary source",
            "Link",
        ],
        csv4_rows,
    )

    (out_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_args(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    import argparse

    p = argparse.ArgumentParser(description="Run ReAct revenue segmentation pipeline for tickers.")
    p.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g. MSFT,AAPL,...")
    p.add_argument("--model", default="gpt-4.1-mini")
    p.add_argument("--max-react-iters", type=int, default=3)
    p.add_argument("--out-dir", default="data/outputs")
    args = p.parse_args(argv)
    return {
        "tickers": [t.strip().upper() for t in args.tickers.split(",") if t.strip()],
        "model": args.model,
        "max_react_iters": int(args.max_react_iters),
        "out_dir": Path(args.out_dir),
    }


if __name__ == "__main__":
    cfg = _parse_args()
    run_pipeline(**cfg)

