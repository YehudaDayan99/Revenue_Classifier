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
    extract_revenue_rows_from_grid,
    extract_table_grid_normalized,
    infer_table_layout,
    rank_candidates_for_financial_tables,
    extract_keyword_windows,
    select_other_revenue_tables,
    select_segment_revenue_table,
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

            # ReAct loop for segment revenue table
            seg_choice: Dict[str, Any] = {}
            seg_layout: Dict[str, Any] = {}
            seg_year: Optional[int] = None
            seg_values: Dict[str, int] = {}
            validation = None
            ambiguous = False

            for it in range(1, max_react_iters + 1):
                seg_choice = select_segment_revenue_table(
                    llm,
                    ticker=ticker,
                    company_name=company_name,
                    candidates=candidates,
                    scout=scout,
                    snippets=snippets,
                )
                (t_art / f"segment_choice_iter{it}.json").write_text(
                    json.dumps(seg_choice, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                _trace_append(t_art, {"stage": "segment_select", "iter": it, "choice": seg_choice})

                if seg_choice.get("kind") == "not_found":
                    per["errors"].append("Segment revenue table not found by LLM")
                    ambiguous = True
                    break

                table_id = str(seg_choice.get("table_id") or "")
                if not table_id:
                    per["errors"].append("LLM did not return table_id")
                    ambiguous = True
                    break

                cand = next((c for c in candidates if c.table_id == table_id), None)
                if cand is None:
                    per["errors"].append(f"LLM returned unknown table_id: {table_id}")
                    ambiguous = True
                    break

                grid = extract_table_grid_normalized(html_path, table_id)
                seg_layout = infer_table_layout(
                    llm,
                    ticker=ticker,
                    company_name=company_name,
                    table_id=table_id,
                    candidate=cand,
                    grid=grid,
                )
                (t_art / f"segment_layout_iter{it}.json").write_text(
                    json.dumps(seg_layout, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                _trace_append(t_art, {"stage": "segment_layout", "iter": it, "table_id": table_id, "layout": seg_layout})

                seg_year, seg_values = extract_revenue_rows_from_grid(grid, layout=seg_layout)
                # Validate vs total revenue (primary: income statement; fallback: SEC companyfacts)
                total_rev: Optional[int] = None
                if income_cand is not None:
                    try:
                        inc_grid = extract_table_grid_normalized(html_path, income_cand.table_id)
                        inc_layout = infer_table_layout(
                            llm,
                            ticker=ticker,
                            company_name=company_name,
                            table_id=income_cand.table_id,
                            candidate=income_cand,
                            grid=inc_grid,
                        )
                        (t_art / f"income_statement_layout_{income_cand.table_id}.json").write_text(
                            json.dumps(inc_layout, indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                        total_rev = _extract_row_value_for_year(
                            inc_grid, layout=inc_layout, row_label_regex=r"^revenue[s]?$", year=seg_year
                        )
                    except Exception:
                        total_rev = None
                if total_rev is None:
                    total_rev = fetch_companyfacts_total_revenue_usd(cik, seg_year)
                validation = validate_segment_table(
                    segment_revenues_usd=seg_values,
                    total_revenue_usd=total_rev,
                    tolerance_pct=validation_tolerance_pct,
                )
                (t_art / f"segment_validation_iter{it}.json").write_text(
                    json.dumps(asdict(validation), indent=2), encoding="utf-8"
                )
                _trace_append(t_art, {"stage": "segment_validate", "iter": it, "year": seg_year, "validation": asdict(validation)})

                if validation.ok:
                    break
                ambiguous = True

            if not seg_values or seg_year is None:
                per["errors"].append("Failed to extract segment revenues")
                continue
            # If we could not validate after retries, mark ambiguous and skip emitting CSV outputs for this ticker.
            if validation is not None and not validation.ok:
                per["errors"].append("Validation failed after retries; marking ambiguous and skipping CSV outputs for this ticker.")
                per["ambiguous"] = True
                continue

            # CSV1 rows
            total_for_pct = validation.total_revenue_usd or sum(seg_values.values())
            for seg, rev in seg_values.items():
                pct = (rev / total_for_pct * 100.0) if total_for_pct else 0.0
                csv1_rows.append(
                    {
                        "Year": seg_year,
                        "Company": company_name,
                        "Ticker": ticker,
                        "Segment": seg,
                        "Income $": rev,
                        "Income %": round(pct, 4),
                        "Primary source": f"10-K segment revenue table ({seg_choice.get('table_id')})",
                        "Link": sec_doc_url,
                    }
                )

            # CSV2 + CSV3 via LLM
            html_text = _html_text_for_llm(html_path)
            seg_names = sorted(seg_values.keys())
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

            # CSV4: additional revenue tables
            other = select_other_revenue_tables(
                llm,
                ticker=ticker,
                company_name=company_name,
                candidates=candidates,
                scout=scout,
                snippets=snippets,
                exclude_table_ids=[str(seg_choice.get("table_id") or "")],
            )
            (t_art / "other_revenue_tables.json").write_text(
                json.dumps(other, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            _trace_append(t_art, {"stage": "other_tables_select", "tables": other})

            for tinfo in (other.get("tables") or []):
                table_id = str(tinfo.get("table_id") or "")
                if not table_id:
                    continue
                cand = next((c for c in candidates if c.table_id == table_id), None)
                if cand is None:
                    continue
                grid = extract_table_grid_normalized(html_path, table_id)
                layout = infer_table_layout(
                    llm,
                    ticker=ticker,
                    company_name=company_name,
                    table_id=table_id,
                    candidate=cand,
                    grid=grid,
                )
                (t_art / f"other_layout_{table_id}.json").write_text(
                    json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8"
                )

                year, values = extract_revenue_rows_from_grid(grid, layout=layout)
                for label, rev in values.items():
                    csv4_rows.append(
                        {
                            "Year": year,
                            "Company": company_name,
                            "Ticker": ticker,
                            "Associated segment": "",  # best-effort later
                            "Item": label,
                            "Revenue $": rev,
                            "Table kind": tinfo.get("kind", ""),
                            "Table id": table_id,
                            "Primary source": f"10-K revenue table ({table_id})",
                            "Link": sec_doc_url,
                        }
                    )

            per["ok"] = True
            per["segment_year"] = seg_year
            per["n_segments"] = len(seg_values)
            per["validation"] = asdict(validation) if validation else None

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

