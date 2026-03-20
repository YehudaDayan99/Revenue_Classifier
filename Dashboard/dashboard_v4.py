#!/usr/bin/env python3
"""
Revenue Extraction Validator Dashboard v0.4

Changes in v0.4:
- Removed title bar + metrics cards (reclaim vertical space)
- Inline ticker nav bar: ← TICKER (n/N) →
- Description panel below Extracted Data in left column (beside 10-K)
- Text wrap enabled on Revenue Line column
- 10-K viewer gets full right-column height

Usage:
    streamlit run dashboard_v4.py

Requires:
    pip install streamlit pandas
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# --- Configuration ---
# Resolve paths from project root so they work regardless of cwd
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = _PROJECT_ROOT / "dashboard_data.json"
STATE_FILE = _PROJECT_ROOT / "review_state.json"
FILINGS_DIR = _PROJECT_ROOT / "data" / "filings"


def inject_css():
    st.markdown("""
    <style>
    /* Tighten top padding aggressively */
    .block-container { padding-top: 0.6rem !important; }
    header[data-testid="stHeader"] { height: 0; min-height: 0; }

    /* Ticker nav bar */
    .ticker-nav {
        display: flex; align-items: center; gap: 8px;
        padding: 6px 0; margin-bottom: 4px;
    }
    .ticker-nav .ticker-name {
        font-size: 1.4rem; font-weight: 700; color: #fff; margin: 0 8px;
    }
    .ticker-nav .ticker-pos {
        color: #888; font-size: 0.85rem; margin-left: 4px;
    }
    .ticker-nav .review-badge {
        font-size: 0.78rem; padding: 2px 8px; border-radius: 10px;
        margin-left: 8px;
    }
    .badge-approved { background: #1a4d1a; color: #6f6; }
    .badge-rejected { background: #4d1a1a; color: #f66; }
    .badge-pending  { background: #333; color: #aaa; }

    /* Force text wrap in st.dataframe cells */
    div[data-testid="stDataFrame"] td {
        white-space: normal !important;
        word-wrap: break-word !important;
    }
    div[data-testid="stDataFrame"] th {
        white-space: normal !important;
    }

    /* Description panel */
    .desc-panel {
        background: #14141f; border: 1px solid #333; border-radius: 6px;
        padding: 12px 16px; margin-top: 6px;
        max-height: 320px; overflow-y: auto;
    }
    .desc-panel h4 { margin: 0 0 6px 0; color: #4a9eff; font-size: 0.95rem; }
    .desc-panel .desc-text {
        color: #ccc; line-height: 1.45; font-size: 0.88rem;
        word-wrap: break-word; overflow-wrap: break-word;
    }
    .desc-panel .desc-meta { color: #888; font-size: 0.78rem; margin-top: 6px; }
    </style>
    """, unsafe_allow_html=True)


# --- State Management ---

def load_dashboard_data() -> Dict[str, Any]:
    if not DATA_FILE.exists():
        st.error(f"Data file not found: {DATA_FILE}")
        st.info("Run: `python prepare_dashboard_data.py --input <pipeline_output_dir>`")
        st.stop()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_review_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_review_state(state: Dict[str, Any]):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def get_edgar_url(ticker: str) -> str:
    return f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K&dateb=&owner=include&count=10"


def load_filing_html(ticker: str) -> Optional[str]:
    for suffix in ["_10K.html", "_10-K.html", ".html"]:
        path = FILINGS_DIR / f"{ticker}{suffix}"
        if path.exists():
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    return None


# --- UI Components ---

def render_ticker_nav(tickers: list, current_idx: int, review_state: Dict[str, Any]):
    """Compact inline nav: ← TICKER (n/N) → with review badge."""
    ticker = tickers[current_idx]
    status = review_state.get(ticker, {}).get("status", "pending")

    nav_left, nav_center, nav_right = st.columns([1, 6, 1])

    with nav_left:
        if st.button("←", disabled=current_idx == 0, key="nav_prev", use_container_width=True):
            st.session_state["selected_idx"] = current_idx - 1
            st.session_state.pop("selected_line", None)
            st.rerun()

    with nav_center:
        badge_class = f"badge-{status}"
        badge_label = {"approved": "✅ Approved", "rejected": "❌ Rejected",
                       "pending": "⏳ Pending"}.get(status, status)
        st.markdown(
            f'<div class="ticker-nav">'
            f'<span class="ticker-name">{ticker}</span>'
            f'<span class="ticker-pos">{current_idx + 1} / {len(tickers)}</span>'
            f'<span class="review-badge {badge_class}">{badge_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with nav_right:
        if st.button("→", disabled=current_idx >= len(tickers) - 1, key="nav_next", use_container_width=True):
            st.session_state["selected_idx"] = current_idx + 1
            st.session_state.pop("selected_line", None)
            st.rerun()


def render_extracted_table(lines: list) -> Optional[str]:
    """Table with row selection + text wrap on Revenue Line."""
    if not lines:
        st.warning("No line items extracted")
        return None

    df = pd.DataFrame(lines)
    total = sum(x for x in df.get("revenue", []) if pd.notna(x))

    display_df = pd.DataFrame({
        "Revenue Line": df.get("line", df.get("segment", "")),
        "Revenue ($M)": df["revenue"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
        ),
        "Segment": df.get("segment", "").apply(lambda x: (x or "")[:40]),
    })

    # Column config for text wrapping
    col_config = {
        "Revenue Line": st.column_config.TextColumn(
            "Revenue Line",
            width="medium",
        ),
        "Revenue ($M)": st.column_config.TextColumn(
            "Revenue ($M)",
            width="small",
        ),
        "Segment": st.column_config.TextColumn(
            "Segment",
            width="small",
        ),
    }

    try:
        event = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config=col_config,
            key="extracted_table",
        )
        selected_rows = event.selection.rows if event.selection else []
        if selected_rows:
            idx = selected_rows[0]
            selected = lines[idx].get("line") or lines[idx].get("segment")
            st.session_state["selected_line"] = selected
            st.session_state["selected_line_data"] = lines[idx]

    except (TypeError, AttributeError):
        line_labels = [
            f"{l.get('line', '?')}  —  ${l['revenue']:,.0f}M"
            if l.get("revenue") else l.get("line", "?")
            for l in lines
        ]
        choice = st.radio("Select", line_labels, index=None,
                          key="line_radio", label_visibility="collapsed")
        if choice:
            idx = line_labels.index(choice)
            st.session_state["selected_line"] = lines[idx].get("line")
            st.session_state["selected_line_data"] = lines[idx]

    st.markdown(f"**Total: ${total:,.0f}M**")
    return st.session_state.get("selected_line")


def render_description_panel(lines: list, selected_line: Optional[str]):
    """Description panel — sits below extracted table, left column."""
    if not selected_line:
        st.markdown(
            '<div class="desc-panel"><span style="color:#666;">Click a row above to see its description</span></div>',
            unsafe_allow_html=True,
        )
        return

    line_data = next((l for l in lines if l.get("line") == selected_line), None)
    if not line_data:
        return

    desc = line_data.get("description", "")
    segment = line_data.get("segment", "")
    revenue = line_data.get("revenue")
    fy = line_data.get("fiscal_year", "")

    rev_str = f"${revenue:,.0f}M" if revenue else "—"
    meta_parts = [s for s in [
        f"Segment: {segment}" if segment else None,
        f"Revenue: {rev_str}",
        f"FY{fy}" if fy else None,
    ] if s]

    st.markdown(f"""
    <div class="desc-panel">
        <h4>📝 {selected_line}</h4>
        <div class="desc-text">{desc if desc else '<em>No description extracted</em>'}</div>
        <div class="desc-meta">{'  ·  '.join(meta_parts)}</div>
    </div>
    """, unsafe_allow_html=True)


def render_filing_viewer(ticker: str, ticker_data: Dict[str, Any]):
    """Full 10-K viewer — takes full right column height."""
    filing_html = load_filing_html(ticker)
    table_html = ticker_data.get("html_table", "")
    has_table = table_html and not table_html.startswith("<p><em>")
    edgar_url = ticker_data.get("edgar_url") or get_edgar_url(ticker)

    tab1, tab2 = st.tabs(["📊 Extracted Table", "📄 Full 10-K Filing"])

    with tab1:
        if has_table:
            st.markdown(f"""
            <div style="max-height:750px; overflow-y:auto; border:1px solid #444;
                        padding:15px; background:#1e1e1e; border-radius:5px;">
                <style>
                    table {{ border-collapse:collapse; width:100%; color:#fff; }}
                    th,td {{ border:1px solid #555; padding:8px; text-align:right; }}
                    th {{ background:#333; }} tr:hover {{ background:#2a2a2a; }}
                </style>
                {table_html}
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Original table HTML not found in pipeline artifacts")

    with tab2:
        if filing_html:
            st.success(f"✅ Local 10-K: data/filings/{ticker}_10K.html")
            st.markdown(f"""
            <div style="height:800px; overflow-y:auto; border:1px solid #444;
                        padding:20px; background:white; color:black; border-radius:5px;">
                {filing_html}
            </div>""", unsafe_allow_html=True)
        else:
            st.warning("Full 10-K not cached locally")
            c1, c2 = st.columns(2)
            with c1:
                st.link_button("🔗 SEC EDGAR", edgar_url, use_container_width=True)
            with c2:
                if st.button("📥 Cache Filing", use_container_width=True):
                    st.info(f"Save to data/filings/{ticker}_10K.html → refresh")


def render_review_controls(ticker: str, review_state: Dict[str, Any]) -> Dict[str, Any]:
    ticker_state = review_state.get(ticker, {
        "status": "pending", "notes": "", "reviewed_at": None,
    })

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("✅ Approve", key=f"approve_{ticker}", type="primary"):
            ticker_state["status"] = "approved"
            ticker_state["reviewed_at"] = datetime.now().isoformat()
    with col2:
        if st.button("❌ Reject", key=f"reject_{ticker}"):
            ticker_state["status"] = "rejected"
            ticker_state["reviewed_at"] = datetime.now().isoformat()
    with col3:
        ticker_state["notes"] = st.text_input(
            "Notes", value=ticker_state.get("notes", ""),
            key=f"notes_{ticker}", placeholder="Review notes...",
            label_visibility="collapsed",
        )
    return ticker_state


def render_summary_view(data: Dict[str, Any], review_state: Dict[str, Any]):
    st.subheader("📊 Summary")

    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox("Review Status", ["All", "Pending", "Approved", "Rejected"])
    with col2:
        pipeline_filter = st.selectbox("Pipeline Status", ["All", "PASS", "FAIL", "UNKNOWN"])

    rows = []
    for ticker, td in data["tickers"].items():
        metrics = td.get("metrics", {})
        review = review_state.get(ticker, {})
        rs = review.get("status", "pending")
        ps = metrics.get("status", "UNKNOWN")
        if status_filter != "All" and rs != status_filter.lower():
            continue
        if pipeline_filter != "All" and ps != pipeline_filter:
            continue

        extracted = metrics.get("calculated_sum") or metrics.get("extracted_total") or 0
        expected = metrics.get("expected_total")
        if expected and expected > 1e9:
            expected /= 1e6

        rows.append({
            "Ticker": ticker, "Pipeline": ps, "Review": rs.title(),
            "Lines": len(td.get("lines", [])),
            "Extracted ($M)": f"{extracted:,.0f}" if extracted else "—",
            "Expected ($M)": f"{expected:,.0f}" if expected else "—",
            "Delta %": f"{metrics.get('delta_pct', 0):.1f}%" if metrics.get("delta_pct") is not None else "—",
        })

    if not rows:
        st.info("No tickers match filters")
        return

    df = pd.DataFrame(rows)

    def highlight(row):
        if row["Review"] == "Approved":
            return ["background-color: #1a4d1a"] * len(row)
        elif row["Review"] == "Rejected":
            return ["background-color: #4d1a1a"] * len(row)
        return [""] * len(row)

    st.dataframe(df.style.apply(highlight, axis=1), use_container_width=True, hide_index=True)

    st.markdown("---")
    all_t = list(data["tickers"].keys())
    approved = sum(1 for r in review_state.values() if r.get("status") == "approved")
    rejected = sum(1 for r in review_state.values() if r.get("status") == "rejected")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(all_t))
    c2.metric("Approved", approved)
    c3.metric("Rejected", rejected)
    c4.metric("Pending", len(all_t) - approved - rejected)
    if all_t:
        st.progress((approved + rejected) / len(all_t),
                     text=f"Review Progress: {(approved + rejected) / len(all_t) * 100:.0f}%")


def render_ticker_detail(ticker: str, ticker_data: Dict[str, Any],
                         review_state: Dict[str, Any], tickers: list, current_idx: int) -> Dict[str, Any]:
    """Detail view v0.4 — compact, max data density."""
    lines = ticker_data.get("lines", [])

    # ── Inline ticker nav ──
    render_ticker_nav(tickers, current_idx, review_state)

    # ── Two columns: Left = table + description | Right = 10-K viewer ──
    col_left, col_right = st.columns([3, 5])

    with col_left:
        st.markdown("**Extracted Data**")
        selected_line = render_extracted_table(lines)
        # Description sits directly below the table, still in left column
        render_description_panel(lines, selected_line)

    with col_right:
        st.markdown("**10-K Source**")
        render_filing_viewer(ticker, ticker_data)

    # ── Compact review controls ──
    st.markdown("---")
    return render_review_controls(ticker, review_state)


# --- Main ---

def main():
    st.set_page_config(page_title="Rev Validator", page_icon="📊", layout="wide")
    inject_css()

    data = load_dashboard_data()
    review_state = load_review_state()
    tickers = list(data["tickers"].keys())

    # Sidebar — minimal
    st.sidebar.title("Navigation")
    view_mode = st.sidebar.radio("View", ["Detail", "Summary"], horizontal=True)

    if view_mode == "Detail":
        # Ticker selector in sidebar
        ticker_options = []
        for t in tickers:
            s = review_state.get(t, {}).get("status", "pending")
            icon = {"approved": "✅", "rejected": "❌", "pending": "⏳"}.get(s, "")
            ticker_options.append(f"{icon} {t}")

        default_idx = st.session_state.get("selected_idx", 0)
        if default_idx >= len(ticker_options):
            default_idx = 0

        selected_option = st.sidebar.selectbox("Ticker", ticker_options, index=default_idx)
        selected_ticker = selected_option.split()[-1]
        current_idx = tickers.index(selected_ticker)
        st.session_state["selected_idx"] = current_idx

        if st.session_state.get("last_ticker") != selected_ticker:
            for k in ["selected_line", "selected_line_data"]:
                st.session_state.pop(k, None)
            st.session_state["last_ticker"] = selected_ticker

        st.sidebar.caption(f"{current_idx + 1} / {len(tickers)}")
        st.sidebar.markdown("---")
        if st.sidebar.button("📥 Export Review"):
            st.sidebar.download_button("Download JSON", json.dumps(review_state, indent=2),
                                        file_name="review_state.json", mime="application/json")

        updated = render_ticker_detail(
            selected_ticker, data["tickers"][selected_ticker],
            review_state, tickers, current_idx,
        )
        review_state[selected_ticker] = updated
        save_review_state(review_state)

    else:
        render_summary_view(data, review_state)
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("📥 Export Review State", json.dumps(review_state, indent=2),
                               file_name="review_state.json", mime="application/json")
        with c2:
            rows = []
            for t, td in data["tickers"].items():
                r = review_state.get(t, {})
                m = td.get("metrics", {})
                rows.append({"Ticker": t, "Pipeline": m.get("status"),
                             "Review": r.get("status", "pending"),
                             "Extracted": m.get("calculated_sum"), "Delta%": m.get("delta_pct"),
                             "Notes": r.get("notes", ""), "Reviewed": r.get("reviewed_at", "")})
            st.download_button("📊 Export CSV", pd.DataFrame(rows).to_csv(index=False),
                               file_name="review_export.csv", mime="text/csv")
        with c3:
            if st.button("🔄 Refresh"):
                st.cache_data.clear()
                st.rerun()


if __name__ == "__main__":
    main()
