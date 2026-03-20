#!/usr/bin/env python3
"""
Revenue Extraction Validator Dashboard v0.5

Changes in v0.5:
- Reactive 10-K viewer: click a line → scrolls to source table + highlights row
- "View in 10-K" in description → scrolls to evidence passage + highlights
- Uses st.components.html() for full JS execution inside iframe
- Single unified 10-K panel (no more tabs)
- Provenance display: source section tag + evidence snippet

Usage:
    streamlit run dashboard_v5.py

Requires:
    pip install streamlit pandas
"""

import json
import re
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# --- Configuration ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = _PROJECT_ROOT / "dashboard_data.json"
STATE_FILE = _PROJECT_ROOT / "review_state.json"
FILINGS_DIR = _PROJECT_ROOT / "data" / "filings"
VIEWER_HEIGHT = 780  # px


def inject_css():
    st.markdown("""
    <style>
    .block-container { padding-top: 0.5rem !important; }
    header[data-testid="stHeader"] { height: 0; min-height: 0; }

    /* Ticker nav */
    .ticker-nav {
        display: flex; align-items: center; gap: 8px;
        padding: 4px 0; margin-bottom: 2px;
    }
    .ticker-nav .ticker-name { font-size: 1.4rem; font-weight: 700; color: #fff; margin: 0 8px; }
    .ticker-nav .ticker-pos  { color: #888; font-size: 0.85rem; }
    .review-badge { font-size: 0.78rem; padding: 2px 8px; border-radius: 10px; margin-left: 8px; }
    .badge-approved { background: #1a4d1a; color: #6f6; }
    .badge-rejected { background: #4d1a1a; color: #f66; }
    .badge-pending  { background: #333; color: #aaa; }

    /* Force text wrap in dataframe */
    div[data-testid="stDataFrame"] td,
    div[data-testid="stDataFrame"] th {
        white-space: normal !important;
        word-wrap: break-word !important;
    }

    /* Description panel */
    .desc-panel {
        background: #14141f; border: 1px solid #333; border-radius: 6px;
        padding: 10px 14px; margin-top: 4px;
        max-height: 280px; overflow-y: auto;
    }
    .desc-panel h4 { margin: 0 0 4px 0; color: #4a9eff; font-size: 0.92rem; }
    .desc-panel .desc-text {
        color: #ccc; line-height: 1.4; font-size: 0.85rem;
        word-wrap: break-word; overflow-wrap: break-word;
    }
    .desc-panel .desc-meta { color: #888; font-size: 0.75rem; margin-top: 4px; }
    .desc-panel .source-tag {
        display: inline-block; font-size: 0.72rem; padding: 1px 6px;
        border-radius: 3px; margin-right: 6px;
    }
    .tag-footnote    { background: #2d4a2d; color: #8f8; }
    .tag-item1       { background: #2d2d4a; color: #88f; }
    .tag-item8       { background: #4a3d2d; color: #fb8; }
    .tag-table       { background: #4a2d3d; color: #f8b; }
    .tag-rag         { background: #2d4a4a; color: #8ff; }
    .tag-other       { background: #333;    color: #aaa; }
    </style>
    """, unsafe_allow_html=True)


# --- State ---

def load_dashboard_data() -> Dict[str, Any]:
    if not DATA_FILE.exists():
        st.error(f"Data file not found: {DATA_FILE}")
        st.info("Run: python prepare_dashboard_data_v3.py --input <dir>")
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


def load_filing_html(ticker: str) -> Optional[str]:
    """Load filing HTML — prefer annotated version with evidence anchors."""
    # Annotated version has evidence anchor spans injected
    annotated = FILINGS_DIR / f"{ticker}_10K_annotated.html"
    if annotated.exists():
        return annotated.read_text(encoding="utf-8", errors="ignore")
    # Fallback to raw
    for suffix in ["_10K.html", "_10-K.html", ".html"]:
        path = FILINGS_DIR / f"{ticker}{suffix}"
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    return None


def get_edgar_url(ticker: str) -> str:
    return f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K&dateb=&owner=include&count=10"


# --- Reactive 10-K Viewer ---

def build_viewer_html(
    filing_html: str,
    table_id: Optional[str],
    source_table_html: Optional[str],
    scroll_target: Optional[str] = None,
    scroll_mode: str = "table",
    highlight_line: Optional[str] = None,
) -> str:
    """Build the full HTML document for the 10-K viewer iframe.

    scroll_mode:
        "table"    → scroll to source table, highlight matching row
        "evidence" → scroll to evidence anchor span
        "none"     → no scroll (initial state)
    """
    # Escape line name for JS string
    hl_line_js = json.dumps(highlight_line or "")

    scroll_js = ""
    if scroll_mode == "table" and table_id:
        scroll_js = f"""
        (function() {{
            // Scroll to source table
            var table = document.getElementById('{table_id}');
            if (table) {{
                table.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                // Highlight the matched row
                var line = {hl_line_js};
                if (line) {{
                    var rows = table.querySelectorAll('tr[data-line]');
                    rows.forEach(function(r) {{
                        if (r.getAttribute('data-line') === line) {{
                            r.classList.add('active-highlight');
                        }}
                    }});
                    // Also try fuzzy match on row text
                    if (!document.querySelector('.active-highlight')) {{
                        var allRows = table.querySelectorAll('tr');
                        var lineLower = line.toLowerCase();
                        allRows.forEach(function(r) {{
                            var cellText = r.cells && r.cells[0] ? r.cells[0].textContent.trim().toLowerCase() : '';
                            if (cellText && (cellText.indexOf(lineLower) >= 0 || lineLower.indexOf(cellText) >= 0)) {{
                                r.classList.add('active-highlight');
                            }}
                        }});
                    }}
                }}
            }}
        }})();
        """
    elif scroll_mode == "evidence" and scroll_target:
        scroll_js = f"""
        (function() {{
            var anchor = document.getElementById('{scroll_target}');
            if (anchor) {{
                anchor.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                anchor.classList.add('evidence-highlight');
            }}
        }})();
        """

    # Build the full viewer document
    viewer = f"""<!DOCTYPE html>
<html>
<head>
<style>
    body {{
        margin: 0; padding: 16px;
        background: white; color: black;
        font-family: serif;
    }}
    /* Highlight styles for extracted rows */
    .hl-extracted-row td {{
        background: #e8f4e8 !important;
    }}
    .active-highlight td,
    .active-highlight {{
        background: #fff3b0 !important;
        outline: 2px solid #e6a800;
        transition: background 0.3s ease;
    }}
    /* Evidence anchor highlight */
    .evidence-anchor {{
        /* subtle underline when not active */
        border-bottom: 1px dotted #999;
    }}
    .evidence-highlight {{
        background: #fff3b0 !important;
        outline: 2px solid #e6a800;
        padding: 2px 4px;
        border-radius: 3px;
        transition: background 0.3s ease;
    }}
    /* Source table border for visibility */
    #{table_id or '_none'} {{
        border: 2px solid #2a7fff !important;
        box-shadow: 0 0 8px rgba(42, 127, 255, 0.3);
    }}
    table {{
        border-collapse: collapse;
    }}
</style>
</head>
<body>
{filing_html}
<script>
    // Wait for DOM + images, then scroll
    window.addEventListener('load', function() {{
        setTimeout(function() {{
            {scroll_js}
        }}, 200);
    }});
</script>
</body>
</html>"""
    return viewer


# --- UI Components ---

def render_ticker_nav(tickers: list, current_idx: int, review_state: Dict[str, Any]):
    ticker = tickers[current_idx]
    status = review_state.get(ticker, {}).get("status", "pending")

    nav_l, nav_c, nav_r = st.columns([1, 6, 1])
    with nav_l:
        if st.button("←", disabled=current_idx == 0, key="nav_prev", use_container_width=True):
            st.session_state["selected_idx"] = current_idx - 1
            _clear_selection()
            st.rerun()
    with nav_c:
        badge_cls = f"badge-{status}"
        badge_lbl = {"approved": "✅ Approved", "rejected": "❌ Rejected",
                     "pending": "⏳ Pending"}.get(status, status)
        st.markdown(
            f'<div class="ticker-nav">'
            f'<span class="ticker-name">{ticker}</span>'
            f'<span class="ticker-pos">{current_idx+1}/{len(tickers)}</span>'
            f'<span class="review-badge {badge_cls}">{badge_lbl}</span>'
            f'</div>', unsafe_allow_html=True)
    with nav_r:
        if st.button("→", disabled=current_idx >= len(tickers)-1, key="nav_next", use_container_width=True):
            st.session_state["selected_idx"] = current_idx + 1
            _clear_selection()
            st.rerun()


def _clear_selection():
    for k in ["selected_line", "selected_line_data", "scroll_mode", "scroll_target"]:
        st.session_state.pop(k, None)


def render_extracted_table(lines: list) -> Optional[str]:
    if not lines:
        st.warning("No line items extracted")
        return None

    df = pd.DataFrame(lines)
    total = sum(x for x in df.get("revenue", []) if pd.notna(x))

    display_df = pd.DataFrame({
        "Revenue Line": df.get("line", df.get("segment", "")),
        "Revenue ($M)": df["revenue"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—"),
        "Segment": df.get("segment", "").apply(lambda x: (x or "")[:40]),
    })

    col_config = {
        "Revenue Line": st.column_config.TextColumn("Revenue Line", width="medium"),
        "Revenue ($M)": st.column_config.TextColumn("Revenue ($M)", width="small"),
        "Segment": st.column_config.TextColumn("Segment", width="small"),
    }

    try:
        event = st.dataframe(
            display_df, use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row",
            column_config=col_config, key="extracted_table",
        )
        selected_rows = event.selection.rows if event.selection else []
        if selected_rows:
            idx = selected_rows[0]
            line_name = lines[idx].get("line") or lines[idx].get("segment")
            st.session_state["selected_line"] = line_name
            st.session_state["selected_line_data"] = lines[idx]
            # Default: scroll to table + highlight row
            st.session_state["scroll_mode"] = "table"
            st.session_state["scroll_target"] = None  # table_id used directly

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
            st.session_state["scroll_mode"] = "table"

    st.caption(f"Total: **${total:,.0f}M**")
    return st.session_state.get("selected_line")


def _source_tag_html(source: str) -> str:
    """Map source string to a colored tag."""
    src = (source or "").lower()
    if "footnote" in src:
        return '<span class="source-tag tag-footnote">📎 Footnote</span>'
    if "item1" in src:
        return '<span class="source-tag tag-item1">📘 Item 1 – Business</span>'
    if "item8" in src:
        return '<span class="source-tag tag-item8">📙 Item 8 – Notes</span>'
    if "table" in src:
        return '<span class="source-tag tag-table">📋 Table Context</span>'
    if "rag" in src or "retrieval" in src:
        return '<span class="source-tag tag-rag">🔍 RAG Retrieval</span>'
    if "segment_enum" in src:
        return '<span class="source-tag tag-item1">📘 Segment Definition</span>'
    if source:
        return f'<span class="source-tag tag-other">📄 {source}</span>'
    return '<span class="source-tag tag-other">—</span>'


def render_description_panel(lines: list, selected_line: Optional[str]):
    if not selected_line:
        st.markdown(
            '<div class="desc-panel"><span style="color:#555;">Click a row to see description + source</span></div>',
            unsafe_allow_html=True)
        return

    line_data = next((l for l in lines if l.get("line") == selected_line), None)
    if not line_data:
        return

    desc = line_data.get("description", "")
    prov = line_data.get("provenance", {})
    source = prov.get("source", "")
    evidence = prov.get("evidence_snippet", "")
    anchor_id = prov.get("evidence_anchor_id", "")

    segment = line_data.get("segment", "")
    revenue = line_data.get("revenue")
    rev_str = f"${revenue:,.0f}M" if revenue else "—"

    # Source tag
    tag_html = _source_tag_html(source)

    # "View in 10-K" button — only if we have an anchor
    view_btn = ""
    if anchor_id and evidence:
        view_btn = "  ←  *click below to locate in 10-K*"

    st.markdown(f"""
    <div class="desc-panel">
        <h4>📝 {selected_line}</h4>
        <div class="desc-text">{desc if desc else '<em>No description extracted</em>'}</div>
        <div class="desc-meta">
            {tag_html} {segment} · {rev_str}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # "View in 10-K" as a real Streamlit button so it can set session_state
    if anchor_id and evidence:
        if st.button(f"📍 View source in 10-K", key="view_evidence", use_container_width=True):
            st.session_state["scroll_mode"] = "evidence"
            st.session_state["scroll_target"] = anchor_id
            st.rerun()


def render_reactive_viewer(ticker: str, ticker_data: Dict[str, Any]):
    """Render the 10-K viewer with reactive scroll."""
    filing_html = load_filing_html(ticker)
    table_id = ticker_data.get("table_id")
    source_table_html = ticker_data.get("source_table_html", "")
    edgar_url = ticker_data.get("edgar_url") or get_edgar_url(ticker)

    selected_line = st.session_state.get("selected_line")
    scroll_mode = st.session_state.get("scroll_mode", "none")
    scroll_target = st.session_state.get("scroll_target")

    if filing_html:
        # Build viewer with scroll instructions
        viewer_html = build_viewer_html(
            filing_html=filing_html,
            table_id=table_id,
            source_table_html=source_table_html,
            scroll_target=scroll_target,
            scroll_mode=scroll_mode if selected_line else "none",
            highlight_line=selected_line,
        )

        # Indicator line
        if scroll_mode == "table" and selected_line and table_id:
            st.caption(f"📍 Showing table `{table_id}` → **{selected_line}**")
        elif scroll_mode == "evidence" and scroll_target:
            st.caption(f"📍 Showing evidence for **{selected_line}**")
        else:
            st.caption("Select a row to navigate the filing")

        components.html(viewer_html, height=VIEWER_HEIGHT, scrolling=True)

    elif source_table_html:
        # No full filing but we have the extracted table
        st.warning("Full 10-K not cached. Showing extracted table only.")
        styled = f"""
        <div style="max-height:600px; overflow-y:auto; border:1px solid #444;
                    padding:15px; background:#1e1e1e; border-radius:5px;">
            <style>
                table {{ border-collapse:collapse; width:100%; color:#fff; }}
                th,td {{ border:1px solid #555; padding:8px; text-align:right; }}
                th {{ background:#333; }} tr:hover {{ background:#2a2a2a; }}
                .hl-extracted-row td {{ background:#1a3a1a !important; }}
            </style>
            {source_table_html}
        </div>"""
        st.markdown(styled, unsafe_allow_html=True)
    else:
        st.warning("No filing data available")
        st.link_button("🔗 Open on SEC EDGAR", edgar_url, use_container_width=True)


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
            label_visibility="collapsed")
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
        m = td.get("metrics", {})
        r = review_state.get(ticker, {})
        rs, ps = r.get("status", "pending"), m.get("status", "UNKNOWN")
        if status_filter != "All" and rs != status_filter.lower():
            continue
        if pipeline_filter != "All" and ps != pipeline_filter:
            continue
        extracted = m.get("calculated_sum") or m.get("extracted_total") or 0
        expected = m.get("expected_total")
        if expected and expected > 1e9:
            expected /= 1e6
        rows.append({
            "Ticker": ticker, "Pipeline": ps, "Review": rs.title(),
            "Lines": len(td.get("lines", [])),
            "Extracted ($M)": f"{extracted:,.0f}" if extracted else "—",
            "Expected ($M)": f"{expected:,.0f}" if expected else "—",
            "Delta %": f"{m.get('delta_pct', 0):.1f}%" if m.get("delta_pct") is not None else "—",
        })
    if not rows:
        st.info("No tickers match filters")
        return

    df = pd.DataFrame(rows)
    def hl(row):
        if row["Review"] == "Approved": return ["background-color:#1a4d1a"]*len(row)
        if row["Review"] == "Rejected": return ["background-color:#4d1a1a"]*len(row)
        return [""]*len(row)
    st.dataframe(df.style.apply(hl, axis=1), use_container_width=True, hide_index=True)

    st.markdown("---")
    all_t = list(data["tickers"].keys())
    approved = sum(1 for v in review_state.values() if v.get("status") == "approved")
    rejected = sum(1 for v in review_state.values() if v.get("status") == "rejected")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(all_t)); c2.metric("Approved", approved)
    c3.metric("Rejected", rejected); c4.metric("Pending", len(all_t)-approved-rejected)
    if all_t:
        st.progress((approved+rejected)/len(all_t),
                     text=f"Review: {(approved+rejected)/len(all_t)*100:.0f}%")


def render_ticker_detail(
    ticker: str, ticker_data: Dict[str, Any],
    review_state: Dict[str, Any], tickers: list, current_idx: int,
) -> Dict[str, Any]:
    """Detail view v0.5 — reactive viewer."""
    lines = ticker_data.get("lines", [])

    render_ticker_nav(tickers, current_idx, review_state)

    # ── Two columns: Left (table + desc) | Right (reactive 10-K) ──
    col_left, col_right = st.columns([3, 5])

    with col_left:
        st.markdown("**Extracted Data**")
        selected_line = render_extracted_table(lines)
        render_description_panel(lines, selected_line)

    with col_right:
        st.markdown("**10-K Source**")
        render_reactive_viewer(ticker, ticker_data)

    st.markdown("---")
    return render_review_controls(ticker, review_state)


# --- Main ---

def main():
    st.set_page_config(page_title="Rev Validator", page_icon="📊", layout="wide")
    inject_css()

    data = load_dashboard_data()
    review_state = load_review_state()
    tickers = list(data["tickers"].keys())

    st.sidebar.title("Navigation")
    view_mode = st.sidebar.radio("View", ["Detail", "Summary"], horizontal=True)

    if view_mode == "Detail":
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
            _clear_selection()
            st.session_state["last_ticker"] = selected_ticker

        st.sidebar.caption(f"{current_idx+1} / {len(tickers)}")
        st.sidebar.markdown("---")
        if st.sidebar.button("📥 Export Review"):
            st.sidebar.download_button("Download JSON", json.dumps(review_state, indent=2),
                                        file_name="review_state.json", mime="application/json")

        updated = render_ticker_detail(
            selected_ticker, data["tickers"][selected_ticker],
            review_state, tickers, current_idx)
        review_state[selected_ticker] = updated
        save_review_state(review_state)

    else:
        render_summary_view(data, review_state)
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("📥 Export Review", json.dumps(review_state, indent=2),
                               file_name="review_state.json", mime="application/json")
        with c2:
            rows = []
            for t, td in data["tickers"].items():
                r = review_state.get(t, {}); m = td.get("metrics", {})
                rows.append({"Ticker": t, "Pipeline": m.get("status"),
                             "Review": r.get("status", "pending"),
                             "Extracted": m.get("calculated_sum"), "Delta%": m.get("delta_pct"),
                             "Notes": r.get("notes",""), "Reviewed": r.get("reviewed_at","")})
            st.download_button("📊 Export CSV", pd.DataFrame(rows).to_csv(index=False),
                               file_name="review_export.csv", mime="text/csv")
        with c3:
            if st.button("🔄 Refresh"):
                st.cache_data.clear(); st.rerun()


if __name__ == "__main__":
    main()
