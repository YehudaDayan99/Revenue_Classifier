#!/usr/bin/env python3
"""
Revenue Extraction Validator Dashboard v0.2

New in v0.2:
- Full 10-K viewer (HTML iframe or EDGAR link)
- Clickable line items to show descriptions
- Better artifact loading

Usage:
    streamlit run dashboard.py

Requires:
    pip install streamlit pandas
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import urllib.parse

# --- Configuration ---
# Resolve paths from project root (parent of Dashboard/) so they work regardless of cwd
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = _PROJECT_ROOT / "dashboard_data.json"
STATE_FILE = _PROJECT_ROOT / "review_state.json"
FILINGS_DIR = _PROJECT_ROOT / "data" / "filings"  # Local cache of 10-K HTML files

# --- State Management ---

def load_dashboard_data() -> Dict[str, Any]:
    """Load dashboard data from JSON."""
    if not Path(DATA_FILE).exists():
        st.error(f"Data file not found: {DATA_FILE}")
        st.info("Run: `python prepare_dashboard_data.py --input <pipeline_output_dir>`")
        st.stop()
    
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_review_state() -> Dict[str, Any]:
    """Load persistent review state."""
    if Path(STATE_FILE).exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_review_state(state: Dict[str, Any]):
    """Save review state to disk."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def get_edgar_url(ticker: str, cik: Optional[str] = None) -> str:
    """Generate EDGAR search URL for ticker."""
    # Direct search URL - user can find the 10-K from here
    return f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K&dateb=&owner=include&count=10"


def load_filing_html(ticker: str) -> Optional[str]:
    """Load cached 10-K HTML if available."""
    patterns = [
        FILINGS_DIR / f"{ticker}_10K.html",
        FILINGS_DIR / f"{ticker}_10-K.html", 
        FILINGS_DIR / f"{ticker}.html",
    ]
    
    for path in patterns:
        if path.exists():
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    
    return None


# --- UI Components ---

def render_metrics_card(metrics: Dict[str, Any]):
    """Render validation metrics as cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    status = metrics.get("status", "UNKNOWN")
    status_color = "🟢" if status == "PASS" else "🔴" if status == "FAIL" else "🟡"
    
    with col1:
        st.metric("Status", f"{status_color} {status}")
    
    with col2:
        extracted = metrics.get("calculated_sum") or metrics.get("extracted_total")
        if extracted:
            st.metric("Extracted Total", f"${extracted:,.0f}M")
        else:
            st.metric("Extracted Total", "—")
    
    with col3:
        expected = metrics.get("expected_total")
        if expected:
            # Handle both raw and already-divided values
            if expected > 1e9:
                st.metric("Expected Total", f"${expected/1e6:,.0f}M")
            else:
                st.metric("Expected Total", f"${expected:,.0f}M")
        else:
            st.metric("Expected Total", "—")
    
    with col4:
        delta = metrics.get("delta_pct")
        if delta is not None:
            color = "normal" if abs(delta) < 2 else "inverse"
            st.metric("Delta", f"{delta:.2f}%", delta_color=color)
        else:
            st.metric("Delta", "—")


def render_extracted_data_interactive(lines: list) -> Optional[str]:
    """Render extracted line items as clickable table. Returns selected line name."""
    if not lines:
        st.warning("No line items extracted")
        return None
    
    df = pd.DataFrame(lines)
    
    # Format revenue column
    if "revenue" in df.columns:
        df["revenue_fmt"] = df["revenue"].apply(
            lambda x: f"${x:,.0f}M" if pd.notna(x) else "—"
        )
    
    # Add index for selection
    df["idx"] = range(len(df))
    
    # Calculate total
    total = sum(x for x in df["revenue"] if pd.notna(x))
    
    # Create clickable buttons for each line
    selected_line = None
    
    # Header
    cols = st.columns([3, 2, 2])
    cols[0].markdown("**Revenue Line**")
    cols[1].markdown("**Revenue ($M)**")
    cols[2].markdown("**Segment**")
    
    # Rows as buttons
    for idx, row in df.iterrows():
        cols = st.columns([3, 2, 2])
        
        line_name = row.get("line", "Unknown")
        revenue = row.get("revenue_fmt", "—")
        segment = row.get("segment", "")[:30]
        
        # Make line name a button
        if cols[0].button(f"📄 {line_name}", key=f"line_{idx}", use_container_width=True):
            selected_line = line_name
            st.session_state["selected_line"] = line_name
            st.session_state["selected_line_data"] = row.to_dict()
        
        cols[1].markdown(revenue)
        cols[2].markdown(segment if segment else "—")
    
    # Total row
    st.markdown(f"**Total: ${total:,.0f}M**")
    
    # Check session state for selection
    if "selected_line" in st.session_state:
        selected_line = st.session_state["selected_line"]
    
    return selected_line


def render_description_panel(lines: list, selected_line: Optional[str]):
    """Render description for selected line item."""
    if not selected_line:
        st.info("👆 Click a revenue line to see its description")
        return
    
    # Find the line data
    line_data = None
    for line in lines:
        if line.get("line") == selected_line:
            line_data = line
            break
    
    if not line_data:
        st.warning(f"No data found for: {selected_line}")
        return
    
    desc = line_data.get("description", "")
    
    st.markdown(f"### 📝 {selected_line}")
    
    if desc:
        st.markdown(f"> {desc}")
    else:
        st.warning("*No description extracted for this line item*")
    
    # Show additional metadata
    with st.expander("Details"):
        st.json({
            "segment": line_data.get("segment", ""),
            "revenue": line_data.get("revenue"),
            "fiscal_year": line_data.get("fiscal_year", ""),
        })


def render_filing_viewer(ticker: str, ticker_data: Dict[str, Any]):
    """Render full 10-K filing viewer."""
    
    # Try to load local HTML first
    filing_html = load_filing_html(ticker)
    
    # Check for embedded table HTML
    table_html = ticker_data.get("html_table", "")
    has_table = table_html and not table_html.startswith("<p><em>")
    
    # Check for EDGAR URL in data
    edgar_url = ticker_data.get("edgar_url") or get_edgar_url(ticker)
    
    # Viewer tabs
    tab1, tab2 = st.tabs(["📊 Extracted Table", "📄 Full 10-K Filing"])
    
    with tab1:
        if has_table:
            # Wrap in scrollable container
            styled_html = f"""
            <div style="
                max-height: 500px; 
                overflow-y: auto; 
                border: 1px solid #444; 
                padding: 15px;
                background: #1e1e1e;
                border-radius: 5px;
            ">
                <style>
                    table {{ border-collapse: collapse; width: 100%; color: #fff; }}
                    th, td {{ border: 1px solid #555; padding: 8px; text-align: right; }}
                    th {{ background: #333; }}
                    tr:hover {{ background: #2a2a2a; }}
                </style>
                {table_html}
            </div>
            """
            st.markdown(styled_html, unsafe_allow_html=True)
        else:
            st.info("Original table HTML not found in pipeline artifacts")
            st.caption("The pipeline needs to save selected table HTML to artifacts directory")
    
    with tab2:
        if filing_html:
            # Full filing available locally
            st.success(f"✅ Local 10-K loaded from: data/filings/{ticker}_10K.html")
            
            # Embed in iframe-like container
            filing_container = f"""
            <div style="
                height: 600px; 
                overflow-y: auto; 
                border: 1px solid #444; 
                padding: 20px;
                background: white;
                color: black;
                border-radius: 5px;
            ">
                {filing_html}
            </div>
            """
            st.markdown(filing_container, unsafe_allow_html=True)
            
        else:
            # No local file - show options
            st.warning("Full 10-K not cached locally")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.link_button(
                    "🔗 Open in SEC EDGAR",
                    edgar_url,
                    use_container_width=True,
                )
            
            with col2:
                if st.button("📥 Cache Filing Locally", use_container_width=True):
                    st.info(f"""
                    To cache filings locally:
                    1. Download 10-K HTML from EDGAR
                    2. Save to: `data/filings/{ticker}_10K.html`
                    3. Refresh dashboard
                    """)
            
            st.markdown("---")
            st.caption(f"EDGAR Search URL: {edgar_url}")


def render_review_controls(ticker: str, review_state: Dict[str, Any]) -> Dict[str, Any]:
    """Render review controls and return updated state."""
    ticker_state = review_state.get(ticker, {
        "status": "pending",
        "notes": "",
        "reviewed_at": None,
    })
    
    st.markdown("---")
    st.subheader("Review")
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        if st.button("✅ Approve", key=f"approve_{ticker}", type="primary"):
            ticker_state["status"] = "approved"
            ticker_state["reviewed_at"] = datetime.now().isoformat()
            # Clear line selection on status change
            if "selected_line" in st.session_state:
                del st.session_state["selected_line"]
    
    with col2:
        if st.button("❌ Reject", key=f"reject_{ticker}"):
            ticker_state["status"] = "rejected"
            ticker_state["reviewed_at"] = datetime.now().isoformat()
            if "selected_line" in st.session_state:
                del st.session_state["selected_line"]
    
    with col3:
        current_status = ticker_state.get("status", "pending")
        status_icon = {
            "approved": "✅ Approved",
            "rejected": "❌ Rejected", 
            "pending": "⏳ Pending Review"
        }.get(current_status, current_status)
        st.markdown(f"**Status:** {status_icon}")
        
        if ticker_state.get("reviewed_at"):
            st.caption(f"Reviewed: {ticker_state['reviewed_at'][:16]}")
    
    # Notes field
    notes = st.text_area(
        "Notes",
        value=ticker_state.get("notes", ""),
        key=f"notes_{ticker}",
        placeholder="Add review notes here...",
        height=80,
    )
    ticker_state["notes"] = notes
    
    return ticker_state


def render_summary_view(data: Dict[str, Any], review_state: Dict[str, Any]):
    """Render summary table of all tickers."""
    st.subheader("📊 Summary")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox(
            "Filter by Review Status",
            ["All", "Pending", "Approved", "Rejected"],
        )
    with col2:
        pipeline_filter = st.selectbox(
            "Filter by Pipeline Status", 
            ["All", "PASS", "FAIL", "UNKNOWN"],
        )
    
    rows = []
    for ticker, ticker_data in data["tickers"].items():
        metrics = ticker_data.get("metrics", {})
        review = review_state.get(ticker, {})
        
        # Apply filters
        review_status = review.get("status", "pending")
        pipeline_status = metrics.get("status", "UNKNOWN")
        
        if status_filter != "All" and review_status != status_filter.lower():
            continue
        if pipeline_filter != "All" and pipeline_status != pipeline_filter:
            continue
        
        extracted = metrics.get("calculated_sum") or metrics.get("extracted_total") or 0
        expected = metrics.get("expected_total")
        if expected and expected > 1e9:
            expected = expected / 1e6
        
        rows.append({
            "Ticker": ticker,
            "Pipeline": pipeline_status,
            "Review": review_status.title(),
            "Lines": len(ticker_data.get("lines", [])),
            "Extracted ($M)": f"{extracted:,.0f}" if extracted else "—",
            "Expected ($M)": f"{expected:,.0f}" if expected else "—",
            "Delta %": f"{metrics.get('delta_pct', 0):.1f}%" if metrics.get("delta_pct") is not None else "—",
            "Notes": (review.get("notes", "")[:30] + "...") if len(review.get("notes", "")) > 30 else review.get("notes", ""),
        })
    
    if not rows:
        st.info("No tickers match the current filters")
        return
    
    df = pd.DataFrame(rows)
    
    # Color code by review status
    def highlight_status(row):
        if row["Review"] == "Approved":
            return ["background-color: #1a4d1a"] * len(row)
        elif row["Review"] == "Rejected":
            return ["background-color: #4d1a1a"] * len(row)
        return [""] * len(row)
    
    styled_df = df.style.apply(highlight_status, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Summary stats
    st.markdown("---")
    
    all_tickers = list(data["tickers"].keys())
    approved = sum(1 for r in review_state.values() if r.get("status") == "approved")
    rejected = sum(1 for r in review_state.values() if r.get("status") == "rejected")
    pending = len(all_tickers) - approved - rejected
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", len(all_tickers))
    with col2:
        st.metric("Approved", approved)
    with col3:
        st.metric("Rejected", rejected)
    with col4:
        st.metric("Pending", pending)
    
    # Progress bar
    if all_tickers:
        progress = (approved + rejected) / len(all_tickers)
        st.progress(progress, text=f"Review Progress: {progress*100:.0f}%")


def render_ticker_detail(ticker: str, ticker_data: Dict[str, Any], review_state: Dict[str, Any]) -> Dict[str, Any]:
    """Render detailed view for a single ticker."""
    st.header(f"📄 {ticker}")
    
    metrics = ticker_data.get("metrics", {})
    lines = ticker_data.get("lines", [])
    
    # Metrics cards
    render_metrics_card(metrics)
    
    # Validation notes if any
    if metrics.get("validation_notes"):
        st.caption(f"ℹ️ {metrics['validation_notes']}")
    
    st.markdown("---")
    
    # Main content: 3 columns
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col1:
        st.subheader("Extracted Data")
        selected_line = render_extracted_data_interactive(lines)
    
    with col2:
        st.subheader("10-K Source")
        render_filing_viewer(ticker, ticker_data)
    
    with col3:
        st.subheader("Description")
        render_description_panel(lines, selected_line)
    
    # Review controls
    updated_state = render_review_controls(ticker, review_state)
    
    return updated_state


# --- Main App ---

def main():
    st.set_page_config(
        page_title="Revenue Extraction Validator",
        page_icon="📊",
        layout="wide",
    )
    
    st.title("📊 Revenue Extraction Validator")
    
    # Load data
    data = load_dashboard_data()
    review_state = load_review_state()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    tickers = list(data["tickers"].keys())
    
    view_mode = st.sidebar.radio(
        "View",
        ["Summary", "Detail"],
        horizontal=True,
    )
    
    if view_mode == "Detail":
        # Ticker selector with status indicators
        ticker_options = []
        for t in tickers:
            status = review_state.get(t, {}).get("status", "pending")
            icon = {"approved": "✅", "rejected": "❌", "pending": "⏳"}.get(status, "")
            ticker_options.append(f"{icon} {t}")
        
        # Handle index from session state
        default_idx = st.session_state.get("selected_idx", 0)
        if default_idx >= len(ticker_options):
            default_idx = 0
        
        selected_option = st.sidebar.selectbox(
            "Select Ticker",
            ticker_options,
            index=default_idx,
        )
        selected_ticker = selected_option.split(" ")[-1]
        
        # Update session state
        current_idx = tickers.index(selected_ticker)
        st.session_state["selected_idx"] = current_idx
        
        # Clear line selection when ticker changes
        if st.session_state.get("last_ticker") != selected_ticker:
            if "selected_line" in st.session_state:
                del st.session_state["selected_line"]
            st.session_state["last_ticker"] = selected_ticker
        
        # Navigation buttons
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("← Prev", disabled=current_idx == 0):
                st.session_state["selected_idx"] = current_idx - 1
                if "selected_line" in st.session_state:
                    del st.session_state["selected_line"]
                st.rerun()
        
        with col2:
            if st.button("Next →", disabled=current_idx >= len(tickers) - 1):
                st.session_state["selected_idx"] = current_idx + 1
                if "selected_line" in st.session_state:
                    del st.session_state["selected_line"]
                st.rerun()
        
        # Progress indicator
        st.sidebar.markdown("---")
        st.sidebar.caption(f"Ticker {current_idx + 1} of {len(tickers)}")
        
        # Quick actions
        st.sidebar.markdown("---")
        st.sidebar.subheader("Quick Actions")
        
        if st.sidebar.button("📥 Export Review State"):
            st.sidebar.download_button(
                "Download JSON",
                json.dumps(review_state, indent=2),
                file_name="review_state.json",
                mime="application/json",
            )
        
        # Render detail view
        ticker_data = data["tickers"][selected_ticker]
        updated_state = render_ticker_detail(selected_ticker, ticker_data, review_state)
        
        # Save state
        review_state[selected_ticker] = updated_state
        save_review_state(review_state)
    
    else:
        # Summary view
        render_summary_view(data, review_state)
        
        st.markdown("---")
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📥 Export Review State",
                json.dumps(review_state, indent=2),
                file_name="review_state.json",
                mime="application/json",
            )
        
        with col2:
            # Export to CSV
            export_rows = []
            for ticker, ticker_data in data["tickers"].items():
                review = review_state.get(ticker, {})
                metrics = ticker_data.get("metrics", {})
                export_rows.append({
                    "Ticker": ticker,
                    "Pipeline Status": metrics.get("status"),
                    "Review Status": review.get("status", "pending"),
                    "Extracted Total": metrics.get("calculated_sum"),
                    "Delta %": metrics.get("delta_pct"),
                    "Notes": review.get("notes", ""),
                    "Reviewed At": review.get("reviewed_at", ""),
                })
            export_df = pd.DataFrame(export_rows)
            
            st.download_button(
                "📊 Export to CSV",
                export_df.to_csv(index=False),
                file_name="review_export.csv",
                mime="text/csv",
            )
        
        with col3:
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()


if __name__ == "__main__":
    main()
