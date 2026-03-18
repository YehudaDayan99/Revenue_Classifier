#!/usr/bin/env python3
"""
Revenue Extraction Validator Dashboard

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
from typing import Dict, Any

# --- Configuration ---
DATA_FILE = "dashboard_data.json"
STATE_FILE = "review_state.json"

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
            st.metric("Expected Total", f"${expected/1e6:,.0f}M")
        else:
            st.metric("Expected Total", "—")
    
    with col4:
        delta = metrics.get("delta_pct")
        if delta is not None:
            color = "normal" if abs(delta) < 2 else "inverse"
            st.metric("Delta", f"{delta:.2f}%", delta_color=color)
        else:
            st.metric("Delta", "—")


def render_extracted_data(lines: list):
    """Render extracted line items as a table."""
    if not lines:
        st.warning("No line items extracted")
        return
    
    df = pd.DataFrame(lines)
    
    # Format revenue column
    if "revenue" in df.columns:
        df["revenue_fmt"] = df["revenue"].apply(
            lambda x: f"${x:,.0f}M" if pd.notna(x) else "—"
        )
    
    # Display table
    display_cols = ["line", "revenue_fmt", "segment"]
    display_cols = [c for c in display_cols if c in df.columns]
    
    st.dataframe(
        df[display_cols].rename(columns={
            "line": "Revenue Line",
            "revenue_fmt": "Revenue ($M)",
            "segment": "Segment",
        }),
        use_container_width=True,
        hide_index=True,
    )
    
    # Total row
    total = sum(x for x in df["revenue"] if pd.notna(x))
    st.markdown(f"**Total: ${total:,.0f}M**")


def render_descriptions(lines: list):
    """Render description quality section."""
    if not lines:
        return
    
    has_descriptions = any(line.get("description") for line in lines)
    
    if has_descriptions:
        with st.expander("📝 Descriptions", expanded=False):
            for line in lines:
                desc = line.get("description", "")
                name = line.get("line", "Unknown")
                if desc:
                    st.markdown(f"**{name}:** {desc[:500]}{'...' if len(desc) > 500 else ''}")
                else:
                    st.markdown(f"**{name}:** *No description*")
    else:
        st.caption("⚠️ No descriptions extracted")


def render_original_table(html: str):
    """Render original 10-K table."""
    if not html or html.startswith("<p><em>"):
        st.info("Original table not available in artifacts")
        return
    
    # Wrap in scrollable container with styling
    styled_html = f"""
    <div style="
        max-height: 400px; 
        overflow-y: auto; 
        border: 1px solid #ddd; 
        padding: 10px;
        background: white;
        font-size: 12px;
    ">
        {html}
    </div>
    """
    st.markdown(styled_html, unsafe_allow_html=True)


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
    
    with col2:
        if st.button("❌ Reject", key=f"reject_{ticker}"):
            ticker_state["status"] = "rejected"
            ticker_state["reviewed_at"] = datetime.now().isoformat()
    
    with col3:
        current_status = ticker_state.get("status", "pending")
        status_icon = {
            "approved": "✅ Approved",
            "rejected": "❌ Rejected", 
            "pending": "⏳ Pending Review"
        }.get(current_status, current_status)
        st.markdown(f"**Status:** {status_icon}")
    
    # Notes field
    notes = st.text_area(
        "Notes",
        value=ticker_state.get("notes", ""),
        key=f"notes_{ticker}",
        placeholder="Add review notes here...",
    )
    ticker_state["notes"] = notes
    
    return ticker_state


def render_summary_view(data: Dict[str, Any], review_state: Dict[str, Any]):
    """Render summary table of all tickers."""
    st.subheader("📊 Summary")
    
    rows = []
    for ticker, ticker_data in data["tickers"].items():
        metrics = ticker_data.get("metrics", {})
        review = review_state.get(ticker, {})
        
        extracted = metrics.get("calculated_sum") or metrics.get("extracted_total") or 0
        expected = (metrics.get("expected_total") or 0) / 1e6 if metrics.get("expected_total") else None
        
        rows.append({
            "Ticker": ticker,
            "Pipeline": metrics.get("status", "—"),
            "Review": review.get("status", "pending").title(),
            "Lines": len(ticker_data.get("lines", [])),
            "Extracted ($M)": f"{extracted:,.0f}" if extracted else "—",
            "Expected ($M)": f"{expected:,.0f}" if expected else "—",
            "Delta %": f"{metrics.get('delta_pct', 0):.1f}%" if metrics.get("delta_pct") is not None else "—",
            "Notes": (review.get("notes", "")[:30] + "...") if len(review.get("notes", "")) > 30 else review.get("notes", ""),
        })
    
    df = pd.DataFrame(rows)
    
    # Color code by review status
    def highlight_status(row):
        if row["Review"] == "Approved":
            return ["background-color: #d4edda"] * len(row)
        elif row["Review"] == "Rejected":
            return ["background-color: #f8d7da"] * len(row)
        return [""] * len(row)
    
    styled_df = df.style.apply(highlight_status, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    approved = sum(1 for r in review_state.values() if r.get("status") == "approved")
    rejected = sum(1 for r in review_state.values() if r.get("status") == "rejected")
    pending = len(data["tickers"]) - approved - rejected
    
    with col1:
        st.metric("Total", len(data["tickers"]))
    with col2:
        st.metric("Approved", approved)
    with col3:
        st.metric("Rejected", rejected)
    with col4:
        st.metric("Pending", pending)


def render_ticker_detail(ticker: str, ticker_data: Dict[str, Any], review_state: Dict[str, Any]) -> Dict[str, Any]:
    """Render detailed view for a single ticker."""
    st.header(f"📄 {ticker}")
    
    metrics = ticker_data.get("metrics", {})
    lines = ticker_data.get("lines", [])
    html_table = ticker_data.get("html_table", "")
    
    # Metrics cards
    render_metrics_card(metrics)
    
    # Validation notes if any
    if metrics.get("validation_notes"):
        st.caption(f"ℹ️ {metrics['validation_notes']}")
    
    st.markdown("---")
    
    # Side-by-side view
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Extracted Data")
        render_extracted_data(lines)
        render_descriptions(lines)
    
    with col2:
        st.subheader("Original 10-K Table")
        render_original_table(html_table)
    
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
        
        selected_option = st.sidebar.selectbox(
            "Select Ticker",
            ticker_options,
            index=0,
        )
        selected_ticker = selected_option.split(" ")[-1]  # Extract ticker from "✅ NVDA"
        
        # Navigation buttons
        col1, col2 = st.sidebar.columns(2)
        current_idx = tickers.index(selected_ticker)
        
        with col1:
            if st.button("← Prev") and current_idx > 0:
                st.session_state["selected_idx"] = current_idx - 1
                st.rerun()
        
        with col2:
            if st.button("Next →") and current_idx < len(tickers) - 1:
                st.session_state["selected_idx"] = current_idx + 1
                st.rerun()
        
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
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Review State"):
                st.download_button(
                    "Download JSON",
                    json.dumps(review_state, indent=2),
                    file_name="review_state.json",
                    mime="application/json",
                )
        
        with col2:
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()


if __name__ == "__main__":
    main()
