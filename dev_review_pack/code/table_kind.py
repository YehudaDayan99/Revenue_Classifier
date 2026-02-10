"""Deterministic table kind gating to filter out non-target tables."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from revseg.table_candidates import TableCandidate


_WS_RE = re.compile(r"\s+")


def _clean(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip())


# =====================================================================
# STRICT NEGATIVE PATTERNS - Only reject if table's CORE topic is these
# Applied to: caption_text + first 8 row labels + nearby_text
# =====================================================================
NEGATIVE_PATTERNS_STRICT: List[re.Pattern] = [
    # Tables specifically about deferred/unearned revenue schedules
    re.compile(r"\bunearned\s+revenue\b", re.IGNORECASE),
    re.compile(r"\bdeferred\s+revenue\b(?!\s+recognition)", re.IGNORECASE),  # Allow "deferred revenue recognition"
    re.compile(r"\bcontract\s+liabilities?\b", re.IGNORECASE),
    re.compile(r"\bremaining\s+performance\s+obligation", re.IGNORECASE),
    # Derivative/hedging instrument tables (not hedging gains in revenue tables)
    re.compile(r"\bderivative\s+instruments?\b", re.IGNORECASE),
    re.compile(r"\bhedging\s+instruments?\b", re.IGNORECASE),
    # Lease tables
    re.compile(r"\blease\s+(assets?|liabilities?|obligations?)\b", re.IGNORECASE),
    re.compile(r"\boperating\s+lease\s+(cost|liability|asset)\b", re.IGNORECASE),
    re.compile(r"\bfinance\s+lease\s+(cost|liability|asset)\b", re.IGNORECASE),
    # Tax tables
    re.compile(r"\bdeferred\s+tax\b", re.IGNORECASE),
    re.compile(r"\bunrecognized\s+tax\b", re.IGNORECASE),
]

# =====================================================================
# CONTEXT NEGATIVE PATTERNS - Only reject if these appear AND no positive signals
# Applied to: full text blob
# These are only applied when has_revenue_signal is False
# =====================================================================
NEGATIVE_PATTERNS_CONTEXT: List[re.Pattern] = [
    # Geography tables (when not part of a larger revenue disclosure)
    re.compile(r"\brevenue\s+by\s+geography\b", re.IGNORECASE),
    re.compile(r"\bgeographic\s+(regions?|areas?)\b", re.IGNORECASE),
    
    # --- P0 additions (Jan 2026) ---
    
    # Stock performance / shareholder return tables
    re.compile(r"\$\s*100\s+invested", re.IGNORECASE),
    re.compile(r"\bcumulative\s+total\s+return\b", re.IGNORECASE),
    re.compile(r"\bshareholder\s+return\b", re.IGNORECASE),
    re.compile(r"\b(stock|share)\s+(price\s+)?performance\b", re.IGNORECASE),
    re.compile(r"\bcomparison\s+of\s+\$?\s*100\b", re.IGNORECASE),
    re.compile(r"\b(?:s&p|nasdaq|russell|dow)\b", re.IGNORECASE),
    re.compile(r"\b(philadelphia|p\.?h\.?l\.?x\.?)\s*semiconductor\b", re.IGNORECASE),
    re.compile(r"\bindex\b.*\b(?:return|performance|value)\b", re.IGNORECASE),
    
    # Earnings / income statement-like tables (when not a revenue disaggregation)
    re.compile(r"\bnet\s+(income|earnings)\b", re.IGNORECASE),
    re.compile(r"\bearnings?\s+(before|after)\s+(?:income\s+)?tax\b", re.IGNORECASE),
    re.compile(r"\bincome\s+before\s+(?:income\s+)?tax\b", re.IGNORECASE),
    re.compile(r"\bconsolidated\s+statements?\s+of\s+(operations|income|earnings)\b", re.IGNORECASE),
    re.compile(r"\bsegment\s+(operating\s+)?(income|earnings|profit)\b", re.IGNORECASE),
    
    # Volume/throughput metrics (payment networks like MA, V)
    re.compile(r"\bgross\s+dollar\s+volume\b", re.IGNORECASE),
    re.compile(r"\bgdv\b", re.IGNORECASE),
    re.compile(r"\bpayment[s]?\s+volume\b", re.IGNORECASE),
    re.compile(r"\bprocessed\s+transactions?\b", re.IGNORECASE),
    re.compile(r"\btotal\s+volume\s+\(\$?b\)", re.IGNORECASE),
    re.compile(r"\bcards?\s+\([mb]\)", re.IGNORECASE),
]

# =====================================================================
# POSITIVE PATTERNS - If ANY match, allow the table
# Applied to: full text blob (overrides negatives)
# =====================================================================
POSITIVE_PATTERNS: List[re.Pattern] = [
    # Revenue disaggregation tables
    re.compile(r"\bdisaggregation\s+of\s+revenue\b", re.IGNORECASE),
    re.compile(r"\brevenue\s+classified\s+by\b", re.IGNORECASE),
    re.compile(r"\bnet\s+sales\s+by\s+(product|category|segment)\b", re.IGNORECASE),
    re.compile(r"\brevenue\s+by\s+(product|service|segment)\b", re.IGNORECASE),
    re.compile(r"\bsignificant\s+product\s+and\s+service\s+offerings\b", re.IGNORECASE),
    # Total revenue/sales indicators
    re.compile(r"\btotal\s+net\s+sales\b", re.IGNORECASE),
    re.compile(r"\btotal\s+revenues?\b", re.IGNORECASE),
    # Segment results tables (MSFT-style)
    re.compile(r"\bsegment\s+results\b", re.IGNORECASE),
    re.compile(r"\bresults\s+of\s+operations\s+by\s+segment\b", re.IGNORECASE),
    # Product category indicators
    re.compile(r"\biphone\b.*\bmac\b", re.IGNORECASE),  # AAPL
    re.compile(r"\bintelligent\s+cloud\b", re.IGNORECASE),  # MSFT
    re.compile(r"\bgoogle\s+services\b", re.IGNORECASE),  # GOOGL
    re.compile(r"\bgoogle\s+cloud\b", re.IGNORECASE),  # GOOGL
]


@dataclass(frozen=True)
class TableKindDecision:
    ok: bool
    reason: str
    positive_hit: str | None = None
    negative_hit: str | None = None


def _get_core_text(c: TableCandidate) -> str:
    """Get the core text that defines the table's topic (caption + row labels + nearby)."""
    parts: List[str] = []
    parts.append(str(getattr(c, "caption_text", "") or ""))
    parts.append(str(getattr(c, "heading_context", "") or ""))
    # First 8 row labels (enough to catch "unearned revenue" patterns)
    row_labels = (getattr(c, "row_label_preview", []) or [])[:8]
    parts.append(" ".join(row_labels))
    # Also include nearby text context
    parts.append(str(getattr(c, "nearby_text_context", "") or ""))
    return _clean(" ".join(parts))


def _get_full_text(c: TableCandidate) -> str:
    """Get all available text for the candidate."""
    parts: List[str] = []
    parts.append(str(getattr(c, "caption_text", "") or ""))
    parts.append(str(getattr(c, "heading_context", "") or ""))
    parts.append(str(getattr(c, "nearby_text_context", "") or ""))
    parts.append(str(getattr(c, "year_header_text", "") or ""))
    parts.append(" ".join(getattr(c, "row_label_preview", []) or []))
    # Preview grid text
    try:
        prev = getattr(c, "preview", []) or []
        parts.append(" ".join([" ".join(r) for r in prev[:10]]))
    except Exception:
        pass
    return _clean(" ".join(parts))


def tablekind_gate(
    c: TableCandidate,
    *,
    strict_negative_patterns: Iterable[re.Pattern] = NEGATIVE_PATTERNS_STRICT,
    context_negative_patterns: Iterable[re.Pattern] = NEGATIVE_PATTERNS_CONTEXT,
    positive_patterns: Iterable[re.Pattern] = POSITIVE_PATTERNS,
) -> TableKindDecision:
    """
    Deterministic gate: reject common non-target tables before LLM selection.
    
    Strategy:
    1. Check strict negative patterns FIRST (unearned/deferred/lease/etc always reject)
    2. Check positive patterns - allow if no strict negative hit
    3. Check context negatives on full text (only if no revenue signals)
    4. Default: allow
    """
    core_text = _get_core_text(c)
    full_text = _get_full_text(c)
    
    # Step 1: Check strict negatives FIRST - these always reject (liability tables)
    for p in strict_negative_patterns:
        if p.search(core_text):
            return TableKindDecision(ok=False, reason="negative_gate_strict", negative_hit=p.pattern)
    
    # Step 2: Check positive patterns - these rescue tables if no strict negative hit
    for p in positive_patterns:
        if p.search(full_text):
            return TableKindDecision(ok=True, reason="positive_allow", positive_hit=p.pattern)
    
    # Step 3: Check context negatives only if no revenue signals in full text
    has_revenue_signal = bool(re.search(r"\brevenue\b|\bnet\s+sales\b", full_text, re.IGNORECASE))
    if not has_revenue_signal:
        for p in context_negative_patterns:
            if p.search(full_text):
                return TableKindDecision(ok=False, reason="negative_gate_context", negative_hit=p.pattern)
    
    # Step 4: Default allow - rely on extraction validation to filter bad tables
    return TableKindDecision(ok=True, reason="default_allow")


# Keep backward-compatible function signature
def candidate_text_blob(c: TableCandidate) -> str:
    """Get all text from a candidate (for backward compatibility)."""
    return _get_full_text(c)
