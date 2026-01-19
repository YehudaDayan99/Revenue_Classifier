from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from revseg.table_candidates import TableCandidate


_WS_RE = re.compile(r"\s+")


def _clean(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip())


# Priority-0 negative gates (common iXBRL traps)
NEGATIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bunearned\b", re.IGNORECASE),
    re.compile(r"\bdeferred\b", re.IGNORECASE),
    re.compile(r"\bcontract\s+liabil", re.IGNORECASE),
    re.compile(r"\bremaining\s+performance\s+obligation", re.IGNORECASE),
    re.compile(r"\brpo\b", re.IGNORECASE),
    re.compile(r"\bhedging\b", re.IGNORECASE),
    re.compile(r"\bderivative", re.IGNORECASE),
    re.compile(r"\blease\b|\bleases\b", re.IGNORECASE),
    re.compile(r"\btax\b|\btaxes\b", re.IGNORECASE),
    # Geography signals (explicitly out-of-scope for CSV1)
    re.compile(r"\bgeograph", re.IGNORECASE),
    re.compile(r"\bemea\b|\bapac\b|\bamericas\b|\bcountry\b|\bregion\b", re.IGNORECASE),
]

# Short allowlist for target disclosures (high precision)
POSITIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brevenue\b.*\bclassified\b", re.IGNORECASE),
    re.compile(r"\bnet\s+sales\b.*\bby\b", re.IGNORECASE),
    re.compile(r"\brevenue\b.*\bby\b.*\bproduct\b", re.IGNORECASE),
    re.compile(r"\bsignificant\s+product\s+and\s+service\s+offerings\b", re.IGNORECASE),
    re.compile(r"\bdisaggregation\b.*\brevenue\b", re.IGNORECASE),
]


@dataclass(frozen=True)
class TableKindDecision:
    ok: bool
    reason: str
    positive_hit: str | None = None
    negative_hit: str | None = None


def candidate_text_blob(c: TableCandidate) -> str:
    parts: list[str] = []
    parts.append(str(getattr(c, "caption_text", "") or ""))
    parts.append(str(getattr(c, "heading_context", "") or ""))
    parts.append(str(getattr(c, "nearby_text_context", "") or ""))
    parts.append(str(getattr(c, "year_header_text", "") or ""))
    parts.append(" ".join(getattr(c, "row_label_preview", []) or []))
    # preview grid text is high-signal for MSFT (product/service labels)
    try:
        prev = getattr(c, "preview", []) or []
        parts.append(" ".join([" ".join(r) for r in prev[:10]]))
    except Exception:
        pass
    return _clean(" ".join(parts))


def tablekind_gate(
    c: TableCandidate,
    *,
    negative_patterns: Iterable[re.Pattern] = NEGATIVE_PATTERNS,
    positive_patterns: Iterable[re.Pattern] = POSITIVE_PATTERNS,
) -> TableKindDecision:
    """Deterministic gate: reject common non-target tables before LLM selection."""
    blob = candidate_text_blob(c)
    for p in negative_patterns:
        m = p.search(blob)
        if m:
            return TableKindDecision(ok=False, reason="negative_gate", negative_hit=p.pattern)
    for p in positive_patterns:
        m = p.search(blob)
        if m:
            return TableKindDecision(ok=True, reason="positive_allow", positive_hit=p.pattern)
    # If no positive hit, still allow (we'll rely on overlap+validation), but mark as weak.
    return TableKindDecision(ok=True, reason="no_positive_hit")

