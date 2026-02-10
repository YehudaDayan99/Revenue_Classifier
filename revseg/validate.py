from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests


class ValidationError(RuntimeError):
    pass


def _sec_user_agent() -> str:
    ua = os.getenv("SEC_USER_AGENT")
    if not ua or "@" not in ua:
        raise ValidationError(
            "SEC_USER_AGENT env var must be set and include contact info (e.g., email)."
        )
    return ua


def _sec_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": _sec_user_agent(),
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "application/json",
            "Connection": "keep-alive",
        }
    )
    return s


@dataclass(frozen=True)
class RevenueValidation:
    ok: bool
    total_revenue_usd: Optional[int]
    sum_segments_usd: int
    abs_delta_usd: Optional[int]
    pct_delta: Optional[float]
    notes: str


SEC_COMPANYFACTS_TMPL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"


def fetch_companyfacts_total_revenue_usd(
    cik: int,
    fiscal_year: int,
    *,
    min_interval_s: float = 0.2,
    timeout_s: int = 30,
) -> Optional[int]:
    """Fetch total revenue (USD) for a given fiscal year from SEC companyfacts.

    Returns None if unavailable.
    """
    cik10 = f"{cik:010d}"
    url = SEC_COMPANYFACTS_TMPL.format(cik10=cik10)
    s = _sec_session()
    time.sleep(min_interval_s)
    r = s.get(url, timeout=timeout_s)
    if r.status_code != 200:
        return None
    data = r.json()

    facts = (data.get("facts") or {}).get("us-gaap") or {}
    
    # P0.1 FIX: Multiple XBRL tag fallbacks for total revenue (order matters)
    REVENUE_TAGS = [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        "NetRevenues",
    ]
    
    rev = None
    for tag in REVENUE_TAGS:
        rev = facts.get(tag)
        if rev:
            break
    
    if not rev:
        return None
    units = (rev.get("units") or {}).get("USD") or []

    # Prefer FY facts where fy matches and fp is FY
    candidates = [
        x
        for x in units
        if str(x.get("fp", "")).upper() == "FY"
        and int(x.get("fy") or 0) == int(fiscal_year)
        and isinstance(x.get("val"), (int, float))
    ]
    if not candidates:
        return None
    # Take the latest filed for the year
    candidates.sort(key=lambda x: (x.get("filed") or "", x.get("end") or ""), reverse=True)
    val = candidates[0]["val"]
    return int(round(float(val)))


def validate_segment_table(
    *,
    segment_revenues_usd: Dict[str, int],
    total_revenue_usd: Optional[int],
    tolerance_pct: float = 0.02,
) -> RevenueValidation:
    sum_segments = int(sum(int(v) for v in segment_revenues_usd.values() if v is not None))
    if total_revenue_usd is None:
        return RevenueValidation(
            ok=False,
            total_revenue_usd=None,
            sum_segments_usd=sum_segments,
            abs_delta_usd=None,
            pct_delta=None,
            notes="No total revenue reference available; cannot validate segments-to-total.",
        )

    abs_delta = int(abs(sum_segments - int(total_revenue_usd)))
    pct_delta = abs_delta / max(1, int(total_revenue_usd))
    ok = pct_delta <= tolerance_pct
    notes = "OK" if ok else f"Delta {pct_delta:.3%} exceeds tolerance {tolerance_pct:.2%}"
    return RevenueValidation(
        ok=ok,
        total_revenue_usd=int(total_revenue_usd),
        sum_segments_usd=sum_segments,
        abs_delta_usd=abs_delta,
        pct_delta=float(pct_delta),
        notes=notes,
    )

