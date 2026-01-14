from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TMPL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"


class SecEdgarError(RuntimeError):
    pass


@dataclass(frozen=True)
class FilingRef:
    ticker: str
    cik: int
    form: str
    accession_number: str
    filing_date: str
    primary_document: str

    @property
    def cik_no_pad(self) -> str:
        return str(self.cik)

    @property
    def cik10(self) -> str:
        return f"{self.cik:010d}"

    @property
    def accession_no_dashes(self) -> str:
        return self.accession_number.replace("-", "")

    @property
    def sec_doc_url(self) -> str:
        return (
            f"{SEC_ARCHIVES_BASE}/"
            f"{self.cik_no_pad}/"
            f"{self.accession_no_dashes}/"
            f"{self.primary_document}"
        )

    @property
    def sec_index_url(self) -> str:
        return (
            f"{SEC_ARCHIVES_BASE}/"
            f"{self.cik_no_pad}/"
            f"{self.accession_no_dashes}/index.json"
        )


def _sec_user_agent() -> str:
    ua = os.getenv("SEC_USER_AGENT")
    if not ua or "@" not in ua:
        raise SecEdgarError(
            "SEC_USER_AGENT env var must be set and include contact info (e.g., email). "
            'Example: SEC_USER_AGENT="RevenueSegBot/0.1 (your.email@domain.com)"'
        )
    return ua


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": _sec_user_agent(),
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Connection": "keep-alive",
        }
    )
    return s


def _sleep_rate_limit(min_interval_s: float, last_call_ts: List[float]) -> None:
    """Ensure at least min_interval_s seconds between calls."""
    now = time.time()
    if last_call_ts and (now - last_call_ts[0]) < min_interval_s:
        time.sleep(min_interval_s - (now - last_call_ts[0]))
    if last_call_ts:
        last_call_ts[0] = time.time()
    else:
        last_call_ts.append(time.time())


def fetch_ticker_cik_map(
    cache_path: Optional[Path] = None, *, min_interval_s: float = 0.2
) -> Dict[str, int]:
    """Fetch SEC ticker->CIK mapping and normalize tickers to uppercase.

    If cache_path exists, reads from cache. If provided and missing, writes cache.
    """
    if cache_path and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {k.upper(): int(v) for k, v in data.items()}

    s = _session()
    last_ts: List[float] = []
    _sleep_rate_limit(min_interval_s, last_ts)

    r = s.get(SEC_TICKER_CIK_URL, timeout=30)
    if r.status_code != 200:
        raise SecEdgarError(f"Failed to fetch ticker->CIK map: HTTP {r.status_code}")

    raw = r.json()
    out: Dict[str, int] = {}
    for _, rec in raw.items():
        t = str(rec.get("ticker", "")).upper().strip()
        cik = rec.get("cik_str")
        if t and cik is not None:
            out[t] = int(cik)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)

    return out


def get_company_submissions(
    cik: int, *, min_interval_s: float = 0.2, cache_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Fetch company submissions JSON for a given CIK."""
    cik10 = f"{cik:010d}"
    cache_path = (cache_dir / f"CIK{cik10}.json") if cache_dir else None

    if cache_path and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    s = _session()
    last_ts: List[float] = []
    _sleep_rate_limit(min_interval_s, last_ts)

    url = SEC_SUBMISSIONS_URL_TMPL.format(cik10=cik10)
    r = s.get(url, timeout=30)
    if r.status_code != 200:
        raise SecEdgarError(f"Failed to fetch submissions for CIK {cik10}: HTTP {r.status_code}")

    data = r.json()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    return data


def select_latest_10k(
    ticker: str,
    cik: int,
    submissions: Dict[str, Any],
    *,
    include_amendments: bool = False,
) -> FilingRef:
    """Select the most recent 10-K (or optionally 10-K/A) from submissions."""
    recent = submissions.get("filings", {}).get("recent", {})
    forms: List[str] = recent.get("form", [])
    accession: List[str] = recent.get("accessionNumber", [])
    filing_dates: List[str] = recent.get("filingDate", [])
    primary_docs: List[str] = recent.get("primaryDocument", [])

    if not (len(forms) == len(accession) == len(filing_dates) == len(primary_docs)):
        raise SecEdgarError(f"Malformed submissions JSON for {ticker}/{cik}")

    allowed = {"10-K"} if not include_amendments else {"10-K", "10-K/A"}

    for form, acc, fdate, pdoc in zip(forms, accession, filing_dates, primary_docs):
        if form in allowed:
            return FilingRef(
                ticker=ticker.upper(),
                cik=cik,
                form=form,
                accession_number=acc,
                filing_date=fdate,
                primary_document=pdoc,
            )

    raise SecEdgarError(f"No 10-K found for {ticker} (include_amendments={include_amendments})")


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def download_latest_10k(
    ticker: str,
    out_dir: Path,
    *,
    include_amendments: bool = False,
    cache_dir: Optional[Path] = None,
    min_interval_s: float = 0.2,
) -> Path:
    """Download latest 10-K primary document and metadata. Returns output folder path."""
    ticker = ticker.upper().strip()
    out_dir = out_dir.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve() if cache_dir else None

    print(f"  {ticker}: Fetching ticker->CIK map...", flush=True)
    ticker_cik = fetch_ticker_cik_map(
        cache_path=(cache_dir / "ticker_cik.json") if cache_dir else None,
        min_interval_s=min_interval_s,
    )
    if ticker not in ticker_cik:
        raise SecEdgarError(f"Ticker not found in SEC mapping: {ticker}")

    cik = ticker_cik[ticker]
    print(f"  {ticker}: Fetching submissions for CIK {cik}...", flush=True)
    subs = get_company_submissions(cik, min_interval_s=min_interval_s, cache_dir=cache_dir)
    filing = select_latest_10k(ticker, cik, subs, include_amendments=include_amendments)
    print(f"  {ticker}: Found {filing.form} filed on {filing.filing_date}", flush=True)

    folder = out_dir / ticker / f"{filing.filing_date}_{filing.accession_no_dashes}"
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "filing_ref.json").write_text(json.dumps(filing.__dict__, indent=2), encoding="utf-8")
    (folder / "submission.json").write_text(json.dumps(subs, indent=2), encoding="utf-8")

    s = _session()
    last_ts: List[float] = []

    # Filing index (non-fatal)
    print(f"  {ticker}: Downloading filing index...", flush=True)
    _sleep_rate_limit(min_interval_s, last_ts)
    idx = s.get(filing.sec_index_url, timeout=60)
    if idx.status_code == 200:
        (folder / "filing_index.json").write_text(idx.text, encoding="utf-8")
    else:
        (folder / "filing_index_error.txt").write_text(
            f"HTTP {idx.status_code} fetching {filing.sec_index_url}\n{idx.text[:2000]}",
            encoding="utf-8",
        )

    # Primary document (fatal)
    print(f"  {ticker}: Downloading primary document ({filing.primary_document})...", flush=True)
    _sleep_rate_limit(min_interval_s, last_ts)
    doc = s.get(filing.sec_doc_url, timeout=120)
    if doc.status_code != 200:
        raise SecEdgarError(f"Failed to download primary document: HTTP {doc.status_code} ({filing.sec_doc_url})")

    pdoc_name = _sanitize_filename(filing.primary_document)
    (folder / pdoc_name).write_bytes(doc.content)

    suffix = Path(pdoc_name).suffix.lower()
    conv_name = "primary_document.html" if suffix in {".htm", ".html"} else f"primary_document{suffix or '.bin'}"
    if (folder / conv_name).name != pdoc_name:
        (folder / conv_name).write_bytes(doc.content)

    return folder


def download_many_latest_10k(
    tickers: Iterable[str],
    out_dir: Path,
    *,
    include_amendments: bool = False,
    cache_dir: Optional[Path] = None,
    min_interval_s: float = 0.2,
) -> Dict[str, Tuple[bool, str]]:
    """Download latest 10-K for multiple tickers.

    Returns: ticker -> (success, folder_path_or_error_message)
    """
    ticker_list = list(tickers)
    total = len(ticker_list)
    results: Dict[str, Tuple[bool, str]] = {}
    for idx, t in enumerate(ticker_list, 1):
        t_up = str(t).upper().strip()
        print(f"[{idx}/{total}] Processing {t_up}...", flush=True)
        try:
            folder = download_latest_10k(
                t_up,
                out_dir,
                include_amendments=include_amendments,
                cache_dir=cache_dir,
                min_interval_s=min_interval_s,
            )
            print(f"[{idx}/{total}] ✓ {t_up} completed: {folder}", flush=True)
            results[t_up] = (True, str(folder))
        except Exception as e:
            print(f"[{idx}/{total}] ✗ {t_up} failed: {type(e).__name__}: {e}", flush=True)
            results[t_up] = (False, f"{type(e).__name__}: {e}")
    return results
