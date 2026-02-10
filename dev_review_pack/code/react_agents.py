from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup

from revseg.llm_client import OpenAIChatClient
from revseg.table_candidates import TableCandidate, extract_table_grid_normalized, _get_cached_soup


_WS_RE = re.compile(r"\s+")
_MONEY_CLEAN_RE = re.compile(r"[^0-9.\-]")
_ITEM8_RE = re.compile(
    r"\bitem\s*8\b|\bfinancial statements\b|\bnotes to (?:the )?financial statements\b",
    re.IGNORECASE,
)
_ITEM7_RE = re.compile(r"\bitem\s*7\b|\bmanagement['']s discussion\b|\bmd&a\b", re.IGNORECASE)
_ITEM1_RE = re.compile(r"\bitem\s*1[.\s]+business\b|\bitem\s*1\b(?![\d])", re.IGNORECASE)
_SEGMENT_NOTE_RE = re.compile(r"\bsegment(s)?\b|\breportable segment(s)?\b", re.IGNORECASE)

# Revenue line label expansions for better matching
_LABEL_EXPANSIONS: Dict[str, List[str]] = {
    "advertising": ["advertising", "ad revenue", "ads", "advertising services"],
    "subscription": ["subscription", "subscriptions", "subscriber"],
    "services": ["services", "service revenue", "service offerings"],
    "cloud": ["cloud", "cloud services", "cloud computing"],
    "gaming": ["gaming", "game", "games"],
    "compute": ["compute", "data center", "computing"],
    "networking": ["networking", "network"],
    "professional visualization": ["professional visualization", "visualization", "professional"],
    "automotive": ["automotive", "auto", "vehicle"],
    "online stores": ["online stores", "online store", "e-commerce"],
    "physical stores": ["physical stores", "physical store", "retail stores"],
    "third-party seller": ["third-party seller", "third party seller", "marketplace", "3p seller"],
    "aws": ["aws", "amazon web services", "cloud services"],
}

# Patterns for identifying numeric/currency cells
_CURRENCY_NUM_RE = re.compile(r"^[\s$€£¥(),.0-9\-]+$")
_PURE_NUMBER_RE = re.compile(r"^[\s0-9,.\-()]+$")


def choose_item_col(
    grid: List[List[str]],
    header_rows: Optional[List[int]] = None,
    llm_proposed_col: Optional[int] = None,
) -> Tuple[int, str]:
    """
    Deterministically select the best label/item column in a table grid.
    
    Ranks columns by:
    1. numeric_ratio (lower = better for label column)
    2. alpha_ratio (higher = better)
    3. uniqueness (labels tend to be diverse)
    4. mean string length (labels typically > 2 chars)
    
    Args:
        grid: Table grid as list of lists
        header_rows: Row indices to skip (headers)
        llm_proposed_col: Column proposed by LLM (will validate)
    
    Returns:
        (best_col_index, reason_string)
    """
    if not grid:
        return (0, "empty grid, defaulting to 0")
    
    header_rows = set(header_rows or [])
    
    # Compute metrics for each column
    col_scores = []
    n_cols = max(len(row) for row in grid) if grid else 0
    
    for col_idx in range(n_cols):
        cells = []
        for row_idx, row in enumerate(grid):
            if row_idx in header_rows:
                continue
            if col_idx < len(row):
                cell = str(row[col_idx]).strip()
                if cell:
                    cells.append(cell)
        
        if not cells:
            col_scores.append({
                "col": col_idx,
                "numeric_ratio": 1.0,
                "alpha_ratio": 0.0,
                "uniqueness": 0.0,
                "mean_len": 0.0,
                "score": -999,
            })
            continue
        
        # Metric 1: numeric_ratio (lower = better for label column)
        numeric_count = sum(1 for c in cells if _CURRENCY_NUM_RE.match(c))
        numeric_ratio = numeric_count / len(cells)
        
        # Metric 2: alpha_ratio (contains letters, higher = better)
        alpha_count = sum(1 for c in cells if any(ch.isalpha() for ch in c))
        alpha_ratio = alpha_count / len(cells)
        
        # Metric 3: uniqueness (unique values / total, higher = better for labels)
        unique_values = len(set(c.lower() for c in cells))
        uniqueness = unique_values / len(cells) if cells else 0
        
        # Metric 4: mean string length (labels tend to be longer than numbers)
        mean_len = sum(len(c) for c in cells) / len(cells) if cells else 0
        
        # Combined score: 
        # - Penalize high numeric_ratio heavily (-5x weight)
        # - Reward alpha_ratio (+3x weight)
        # - Reward uniqueness slightly (+1x weight)
        # - Reward reasonable length (+0.05x weight)
        score = (-5 * numeric_ratio) + (3 * alpha_ratio) + (1 * uniqueness) + (0.05 * mean_len)
        
        col_scores.append({
            "col": col_idx,
            "numeric_ratio": round(numeric_ratio, 3),
            "alpha_ratio": round(alpha_ratio, 3),
            "uniqueness": round(uniqueness, 3),
            "mean_len": round(mean_len, 1),
            "score": round(score, 3),
        })
    
    if not col_scores:
        return (0, "no columns found, defaulting to 0")
    
    # Sort by score descending
    col_scores.sort(key=lambda x: x["score"], reverse=True)
    heuristic_best = col_scores[0]
    
    # Validate LLM's proposed column
    if llm_proposed_col is not None and 0 <= llm_proposed_col < len(col_scores):
        llm_col_data = next((c for c in col_scores if c["col"] == llm_proposed_col), None)
        
        if llm_col_data:
            # Accept LLM choice if:
            # 1. numeric_ratio < 0.5 (not mostly numbers)
            # 2. alpha_ratio > 0.3 (has some text)
            if llm_col_data["numeric_ratio"] < 0.5 and llm_col_data["alpha_ratio"] > 0.3:
                return (llm_proposed_col, f"LLM choice validated (num={llm_col_data['numeric_ratio']}, alpha={llm_col_data['alpha_ratio']})")
            else:
                # LLM choice failed validation, override with heuristic best
                return (
                    heuristic_best["col"],
                    f"LLM col {llm_proposed_col} OVERRIDDEN (num={llm_col_data['numeric_ratio']:.2f}, alpha={llm_col_data['alpha_ratio']:.2f}) → col {heuristic_best['col']} (score={heuristic_best['score']:.2f})"
                )
    
    # No LLM proposal or invalid index, use heuristic
    return (heuristic_best["col"], f"heuristic best (score={heuristic_best['score']:.2f})")


def validate_extracted_labels(labels: List[str], threshold: float = 0.5) -> Tuple[bool, str]:
    """
    Validate that extracted revenue line labels are not mostly numeric/currency.
    
    Args:
        labels: List of extracted revenue line labels
        threshold: Maximum allowed ratio of numeric labels (default 50%)
    
    Returns:
        (is_valid, reason)
    """
    if not labels:
        return (False, "no labels extracted")
    
    numeric_count = sum(1 for label in labels if _CURRENCY_NUM_RE.match(label.strip()))
    numeric_ratio = numeric_count / len(labels)
    
    if numeric_ratio > threshold:
        return (False, f"FAIL: {numeric_ratio*100:.0f}% of labels are numeric/currency (threshold: {threshold*100:.0f}%)")
    
    return (True, f"OK: {numeric_ratio*100:.0f}% numeric labels")


def _clean(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip())


def _extract_section(text: str, section_pattern: re.Pattern, max_chars: int = 50000) -> str:
    """
    Extract a specific section from the filing text.
    
    Finds the section start using the pattern and extracts up to max_chars,
    stopping at the next major section boundary (Item X).
    """
    if not text:
        return ""
    
    match = section_pattern.search(text)
    if not match:
        return ""
    
    start_idx = match.start()
    
    # Find the next section boundary (Item followed by number)
    next_item_pattern = re.compile(r"\bitem\s+\d+", re.IGNORECASE)
    search_start = start_idx + len(match.group())
    
    end_idx = start_idx + max_chars
    for next_match in next_item_pattern.finditer(text, search_start):
        if next_match.start() > start_idx + 500:  # Must be at least 500 chars after start
            end_idx = min(end_idx, next_match.start())
            break
    
    return text[start_idx:end_idx]


def _expand_search_terms(label: str) -> List[str]:
    """
    Expand a revenue line label into multiple search terms for better matching.
    Returns the original label plus any known variations.
    """
    terms = [label.lower()]
    label_low = label.lower().strip()
    
    # Check for known expansions
    for key, expansions in _LABEL_EXPANSIONS.items():
        if key in label_low or label_low in key:
            for exp in expansions:
                if exp.lower() not in terms:
                    terms.append(exp.lower())
    
    # Also add individual significant words (3+ chars)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', label)
    for word in words:
        word_low = word.lower()
        if word_low not in terms and word_low not in ('the', 'and', 'for', 'from', 'other'):
            terms.append(word_low)
    
    return terms


# Pattern to extract footnote markers from labels like "Online stores (1)"
_FOOTNOTE_MARKER_RE = re.compile(r'\((\d+)\)\s*$')


# ========================================================================
# Fix A: Accounting/driver sentence filter
# ========================================================================
_ACCOUNTING_DENY_PATTERNS = [
    # Revenue recognition / accounting language
    r"\bperformance obligation\b",
    r"\bstand-alone selling price\b",
    r"\bssp\b",
    r"\ballocated\b",
    r"\brecognized\b",
    r"\bdeferred\b",
    r"\bamortization\b",
    r"\bcontract liabilit",
    r"\bunearned\b",
    r"\bASC\s+\d",
    r"\bGAAP\b",
    r"\brevenue recognition\b",
    # Reporting mechanics
    r"\brecord revenue\b",
    r"\bgross\b.*\bnet of\b",
    r"\bconsolidated\b",
    r"\breclassification\b",
    # Table/reference mechanics
    r"\bsee note\b",
    r"\brefer to\b",
    r"\bin the table\b",
    r"\bas shown\b",
    # Performance drivers (MD&A-style) - we want product definitions, not performance
    r"\bincreased due to\b",
    r"\bdecreased due to\b",
    r"\bprimarily due to\b",
    r"\bhigher sales of\b",
    r"\blower sales of\b",
    r"\bdriven by\b",
    r"\bprimarily driven\b",
    r"\bcompared to\b.*\bprior\b",
    r"\byear over year\b",
    r"\bfrom a year ago\b",
    # P1.4: Additional patterns from Dev review
    r"\bgrowth depends\b",
    r"\bwe expect\b",
    r"\bwe anticipate\b",
    r"\bwe believe\b",
    r"\bwe continue\b",
    r"\bfiscal\s+\d{4}\b",  # Fiscal year references
    r"\bfor the year\b",
    r"\bduring the year\b",
    r"\bquarter\b",
]
_ACCOUNTING_DENY_RE = re.compile("|".join(_ACCOUNTING_DENY_PATTERNS), re.IGNORECASE)


def strip_accounting_sentences(text: str) -> str:
    """
    Remove sentences containing accounting/regulatory or performance driver language.
    
    This ensures descriptions focus on WHAT the product/service IS,
    not HOW it performed or accounting treatment.
    
    P1.4: Also strips ToC/page artifacts.
    """
    if not text:
        return ""
    
    # P1.4: Strip ToC/page artifacts before processing
    text = _strip_toc_and_page_noise(text)
    
    # Split into sentences (handling abbreviations like "Inc.")
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    clean_sentences = []
    
    for sentence in sentences:
        if not _ACCOUNTING_DENY_RE.search(sentence):
            clean_sentences.append(sentence)
    
    result = " ".join(clean_sentences).strip()
    return result


# P1.4: ToC and page noise patterns
_TOC_NOISE_PATTERNS = [
    r"Table\s+of\s+Contents",
    r"^\d+\s*$",  # Standalone page numbers
    r"Page\s+\d+",
    r"F-\d+",  # Financial statement page refs
    r"Item\s+\d+[A-Z]?\.",  # SEC item references
    r"PART\s+[IVX]+",
    r"^[\d\s]+Table of Contents",
]
_TOC_NOISE_RE = re.compile("|".join(_TOC_NOISE_PATTERNS), re.IGNORECASE | re.MULTILINE)


def _strip_toc_and_page_noise(text: str) -> str:
    """Remove Table of Contents artifacts and page number noise."""
    if not text:
        return ""
    # Remove ToC patterns
    text = _TOC_NOISE_RE.sub(" ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ========================================================================
# P0: Heading-based definition extraction (AAPL Services fix)
# ========================================================================
# Phase 5: Table-header contamination detection
TABLE_HEADER_MARKERS = [
    "Year Ended",
    "December 31",
    "June 30", 
    "September 30",
    "(in millions)",
    "(in thousands)",
    "% change",
    "% Change",
    "Total Revenue",
    "Total Net Sales",
    "Year-Over-Year",
    "Fiscal Year",
]

def _is_table_header_contaminated(text: str) -> bool:
    """
    Check if extracted text is contaminated with table header/structure content.
    
    This happens when heading-based extraction captures table structure instead
    of actual product/service definitions (e.g., META Reality Labs).
    
    Returns True if the text appears to be table structure, not a definition.
    """
    if not text:
        return False
    
    # Check for table header markers
    marker_count = sum(1 for marker in TABLE_HEADER_MARKERS if marker.lower() in text.lower())
    
    # If 2+ markers found, likely table header
    if marker_count >= 2:
        return True
    
    # Check for column-like patterns (e.g., "2024 2023 2022")
    year_pattern = re.compile(r'\b20\d{2}\s+20\d{2}\s+20\d{2}\b')
    if year_pattern.search(text):
        return True
    
    # Check for dollar amounts at the start (revenue table data)
    dollar_start = re.compile(r'^\s*\$?\s*\d{1,3}(?:,\d{3})+')
    if dollar_start.match(text):
        return True
    
    # Check for percentage patterns that indicate table data
    pct_pattern = re.compile(r'\b\d+\s*%\s*(change|increase|decrease)\b', re.IGNORECASE)
    if pct_pattern.search(text[:200]):  # Check first 200 chars
        return True
    
    return False


# Phase 5: Note 2 Revenue paragraph extraction (for META-style filings)
# Patterns for prose-style definitions in Note 2 - Revenue
NOTE2_DEFINITION_PATTERNS = [
    # Pattern: "X revenue is generated from..." or "X revenue includes..."
    re.compile(
        r'(?P<label>Advertising|Other|Reality Labs|Family of Apps|Subscription|Premium|Platform)'
        r'(?:\s+(?:revenue|revenues))?\s+'
        r'(?:is generated from|includes|consists of|is comprised of|is derived from|represents)'
        r'\s+(?P<definition>[^.]{20,400}\.)',
        re.IGNORECASE | re.DOTALL
    ),
    # Pattern: "Revenue from X includes..." 
    re.compile(
        r'Revenue\s+from\s+'
        r'(?P<label>advertising|delivery|subscriptions?|hardware|devices?|content)'
        r'\s+(?:includes|consists of|is comprised of)'
        r'\s+(?P<definition>[^.]{20,400}\.)',
        re.IGNORECASE | re.DOTALL
    ),
    # Pattern: "X consists of revenue from..."
    re.compile(
        r'(?P<label>Advertising|Other revenue|Reality Labs|FoA|RL)'
        r'\s+(?:consists of|includes)\s+'
        r'(?:revenue from\s+)?(?P<definition>[^.]{20,400}\.)',
        re.IGNORECASE | re.DOTALL
    ),
]

# Note 2 section extraction pattern
NOTE2_SECTION_PATTERN = re.compile(
    r'(?:Note\s*2|NOTE\s*2)[^A-Za-z]*(?:Revenue|REVENUE)',
    re.IGNORECASE
)


def _extract_note2_section(html_text: str, max_chars: int = 50000) -> Optional[str]:
    """Extract Note 2 - Revenue section from the filing."""
    match = NOTE2_SECTION_PATTERN.search(html_text)
    if not match:
        return None
    
    start = match.start()
    # Find end by looking for Note 3 or next major section
    end_pattern = re.compile(r'(?:Note\s*3|NOTE\s*3|Note\s*4|NOTE\s*4)', re.IGNORECASE)
    end_match = end_pattern.search(html_text, start + 100)
    
    if end_match:
        end = min(end_match.start(), start + max_chars)
    else:
        end = start + max_chars
    
    return html_text[start:end]


def _extract_note2_paragraph_definition(
    html_text: str,
    label: str,
) -> Optional[str]:
    """
    Extract definition for a label from Note 2 - Revenue section.
    
    This handles META-style filings where definitions are in prose paragraphs
    like "Advertising revenue is generated from marketers advertising on our apps..."
    
    Args:
        html_text: Full HTML text of the filing
        label: Revenue line label to find (e.g., "Advertising", "Reality Labs")
        
    Returns:
        Definition text if found, None otherwise
    """
    # First, try to extract Note 2 section for focused search
    note2_section = _extract_note2_section(html_text)
    search_text = note2_section if note2_section else html_text
    
    # Clean label for matching
    label_clean = label.strip().lower()
    
    # Build label-specific patterns
    label_escaped = re.escape(label.strip())
    
    # Phase 5: For "advertising", try the META-specific pattern FIRST
    # This is more reliable than the generic patterns which may match wrong context
    # NOTE: Search full html_text, not just Note 2 section, because the advertising
    # definition may be in Item 1 Business (before Note 2)
    if label_clean in ('advertising', 'advertising revenue'):
        # Pattern: "substantially all of our revenue from selling advertising placements..."
        advertising_pattern = re.compile(
            r'(?:substantially all|majority)\s+of\s+(?:our\s+)?revenue\s+'
            r'from\s+(?:selling\s+)?advertising\s+'
            r'([^.]{30,400}\.(?:\s+[A-Z][^.]{20,200}\.)?)',
            re.IGNORECASE | re.DOTALL
        )
        # Search full text for advertising (it may be in Item 1, not Note 2)
        match = advertising_pattern.search(html_text)
        if match:
            definition = match.group(1).strip()
            if not _is_table_header_contaminated(definition):
                # Prepend context for clarity
                full_desc = f"Revenue from selling advertising {definition}"
                return strip_accounting_sentences(full_desc)
        
        # Alternative: look for "revenue from marketers advertising on..."
        marketers_pattern = re.compile(
            r'revenue\s+from\s+marketers\s+advertising\s+on\s+'
            r'([^.]{20,300}\.)',
            re.IGNORECASE | re.DOTALL
        )
        match = marketers_pattern.search(html_text)  # Search full text
        if match:
            definition = match.group(1).strip()
            if not _is_table_header_contaminated(definition):
                full_desc = f"Revenue from marketers advertising on {definition}"
                return strip_accounting_sentences(full_desc)
        
        # Skip generic patterns for advertising - they match wrong context
        return None
    
    # Pattern 1: Direct label match "X revenue includes..."
    direct_pattern = re.compile(
        rf'{label_escaped}'
        r'(?:\s+(?:revenue|revenues|segment))?\s+'
        r'(?:is generated from|includes|consists of|is comprised of|is derived from|represents|are generated from)'
        r'\s+([^.]{20,500}\.)',
        re.IGNORECASE | re.DOTALL
    )
    
    match = direct_pattern.search(search_text)
    if match:
        definition = match.group(1).strip()
        # Check for contamination
        if not _is_table_header_contaminated(definition):
            return strip_accounting_sentences(definition)
    
    # Pattern 2: For "Other revenue" or "Other" - special handling
    if label_clean in ('other', 'other revenue'):
        other_pattern = re.compile(
            r'Other\s+(?:revenue|revenues)\s+'
            r'(?:consists of|includes|is comprised of|represents)'
            r'\s+([^.]{20,500}\.)',
            re.IGNORECASE | re.DOTALL
        )
        match = other_pattern.search(search_text)
        if match:
            definition = match.group(1).strip()
            if not _is_table_header_contaminated(definition):
                return strip_accounting_sentences(definition)
    
    # Pattern 3: For segment labels like "Reality Labs" or "Family of Apps"
    if label_clean in ('reality labs', 'rl', 'family of apps', 'foa'):
        segment_pattern = re.compile(
            rf'{label_escaped}'
            r'[^.]*?'
            r'(?:revenue|revenues)\s+'
            r'(?:is generated from|includes|consists of|are generated from)'
            r'\s+([^.]{20,500}\.)',
            re.IGNORECASE | re.DOTALL
        )
        match = segment_pattern.search(search_text)
        if match:
            definition = match.group(1).strip()
            if not _is_table_header_contaminated(definition):
                return strip_accounting_sentences(definition)
    
    # Pattern 4: Look for "revenue from [label] includes"
    revenue_from_pattern = re.compile(
        rf'revenue\s+from\s+{label_escaped}'
        r'\s+(?:includes|consists of|is comprised of|represents)'
        r'\s+([^.]{20,500}\.)',
        re.IGNORECASE | re.DOTALL
    )
    match = revenue_from_pattern.search(search_text)
    if match:
        definition = match.group(1).strip()
        if not _is_table_header_contaminated(definition):
            return strip_accounting_sentences(definition)
    
    return None


# ========================================================================
# Phase 6: Segment enumeration extraction (for NVDA-style filings)
# ========================================================================

def _extract_from_segment_enumeration(
    html_text: str,
    revenue_line: str,
    revenue_group: str,
) -> Optional[str]:
    """
    Extract description from segment enumeration pattern.
    
    Handles filings where the segment definition is:
    "{Segment} includes X; Y; Z" and revenue lines are components (X, Y, Z).
    
    Example (NVDA):
    - revenue_group: "Compute & Networking"
    - revenue_line: "Compute"
    - Pattern matches: "Compute & Networking segment includes our Data Center 
      accelerated computing platforms...; networking; automotive..."
    - Returns: "Data Center accelerated computing platforms and AI solutions and software"
    
    This is a fallback for generic labels like "Compute" where the definition
    exists only at the segment level, not as a standalone heading.
    
    Args:
        html_text: Full HTML text of the filing
        revenue_line: The specific revenue line label (e.g., "Compute")
        revenue_group: The parent segment/group (e.g., "Compute & Networking")
        
    Returns:
        Definition text if found, None otherwise
    """
    if not revenue_group or not revenue_line:
        return None
    
    # Skip if revenue_group is a generic fallback
    generic_groups = {'product/service disclosure', 'other', 'total'}
    if revenue_group.lower().strip() in generic_groups:
        return None
    
    # Build multiple regex patterns to handle HTML encoding variations
    # "Compute & Networking" might appear as:
    # - "Compute & Networking" (plain)
    # - "Compute &amp; Networking" (HTML encoded)
    # - "ComputeAndNetworking" (no space/symbol)
    group_stripped = revenue_group.strip()
    
    # Create variations of the group name for matching
    group_patterns = []
    
    # 1. Plain escaped version
    group_patterns.append(re.escape(group_stripped))
    
    # 2. HTML-encoded ampersand version (Compute & Networking -> Compute &amp; Networking)
    if '&' in group_stripped:
        html_encoded = group_stripped.replace('&', '&amp;')
        group_patterns.append(re.escape(html_encoded))
    
    # 3. No-symbol version (Compute & Networking -> ComputeAndNetworking)
    no_symbol = re.sub(r'\s*&\s*', 'And', group_stripped)
    no_symbol = re.sub(r'\s+', '', no_symbol)  # Remove spaces
    if no_symbol != group_stripped:
        group_patterns.append(re.escape(no_symbol))
    
    # Try each pattern variation
    for group_escaped in group_patterns:
        # Pattern: "{group} segment includes X; Y; Z" or "{group} includes X; Y; Z"
        pattern = re.compile(
            rf'{group_escaped}\s*(?:segment\s+)?includes?\s+'
            r'(?:our\s+)?'
            r'([^.]{50,600})',  # Capture the enumeration (allow semicolons)
            re.IGNORECASE | re.DOTALL
        )
        
        match = pattern.search(html_text)
        if match:
            break
    else:
        # No pattern matched
        return None
    
    # Extract the enumeration from the matched pattern
    enumeration = match.group(1).strip()
    
    # Split by semicolons to get individual items
    items = [item.strip() for item in enumeration.split(';') if item.strip()]
    
    if not items:
        return None
    
    line_lower = revenue_line.lower().strip()
    
    # For "Compute" - FIRST try to find detailed "offerings include" pattern
    if line_lower == 'compute':
        # Priority 1: Look for detailed compute description with "offerings include"
        compute_pattern = re.compile(
            r'(?:our\s+)?compute\s+offerings?\s+include[s]?\s+'
            r'([^.]{30,400}\.)',
            re.IGNORECASE
        )
        cm = compute_pattern.search(html_text)
        if cm:
            full_match = "Our compute offerings include " + cm.group(1).strip()
            return strip_accounting_sentences(full_match)
        
        # Priority 2: Fallback to first item from segment enumeration
        desc = items[0]
        # Clean up any trailing conjunctions
        desc = re.sub(r'\s+and\s*$', '', desc, flags=re.IGNORECASE)
        if len(desc) >= 20:
            return strip_accounting_sentences(desc)
    
    # For "Networking" - FIRST try to find detailed "offerings include" pattern
    # This is richer than the segment enumeration fragments
    if line_lower == 'networking':
        # Priority 1: Look for detailed networking description with "offerings include"
        networking_pattern = re.compile(
            r'(?:our\s+)?networking\s+offerings?\s+include[s]?\s+'
            r'([^.]{30,400}\.)',
            re.IGNORECASE
        )
        nm = networking_pattern.search(html_text)
        if nm:
            full_match = "Our networking offerings include " + nm.group(1).strip()
            return strip_accounting_sentences(full_match)
        
        # Priority 2: Fallback to segment enumeration item (but only if substantial)
        for item in items:
            if 'network' in item.lower() and 'compute' not in item.lower():
                # Only use if it's a real description, not just the word "networking"
                if len(item.strip()) > 30:
                    return strip_accounting_sentences(item)
    
    # For other generic labels, try to find a matching item by keyword
    for item in items:
        # Check if the revenue_line keyword appears in this item
        if line_lower in item.lower():
            return strip_accounting_sentences(item)
    
    return None


def _extract_heading_based_definition(
    html_text: str,
    label: str,
    section_text: Optional[str] = None,
) -> Optional[str]:
    """
    Extract definition for a label that appears as a heading in Item 1.
    
    For Apple-style 10-Ks where products/services have dedicated headings
    (e.g., <b>Services</b>, <strong>iPhone</strong>) followed by descriptive paragraphs.
    
    Phase 3 enhancement: For parent headings like "Services" that have subheadings
    (Advertising, AppleCare, etc.), aggregate ALL child content until a true
    peer heading (another major product category like "iPhone", "Mac").
    
    Strategy:
    1. Find label as a HEADING (bold/strong/h1-h3 tag)
    2. Identify known major section headings (peer headings to stop at)
    3. Continue collecting content past child subheadings
    4. Stop only at a peer heading or max chars
    5. Apply accounting sentence filter
    
    Args:
        html_text: Full HTML text of the filing
        label: Revenue line label to find (e.g., "Services", "iPhone")
        section_text: Optional pre-extracted section (Item 1) to search within
        
    Returns:
        Definition text if found, None otherwise
    """
    # Use section_text if provided, otherwise search full HTML
    search_text = section_text if section_text else html_text
    if not search_text:
        return None
    
    # Escape label for regex (handle special chars)
    label_escaped = re.escape(label.strip())
    
    # Known major product/service headings (peer level) - used to stop aggregation
    # These are typically the top-level revenue line items
    PEER_HEADINGS = {
        'iphone', 'mac', 'ipad', 'services', 'wearables', 'home', 'accessories',
        'wearables, home and accessories', 'products', 'total net sales',
        # Generic peer markers
        'item 2', 'item 3', 'business', 'properties', 'legal proceedings',
    }
    
    # Known child subheadings (not peer level) - continue past these
    # These are subsections within a major category
    CHILD_SUBHEADINGS = {
        'advertising', 'apple care', 'applecare', 'cloud services', 
        'digital content', 'payment services', 'other services',
        'app store', 'apple music', 'apple tv+', 'apple arcade', 'apple news+',
        'apple fitness+', 'icloud', 'apple card', 'apple pay',
    }
    
    heading_tags = ['b', 'strong', 'h1', 'h2', 'h3', 'h4']
    
    for tag in heading_tags:
        pattern = rf'<{tag}[^>]*>\s*{label_escaped}\s*</{tag}>'
        match = re.search(pattern, search_text, re.IGNORECASE)
        
        if match:
            start_pos = match.end()
            remaining_text = search_text[start_pos:]
            
            # Pattern to find ANY heading (we'll check if it's peer or child)
            any_heading_pattern = rf'<(?:{"|".join(heading_tags)})[^>]*>([A-Z][^<]{{1,100}})</(?:{"|".join(heading_tags)})>'
            
            # Find ALL headings in the remaining text
            content_end = len(remaining_text)
            max_content = 15000  # Allow up to 15k chars for parent sections with subsections
            
            for heading_match in re.finditer(any_heading_pattern, remaining_text[:max_content], re.IGNORECASE):
                heading_text = heading_match.group(1).strip().lower()
                
                # Check if this is a peer heading (stop point)
                is_peer = heading_text in PEER_HEADINGS
                # Also treat it as peer if it's a different major product (single capitalized word > 3 chars)
                # that's NOT in the child list
                is_major_section = (
                    len(heading_text.split()) <= 2 and  # Short heading (1-2 words)
                    heading_text not in CHILD_SUBHEADINGS and
                    heading_text != label.strip().lower() and  # Not the label itself
                    heading_match.start() > 200  # Must be at least 200 chars in
                )
                
                if is_peer or is_major_section:
                    content_end = heading_match.start()
                    break
            
            # Extract content
            content = remaining_text[:min(content_end, max_content)]
            
            # Parse HTML to extract clean text
            try:
                soup = BeautifulSoup(content, 'lxml')
                text_content = soup.get_text(separator=' ', strip=True)
            except Exception:
                text_content = re.sub(r'<[^>]+>', ' ', content)
            
            # Clean and filter
            cleaned = _clean(text_content)
            filtered = strip_accounting_sentences(cleaned)
            
            if filtered and len(filtered) >= 50:
                # Phase 5: Check for table-header contamination before returning
                if _is_table_header_contaminated(filtered):
                    # Skip this match - it's table structure, not a definition
                    continue
                
                # For parent sections with subsections, allow longer descriptions
                # but still summarize to key sentences
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', filtered)
                if len(sentences) > 6:
                    # Take first 6 sentences for comprehensive summary
                    filtered = ' '.join(sentences[:6])
                return filtered[:1500]  # Increased cap for parent sections
    
    # Also try finding the label in a slightly different format:
    # Sometimes it's: <span style="font-weight:bold">Services</span>
    bold_style_pattern = rf'<[^>]*(?:font-weight\s*:\s*bold|font-weight\s*:\s*700)[^>]*>\s*{label_escaped}\s*</[^>]+>'
    match = re.search(bold_style_pattern, search_text, re.IGNORECASE)
    
    if match:
        start_pos = match.end()
        content = search_text[start_pos:start_pos + 8000]
        
        # Fix: Strip leading orphan closing tags (like </div></span>) that confuse BS4
        # These occur because we start extraction after the opening tag's closing >
        content = re.sub(r'^(?:</\w+>)+\s*', '', content)
        
        try:
            soup = BeautifulSoup(content, 'lxml')
            text_content = soup.get_text(separator=' ', strip=True)
        except Exception:
            text_content = re.sub(r'<[^>]+>', ' ', content)
        
        cleaned = _clean(text_content)
        filtered = strip_accounting_sentences(cleaned)
        
        if filtered and len(filtered) >= 50:
            # Phase 5: Check for table-header contamination
            if _is_table_header_contaminated(filtered):
                return None  # Don't return contaminated text
            
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', filtered)
            if len(sentences) > 6:
                filtered = ' '.join(sentences[:6])
            return filtered[:1500]
    
    return None


# ========================================================================
# Phase 2: DOM-based table-local context and footnote extraction
# ========================================================================
def build_table_local_context_dom(table_elem, html: str, n_siblings: int = 5) -> str:
    """
    Build table-local context using DOM adjacency for robust footnote extraction.
    
    This addresses the issue where footnote markers/parentheses are split across 
    HTML tags, making regex on raw HTML unreliable.
    
    Strategy:
    1. Get the table element
    2. Collect preceding sibling blocks (for captions/headers)
    3. Collect following sibling blocks (where footnotes typically appear)
    4. Return normalized text using get_text()
    
    Args:
        table_elem: BeautifulSoup table element
        html: Full HTML string (used as fallback)
        n_siblings: Number of sibling elements to collect on each side
        
    Returns:
        Normalized text from table-local context
    """
    if table_elem is None:
        return ""
    
    parts = []
    
    # Collect preceding siblings (captions, headers)
    preceding = []
    sibling = table_elem.find_previous_sibling()
    count = 0
    while sibling and count < n_siblings:
        if sibling.name in ('div', 'p', 'span', 'td', 'tr', 'table'):
            text = sibling.get_text(" ", strip=True)
            if text and len(text) < 5000:  # Skip very long blocks
                preceding.append(text)
        sibling = sibling.find_previous_sibling()
        count += 1
    
    # Reverse to maintain order
    for text in reversed(preceding):
        parts.append(text)
    
    # Get table text
    table_text = table_elem.get_text(" ", strip=True)
    parts.append(table_text)
    
    # Collect following siblings (footnotes typically appear here)
    sibling = table_elem.find_next_sibling()
    count = 0
    while sibling and count < n_siblings:
        if sibling.name in ('div', 'p', 'span', 'td', 'tr', 'table'):
            text = sibling.get_text(" ", strip=True)
            if text and len(text) < 5000:
                parts.append(text)
                # Check for footnote separator pattern (end of footnote area)
                if '___' in text and count > 2:
                    break
        sibling = sibling.find_next_sibling()
        count += 1
    
    return "\n".join(parts)


def extract_footnotes_from_dom_context(table_elem, html: str) -> Dict[str, str]:
    """
    Extract footnote definitions from table-local context using DOM + normalized text.
    
    This is more robust than regex on raw HTML because it handles:
    - Footnote markers split across tags
    - Whitespace normalization
    - iXBRL tag complexity
    
    Args:
        table_elem: BeautifulSoup table element
        html: Full HTML string
        
    Returns:
        Dict mapping footnote number to definition text
    """
    # Build normalized text from table-local context
    context_text = build_table_local_context_dom(table_elem, html)
    
    if not context_text:
        return {}
    
    # Extract footnotes from normalized text
    return _extract_footnotes_from_text(context_text, prioritize_includes=True)


# ========================================================================
# Fix D: Extract footnote IDs from table DOM (handles iXBRL superscripts)
# ========================================================================
def extract_footnote_ids_from_table(table_elem) -> Dict[str, List[str]]:
    """
    Extract footnote IDs from table row labels using DOM parsing.
    
    This handles iXBRL HTML where superscripts (<sup>) or anchors (<a>) 
    are used for footnote markers but get stripped during text extraction.
    
    Args:
        table_elem: BeautifulSoup table element
        
    Returns:
        Dict mapping row label text → list of footnote IDs found in that cell
        e.g. {"Online stores": ["1"], "Third-party seller services": ["3"]}
    """
    if table_elem is None:
        return {}
    
    label_to_footnotes: Dict[str, List[str]] = {}
    
    for row in table_elem.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        
        # First non-empty cell is typically the label
        label_cell = None
        for cell in cells:
            text = cell.get_text(strip=True)
            if text and len(text) > 1:  # Skip empty or single-char cells
                label_cell = cell
                break
        
        if not label_cell:
            continue
        
        # Get clean label text (without footnote markers)
        label_text = label_cell.get_text(strip=True)
        # Remove trailing footnote markers from text
        label_clean = _FOOTNOTE_MARKER_RE.sub('', label_text).strip()
        
        # Find footnote markers in DOM: <sup>, <a>, ix:footnoteReference
        footnote_ids: List[str] = []
        
        # Check <sup> tags
        for sup in label_cell.find_all("sup"):
            fn_id = sup.get_text(strip=True)
            if fn_id and (fn_id.isdigit() or fn_id in "123456789"):
                footnote_ids.append(fn_id)
        
        # Check <a> anchors with href like "#footnote_1"
        for anchor in label_cell.find_all("a"):
            href = anchor.get("href", "")
            anchor_text = anchor.get_text(strip=True)
            # href="#fn_1" or anchor text is a number
            if anchor_text and anchor_text.isdigit():
                footnote_ids.append(anchor_text)
            elif "#" in href:
                # Extract number from href like #footnote_1, #fn1, etc.
                fn_match = re.search(r'(\d+)', href)
                if fn_match:
                    footnote_ids.append(fn_match.group(1))
        
        # Check for ix:footnoteReference (iXBRL specific)
        for fn_ref in label_cell.find_all(lambda tag: 'footnote' in tag.name.lower()):
            ref_text = fn_ref.get_text(strip=True)
            if ref_text and ref_text.isdigit():
                footnote_ids.append(ref_text)
        
        if label_clean and footnote_ids:
            # Deduplicate while preserving order
            unique_ids = list(dict.fromkeys(footnote_ids))
            label_to_footnotes[label_clean] = unique_ids
    
    return label_to_footnotes


def _extract_footnotes_from_text(text: str, prioritize_includes: bool = True) -> Dict[str, str]:
    """
    Extract footnote definitions from text.
    
    Looks for patterns like:
    - "(1) Includes product sales..."
    - "(2) Includes product sales where..."
    
    Args:
        text: Text to search for footnotes
        prioritize_includes: If True, strongly prefer footnotes starting with "Includes"
        
    Returns dict mapping footnote number to definition text.
    """
    if not text:
        return {}
    
    footnotes: Dict[str, str] = {}
    
    # Pattern: (N) followed by "Includes" - this is the most reliable pattern for revenue footnotes
    for i in range(1, 10):
        # First priority: (N) Includes... (standard revenue footnote format)
        pattern_includes = rf'\({i}\)\s*(Includes\s+[^(]*?)(?=\(\d+\)|_____|$)'
        matches = re.findall(pattern_includes, text, re.IGNORECASE | re.DOTALL)
        if matches:
            for match in matches:
                cleaned = _clean(match)
                if len(cleaned) >= 20:
                    footnotes[str(i)] = cleaned[:600]
                    break
        
        # Second priority: Other substantive verbs
        if str(i) not in footnotes:
            pattern_verbs = rf'\({i}\)\s*((?:Represents|Consists|Comprises|Contains)\s+[^(]*?)(?=\(\d+\)|_____|$)'
            matches2 = re.findall(pattern_verbs, text, re.IGNORECASE | re.DOTALL)
            if matches2:
                for match in matches2:
                    cleaned = _clean(match)
                    if len(cleaned) >= 20:
                        footnotes[str(i)] = cleaned[:600]
                        break
        
        # Third priority (only if not prioritizing includes): Capital letter start
        if str(i) not in footnotes and not prioritize_includes:
            pattern_capital = rf'\({i}\)\s*([A-Z][^(]*?)(?=\(\d+\)|_____|$)'
            matches3 = re.findall(pattern_capital, text, re.DOTALL)
            if matches3:
                for match in matches3:
                    cleaned = _clean(match)
                    # Filter out non-definition matches
                    if len(cleaned) >= 30 and not cleaned[0].isdigit():
                        if cleaned.count('$') <= 1:
                            footnotes[str(i)] = cleaned[:600]
                            break
    
    return footnotes


def _extract_footnote_for_label(label: str, html_text: str, table_context_text: str) -> Optional[str]:
    """
    Extract footnote definition for a revenue line label that contains a footnote marker.
    
    Args:
        label: Revenue line label like "Online stores (1)"
        html_text: Full HTML text to search
        table_context_text: Table's nearby text context
        
    Returns:
        Footnote definition text if found, None otherwise.
    """
    # Check if label has a footnote marker
    match = _FOOTNOTE_MARKER_RE.search(label)
    if not match:
        return None
    
    footnote_num = match.group(1)
    label_clean = _FOOTNOTE_MARKER_RE.sub('', label).strip().lower()
    
    # Strategy: Look for table separator (___) followed by footnotes
    # This is the most reliable way to find footnotes for a specific table
    
    # Step 1: Find the label in context of revenue table, then look for separator + footnotes
    low = html_text.lower() if html_text else ""
    
    # Find occurrences of the label
    label_positions = []
    search_pos = 0
    while True:
        idx = low.find(label_clean, search_pos)
        if idx == -1:
            break
        label_positions.append(idx)
        search_pos = idx + len(label_clean)
        if len(label_positions) >= 10:  # Limit search
            break
    
    # For each label position, look for "_____" separator followed by footnotes
    for label_idx in label_positions:
        # Look for separator after the label (within 3000 chars)
        separator_idx = html_text.find("_____", label_idx, label_idx + 3000)
        if separator_idx != -1:
            # Found separator - extract footnotes from right after it
            footnote_window = html_text[separator_idx:separator_idx + 10000]
            
            # Look specifically for (N) Includes pattern in this window
            pattern = rf'\({footnote_num}\)\s*(Includes\s+[^(]+?)(?=\(\d+\)|_____|$)'
            matches = re.findall(pattern, footnote_window, re.IGNORECASE | re.DOTALL)
            if matches:
                cleaned = _clean(matches[0])
                if len(cleaned) >= 20:
                    return cleaned[:600]
    
    # Step 2: Fallback - search for all (N) Includes patterns near the label
    for label_idx in label_positions:
        window = html_text[label_idx:label_idx + 15000]
        
        # Look for (N) Includes pattern
        pattern = rf'\({footnote_num}\)\s*(Includes\s+[^(]+?)(?=\(\d+\)|_____|$)'
        matches = re.findall(pattern, window, re.IGNORECASE | re.DOTALL)
        if matches:
            cleaned = _clean(matches[0])
            if len(cleaned) >= 20:
                return cleaned[:600]
    
    # Step 3: Try table context
    if table_context_text:
        footnotes = _extract_footnotes_from_text(table_context_text)
        if footnote_num in footnotes:
            return footnotes[footnote_num]
    
    # Step 4: Fallback - search Item 8 section
    item8_match = _ITEM8_RE.search(html_text)
    if item8_match:
        item8_start = item8_match.start()
        item8_section = html_text[item8_start:item8_start + 200000]
        
        # Look for separator + footnotes pattern
        separator_pattern = r'_____+\s*\(\d+\)'
        sep_matches = list(re.finditer(separator_pattern, item8_section))
        for sep_match in sep_matches:
            footnote_window = item8_section[sep_match.start():sep_match.start() + 10000]
            pattern = rf'\({footnote_num}\)\s*(Includes\s+[^(]+?)(?=\(\d+\)|_____|$)'
            matches = re.findall(pattern, footnote_window, re.IGNORECASE | re.DOTALL)
            if matches:
                cleaned = _clean(matches[0])
                if len(cleaned) >= 20:
                    return cleaned[:600]
    
    return None


def _parse_number(s: str) -> Optional[float]:
    if s is None:
        return None
    t = _clean(s)
    if t in {"", "-", "—", "–"}:
        return None
    # Handle parentheses negatives
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg = True
        t = t[1:-1]
    t = t.replace("$", "").replace(",", "").strip()
    try:
        v = float(t)
        return -v if neg else v
    except Exception:
        return None


def _parse_money_to_int(s: str) -> Optional[int]:
    v = _parse_number(s)
    if v is None:
        return None
    return int(round(v))


def rank_candidates_for_financial_tables(candidates: List[TableCandidate]) -> List[TableCandidate]:
    return sorted(
        candidates,
        key=lambda c: (
            float(guess_item8_score(c)),
            bool(getattr(c, "has_year_header", False)),
            bool(getattr(c, "has_units_marker", False)),
            float(getattr(c, "money_cell_ratio", 0.0)),
            float(getattr(c, "numeric_cell_ratio", 0.0)),
            len(getattr(c, "keyword_hits", []) or []),
            int(getattr(c, "n_rows", 0)) * int(getattr(c, "n_cols", 0)),
        ),
        reverse=True,
    )


def guess_item8_score(c: TableCandidate) -> float:
    """Soft signal: does the local context look like Item 8 / Notes / Segment Note?"""
    blob = " ".join(
        [
            str(getattr(c, "heading_context", "") or ""),
            str(getattr(c, "caption_text", "") or ""),
            str(getattr(c, "nearby_text_context", "") or ""),
        ]
    )
    blob = _clean(blob)
    score = 0.0
    if _ITEM8_RE.search(blob):
        score += 3.0
    if _SEGMENT_NOTE_RE.search(blob):
        score += 1.0
    # If it looks like Item 7/MD&A, slightly downweight (soft preference, not exclusion)
    if _ITEM7_RE.search(blob):
        score -= 1.0
    return score


def extract_keyword_windows(
    html_path: Path,
    *,
    keywords: List[str],
    window_chars: int = 2500,
    max_windows: int = 12,
) -> List[str]:
    """Deterministically extract short text windows around keywords for LLM context."""
    # Use cached soup to avoid re-parsing large HTML files
    soup = _get_cached_soup(html_path)
    text = soup.get_text(" ", strip=True)
    text = _clean(text)
    low = text.lower()

    windows: List[str] = []
    for kw in keywords:
        k = kw.lower()
        start = 0
        while True:
            i = low.find(k, start)
            if i == -1:
                break
            a = max(0, i - window_chars // 3)
            b = min(len(text), i + window_chars)
            snippet = _clean(text[a:b])
            if snippet and snippet not in windows:
                windows.append(snippet)
            start = i + max(1, len(k))
            if len(windows) >= max_windows:
                return windows
    return windows


def document_scout(html_path: Path, *, max_headings: int = 80) -> Dict[str, Any]:
    """Lightweight scan of headings to help the LLM orient itself."""
    # Use cached soup to avoid re-parsing large HTML files
    soup = _get_cached_soup(html_path)
    headings: List[str] = []
    for tag in soup.find_all(["h1", "h2", "h3", "b", "strong"]):
        txt = _clean(tag.get_text(" ", strip=True))
        if 5 <= len(txt) <= 180 and txt not in headings:
            headings.append(txt)
        if len(headings) >= max_headings:
            break
    return {"headings": headings}


def _candidate_summary(c: TableCandidate) -> Dict[str, Any]:
    return {
        "table_id": c.table_id,
        "n_rows": c.n_rows,
        "n_cols": c.n_cols,
        "detected_years": c.detected_years,
        "keyword_hits": c.keyword_hits,
        "item8_score": guess_item8_score(c),
        "has_year_header": getattr(c, "has_year_header", False),
        "has_units_marker": getattr(c, "has_units_marker", False),
        "units_hint": getattr(c, "units_hint", ""),
        "money_cell_ratio": getattr(c, "money_cell_ratio", 0.0),
        "numeric_cell_ratio": getattr(c, "numeric_cell_ratio", 0.0),
        "row_label_preview": getattr(c, "row_label_preview", [])[:12],
        "caption_text": getattr(c, "caption_text", "")[:200],
        "heading_context": getattr(c, "heading_context", "")[:200],
        "nearby_text_context": getattr(c, "nearby_text_context", "")[:280],
    }


def select_segment_revenue_table(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    candidates: List[TableCandidate],
    scout: Dict[str, Any],
    snippets: List[str],
    max_candidates: int = 80,
) -> Dict[str, Any]:
    ranked = rank_candidates_for_financial_tables(candidates)[:max_candidates]
    payload = [_candidate_summary(c) for c in ranked]

    system = (
        "You are a financial filings analyst. You select the single best HTML table candidate "
        "that represents REVENUE BY REPORTABLE SEGMENT (or equivalent business segments) for the latest fiscal year. "
        "Prefer tables from Item 8 / Notes to Financial Statements when possible, but you may select other sections if they clearly match and are consistent. "
        "Output must be STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "objective": "Find the reportable segment revenue table (e.g., segments with revenue totals).",
            "headings": scout.get("headings", [])[:40],
            "retrieved_snippets": snippets[:10],
            "table_candidates": payload,
            "output_schema": {
                "table_id": "string like t0071",
                "confidence": "number 0..1",
                "kind": "string, use 'segment_revenue' or 'not_found'",
                "rationale": "short string",
            },
        },
        ensure_ascii=False,
    )
    out = llm.json_call(system=system, user=user, max_output_tokens=700)
    return out


TABLE_KINDS = [
    "segment_revenue",
    "product_service_revenue",
    "segment_results_of_operations",
    "other",
]


def discover_primary_business_lines(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    snippets: List[str],
) -> Dict[str, Any]:
    """Text-first agent: infer primary business lines for CSV1 (Option 1).

    Output contracts:
      - dimension: product_category | reportable_segments
      - segments: list[str] (primary business lines)
      - include_segments_optional: list[str] (e.g., Corporate adjustments) if needed for reconciliation
    """
    system = (
        "You are a financial filings analyst. Determine the primary business-line dimension for CSV1.\n"
        "Rules:\n"
        "- For AAPL, treat business lines as product categories (iPhone, Mac, iPad, Wearables/Home/Accessories, Services).\n"
        "- For MSFT and GOOGL, treat business lines as reportable segments (e.g., Intelligent Cloud).\n"
        "- If the filing includes corporate adjustments (e.g., hedging gains/losses) that are included in Total Revenues, "
        "put that under include_segments_optional=['Corporate'].\n"
        "Output STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "snippets": snippets[:10],
            "few_shot_examples": [
                {
                    "ticker": "AAPL",
                    "dimension": "product_category",
                    "segments": ["iPhone", "Mac", "iPad", "Wearables, Home and Accessories", "Services"],
                    "include_segments_optional": [],
                },
                {
                    "ticker": "MSFT",
                    "dimension": "reportable_segments",
                    "segments": [
                        "Productivity and Business Processes",
                        "Intelligent Cloud",
                        "More Personal Computing",
                    ],
                    "include_segments_optional": [],
                },
                {
                    "ticker": "GOOGL",
                    "dimension": "reportable_segments",
                    "segments": ["Google Services", "Google Cloud", "Other Bets"],
                    "include_segments_optional": ["Corporate"],
                },
            ],
            "output_schema": {
                "dimension": "product_category | reportable_segments",
                "segments": "list[string]",
                "include_segments_optional": "list[string]",
                "notes": "short string",
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=700)


def _classify_table_dimension(c: TableCandidate) -> str:
    """
    Pre-classify a table candidate's disclosure dimension based on metadata.
    
    Returns: 'product_service', 'segment', 'geography', or 'unknown'
    """
    text = " ".join([
        str(getattr(c, "caption_text", "") or ""),
        str(getattr(c, "heading_context", "") or ""),
        " ".join(getattr(c, "row_label_preview", []) or []),
    ]).lower()
    
    # Product/service patterns (most specific first)
    product_service_patterns = [
        r"groups?\s+of\s+similar\s+products?\s+(and|&)\s+services?",
        r"disaggregat(ed|ion)\s+(of\s+)?revenue",
        r"revenue\s+(from\s+external\s+customers\s+)?by\s+(product|service|category)",
        r"net\s+sales\s+by\s+(product|category|type)",
        r"by\s+(product|service)\s+(line|category|type)",
    ]
    for pattern in product_service_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "product_service"
    
    # Geography patterns
    geography_patterns = [
        r"by\s+geograph",
        r"geographic\s+(area|region)",
        r"revenue\s+by\s+region",
    ]
    for pattern in geography_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "geography"
    
    # Segment patterns
    segment_patterns = [
        r"reportable\s+segment",
        r"operating\s+segment",
        r"segment\s+(revenue|result)",
        r"revenue\s+by\s+segment",
    ]
    for pattern in segment_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "segment"
    
    return "unknown"


def select_revenue_disaggregation_table(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    candidates: List[TableCandidate],
    scout: Dict[str, Any],
    snippets: List[str],
    segments: List[str],
    keyword_hints: Optional[List[str]] = None,
    max_candidates: int = 80,
    prefer_granular: bool = True,
) -> Dict[str, Any]:
    """Select the most granular revenue disaggregation table that includes a Total row.
    
    When prefer_granular=True, prioritize tables with product/service line items
    (e.g., 'Revenue by Products and Services') over segment-level totals.
    """
    ranked = rank_candidates_for_financial_tables(candidates)[:max_candidates]
    
    # Pre-classify each candidate's dimension
    payload = []
    for c in ranked:
        summary = _candidate_summary(c)
        summary["inferred_dimension"] = _classify_table_dimension(c)
        payload.append(summary)
    
    # Sort to prioritize product_service dimension tables
    dimension_priority = {"product_service": 0, "segment": 1, "unknown": 2, "geography": 3}
    payload.sort(key=lambda x: dimension_priority.get(x.get("inferred_dimension", "unknown"), 2))
    
    granular_guidance = ""
    if prefer_granular:
        granular_guidance = (
            "- **CRITICAL**: When multiple tables exist, PREFER tables with inferred_dimension='product_service' "
            "over tables with inferred_dimension='segment'. Product/service tables provide the most granular "
            "revenue breakdown (e.g., 'Online stores', 'Third-party seller services', 'Subscription services').\n"
            "- Tables titled 'Net sales by groups of similar products and services' or 'Disaggregation of Revenue' "
            "are BETTER than 'Revenue by Segment' or 'Segment Information' tables.\n"
            "- For Amazon (AMZN), the product/service table has: Online stores, Physical stores, Third-party seller "
            "services, Subscription services, Advertising services, AWS, Other.\n"
        )
    
    system = (
        "You are a financial filings analyst. Select the single best table that DISAGGREGATES revenue "
        "by business lines (segments or product categories) and includes a Total Revenue/Net Sales row.\n"
        "Constraints:\n"
        f"{granular_guidance}"
        "- Ignore geography-only tables (inferred_dimension='geography').\n"
        "- Prefer Item 8 / Notes (Note 17 or Note 18 often has the most granular breakdown).\n"
        "- Prefer tables whose year columns are recent fiscal years (>= 2018).\n"
        "- Each candidate includes 'inferred_dimension' field indicating detected dimension type.\n"
        "Output STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "business_lines": segments,
            "keyword_hints": keyword_hints or [],
            "headings": scout.get("headings", [])[:40],
            "retrieved_snippets": snippets[:10],
            "table_candidates": payload,
            "output_schema": {
                "table_id": "tXXXX",
                "confidence": "0..1",
                "selected_dimension": "product_service|segment|other",
                "rationale": "short string",
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=700)


def infer_disaggregation_layout(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    table_id: str,
    candidate: TableCandidate,
    grid: List[List[str]],
    business_lines: List[str],
    max_rows_for_llm: int = 40,
) -> Dict[str, Any]:
    """Infer layout for tables like:
    - AAPL: Category | Product/Service | FY2025 | FY2024 | ...
    - MSFT/GOOGL: Segment | Product/Service | FY... | ...
    """
    preview = grid[:max_rows_for_llm]
    system = (
        "You analyze a revenue disaggregation table from a 10-K. "
        "Identify which columns correspond to Segment (optional), Item/Product (required), and years, "
        "and how to identify the Total row.\n"
        "Important: year columns should be recent fiscal years (>= 2018) and usually appear as FY2025/FY2024 or 2025/2024.\n"
        "Output STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "table_id": table_id,
            "business_lines": business_lines,
            "candidate_summary": _candidate_summary(candidate),
            "table_grid_preview": preview,
            "output_schema": {
                "segment_col": "int|null (e.g., 0 for Segment; null if no segment column)",
                "item_col": "int (e.g., Product / Service column)",
                "year_cols": {"YYYY": "int column index"},
                "header_rows": "list[int]",
                "total_row_regex": "string regex matching the Total row label (e.g., Total Revenues|Total Net Sales)",
                "exclude_row_regex": "string regex for rows to exclude (e.g., Hedging gains)",
                "units_multiplier": "int (1, 1000, 1000000, 1000000000)",
                "notes": "short string",
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=900)


def extract_disaggregation_rows_from_grid(
    grid: List[List[str]],
    *,
    layout: Dict[str, Any],
    target_year: Optional[int] = None,
    business_lines: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Deterministically extract (segment,item,value) rows and a table total."""
    # Pad rows so column indices inferred from a wide header row work across short rows.
    max_len = max((len(r) for r in grid), default=0)
    if max_len > 0:
        grid = [list(r) + [""] * (max_len - len(r)) for r in grid]

    seg_col = layout.get("segment_col")
    seg_col = int(seg_col) if seg_col is not None else None
    item_col = int(layout["item_col"])
    year_cols_raw = layout.get("year_cols") or {}
    year_cols: Dict[int, int] = {int(y): int(ci) for y, ci in year_cols_raw.items()}
    if not year_cols:
        raise ValueError("No year_cols detected")
    year = target_year or max(year_cols.keys())
    if year not in year_cols:
        year = max(year_cols.keys())
    val_col = year_cols[year]

    header_rows = set(int(i) for i in (layout.get("header_rows") or []))
    total_re = re.compile(layout.get("total_row_regex") or r"total", re.IGNORECASE)
    exclude_re = re.compile(layout.get("exclude_row_regex") or r"$^", re.IGNORECASE)
    mult = int(layout.get("units_multiplier") or 1)
    if mult <= 0:
        mult = 1

    bl_norm = {b.lower(): b for b in (business_lines or [])}
    def _is_business_line(s: str) -> bool:
        if not bl_norm:
            return True
        return s.lower() in bl_norm

    rows: List[Dict[str, Any]] = []
    total_val: Optional[int] = None
    last_seg: str = ""

    for r_i, row in enumerate(grid):
        if r_i in header_rows:
            continue
        if item_col >= len(row) or val_col >= len(row):
            continue
        seg = _clean(row[seg_col]) if seg_col is not None and seg_col < len(row) else ""
        if seg_col is not None:
            if seg:
                last_seg = seg
            else:
                # iXBRL often blanks repeated segment labels; fill down.
                seg = last_seg
        item = _clean(row[item_col])
        if not item:
            continue
        if exclude_re.search(item) or exclude_re.search(seg):
            continue

        # Some tables put a currency symbol column before the number (e.g., '$', '209,586').
        raw_val = _parse_money_to_int(row[val_col])
        if raw_val is None and (val_col + 1) < len(row):
            raw_val = _parse_money_to_int(row[val_col + 1])
        if raw_val is None and (val_col + 2) < len(row):
            raw_val = _parse_money_to_int(row[val_col + 2])
        if raw_val is None:
            continue
        val = int(raw_val) * mult

        # Total row detection: match across the row, not just item/segment cell.
        if total_re.search(item) or total_re.search(seg) or any(total_re.search(_clean(c)) for c in row if c):
            total_val = val
            continue

        if seg and not _is_business_line(seg) and seg.lower() != "corporate":
            # keep corporate as optional; otherwise require match if business lines provided
            continue

        rows.append({"segment": seg, "item": item, "value": val, "year": year})

    return {"year": year, "rows": rows, "total_value": total_val}


def extract_segment_revenue_from_segment_results_grid(
    grid: List[List[str]],
    *,
    segments: List[str],
    target_year: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract segment revenues from a 'segment results of operations' style table.

    Shape example (MSFT t0071):
      - segment header rows: 'Productivity and Business Processes'
      - metric rows under each segment: 'Revenue', 'Cost of revenue', ...
      - final 'Total' section with 'Revenue'
    """
    import re

    # Pad rows to a common width
    max_len = max((len(r) for r in grid), default=0)
    if max_len > 0:
        grid = [list(r) + [""] * (max_len - len(r)) for r in grid]

    year_re = re.compile(r"\b(20\d{2})\b")
    year_cols: dict[int, int] = {}
    for r in grid[:15]:
        for ci, cell in enumerate(r):
            m = year_re.search(str(cell or ""))
            if not m:
                continue
            y = int(m.group(1))
            if 2015 <= y <= 2100:
                year_cols.setdefault(y, ci)
    if not year_cols:
        raise ValueError("No year columns detected in segment results grid")

    year = target_year or max(year_cols.keys())
    if year not in year_cols:
        year = max(year_cols.keys())
    val_col = year_cols[year]

    seg_norm = {s.lower(): s for s in segments}
    current_seg = ""
    out: dict[str, int] = {}
    total_value: Optional[int] = None

    for row in grid:
        if not row:
            continue
        first = _clean(row[0] or "")
        if not first:
            continue

        # Segment header row
        if first.lower() in seg_norm or first.lower() == "total":
            current_seg = seg_norm.get(first.lower(), "Total")
            continue

        # Metric row under current segment
        if first.lower() == "revenue" and current_seg:
            raw = _parse_money_to_int(row[val_col])
            if raw is None and (val_col + 1) < len(row):
                raw = _parse_money_to_int(row[val_col + 1])
            if raw is None and (val_col + 2) < len(row):
                raw = _parse_money_to_int(row[val_col + 2])
            if raw is None:
                continue
            if current_seg == "Total":
                total_value = int(raw)
            else:
                out[current_seg] = int(raw)

    if not out:
        raise ValueError("No segment revenues extracted from segment results grid")
    return {"year": year, "segment_totals": out, "total_value": total_value}


def classify_table_candidates(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    candidates: List[TableCandidate],
    scout: Dict[str, Any],
    snippets: List[str],
    max_candidates: int = 60,
) -> Dict[str, Any]:
    """Classify top candidates into a strict table_kind enum for routing."""
    ranked = rank_candidates_for_financial_tables(candidates)[:max_candidates]
    payload = [_candidate_summary(c) for c in ranked]

    system = (
        "You are a financial filings analyst. Classify each table candidate into a strict table_kind enum.\n"
        "Definitions:\n"
        "- segment_revenue: revenue by reportable segment/business segment\n"
        "- product_service_revenue: revenue by product/service offerings or disaggregation\n"
        "- segment_results_of_operations: segment operating income/costs/expenses (NOT revenue)\n"
        "- other: anything else\n"
        "Output STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "retrieved_snippets": snippets[:8],
            "headings": scout.get("headings", [])[:30],
            "table_candidates": payload,
            "table_kind_enum": TABLE_KINDS,
            "output_schema": {
                "tables": [
                    {
                        "table_id": "tXXXX",
                        "table_kind": "one of table_kind_enum",
                        "confidence": "0..1",
                        "rationale": "short string",
                    }
                ]
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=1200)


def select_other_revenue_tables(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    candidates: List[TableCandidate],
    scout: Dict[str, Any],
    snippets: List[str],
    exclude_table_ids: Iterable[str],
    max_tables: int = 3,
    max_candidates: int = 120,
) -> Dict[str, Any]:
    ranked = rank_candidates_for_financial_tables(candidates)[:max_candidates]
    payload = [_candidate_summary(c) for c in ranked if c.table_id not in set(exclude_table_ids)]

    system = (
        "You are a financial filings analyst. Identify up to N additional REVENUE tables (not the main segments table), "
        "such as revenue by product/service offering, geography, customer type, or disaggregation. "
        "Prefer Item 8 / Notes sources when available; otherwise select the best matching revenue disclosures. "
        "Output must be STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "objective": "Find other revenue tables (product/service offerings etc.)",
            "N": max_tables,
            "headings": scout.get("headings", [])[:40],
            "retrieved_snippets": snippets[:10],
            "table_candidates": payload,
            "output_schema": {
                "tables": [
                    {
                        "table_id": "tXXXX",
                        "kind": "revenue_by_product_service | revenue_by_geography | other_revenue",
                        "confidence": "0..1",
                        "rationale": "short string",
                    }
                ]
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=900)


def extract_table_grid_normalized_with_fallback(
    html_path: Path, table_id: str, *, max_rows: int = 250
) -> List[List[str]]:
    # Wrapper in case we want to add fallbacks later (e.g., pandas.read_html)
    return extract_table_grid_normalized(html_path, table_id, max_rows=max_rows)


def infer_table_layout(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    table_id: str,
    candidate: TableCandidate,
    grid: List[List[str]],
    max_rows_for_llm: int = 30,
) -> Dict[str, Any]:
    """Ask the LLM to identify label/year columns and which rows are data."""
    preview = grid[:max_rows_for_llm]
    system = (
        "You analyze HTML tables from SEC 10-K filings. "
        "Your job: identify which column contains row labels and which columns correspond to fiscal years. "
        "Output STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "table_id": table_id,
            "candidate_summary": _candidate_summary(candidate),
            "table_grid_preview": preview,
            "output_schema": {
                "label_col": "int",
                "year_cols": {"YYYY": "int column index"},
                "header_rows": "list[int] (rows to ignore as header, from the preview)",
                "skip_row_regex": "string regex for rows to skip (e.g., totals, separators) or empty",
                "units_multiplier": "int (1, 1000, 1000000, 1000000000) inferred from units_hint if possible",
                "notes": "short string",
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=900)


def extract_revenue_rows_from_grid(
    grid: List[List[str]],
    *,
    layout: Dict[str, Any],
    target_year: Optional[int] = None,
) -> Tuple[int, Dict[str, int]]:
    """Return (year, {label -> revenue_usd_scaled}). Values are scaled by units_multiplier."""
    label_col = int(layout["label_col"])
    year_cols_raw = layout.get("year_cols") or {}
    year_cols: Dict[int, int] = {int(y): int(ci) for y, ci in year_cols_raw.items()}
    if not year_cols:
        raise ValueError("No year_cols detected")

    year = target_year or max(year_cols.keys())
    if year not in year_cols:
        year = max(year_cols.keys())
    value_col = year_cols[year]

    header_rows = set(int(i) for i in (layout.get("header_rows") or []))
    skip_row_re = layout.get("skip_row_regex") or ""
    skip_pat = re.compile(skip_row_re, re.IGNORECASE) if skip_row_re else None
    mult = int(layout.get("units_multiplier") or 1)
    if mult <= 0:
        mult = 1

    out: Dict[str, int] = {}
    for r_i, row in enumerate(grid):
        if r_i in header_rows:
            continue
        if label_col >= len(row) or value_col >= len(row):
            continue
        label = _clean(row[label_col])
        if not label:
            continue
        if skip_pat and skip_pat.search(label):
            continue
        if label.lower() in {"total", "total revenue", "revenues", "net sales"}:
            continue

        val = _parse_money_to_int(row[value_col])
        if val is None:
            continue
        out[label] = int(val) * mult

    return year, out


def summarize_segment_descriptions(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    sec_doc_url: str,
    html_text: str,
    segment_names: List[str],
    revenue_items: Optional[List[str]] = None,
    max_chars_per_segment: int = 6000,
    dimension: str = "segment",
    table_context: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Produce CSV2-style rows via LLM from extracted filing text snippets.
    
    Enhanced to:
    1. Find segment descriptions in Notes sections with better boundary detection
    2. Use revenue_items (from CSV1) as "must include" keywords for grounding
    3. Extract bounded segments (from one segment heading to the next)
    4. For product_service dimension: use table_context (caption/footnotes) instead of global search
    
    Args:
        dimension: The disclosure dimension ('segment', 'product_service', 'end_market').
                   "Other" is only skipped for dimension='segment'.
        table_context: Dict with 'caption', 'heading', 'nearby_text' from accepted table.
                       Used for product_service dimension instead of global text search.
    """
    snippets: Dict[str, str] = {}
    
    # For product_service dimension: use table context instead of global text search
    # This prevents cross-category contamination (e.g., AWS getting advertising terms)
    if dimension == "product_service" and table_context:
        # Build combined table context for product/service category definitions
        table_context_text = " ".join([
            table_context.get("caption", ""),
            table_context.get("heading", ""),
            table_context.get("nearby_text", ""),
        ])
        
        # For product_service categories, use the same snippet for all
        # (table context contains the category definitions/footnotes)
        for seg in segment_names:
            if seg.lower() in ("other", "other revenue", "corporate"):
                continue
            snippets[seg] = _clean(table_context_text) if table_context_text.strip() else ""
        
        # Skip the global search path
        filtered_segments = [s for s in segment_names if s.lower() not in ("other", "other revenue", "corporate")]
        
    else:
        # For segment/end_market dimensions: use global search (existing logic)
        t = html_text
        low = t.lower()
        
        # Find all segment boundary positions for bounded extraction
        segment_positions: Dict[str, List[int]] = {}
        for seg in segment_names:
            seg_low = seg.lower()
            positions = []
            idx = 0
            while True:
                found = low.find(seg_low, idx)
                if found == -1:
                    break
                positions.append(found)
                idx = found + 1
            segment_positions[seg] = positions
        
        # Key section markers
        segment_info_patterns = [
            "segment information",
            "reportable segments",
            "note 18",
            "note 17",
            "segment results",
        ]
        segment_info_idx = -1
        for pattern in segment_info_patterns:
            idx = low.find(pattern)
            if idx >= 0:
                segment_info_idx = idx
                break
        
        # Notes section
        notes_idx = low.find("notes to consolidated financial statements")
        if notes_idx == -1:
            notes_idx = low.find("notes to financial statements")
        
        # Item 1 Business section
        item1_idx = low.find("item 1")
        item1_business_idx = low.find("item 1.", item1_idx) if item1_idx >= 0 else -1
        
        for seg in segment_names:
            # Only skip "Other" for segment dimensions (residual catch-all)
            # Include "Other" for product_service/end_market (explicit revenue line)
            if dimension == "segment" and seg.lower() in ("other", "other revenue", "corporate"):
                continue
                
            key = seg
            seg_low = seg.lower()
            positions = segment_positions.get(seg, [])
            
            if not positions:
                snippets[key] = ""
                continue
            
            # Find the best occurrence using priority:
            # 1. In segment info section (Note 17/18)
            # 2. In Notes section after segment_info_idx
            # 3. In Item 1 Business section
            # 4. First occurrence
            
            best_idx = -1
            
            # Priority 1: In segment info section
            if segment_info_idx >= 0:
                for pos in positions:
                    if pos >= segment_info_idx and pos < segment_info_idx + 100000:
                        best_idx = pos
                        break
            
            # Priority 2: In Notes section
            if best_idx == -1 and notes_idx >= 0:
                for pos in positions:
                    if pos >= notes_idx:
                        best_idx = pos
                        break
            
            # Priority 3: In Item 1
            if best_idx == -1 and item1_business_idx >= 0:
                for pos in positions:
                    if pos >= item1_business_idx:
                        best_idx = pos
                        break
            
            # Priority 4: First occurrence
            if best_idx == -1 and positions:
                best_idx = positions[0]
            
            if best_idx == -1:
                snippets[key] = ""
                continue
            
            # Find the end boundary (next segment heading or max chars)
            end_idx = best_idx + max_chars_per_segment
            other_seg_names = [s for s in segment_names if s.lower() != seg_low and s.lower() not in ("other", "corporate")]
            for other_seg in other_seg_names:
                other_pos = low.find(other_seg.lower(), best_idx + len(seg))
                if other_pos > best_idx and other_pos < end_idx:
                    # Found next segment - use as boundary but include some padding
                    end_idx = min(end_idx, other_pos + 200)
            
            # Extract bounded snippet
            start = max(0, best_idx - 200)
            end = min(len(t), end_idx)
            snippets[key] = _clean(t[start:end])
        
        # Filter out "Other" segments for the else path
        filtered_segments = [s for s in segment_names if s.lower() not in ("other", "other revenue", "corporate")]
    
    system = (
        "You summarize company business segments from SEC 10-K text. "
        "CRITICAL RULES:\n"
        "1. For each segment, write a description GROUNDED in the provided text snippet.\n"
        "2. List SPECIFIC product/brand names that appear in the text "
        "(e.g., 'Azure', 'Microsoft 365 Commercial', 'LinkedIn', 'YouTube ads').\n"
        "3. The 'revenue_items_from_filing' field contains ACTUAL revenue line items from the 10-K. "
        "Map these to the correct segment and include them in key_products_services.\n"
        "4. Do NOT invent products not mentioned in the text or revenue items.\n"
        "Output STRICT JSON ONLY."
    )
    
    # Build segment data with revenue items mapping hint
    segment_data = []
    for s in filtered_segments:
        item_data = {
            "segment": s,
            "text_snippet": snippets.get(s, ""),
        }
        # Add revenue items hint if provided
        if revenue_items:
            item_data["revenue_items_from_filing"] = revenue_items
        segment_data.append(item_data)
    
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "sec_doc_url": sec_doc_url,
            "segments": segment_data,
            "output_schema": {
                "rows": [
                    {
                        "segment": "string",
                        "segment_description": "string (comprehensive, 2-3 sentences, grounded in text)",
                        "key_products_services": "list[string] (specific brand/product names from text and revenue_items)",
                        "primary_source": "string short",
                    }
                ]
            },
        },
        ensure_ascii=False,
    )
    result = llm.json_call(system=system, user=user, max_output_tokens=2000)
    
    # Enrich LLM output rows with original text snippets for downstream validation
    rows = result.get("rows", [])
    for row in rows:
        seg_name = row.get("segment", "")
        row["text_snippet"] = snippets.get(seg_name, "")
    
    return result


def expand_key_items_per_segment(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    sec_doc_url: str,
    segment_rows: List[Dict[str, Any]],
    html_text: str = "",
    dimension: str = "segment",
) -> Dict[str, Any]:
    """Produce CSV3 rows: key items per segment with short + long description.
    
    EVIDENCE-BASED EXTRACTION:
    - Only extract items that appear verbatim in the provided text
    - Each item must include an evidence_span copied from the source
    - Post-validate that evidence_span exists in SEGMENT-SPECIFIC snippet (not full HTML)
    
    Process each segment individually to prevent token truncation and cross-segment leakage.
    
    Args:
        dimension: The disclosure dimension ('segment', 'product_service', 'end_market').
                   "Other" is only skipped for dimension='segment'.
    """
    all_rows: List[Dict[str, Any]] = []
    seen_items: set = set()  # For de-duplication
    html_text_lower = html_text.lower() if html_text else ""
    
    for seg_row in segment_rows:
        segment_name = seg_row.get("segment", "Unknown")
        
        # Only skip "Other" for segment dimensions (residual catch-all)
        # Include "Other" for product_service/end_market (explicit revenue line)
        if dimension == "segment" and segment_name.lower() in ("other", "other revenue", "corporate"):
            continue
        
        segment_description = seg_row.get("segment_description", "")
        key_products = seg_row.get("key_products_services", [])
        
        # Use segment-specific snippet for evidence validation (prevents cross-segment leakage)
        segment_snippet = seg_row.get("text_snippet", "")
        segment_snippet_lower = segment_snippet.lower() if segment_snippet else ""
        
        system = (
            "You are an EXTRACTIVE information retrieval system. You ONLY output items that are "
            "EXPLICITLY NAMED as products, services, or brands in the provided text.\n\n"
            "STRICT RULES - VIOLATIONS WILL BE REJECTED:\n"
            "1. ONLY output items whose EXACT NAME appears VERBATIM in the text.\n"
            "2. 'evidence_span' MUST be a WORD-FOR-WORD quote from the text (15-40 words) that includes the item name.\n"
            "3. Do NOT infer, generalize, or add items based on your knowledge. If it's not in the text, don't output it.\n"
            "4. Do NOT output generic categories (e.g., 'cloud services', 'advertising') unless that EXACT phrase names a distinct product.\n"
            "5. If only 1-3 items are explicitly named, output only those 1-3 items. Empty output is valid.\n"
            "6. Descriptions must summarize ONLY what the text says, not general knowledge.\n\n"
            "Output STRICT JSON ONLY."
        )
        # Use text_snippet as primary source (raw filing text), not segment_description (LLM output)
        user = json.dumps(
            {
                "ticker": ticker,
                "company_name": company_name,
                "segment": segment_name,
                "source_text": segment_snippet if segment_snippet else segment_description,
                "key_products_hint": key_products,
                "instructions": (
                    "EXTRACT items from 'source_text' ONLY. "
                    "For each item, the evidence_span must be copy-pasted VERBATIM from source_text "
                    "and must contain the item name. "
                    "If you cannot find verbatim evidence for an item in source_text, DO NOT include it. "
                    "key_products_hint is for reference only - do not include items not in source_text."
                ),
                "output_schema": {
                    "rows": [
                        {
                            "segment": "string (must match input segment name exactly)",
                            "business_item": "string (EXACT product/brand name as it appears in source_text)",
                            "business_item_short_description": "string (1 sentence from source_text)",
                            "business_item_long_description": "string (2-3 sentences from source_text)",
                            "evidence_span": "string (VERBATIM quote from source_text, 15-40 words, containing the item name)",
                        }
                    ]
                },
            },
            ensure_ascii=False,
        )
        try:
            result = llm.json_call(system=system, user=user, max_output_tokens=1800)
            rows = result.get("rows", [])
            
            for row in rows:
                row["segment"] = segment_name
                item_name = row.get("business_item", "").strip()
                evidence = row.get("evidence_span", "").strip()
                
                # De-duplication check
                item_key = item_name.lower().replace(" ", "").replace("-", "")
                if item_key in seen_items:
                    continue
                
                # SEGMENT-SCOPED Evidence validation (prevents cross-segment leakage)
                # Validate against segment snippet first, fall back to full HTML if no snippet
                validation_text = segment_snippet_lower if segment_snippet_lower else html_text_lower
                
                evidence_found = False
                item_in_text = item_name.lower() in validation_text if item_name else False
                item_in_evidence = item_name.lower() in evidence.lower() if (item_name and evidence) else False
                
                if evidence and validation_text:
                    # Normalize for matching (remove extra spaces, lowercase)
                    evidence_normalized = " ".join(evidence.lower().split())
                    
                    # Check if evidence span exists in segment snippet
                    if evidence_normalized in validation_text:
                        evidence_found = True
                    else:
                        # Stricter fuzzy match: 85% of significant words must appear
                        evidence_words = [w for w in evidence_normalized.split() if len(w) > 3]
                        if len(evidence_words) >= 4:
                            matches = sum(1 for w in evidence_words if w in validation_text)
                            if matches / len(evidence_words) >= 0.85:
                                evidence_found = True
                
                # Strict acceptance criteria:
                # 1. Item name must appear in segment snippet, AND
                # 2. Either evidence is validated OR evidence contains the item name
                accept_item = item_in_text and (evidence_found or item_in_evidence)
                
                if accept_item or not validation_text:
                    row["evidence_validated"] = evidence_found
                    row["item_in_source"] = item_in_text
                    row["item_in_evidence"] = item_in_evidence
                    seen_items.add(item_key)
                    all_rows.append(row)
                else:
                    # Reject items that fail segment-scoped validation
                    reject_reason = []
                    if not item_in_text:
                        reject_reason.append("item_not_in_segment_text")
                    if not evidence_found:
                        reject_reason.append("evidence_not_in_segment")
                    if not item_in_evidence:
                        reject_reason.append("item_not_in_evidence")
                    print(f"[{ticker}] Rejected item '{item_name}' for segment '{segment_name}' - {', '.join(reject_reason)}", flush=True)
                    
        except Exception as e:
            print(f"[{ticker}] Warning: expand_key_items failed for segment '{segment_name}': {e}", flush=True)
            continue
    
    return {"rows": all_rows}


def describe_revenue_lines(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    fiscal_year: int,
    revenue_lines: List[Dict[str, Any]],
    table_context: Dict[str, str],
    html_text: str,
    html_raw: Optional[str] = None,
    max_chars_per_line: int = 5000,
    footnote_id_map: Optional[Dict[str, List[str]]] = None,
    dom_footnotes: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Generate company-language descriptions for each revenue line.
    
    Enhanced with:
    - DOM-based footnote ID recovery (handles iXBRL superscripts)
    - Phase 2: DOM-based footnote extraction using normalized text
    - Phase 3: Raw HTML for heading extraction (html_text is plain text!)
    - Section-aware priority search (Item 1 > Item 8; MD&A excluded)
    - Accounting/driver sentence filtering
    
    Priority for evidence:
    1. DOM-extracted footnotes (highest priority - most reliable for AMZN-style)
    2. Table footnotes from regex (fallback)
    3. Heading-based extraction in Item 1 (AAPL-style product descriptions)
    4. Notes to Financial Statements (Item 8, if definitional)
    5. Full text fallback
    
    NOTE: MD&A (Item 7) is intentionally excluded - it contains performance
    drivers ("increased due to...") not product definitions.
    
    Args:
        revenue_lines: List of dicts with 'item' (revenue line label) and 'value'
        table_context: Dict with 'caption', 'heading', 'nearby_text' from accepted table
        html_text: Full filing text for evidence retrieval (plain text, tags removed)
        html_raw: Raw HTML for heading extraction (Phase 3 - keeps tags for pattern matching)
        footnote_id_map: Optional dict from DOM extraction mapping labels to footnote IDs
        dom_footnotes: Optional dict from Phase 2 DOM-based extraction mapping footnote IDs to definitions
        
    Returns:
        Dict with 'rows' containing line descriptions
    """
    if not revenue_lines:
        return {"rows": []}
    
    # Build combined table context
    table_context_text = " ".join([
        table_context.get("caption", ""),
        table_context.get("heading", ""),
        table_context.get("nearby_text", ""),
    ])
    
    # Initialize optional parameters
    if footnote_id_map is None:
        footnote_id_map = {}
    if dom_footnotes is None:
        dom_footnotes = {}
    
    # Phase 4: Track provenance for each description
    # Structure: {item_label: {"description": str, "source": str, "evidence_snippet": str, "footnote_id": str|None}}
    provenance: Dict[str, Dict[str, Any]] = {}
    
    # STEP 1: Try to extract footnote definitions
    # Priority: DOM-extracted > regex table-local > regex full-text
    footnote_descriptions: Dict[str, str] = {}
    
    # Phase 2: Use DOM-extracted footnotes as highest priority
    # These are extracted using normalized text (get_text()) which handles split tags
    all_footnotes = dict(dom_footnotes)  # Start with DOM footnotes
    
    # Fallback: Also extract via regex (less reliable but catches some cases)
    regex_footnotes = _extract_footnotes_from_text(html_text)
    table_context_footnotes = _extract_footnotes_from_text(table_context_text)
    
    # Merge, preferring DOM > table-context > full-text
    for fn_id, fn_text in regex_footnotes.items():
        if fn_id not in all_footnotes:
            all_footnotes[fn_id] = fn_text
    for fn_id, fn_text in table_context_footnotes.items():
        if fn_id not in all_footnotes or len(fn_text) > len(all_footnotes.get(fn_id, "")):
            all_footnotes[fn_id] = fn_text
    
    for line_info in revenue_lines:
        item_label = line_info.get("item", "")
        if not item_label:
            continue
        
        # Clean label for lookup (remove existing markers)
        label_clean = _FOOTNOTE_MARKER_RE.sub('', item_label).strip()
        
        # Approach A: Check DOM-extracted footnote IDs (handles iXBRL superscripts)
        fn_ids = footnote_id_map.get(label_clean, [])
        if not fn_ids:
            fn_ids = footnote_id_map.get(item_label, [])
        
        for fn_id in fn_ids:
            if fn_id in all_footnotes:
                desc = strip_accounting_sentences(all_footnotes[fn_id])
                if desc and len(desc) >= 20:
                    footnote_descriptions[item_label] = desc
                    # Phase 4: Record provenance
                    provenance[item_label] = {
                        "description": desc,
                        "source": "table_footnote_dom",
                        "evidence_snippet": all_footnotes[fn_id][:500],
                        "footnote_id": fn_id,
                    }
                    break
        
        # Approach B: Fall back to regex detection if not found via DOM
        if item_label not in footnote_descriptions:
            footnote_desc = _extract_footnote_for_label(item_label, html_text, table_context_text)
            if footnote_desc:
                desc = strip_accounting_sentences(footnote_desc)
                if desc and len(desc) >= 20:
                    footnote_descriptions[item_label] = desc
                    # Phase 4: Record provenance
                    fn_match = _FOOTNOTE_MARKER_RE.search(item_label)
                    provenance[item_label] = {
                        "description": desc,
                        "source": "table_footnote_regex",
                        "evidence_snippet": footnote_desc[:500],
                        "footnote_id": fn_match.group(1) if fn_match else None,
                    }
    
    # Lines that still need description extraction (not found in footnotes)
    lines_needing_more = [
        line_info for line_info in revenue_lines
        if line_info.get("item", "") not in footnote_descriptions
    ]
    
    # If we got footnotes for all lines, skip further extraction
    if not lines_needing_more:
        return {
            "rows": [
                {"revenue_line": line_info.get("item", ""), "description": footnote_descriptions.get(line_info.get("item", ""), "")}
                for line_info in revenue_lines
            ]
        }
    
    # STEP 1.5: Heading-based extraction for AAPL Services-style labels
    # This handles cases where the label is a heading in Item 1 Business
    # (e.g., <b>Services</b> followed by subsection descriptions)
    # Phase 3 fix: Use html_raw (raw HTML with tags) for heading extraction
    #              html_text is plain text with tags removed - can't match heading patterns!
    search_html = html_raw if html_raw else html_text
    item1_section = _extract_section(search_html, _ITEM1_RE, max_chars=80000)
    
    heading_descriptions: Dict[str, str] = {}
    for line_info in lines_needing_more:
        item_label = line_info.get("item", "")
        if not item_label:
            continue
        
        # Clean label (remove footnote markers like "(1)")
        label_clean = _FOOTNOTE_MARKER_RE.sub('', item_label).strip()
        
        # Try heading-based extraction in Item 1 (most reliable for product definitions)
        # First try with section, then fallback to full HTML if not found
        heading_desc = _extract_heading_based_definition(search_html, label_clean, item1_section)
        source_section = "item1" if item1_section and heading_desc else None
        
        # If not found in section, try full HTML (handles styled spans outside Item 1)
        if not heading_desc:
            heading_desc = _extract_heading_based_definition(search_html, label_clean, None)
            source_section = "html_heading" if heading_desc else None
        
        if heading_desc and len(heading_desc) >= 50:
            heading_descriptions[item_label] = heading_desc
            # Phase 4: Record provenance for heading-based extraction
            provenance[item_label] = {
                "description": heading_desc,
                "source": f"heading_based_{source_section}" if source_section else "heading_based",
                "evidence_snippet": heading_desc[:500],
                "footnote_id": None,
            }
            print(f"[heading-based] Found definition for '{label_clean}': {heading_desc[:80]}...", flush=True)
        else:
            # Phase 5a: Try "offerings include" pattern first (for NVDA-style filings)
            # This produces richer definitions than segment enumerations
            offerings_pattern = re.compile(
                r'(?:our\s+)?' + re.escape(label_clean) + r'\s+offerings?\s+include[s]?\s+'
                r'([^.]{30,400}\.)',
                re.IGNORECASE
            )
            offerings_match = offerings_pattern.search(html_text)
            if offerings_match:
                offerings_desc = f"Our {label_clean.lower()} offerings include {offerings_match.group(1).strip()}"
                offerings_desc = strip_accounting_sentences(offerings_desc)
                if offerings_desc and len(offerings_desc) >= 40:
                    heading_descriptions[item_label] = offerings_desc
                    provenance[item_label] = {
                        "description": offerings_desc,
                        "source": "offerings_pattern",
                        "evidence_snippet": offerings_desc[:500],
                        "footnote_id": None,
                    }
                    print(f"[offerings-pattern] Found definition for '{label_clean}': {offerings_desc[:80]}...", flush=True)
                    continue  # Skip note2 and other fallbacks
            
            # Phase 5b: Try Note 2 paragraph extraction (for META-style filings)
            note2_desc = _extract_note2_paragraph_definition(html_text, label_clean)
            if note2_desc and len(note2_desc) >= 30:
                heading_descriptions[item_label] = note2_desc
                # Phase 4: Record provenance for Note 2 extraction
                provenance[item_label] = {
                    "description": note2_desc,
                    "source": "note2_paragraph",
                    "evidence_snippet": note2_desc[:500],
                    "footnote_id": None,
                }
                print(f"[note2-paragraph] Found definition for '{label_clean}': {note2_desc[:80]}...", flush=True)
    
    # Merge: footnotes take priority, then headings/note2
    for item, desc in heading_descriptions.items():
        if item not in footnote_descriptions:
            footnote_descriptions[item] = desc
    
    # Lines that still need LLM-based description extraction
    lines_needing_llm = [
        line_info for line_info in revenue_lines
        if line_info.get("item", "") not in footnote_descriptions
    ]
    
    # If we now have all descriptions, skip LLM
    if not lines_needing_llm:
        return {
            "rows": [
                {"revenue_line": line_info.get("item", ""), "description": footnote_descriptions.get(line_info.get("item", ""), "")}
                for line_info in revenue_lines
            ],
            "provenance": provenance,  # Phase 4: Include provenance
        }
    
    # STEP 2: Section-aware search for remaining lines
    # Pre-extract major sections for priority search
    # NOTE: Item 7 (MD&A) is intentionally EXCLUDED - it contains performance
    # drivers ("increased/decreased due to...") not product definitions.
    # item1_section already extracted above
    item8_section = _extract_section(html_text, _ITEM8_RE, max_chars=80000)
    
    # Priority order: Item 1 (Business) → Item 8 (Notes) → Full text
    # MD&A (Item 7) excluded as it contains drivers, not definitions
    sections_priority = [
        ("item1", item1_section),
        ("item8", item8_section),
        ("full", html_text),
    ]
    
    evidence_by_line: Dict[str, str] = {}
    
    for line_info in lines_needing_llm:
        item_label = line_info.get("item", "")
        if not item_label:
            continue
        
        # Expand search terms for this label
        search_terms = _expand_search_terms(item_label)
        
        snippets = []
        
        # 1. Always include table context first (footnotes often have descriptions)
        if table_context_text:
            snippets.append(f"[TABLE CONTEXT] {table_context_text[:2000]}")
        
        # 2. Search through sections in priority order
        for section_name, section_text in sections_priority:
            if not section_text:
                continue
            
            section_low = section_text.lower()
            
            # Try each search term
            for term in search_terms:
                idx = section_low.find(term)
                if idx != -1:
                    # Found a match - extract larger window
                    start = max(0, idx - 800)
                    end = min(len(section_text), idx + 2500)
                    window = section_text[start:end]
                    snippet = f"[{section_name.upper()}] {_clean(window)}"
                    
                    # Avoid duplicates
                    if snippet not in snippets:
                        snippets.append(snippet)
                    
                    # Found in this section, try to get more context
                    # Look for additional matches in this section
                    next_idx = section_low.find(term, idx + len(term) + 500)
                    if next_idx != -1 and len(snippets) < 5:
                        start2 = max(0, next_idx - 500)
                        end2 = min(len(section_text), next_idx + 2000)
                        window2 = section_text[start2:end2]
                        snippet2 = f"[{section_name.upper()}] {_clean(window2)}"
                        if snippet2 not in snippets:
                            snippets.append(snippet2)
                    
                    break  # Found in this section, move to next section
            
            # Stop if we have enough snippets
            if len(snippets) >= 4:
                break
        
        # Combine snippets for this line
        combined = " [...] ".join(snippets)[:max_chars_per_line]
        evidence_by_line[item_label] = combined
    
    # Build LLM prompt for lines that need LLM extraction
    llm_descriptions: Dict[str, str] = {}
    
    if lines_needing_llm and evidence_by_line:
        system = (
            "You are extracting product/service DEFINITIONS from SEC 10-K filings.\n\n"
            "CRITICAL RULES:\n"
            "1. Describe WHAT the product/service IS, not how it performed.\n"
            "2. Use company language from the evidence text.\n"
            "3. Each description should be 1-2 sentences explaining what the revenue line includes.\n"
            "4. PREFER definitions that look like:\n"
            "   - 'X is the Company's line of...'\n"
            "   - 'Includes...'\n"
            "   - 'Provides...services such as...'\n"
            "5. EXCLUDE the following - return empty string if only this type of text is found:\n"
            "   - Accounting language: 'recognized', 'deferred', 'amortization', 'performance obligation'\n"
            "   - Performance drivers: 'increased due to', 'decreased due to', 'primarily driven by'\n"
            "   - YoY comparisons: 'compared to prior year', 'from a year ago'\n"
            "6. If no product/service definition is found, return an empty string.\n"
            "7. Do NOT invent or infer descriptions.\n\n"
            "Output STRICT JSON ONLY."
        )
        
        lines_data = []
        for line_info in lines_needing_llm:
            item = line_info.get("item", "")
            lines_data.append({
                "revenue_line": item,
                "evidence_text": evidence_by_line.get(item, "")[:4000],
            })
        
        user = json.dumps(
            {
                "ticker": ticker,
                "company_name": company_name,
                "fiscal_year": fiscal_year,
                "revenue_lines": lines_data,
                "instructions": (
                    "For each revenue_line, extract a PRODUCT/SERVICE DEFINITION from evidence_text. "
                    "Priority: [TABLE CONTEXT] for footnote definitions, [ITEM1] for business descriptions. "
                    "Focus on WHAT the product/service IS and WHAT IT INCLUDES. "
                    "EXCLUDE performance commentary ('increased due to...') and accounting language. "
                    "If only performance/accounting text is found, return empty string."
                ),
                "output_schema": {
                    "rows": [
                        {
                            "revenue_line": "string (exact match from input)",
                            "description": "string (1-2 sentences in company language, or empty)",
                        }
                    ]
                },
            },
            ensure_ascii=False,
        )
        
        try:
            result = llm.json_call(system=system, user=user, max_output_tokens=2500)
            rows = result.get("rows", [])
            
            for row in rows:
                line = row.get("revenue_line", "")
                desc = row.get("description", "")
                if line:
                    # Apply accounting sentence filter to LLM output
                    filtered_desc = strip_accounting_sentences(desc)
                    llm_descriptions[line] = filtered_desc
                    
                    # Phase 4: Record provenance for LLM-extracted descriptions
                    if filtered_desc and line not in provenance:
                        evidence = evidence_by_line.get(line, "")
                        # Determine source section from evidence markers
                        source = "llm_section_search"
                        if "[ITEM1]" in evidence:
                            source = "llm_item1"
                        elif "[ITEM8]" in evidence:
                            source = "llm_item8"
                        elif "[TABLE CONTEXT]" in evidence:
                            source = "llm_table_context"
                        
                        provenance[line] = {
                            "description": filtered_desc,
                            "source": source,
                            "evidence_snippet": evidence[:500],
                            "footnote_id": None,
                        }
                
        except Exception as e:
            print(f"[{ticker}] Warning: describe_revenue_lines LLM call failed: {e}", flush=True)
    
    # STEP 3: Merge footnote descriptions with LLM descriptions
    # Footnote descriptions take priority (they are direct quotes from the filing)
    # Both have already been filtered by strip_accounting_sentences()
    # Phase 6: Add segment enumeration fallback for NVDA-style filings
    output_rows = []
    for line_info in revenue_lines:
        item = line_info.get("item", "")
        revenue_group = line_info.get("revenue_group", "")
        
        # Priority: footnote > LLM > segment_enumeration
        description = footnote_descriptions.get(item, "") or llm_descriptions.get(item, "")
        
        # Phase 6: Fallback to segment enumeration extraction
        # For generic labels like "Compute" that exist only in segment narratives
        if not description and revenue_group:
            try:
                # Use html_raw if available (has HTML structure for pattern matching)
                search_html = html_raw if html_raw else html_text
                enum_desc = _extract_from_segment_enumeration(search_html, item, revenue_group)
                if enum_desc:
                    description = enum_desc
                    # Record provenance for segment enumeration extraction
                    if item not in provenance:
                        provenance[item] = {
                            "description": enum_desc,
                            "source": "segment_enumeration",
                            "evidence_snippet": f"Extracted from '{revenue_group}' segment definition",
                            "footnote_id": None,
                        }
                    print(f"[{ticker}] Phase 6: Extracted description for '{item}' from segment enumeration", flush=True)
            except Exception as e:
                print(f"[{ticker}] Warning: segment enumeration extraction failed for '{item}': {e}", flush=True)
        
        output_rows.append({
            "revenue_line": item,
            "description": description,
        })
    
    return {
        "rows": output_rows,
        "provenance": provenance,  # Phase 4: Include provenance
    }

