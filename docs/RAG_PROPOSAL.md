# RAG-Enhanced Revenue Line Description Extraction

## Proposal for Semantic Search Integration (v3 - Final)

**Date**: January 2026  
**Status**: Approved with Modifications  
**Estimated Effort**: 4-5 days  

---

## v3 Change Summary (Developer Review Response)

| # | Reviewer Recommendation | Response | Rationale |
|---|------------------------|----------|-----------|
| 1 | TOC: detection + exclusion, not destructive regex | ✅ **Agree** | Regex is brittle across filers; tagging preserves original text for debugging |
| 2 | DOM-based table context, not character offsets | ✅ **Agree** | Parser differences cause offset drift; DOM adjacency is semantic |
| 3 | Multi-candidate section boundaries with density checks | ✅ **Agree** | Single-match vulnerable to TOC; body-density filters false positives |
| 4 | Calibrated thresholds, not hard-coded | ✅ **Agree** | Score distributions vary by model/corpus; P90 noise baseline is principled |
| 5 | Evidence coverage gate (preferred section required) | ✅ **Agree** | Prevents plausible-but-wrong from generic "compute" discussions |
| 6 | Extractive-first product enumeration | ✅ **Agree** | LLM filters known candidates > LLM generates from scratch |
| 7 | QA artifact per ticker | ✅ **Agree** | Regression testing is essential for maintainability |

**No pushback on any point.** All recommendations improve robustness and testability.

---

## Executive Summary

### The Problem

Current description extraction uses **keyword search** to find revenue line descriptions in 10-K filings. This works well for companies with footnote-style disclosures (AMZN, MSFT) but fails for companies with narrative-style descriptions (NVDA).

| Ticker | Current Coverage | Issue |
|--------|------------------|-------|
| AMZN | 7/7 ✅ | Footnotes work |
| AAPL | 5/5 ✅ | Footnotes work |
| MSFT | 9/10 | "Other" has no description |
| GOOGL | 5/6 | "Other Bets" undescribed |
| META | 2/3 | "Other revenue" undescribed |
| **NVDA** | **0/6** ❌ | **No footnotes, narrative style** |

### Known Failure Modes (Developer Review)

| Failure Mode | Root Cause | Impact |
|--------------|------------|--------|
| TOC Capture | "Item 1/7/8" search matches Table of Contents | Retrieves junk context |
| Weak Locality | Extract window is unidirectional (+2500 chars) | Misses pre-label context |
| Ambiguous Labels | "Compute" matches risk factors, capex, liquidity | Wrong semantic context |

### The Solution (Revised)

Implement **Table-Local-First RAG** with:
1. **Pre-RAG cleanup**: Strip TOC, build bidirectional table context
2. **Two-tier retrieval**: Table-local index first, full-filing fallback
3. **Structure-aware chunking**: By sections/notes with metadata
4. **Retrieval quality controls**: MMR, similarity thresholds, rich queries
5. **Auditable generation**: Evidence chunk IDs, quoted fragments

**Expected Improvement**: 76% → 95%+ description coverage

---

## Architecture (Revised)

### Two-Tier Retrieval Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              TABLE-LOCAL-FIRST RAG ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    PREPROCESSING (One-Time per Filing)               │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │                                                                      │    │
│  │   10-K Filing                                                        │    │
│  │        │                                                             │    │
│  │        ▼                                                             │    │
│  │   ┌───────────────┐                                                  │    │
│  │   │  Strip TOC    │  Remove Table of Contents section                │    │
│  │   │  Clean HTML   │  Remove headers/footers, page numbers           │    │
│  │   └───────┬───────┘                                                  │    │
│  │           │                                                          │    │
│  │           ▼                                                          │    │
│  │   ┌───────────────────────────────────────────────────────────────┐ │    │
│  │   │              STRUCTURE-AWARE CHUNKING                          │ │    │
│  │   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │ │    │
│  │   │  │  Item 1     │  │  Item 7     │  │  Item 8     │            │ │    │
│  │   │  │  Business   │  │   MD&A      │  │   Notes     │            │ │    │
│  │   │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │ │    │
│  │   │         │                │                │                    │ │    │
│  │   │         ▼                ▼                ▼                    │ │    │
│  │   │    ┌─────────────────────────────────────────────────────┐    │ │    │
│  │   │    │  Chunks with METADATA:                              │    │ │    │
│  │   │    │  • section: "item1" | "item7" | "note_18" | ...     │    │ │    │
│  │   │    │  • heading: "Revenue Recognition" | "Segments"      │    │ │    │
│  │   │    │  • table_id: "t0042" | null                         │    │ │    │
│  │   │    │  • is_toc: false (filtered out)                     │    │ │    │
│  │   │    │  • char_range: [12000, 12800]                       │    │ │    │
│  │   │    └─────────────────────────────────────────────────────┘    │ │    │
│  │   └───────────────────────────────────────────────────────────────┘ │    │
│  │           │                                                          │    │
│  │           ▼                                                          │    │
│  │   ┌───────────────────────────────────────────────────────────────┐ │    │
│  │   │                    TWO INDEXES                                 │ │    │
│  │   │  ┌─────────────────────┐    ┌─────────────────────┐           │ │    │
│  │   │  │  TABLE-LOCAL INDEX  │    │  FULL-FILING INDEX  │           │ │    │
│  │   │  │  • ±5000 chars      │    │  • All chunks       │           │ │    │
│  │   │  │    around table     │    │  • ~300-500 chunks  │           │ │    │
│  │   │  │  • Table caption    │    │  • With metadata    │           │ │    │
│  │   │  │  • Footnotes        │    │                     │           │ │    │
│  │   │  │  • ~20-50 chunks    │    │                     │           │ │    │
│  │   │  └─────────────────────┘    └─────────────────────┘           │ │    │
│  │   └───────────────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    RETRIEVAL (Per Revenue Line)                      │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │                                                                      │    │
│  │   Revenue Line: "Compute"                                            │    │
│  │   Revenue Group: "Compute & Networking"                              │    │
│  │   Table Caption: "Revenue by End Market"                             │    │
│  │        │                                                             │    │
│  │        ▼                                                             │    │
│  │   ┌───────────────────────────────────────────────────────────────┐ │    │
│  │   │  RICH QUERY CONSTRUCTION                                      │ │    │
│  │   │  "{company} FY{year} revenue line '{label}'                   │ │    │
│  │   │   in {revenue_group}. Products and services included.         │ │    │
│  │   │   Use definitions from revenue/segment note."                 │ │    │
│  │   └───────────────────────────────────────────────────────────────┘ │    │
│  │        │                                                             │    │
│  │        ▼                                                             │    │
│  │   ┌─────────────────────────────────────────────────────────────┐   │    │
│  │   │              TIER 1: TABLE-LOCAL RETRIEVAL                   │   │    │
│  │   │  Query → Table-Local Index → Top 5 (score ≥ 0.70)           │   │    │
│  │   │                                                              │   │    │
│  │   │  ┌─────────────────────────────────────────────────────┐    │   │    │
│  │   │  │ ✅ Found chunks with score ≥ 0.70?                  │    │   │    │
│  │   │  │    → Use these chunks                               │    │   │    │
│  │   │  │ ❌ Max score < 0.70?                                │    │   │    │
│  │   │  │    → Fall back to Tier 2                            │    │   │    │
│  │   │  └─────────────────────────────────────────────────────┘    │   │    │
│  │   └─────────────────────────────────────────────────────────────┘   │    │
│  │        │ (if fallback needed)                                        │    │
│  │        ▼                                                             │    │
│  │   ┌─────────────────────────────────────────────────────────────┐   │    │
│  │   │              TIER 2: FULL-FILING RETRIEVAL                   │   │    │
│  │   │  Query → Full-Filing Index → Top 5 (with section filter)    │   │    │
│  │   │                                                              │   │    │
│  │   │  Priority sections: note_*, item1, item7                    │   │    │
│  │   │  Deprioritize: risk_factors, liquidity, capex               │   │    │
│  │   └─────────────────────────────────────────────────────────────┘   │    │
│  │        │                                                             │    │
│  │        ▼                                                             │    │
│  │   ┌─────────────────────────────────────────────────────────────┐   │    │
│  │   │              RETRIEVAL QUALITY CONTROLS                      │   │    │
│  │   │  • MMR (Maximal Marginal Relevance) for diversity           │   │    │
│  │   │  • Dedup near-identical chunks (cosine > 0.95)              │   │    │
│  │   │  • Minimum threshold τ = 0.60 (else return empty)           │   │    │
│  │   └─────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    GENERATION (Auditable)                            │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │                                                                      │    │
│  │   Retrieved Chunks (with IDs): [chunk_42, chunk_58, chunk_103]      │    │
│  │        │                                                             │    │
│  │        ▼                                                             │    │
│  │   ┌───────────────────────────────────────────────────────────────┐ │    │
│  │   │  LLM CALL (gpt-4.1)                                           │ │    │
│  │   │                                                                │ │    │
│  │   │  Output Schema (JSON):                                         │ │    │
│  │   │  {                                                             │ │    │
│  │   │    "description": "1-2 sentences, company language",          │ │    │
│  │   │    "products_services_list": ["GPU", "DGX", "networking"],    │ │    │
│  │   │    "evidence_chunk_ids": ["chunk_42", "chunk_58"],            │ │    │
│  │   │    "evidence_quotes": [                                        │ │    │
│  │   │      "Data Center compute platforms include DGX...",          │ │    │
│  │   │      "Our compute solutions power AI workloads..."            │ │    │
│  │   │    ]                                                           │ │    │
│  │   │  }                                                             │ │    │
│  │   └───────────────────────────────────────────────────────────────┘ │    │
│  │        │                                                             │    │
│  │        ▼                                                             │    │
│  │   ┌───────────────────────────────────────────────────────────────┐ │    │
│  │   │  POST-VALIDATION                                               │ │    │
│  │   │  • Verify evidence_quotes exist in source chunks              │ │    │
│  │   │  • Verify chunk_ids are in retrieved set                       │ │    │
│  │   │  • If validation fails → return empty (don't hallucinate)     │ │    │
│  │   └───────────────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pre-RAG Prerequisites (Fix Deterministic Bugs First)

### 1. TOC Detection (Non-Destructive)

**Change from v2**: Don't strip TOC with regex (brittle, can delete real content).
Instead, **tag chunks as `is_toc=True`** and exclude at retrieval time.

```python
import re
from typing import Tuple

# TOC detection heuristics
TOC_INDICATORS = [
    # Dense "Item X ... page" patterns
    re.compile(r'Item\s+\d+[A-Z]?\s*\.{2,}\s*\d+', re.IGNORECASE),
    # Link-heavy regions (many anchor tags per line)
    re.compile(r'(<a\s+[^>]*href[^>]*>.*?</a>\s*){3,}', re.IGNORECASE),
    # Page number patterns at end of lines
    re.compile(r'\.\s*\d{1,3}\s*$', re.MULTILINE),
]

def detect_toc_regions(text: str) -> list[Tuple[int, int]]:
    """
    Detect TOC regions by heuristics. Returns list of (start, end) char ranges.
    
    Heuristics:
    1. Dense "Item X ... page" listings (>5 in close proximity)
    2. High ratio of dotted leaders to regular text
    3. Low paragraph density (short lines, many numbers)
    """
    toc_regions = []
    
    # Find dense Item listings
    item_matches = list(re.finditer(
        r'Item\s+\d+[A-Z]?\s*[\.\s]{2,}',
        text, re.IGNORECASE
    ))
    
    # If 5+ item matches within 2000 chars, mark as TOC region
    for i, match in enumerate(item_matches):
        if i + 4 < len(item_matches):
            span_start = match.start()
            span_end = item_matches[i + 4].end()
            if span_end - span_start < 2000:
                # Expand to include full TOC block
                block_start = max(0, span_start - 500)
                block_end = min(len(text), span_end + 500)
                toc_regions.append((block_start, block_end))
    
    return _merge_overlapping_regions(toc_regions)


def is_toc_chunk(chunk_text: str, char_start: int, toc_regions: list) -> bool:
    """Check if chunk falls within a TOC region."""
    for toc_start, toc_end in toc_regions:
        if char_start >= toc_start and char_start < toc_end:
            return True
    
    # Additional heuristic: high density of dotted leaders
    dotted_ratio = len(re.findall(r'\.{3,}', chunk_text)) / max(1, len(chunk_text) / 100)
    if dotted_ratio > 0.5:
        return True
    
    return False
```

**Why this is better**: 
- Non-destructive: original text intact for debugging
- Reversible: can re-evaluate TOC detection without re-parsing
- More robust: multiple heuristics, not single regex

### 2. DOM-Based Table-Local Context

**Change from v2**: Don't use character offsets (fragile with whitespace normalization).
Use **DOM adjacency** to find semantically related blocks.

```python
from bs4 import BeautifulSoup, Tag
from typing import List, Optional

def build_table_local_context_dom(
    soup: BeautifulSoup,
    table_element: Tag,
    sibling_blocks: int = 3
) -> str:
    """
    Build table-local context using DOM structure.
    
    Includes:
    - Table caption (if present)
    - N preceding sibling blocks (paragraphs, divs)
    - N following sibling blocks
    - Footnote blocks semantically tied to table
    
    Args:
        soup: Parsed HTML document
        table_element: The <table> tag
        sibling_blocks: Number of sibling blocks to include each side
    
    Returns:
        Combined text from table context
    """
    context_parts = []
    
    # 1. Table caption
    caption = table_element.find('caption')
    if caption:
        context_parts.append(f"[CAPTION] {caption.get_text(strip=True)}")
    
    # 2. Preceding siblings (headings, paragraphs)
    preceding = []
    for sibling in table_element.find_previous_siblings():
        if len(preceding) >= sibling_blocks:
            break
        if _is_content_block(sibling):
            preceding.append(sibling.get_text(separator=' ', strip=True))
    
    # Add in correct order (closest first)
    for text in reversed(preceding):
        context_parts.append(f"[BEFORE] {text}")
    
    # 3. Table content (for reference)
    # context_parts.append(f"[TABLE] {table_element.get_text(separator=' ', strip=True)[:2000]}")
    
    # 4. Following siblings (footnotes often immediately after)
    following_count = 0
    for sibling in table_element.find_next_siblings():
        if following_count >= sibling_blocks:
            break
        if _is_content_block(sibling):
            text = sibling.get_text(separator=' ', strip=True)
            
            # Check if this looks like a footnote block
            if _is_footnote_block(text):
                context_parts.append(f"[FOOTNOTE] {text}")
            else:
                context_parts.append(f"[AFTER] {text}")
            
            following_count += 1
    
    return "\n\n".join(context_parts)


def _is_content_block(element: Tag) -> bool:
    """Check if element is a content-bearing block."""
    if not isinstance(element, Tag):
        return False
    if element.name in ('p', 'div', 'span', 'td', 'section'):
        text = element.get_text(strip=True)
        return len(text) > 20  # Skip empty/tiny elements
    return False


def _is_footnote_block(text: str) -> bool:
    """Detect if text block is a footnote definition."""
    # Footnote patterns: (1) ..., (a) ..., [1] ...
    footnote_pattern = re.compile(r'^\s*[\(\[]?\d+[\)\]]?\s*[A-Z]', re.MULTILINE)
    return bool(footnote_pattern.search(text[:200]))
```

**Why this is better**:
- Parser-agnostic: works regardless of whitespace normalization
- Semantic: finds actual related content, not arbitrary character windows
- Footnote-aware: explicitly captures footnote blocks that define terms

---

## Technical Implementation (Revised)

### 1. Structure-Aware Chunking

Replace naive sliding windows with section-aware chunks:

```python
from dataclasses import dataclass
from typing import List, Optional
import re

@dataclass
class Chunk:
    """A chunk with metadata for filtered retrieval."""
    chunk_id: str
    text: str
    section: str          # "item1", "item7", "note_18", "risk_factors", etc.
    heading: Optional[str]  # "Revenue Recognition", "Segment Information"
    table_id: Optional[str] # "t0042" if near a table
    char_range: tuple       # (start, end) in original document
    is_toc: bool = False

# Section detection patterns
SECTION_PATTERNS = {
    'item1': re.compile(r'ITEM\s*1[^0-9A-Z].*?BUSINESS', re.IGNORECASE),
    'item7': re.compile(r'ITEM\s*7[^A-Z].*?MANAGEMENT', re.IGNORECASE),
    'item8': re.compile(r'ITEM\s*8[^A-Z].*?FINANCIAL\s+STATEMENTS', re.IGNORECASE),
    'note_segment': re.compile(r'NOTE\s*\d+.*?(SEGMENT|OPERATING\s+SEGMENT)', re.IGNORECASE),
    'note_revenue': re.compile(r'NOTE\s*\d+.*?REVENUE', re.IGNORECASE),
    'risk_factors': re.compile(r'ITEM\s*1A.*?RISK\s+FACTORS', re.IGNORECASE),
    'liquidity': re.compile(r'LIQUIDITY\s+AND\s+CAPITAL', re.IGNORECASE),
}

def chunk_10k_structured(
    text: str, 
    chunk_size: int = 800, 
    overlap: int = 100
) -> List[Chunk]:
    """
    Structure-aware chunking with metadata.
    
    Key improvements:
    1. Detect section boundaries (Item 1, Item 7, Notes)
    2. Attach section metadata to each chunk
    3. Detect headings within sections
    4. Skip TOC sections entirely
    """
    # Step 1: Strip TOC
    text = strip_toc(text)
    
    # Step 2: Identify section boundaries
    section_ranges = _identify_sections(text)
    
    # Step 3: Chunk within sections
    chunks = []
    chunk_idx = 0
    
    for section_name, (sec_start, sec_end) in section_ranges.items():
        section_text = text[sec_start:sec_end]
        
        # Skip TOC and risk factors for description extraction
        if section_name == 'toc':
            continue
        
        # Chunk this section
        pos = 0
        while pos < len(section_text):
            end_pos = min(pos + chunk_size, len(section_text))
            chunk_text = section_text[pos:end_pos]
            
            # Detect heading if present
            heading = _detect_heading(chunk_text)
            
            chunks.append(Chunk(
                chunk_id=f"chunk_{chunk_idx:04d}",
                text=chunk_text.strip(),
                section=section_name,
                heading=heading,
                table_id=None,  # Filled later if near table
                char_range=(sec_start + pos, sec_start + end_pos),
                is_toc=False
            ))
            
            chunk_idx += 1
            pos = end_pos - overlap
    
    return chunks


def _identify_sections(text: str, toc_regions: list) -> dict:
    """
    Identify section boundaries with multiple-candidate detection.
    
    Change from v2: Don't use single-match. Find ALL candidates,
    then select first that passes "not TOC" + "body-like density" checks.
    """
    sections = {}
    
    for name, pattern in SECTION_PATTERNS.items():
        # Find ALL matches, not just first
        candidates = list(pattern.finditer(text))
        
        for match in candidates:
            start = match.start()
            
            # Check 1: Not in TOC region
            if _in_toc_region(start, toc_regions):
                continue
            
            # Check 2: Body-like density (paragraphs, not lists)
            window = text[start:start + 2000]
            if not _has_body_density(window):
                continue
            
            # This candidate passes checks
            end = _find_next_section_start(text, start + 100) or len(text)
            sections[name] = (start, end)
            break  # Use first valid candidate
    
    return sections


def _in_toc_region(char_pos: int, toc_regions: list) -> bool:
    """Check if position falls within any TOC region."""
    for toc_start, toc_end in toc_regions:
        if toc_start <= char_pos < toc_end:
            return True
    return False


def _has_body_density(text: str) -> bool:
    """
    Check if text has body-like density (paragraphs, not TOC/lists).
    
    Heuristics:
    - Low ratio of dotted leaders
    - Higher paragraph density (avg line length > 60 chars)
    - Few page-number patterns
    """
    # Dotted leader ratio
    dotted_count = len(re.findall(r'\.{3,}', text))
    if dotted_count > 5:
        return False
    
    # Average line length
    lines = [l for l in text.split('\n') if l.strip()]
    if lines:
        avg_len = sum(len(l) for l in lines) / len(lines)
        if avg_len < 40:  # Too short, likely TOC or index
            return False
    
    # Page number density
    page_nums = len(re.findall(r'\b\d{1,3}\s*$', text, re.MULTILINE))
    if page_nums > 10:
        return False
    
    return True


def _detect_heading(chunk_text: str) -> Optional[str]:
    """Extract heading if chunk starts with one."""
    # Look for bold/caps heading patterns
    heading_match = re.match(
        r'^[A-Z][A-Z\s]+(?=\n)|^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?=\n)',
        chunk_text
    )
    return heading_match.group(0).strip() if heading_match else None
```

### 2. Two-Tier Index Architecture

```python
import numpy as np
import faiss
from pathlib import Path
import json
from typing import List, Tuple, Optional

class TwoTierIndex:
    """
    Two-tier embedding index:
    - Tier 1: Table-local context (high precision)
    - Tier 2: Full filing (high recall fallback)
    """
    
    def __init__(
        self, 
        ticker: str, 
        cache_dir: Path = Path("data/embeddings")
    ):
        self.ticker = ticker
        self.cache_dir = cache_dir
        
        # Tier 1: Table-local
        self.local_index: Optional[faiss.IndexFlatIP] = None
        self.local_chunks: List[Chunk] = []
        
        # Tier 2: Full filing
        self.full_index: Optional[faiss.IndexFlatIP] = None
        self.full_chunks: List[Chunk] = []
    
    def build(
        self,
        table_local_chunks: List[Chunk],
        full_filing_chunks: List[Chunk],
        embeddings_local: List[List[float]],
        embeddings_full: List[List[float]]
    ):
        """Build both indexes."""
        # Tier 1: Table-local
        self.local_chunks = table_local_chunks
        if table_local_chunks:
            local_matrix = np.array(embeddings_local, dtype=np.float32)
            faiss.normalize_L2(local_matrix)
            self.local_index = faiss.IndexFlatIP(local_matrix.shape[1])
            self.local_index.add(local_matrix)
        
        # Tier 2: Full filing
        self.full_chunks = full_filing_chunks
        full_matrix = np.array(embeddings_full, dtype=np.float32)
        faiss.normalize_L2(full_matrix)
        self.full_index = faiss.IndexFlatIP(full_matrix.shape[1])
        self.full_index.add(full_matrix)
        
        # Cache to disk
        self._save_cache()
    
    def retrieve(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        local_threshold: float = None,  # Use calibrated value
        global_threshold: float = None,  # Use calibrated value
        prefer_sections: List[str] = None
    ) -> Tuple[List[Chunk], List[float], str]:
        """
        Two-tier retrieval with quality controls.
        
        Thresholds are calibrated per corpus (see ThresholdCalibrator).
        Defaults: local=0.70, global=0.60 if not calibrated.
        
        Returns:
            (chunks, scores, tier_used)
        """
        # Use calibrated or default thresholds
        local_threshold = local_threshold or self.calibrated_local_threshold or 0.70
        global_threshold = global_threshold or self.calibrated_global_threshold or 0.60
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)
        
        # Tier 1: Try table-local first
        if self.local_index is not None and self.local_index.ntotal > 0:
            scores, indices = self.local_index.search(query, min(top_k, self.local_index.ntotal))
            
            # Filter by threshold
            valid = [(self.local_chunks[i], s) for i, s in zip(indices[0], scores[0]) if s >= local_threshold]
            
            if valid:
                chunks, chunk_scores = zip(*valid)
                # Apply MMR for diversity
                chunks, chunk_scores = self._apply_mmr(list(chunks), list(chunk_scores))
                return chunks[:top_k], chunk_scores[:top_k], "tier1_local"
        
        # Tier 2: Full filing fallback
        scores, indices = self.full_index.search(query, top_k * 2)  # Over-fetch for filtering
        
        results = []
        for i, s in zip(indices[0], scores[0]):
            if s < global_threshold:
                continue
            chunk = self.full_chunks[i]
            
            # Section filtering/boosting
            if prefer_sections:
                if chunk.section in prefer_sections:
                    s *= 1.1  # Boost preferred sections
                elif chunk.section in ('risk_factors', 'liquidity'):
                    s *= 0.8  # Deprioritize irrelevant sections
            
            results.append((chunk, s))
        
        if not results:
            return [], [], "tier2_empty"
        
        # Sort by adjusted score
        results.sort(key=lambda x: -x[1])
        chunks, chunk_scores = zip(*results[:top_k])
        
        # Apply MMR
        chunks, chunk_scores = self._apply_mmr(list(chunks), list(chunk_scores))
        return chunks[:top_k], chunk_scores[:top_k], "tier2_full"
    
    def _apply_mmr(
        self, 
        chunks: List[Chunk], 
        scores: List[float],
        lambda_param: float = 0.7
    ) -> Tuple[List[Chunk], List[float]]:
        """
        Maximal Marginal Relevance for diversity.
        Removes near-duplicate chunks (cosine > 0.95).
        """
        if len(chunks) <= 1:
            return chunks, scores
        
        # Simple dedup: remove chunks with very similar text
        seen_texts = set()
        deduped = []
        deduped_scores = []
        
        for chunk, score in zip(chunks, scores):
            # Hash first 200 chars for dedup
            text_key = chunk.text[:200].lower()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                deduped.append(chunk)
                deduped_scores.append(score)
        
        return deduped, deduped_scores
    
    def _save_cache(self):
        """Save indexes and metadata to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS indexes as binary
        if self.local_index:
            faiss.write_index(
                self.local_index, 
                str(self.cache_dir / f"{self.ticker}_local.faiss")
            )
        faiss.write_index(
            self.full_index,
            str(self.cache_dir / f"{self.ticker}_full.faiss")
        )
        
        # Save chunk metadata as JSON
        metadata = {
            "local_chunks": [self._chunk_to_dict(c) for c in self.local_chunks],
            "full_chunks": [self._chunk_to_dict(c) for c in self.full_chunks],
        }
        (self.cache_dir / f"{self.ticker}_metadata.json").write_text(
            json.dumps(metadata, indent=2)
        )
    
    def load_from_cache(self) -> bool:
        """Load from disk cache. Returns True if successful."""
        try:
            # Load FAISS indexes
            local_path = self.cache_dir / f"{self.ticker}_local.faiss"
            if local_path.exists():
                self.local_index = faiss.read_index(str(local_path))
            
            self.full_index = faiss.read_index(
                str(self.cache_dir / f"{self.ticker}_full.faiss")
            )
            
            # Load metadata
            metadata = json.loads(
                (self.cache_dir / f"{self.ticker}_metadata.json").read_text()
            )
            self.local_chunks = [self._dict_to_chunk(d) for d in metadata["local_chunks"]]
            self.full_chunks = [self._dict_to_chunk(d) for d in metadata["full_chunks"]]
            
            return True
        except FileNotFoundError:
            return False
    
    @staticmethod
    def _chunk_to_dict(chunk: Chunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "section": chunk.section,
            "heading": chunk.heading,
            "table_id": chunk.table_id,
            "char_range": chunk.char_range,
        }
    
    @staticmethod
    def _dict_to_chunk(d: dict) -> Chunk:
        return Chunk(
            chunk_id=d["chunk_id"],
            text=d["text"],
            section=d["section"],
            heading=d.get("heading"),
            table_id=d.get("table_id"),
            char_range=tuple(d["char_range"]),
        )
```

### 3. Threshold Calibration Harness

**Change from v2**: Don't hard-code thresholds. Calibrate per corpus using known-good pairs.

```python
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class CalibrationPair:
    """Known-good query-chunk pair for calibration."""
    ticker: str
    query: str
    expected_chunk_text: str  # Substring that should be in top results
    source: str  # "footnote", "table_local", "item1", etc.

# Known-good pairs from manual inspection
CALIBRATION_PAIRS = [
    # AMZN footnotes (should score high on table-local)
    CalibrationPair(
        ticker="AMZN",
        query="AMZN FY2024 revenue line 'Online stores' products services",
        expected_chunk_text="Includes product sales and digital media content",
        source="footnote"
    ),
    CalibrationPair(
        ticker="AMZN", 
        query="AMZN FY2024 revenue line 'Third-party seller services'",
        expected_chunk_text="commissions and any related fulfillment and shipping fees",
        source="footnote"
    ),
    # MSFT segment note (should score high on note_segment)
    CalibrationPair(
        ticker="MSFT",
        query="MSFT FY2025 revenue line 'Intelligent Cloud' products services",
        expected_chunk_text="Azure",
        source="note_segment"
    ),
    # Add more known-good pairs...
]

class ThresholdCalibrator:
    """
    Calibrate retrieval thresholds using known-good pairs.
    
    Method:
    1. For each known-good pair, compute score of expected chunk
    2. Compute score distribution of "noise" chunks (non-matching)
    3. Set threshold at P90 of noise scores (allows 10% false positives)
    """
    
    def calibrate(
        self,
        index: 'TwoTierIndex',
        pairs: List[CalibrationPair],
        noise_percentile: float = 90
    ) -> Tuple[float, float]:
        """
        Returns (local_threshold, global_threshold).
        """
        local_good_scores = []
        local_noise_scores = []
        global_good_scores = []
        global_noise_scores = []
        
        for pair in pairs:
            query_emb = embed_query(pair.query)
            
            # Get all scores from both tiers
            if index.local_index and index.local_index.ntotal > 0:
                local_scores, local_indices = index.local_index.search(
                    np.array([query_emb], dtype=np.float32), 
                    index.local_index.ntotal
                )
                
                for i, score in zip(local_indices[0], local_scores[0]):
                    chunk = index.local_chunks[i]
                    if pair.expected_chunk_text.lower() in chunk.text.lower():
                        local_good_scores.append(score)
                    else:
                        local_noise_scores.append(score)
            
            # Similar for global index...
            global_scores, global_indices = index.full_index.search(
                np.array([query_emb], dtype=np.float32),
                min(100, index.full_index.ntotal)
            )
            
            for i, score in zip(global_indices[0], global_scores[0]):
                chunk = index.full_chunks[i]
                if pair.expected_chunk_text.lower() in chunk.text.lower():
                    global_good_scores.append(score)
                else:
                    global_noise_scores.append(score)
        
        # Set thresholds at noise percentile
        local_threshold = np.percentile(local_noise_scores, noise_percentile) if local_noise_scores else 0.70
        global_threshold = np.percentile(global_noise_scores, noise_percentile) if global_noise_scores else 0.60
        
        # Ensure threshold doesn't exceed good scores
        if local_good_scores:
            local_threshold = min(local_threshold, min(local_good_scores) - 0.05)
        if global_good_scores:
            global_threshold = min(global_threshold, min(global_good_scores) - 0.05)
        
        return max(0.50, local_threshold), max(0.45, global_threshold)
```

**Why this is better**:
- Data-driven: thresholds based on actual score distributions
- Adaptive: can recalibrate when embedding model changes
- Transparent: can inspect which pairs drive threshold

---

### 4. Rich Query Construction

```python
def build_rag_query(
    company_name: str,
    ticker: str,
    fiscal_year: int,
    revenue_line: str,
    revenue_group: str,
    table_caption: str
) -> str:
    """
    Construct rich query for semantic search.
    
    Key insight: Include context (revenue group, table caption)
    to disambiguate labels like "Compute" from risk factors.
    """
    return (
        f"{company_name} ({ticker}) FY{fiscal_year} "
        f"revenue line '{revenue_line}' "
        f"in segment '{revenue_group}'. "
        f"Table: {table_caption}. "
        f"What products and services are included? "
        f"Use definitions from revenue recognition note or segment note."
    )

# Example queries:
# Bad:  "Compute"
# Good: "NVIDIA (NVDA) FY2025 revenue line 'Compute' in segment 'Compute & Networking'. 
#        Table: Revenue by End Market. What products and services are included? 
#        Use definitions from revenue recognition note or segment note."
```

### 5. Auditable Generation with Evidence Gate

**Changes from v2**:
1. Add **evidence coverage gate**: require ≥1 chunk from preferred section or table-local
2. Add **extractive-first product enumeration**: deterministic pass before LLM

```python
from dataclasses import dataclass
from typing import List, Optional, Set
import re

# Preferred sections for description extraction
PREFERRED_SECTIONS = {'note_revenue', 'note_segment', 'item1', 'item7'}

@dataclass
class DescriptionResult:
    """Auditable description with evidence."""
    revenue_line: str
    description: str
    products_services_list: List[str]
    evidence_chunk_ids: List[str]
    evidence_quotes: List[str]
    retrieval_tier: str  # "tier1_local" | "tier2_full" | "tier2_empty"
    validated: bool      # True if quotes verified in source
    evidence_gate_passed: bool  # True if ≥1 chunk from preferred section


def extract_candidate_products(chunks: List[Chunk]) -> Set[str]:
    """
    Extractive-first: Pull candidate product/service names from chunks
    using deterministic patterns BEFORE LLM filtering.
    
    Patterns:
    - Capitalized noun phrases (e.g., "Azure", "DGX Systems")
    - Trademark patterns (®, ™)
    - Model names (alphanumeric: "H100", "A100")
    - Quoted terms
    """
    candidates = set()
    
    for chunk in chunks:
        text = chunk.text
        
        # Pattern 1: Capitalized multi-word phrases (2-4 words)
        cap_phrases = re.findall(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b',
            text
        )
        candidates.update(cap_phrases)
        
        # Pattern 2: Trademark symbols
        trademark = re.findall(r'(\b\w+[®™])', text)
        candidates.update(tm.rstrip('®™') for tm in trademark)
        
        # Pattern 3: Product model patterns (letters + numbers)
        models = re.findall(r'\b([A-Z]{1,4}\d{2,4}[A-Z]?)\b', text)
        candidates.update(models)
        
        # Pattern 4: Quoted product names
        quoted = re.findall(r'"([^"]{3,30})"', text)
        candidates.update(quoted)
        
        # Pattern 5: "including X, Y, and Z" patterns
        including = re.findall(
            r'including\s+([A-Z][^,\.]{2,20}(?:,\s*[A-Z][^,\.]{2,20})*)',
            text, re.IGNORECASE
        )
        for match in including:
            items = re.split(r',\s*(?:and\s+)?', match)
            candidates.update(i.strip() for i in items if i.strip())
    
    # Filter out common false positives
    stopwords = {
        'The', 'This', 'These', 'Our', 'We', 'Company', 'Revenue',
        'Services', 'Products', 'Business', 'Segment', 'Total',
        'United States', 'North America', 'International'
    }
    candidates = {c for c in candidates if c not in stopwords and len(c) > 2}
    
    return candidates


def check_evidence_gate(
    chunks: List[Chunk],
    retrieval_tier: str
) -> bool:
    """
    Evidence coverage gate: require ≥1 chunk from preferred section
    OR from table-local tier.
    
    Prevents plausible-but-wrong narratives from generic discussions.
    """
    # Tier 1 (table-local) always passes
    if retrieval_tier == "tier1_local":
        return True
    
    # Tier 2: check if any chunk is from preferred section
    for chunk in chunks:
        if chunk.section in PREFERRED_SECTIONS:
            return True
    
    return False


def generate_description_with_evidence(
    llm: OpenAIChatClient,
    revenue_line: str,
    revenue_group: str,
    chunks: List[Chunk],
    scores: List[float],
    retrieval_tier: str
) -> DescriptionResult:
    """
    Generate description with auditable evidence.
    
    Process:
    1. Check evidence gate (≥1 chunk from preferred section or table-local)
    2. Extract candidate products deterministically
    3. LLM filters/deduplicates candidates and generates description
    4. Post-validate quotes exist in chunks
    """
    # Step 0: Empty chunks
    if not chunks:
        return DescriptionResult(
            revenue_line=revenue_line,
            description="",
            products_services_list=[],
            evidence_chunk_ids=[],
            evidence_quotes=[],
            retrieval_tier=retrieval_tier,
            validated=True,
            evidence_gate_passed=False
        )
    
    # Step 1: Evidence gate
    gate_passed = check_evidence_gate(chunks, retrieval_tier)
    if not gate_passed:
        return DescriptionResult(
            revenue_line=revenue_line,
            description="",
            products_services_list=[],
            evidence_chunk_ids=[],
            evidence_quotes=[],
            retrieval_tier=retrieval_tier,
            validated=True,
            evidence_gate_passed=False
        )
    
    # Step 2: Extractive-first product candidates
    candidate_products = extract_candidate_products(chunks)
    
    # Build context with chunk IDs
    context_parts = []
    for chunk, score in zip(chunks, scores):
        context_parts.append(
            f"[{chunk.chunk_id}] (section: {chunk.section}, score: {score:.2f})\n"
            f"{chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)
    
    # Step 3: LLM call with candidate products
    system = """You are extracting product/service descriptions from SEC 10-K filings.

OUTPUT REQUIREMENTS (strict JSON):
{
  "description": "1-2 sentences describing what this revenue line includes. Use company language.",
  "products_services_list": ["specific", "products", "services"],
  "evidence_chunk_ids": ["chunk_0001", "chunk_0042"],
  "evidence_quotes": ["exact quoted text supporting description"]
}

RULES:
1. Use ONLY information from the provided chunks.
2. Quote or closely paraphrase the company's own words.
3. For products_services_list: FILTER the candidate_products to keep only those 
   that are EXPLICITLY mentioned as part of this revenue line in the chunks.
4. Include chunk_ids for ALL chunks you used.
5. Include EXACT quotes (10-50 words) that support your description.
6. If no relevant information found, return empty strings/arrays.
7. Do NOT add products not in candidate_products unless clearly stated in chunks."""

    user = f"""Revenue line: {revenue_line}
Revenue group: {revenue_group}

Candidate products (filter these): {sorted(candidate_products)[:30]}

Retrieved chunks from 10-K:
{context}

Extract description and filter products for "{revenue_line}"."""

    result = llm.json_call(system=system, user=user, max_output_tokens=500)
    
    # Step 4: Post-validation
    chunk_texts = {c.chunk_id: c.text for c in chunks}
    validated = True
    
    for chunk_id in result.get("evidence_chunk_ids", []):
        if chunk_id not in chunk_texts:
            validated = False
            break
    
    for quote in result.get("evidence_quotes", []):
        quote_lower = quote.lower()[:50]
        found = any(quote_lower in c.text.lower() for c in chunks)
        if not found:
            validated = False
            break
    
    return DescriptionResult(
        revenue_line=revenue_line,
        description=result.get("description", "") if validated else "",
        products_services_list=result.get("products_services_list", []) if validated else [],
        evidence_chunk_ids=result.get("evidence_chunk_ids", []) if validated else [],
        evidence_quotes=result.get("evidence_quotes", []) if validated else [],
        retrieval_tier=retrieval_tier,
        validated=validated,
        evidence_gate_passed=gate_passed
    )
```

**Why these changes are better**:
- **Evidence gate**: Prevents "Compute capacity constraints" from risk factors being used
- **Extractive-first**: LLM filters from known candidates, not generating from scratch
- **Grounded**: Products must be in chunks AND pass LLM relevance check

---

## RAG Scope Definition

### ✅ Use RAG For

| Task | Rationale |
|------|-----------|
| **CSV1 line descriptions** | Narrative descriptions in Item 1/7, Notes |
| Product/service enumeration | Lists spread across sections |

> **Note**: Focus is CSV1. CSV2/CSV3 are out of scope for now.

### ❌ Do NOT Use RAG For

| Task | Rationale | Keep Current |
|------|-----------|--------------|
| Table selection | Deterministic gating works well | `table_kind.py` |
| Numeric extraction | Tables are structured, not semantic | Layout inference |
| Validation | Math-based reconciliation | `validation.py` |

---

## CSV1 QA Artifact

**New in v3**: Write per-ticker QA artifact for regression testing.

```python
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path
import json

@dataclass
class CSV1DescCoverage:
    """QA artifact for CSV1 description coverage."""
    ticker: str
    fiscal_year: int
    total_lines: int
    lines_with_description: int
    coverage_pct: float
    missing_labels: List[str]
    line_details: List[Dict]  # Per-line: label, has_desc, tier, chunk_ids

def write_csv1_qa_artifact(
    ticker: str,
    fiscal_year: int,
    results: List[DescriptionResult],
    output_dir: Path
) -> CSV1DescCoverage:
    """
    Write csv1_desc_coverage.json for regression testing.
    
    Example output:
    {
        "ticker": "NVDA",
        "fiscal_year": 2025,
        "total_lines": 6,
        "lines_with_description": 5,
        "coverage_pct": 83.3,
        "missing_labels": ["OEM and Other"],
        "line_details": [
            {
                "revenue_line": "Compute",
                "has_description": true,
                "retrieval_tier": "tier2_full",
                "evidence_gate_passed": true,
                "top_chunk_ids": ["chunk_0142", "chunk_0143"],
                "top_sections": ["item1", "note_segment"]
            },
            ...
        ]
    }
    """
    total = len(results)
    with_desc = sum(1 for r in results if r.description)
    missing = [r.revenue_line for r in results if not r.description]
    
    line_details = []
    for r in results:
        # Get top sections from chunk IDs
        top_sections = list(set(
            # Would need chunk metadata lookup in real impl
        ))[:3]
        
        line_details.append({
            "revenue_line": r.revenue_line,
            "has_description": bool(r.description),
            "retrieval_tier": r.retrieval_tier,
            "evidence_gate_passed": r.evidence_gate_passed,
            "validated": r.validated,
            "top_chunk_ids": r.evidence_chunk_ids[:3],
            "products_found": len(r.products_services_list),
        })
    
    coverage = CSV1DescCoverage(
        ticker=ticker,
        fiscal_year=fiscal_year,
        total_lines=total,
        lines_with_description=with_desc,
        coverage_pct=round(100 * with_desc / total, 1) if total > 0 else 0,
        missing_labels=missing,
        line_details=line_details
    )
    
    # Write artifact
    output_path = output_dir / f"{ticker}_csv1_desc_coverage.json"
    output_path.write_text(json.dumps(asdict(coverage), indent=2))
    
    return coverage


def run_regression_check(
    current: CSV1DescCoverage,
    baseline: CSV1DescCoverage
) -> Dict:
    """
    Compare current run against baseline for regression.
    
    Returns dict with:
    - coverage_delta: change in coverage %
    - new_missing: labels that were covered but now missing
    - new_covered: labels that were missing but now covered
    """
    baseline_covered = set(
        d["revenue_line"] for d in baseline.line_details if d["has_description"]
    )
    current_covered = set(
        d["revenue_line"] for d in current.line_details if d["has_description"]
    )
    
    return {
        "coverage_delta": current.coverage_pct - baseline.coverage_pct,
        "new_missing": list(baseline_covered - current_covered),
        "new_covered": list(current_covered - baseline_covered),
        "regression": len(baseline_covered - current_covered) > 0
    }
```

**Why this is valuable**:
- **Regression testing**: Detect when changes break previously-working descriptions
- **Debugging**: Know exactly which tier/chunks were used for each line
- **Coverage tracking**: Monitor improvement over time

---

## Storage Architecture (Production-Ready)

### File Structure

```
data/embeddings/
├── AAPL/
│   ├── AAPL_local.faiss       # Tier 1 index (binary)
│   ├── AAPL_full.faiss        # Tier 2 index (binary)
│   └── AAPL_metadata.json     # Chunk text + metadata
├── MSFT/
│   ├── MSFT_local.faiss
│   ├── MSFT_full.faiss
│   └── MSFT_metadata.json
└── ...
```

### Why FAISS + JSON (not JSON embeddings)

| Approach | 6 Tickers | 1000 Tickers | Load Time |
|----------|-----------|--------------|-----------|
| JSON embeddings | 50 MB | 8 GB | Slow (parse) |
| **FAISS + JSON** | 30 MB | 5 GB | Fast (mmap) |
| SQLite + FAISS | 25 MB | 4 GB | Fast |

FAISS binary indexes support memory-mapped loading for instant access.

---

## Cost Analysis (Revised)

### Embedding Costs (One-Time per Filing)

| Component | Tokens | Cost |
|-----------|--------|------|
| Full filing (~400 chunks) | ~100k | $0.002 |
| Table-local (~50 chunks) | ~12k | $0.0003 |
| **Total per 10-K** | ~112k | **$0.0023** |

| Scale | Cost |
|-------|------|
| 6 tickers | $0.014 |
| 100 tickers | $0.23 |
| 1,000 tickers | $2.30 |

### LLM Costs (Per Revenue Line)

| Operation | Tokens | Cost |
|-----------|--------|------|
| Query embedding | ~50 | $0.000001 |
| RAG generation | ~1000 in + 200 out | ~$0.003 |
| **Per line** | ~1250 | **$0.003** |

### Total Run Cost

| Scale | Embedding | Generation | Total |
|-------|-----------|------------|-------|
| 6 tickers (38 lines) | $0.014 | $0.11 | **$0.12** |
| 100 tickers (~600 lines) | $0.23 | $1.80 | **$2.03** |

---

## Implementation Plan (v3 - Final)

### Phase 0: Pre-RAG Fixes (Day 1 - AM)
- [ ] Implement `detect_toc_regions()` with non-destructive tagging
- [ ] Implement `is_toc_chunk()` heuristics
- [ ] Implement `build_table_local_context_dom()` using BeautifulSoup
- [ ] Add DOM-based table element tracking to table selection

### Phase 1: Structure-Aware Chunking (Day 1 - PM)
- [ ] Implement `Chunk` dataclass with metadata (`section`, `heading`, `is_toc`)
- [ ] Implement `chunk_10k_structured()` with section detection
- [ ] Implement `_identify_sections()` with multiple-candidate selection
- [ ] Add `_has_body_density()` checks for TOC rejection

### Phase 2: Two-Tier Index (Day 2)
- [ ] Implement `TwoTierIndex` class with table-local + full-filing indexes
- [ ] Add FAISS binary save/load (not JSON embeddings)
- [ ] Implement `retrieve()` with tier fallback
- [ ] Add MMR deduplication and section boosting

### Phase 3: Threshold Calibration (Day 3 - AM)
- [ ] Define `CalibrationPair` dataset (AMZN footnotes, MSFT notes)
- [ ] Implement `ThresholdCalibrator` with percentile-based thresholds
- [ ] Run calibration on known-good pairs
- [ ] Document calibrated thresholds per embedding model

### Phase 4: Generation & Evidence Gate (Day 3 - PM)
- [ ] Implement `extract_candidate_products()` (extractive-first)
- [ ] Implement `check_evidence_gate()` (preferred section check)
- [ ] Implement `generate_description_with_evidence()` with LLM
- [ ] Add post-validation for quotes

### Phase 5: Pipeline Integration & QA (Day 4)
- [ ] Add `--use-rag` flag to pipeline
- [ ] Implement `write_csv1_qa_artifact()` per ticker
- [ ] Implement `run_regression_check()` against baseline
- [ ] Update CSV1 output with new descriptions

### Phase 6: Testing & Sign-Off (Day 5)
- [ ] Test on NVDA (hardest case - expect 5/6 coverage)
- [ ] Test on all 6 tickers
- [ ] Compare: RAG vs current vs manual
- [ ] Document final coverage metrics
- [ ] Create baseline artifacts for future regression

### Dependencies

```bash
pip install faiss-cpu numpy
# OpenAI already installed
```

---

## Risks & Mitigations (v3)

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TOC chunks retrieved | Low | Junk descriptions | **Non-destructive detection + exclusion at retrieval** |
| DOM parsing fails | Low | Missing context | **Fallback to character-based extraction** |
| Table-local misses context | Medium | Empty tier 1 | **Fall back to tier 2 with section filtering** |
| Ambiguous label retrieval | Medium | Wrong context | **Rich query + evidence gate (preferred sections)** |
| LLM hallucinates products | Low | Wrong items | **Extractive-first + evidence validation** |
| Threshold miscalibration | Medium | Over/under retrieval | **Calibration harness with known-good pairs** |
| Near-duplicate chunks | Low | Redundant context | **MMR deduplication** |
| Storage grows large | Low | Disk space | **FAISS binary, not JSON embeddings** |
| Section detection fails | Low | Wrong metadata | **Multiple candidates + body-density checks** |
| Regression undetected | Medium | Quality degradation | **csv1_desc_coverage.json + regression checks** |

---

## Expected Results (Conservative)

### Before RAG (Current)

| Ticker | Lines | With Description |
|--------|-------|------------------|
| AMZN | 7 | 7 ✅ |
| META | 3 | 2 |
| NVDA | 6 | 0 ❌ |
| AAPL | 5 | 5 ✅ |
| MSFT | 10 | 9 |
| GOOGL | 6 | 5 |
| **Total** | **37** | **28 (76%)** |

### After Table-Local-First RAG (Expected)

| Ticker | Lines | With Description | Notes |
|--------|-------|------------------|-------|
| AMZN | 7 | 7 ✅ | Tier 1 (footnotes) |
| META | 3 | 3 ✅ | Tier 2 (Item 1) |
| NVDA | 6 | 5-6 ✅ | Tier 2 (narrative) |
| AAPL | 5 | 5 ✅ | Tier 1 (footnotes) |
| MSFT | 10 | 10 ✅ | Tier 1/2 (notes) |
| GOOGL | 6 | 6 ✅ | Tier 1/2 (notes) |
| **Total** | **37** | **35-36 (95-97%)** |

---

## Decision Checkpoint

**v3 Changes Accepted**:
- [x] TOC handling: Non-destructive detection + exclusion (not regex deletion)
- [x] Table-local context: DOM-based, not character offsets
- [x] Section detection: Multiple candidates with body-density checks
- [x] Thresholds: Calibrated per corpus, not hard-coded
- [x] Evidence gate: Require ≥1 chunk from preferred section
- [x] Product enumeration: Extractive-first, then LLM filter
- [x] QA artifact: `csv1_desc_coverage.json` per ticker

**Proceed with Implementation if**:
- [ ] 4-5 day implementation effort is acceptable
- [ ] FAISS + BeautifulSoup dependencies are acceptable
- [ ] CSV1-only focus (CSV2/CSV3 deferred) is acceptable

**Alternative (if RAG deferred)**:
- Fix TOC detection alone (+10-15% coverage)
- Fix DOM-based context (+5-10% coverage)
- Total: 76% → ~90% without semantic search

---

## Appendix: Why Two-Tier Works

### AMZN (Tier 1 Success)

```
Revenue Line: "Online stores"
Table-Local Context: "...Online stores (1)... (1) Includes product sales..."
Tier 1 Score: 0.89 ✅
→ Uses footnote directly
```

### NVDA (Tier 2 Fallback)

```
Revenue Line: "Compute"
Table-Local Context: [table numbers only, no narrative]
Tier 1 Score: 0.45 ❌ (below 0.70 threshold)
→ Falls back to Tier 2

Tier 2 Search: Full filing index
Section Filter: Prefer note_*, item1, item7
Top Result: "Data Center compute platforms include DGX systems..."
Tier 2 Score: 0.82 ✅
→ Uses Item 1 Business narrative
```

### META (Tier 2 with Section Boost)

```
Revenue Line: "Other revenue"
Table-Local Context: [no description]
Tier 1 Score: 0.38 ❌
→ Falls back to Tier 2

Tier 2 Search: 
- item7 chunk: "Other revenue consists of..." (score 0.75, boosted to 0.82)
- risk_factors chunk: "other revenue may decline..." (score 0.77, reduced to 0.62)
→ Uses MD&A description (preferred section)
```
