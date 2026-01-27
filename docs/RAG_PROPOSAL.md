# RAG-Enhanced Revenue Line Description Extraction

## Proposal for Semantic Search Integration (v2 - Revised)

**Date**: January 2026  
**Status**: Proposal (Revised with Developer Review)  
**Estimated Effort**: 3-4 days  

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

### 1. TOC Stripping

Before any chunking/embedding, strip the Table of Contents to prevent junk retrieval:

```python
import re

TOC_PATTERNS = [
    # SEC filings typically have explicit TOC markers
    re.compile(r'TABLE\s+OF\s+CONTENTS.*?(?=PART\s+I[^V]|ITEM\s+1[^0-9])', re.IGNORECASE | re.DOTALL),
    # Alternative: detect dense "Item N" listings
    re.compile(r'(Item\s+\d+[A-Z]?\s*\.{2,}.*\n){5,}', re.IGNORECASE),
]

def strip_toc(text: str) -> str:
    """Remove Table of Contents section to prevent junk retrieval."""
    for pattern in TOC_PATTERNS:
        text = pattern.sub('', text)
    return text
```

### 2. Bidirectional Table-Local Context

Current approach extracts `text[idx-800 : idx+2500]` (unidirectional). Fix to ±N chars:

```python
def build_table_local_context(
    html_text: str,
    table_start_char: int,
    table_end_char: int,
    context_chars: int = 5000
) -> str:
    """
    Build bidirectional context around the accepted revenue table.
    
    Args:
        table_start_char: Character position where table HTML starts
        table_end_char: Character position where table HTML ends
        context_chars: Characters to include before AND after table
    
    Returns:
        Text: [pre-context] + [table caption + footnotes] + [post-context]
    """
    start = max(0, table_start_char - context_chars)
    end = min(len(html_text), table_end_char + context_chars)
    
    return html_text[start:end]
```

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


def _identify_sections(text: str) -> dict:
    """Identify section boundaries in 10-K text."""
    sections = {}
    
    for name, pattern in SECTION_PATTERNS.items():
        match = pattern.search(text)
        if match:
            # Find next section start as this section's end
            start = match.start()
            end = _find_next_section_start(text, start + 100) or len(text)
            sections[name] = (start, end)
    
    # Add "other" for content not in identified sections
    # (implementation simplified for clarity)
    
    return sections


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
        local_threshold: float = 0.70,
        global_threshold: float = 0.60,
        prefer_sections: List[str] = None
    ) -> Tuple[List[Chunk], List[float], str]:
        """
        Two-tier retrieval with quality controls.
        
        Returns:
            (chunks, scores, tier_used)
        """
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

### 3. Rich Query Construction

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

### 4. Auditable Generation

```python
from dataclasses import dataclass
from typing import List, Optional

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
    
    LLM must output:
    - description: 1-2 sentences
    - products_services_list: specific items mentioned
    - evidence_chunk_ids: which chunks support the description
    - evidence_quotes: exact quoted text
    """
    if not chunks:
        return DescriptionResult(
            revenue_line=revenue_line,
            description="",
            products_services_list=[],
            evidence_chunk_ids=[],
            evidence_quotes=[],
            retrieval_tier=retrieval_tier,
            validated=True  # Empty is valid
        )
    
    # Build context with chunk IDs
    context_parts = []
    for chunk, score in zip(chunks, scores):
        context_parts.append(
            f"[{chunk.chunk_id}] (section: {chunk.section}, score: {score:.2f})\n"
            f"{chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)
    
    system = """You are extracting product/service descriptions from SEC 10-K filings.

OUTPUT REQUIREMENTS (strict JSON):
{
  "description": "1-2 sentences describing what this revenue line includes. Use company language.",
  "products_services_list": ["specific", "products", "or", "services", "mentioned"],
  "evidence_chunk_ids": ["chunk_0001", "chunk_0042"],
  "evidence_quotes": ["exact quoted text from chunks that support description"]
}

RULES:
1. Use ONLY information from the provided chunks.
2. Quote or closely paraphrase the company's own words.
3. List specific products/services mentioned (e.g., "DGX systems", "Azure", not generic terms).
4. Include chunk_ids for ALL chunks you used.
5. Include EXACT quotes that support your description.
6. If no relevant information found, return empty strings/arrays.
7. Do NOT hallucinate products not mentioned in the chunks."""

    user = f"""Revenue line: {revenue_line}
Revenue group: {revenue_group}

Retrieved chunks from 10-K:
{context}

Extract description and evidence for "{revenue_line}"."""

    result = llm.json_call(system=system, user=user, max_output_tokens=500)
    
    # Post-validation: verify quotes exist in chunks
    chunk_texts = {c.chunk_id: c.text for c in chunks}
    validated = True
    
    for chunk_id in result.get("evidence_chunk_ids", []):
        if chunk_id not in chunk_texts:
            validated = False
            break
    
    for quote in result.get("evidence_quotes", []):
        # Check if quote (or close variant) exists in any chunk
        quote_lower = quote.lower()[:50]  # First 50 chars
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
        validated=validated
    )
```

---

## RAG Scope Definition

### ✅ Use RAG For

| Task | Rationale |
|------|-----------|
| CSV1 line descriptions | Narrative descriptions in Item 1/7, Notes |
| CSV2 segment descriptions | Segment note text extraction |
| Product/service enumeration | Lists spread across sections |

### ❌ Do NOT Use RAG For

| Task | Rationale | Keep Current |
|------|-----------|--------------|
| Table selection | Deterministic gating works well | `table_kind.py` |
| Numeric extraction | Tables are structured, not semantic | Layout inference |
| Validation | Math-based reconciliation | `validation.py` |

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

## Implementation Plan (Revised)

### Phase 0: Pre-RAG Fixes (Day 1 - AM)
- [ ] Implement `strip_toc()` function
- [ ] Implement `build_table_local_context()` with ±5000 chars
- [ ] Add `table_start_char`, `table_end_char` tracking to table selection

### Phase 1: Structure-Aware Chunking (Day 1 - PM)
- [ ] Implement `Chunk` dataclass with metadata
- [ ] Implement `chunk_10k_structured()` with section detection
- [ ] Add `_identify_sections()` for Item 1/7/8, Notes

### Phase 2: Two-Tier Index (Day 2)
- [ ] Implement `TwoTierIndex` class
- [ ] Add FAISS binary save/load
- [ ] Implement `retrieve()` with thresholds
- [ ] Add MMR deduplication

### Phase 3: Rich Query & Generation (Day 3)
- [ ] Implement `build_rag_query()` with context
- [ ] Implement `generate_description_with_evidence()`
- [ ] Add post-validation for evidence quotes
- [ ] Integrate with pipeline (flag: `--use-rag`)

### Phase 4: Testing & Tuning (Day 4)
- [ ] Test on NVDA (hardest case)
- [ ] Test on all 6 tickers
- [ ] Tune thresholds (local: 0.70, global: 0.60)
- [ ] Compare: RAG vs current vs manual

### Dependencies

```bash
pip install faiss-cpu numpy
# OpenAI already installed
```

---

## Risks & Mitigations (Revised)

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TOC chunks retrieved | Medium | Junk descriptions | **Strip TOC before chunking** |
| Table-local misses context | Medium | Empty tier 1 | **Fall back to tier 2** |
| Ambiguous label retrieval | Medium | Wrong context | **Rich query with revenue group** |
| LLM hallucinates products | Medium | Wrong items | **Evidence validation** |
| Near-duplicate chunks | Low | Redundant context | **MMR deduplication** |
| Storage grows large | Low | Disk space | **FAISS binary, not JSON** |
| Section detection fails | Low | Wrong metadata | **Fallback to "other" section** |

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

**Proceed with Table-Local-First RAG if**:
- [x] Agree that TOC stripping is highest ROI fix
- [x] Agree that two-tier retrieval balances precision/recall
- [x] Agree that structure-aware chunking improves relevance
- [x] Agree that auditable evidence is critical for QA
- [ ] 3-4 day implementation effort is acceptable
- [ ] FAISS dependency is acceptable

**Alternative (if RAG deferred)**:
- Fix TOC stripping alone (+10-15% coverage)
- Fix bidirectional context (+5-10% coverage)
- Total: 76% → ~90% without RAG

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
