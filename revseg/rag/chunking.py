"""
Structure-aware chunking with TOC detection and DOM-based context extraction.

Key features:
- Non-destructive TOC detection (tag, don't delete)
- DOM-based table context extraction
- Section-aware chunking with metadata
- Multiple-candidate section boundary detection
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from bs4 import BeautifulSoup, Tag


@dataclass
class Chunk:
    """A text chunk with metadata for filtered retrieval."""
    chunk_id: str
    text: str
    section: str           # "item1", "item7", "note_segment", "note_revenue", etc.
    heading: Optional[str] # "Revenue Recognition", "Segment Information"
    table_id: Optional[str] # "t0042" if near a table
    char_range: Tuple[int, int]  # (start, end) in original document
    is_toc: bool = False
    
    def __hash__(self):
        return hash(self.chunk_id)
    
    def __eq__(self, other):
        if not isinstance(other, Chunk):
            return False
        return self.chunk_id == other.chunk_id


# =============================================================================
# TOC Detection (Non-Destructive)
# =============================================================================

def detect_toc_regions(text: str) -> List[Tuple[int, int]]:
    """
    Detect TOC regions by heuristics. Returns list of (start, end) char ranges.
    
    Heuristics:
    1. Dense "Item X ... page" listings (>5 in close proximity)
    2. High ratio of dotted leaders to regular text
    3. Low paragraph density (short lines, many numbers)
    
    Non-destructive: returns regions to exclude, doesn't modify text.
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
    
    # Also detect "TABLE OF CONTENTS" header
    toc_header = re.search(r'TABLE\s+OF\s+CONTENTS', text, re.IGNORECASE)
    if toc_header:
        # Find next major section start
        next_section = re.search(
            r'\n\s*(PART\s+I[^V]|ITEM\s+1[^0-9A-Z])',
            text[toc_header.end():],
            re.IGNORECASE
        )
        if next_section:
            toc_regions.append((
                toc_header.start(),
                toc_header.end() + next_section.start()
            ))
    
    return _merge_overlapping_regions(toc_regions)


def _merge_overlapping_regions(regions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping (start, end) regions."""
    if not regions:
        return []
    
    sorted_regions = sorted(regions, key=lambda x: x[0])
    merged = [sorted_regions[0]]
    
    for start, end in sorted_regions[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:  # Overlapping
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    
    return merged


def is_toc_chunk(chunk_text: str, char_start: int, toc_regions: List[Tuple[int, int]]) -> bool:
    """Check if chunk falls within a TOC region or has TOC characteristics."""
    # Check if in detected TOC region
    for toc_start, toc_end in toc_regions:
        if char_start >= toc_start and char_start < toc_end:
            return True
    
    # Additional heuristic: high density of dotted leaders
    dotted_count = len(re.findall(r'\.{3,}', chunk_text))
    if dotted_count > 5:
        return True
    
    # High density of page number patterns
    page_nums = len(re.findall(r'\.\s*\d{1,3}\s*$', chunk_text, re.MULTILINE))
    if page_nums > 8:
        return True
    
    return False


# =============================================================================
# DOM-Based Table Context
# =============================================================================

def build_table_local_context_dom(
    soup: BeautifulSoup,
    table_element: Tag,
    sibling_blocks: int = 3
) -> Tuple[str, List[Chunk]]:
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
        Tuple of (combined_text, list of Chunk objects)
    """
    context_parts = []
    chunks = []
    chunk_idx = 0
    
    # 1. Table caption
    caption = table_element.find('caption')
    if caption:
        caption_text = caption.get_text(strip=True)
        context_parts.append(f"[CAPTION] {caption_text}")
        chunks.append(Chunk(
            chunk_id=f"local_{chunk_idx:04d}",
            text=caption_text,
            section="table_caption",
            heading=None,
            table_id=None,
            char_range=(0, len(caption_text)),
            is_toc=False
        ))
        chunk_idx += 1
    
    # 2. Preceding siblings (headings, paragraphs)
    preceding = []
    for sibling in table_element.find_previous_siblings():
        if len(preceding) >= sibling_blocks:
            break
        if _is_content_block(sibling):
            text = sibling.get_text(separator=' ', strip=True)
            if text and len(text) > 20:
                preceding.append(text)
    
    # Add in correct order (farthest first)
    for text in reversed(preceding):
        context_parts.append(f"[BEFORE] {text}")
        chunks.append(Chunk(
            chunk_id=f"local_{chunk_idx:04d}",
            text=text,
            section="table_before",
            heading=_detect_heading(text),
            table_id=None,
            char_range=(0, len(text)),
            is_toc=False
        ))
        chunk_idx += 1
    
    # 3. Following siblings (footnotes often immediately after)
    following_count = 0
    for sibling in table_element.find_next_siblings():
        if following_count >= sibling_blocks + 2:  # Extra for footnotes
            break
        if _is_content_block(sibling):
            text = sibling.get_text(separator=' ', strip=True)
            if not text or len(text) < 20:
                continue
            
            # Check if this looks like a footnote block
            if _is_footnote_block(text):
                context_parts.append(f"[FOOTNOTE] {text}")
                section = "table_footnote"
            else:
                context_parts.append(f"[AFTER] {text}")
                section = "table_after"
            
            chunks.append(Chunk(
                chunk_id=f"local_{chunk_idx:04d}",
                text=text,
                section=section,
                heading=None,
                table_id=None,
                char_range=(0, len(text)),
                is_toc=False
            ))
            chunk_idx += 1
            following_count += 1
    
    combined = "\n\n".join(context_parts)
    return combined, chunks


def _is_content_block(element) -> bool:
    """Check if element is a content-bearing block."""
    if not isinstance(element, Tag):
        return False
    if element.name in ('p', 'div', 'span', 'section', 'article'):
        text = element.get_text(strip=True)
        return len(text) > 20  # Skip empty/tiny elements
    return False


def _is_footnote_block(text: str) -> bool:
    """Detect if text block is a footnote definition."""
    # Footnote patterns: (1) ..., (a) ..., [1] ..., ¹ ...
    footnote_pattern = re.compile(r'^\s*[\(\[]?\d+[\)\]]?\s+[A-Z]', re.MULTILINE)
    superscript_pattern = re.compile(r'^[¹²³⁴⁵⁶⁷⁸⁹]\s+', re.MULTILINE)
    
    return bool(footnote_pattern.search(text[:200])) or bool(superscript_pattern.search(text[:200]))


def _detect_heading(chunk_text: str) -> Optional[str]:
    """Extract heading if chunk starts with one."""
    if not chunk_text:
        return None
    
    # Look for all-caps heading or title case heading at start
    lines = chunk_text.strip().split('\n')
    if not lines:
        return None
    
    first_line = lines[0].strip()
    
    # All caps heading (e.g., "SEGMENT INFORMATION")
    if first_line.isupper() and len(first_line) < 100:
        return first_line
    
    # Title case heading pattern
    heading_match = re.match(
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5})\s*(?:$|\n)',
        first_line
    )
    if heading_match:
        return heading_match.group(1).strip()
    
    return None


# =============================================================================
# Section Detection
# =============================================================================

# Section detection patterns
SECTION_PATTERNS = {
    'item1': re.compile(r'ITEM\s*1[^0-9A-Z].*?BUSINESS', re.IGNORECASE),
    'item1a': re.compile(r'ITEM\s*1A.*?RISK\s+FACTORS', re.IGNORECASE),
    'item7': re.compile(r'ITEM\s*7[^A-Z].*?MANAGEMENT.{0,30}DISCUSSION', re.IGNORECASE),
    'item8': re.compile(r'ITEM\s*8[^A-Z].*?FINANCIAL\s+STATEMENTS', re.IGNORECASE),
    'note_segment': re.compile(r'NOTE\s*\d+\s*[-–—]?\s*(SEGMENT|OPERATING\s+SEGMENT)', re.IGNORECASE),
    'note_revenue': re.compile(r'NOTE\s*\d+\s*[-–—]?\s*REVENUE', re.IGNORECASE),
}

# Sections to deprioritize for description extraction
DEPRIORITIZE_SECTIONS = {'item1a', 'risk_factors', 'liquidity', 'capex'}

# =============================================================================
# P1: Note 2 (Revenue) Sub-classification
# =============================================================================

# Patterns indicating DEFINITION content (what the revenue IS) - keep these
DEFINITION_PATTERNS = [
    re.compile(r'\bconsists?\s+of\b', re.IGNORECASE),
    re.compile(r'\bincludes?\b.*\b(?:products?|services?|offerings?)\b', re.IGNORECASE),
    re.compile(r'\bgenerat(?:es?|ed)\s+(?:from|by)\b', re.IGNORECASE),
    re.compile(r'\bcomprises?\b', re.IGNORECASE),
    re.compile(r'\bprovides?\s+(?:products?|services?)\b', re.IGNORECASE),
    re.compile(r'\bofferings?\s+(?:include|such as)\b', re.IGNORECASE),
    re.compile(r'\bsales?\s+of\b', re.IGNORECASE),
    re.compile(r'\bdelivery\s+of\b', re.IGNORECASE),
    re.compile(r'\bsuch\s+as\b.*\b(?:products?|services?)\b', re.IGNORECASE),
]

# Patterns indicating ACCOUNTING/RECOGNITION content (how revenue is recognized) - block these
ACCOUNTING_PATTERNS = [
    re.compile(r'\bperformance\s+obligat', re.IGNORECASE),
    re.compile(r'\brecogniz(?:es?|ed|ing)\s+(?:revenue|when|upon|at)\b', re.IGNORECASE),
    re.compile(r'\bprincipal\s+(?:vs\.?|versus)\s+agent\b', re.IGNORECASE),
    re.compile(r'\bSSP\b|\bstand-alone\s+selling\s+price\b', re.IGNORECASE),
    re.compile(r'\bASC\s+\d{3}\b', re.IGNORECASE),
    re.compile(r'\bcontract\s+(?:liability|liabilities|asset)\b', re.IGNORECASE),
    re.compile(r'\ballocation\b', re.IGNORECASE),
    re.compile(r'\bcontrol\s+transfers?\b', re.IGNORECASE),
    re.compile(r'\bsatisf(?:ied|action)\b.*\bobligation\b', re.IGNORECASE),
    re.compile(r'\btransaction\s+price\b', re.IGNORECASE),
    re.compile(r'\bvariable\s+consideration\b', re.IGNORECASE),
    re.compile(r'\bpoint\s+in\s+time\b', re.IGNORECASE),
    re.compile(r'\bover\s+time\b.*\brecog', re.IGNORECASE),
    re.compile(r'\bdeferred\s+revenue\b', re.IGNORECASE),
    re.compile(r'\bunearned\s+revenue\b', re.IGNORECASE),
    re.compile(r'\bcontract\s+cost\b', re.IGNORECASE),
]


def classify_note_revenue_chunk(chunk_text: str) -> str:
    """
    Classify a chunk within note_revenue as either:
    - "note_revenue_sources" (product/service definitions) - KEEP for retrieval
    - "note_revenue_recognition" (accounting mechanics) - BLOCK from retrieval
    - "note_revenue" (ambiguous) - keep as fallback
    
    This is the key to META's "Other revenue" fix:
    The definition "Other revenue consists of WhatsApp Business Platform..." 
    is adjacent to "Revenue is recognized when performance obligations are satisfied..."
    We need to separate these at chunk level.
    """
    # Count matches for each pattern type
    definition_matches = sum(1 for p in DEFINITION_PATTERNS if p.search(chunk_text))
    accounting_matches = sum(1 for p in ACCOUNTING_PATTERNS if p.search(chunk_text))
    
    # Strong signal: if accounting patterns dominate, it's recognition text
    if accounting_matches >= 2 and accounting_matches > definition_matches:
        return "note_revenue_recognition"
    
    # Strong signal: if definition patterns dominate, it's source description
    if definition_matches >= 2 and definition_matches > accounting_matches:
        return "note_revenue_sources"
    
    # Mixed or ambiguous: check for specific high-confidence indicators
    if any(p.search(chunk_text) for p in [
        re.compile(r'\bperformance\s+obligat', re.IGNORECASE),
        re.compile(r'\bSSP\b', re.IGNORECASE),
        re.compile(r'\bASC\s+606\b', re.IGNORECASE),
        re.compile(r'\bprincipal\s+(?:vs|versus)\s+agent\b', re.IGNORECASE),
    ]):
        return "note_revenue_recognition"
    
    # Default: if no strong signal, keep as generic note_revenue
    # This allows retrieval but without preference
    return "note_revenue"


def _identify_sections(text: str, toc_regions: List[Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
    """
    Identify section boundaries with multiple-candidate detection.
    
    For each section pattern, find ALL matches, then select first that passes:
    1. Not in TOC region
    2. Has body-like density (paragraphs, not lists)
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
            end = _find_next_section_start(text, start + 500, toc_regions) or len(text)
            sections[name] = (start, end)
            break  # Use first valid candidate
    
    return sections


def _in_toc_region(char_pos: int, toc_regions: List[Tuple[int, int]]) -> bool:
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
    - Higher paragraph density (avg line length > 40 chars)
    - Few page-number patterns
    """
    if not text:
        return False
    
    # Dotted leader ratio
    dotted_count = len(re.findall(r'\.{3,}', text))
    if dotted_count > 5:
        return False
    
    # Average line length
    lines = [l for l in text.split('\n') if l.strip()]
    if lines:
        avg_len = sum(len(l) for l in lines) / len(lines)
        if avg_len < 30:  # Too short, likely TOC or index
            return False
    
    # Page number density
    page_nums = len(re.findall(r'\b\d{1,3}\s*$', text, re.MULTILINE))
    if page_nums > 10:
        return False
    
    return True


def _find_next_section_start(text: str, after_pos: int, toc_regions: List[Tuple[int, int]]) -> Optional[int]:
    """Find the start of the next major section after given position."""
    # Look for ITEM or NOTE patterns
    next_pattern = re.compile(
        r'\n\s*(ITEM\s+\d+[A-Z]?|NOTE\s+\d+\s*[-–—])',
        re.IGNORECASE
    )
    
    search_text = text[after_pos:]
    matches = list(next_pattern.finditer(search_text))
    
    for match in matches:
        abs_pos = after_pos + match.start()
        if not _in_toc_region(abs_pos, toc_regions):
            return abs_pos
    
    return None


# =============================================================================
# Structure-Aware Chunking
# =============================================================================

def chunk_10k_structured(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
    toc_regions: Optional[List[Tuple[int, int]]] = None
) -> List[Chunk]:
    """
    Structure-aware chunking with metadata.
    
    Key improvements:
    1. Detect section boundaries (Item 1, Item 7, Notes)
    2. Attach section metadata to each chunk
    3. Detect headings within sections
    4. Tag TOC chunks (don't delete, for transparency)
    
    Args:
        text: Full 10-K text
        chunk_size: Target chunk size in characters (~200 tokens)
        overlap: Overlap between chunks
        toc_regions: Pre-detected TOC regions (if None, will detect)
    
    Returns:
        List of Chunk objects with metadata
    """
    if toc_regions is None:
        toc_regions = detect_toc_regions(text)
    
    # Step 1: Identify section boundaries
    section_ranges = _identify_sections(text, toc_regions)
    
    # Step 2: Build section map for any position
    def get_section(pos: int) -> str:
        for name, (start, end) in section_ranges.items():
            if start <= pos < end:
                return name
        return "other"
    
    # Step 3: Chunk the full text with overlap
    chunks = []
    chunk_idx = 0
    pos = 0
    
    while pos < len(text):
        end_pos = min(pos + chunk_size, len(text))
        chunk_text = text[pos:end_pos]
        
        # Don't cut mid-sentence if possible
        if end_pos < len(text):
            last_period = chunk_text.rfind('.')
            if last_period > chunk_size // 2:
                end_pos = pos + last_period + 1
                chunk_text = text[pos:end_pos]
        
        # Determine section
        section = get_section(pos)
        
        # P1: Sub-classify note_revenue chunks to separate definitions from accounting
        if section == "note_revenue":
            section = classify_note_revenue_chunk(chunk_text)
        
        # Detect if this is TOC
        is_toc = is_toc_chunk(chunk_text, pos, toc_regions)
        
        # Detect heading
        heading = _detect_heading(chunk_text)
        
        chunk = Chunk(
            chunk_id=f"chunk_{chunk_idx:04d}",
            text=chunk_text.strip(),
            section=section,
            heading=heading,
            table_id=None,
            char_range=(pos, end_pos),
            is_toc=is_toc
        )
        
        chunks.append(chunk)
        chunk_idx += 1
        pos = end_pos - overlap
        
        # Prevent infinite loop
        if pos >= len(text) - overlap:
            break
    
    return chunks
