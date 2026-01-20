"""Token-based fuzzy matching for segment names."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

_STOPWORDS = {"and", "the", "of", "a", "an", "&", "to", "in", "for"}
_SEPARATOR_RE = re.compile(r"[,/\-–—&]+")
_FOOTNOTE_RE = re.compile(r"\s*\([^)]*\)\s*$")  # "Services (1)" -> "Services"
_WHITESPACE_RE = re.compile(r"\s+")


def tokenize_label(label: str) -> Set[str]:
    """
    Normalize and tokenize a label for fuzzy matching.
    
    Examples:
        'Wearables, Home and Accessories' -> {'wearables', 'home', 'accessories'}
        'Google Services - YouTube Ads' -> {'google', 'services', 'youtube', 'ads'}
        'Services (1)' -> {'services'}
    """
    s = str(label or "").strip()
    if not s:
        return set()
    
    # Strip footnotes like "(1)", "(a)", etc.
    s = _FOOTNOTE_RE.sub("", s)
    
    # Normalize separators to spaces
    s = _SEPARATOR_RE.sub(" ", s)
    
    # Lowercase and normalize whitespace
    s = _WHITESPACE_RE.sub(" ", s.lower()).strip()
    
    # Split and filter
    tokens = set(s.split()) - _STOPWORDS
    return {t for t in tokens if len(t) > 1}


def token_overlap_score(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """
    Compute overlap score between two token sets.
    
    Returns: |intersection| / |smaller set|
    This ensures that a subset matches well (e.g., "Services" matches "Google Services").
    """
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    smaller = min(len(tokens_a), len(tokens_b))
    return intersection / smaller


def fuzzy_match_segment(
    label: str,
    expected_segments: List[str],
    *,
    threshold: float = 0.6,
) -> Optional[str]:
    """
    Return the best-matching expected segment if overlap >= threshold.
    
    Args:
        label: The label to match (e.g., from a table row)
        expected_segments: List of canonical segment names to match against
        threshold: Minimum token overlap score (0.0 to 1.0)
    
    Returns:
        The best-matching segment name, or None if no match found.
    
    Examples:
        fuzzy_match_segment("Wearables/Home/Accessories", 
                           ["Wearables, Home and Accessories", "Services"])
        -> "Wearables, Home and Accessories"
    """
    label_tokens = tokenize_label(label)
    if not label_tokens:
        return None
    
    best_match: Optional[str] = None
    best_score = 0.0
    
    for seg in expected_segments:
        seg_tokens = tokenize_label(seg)
        if not seg_tokens:
            continue
        
        score = token_overlap_score(label_tokens, seg_tokens)
        
        # Prefer longer matches when scores are equal
        if score > best_score or (score == best_score and best_match and len(seg) > len(best_match)):
            if score >= threshold:
                best_score = score
                best_match = seg
    
    return best_match


def build_segment_matcher(expected_segments: List[str]) -> Dict[str, str]:
    """
    Build a lookup dict that maps normalized labels to canonical segment names.
    
    This pre-computes exact matches and common variations for fast lookup.
    
    Args:
        expected_segments: List of canonical segment names
    
    Returns:
        Dict mapping lowercase/normalized variants to canonical names
    """
    matcher: Dict[str, str] = {}
    
    for seg in expected_segments:
        # Exact lowercase match
        matcher[seg.lower()] = seg
        
        # Without footnotes
        clean = _FOOTNOTE_RE.sub("", seg).strip()
        matcher[clean.lower()] = seg
        
        # Tokenized key (sorted tokens as string)
        tokens = tokenize_label(seg)
        if tokens:
            tokens_key = " ".join(sorted(tokens))
            matcher[tokens_key] = seg
    
    return matcher


def normalize_segment_name(name: str) -> str:
    """
    Normalize a segment name for consistent output.
    
    - Strips footnote markers like "(1)"
    - Normalizes whitespace
    """
    s = str(name or "").strip()
    s = _FOOTNOTE_RE.sub("", s).strip()
    s = _WHITESPACE_RE.sub(" ", s)
    return s
