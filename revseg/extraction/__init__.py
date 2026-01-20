"""Unified revenue extraction module for 10-K filings."""

from revseg.extraction.core import (
    ExtractionResult,
    ExtractedRow,
    extract_revenue_unified,
    extract_with_layout_fallback,
    extract_line_items_granular,
)
from revseg.extraction.matching import (
    fuzzy_match_segment,
    tokenize_label,
    build_segment_matcher,
)
from revseg.extraction.validation import (
    ValidationResult,
    validate_extraction,
)

__all__ = [
    "ExtractionResult",
    "ExtractedRow",
    "extract_revenue_unified",
    "extract_with_layout_fallback",
    "extract_line_items_granular",
    "fuzzy_match_segment",
    "tokenize_label",
    "build_segment_matcher",
    "ValidationResult",
    "validate_extraction",
]
