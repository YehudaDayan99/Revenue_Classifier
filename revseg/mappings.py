"""
Item-to-Segment mappings for companies where the disaggregation table
does not include segment labels directly.

These mappings are based on the segment definitions in each company's 10-K.
"""

from typing import Dict, Optional

# MSFT: Revenue by Products and Services table has no segment column
# Mapping based on Note 18 segment definitions in 10-K
MSFT_ITEM_TO_SEGMENT: Dict[str, str] = {
    # Intelligent Cloud
    "Server products and cloud services": "Intelligent Cloud",
    "Enterprise and partner services": "Intelligent Cloud",
    
    # Productivity and Business Processes
    "Microsoft 365 Commercial products and cloud services": "Productivity and Business Processes",
    "LinkedIn": "Productivity and Business Processes",
    "Dynamics products and cloud services": "Productivity and Business Processes",
    "Microsoft 365 Consumer products and cloud services": "Productivity and Business Processes",
    
    # More Personal Computing
    "Gaming": "More Personal Computing",
    "Windows and Devices": "More Personal Computing",
    "Search and news advertising": "More Personal Computing",
    
    # Other/Adjustments
    "Other": "Other",
}

# GOOGL: Item-to-segment mappings
# Google Services is a parent segment with sub-items
# Google Cloud and Other Bets are standalone segments
GOOGL_ITEM_TO_SEGMENT: Dict[str, str] = {
    # Google Services sub-items (advertising)
    "Google Search & other": "Google Services",
    "YouTube ads": "Google Services",
    "Google Network": "Google Services",
    # Google Services sub-items (subscriptions)
    "Google subscriptions, platforms, and devices": "Google Services",
    # Standalone segments
    "Google Cloud": "Google Cloud",
    "Other Bets": "Other Bets",
}

# GOOGL: Items that are subtotals (to be excluded)
GOOGL_SUBTOTAL_ITEMS: set = {
    "Google Services",
    "Google Services total",
    "Google advertising",  # Subtotal of Search + YouTube + Network
}

# GOOGL: Adjustment items
GOOGL_ADJUSTMENT_ITEMS: set = {
    "Hedging gains (losses)",
    "Hedging gains",
    "Hedging losses",
}

# AAPL: Product categories are the line items, segment is effectively "Apple"
# No mapping needed - items are already at the right granularity


def get_segment_for_item(ticker: str, item_label: str) -> Optional[str]:
    """
    Look up segment for a given item label.
    Returns None if no mapping exists (use LLM or default to item as segment).
    """
    ticker_upper = ticker.upper()
    item_clean = item_label.strip()
    
    # Get the appropriate mapping dict
    mapping: Dict[str, str] = {}
    if ticker_upper == "MSFT":
        mapping = MSFT_ITEM_TO_SEGMENT
    elif ticker_upper == "GOOGL":
        mapping = GOOGL_ITEM_TO_SEGMENT
    
    if mapping:
        # Try exact match first
        if item_clean in mapping:
            return mapping[item_clean]
        # Try case-insensitive match
        for key, segment in mapping.items():
            if key.lower() == item_clean.lower():
                return segment
    
    return None


def is_adjustment_item(ticker: str, item_label: str) -> bool:
    """
    Check if an item is an adjustment line (hedging, etc.) rather than a product/service.
    """
    item_lower = item_label.lower().strip()
    
    # General adjustment patterns
    adjustment_patterns = [
        "hedging",
        "hedge",
        "corporate",
        "elimination",
        "reconcil",
    ]
    
    for pattern in adjustment_patterns:
        if pattern in item_lower:
            return True
    
    # GOOGL-specific
    if ticker.upper() == "GOOGL":
        if item_label.strip() in GOOGL_ADJUSTMENT_ITEMS:
            return True
    
    return False


def is_total_row(label: str) -> bool:
    """
    Check if a row label indicates a total/subtotal row that should be excluded
    from line item extraction.
    """
    label_lower = label.lower().strip()
    
    total_patterns = [
        "total revenue",
        "total revenues",
        "total net sales",
        "total net revenue",
    ]
    
    for pattern in total_patterns:
        if label_lower == pattern or label_lower.startswith(pattern):
            return True
    
    return False


def is_subtotal_row(label: str, ticker: str = "") -> bool:
    """
    Check if a row is a segment subtotal that should be excluded when we have
    more granular line items.
    """
    label_clean = label.strip()
    label_lower = label_clean.lower()
    
    ticker_upper = ticker.upper()
    
    # GOOGL-specific subtotals
    if ticker_upper == "GOOGL":
        if label_clean in GOOGL_SUBTOTAL_ITEMS:
            return True
        # Case-insensitive check
        for item in GOOGL_SUBTOTAL_ITEMS:
            if item.lower() == label_lower:
                return True
        return False
    
    # MSFT: segment names in the revenue table are NOT subtotals
    # (the table is already at the product level)
    if ticker_upper == "MSFT":
        return False
    
    return False
