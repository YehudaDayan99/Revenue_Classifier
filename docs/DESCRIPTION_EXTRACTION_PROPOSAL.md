# Description Extraction Enhancement Proposal

## Status: PENDING REVIEW
**Date**: January 27, 2026  
**Version**: v1.0

---

## Executive Summary

This document proposes enhancements to the revenue line description extraction process to address gaps identified in META and NVDA filings. The core problem is that **product/service definitions** in Note 2 (Revenue) are often adjacent to **revenue recognition mechanics**, causing the current system to either miss definitions or include accounting language.

---

## 1. Problem Statement

### 1.1 Current State (v16)

| Ticker | Coverage | Key Issue |
|--------|----------|-----------|
| AAPL | 80% | "Services" is too generic; 6 subsection headings not detected |
| MSFT | 90% | Works well |
| GOOGL | 100% | Works well |
| AMZN | 100% | **Fixed** with DOM footnote extraction |
| META | 67% | "Other revenue" definition in Note 2, adjacent to accounting text |
| NVDA | 0% | Descriptions scattered in Item 1 narratives |

### 1.2 Developer Analysis

From manual inspection of META and NVDA 10-Ks:

**META:**
> "The clean, product/service definitions for 'Other revenue' and 'Reality Labs revenue' sit in Note 2 — Revenue, but they are adjacent to revenue recognition mechanics (impressions/actions recognition, principal vs agent, etc.). Your pipeline must be able to use Note 2 for 'what it is' while excluding 'how it's recognized.'"

**NVDA:**
> "The product/service descriptions are concentrated in Item 1 (Business) (market/platform narratives + segment enumerations). If your retrieval prioritizes Item 1 and your generator is constrained to 'what is sold / delivered,' you'll consistently land on the right text."

### 1.3 Root Cause

The current `note_revenue` section capture includes BOTH:
- ✅ **Definitions**: "Other revenue consists of WhatsApp Business Platform, Meta Verified..."
- ❌ **Accounting**: "Revenue is recognized when control transfers to the customer..."

These are often in adjacent paragraphs within Note 2, making section-level filtering insufficient.

---

## 2. Proposed Solution

### 2.1 Architecture Change: Split `note_revenue` into Sub-Sections

**Current:**
```
note_revenue → All of Note 2 content
```

**Proposed:**
```
note_revenue_sources      → "Other revenue consists of...", "Reality Labs generates..."
note_revenue_recognition  → "performance obligation", "recognized when", "principal/agent"
```

### 2.2 Implementation Plan

#### Tier 1: Chunk Classification (Highest Priority)

**File**: `revseg/rag/chunking.py`

**New Function**: `classify_note_revenue_chunk()`

```python
# Patterns indicating DEFINITION content (keep)
DEFINITION_PATTERNS = [
    r"\bconsists?\s+of\b",
    r"\bincludes?\b.*\b(?:products?|services?|offerings?)\b",
    r"\bgenerat(?:es?|ed)\s+(?:from|by)\b",
    r"\bcomprises?\b",
    r"\bprovides?\s+(?:products?|services?)\b",
]

# Patterns indicating ACCOUNTING content (exclude)
ACCOUNTING_PATTERNS = [
    r"\bperformance\s+obligat",
    r"\brecogniz(?:es?|ed|ing)\s+(?:revenue|when|upon)\b",
    r"\bprincipal\s+(?:vs\.?|versus)\s+agent\b",
    r"\bSSP\b|\bstand-alone\s+selling\s+price\b",
    r"\bASC\s+\d{3}\b",
    r"\bcontract\s+(?:liability|liabilities|asset)\b",
    r"\ballocation\b",
    r"\bcontrol\s+transfers?\b",
    r"\bsatisf(?:ied|action)\b.*\bobligation\b",
    r"\btransaction\s+price\b",
    r"\bvariable\s+consideration\b",
]

def classify_note_revenue_chunk(chunk_text: str) -> str:
    """
    Classify a chunk within note_revenue as either:
    - "note_revenue_sources" (product/service definitions)
    - "note_revenue_recognition" (accounting mechanics)
    """
    has_definition = any(re.search(p, chunk_text, re.I) for p in DEFINITION_PATTERNS)
    has_accounting = any(re.search(p, chunk_text, re.I) for p in ACCOUNTING_PATTERNS)
    
    # If both present, check density
    if has_definition and has_accounting:
        definition_count = sum(1 for p in DEFINITION_PATTERNS if re.search(p, chunk_text, re.I))
        accounting_count = sum(1 for p in ACCOUNTING_PATTERNS if re.search(p, chunk_text, re.I))
        return "note_revenue_sources" if definition_count > accounting_count else "note_revenue_recognition"
    
    if has_definition:
        return "note_revenue_sources"
    if has_accounting:
        return "note_revenue_recognition"
    
    return "note_revenue"  # Ambiguous, keep as-is
```

**Integration Point**: In `chunk_10k_structured()`, after section identification:

```python
if section == "note_revenue":
    section = classify_note_revenue_chunk(chunk_text)
```

#### Tier 2: Retrieval Filter (High Priority)

**File**: `revseg/rag/index.py`

**Change**: In `TwoTierIndex.retrieve()`, add section filtering:

```python
# BLOCKED sections - never retrieve from these
BLOCKED_SECTIONS = {
    "note_revenue_recognition",
    "item1a",  # Risk factors
}

# PREFERRED sections - boost score 1.15x
PREFERRED_SECTIONS = {
    "note_revenue_sources",
    "item1",
    "note_segment",
    "table_footnote",
}

def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
    # ... existing retrieval code ...
    
    # Filter blocked sections
    results = [(c, s) for c, s in results if c.section not in BLOCKED_SECTIONS]
    
    # Boost preferred sections
    boosted = []
    for chunk, score in results:
        if chunk.section in PREFERRED_SECTIONS:
            score *= 1.15
        boosted.append((chunk, score))
    
    return sorted(boosted, key=lambda x: -x[1])[:top_k]
```

#### Tier 3: Generation Guardrails (Medium Priority)

**File**: `revseg/rag/generation.py`

**Change 1**: Update `build_rag_query()` to remove "revenue recognition":

```python
# REMOVE this line:
# query_parts.append("revenue recognition segment note")

# REPLACE with:
query_parts.append("product service definition consists of includes")
```

**Change 2**: Add accounting token filter to `extract_candidate_products()`:

```python
# Filter out accounting tokens
ACCOUNTING_TOKENS = {"SSP", "ASC", "GAAP", "ASU", "IFRS", "allocation", "recognition"}

def extract_candidate_products(chunk_text: str) -> List[str]:
    candidates = _extract_capitalized_phrases(chunk_text)
    # Filter accounting tokens
    candidates = [c for c in candidates if c.upper() not in ACCOUNTING_TOKENS]
    return candidates
```

**Change 3**: Update generation prompt:

```python
system = """
You are extracting product/service DEFINITIONS from SEC 10-K filings.

CRITICAL RULES:
1. Describe WHAT the product/service IS and WHAT IT INCLUDES.
2. Use company language from the evidence.
3. EXCLUDE the following (return empty if only this type of text is found):
   - Accounting language: "recognized", "deferred", "performance obligation", "SSP"
   - Revenue recognition: "control transfers", "principal/agent", "transaction price"
   - Performance drivers: "increased due to", "decreased due to", "driven by"
"""
```

#### Tier 4: AAPL Services Fix (Separate Enhancement)

**Problem**: "Services" matches too early (TOC, tables) before the actual Item 1 definition.

**Solution**: Heading-based extraction for Apple-style structured sections.

**File**: `revseg/react_agents.py`

**New Function**: `_extract_heading_based_definition()`

```python
def _extract_heading_based_definition(
    html_text: str,
    label: str,
    section_text: str
) -> Optional[str]:
    """
    For Apple-style 10-Ks where products/services have dedicated headings
    in Item 1 (e.g., <b>Services</b> followed by subsections).
    
    1. Find label as a HEADING (bold/strong tag, not just text)
    2. Extract content until next major heading
    3. If subsections exist (Advertising, AppleCare, etc.), aggregate them
    """
    # Pattern: <b>Services</b> or <strong>Services</strong>
    heading_pattern = rf'<(?:b|strong)[^>]*>\s*{re.escape(label)}\s*</(?:b|strong)>'
    match = re.search(heading_pattern, html_text, re.IGNORECASE)
    
    if not match:
        return None
    
    # Extract content from heading to next major heading
    start = match.end()
    next_heading = re.search(r'<(?:b|strong)[^>]*>[A-Z][^<]{3,30}</(?:b|strong)>', html_text[start:])
    end = start + next_heading.start() if next_heading else start + 5000
    
    content = html_text[start:end]
    # Clean HTML and return
    return _clean(BeautifulSoup(content, 'lxml').get_text())
```

**Integration**: Add to `describe_revenue_lines()` before section search:

```python
# Try heading-based extraction first (for AAPL Services-style)
heading_desc = _extract_heading_based_definition(html_text, item_label, item1_section)
if heading_desc and len(heading_desc) > 50:
    descriptions[item_label] = strip_accounting_sentences(heading_desc)
    continue
```

---

## 3. Validation Plan

### 3.1 Regression Testing

Before deployment, verify no regression on currently-working tickers:

| Ticker | Current Coverage | Post-Change Target |
|--------|-----------------|-------------------|
| AMZN | 100% | 100% |
| GOOGL | 100% | 100% |
| MSFT | 90% | 90%+ |

### 3.2 Improvement Testing

| Ticker | Current | Target | Key Test |
|--------|---------|--------|----------|
| META | 67% | 100% | "Other revenue" should get WhatsApp/Verified definition |
| NVDA | 0% | 80%+ | Compute/Networking/Gaming should get Item 1 definitions |
| AAPL | 80% | 100% | Services should aggregate 6 subsections |

### 3.3 Test Cases

**META "Other revenue":**
- ✅ Should extract: "Non-ad revenue that consists of: WhatsApp Business Platform, Meta Verified subscriptions, net fees from developers using Meta's Payments infrastructure"
- ❌ Should NOT include: "Revenue is recognized at a point in time when control of the promised services transfers"

**NVDA "Compute":**
- ✅ Should extract: "Part of NVIDIA's data-center-scale accelerated computing platform for AI... compute explicitly including GPUs, CPUs, and DPUs"
- ❌ Should NOT include: Performance commentary like "increased 77% year over year"

**AAPL "Services":**
- ✅ Should extract: All 6 subsections aggregated (Advertising, AppleCare, Cloud Services, Digital Content, Subscription Services, Payment Services)
- ❌ Should NOT include: Revenue recognition or YoY performance

---

## 4. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Chunk misclassification | Medium | Medium | Calibration harness with known-good pairs |
| Regression on AMZN/GOOGL | Low | High | Automated regression tests before deploy |
| Patterns don't generalize | Medium | Medium | Review with broader corpus before scaling |
| Over-filtering valid content | Low | Medium | Err on side of inclusion, filter at generation |

---

## 5. Files to Modify

| File | Changes |
|------|---------|
| `revseg/rag/chunking.py` | Add `classify_note_revenue_chunk()`, update `chunk_10k_structured()` |
| `revseg/rag/index.py` | Add blocked/preferred section filtering to `retrieve()` |
| `revseg/rag/generation.py` | Update `build_rag_query()`, filter accounting tokens, update prompt |
| `revseg/react_agents.py` | Add `_extract_heading_based_definition()` for AAPL Services |

---

## 6. Questions for Reviewer

1. **Pattern Validation**: Do the proposed `DEFINITION_PATTERNS` and `ACCOUNTING_PATTERNS` generalize to your broader corpus of 10-Ks?

2. **Section Boundaries**: Is the heuristic "definition_count > accounting_count" robust enough, or do we need more sophisticated boundary detection?

3. **AAPL Heading Extraction**: Does Apple consistently use `<b>` or `<strong>` tags for product category headings, or are there other formats we should handle?

4. **Acceptable Empty**: For cases like "OEM and Other" where no definition exists in the 10-K, is returning an empty description the correct behavior?

---

## 7. Implementation Timeline (Estimated)

| Phase | Effort | Dependency |
|-------|--------|------------|
| Tier 1: Chunk classification | 4-6 hours | None |
| Tier 2: Retrieval filter | 2-3 hours | Tier 1 |
| Tier 3: Generation guardrails | 2-3 hours | Can parallel |
| Tier 4: AAPL heading extraction | 3-4 hours | None |
| Testing & validation | 4-6 hours | All above |
| **Total** | **15-22 hours** | |

---

## Appendix A: Developer's Ideal Output (Reference)

### META

| Revenue Line | Ideal Description | Source |
|--------------|-------------------|--------|
| Advertising | Revenue from marketers advertising on Meta's apps—ads displayed on Facebook, Instagram, Messenger. Marketers typically pay based on impressions delivered or actions (e.g., clicks) taken by users. | Note 2 — Revenue |
| Other | Non-ad revenue that consists of: WhatsApp Business Platform, Meta Verified subscriptions, net fees from developers using Meta's Payments infrastructure, plus various other sources. | Note 2 — Revenue |
| Reality Labs | Revenue generated from delivery of consumer hardware products (Meta Quest and Ray-Ban Meta AI glasses) and related software and content. | Note 2 — Revenue |

### NVDA

| Revenue Line | Ideal Description | Source |
|--------------|-------------------|--------|
| Compute | Part of NVIDIA's data-center-scale accelerated computing platform for AI: compute and networking solutions across processing units, interconnects, systems, and software, with compute explicitly including GPUs, CPUs, and DPUs. | Item 1 — Business |
| Networking | End-to-end InfiniBand and Ethernet platforms, consisting of network adapters, cables, DPUs, switch chips and systems, plus a full software stack. | Item 1 — Business |
| Gaming | GeForce GPUs for gaming and PCs, the GeForce NOW game-streaming service, plus solutions for gaming platforms. | Item 1 — Business |
| Professional Visualization | Quadro/NVIDIA RTX GPUs for enterprise workstation graphics, vGPU software for cloud-based visual/virtual computing, and Omniverse Enterprise software for industrial AI / digital-twin applications. | Item 1 — Business |
| Automotive | DRIVE branded AI-based hardware and software solution for autonomous vehicles and electric vehicles. | Item 1 — Business |
| OEM and Other | *(No explicit definition found in 10-K narrative)* | N/A |
