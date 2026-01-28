"""
RAG-based description generation with evidence gate and extractive-first approach.

Features:
- Evidence coverage gate (require ≥1 chunk from preferred section)
- Extractive-first product enumeration
- Auditable output with evidence_chunk_ids and evidence_quotes
- Post-validation of quotes
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Set, Dict, Any, TYPE_CHECKING

from .chunking import Chunk
from .index import TwoTierIndex, embed_query, PREFERRED_SECTIONS

if TYPE_CHECKING:
    from revseg.llm_client import OpenAIChatClient

# Import accounting sentence filter from react_agents
from revseg.react_agents import strip_accounting_sentences


@dataclass
class DescriptionResult:
    """Auditable description with evidence."""
    revenue_line: str
    description: str
    products_services_list: List[str] = field(default_factory=list)
    evidence_chunk_ids: List[str] = field(default_factory=list)
    evidence_quotes: List[str] = field(default_factory=list)
    retrieval_tier: str = ""  # "tier1_local" | "tier2_full" | "tier2_empty"
    validated: bool = True     # True if quotes verified in source
    evidence_gate_passed: bool = False  # True if ≥1 chunk from preferred section
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_candidate_products(chunks: List[Chunk]) -> Set[str]:
    """
    Extractive-first: Pull candidate product/service names from chunks
    using deterministic patterns BEFORE LLM filtering.
    
    Patterns:
    - Capitalized noun phrases (e.g., "Azure", "DGX Systems")
    - Trademark patterns (®, ™)
    - Model names (alphanumeric: "H100", "A100", "GeForce RTX")
    - Quoted terms
    - "including X, Y, and Z" patterns
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
        
        # Pattern 2: Single capitalized words that look like products (4+ chars)
        single_caps = re.findall(r'\b([A-Z][a-z]{3,})\b', text)
        candidates.update(single_caps)
        
        # Pattern 3: All-caps acronyms/products (e.g., AWS, GCP, DGX)
        acronyms = re.findall(r'\b([A-Z]{2,6})\b', text)
        candidates.update(a for a in acronyms if len(a) >= 2)
        
        # Pattern 4: Trademark symbols
        trademark = re.findall(r'(\b\w+[®™])', text)
        candidates.update(tm.rstrip('®™') for tm in trademark)
        
        # Pattern 5: Product model patterns (letters + numbers)
        models = re.findall(r'\b([A-Z]{1,4}\d{2,4}[A-Z]?)\b', text)
        candidates.update(models)
        
        # Pattern 6: CamelCase products (e.g., "GeForce", "iCloud", "LinkedIn")
        camel = re.findall(r'\b([A-Z]?[a-z]+[A-Z][a-z]+(?:[A-Z][a-z]+)?)\b', text)
        candidates.update(camel)
        
        # Pattern 7: Quoted product names
        quoted = re.findall(r'"([^"]{3,30})"', text)
        candidates.update(quoted)
        
        # Pattern 8: "including X, Y, and Z" patterns
        including = re.findall(
            r'includ(?:es?|ing)\s+([A-Z][^,\.]{2,40}(?:,\s*[A-Z][^,\.]{2,40})*)',
            text, re.IGNORECASE
        )
        for match in including:
            items = re.split(r',\s*(?:and\s+)?', match)
            candidates.update(i.strip() for i in items if i.strip() and len(i.strip()) > 2)
        
        # Pattern 9: "such as X, Y, and Z" patterns
        such_as = re.findall(
            r'such\s+as\s+([A-Z][^,\.]{2,40}(?:,\s*[A-Z][^,\.]{2,40})*)',
            text, re.IGNORECASE
        )
        for match in such_as:
            items = re.split(r',\s*(?:and\s+)?', match)
            candidates.update(i.strip() for i in items if i.strip() and len(i.strip()) > 2)
    
    # Filter out common false positives
    stopwords = {
        'The', 'This', 'These', 'Our', 'We', 'Company', 'Revenue',
        'Services', 'Products', 'Business', 'Segment', 'Total',
        'United', 'States', 'North', 'America', 'International',
        'Operating', 'Income', 'Net', 'Sales', 'Cost', 'Gross',
        'Year', 'Period', 'Quarter', 'Annual', 'Fiscal', 'Item',
        'Part', 'Note', 'Table', 'Form', 'SEC', 'Filed',
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December',
        'AND', 'THE', 'FOR', 'FROM', 'WITH', 'THAT', 'HAVE', 'HAS',
        'WAS', 'WERE', 'BEEN', 'BEING', 'WILL', 'WOULD', 'COULD',
    }
    
    # Accounting terms to exclude (would confuse the LLM)
    accounting_stopwords = {
        'SSP', 'ASC', 'GAAP', 'ASU', 'IFRS', 'US', 'USD',
        'Allocation', 'Recognition', 'Deferred', 'Amortization',
        'Performance', 'Obligation', 'Contract', 'Liability',
        'Principal', 'Agent', 'Transfer', 'Control', 'Price',
        'Transaction', 'Variable', 'Consideration', 'Timing',
        'Recognized', 'Recognizes', 'Accounting', 'Policy',
        'Measurement', 'Fair', 'Value', 'Standard',
    }
    stopwords.update(accounting_stopwords)
    
    candidates = {c for c in candidates if c not in stopwords and len(c) > 2}
    
    return candidates


def check_evidence_gate(
    chunks: List[Chunk],
    retrieval_tier: str,
    scores: Optional[List[float]] = None,
    high_score_threshold: float = 0.55
) -> bool:
    """
    Evidence coverage gate: require ≥1 chunk from preferred section
    OR from table-local tier OR high-scoring chunks.
    
    Prevents plausible-but-wrong narratives from generic discussions
    (e.g., "Compute capacity constraints" from risk factors).
    
    The gate is relaxed for high-scoring chunks because:
    - Some companies (AMZN) have descriptions in footnotes not tagged as preferred
    - Semantic similarity above 0.75 indicates strong relevance
    
    Args:
        chunks: Retrieved chunks
        retrieval_tier: Which tier was used ("tier1_local", "tier2_full", etc.)
        scores: Retrieval scores for chunks
        high_score_threshold: Score above which we accept non-preferred sections
    
    Returns:
        True if evidence gate passes
    """
    if not chunks:
        return False
    
    # Tier 1 (table-local) always passes - these are semantically tied to the table
    if retrieval_tier == "tier1_local":
        return True
    
    # Tier 2: check if any chunk is from preferred section
    for chunk in chunks:
        if chunk.section in PREFERRED_SECTIONS:
            return True
    
    # Relaxed gate: if we have high-scoring chunks, allow them
    # This handles cases like AMZN where footnotes aren't tagged as preferred
    if scores:
        max_score = max(scores) if scores else 0
        if max_score >= high_score_threshold:
            return True
    
    return False


def generate_description_with_evidence(
    llm: 'OpenAIChatClient',
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
    
    Args:
        llm: OpenAI chat client
        revenue_line: Revenue line label (e.g., "Compute")
        revenue_group: Revenue group (e.g., "Compute & Networking")
        chunks: Retrieved chunks
        scores: Retrieval scores
        retrieval_tier: Which tier was used
    
    Returns:
        DescriptionResult with description, evidence, and validation status
    """
    # Step 0: Empty chunks
    if not chunks:
        return DescriptionResult(
            revenue_line=revenue_line,
            description="",
            retrieval_tier=retrieval_tier,
            evidence_gate_passed=False
        )
    
    # Step 1: Evidence gate (pass scores for high-score relaxation)
    gate_passed = check_evidence_gate(chunks, retrieval_tier, scores=scores)
    if not gate_passed:
        return DescriptionResult(
            revenue_line=revenue_line,
            description="",
            retrieval_tier=retrieval_tier,
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
    system = """You are extracting product/service DEFINITIONS from SEC 10-K filings.

OUTPUT REQUIREMENTS (strict JSON):
{
  "description": "1-2 sentences describing WHAT this revenue line IS and INCLUDES. Use company language.",
  "products_services_list": ["specific", "products", "services"],
  "evidence_chunk_ids": ["chunk_0001", "chunk_0042"],
  "evidence_quotes": ["exact quoted text supporting description"]
}

CRITICAL RULES:
1. Describe WHAT the product/service IS, not how it performed.
2. Use ONLY information from the provided chunks.
3. Quote or closely paraphrase the company's own words.

4. EXCLUDE the following (return empty if only this type of text is found):
   - Revenue recognition mechanics: "recognized when", "performance obligation", "control transfers"
   - Accounting terms: "SSP", "ASC", "GAAP", "principal/agent", "transaction price", "allocation"
   - Performance drivers: "increased due to", "decreased due to", "driven by", "year over year"
   - Contract terms: "contract liability", "deferred", "unearned"

5. PREFER text that uses definitional verbs:
   - "consists of", "includes", "comprises", "provides", "offerings"
   - "products such as", "services including"

6. For products_services_list: FILTER the candidate_products to keep only those 
   that are EXPLICITLY mentioned as part of this revenue line in the chunks.
7. Include chunk_ids for ALL chunks you used.
8. Include EXACT quotes (10-50 words) that support your description.
9. If no relevant definitional information found, return empty strings/arrays.
10. Do NOT add products not in candidate_products unless clearly stated in chunks."""

    # Limit candidates to avoid prompt overflow
    sorted_candidates = sorted(candidate_products)[:40]
    
    user = f"""Revenue line: {revenue_line}
Revenue group: {revenue_group}

Candidate products (filter these to keep only relevant ones): {sorted_candidates}

Retrieved chunks from 10-K:
{context}

Extract description and filter products for "{revenue_line}"."""

    try:
        result = llm.json_call(system=system, user=user, max_output_tokens=600)
    except Exception as e:
        print(f"[RAG] LLM call failed for {revenue_line}: {e}")
        return DescriptionResult(
            revenue_line=revenue_line,
            description="",
            retrieval_tier=retrieval_tier,
            evidence_gate_passed=gate_passed,
            validated=False
        )
    
    # Step 4: Post-validation
    chunk_texts = {c.chunk_id: c.text for c in chunks}
    validated = True
    
    # Validate chunk IDs exist
    for chunk_id in result.get("evidence_chunk_ids", []):
        if chunk_id not in chunk_texts:
            validated = False
            break
    
    # Validate quotes exist in chunks
    for quote in result.get("evidence_quotes", []):
        if not quote:
            continue
        quote_lower = quote.lower()[:50]
        found = any(quote_lower in c.text.lower() for c in chunks)
        if not found:
            validated = False
            break
    
    # Step 5: Apply accounting sentence filter to remove any remaining accounting/driver language
    raw_description = result.get("description", "") if validated else ""
    filtered_description = strip_accounting_sentences(raw_description) if raw_description else ""
    
    # If filter removes all content, return empty (description was only accounting text)
    if raw_description and not filtered_description:
        print(f"[RAG] Description for '{revenue_line}' was filtered out (only accounting/driver text)")
        filtered_description = ""
    
    return DescriptionResult(
        revenue_line=revenue_line,
        description=filtered_description,
        products_services_list=result.get("products_services_list", []) if validated else [],
        evidence_chunk_ids=result.get("evidence_chunk_ids", []) if validated else [],
        evidence_quotes=result.get("evidence_quotes", []) if validated else [],
        retrieval_tier=retrieval_tier,
        validated=validated,
        evidence_gate_passed=gate_passed
    )


def build_rag_query(
    company_name: str,
    ticker: str,
    fiscal_year: int,
    revenue_line: str,
    revenue_group: str,
    table_caption: Optional[str] = None
) -> str:
    """
    Construct rich query for semantic search.
    
    Key insight: Include context (revenue group, table caption)
    to disambiguate labels like "Compute" from risk factors.
    """
    query_parts = [
        f"{company_name} ({ticker})",
        f"FY{fiscal_year}",
        f"revenue line '{revenue_line}'",
    ]
    
    if revenue_group:
        query_parts.append(f"in segment '{revenue_group}'")
    
    if table_caption:
        query_parts.append(f"Table: {table_caption}")
    
    # Focus on definitional terms that indicate "what it is" content
    # Key pattern: "[Label]. [Label] consists of..." - this is the definition sentence
    query_parts.append(f'"{revenue_line} consists of" "{revenue_line} includes"')
    query_parts.append("products and services offerings generated from")
    
    return " ".join(query_parts)


# Definition patterns for post-retrieval boosting
_DEFINITION_BOOST_PATTERNS = [
    re.compile(r'\bconsists?\s+of\b', re.IGNORECASE),
    re.compile(r'\bincludes?\b', re.IGNORECASE),
    re.compile(r'\bcomprises?\b', re.IGNORECASE),
    re.compile(r'\bgenerat(?:es?|ed)\s+from\b', re.IGNORECASE),
    re.compile(r'\bprovides?\s+(?:products?|services?)\b', re.IGNORECASE),
    re.compile(r'\bsales?\s+of\b', re.IGNORECASE),
]


def _boost_definitional_chunks(
    chunks: List[Chunk],
    scores: List[float],
    label: str,
    boost_factor: float = 1.25
) -> Tuple[List[Chunk], List[float]]:
    """
    Re-rank chunks by boosting those with BOTH the label AND a definition pattern.
    
    This addresses the META "Other revenue" problem where semantic search returns
    chunks that mention "other revenue" in tables/performance text but not the
    actual definition "Other revenue consists of WhatsApp Business Platform...".
    """
    if not chunks:
        return chunks, scores
    
    label_lower = label.lower()
    boosted = []
    
    for chunk, score in zip(chunks, scores):
        text_lower = chunk.text.lower()
        
        # Check if chunk contains the exact label
        has_label = label_lower in text_lower
        
        # Check if chunk has a definition pattern
        has_definition = any(p.search(chunk.text) for p in _DEFINITION_BOOST_PATTERNS)
        
        # Boost if both conditions are met
        if has_label and has_definition:
            boosted.append((chunk, score * boost_factor))
        else:
            boosted.append((chunk, score))
    
    # Re-sort by boosted score
    boosted.sort(key=lambda x: -x[1])
    
    return [c for c, _ in boosted], [s for _, s in boosted]


def describe_revenue_lines_rag(
    llm: 'OpenAIChatClient',
    *,
    ticker: str,
    company_name: str,
    fiscal_year: int,
    revenue_lines: List[Dict[str, Any]],
    index: TwoTierIndex,
    table_caption: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[DescriptionResult]:
    """
    Generate descriptions for multiple revenue lines using RAG.
    
    Args:
        llm: OpenAI chat client for generation
        ticker: Company ticker
        company_name: Company name
        fiscal_year: Fiscal year
        revenue_lines: List of dicts with 'item' (label) and optional 'revenue_group'
        index: Pre-built TwoTierIndex
        table_caption: Optional table caption for query context
        api_key: Optional OpenAI API key for embeddings
    
    Returns:
        List of DescriptionResult objects
    """
    results = []
    
    for line_info in revenue_lines:
        item_label = line_info.get("item", "")
        revenue_group = line_info.get("revenue_group", "")
        
        if not item_label:
            continue
        
        # Build rich query
        query = build_rag_query(
            company_name=company_name,
            ticker=ticker,
            fiscal_year=fiscal_year,
            revenue_line=item_label,
            revenue_group=revenue_group,
            table_caption=table_caption
        )
        
        # Embed query
        try:
            query_embedding = embed_query(query, api_key)
        except Exception as e:
            print(f"[RAG] Failed to embed query for {item_label}: {e}")
            results.append(DescriptionResult(
                revenue_line=item_label,
                description="",
                retrieval_tier="error",
                evidence_gate_passed=False
            ))
            continue
        
        # Retrieve chunks (over-fetch then boost+filter)
        chunks, scores, tier = index.retrieve(query_embedding, top_k=10)
        
        # P1: Post-retrieval boost for definitional chunks
        # This addresses generic labels like "Other revenue" where the definition
        # chunk may have lower raw semantic score than table/performance mentions
        chunks, scores = _boost_definitional_chunks(chunks, scores, item_label)
        
        # Trim to top_k after boosting
        chunks = chunks[:5]
        scores = scores[:5]
        
        # Optional debug: uncomment to see what's being retrieved
        # print(f"[RAG DEBUG] {item_label}: {len(chunks)} chunks (tier={tier}), top={scores[0]:.3f}" if chunks else f"[RAG DEBUG] {item_label}: NO CHUNKS")
        
        # Generate description
        result = generate_description_with_evidence(
            llm=llm,
            revenue_line=item_label,
            revenue_group=revenue_group,
            chunks=chunks,
            scores=scores,
            retrieval_tier=tier
        )
        
        results.append(result)
    
    return results
