"""
Two-tier FAISS index for RAG retrieval.

Features:
- Tier 1: Table-local context (high precision)
- Tier 2: Full filing (high recall fallback)
- FAISS binary storage (not JSON embeddings)
- MMR deduplication
- Section boosting
"""

from __future__ import annotations

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import asdict

from .chunking import Chunk

# OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def embed_chunks(chunks: List[str], api_key: Optional[str] = None) -> List[List[float]]:
    """
    Embed text chunks using OpenAI.
    
    Cost: ~$0.002 per 10-K (100k tokens)
    
    Args:
        chunks: List of text strings to embed
        api_key: OpenAI API key (uses env var if not provided)
    
    Returns:
        List of embedding vectors (1536 dimensions each)
    """
    import requests
    from revseg.secrets import get_openai_api_key
    
    key = api_key or get_openai_api_key()
    if not key:
        raise ValueError("OPENAI_API_KEY not set")
    
    # Batch in groups of 100 (OpenAI limit is 2048)
    all_embeddings = []
    batch_size = 100
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            },
            json={
                "model": EMBEDDING_MODEL,
                "input": batch
            },
            timeout=60
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI embedding error: {response.text}")
        
        data = response.json()
        batch_embeddings = [item["embedding"] for item in data["data"]]
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings


def embed_query(query: str, api_key: Optional[str] = None) -> List[float]:
    """Embed a single query string."""
    embeddings = embed_chunks([query], api_key)
    return embeddings[0]


# Preferred sections for description extraction
# P1: Added note_revenue_sources (definitional content), removed item7 (MD&A - drivers/performance)
PREFERRED_SECTIONS = {'note_revenue_sources', 'note_revenue', 'note_segment', 'item1', 'table_footnote', 'table_before'}

# P1: Sections to BLOCK from retrieval (accounting/recognition mechanics)
BLOCKED_SECTIONS = {'note_revenue_recognition'}


class TwoTierIndex:
    """
    Two-tier embedding index:
    - Tier 1: Table-local context (high precision)
    - Tier 2: Full filing (high recall fallback)
    
    Storage:
    - FAISS binary indexes (.faiss)
    - Metadata JSON (.json)
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
        
        # Calibrated thresholds (can be overridden)
        self.calibrated_local_threshold: Optional[float] = None
        self.calibrated_global_threshold: Optional[float] = None
    
    def build(
        self,
        table_local_chunks: List[Chunk],
        full_filing_chunks: List[Chunk],
        embeddings_local: List[List[float]],
        embeddings_full: List[List[float]]
    ):
        """Build both indexes from chunks and embeddings."""
        # Tier 1: Table-local
        self.local_chunks = table_local_chunks
        if table_local_chunks and embeddings_local:
            local_matrix = np.array(embeddings_local, dtype=np.float32)
            faiss.normalize_L2(local_matrix)
            self.local_index = faiss.IndexFlatIP(local_matrix.shape[1])
            self.local_index.add(local_matrix)
        
        # Tier 2: Full filing
        self.full_chunks = full_filing_chunks
        if full_filing_chunks and embeddings_full:
            full_matrix = np.array(embeddings_full, dtype=np.float32)
            faiss.normalize_L2(full_matrix)
            self.full_index = faiss.IndexFlatIP(full_matrix.shape[1])
            self.full_index.add(full_matrix)
        
        # Save to disk
        self._save_cache(embeddings_local, embeddings_full)
    
    def retrieve(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        local_threshold: Optional[float] = None,
        global_threshold: Optional[float] = None,
        prefer_sections: Optional[List[str]] = None
    ) -> Tuple[List[Chunk], List[float], str]:
        """
        Two-tier retrieval with quality controls.
        
        Process:
        1. Try table-local index first (Tier 1)
        2. If max score < local_threshold, fall back to full-filing (Tier 2)
        3. Apply section boosting and MMR deduplication
        
        Args:
            query_embedding: Query vector (1536 dims)
            top_k: Number of chunks to return
            local_threshold: Minimum score for Tier 1 (default: calibrated or 0.70)
            global_threshold: Minimum score for Tier 2 (default: calibrated or 0.60)
            prefer_sections: Sections to boost in Tier 2
        
        Returns:
            (chunks, scores, tier_used)
            tier_used: "tier1_local" | "tier2_full" | "tier2_empty"
        """
        # Use calibrated or default thresholds
        # Note: 0.50/0.45 are more permissive to handle varied 10-K formats
        local_threshold = local_threshold or self.calibrated_local_threshold or 0.55
        global_threshold = global_threshold or self.calibrated_global_threshold or 0.45
        prefer_sections = prefer_sections or list(PREFERRED_SECTIONS)
        
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)
        
        # Tier 1: Try table-local first
        if self.local_index is not None and self.local_index.ntotal > 0:
            k = min(top_k, self.local_index.ntotal)
            scores, indices = self.local_index.search(query, k)
            
            # Filter by threshold and exclude TOC / blocked section chunks
            valid = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < 0 or idx >= len(self.local_chunks):
                    continue
                chunk = self.local_chunks[idx]
                # P1: Skip blocked sections (accounting/recognition mechanics)
                if chunk.section in BLOCKED_SECTIONS:
                    continue
                if score >= local_threshold and not chunk.is_toc:
                    valid.append((chunk, float(score)))
            
            if valid:
                chunks, chunk_scores = zip(*valid)
                # Apply MMR for diversity
                chunks, chunk_scores = self._apply_mmr(list(chunks), list(chunk_scores))
                return list(chunks[:top_k]), list(chunk_scores[:top_k]), "tier1_local"
        
        # Tier 2: Full filing fallback
        if self.full_index is None or self.full_index.ntotal == 0:
            return [], [], "tier2_empty"
        
        k = min(top_k * 3, self.full_index.ntotal)  # Over-fetch for filtering
        scores, indices = self.full_index.search(query, k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.full_chunks):
                continue
            if score < global_threshold:
                continue
            
            chunk = self.full_chunks[idx]
            
            # Skip TOC chunks
            if chunk.is_toc:
                continue
            
            # P1: Skip blocked sections (accounting/recognition mechanics)
            if chunk.section in BLOCKED_SECTIONS:
                continue
            
            adjusted_score = float(score)
            
            # Section boosting
            if chunk.section in prefer_sections:
                adjusted_score *= 1.15  # Boost preferred sections
            elif chunk.section in ('item1a', 'risk_factors', 'liquidity'):
                adjusted_score *= 0.75  # Deprioritize irrelevant sections
            
            results.append((chunk, adjusted_score))
        
        if not results:
            return [], [], "tier2_empty"
        
        # Sort by adjusted score
        results.sort(key=lambda x: -x[1])
        chunks, chunk_scores = zip(*results[:top_k * 2])
        
        # Apply MMR
        chunks, chunk_scores = self._apply_mmr(list(chunks), list(chunk_scores))
        return list(chunks[:top_k]), list(chunk_scores[:top_k]), "tier2_full"
    
    def _apply_mmr(
        self,
        chunks: List[Chunk],
        scores: List[float],
        similarity_threshold: float = 0.85
    ) -> Tuple[List[Chunk], List[float]]:
        """
        Maximal Marginal Relevance for diversity.
        Removes near-duplicate chunks based on text similarity.
        """
        if len(chunks) <= 1:
            return chunks, scores
        
        # Simple dedup: remove chunks with very similar text
        seen_texts = set()
        deduped = []
        deduped_scores = []
        
        for chunk, score in zip(chunks, scores):
            # Hash first 150 chars (normalized) for dedup
            text_key = ' '.join(chunk.text[:150].lower().split())
            
            # Check if we've seen similar text
            is_duplicate = False
            for seen in seen_texts:
                if self._text_similarity(text_key, seen) > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_texts.add(text_key)
                deduped.append(chunk)
                deduped_scores.append(score)
        
        return deduped, deduped_scores
    
    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """Simple Jaccard similarity for text deduplication."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    def _save_cache(
        self,
        embeddings_local: List[List[float]],
        embeddings_full: List[List[float]]
    ):
        """Save indexes and metadata to disk."""
        ticker_dir = self.cache_dir / self.ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS indexes as binary
        if self.local_index is not None:
            faiss.write_index(
                self.local_index,
                str(ticker_dir / "local.faiss")
            )
        
        if self.full_index is not None:
            faiss.write_index(
                self.full_index,
                str(ticker_dir / "full.faiss")
            )
        
        # Save chunk metadata as JSON
        metadata = {
            "ticker": self.ticker,
            "local_chunks": [self._chunk_to_dict(c) for c in self.local_chunks],
            "full_chunks": [self._chunk_to_dict(c) for c in self.full_chunks],
            "calibrated_local_threshold": self.calibrated_local_threshold,
            "calibrated_global_threshold": self.calibrated_global_threshold,
        }
        (ticker_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    
    def load_from_cache(self) -> bool:
        """Load from disk cache. Returns True if successful."""
        ticker_dir = self.cache_dir / self.ticker
        
        try:
            # Load metadata first
            metadata_path = ticker_dir / "metadata.json"
            if not metadata_path.exists():
                return False
            
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            
            # Load FAISS indexes
            local_path = ticker_dir / "local.faiss"
            if local_path.exists():
                self.local_index = faiss.read_index(str(local_path))
            
            full_path = ticker_dir / "full.faiss"
            if full_path.exists():
                self.full_index = faiss.read_index(str(full_path))
            
            # Load chunk metadata
            self.local_chunks = [self._dict_to_chunk(d) for d in metadata.get("local_chunks", [])]
            self.full_chunks = [self._dict_to_chunk(d) for d in metadata.get("full_chunks", [])]
            
            # Load calibrated thresholds
            self.calibrated_local_threshold = metadata.get("calibrated_local_threshold")
            self.calibrated_global_threshold = metadata.get("calibrated_global_threshold")
            
            return True
        except Exception as e:
            print(f"[{self.ticker}] Failed to load from cache: {e}")
            return False
    
    def cache_exists(self) -> bool:
        """Check if cache exists for this ticker."""
        ticker_dir = self.cache_dir / self.ticker
        return (ticker_dir / "metadata.json").exists()
    
    @staticmethod
    def _chunk_to_dict(chunk: Chunk) -> Dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "section": chunk.section,
            "heading": chunk.heading,
            "table_id": chunk.table_id,
            "char_range": list(chunk.char_range),
            "is_toc": chunk.is_toc,
        }
    
    @staticmethod
    def _dict_to_chunk(d: Dict[str, Any]) -> Chunk:
        return Chunk(
            chunk_id=d["chunk_id"],
            text=d["text"],
            section=d["section"],
            heading=d.get("heading"),
            table_id=d.get("table_id"),
            char_range=tuple(d["char_range"]),
            is_toc=d.get("is_toc", False),
        )
