"""
Hybrid Retriever combining Vector Search and BM25

This module combines semantic vector search (Qdrant) with lexical BM25 search
for improved retrieval performance. Optionally supports cross-encoder re-ranking.
"""

from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from loguru import logger

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index
from src.retrieval.reranker import CrossEncoderReranker


class HybridRetriever:
    """
    Hybrid retrieval system combining vector and BM25 search
    with optional cross-encoder re-ranking
    """
    
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        alpha: float = 0.5,
        use_reranker: bool = False,
        reranker_model: Optional[str] = None
    ):
        """
        Initialize hybrid retriever
        
        Args:
            embedder: Embedder instance for query encoding
            vector_store: Vector store for semantic search
            bm25_index: BM25 index for lexical search
            alpha: Weight for combining scores (0=BM25 only, 1=vector only, 0.5=balanced)
            use_reranker: Whether to use cross-encoder re-ranking
            reranker_model: Cross-encoder model name (optional)
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.alpha = alpha
        self.use_reranker = use_reranker
        
        # Initialize reranker if requested
        self.reranker = None
        if use_reranker:
            self.reranker = CrossEncoderReranker(
                model_name=reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            logger.info(f"HybridRetriever initialized with re-ranking (alpha={alpha})")
        else:
            logger.info(f"HybridRetriever initialized (alpha={alpha})")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filter_dict: Optional[Dict] = None,
        rerank_top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve documents using hybrid search with optional re-ranking
        
        Args:
            query: Search query
            top_k: Number of results to return (before re-ranking)
            filter_dict: Optional metadata filter for vector search
            rerank_top_k: Number of results after re-ranking (if None, uses top_k)
        
        Returns:
            List of results with combined scores (and rerank scores if enabled)
        """
        logger.info(f"Hybrid search for query: '{query[:50]}...'")
        
        # 1. Vector search
        query_embedding = self.embedder.embed_text(query)
        vector_results = self.vector_store.search(
            query_embedding,
            top_k=top_k * 2,  # Get more candidates
            filter_dict=filter_dict
        )
        
        # 2. BM25 search
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)
        
        # 3. Combine results
        combined_results = self._combine_results(
            vector_results,
            bm25_results,
            alpha=self.alpha
        )
        
        # 4. Get top-k candidates for re-ranking
        candidates = combined_results[:top_k]
        
        # 5. Optional re-ranking
        if self.use_reranker and self.reranker:
            final_top_k = rerank_top_k or (top_k // 4)  # Default to top 25% after reranking
            final_results = self.reranker.rerank(query, candidates, top_k=final_top_k)
            logger.success(
                f"✓ Hybrid search with re-ranking: "
                f"{len(vector_results)} vector + {len(bm25_results)} BM25 "
                f"→ {len(candidates)} candidates → {len(final_results)} re-ranked"
            )
        else:
            final_results = candidates
            logger.success(
                f"✓ Hybrid search complete: "
                f"{len(vector_results)} vector + {len(bm25_results)} BM25 "
                f"→ {len(final_results)} final results"
            )
        
        return final_results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-max normalization of scores to [0, 1]"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def _combine_results(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        alpha: float
    ) -> List[Dict]:
        """
        Combine vector and BM25 results using weighted sum
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            alpha: Weight (0-1) for combining scores
        
        Returns:
            Combined and sorted results
        """
        # Normalize scores
        vector_scores = [r["score"] for r in vector_results]
        bm25_scores = [r["score"] for r in bm25_results]
        
        vector_scores_norm = self._normalize_scores(vector_scores)
        bm25_scores_norm = self._normalize_scores(bm25_scores)
        
        # Create score dictionaries (keyed by text for matching)
        vector_score_dict = {}
        for result, norm_score in zip(vector_results, vector_scores_norm):
            text = result.get("text", "")
            vector_score_dict[text] = {
                "score": norm_score,
                "result": result
            }
        
        bm25_score_dict = {}
        for result, norm_score in zip(bm25_results, bm25_scores_norm):
            text = result.get("text", "")
            bm25_score_dict[text] = {
                "score": norm_score,
                "result": result
            }
        
        # Combine scores
        all_texts = set(vector_score_dict.keys()) | set(bm25_score_dict.keys())
        combined_results = []
        
        for text in all_texts:
            vector_score = vector_score_dict.get(text, {}).get("score", 0.0)
            bm25_score = bm25_score_dict.get(text, {}).get("score", 0.0)
            
            # Weighted combination
            combined_score = alpha * vector_score + (1 - alpha) * bm25_score
            
            # Get result metadata (prefer vector search result for metadata)
            if text in vector_score_dict:
                result = vector_score_dict[text]["result"]
            else:
                result = bm25_score_dict[text]["result"]
            
            combined_results.append({
                **result,
                "combined_score": combined_score,
                "vector_score": vector_score,
                "bm25_score": bm25_score
            })
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return combined_results
    
    def set_alpha(self, alpha: float):
        """Update alpha parameter"""
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.alpha = alpha
        logger.info(f"Alpha updated to {alpha}")


def main():
    """Example usage"""
    from src.config import settings
    
    print("\n" + "="*70)
    print("HybridRetriever Demo")
    print("="*70)
    
    # Sample documents
    documents = [
        "Transformers use self-attention mechanisms to process sequences in parallel.",
        "BERT is a bidirectional transformer for natural language understanding.",
        "Graph neural networks operate on graph-structured data.",
        "Convolutional neural networks excel at computer vision tasks.",
        "Recurrent neural networks process sequential data with memory.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Deep learning has revolutionized artificial intelligence applications.",
        "Neural networks learn hierarchical representations from data."
    ]
    
    metadatas = [
        {"id": str(i), "source": f"doc{i}"} 
        for i in range(len(documents))
    ]
    
    # Initialize components
    print("\n1. Initializing components...")
    embedder = Embedder(settings.EMBEDDING_MODEL)
    
    vector_store = VectorStore(
        collection_name="hybrid_test",
        embedding_dim=embedder.get_dimension(),
        persist_dir=settings.VECTOR_DB_DIR / "hybrid_test"
    )
    
    bm25_index = BM25Index()
    
    # Build indices
    print("\n2. Building indices...")
    embeddings = embedder.embed_batch(documents, show_progress=False)
    vector_store.add_documents(documents, embeddings, metadatas)
    bm25_index.build_index(documents, metadatas)
    
    # Create hybrid retriever
    hybrid = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25_index,
        alpha=0.5  # Balanced combination
    )
    
    # Test queries
    queries = [
        "attention mechanisms in transformers",
        "graph neural networks",
        "deep learning applications"
    ]
    
    print("\n3. Testing hybrid search...")
    for query in queries:
        print(f"\n   Query: '{query}'")
        results = hybrid.retrieve(query, top_k=3)
        
        print(f"   Top {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n   {i}. Combined: {result['combined_score']:.4f} "
                  f"(Vector: {result['vector_score']:.4f}, "
                  f"BM25: {result['bm25_score']:.4f})")
            print(f"      {result['text'][:60]}...")
    
    # Compare alpha values
    print("\n4. Testing different alpha values...")
    test_query = "attention mechanisms"
    
    for alpha_val in [0.0, 0.5, 1.0]:
        hybrid.set_alpha(alpha_val)
        results = hybrid.retrieve(test_query, top_k=2)
        
        alpha_name = {0.0: "BM25 only", 0.5: "Balanced", 1.0: "Vector only"}[alpha_val]
        print(f"\n   Alpha {alpha_val} ({alpha_name}):")
        for i, result in enumerate(results, 1):
            print(f"   {i}. Score: {result['combined_score']:.4f} - {result['text'][:50]}...")
    
    # Cleanup
    print("\n5. Cleanup...")
    vector_store.delete_collection()
    print("   ✓ Vector store cleaned")
    
    print("\n" + "="*70)
    print("✓ HybridRetriever working correctly!")
    print("="*70)


if __name__ == "__main__":
    main()

