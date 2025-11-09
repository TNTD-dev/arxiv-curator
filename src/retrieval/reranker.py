"""
Cross-Encoder Re-ranker for improving retrieval results

This module provides cross-encoder based re-ranking to improve the quality
of retrieved documents by computing relevance scores with bidirectional attention.
"""

from typing import List, Dict, Tuple
from loguru import logger
import torch
from sentence_transformers import CrossEncoder
import time


class CrossEncoderReranker:
    """
    Cross-encoder based re-ranker for semantic search results
    
    Uses a fine-tuned cross-encoder model to compute relevance scores
    between query and document pairs with full bidirectional attention.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder re-ranker
        
        Args:
            model_name: HuggingFace model name for cross-encoder
                       Default: ms-marco-MiniLM-L-6-v2 (fast, good quality)
                       Alternatives:
                       - cross-encoder/ms-marco-MiniLM-L-12-v2 (better quality)
                       - cross-encoder/ms-marco-TinyBERT-L-2-v2 (faster)
        """
        logger.info(f"Loading cross-encoder model: {model_name}")
        start_time = time.time()
        
        # Load model with GPU support if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=self.device)
        self.model_name = model_name
        
        load_time = time.time() - start_time
        logger.success(f"✓ Cross-encoder loaded on {self.device} in {load_time:.2f}s")
    
    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Re-rank search results using cross-encoder
        
        Args:
            query: Search query
            results: List of search results with 'text' field
            top_k: Number of top results to return after re-ranking
        
        Returns:
            Re-ranked results with 'rerank_score' field
        """
        if not results:
            return []
        
        logger.info(f"Re-ranking {len(results)} results for query: '{query[:50]}...'")
        start_time = time.time()
        
        # Prepare query-document pairs
        pairs = [(query, result.get('text', '')) for result in results]
        
        # Compute cross-encoder scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Add scores to results
        reranked_results = []
        for result, score in zip(results, scores):
            reranked_result = result.copy()
            reranked_result['rerank_score'] = float(score)
            # Keep original scores for comparison
            if 'combined_score' in result:
                reranked_result['original_score'] = result['combined_score']
            reranked_results.append(reranked_result)
        
        # Sort by rerank score
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Return top-k
        final_results = reranked_results[:top_k]
        
        rerank_time = time.time() - start_time
        logger.success(
            f"✓ Re-ranking complete: {len(results)} → {len(final_results)} results "
            f"in {rerank_time:.3f}s"
        )
        
        return final_results
    
    def rerank_with_batch(
        self,
        queries: List[str],
        results_list: List[List[Dict]],
        top_k: int = 5
    ) -> List[List[Dict]]:
        """
        Re-rank multiple queries in batch (more efficient)
        
        Args:
            queries: List of queries
            results_list: List of result lists (one per query)
            top_k: Number of top results per query
        
        Returns:
            List of re-ranked results for each query
        """
        logger.info(f"Batch re-ranking {len(queries)} queries")
        
        all_reranked = []
        for query, results in zip(queries, results_list):
            reranked = self.rerank(query, results, top_k)
            all_reranked.append(reranked)
        
        return all_reranked
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model": self.model_name,
            "device": self.device,
            "type": "cross-encoder"
        }


class RerankerComparator:
    """
    Utility for comparing retrieval with and without re-ranking
    """
    
    @staticmethod
    def compare_results(
        original_results: List[Dict],
        reranked_results: List[Dict],
        top_k: int = 5
    ) -> Dict:
        """
        Compare original and re-ranked results
        
        Args:
            original_results: Original retrieval results
            reranked_results: Re-ranked results
            top_k: Number of top results to compare
        
        Returns:
            Comparison metrics
        """
        # Get top-k from both
        original_top_k = original_results[:top_k]
        reranked_top_k = reranked_results[:top_k]
        
        # Extract IDs/texts for comparison
        original_texts = set(r.get('text', '')[:100] for r in original_top_k)
        reranked_texts = set(r.get('text', '')[:100] for r in reranked_top_k)
        
        # Calculate overlap
        overlap = len(original_texts & reranked_texts)
        overlap_ratio = overlap / top_k if top_k > 0 else 0
        
        # Calculate position changes
        position_changes = []
        for i, orig_result in enumerate(original_top_k):
            orig_text = orig_result.get('text', '')[:100]
            # Find in reranked
            for j, rerank_result in enumerate(reranked_top_k):
                if rerank_result.get('text', '')[:100] == orig_text:
                    position_changes.append(abs(i - j))
                    break
        
        avg_position_change = sum(position_changes) / len(position_changes) if position_changes else 0
        
        # Score improvements
        score_improvements = []
        for rerank_result in reranked_top_k:
            if 'rerank_score' in rerank_result and 'original_score' in rerank_result:
                # Normalize to 0-1 for comparison
                improvement = rerank_result['rerank_score'] - rerank_result['original_score']
                score_improvements.append(improvement)
        
        return {
            "overlap": overlap,
            "overlap_ratio": overlap_ratio,
            "avg_position_change": avg_position_change,
            "max_position_change": max(position_changes) if position_changes else 0,
            "score_changes": {
                "mean": sum(score_improvements) / len(score_improvements) if score_improvements else 0,
                "min": min(score_improvements) if score_improvements else 0,
                "max": max(score_improvements) if score_improvements else 0
            }
        }
    
    @staticmethod
    def print_comparison(original: List[Dict], reranked: List[Dict], query: str):
        """
        Print a visual comparison of results
        
        Args:
            original: Original results
            reranked: Re-ranked results
            query: Search query
        """
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}\n")
        
        print("BEFORE RE-RANKING:")
        print("-"*80)
        for i, result in enumerate(original[:5], 1):
            score = result.get('combined_score', result.get('score', 0))
            text = result.get('text', '')[:100]
            print(f"{i}. Score: {score:.4f}")
            print(f"   {text}...")
            print()
        
        print("\nAFTER RE-RANKING:")
        print("-"*80)
        for i, result in enumerate(reranked[:5], 1):
            rerank_score = result.get('rerank_score', 0)
            original_score = result.get('original_score', 0)
            text = result.get('text', '')[:100]
            change = "↑" if rerank_score > original_score else "↓" if rerank_score < original_score else "="
            print(f"{i}. Rerank: {rerank_score:.4f} {change} (Original: {original_score:.4f})")
            print(f"   {text}...")
            print()
        
        # Show metrics
        metrics = RerankerComparator.compare_results(original, reranked, 5)
        print("\nMETRICS:")
        print("-"*80)
        print(f"Overlap ratio: {metrics['overlap_ratio']:.2%} ({metrics['overlap']}/5 same documents)")
        print(f"Avg position change: {metrics['avg_position_change']:.2f}")
        print(f"Max position change: {metrics['max_position_change']}")
        print(f"{'='*80}\n")


def main():
    """Example usage and testing"""
    print("\n" + "="*80)
    print("Cross-Encoder Re-ranker Demo")
    print("="*80)
    
    # Sample query and documents
    query = "What are transformer models in deep learning?"
    
    # Simulate retrieval results (with varying relevance)
    documents = [
        "Convolutional neural networks are good for image processing tasks.",
        "Transformers use self-attention mechanisms for sequence processing.",
        "Deep learning models require large amounts of training data.",
        "The transformer architecture revolutionized natural language processing.",
        "BERT is a bidirectional transformer model for language understanding.",
        "Graph neural networks work with graph-structured data.",
        "Attention mechanisms allow models to focus on relevant parts of input.",
        "Recurrent neural networks process sequential data with hidden states."
    ]
    
    # Create mock results with random scores
    import random
    results = [
        {
            "text": doc,
            "combined_score": random.uniform(0.4, 0.8),
            "metadata": {"id": i}
        }
        for i, doc in enumerate(documents)
    ]
    
    # Sort by combined score (simulating hybrid search)
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    print("\n1. Original Results (by hybrid search):")
    print("-"*80)
    for i, r in enumerate(results[:5], 1):
        print(f"{i}. Score: {r['combined_score']:.4f}")
        print(f"   {r['text'][:70]}...")
    
    # Initialize re-ranker
    print("\n2. Initializing Cross-Encoder Re-ranker...")
    print("-"*80)
    reranker = CrossEncoderReranker()
    
    # Re-rank
    print("\n3. Re-ranking results...")
    print("-"*80)
    reranked = reranker.rerank(query, results, top_k=5)
    
    print("\n4. Re-ranked Results:")
    print("-"*80)
    for i, r in enumerate(reranked, 1):
        print(f"{i}. Rerank Score: {r['rerank_score']:.4f} (Original: {r['original_score']:.4f})")
        print(f"   {r['text'][:70]}...")
    
    # Comparison
    print("\n5. Comparison Analysis:")
    print("-"*80)
    metrics = RerankerComparator.compare_results(results, reranked, 5)
    print(f"Overlap: {metrics['overlap']}/5 documents remain in top-5")
    print(f"Overlap ratio: {metrics['overlap_ratio']:.1%}")
    print(f"Average position change: {metrics['avg_position_change']:.2f}")
    print(f"Max position change: {metrics['max_position_change']}")
    
    # Visual comparison
    RerankerComparator.print_comparison(results, reranked, query)
    
    print("="*80)
    print("✓ Re-ranker Demo Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

