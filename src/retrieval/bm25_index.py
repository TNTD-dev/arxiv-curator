"""
BM25 Index for lexical/keyword-based search

This module provides BM25 ranking for traditional keyword search,
complementing the semantic vector search.
"""

from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional
import pickle
from pathlib import Path
from loguru import logger
import re


class BM25Index:
    """
    BM25 index for lexical search
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        """
        Initialize BM25 index
        
        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
            epsilon: Floor value for IDF (default: 0.25)
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.bm25 = None
        self.documents = []
        self.metadatas = []
        self.tokenized_corpus = []
        
        logger.info(f"BM25Index initialized (k1={k1}, b={b}, epsilon={epsilon})")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization (can be improved with more sophisticated methods)
        
        Args:
            text: Input text
        
        Returns:
            List of tokens
        """
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def build_index(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None
    ):
        """
        Build BM25 index from documents
        
        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries
        """
        if not documents:
            logger.warning("No documents provided to build index")
            return
        
        logger.info(f"Building BM25 index for {len(documents)} documents...")
        
        self.documents = documents
        self.metadatas = metadatas if metadatas else [{} for _ in documents]
        
        # Tokenize all documents
        self.tokenized_corpus = [self.tokenize(doc) for doc in documents]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon
        )
        
        logger.success(f"✓ BM25 index built with {len(documents)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Search using BM25 ranking
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of results with scores and metadata
        """
        if self.bm25 is None:
            logger.error("BM25 index not built. Call build_index() first.")
            return []
        
        # Tokenize query
        query_tokens = self.tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                results.append({
                    "index": idx,
                    "score": float(scores[idx]),
                    "text": self.documents[idx],
                    "metadata": self.metadatas[idx]
                })
        
        logger.info(f"BM25 search found {len(results)} results for query: '{query[:50]}...'")
        return results
    
    def save(self, filepath: Path):
        """
        Save BM25 index to disk
        
        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "documents": self.documents,
            "metadatas": self.metadatas,
            "tokenized_corpus": self.tokenized_corpus
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.success(f"✓ BM25 index saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving BM25 index: {e}")
            raise
    
    def load(self, filepath: Path):
        """
        Load BM25 index from disk
        
        Args:
            filepath: Path to load file
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"BM25 index file not found: {filepath}")
            raise FileNotFoundError(f"Index file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.k1 = data["k1"]
            self.b = data["b"]
            self.epsilon = data["epsilon"]
            self.documents = data["documents"]
            self.metadatas = data["metadatas"]
            self.tokenized_corpus = data["tokenized_corpus"]
            
            # Rebuild BM25 object
            self.bm25 = BM25Okapi(
                self.tokenized_corpus,
                k1=self.k1,
                b=self.b,
                epsilon=self.epsilon
            )
            
            logger.success(f"✓ BM25 index loaded from {filepath} ({len(self.documents)} documents)")
            
        except Exception as e:
            logger.error(f"Error loading BM25 index: {e}")
            raise
    
    def get_document_count(self) -> int:
        """Get number of documents in index"""
        return len(self.documents)
    
    def get_avg_document_length(self) -> float:
        """Get average document length"""
        if not self.tokenized_corpus:
            return 0.0
        return sum(len(doc) for doc in self.tokenized_corpus) / len(self.tokenized_corpus)


def main():
    """Example usage"""
    from src.config import settings
    
    print("\n" + "="*70)
    print("BM25Index Demo")
    print("="*70)
    
    # Sample documents
    documents = [
        "Transformers use self-attention mechanisms to process input sequences efficiently.",
        "BERT is a bidirectional transformer model for natural language understanding tasks.",
        "Graph neural networks operate on graph-structured data using message passing.",
        "Convolutional neural networks are particularly effective for image classification.",
        "Recurrent neural networks process sequential data with hidden states."
    ]
    
    metadatas = [
        {"doc_id": "1", "category": "transformers"},
        {"doc_id": "2", "category": "transformers"},
        {"doc_id": "3", "category": "graphs"},
        {"doc_id": "4", "category": "vision"},
        {"doc_id": "5", "category": "sequential"}
    ]
    
    # Build index
    print("\n1. Building BM25 index...")
    bm25 = BM25Index(k1=settings.BM25_K1, b=settings.BM25_B)
    bm25.build_index(documents, metadatas)
    print(f"   Documents indexed: {bm25.get_document_count()}")
    print(f"   Avg doc length: {bm25.get_avg_document_length():.1f} tokens")
    
    # Search
    print("\n2. Searching...")
    queries = [
        "attention mechanisms transformers",
        "graph neural networks",
        "image classification CNN"
    ]
    
    for query in queries:
        print(f"\n   Query: '{query}'")
        results = bm25.search(query, top_k=3)
        print(f"   Found {len(results)} results:")
        
        for i, result in enumerate(results, 1):
            print(f"\n   {i}. Score: {result['score']:.4f}")
            print(f"      Text: {result['text'][:60]}...")
            print(f"      Category: {result['metadata']['category']}")
    
    # Save and load
    print("\n3. Testing save/load...")
    test_path = settings.VECTOR_DB_DIR / "test_bm25.pkl"
    bm25.save(test_path)
    print(f"   ✓ Saved to {test_path}")
    
    # Create new instance and load
    bm25_loaded = BM25Index()
    bm25_loaded.load(test_path)
    print(f"   ✓ Loaded {bm25_loaded.get_document_count()} documents")
    
    # Test loaded index
    results = bm25_loaded.search("attention mechanisms", top_k=2)
    print(f"   ✓ Search on loaded index: {len(results)} results")
    
    # Cleanup
    test_path.unlink()
    print("   ✓ Cleanup complete")
    
    print("\n" + "="*70)
    print("✓ BM25Index working correctly!")
    print("="*70)


if __name__ == "__main__":
    main()

