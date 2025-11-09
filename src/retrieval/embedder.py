"""
Embedding Generator using Sentence Transformers

This module provides functionality to generate dense embeddings for text chunks
using pre-trained sentence transformer models.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
from loguru import logger
from pathlib import Path
import torch


class Embedder:
    """
    Generate embeddings for text using Sentence Transformers
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None
    ):
        """
        Initialize embedder with sentence transformer model
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Using device: {device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            logger.success(
                f"✓ Embedder loaded: {model_name} "
                f"(dim={self.embedding_dim}, device={device})"
            )
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text string
        
        Returns:
            Embedding vector as numpy array
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise
    
    def embed_batch(
        self, 
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        
        Returns:
            Array of embeddings (shape: [num_texts, embedding_dim])
        """
        if not texts:
            logger.warning("Empty text list provided")
            return np.array([])
        
        try:
            logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
            
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            logger.success(f"✓ Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            raise
    
    def embed_chunks(
        self,
        chunks: List[dict],
        text_field: str = "text",
        batch_size: int = 32
    ) -> List[dict]:
        """
        Embed chunks and add embeddings to chunk dictionaries
        
        Args:
            chunks: List of chunk dictionaries
            text_field: Field name containing text to embed
            batch_size: Batch size for encoding
        
        Returns:
            Chunks with added 'embedding' field
        """
        if not chunks:
            logger.warning("No chunks provided")
            return []
        
        # Extract texts
        texts = [chunk.get(text_field, "") for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_batch(texts, batch_size=batch_size)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()  # Convert to list for JSON serialization
        
        logger.info(f"✓ Added embeddings to {len(chunks)} chunks")
        return chunks
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Cosine similarity score (0-1)
        """
        from numpy.linalg import norm
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
        return float(similarity)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim
    
    def get_model_name(self) -> str:
        """Get model name"""
        return self.model_name


def main():
    """Example usage"""
    from src.config import settings
    
    # Initialize embedder
    embedder = Embedder(settings.EMBEDDING_MODEL)
    
    # Test with sample texts
    texts = [
        "Transformer models have revolutionized natural language processing.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Graph neural networks are powerful for processing structured data."
    ]
    
    print("\n" + "="*70)
    print("Embedding Demo")
    print("="*70)
    
    # Embed single text
    print("\n1. Single text embedding:")
    embedding = embedder.embed_text(texts[0])
    print(f"   Text: {texts[0][:50]}...")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   First 5 values: {embedding[:5]}")
    
    # Embed batch
    print("\n2. Batch embedding:")
    embeddings = embedder.embed_batch(texts, show_progress=False)
    print(f"   Number of texts: {len(texts)}")
    print(f"   Embeddings shape: {embeddings.shape}")
    
    # Calculate similarities
    print("\n3. Similarity scores:")
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            sim = embedder.similarity(embeddings[i], embeddings[j])
            print(f"   Text {i+1} vs Text {j+1}: {sim:.4f}")
    
    # Test with chunks
    print("\n4. Embedding chunks:")
    sample_chunks = [
        {"chunk_id": "1", "text": texts[0]},
        {"chunk_id": "2", "text": texts[1]},
        {"chunk_id": "3", "text": texts[2]}
    ]
    
    chunks_with_embeddings = embedder.embed_chunks(sample_chunks)
    print(f"   Chunks embedded: {len(chunks_with_embeddings)}")
    print(f"   First chunk has embedding: {'embedding' in chunks_with_embeddings[0]}")
    print(f"   Embedding dimension: {len(chunks_with_embeddings[0]['embedding'])}")
    
    print("\n" + "="*70)
    print("✓ Embedder working correctly!")
    print("="*70)


if __name__ == "__main__":
    main()

