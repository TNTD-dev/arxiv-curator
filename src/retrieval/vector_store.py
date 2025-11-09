"""
Qdrant Vector Store for semantic search

This module provides Qdrant integration for storing and retrieving document embeddings.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)
from typing import List, Dict, Optional, Union
from pathlib import Path
import numpy as np
from loguru import logger
from uuid import uuid4, UUID
import hashlib


class VectorStore:
    """
    Vector store using Qdrant for semantic search
    """
    
    def __init__(
        self,
        collection_name: str,
        embedding_dim: int,
        persist_dir: Path,
        distance: str = "Cosine"
    ):
        """
        Initialize Qdrant vector store
        
        Args:
            collection_name: Name of the collection
            embedding_dim: Dimension of embeddings
            persist_dir: Directory to persist data
            distance: Distance metric (Cosine, Euclid, Dot)
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Map distance names
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT
        }
        self.distance = distance_map.get(distance, Distance.COSINE)
        
        logger.info(f"Initializing Qdrant client at {persist_dir}")
        
        try:
            # Initialize Qdrant client with persistence
            self.client = QdrantClient(path=str(persist_dir))
            
            # Create collection if it doesn't exist
            self._ensure_collection()
            
            logger.success(
                f"✓ VectorStore initialized: collection='{collection_name}', "
                f"dim={embedding_dim}, distance={distance}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {e}")
            raise
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=self.distance
                    )
                )
                
                logger.success(f"✓ Collection '{self.collection_name}' created")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
    
    def _convert_id_to_uuid(self, doc_id: Union[str, int, UUID]) -> str:
        """
        Convert various ID types to UUID string for Qdrant
        
        Args:
            doc_id: Document ID (string, int, or UUID)
        
        Returns:
            UUID string
        """
        if isinstance(doc_id, UUID):
            return str(doc_id)
        elif isinstance(doc_id, int):
            # Convert int to UUID using uuid4 with seed
            return str(uuid4())
        else:
            # Hash string ID to create deterministic UUID
            # This ensures same string always maps to same UUID
            hash_object = hashlib.md5(str(doc_id).encode())
            uuid_from_hash = UUID(hash_object.hexdigest())
            return str(uuid_from_hash)
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict],
        ids: Optional[List[Union[str, int, UUID]]] = None
    ) -> List[str]:
        """
        Add documents to vector store
        
        Args:
            texts: List of text strings
            embeddings: Array of embeddings
            metadatas: List of metadata dictionaries
            ids: Optional list of IDs (string, int, or UUID - will be converted to UUID)
        
        Returns:
            List of document IDs (as UUID strings)
        """
        if not texts:
            logger.warning("No documents to add")
            return []
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(texts))]
        
        # Convert all IDs to UUID format and store mapping
        uuid_ids = []
        id_mapping = {}  # Maps UUID back to original ID
        
        for original_id in ids:
            uuid_id = self._convert_id_to_uuid(original_id)
            uuid_ids.append(uuid_id)
            id_mapping[uuid_id] = str(original_id)
        
        # Prepare points for Qdrant
        points = []
        for idx, (text, embedding, metadata, uuid_id, orig_id) in enumerate(
            zip(texts, embeddings, metadatas, uuid_ids, ids)
        ):
            # Add text and original ID to metadata
            payload = {
                **metadata, 
                "text": text,
                "original_id": str(orig_id)  # Store original ID in metadata
            }
            
            point = PointStruct(
                id=uuid_id,  # Use UUID for Qdrant
                vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                payload=payload
            )
            points.append(point)
        
        try:
            # Upload points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            logger.success(f"✓ Added {len(points)} documents to vector store")
            return uuid_ids  # Return UUID IDs
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter
        
        Returns:
            List of results with scores and metadata
        """
        try:
            # Convert numpy array to list
            query_vector = query_embedding.tolist() if isinstance(
                query_embedding, np.ndarray
            ) else query_embedding
            
            # Build filter if provided
            query_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                query_filter = Filter(must=conditions)
            
            # Search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter
            )
            
            # Format results
            formatted_results = []
            for result in results:
                # Use original_id from metadata if available
                result_id = result.payload.get("original_id", result.id)
                
                formatted_results.append({
                    "id": result_id,  # Return original ID
                    "uuid": result.id,  # Also include UUID
                    "score": result.score,
                    "metadata": result.payload,
                    "text": result.payload.get("text", "")
                })
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Error deleting collection: {e}")
    
    def get_collection_info(self) -> Dict:
        """Get collection information"""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def count(self) -> int:
        """Get number of documents in collection"""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return info.points_count
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0


def main():
    """Example usage"""
    from src.config import settings
    from src.retrieval.embedder import Embedder
    
    print("\n" + "="*70)
    print("VectorStore Demo")
    print("="*70)
    
    # Initialize embedder and vector store
    embedder = Embedder(settings.EMBEDDING_MODEL)
    
    vector_store = VectorStore(
        collection_name="test_collection",
        embedding_dim=embedder.get_dimension(),
        persist_dir=settings.VECTOR_DB_DIR / "test"
    )
    
    # Sample documents
    documents = [
        "Transformers use self-attention mechanisms to process sequences.",
        "BERT is a transformer-based model for natural language understanding.",
        "Graph neural networks operate on graph-structured data.",
        "Convolutional neural networks are effective for image processing.",
        "Recurrent neural networks process sequential data."
    ]
    
    metadatas = [
        {"source": "doc1", "category": "transformers"},
        {"source": "doc2", "category": "transformers"},
        {"source": "doc3", "category": "graphs"},
        {"source": "doc4", "category": "vision"},
        {"source": "doc5", "category": "sequential"}
    ]
    
    print("\n1. Adding documents...")
    embeddings = embedder.embed_batch(documents, show_progress=False)
    doc_ids = vector_store.add_documents(documents, embeddings, metadatas)
    print(f"   Added {len(doc_ids)} documents")
    
    print("\n2. Collection info:")
    info = vector_store.get_collection_info()
    print(f"   Name: {info['name']}")
    print(f"   Documents: {info['points_count']}")
    print(f"   Status: {info['status']}")
    
    print("\n3. Searching...")
    query = "What are attention mechanisms?"
    query_embedding = embedder.embed_text(query)
    results = vector_store.search(query_embedding, top_k=3)
    
    print(f"   Query: {query}")
    print(f"   Top {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n   {i}. Score: {result['score']:.4f}")
        print(f"      Text: {result['text'][:60]}...")
        print(f"      Category: {result['metadata']['category']}")
    
    print("\n4. Filtered search...")
    results = vector_store.search(
        query_embedding, 
        top_k=3,
        filter_dict={"category": "transformers"}
    )
    print(f"   Filter: category='transformers'")
    print(f"   Results: {len(results)}")
    
    # Cleanup
    print("\n5. Cleanup...")
    vector_store.delete_collection()
    print("   ✓ Collection deleted")
    
    print("\n" + "="*70)
    print("✓ VectorStore working correctly!")
    print("="*70)


if __name__ == "__main__":
    main()

