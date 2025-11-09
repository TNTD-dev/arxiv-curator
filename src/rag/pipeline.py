"""
End-to-End RAG Pipeline

This module provides the complete RAG pipeline integrating:
- Document chunking (Phase 2)
- Hybrid retrieval (Phase 3)
- LLM generation (Phase 4)
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger

from src.config import settings
from src.data.arxiv_client import ArxivClient
from src.data.docling_processor import DoclingProcessor
from src.data.chunker import SemanticChunker
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.llm.groq_client import GroqClient
from src.llm.prompts import format_rag_prompt, get_system_message


class RAGPipeline:
    """
    Complete RAG pipeline for research paper Q&A
    """
    
    def __init__(
        self,
        groq_api_key: str,
        embedding_model: Optional[str] = None,
        groq_model: Optional[str] = None,
        vector_db_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
        alpha: float = 0.5,
        use_reranker: bool = False,
        reranker_model: Optional[str] = None
    ):
        """
        Initialize RAG pipeline
        
        Args:
            groq_api_key: Groq API key
            embedding_model: Sentence transformer model name
            groq_model: Groq model name
            vector_db_dir: Vector database directory
            collection_name: Qdrant collection name
            alpha: Hybrid search alpha (0=BM25, 1=vector, 0.5=balanced)
            use_reranker: Enable cross-encoder re-ranking
            reranker_model: Cross-encoder model name (optional)
        """
        logger.info("Initializing RAG Pipeline...")
        
        # Set defaults from config
        self.embedding_model_name = embedding_model or settings.EMBEDDING_MODEL
        self.groq_model_name = groq_model or settings.GROQ_MODEL
        self.vector_db_dir = vector_db_dir or settings.VECTOR_DB_DIR
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.alpha = alpha
        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model or settings.RERANK_MODEL
        
        # Initialize components
        self.embedder = None
        self.vector_store = None
        self.bm25_index = None
        self.hybrid_retriever = None
        self.llm = None
        
        # Initialize LLM first (most likely to fail)
        self._init_llm(groq_api_key)
        
        # Initialize retrieval components
        self._init_retrieval()
        
        logger.success("✓ RAG Pipeline initialized successfully")
    
    def _init_llm(self, api_key: str):
        """Initialize LLM client"""
        try:
            self.llm = GroqClient(
                api_key=api_key,
                model=self.groq_model_name,
                temperature=settings.GROQ_TEMPERATURE,
                max_tokens=settings.GROQ_MAX_TOKENS
            )
            logger.info(f"✓ LLM initialized: {self.groq_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _init_retrieval(self):
        """Initialize retrieval components"""
        try:
            # Embedder
            self.embedder = Embedder(self.embedding_model_name)
            logger.info(f"✓ Embedder initialized: {self.embedding_model_name}")
            
            # Vector store
            self.vector_store = VectorStore(
                collection_name=self.collection_name,
                embedding_dim=self.embedder.get_dimension(),
                persist_dir=self.vector_db_dir,
                distance=settings.QDRANT_DISTANCE
            )
            logger.info(f"✓ Vector store initialized: {self.collection_name}")
            
            # Check if vector store has data
            count = self.vector_store.count()
            if count == 0:
                logger.warning("Vector store is empty! Index documents first.")
            else:
                logger.info(f"✓ Vector store loaded: {count} documents")
            
            # BM25 index
            bm25_path = self.vector_db_dir / "bm25_index.pkl"
            if bm25_path.exists():
                self.bm25_index = BM25Index()
                self.bm25_index.load(bm25_path)
                logger.info(f"✓ BM25 index loaded: {self.bm25_index.get_document_count()} documents")
            else:
                logger.warning(f"BM25 index not found at {bm25_path}")
                self.bm25_index = BM25Index()
            
            # Hybrid retriever
            self.hybrid_retriever = HybridRetriever(
                embedder=self.embedder,
                vector_store=self.vector_store,
                bm25_index=self.bm25_index,
                alpha=self.alpha,
                use_reranker=self.use_reranker,
                reranker_model=self.reranker_model_name
            )
            if self.use_reranker:
                logger.info(f"✓ Hybrid retriever with re-ranking initialized (alpha={self.alpha})")
            else:
                logger.info(f"✓ Hybrid retriever initialized (alpha={self.alpha})")
            
        except Exception as e:
            logger.error(f"Failed to initialize retrieval: {e}")
            raise
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        mode: str = "default",
        stream: bool = False,
        return_context: bool = True,
        rerank_top_k: Optional[int] = None
    ) -> Dict:
        """
        Query the RAG pipeline
        
        Args:
            question: User question
            top_k: Number of contexts to retrieve (before re-ranking)
            mode: System message mode (default, technical, beginner_friendly)
            stream: Enable streaming response
            return_context: Include retrieved contexts in response
            rerank_top_k: Number of contexts after re-ranking (if None, uses top_k)
        
        Returns:
            Dictionary with answer, contexts, and metadata
        """
        logger.info(f"Query: '{question[:50]}...'")
        
        try:
            # Step 1: Retrieve relevant contexts
            contexts = self.hybrid_retriever.retrieve(
                question, 
                top_k=top_k,
                rerank_top_k=rerank_top_k
            )
            
            if not contexts:
                logger.warning("No contexts retrieved")
                return {
                    "answer": "I don't have enough information to answer this question based on the indexed papers.",
                    "contexts": [],
                    "metadata": {"status": "no_context"}
                }
            
            logger.info(f"Retrieved {len(contexts)} contexts")
            
            # Step 2: Generate response with LLM
            system_message = get_system_message(mode)
            
            response = self.llm.generate_with_context(
                query=question,
                contexts=contexts,
                system_message=system_message,
                stream=stream
            )
            
            # Step 3: Build result
            # Get the appropriate score field
            if self.use_reranker and contexts and 'rerank_score' in contexts[0]:
                top_score = contexts[0].get('rerank_score', 0)
                score_type = "rerank_score"
            else:
                top_score = contexts[0].get('combined_score', 0) if contexts else 0
                score_type = "combined_score"
            
            result = {
                "answer": response,
                "metadata": {
                    "num_contexts": len(contexts),
                    "top_score": top_score,
                    "score_type": score_type,
                    "model": self.groq_model_name,
                    "mode": mode,
                    "use_reranker": self.use_reranker
                }
            }
            
            if return_context:
                result["contexts"] = [
                    {
                        "text": ctx['text'][:200] + "..." if len(ctx['text']) > 200 else ctx['text'],
                        "score": ctx.get('rerank_score', ctx.get('combined_score', 0)),
                        "combined_score": ctx.get('combined_score', 0),
                        "rerank_score": ctx.get('rerank_score', None),
                        "paper_id": ctx.get('metadata', {}).get('paper_id', 'Unknown'),
                        "section": ctx.get('metadata', {}).get('section', 'Unknown')
                    }
                    for ctx in contexts
                ]
            
            logger.success("✓ Query completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in query: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "contexts": [],
                "metadata": {"status": "error", "error": str(e)}
            }
    
    def index_documents(
        self,
        pdf_paths: List[Path],
        force_reindex: bool = False
    ) -> Dict:
        """
        Index documents into the RAG pipeline
        
        Args:
            pdf_paths: List of PDF file paths
            force_reindex: Force reindexing even if documents exist
        
        Returns:
            Indexing statistics
        """
        logger.info(f"Indexing {len(pdf_paths)} documents...")
        
        # Check if already indexed
        current_count = self.vector_store.count()
        if current_count > 0 and not force_reindex:
            logger.info(f"Vector store already has {current_count} documents")
            return {
                "status": "already_indexed",
                "count": current_count
            }
        
        try:
            # Step 1: Process PDFs
            processor = DoclingProcessor(
                texts_dir=settings.TEXTS_DIR,
                tables_dir=settings.TABLES_DIR,
                figures_dir=settings.FIGURES_DIR
            )
            
            documents = processor.process_batch(pdf_paths)
            logger.info(f"✓ Processed {len(documents)} PDFs")
            
            # Step 2: Chunk documents
            chunker = SemanticChunker(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            
            all_chunks = []
            for doc in documents:
                text_file = Path(doc['text_file'])
                tables_file = Path(doc['tables_file']) if doc.get('tables_file') else None
                chunks = chunker.chunk_document(text_file, doc, tables_file)
                all_chunks.extend(chunks)
            
            logger.info(f"✓ Created {len(all_chunks)} chunks")
            
            # Step 3: Generate embeddings
            texts = [chunk['text'] for chunk in all_chunks]
            embeddings = self.embedder.embed_batch(texts, batch_size=32)
            logger.info(f"✓ Generated embeddings")
            
            # Step 4: Build vector store
            metadatas = [
                {
                    "chunk_id": chunk['chunk_id'],
                    "paper_id": chunk['paper_id'],
                    "section": chunk['section'],
                    "has_table": chunk.get('has_table', False)
                }
                for chunk in all_chunks
            ]
            
            ids = [chunk['chunk_id'] for chunk in all_chunks]
            self.vector_store.add_documents(texts, embeddings, metadatas, ids)
            logger.info(f"✓ Vector store built")
            
            # Step 5: Build BM25 index
            self.bm25_index.build_index(texts, metadatas)
            
            # Save BM25 index
            bm25_path = self.vector_db_dir / "bm25_index.pkl"
            self.bm25_index.save(bm25_path)
            logger.info(f"✓ BM25 index saved")
            
            return {
                "status": "success",
                "num_documents": len(documents),
                "num_chunks": len(all_chunks),
                "vector_store_count": self.vector_store.count(),
                "bm25_count": self.bm25_index.get_document_count()
            }
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        stats = {
            "vector_store": {
                "count": self.vector_store.count() if self.vector_store else 0,
                "collection": self.collection_name
            },
            "bm25": {
                "count": self.bm25_index.get_document_count() if self.bm25_index else 0
            },
            "llm": self.llm.get_model_info() if self.llm else {},
            "embedder": {
                "model": self.embedding_model_name,
                "dimension": self.embedder.get_dimension() if self.embedder else 0
            },
            "reranker": {
                "enabled": self.use_reranker,
                "model": self.reranker_model_name if self.use_reranker else None
            }
        }
        return stats


def main():
    """Example usage"""
    from src.config import settings
    import os
    
    print("\n" + "="*70)
    print("RAG Pipeline Demo")
    print("="*70)
    
    # Check API key
    api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\n❌ GROQ_API_KEY not found!")
        print("Set it in .env file: GROQ_API_KEY=gsk_...")
        return
    
    # Initialize pipeline
    print("\nInitializing RAG Pipeline...")
    rag = RAGPipeline(groq_api_key=api_key)
    
    # Show stats
    print("\nPipeline Statistics:")
    stats = rag.get_stats()
    print(f"  Vector Store: {stats['vector_store']['count']} documents")
    print(f"  BM25 Index: {stats['bm25']['count']} documents")
    print(f"  LLM: {stats['llm']['model']}")
    
    # Test queries
    test_queries = [
        "What are transformer models?",
        "How does attention mechanism work?",
        "What are the main contributions of this paper?"
    ]
    
    print("\nTesting RAG Pipeline:")
    print("-"*70)
    
    for query in test_queries:
        print(f"\nQ: {query}")
        
        result = rag.query(query, top_k=3)
        
        print(f"A: {result['answer'][:200]}...")
        print(f"\nContexts used: {result['metadata']['num_contexts']}")
        print(f"Top score: {result['metadata']['top_score']:.4f}")
    
    print("\n" + "="*70)
    print("✓ RAG Pipeline working correctly!")
    print("="*70)


if __name__ == "__main__":
    main()

