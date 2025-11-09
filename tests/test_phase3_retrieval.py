"""
Phase 3 Integration Test - Retrieval Pipeline

Test the complete retrieval flow: embeddings → vector store → BM25 → hybrid search
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.data.chunker import SemanticChunker
import json
from loguru import logger


def test_phase3_pipeline():
    """
    Run complete Phase 3 retrieval pipeline test
    """
    print("="*80)
    print("PHASE 3 INTEGRATION TEST - RETRIEVAL PIPELINE")
    print("="*80)
    print()
    
    # Check if Phase 2 data exists
    processed_files = list(settings.TEXTS_DIR.glob("*.md")) or list(settings.TEXTS_DIR.glob("*.txt"))
    
    if not processed_files:
        print("❌ No processed documents found!")
        print(f"   Expected location: {settings.TEXTS_DIR}")
        print()
        print("Please run Phase 2 first:")
        print("  python tests/test_phase2_pipeline.py --num-papers 3")
        return False
    
    print(f"✓ Found {len(processed_files)} processed documents")
    print()
    
    # Step 1: Load chunks from Phase 2
    print("Step 1: Loading chunks from Phase 2...")
    print("-"*80)
    
    try:
        # Load documents metadata
        chunker = SemanticChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        # Process each document
        all_chunks = []
        for text_file in processed_files[:3]:  # Limit to 3 documents for test
            paper_id = text_file.stem
            metadata = {
                "paper_id": paper_id,
                "text_file": str(text_file),
                "processor": "docling" if text_file.suffix == ".md" else "pymupdf"
            }
            
            chunks = chunker.chunk_document(text_file, metadata, None)
            all_chunks.extend(chunks)
        
        print(f"✓ Loaded {len(all_chunks)} chunks from {len(processed_files[:3])} documents")
        
        # Show sample
        if all_chunks:
            sample = all_chunks[0]
            print(f"\n   Sample chunk:")
            print(f"   - ID: {sample['chunk_id']}")
            print(f"   - Section: {sample['section']}")
            print(f"   - Length: {sample['char_count']} chars")
            print(f"   - Text preview: {sample['text'][:100]}...")
        
    except Exception as e:
        logger.error(f"Failed to load chunks: {e}")
        return False
    
    print()
    
    # Step 2: Initialize Embedder
    print("Step 2: Initializing Embedder...")
    print("-"*80)
    
    try:
        embedder = Embedder(settings.EMBEDDING_MODEL)
        print(f"✓ Embedder loaded")
        print(f"   Model: {embedder.get_model_name()}")
        print(f"   Dimension: {embedder.get_dimension()}")
        print(f"   Device: {embedder.device}")
        
    except Exception as e:
        logger.error(f"Failed to initialize embedder: {e}")
        return False
    
    print()
    
    # Step 3: Generate Embeddings
    print("Step 3: Generating embeddings...")
    print("-"*80)
    
    try:
        texts = [chunk['text'] for chunk in all_chunks]
        embeddings = embedder.embed_batch(texts, batch_size=32, show_progress=True)
        
        print(f"✓ Generated {len(embeddings)} embeddings")
        print(f"   Shape: {embeddings.shape}")
        
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        return False
    
    print()
    
    # Step 4: Build Vector Store
    print("Step 4: Building vector store (Qdrant)...")
    print("-"*80)
    
    try:
        vector_store = VectorStore(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embedding_dim=embedder.get_dimension(),
            persist_dir=settings.VECTOR_DB_DIR,
            distance=settings.QDRANT_DISTANCE
        )
        
        # Prepare metadata
        metadatas = [
            {
                "chunk_id": chunk['chunk_id'],
                "paper_id": chunk['paper_id'],
                "section": chunk['section'],
                "has_table": chunk.get('has_table', False)
            }
            for chunk in all_chunks
        ]
        
        # Add documents
        ids = [chunk['chunk_id'] for chunk in all_chunks]
        vector_store.add_documents(texts, embeddings, metadatas, ids)
        
        # Get info
        info = vector_store.get_collection_info()
        print(f"✓ Vector store built")
        print(f"   Collection: {info['name']}")
        print(f"   Documents: {info['points_count']}")
        print(f"   Status: {info['status']}")
        
    except Exception as e:
        logger.error(f"Failed to build vector store: {e}")
        return False
    
    print()
    
    # Step 5: Build BM25 Index
    print("Step 5: Building BM25 index...")
    print("-"*80)
    
    try:
        bm25_index = BM25Index(k1=settings.BM25_K1, b=settings.BM25_B)
        bm25_index.build_index(texts, metadatas)
        
        print(f"✓ BM25 index built")
        print(f"   Documents: {bm25_index.get_document_count()}")
        print(f"   Avg doc length: {bm25_index.get_avg_document_length():.1f} tokens")
        
        # Save index
        bm25_path = settings.VECTOR_DB_DIR / "bm25_index.pkl"
        bm25_index.save(bm25_path)
        print(f"   Saved to: {bm25_path}")
        
    except Exception as e:
        logger.error(f"Failed to build BM25 index: {e}")
        return False
    
    print()
    
    # Step 6: Test Hybrid Retrieval
    print("Step 6: Testing hybrid retrieval...")
    print("-"*80)
    
    try:
        hybrid = HybridRetriever(
            embedder=embedder,
            vector_store=vector_store,
            bm25_index=bm25_index,
            alpha=0.5  # Balanced
        )
        
        # Test queries
        test_queries = [
            "What are transformer models?",
            "How does attention mechanism work?",
            "Graph neural networks for relational data"
        ]
        
        results_summary = []
        
        for query in test_queries:
            print(f"\n   Query: '{query}'")
            results = hybrid.retrieve(query, top_k=5)
            
            print(f"   Found {len(results)} results:")
            for i, result in enumerate(results[:3], 1):
                print(f"\n   {i}. Combined: {result['combined_score']:.4f} "
                      f"(V:{result.get('vector_score', 0):.3f}, "
                      f"B:{result.get('bm25_score', 0):.3f})")
                print(f"      Section: {result['metadata']['section']}")
                print(f"      Paper: {result['metadata']['paper_id']}")
                print(f"      Text: {result['text'][:80]}...")
            
            results_summary.append({
                "query": query,
                "num_results": len(results),
                "top_score": results[0]['combined_score'] if results else 0
            })
        
        print()
        print("   ✓ Hybrid retrieval working!")
        
    except Exception as e:
        logger.error(f"Failed hybrid retrieval: {e}")
        return False
    
    print()
    
    # Step 7: Save Results
    print("Step 7: Saving results...")
    print("-"*80)
    
    try:
        results_file = settings.DATA_DIR / "phase3_test_results.json"
        
        results = {
            "num_chunks": len(all_chunks),
            "embedding_dim": embedder.get_dimension(),
            "vector_store_count": info['points_count'],
            "bm25_index_count": bm25_index.get_document_count(),
            "test_queries": results_summary,
            "retrieval_config": {
                "top_k_retrieval": settings.TOP_K_RETRIEVAL,
                "alpha": 0.5
            }
        }
        
        results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"✓ Results saved to: {results_file}")
        
    except Exception as e:
        logger.warning(f"Could not save results: {e}")
    
    print()
    print("="*80)
    print("PHASE 3 TEST COMPLETED SUCCESSFULLY! ✓")
    print("="*80)
    print()
    print("Summary:")
    print(f"  • Chunks indexed: {len(all_chunks)}")
    print(f"  • Vector store: {info['points_count']} documents")
    print(f"  • BM25 index: {bm25_index.get_document_count()} documents")
    print(f"  • Hybrid retrieval: Working")
    print()
    print("Next steps:")
    print("  1. Vector store ready for Phase 4 (RAG)")
    print("  2. Test with: python tests/test_phase3_retrieval.py")
    print("  3. Ready to integrate with LLM (Groq)")
    print()
    
    return True


def test_individual_components():
    """Test each component individually"""
    print("\n" + "="*80)
    print("INDIVIDUAL COMPONENT TESTS")
    print("="*80 + "\n")
    
    # Test 1: Embedder
    print("Test 1: Embedder")
    print("-"*80)
    try:
        embedder = Embedder(settings.EMBEDDING_MODEL)
        test_text = "This is a test sentence."
        embedding = embedder.embed_text(test_text)
        print(f"✓ Embedder working: generated {len(embedding)}-dim embedding")
    except Exception as e:
        print(f"✗ Embedder failed: {e}")
    
    print()
    
    # Test 2: Vector Store
    print("Test 2: Vector Store (Qdrant)")
    print("-"*80)
    try:
        vs = VectorStore(
            collection_name="test",
            embedding_dim=384,
            persist_dir=settings.VECTOR_DB_DIR / "test"
        )
        print(f"✓ VectorStore initialized")
        vs.delete_collection()
    except Exception as e:
        print(f"✗ VectorStore failed: {e}")
    
    print()
    
    # Test 3: BM25 Index
    print("Test 3: BM25 Index")
    print("-"*80)
    try:
        bm25 = BM25Index()
        test_docs = ["test document one", "test document two"]
        bm25.build_index(test_docs)
        results = bm25.search("test", top_k=2)
        print(f"✓ BM25Index working: found {len(results)} results")
    except Exception as e:
        print(f"✗ BM25Index failed: {e}")
    
    print()


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Phase 3 Retrieval Pipeline")
    parser.add_argument(
        "--components-only",
        action="store_true",
        help="Only test individual components"
    )
    
    args = parser.parse_args()
    
    if args.components_only:
        test_individual_components()
    else:
        success = test_phase3_pipeline()
        
        if not success:
            print("\n⚠️  Some tests failed. Check logs for details.")
            sys.exit(1)


if __name__ == "__main__":
    main()

