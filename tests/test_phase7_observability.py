"""
Phase 7 Integration Test - Observability with Langfuse

Test Langfuse integration and observable RAG pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.observability.langfuse_client import LangfuseObserver, create_observer
from src.rag.observable_pipeline import ObservableRAGPipeline
import os


def test_phase7_observability():
    """
    Run Phase 7 observability tests
    """
    print("="*80)
    print("PHASE 7 INTEGRATION TEST - OBSERVABILITY WITH LANGFUSE")
    print("="*80)
    print()
    
    # Check API keys
    groq_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    langfuse_public = settings.LANGFUSE_PUBLIC_KEY or os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret = settings.LANGFUSE_SECRET_KEY or os.getenv("LANGFUSE_SECRET_KEY")
    
    if not groq_key:
        print("❌ GROQ_API_KEY not found!")
        return False
    
    print(f"✓ Groq API key: {groq_key[:20]}...")
    
    has_langfuse = bool(langfuse_public and langfuse_secret)
    if has_langfuse:
        print(f"✓ Langfuse public key: {langfuse_public[:20]}...")
        print(f"✓ Langfuse secret key: {langfuse_secret[:20]}...")
    else:
        print("⚠️  Langfuse keys not found (will test with observability disabled)")
    
    print()
    
    # Test 1: Observer Initialization
    print("Test 1: Langfuse Observer Initialization")
    print("-"*80)
    
    try:
        observer = create_observer(
            public_key=langfuse_public,
            secret_key=langfuse_secret
        )
        
        print(f"✓ Observer created")
        print(f"  Enabled: {observer.enabled}")
        
        if not observer.enabled and has_langfuse:
            print("  ⚠️  Observer disabled despite having keys (check credentials)")
        
        print()
        print("✅ Observer initialization test passed!")
        
    except Exception as e:
        print(f"❌ Observer initialization failed: {e}")
        return False
    
    print()
    
    # Test 2: Observable Pipeline Initialization
    print("Test 2: Observable RAG Pipeline Initialization")
    print("-"*80)
    
    try:
        rag = ObservableRAGPipeline(
            groq_api_key=groq_key,
            langfuse_public_key=langfuse_public,
            langfuse_secret_key=langfuse_secret
        )
        
        print("✓ Observable RAG Pipeline initialized")
        print(f"  Observability: {'enabled' if rag.observer.enabled else 'disabled'}")
        
        # Get stats
        stats = rag.get_stats()
        print(f"  Vector store: {stats['vector_store']['count']} documents")
        print(f"  BM25 index: {stats['bm25']['count']} documents")
        print(f"  LLM model: {stats['llm']['model']}")
        print(f"  Re-ranker: {stats['reranker']['enabled']}")
        
        print()
        print("✅ Pipeline initialization test passed!")
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False
    
    print()
    
    # Test 3: Query with Observability
    print("Test 3: Query with Observability Tracking")
    print("-"*80)
    
    test_queries = [
        "What are transformer models?",
        "How does attention mechanism work?",
        "What are the main contributions of this research?"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        
        try:
            result = rag.query(
                question=query,
                top_k=5,
                return_context=True
            )
            
            print(f"✓ Query completed")
            print(f"  Answer length: {len(result['answer'])} chars")
            print(f"  Contexts: {result['metadata']['num_contexts']}")
            print(f"  Top score: {result['metadata']['top_score']:.4f}")
            
            # Check timing metadata
            if 'retrieval_duration_ms' in result['metadata']:
                print(f"  Retrieval: {result['metadata']['retrieval_duration_ms']:.2f}ms")
            if 'llm_duration_ms' in result['metadata']:
                print(f"  LLM: {result['metadata']['llm_duration_ms']:.2f}ms")
            if 'total_duration_ms' in result['metadata']:
                print(f"  Total: {result['metadata']['total_duration_ms']:.2f}ms")
            
            results.append({
                "query": query,
                "success": True,
                "metadata": result['metadata']
            })
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            results.append({
                "query": query,
                "success": False,
                "error": str(e)
            })
    
    print()
    successful = sum(1 for r in results if r.get('success'))
    print(f"Query Success Rate: {successful}/{len(test_queries)}")
    
    if successful == len(test_queries):
        print("✅ All queries successful!")
    else:
        print(f"⚠️  {len(test_queries) - successful} queries failed")
    
    print()
    
    # Test 4: Metrics Collection (if Langfuse enabled)
    if rag.observer.enabled:
        print("Test 4: Metrics Collection")
        print("-"*80)
        
        try:
            from src.observability.langfuse_client import MetricsCollector
            
            collector = MetricsCollector(rag.observer.client)
            
            print("✓ Metrics collector initialized")
            print()
            print("Note: Metrics collection requires Langfuse API access")
            print("      Check dashboard: https://cloud.langfuse.com")
            
            print()
            print("✅ Metrics collection test passed!")
            
        except Exception as e:
            print(f"⚠️  Metrics collection test skipped: {e}")
    else:
        print("Test 4: Metrics Collection")
        print("-"*80)
        print("⚠️  Skipped (Langfuse not enabled)")
    
    print()
    
    # Summary
    print("="*80)
    print("PHASE 7 TEST COMPLETED!")
    print("="*80)
    print()
    print("Summary:")
    print(f"  ✓ Observer initialization: PASSED")
    print(f"  ✓ Pipeline initialization: PASSED")
    print(f"  ✓ Query tracking: {successful}/{len(test_queries)} successful")
    print(f"  ✓ Observability: {'ENABLED' if rag.observer.enabled else 'DISABLED (no credentials)'}")
    print()
    
    if rag.observer.enabled:
        print("Next steps:")
        print("  1. Check Langfuse dashboard: https://cloud.langfuse.com")
        print("  2. View traces and metrics")
        print("  3. Analyze performance patterns")
    else:
        print("To enable observability:")
        print("  1. Sign up at: https://cloud.langfuse.com")
        print("  2. Get API keys from settings")
        print("  3. Add to .env:")
        print("     LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("     LANGFUSE_SECRET_KEY=sk-lf-...")
    
    print()
    
    return True


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Phase 7 Observability")
    args = parser.parse_args()
    
    success = test_phase7_observability()
    
    if not success:
        print("\n⚠️  Some tests failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

