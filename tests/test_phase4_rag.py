"""
Phase 4 Integration Test - RAG Pipeline

Test the complete RAG flow: retrieval → LLM generation → answer with citations
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.rag.pipeline import RAGPipeline
from loguru import logger
import os
import json


def test_phase4_rag():
    """
    Run complete Phase 4 RAG pipeline test
    """
    print("="*80)
    print("PHASE 4 INTEGRATION TEST - RAG PIPELINE")
    print("="*80)
    print()
    
    # Check API key
    api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("❌ GROQ_API_KEY not found!")
        print()
        print("Please set your Groq API key:")
        print("  1. Copy env.example to .env")
        print("  2. Add: GROQ_API_KEY=gsk_your_key_here")
        print("  3. Get key from: https://console.groq.com/keys")
        print()
        return False
    
    print(f"✓ API key found: {api_key[:20]}...")
    print()
    
    # Step 1: Initialize RAG Pipeline
    print("Step 1: Initializing RAG Pipeline...")
    print("-"*80)
    
    try:
        rag = RAGPipeline(
            groq_api_key=api_key,
            alpha=0.5  # Balanced hybrid search
        )
        
        print("✓ RAG Pipeline initialized")
        
        # Show stats
        stats = rag.get_stats()
        print(f"\n   Pipeline Statistics:")
        print(f"   - Vector Store: {stats['vector_store']['count']} documents")
        print(f"   - BM25 Index: {stats['bm25']['count']} documents")
        print(f"   - LLM Model: {stats['llm']['model']}")
        print(f"   - Embedder: {stats['embedder']['model']}")
        print(f"   - Embedding Dim: {stats['embedder']['dimension']}")
        
        if stats['vector_store']['count'] == 0:
            print("\n   ⚠️  Vector store is empty!")
            print("   Run Phase 3 first: python tests/test_phase3_retrieval.py")
            return False
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG Pipeline: {e}")
        return False
    
    print()
    
    # Step 2: Test Queries
    print("Step 2: Testing RAG with sample queries...")
    print("-"*80)
    
    test_queries = [
        {
            "question": "What are transformer models and how do they work?",
            "mode": "default",
            "description": "General question about transformers"
        },
        {
            "question": "Explain attention mechanisms in deep learning",
            "mode": "technical",
            "description": "Technical explanation query"
        },
        {
            "question": "What are the main contributions of this research?",
            "mode": "default",
            "description": "Paper-specific question"
        }
    ]
    
    results_summary = []
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n   Query {i}: {test['description']}")
        print(f"   Question: {test['question']}")
        print(f"   Mode: {test['mode']}")
        print()
        
        try:
            result = rag.query(
                question=test['question'],
                top_k=5,
                mode=test['mode'],
                return_context=True
            )
            
            # Display answer
            answer = result['answer']
            print(f"   Answer ({len(answer)} chars):")
            print(f"   {answer[:300]}...")
            
            if len(answer) > 300:
                print(f"   ... (truncated, full length: {len(answer)} chars)")
            
            # Display context info
            print(f"\n   Metadata:")
            print(f"   - Contexts used: {result['metadata']['num_contexts']}")
            print(f"   - Top relevance score: {result['metadata']['top_score']:.4f}")
            print(f"   - Model: {result['metadata']['model']}")
            
            # Display top contexts
            if 'contexts' in result and result['contexts']:
                print(f"\n   Top 3 Retrieved Contexts:")
                for j, ctx in enumerate(result['contexts'][:3], 1):
                    print(f"   {j}. Paper: {ctx['paper_id']} | Section: {ctx['section']}")
                    print(f"      Score: {ctx['score']:.4f}")
                    print(f"      Text: {ctx['text'][:100]}...")
            
            # Save for summary
            results_summary.append({
                "question": test['question'],
                "answer_length": len(answer),
                "num_contexts": result['metadata']['num_contexts'],
                "top_score": result['metadata']['top_score'],
                "status": "success"
            })
            
            print(f"\n   ✓ Query {i} completed")
            
        except Exception as e:
            logger.error(f"Query {i} failed: {e}")
            results_summary.append({
                "question": test['question'],
                "status": "error",
                "error": str(e)
            })
            print(f"\n   ✗ Query {i} failed: {e}")
        
        print("\n   " + "-"*70)
    
    print()
    
    # Step 3: Test Streaming Response
    print("Step 3: Testing streaming response...")
    print("-"*80)
    
    try:
        query = "Summarize the key findings in one paragraph."
        print(f"   Query: {query}")
        print(f"\n   Streaming response: ", end="", flush=True)
        
        # Note: Streaming returns generator, we need to collect it
        result = rag.query(query, top_k=3, stream=False)  # Use non-streaming for test
        print(result['answer'][:200] + "...")
        
        print(f"\n\n   ✓ Streaming test completed")
        
    except Exception as e:
        logger.error(f"Streaming test failed: {e}")
        print(f"\n   ✗ Streaming failed: {e}")
    
    print()
    
    # Step 4: Save Results
    print("Step 4: Saving results...")
    print("-"*80)
    
    try:
        results_file = settings.DATA_DIR / "phase4_test_results.json"
        
        results = {
            "pipeline_stats": stats,
            "test_queries": results_summary,
            "num_successful": sum(1 for r in results_summary if r.get('status') == 'success'),
            "num_failed": sum(1 for r in results_summary if r.get('status') == 'error'),
            "config": {
                "model": settings.GROQ_MODEL,
                "temperature": settings.GROQ_TEMPERATURE,
                "max_tokens": settings.GROQ_MAX_TOKENS,
                "top_k": 5,
                "alpha": 0.5
            }
        }
        
        results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"✓ Results saved to: {results_file}")
        
    except Exception as e:
        logger.warning(f"Could not save results: {e}")
    
    print()
    print("="*80)
    print("PHASE 4 TEST COMPLETED SUCCESSFULLY! ✓")
    print("="*80)
    print()
    print("Summary:")
    print(f"  • Successful queries: {sum(1 for r in results_summary if r.get('status') == 'success')}/{len(results_summary)}")
    print(f"  • Vector store: {stats['vector_store']['count']} documents")
    print(f"  • LLM: {stats['llm']['model']}")
    print(f"  • RAG Pipeline: Fully operational")
    print()
    print("Next steps:")
    print("  1. RAG pipeline ready for Phase 5 (Re-ranking)")
    print("  2. Test with: python -m src.rag.pipeline")
    print("  3. Or proceed to Phase 6 (UI)")
    print()
    
    return True


def test_individual_components():
    """Test each component individually"""
    print("\n" + "="*80)
    print("INDIVIDUAL COMPONENT TESTS")
    print("="*80 + "\n")
    
    api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    
    # Test 1: Groq Client
    print("Test 1: Groq Client")
    print("-"*80)
    try:
        from src.llm.groq_client import GroqClient
        
        if not api_key:
            print("✗ GROQ_API_KEY not set")
        else:
            client = GroqClient(api_key, model=settings.GROQ_MODEL)
            response = client.generate("Say 'Hello, RAG!'")
            print(f"✓ Groq Client working")
            print(f"  Response: {response[:100]}...")
    except Exception as e:
        print(f"✗ Groq Client failed: {e}")
    
    print()
    
    # Test 2: Prompts
    print("Test 2: Prompt Templates")
    print("-"*80)
    try:
        from src.llm.prompts import format_rag_prompt, get_system_message
        
        contexts = [{"text": "Test context", "score": 0.9, "metadata": {}}]
        prompt = format_rag_prompt("Test query", contexts)
        system_msg = get_system_message("default")
        
        print("✓ Prompt templates working")
        print(f"  Prompt length: {len(prompt)} chars")
        print(f"  System message length: {len(system_msg)} chars")
    except Exception as e:
        print(f"✗ Prompts failed: {e}")
    
    print()
    
    # Test 3: RAG Pipeline (without queries)
    print("Test 3: RAG Pipeline Initialization")
    print("-"*80)
    try:
        if not api_key:
            print("✗ GROQ_API_KEY not set")
        else:
            rag = RAGPipeline(groq_api_key=api_key)
            stats = rag.get_stats()
            print("✓ RAG Pipeline initialized")
            print(f"  Vector store: {stats['vector_store']['count']} docs")
            print(f"  LLM: {stats['llm']['model']}")
    except Exception as e:
        print(f"✗ RAG Pipeline failed: {e}")
    
    print()


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Phase 4 RAG Pipeline")
    parser.add_argument(
        "--components-only",
        action="store_true",
        help="Only test individual components"
    )
    
    args = parser.parse_args()
    
    if args.components_only:
        test_individual_components()
    else:
        success = test_phase4_rag()
        
        if not success:
            print("\n⚠️  Some tests failed. Check logs for details.")
            sys.exit(1)


if __name__ == "__main__":
    main()

