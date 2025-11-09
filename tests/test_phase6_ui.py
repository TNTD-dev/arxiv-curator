"""
Phase 6 Integration Test - Enhanced UI

Test the Gradio UI components and functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.ui.gradio_app import PaperNavigatorUI
from src.ui.components import (
    CitationFormatter,
    TableFormatter,
    FigureFormatter,
    SourceHighlighter,
    MetadataFormatter
)
import os


def test_phase6_ui():
    """
    Run Phase 6 UI component tests
    """
    print("="*80)
    print("PHASE 6 INTEGRATION TEST - ENHANCED UI")
    print("="*80)
    print()
    
    # Check API key
    api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("❌ GROQ_API_KEY not found!")
        print()
        print("Please set your Groq API key in .env file")
        print()
        return False
    
    print(f"✓ API key found: {api_key[:20]}...")
    print()
    
    # Test 1: Component Formatters
    print("Test 1: Testing Component Formatters")
    print("-"*80)
    
    try:
        # Test citation formatter
        test_ctx = {
            'paper_id': '2511_04557v1',
            'section': 'Introduction',
            'score': 0.8523,
            'combined_score': 0.8523,
            'rerank_score': 9.234,
            'text': 'Transformer models have revolutionized natural language processing.'
        }
        
        basic_citation = CitationFormatter.format_basic(test_ctx, 1)
        print("✓ Basic citation formatter working")
        print(f"  Length: {len(basic_citation)} chars")
        
        rerank_citation = CitationFormatter.format_with_rerank(test_ctx, 1)
        print("✓ Re-rank citation formatter working")
        print(f"  Length: {len(rerank_citation)} chars")
        
        # Test source highlighter
        query = "What are transformer models"
        terms = SourceHighlighter.extract_query_terms(query)
        print(f"✓ Source highlighter extracted {len(terms)} terms: {terms}")
        
        highlighted = SourceHighlighter.highlight_text(test_ctx['text'], terms)
        print("✓ Text highlighting working")
        
        # Test metadata formatter
        test_metadata = {
            'model': 'llama-3.3-70b-versatile',
            'num_contexts': 5,
            'top_score': 0.9234,
            'score_type': 'rerank_score',
            'use_reranker': True,
            'mode': 'default'
        }
        
        stats = MetadataFormatter.format_retrieval_stats(test_metadata)
        print("✓ Metadata formatter working")
        print(f"  Stats length: {len(stats)} chars")
        
        print()
        print("✅ All component formatters working correctly!")
        
    except Exception as e:
        print(f"❌ Component formatter test failed: {e}")
        return False
    
    print()
    
    # Test 2: UI Initialization
    print("Test 2: Testing UI Initialization")
    print("-"*80)
    
    try:
        ui = PaperNavigatorUI(groq_api_key=api_key)
        print("✓ PaperNavigatorUI initialized")
        
        # Test stats
        stats = ui.get_stats()
        print("✓ Pipeline stats retrieved")
        print(f"  Stats length: {len(stats)} chars")
        
        # Test history
        history = ui.get_history()
        print("✓ History retrieved (empty)")
        
        print()
        print("✅ UI initialization successful!")
        
    except Exception as e:
        print(f"❌ UI initialization failed: {e}")
        return False
    
    print()
    
    # Test 3: Query Processing (without actual query to save API calls)
    print("Test 3: Testing Query Processing Components")
    print("-"*80)
    
    try:
        # Test citation formatting with real data structure
        test_contexts = [
            {
                'paper_id': '2511_04557v1',
                'section': 'Introduction',
                'score': 0.9234,
                'combined_score': 0.8523,
                'rerank_score': 9.234,
                'text': 'Transformers use self-attention mechanisms to process sequences.'
            },
            {
                'paper_id': '2511_04557v1',
                'section': 'Methods',
                'score': 0.8712,
                'combined_score': 0.7891,
                'text': 'The attention mechanism allows models to focus on relevant parts.'
            }
        ]
        
        # Format citations
        citations_text = "## 📚 Retrieved Contexts\n\n"
        for i, ctx in enumerate(test_contexts, 1):
            citation = ui.format_citation(ctx, i)
            citations_text += citation
            print(f"✓ Formatted citation {i}")
        
        print(f"\n  Total citations length: {len(citations_text)} chars")
        
        print()
        print("✅ Query processing components working!")
        
    except Exception as e:
        print(f"❌ Query processing test failed: {e}")
        return False
    
    print()
    
    # Test 4: Interface Building
    print("Test 4: Testing Gradio Interface Building")
    print("-"*80)
    
    try:
        demo = ui.build_interface()
        print("✓ Gradio interface built successfully")
        print(f"  Interface type: {type(demo).__name__}")
        
        print()
        print("✅ Interface building successful!")
        
    except Exception as e:
        print(f"❌ Interface building failed: {e}")
        return False
    
    print()
    
    # Test 5: History Management
    print("Test 5: Testing History Management")
    print("-"*80)
    
    try:
        # Add mock queries to history
        ui.query_history.append({
            "timestamp": "2025-11-08T15:00:00",
            "question": "What are transformers?",
            "answer": "Transformers are neural network architectures...",
            "num_contexts": 5,
            "top_score": 0.923
        })
        
        ui.query_history.append({
            "timestamp": "2025-11-08T15:01:00",
            "question": "How does attention work?",
            "answer": "Attention mechanisms allow models to...",
            "num_contexts": 5,
            "top_score": 0.856
        })
        
        history = ui.get_history()
        print(f"✓ History retrieved with {len(ui.query_history)} queries")
        print(f"  History text length: {len(history)} chars")
        
        # Test clear history
        ui.clear_history()
        print("✓ History cleared")
        print(f"  History length after clear: {len(ui.query_history)}")
        
        print()
        print("✅ History management working!")
        
    except Exception as e:
        print(f"❌ History management failed: {e}")
        return False
    
    print()
    
    # Summary
    print("="*80)
    print("PHASE 6 TEST COMPLETED SUCCESSFULLY! ✓")
    print("="*80)
    print()
    print("Summary:")
    print("  ✓ Component formatters working")
    print("  ✓ UI initialization successful")
    print("  ✓ Query processing components ready")
    print("  ✓ Gradio interface built")
    print("  ✓ History management working")
    print()
    print("Next steps:")
    print("  1. Launch UI: python -m src.ui.gradio_app")
    print("  2. Or: python demo_ui.py")
    print("  3. Test with real queries in browser")
    print()
    
    return True


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Phase 6 Enhanced UI")
    args = parser.parse_args()
    
    success = test_phase6_ui()
    
    if not success:
        print("\n⚠️  Some tests failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

