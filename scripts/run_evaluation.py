"""
Run RAGAS Evaluation on RAG Pipeline

This script runs a comprehensive evaluation of the RAG pipeline using RAGAS metrics.
It generates test questions, evaluates the pipeline, and produces detailed reports.

Usage:
    python scripts/run_evaluation.py [--num-questions N] [--with-reranker] [--output-dir DIR]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.config import settings
from src.rag.pipeline import RAGPipeline
from src.evaluation.ragas_evaluator import RAGASEvaluator
import os


def main():
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on RAG pipeline"
    )
    parser.add_argument(
        '--num-questions',
        type=int,
        default=10,
        help='Number of test questions to evaluate (default: 10)'
    )
    parser.add_argument(
        '--with-reranker',
        action='store_true',
        help='Enable cross-encoder re-ranking'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: data/evaluation)'
    )
    parser.add_argument(
        '--custom-questions',
        type=str,
        default=None,
        help='Path to JSON file with custom questions'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("RAGAS Evaluation Pipeline")
    print("="*80)
    
    # Check API key
    api_key = os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY
    if not api_key:
        print("\n❌ GROQ_API_KEY not found!")
        print("Please set it in .env file or environment variable")
        print("\nExample:")
        print("  export GROQ_API_KEY=gsk_...")
        print("  # or")
        print("  echo 'GROQ_API_KEY=gsk_...' >> .env")
        return 1
    
    print(f"\n✓ API key found: {api_key[:20]}...")
    
    # Initialize pipeline
    print("\n" + "-"*80)
    print("Step 1: Initializing RAG Pipeline")
    print("-"*80)
    
    try:
        pipeline = RAGPipeline(
            groq_api_key=api_key,
            use_reranker=args.with_reranker
        )
        
        # Check if vector store has data
        count = pipeline.vector_store.count()
        if count == 0:
            print("\n❌ Vector store is empty!")
            print("\nPlease index documents first using:")
            print("  python scripts/fetch_and_index_papers.py")
            return 1
        
        print(f"\n✓ Pipeline initialized successfully")
        print(f"  Documents indexed: {count}")
        print(f"  LLM model: {pipeline.groq_model_name}")
        print(f"  Embedding model: {pipeline.embedding_model_name}")
        print(f"  Re-ranking: {'Enabled' if args.with_reranker else 'Disabled'}")
        
    except Exception as e:
        print(f"\n❌ Failed to initialize pipeline: {e}")
        logger.exception(e)
        return 1
    
    # Initialize evaluator
    print("\n" + "-"*80)
    print("Step 2: Initializing RAGAS Evaluator")
    print("-"*80)
    
    try:
        evaluator = RAGASEvaluator(
            pipeline=pipeline,
            use_groq_for_eval=True
        )
        print("\n✓ Evaluator initialized")
    except Exception as e:
        print(f"\n❌ Failed to initialize evaluator: {e}")
        logger.exception(e)
        return 1
    
    # Load custom questions if provided
    custom_questions = None
    if args.custom_questions:
        print(f"\nLoading custom questions from {args.custom_questions}...")
        try:
            import json
            with open(args.custom_questions, 'r', encoding='utf-8') as f:
                custom_questions = json.load(f)
            print(f"✓ Loaded {len(custom_questions)} custom questions")
        except Exception as e:
            print(f"⚠ Failed to load custom questions: {e}")
            custom_questions = None
    
    # Run evaluation
    print("\n" + "-"*80)
    print("Step 3: Running RAGAS Evaluation")
    print("-"*80)
    print(f"\nThis will evaluate {args.num_questions} questions.")
    print("Evaluation may take 3-5 minutes depending on the number of questions...")
    print()
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else settings.DATA_DIR / "evaluation"
    
    try:
        results = evaluator.evaluate_and_save(
            output_dir=output_dir,
            num_questions=args.num_questions,
            custom_questions=custom_questions,
            save_dataset=True
        )
        
        # Check if mock results
        is_mock = results.get('is_mock', False)
        
        # Display results
        print("\n" + "="*80)
        if is_mock:
            print("EVALUATION RESULTS (MOCK - For Demonstration)")
        else:
            print("EVALUATION RESULTS")
        print("="*80)
        
        if is_mock:
            print("\n⚠️  WARNING: These are MOCK scores (OpenAI API not provided)")
            print("For real evaluation, set OPENAI_API_KEY environment variable")
            print("Get key at: https://platform.openai.com/api-keys\n")
        
        print()
        
        scores = results['scores']
        
        # Display each metric
        for metric, score in scores.items():
            percentage = score * 100
            bar = '█' * int(percentage / 5) + '░' * (20 - int(percentage / 5))
            metric_name = metric.replace('_', ' ').title()
            print(f"{metric_name:.<35} {score:.4f} ({percentage:.1f}%)")
            print(f"{bar}\n")
        
        # Calculate and display average
        avg_score = sum(scores.values()) / len(scores)
        print("-"*80)
        print(f"{'AVERAGE SCORE':.<35} {avg_score:.4f} ({avg_score*100:.1f}%)")
        print("-"*80)
        
        # Performance assessment
        print("\nPERFORMANCE ASSESSMENT:")
        if avg_score >= 0.8:
            print("✅ EXCELLENT - Pipeline is production-ready")
        elif avg_score >= 0.6:
            print("⚠️  GOOD - Pipeline works well but has room for improvement")
        else:
            print("❌ NEEDS IMPROVEMENT - Optimization required")
        
        # Output files
        print("\n" + "="*80)
        print("OUTPUT FILES")
        print("="*80)
        print(f"\n📊 Results:  {output_dir / 'evaluation_results.json'}")
        print(f"📄 Report:   {output_dir / 'evaluation_report.md'}")
        print(f"📝 Dataset:  {output_dir / 'test_dataset.json'}")
        
        print("\n" + "="*80)
        print("✓ EVALUATION COMPLETE!")
        print("="*80)
        
        print(f"\nView the detailed report at:")
        print(f"  {output_dir / 'evaluation_report.md'}")
        
        if is_mock:
            print("\n" + "="*80)
            print("⚠️  IMPORTANT: Mock Evaluation Notice")
            print("="*80)
            print("\nThese are demonstration scores (OpenAI API key not provided).")
            print("\nFor REAL evaluation with actual RAGAS metrics:")
            print("  1. Get OpenAI API key: https://platform.openai.com/api-keys")
            print("  2. Set environment variable:")
            print("     export OPENAI_API_KEY=sk-...")
            print("  3. Re-run: python scripts/run_evaluation.py")
            print("\nNote: OpenAI API has free tier with $5 credit for new accounts.")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

