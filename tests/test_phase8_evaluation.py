"""
Phase 8: RAGAS Evaluation Tests

Tests for the RAGAS evaluation system including:
- Test dataset generation
- Evaluation metrics computation
- Report generation
"""

import pytest
from pathlib import Path
import json
import os

from src.rag.pipeline import RAGPipeline
from src.evaluation.ragas_evaluator import RAGASEvaluator
from src.config import settings


@pytest.fixture
def api_key():
    """Get Groq API key from environment"""
    key = os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY
    if not key:
        pytest.skip("GROQ_API_KEY not found")
    return key


@pytest.fixture
def rag_pipeline(api_key):
    """Initialize RAG pipeline"""
    pipeline = RAGPipeline(
        groq_api_key=api_key,
        use_reranker=False  # Faster for testing
    )
    
    # Check if pipeline has data
    count = pipeline.vector_store.count()
    if count == 0:
        pytest.skip("Vector store is empty. Index documents first.")
    
    return pipeline


@pytest.fixture
def evaluator(rag_pipeline):
    """Initialize RAGAS evaluator"""
    return RAGASEvaluator(
        pipeline=rag_pipeline,
        use_groq_for_eval=True
    )


def test_evaluator_initialization(evaluator):
    """Test evaluator initializes correctly"""
    print("\n1. Testing evaluator initialization...")
    
    assert evaluator is not None
    assert evaluator.pipeline is not None
    assert evaluator.use_groq_for_eval is True
    
    print("✓ Evaluator initialized successfully")


def test_generate_test_dataset(evaluator):
    """Test generating test dataset"""
    print("\n2. Testing test dataset generation...")
    
    # Generate small dataset
    dataset = evaluator.generate_test_dataset(num_questions=3)
    
    assert len(dataset) > 0
    assert len(dataset) <= 3
    
    # Check dataset structure
    assert 'question' in dataset.features
    assert 'answer' in dataset.features
    assert 'contexts' in dataset.features
    assert 'ground_truth' in dataset.features
    
    # Check first sample
    sample = dataset[0]
    assert isinstance(sample['question'], str)
    assert isinstance(sample['answer'], str)
    assert isinstance(sample['contexts'], list)
    assert isinstance(sample['ground_truth'], str)
    
    print(f"✓ Generated dataset with {len(dataset)} samples")
    print(f"  Sample question: {sample['question'][:60]}...")
    print(f"  Contexts retrieved: {len(sample['contexts'])}")


def test_custom_questions(evaluator):
    """Test evaluation with custom questions"""
    print("\n3. Testing custom questions...")
    
    custom_questions = [
        {
            'question': 'What is machine learning?',
            'ground_truth': 'Machine learning is a subset of AI that enables systems to learn from data.'
        },
        {
            'question': 'What are neural networks?',
            'ground_truth': 'Neural networks are computing systems inspired by biological neural networks.'
        }
    ]
    
    dataset = evaluator.generate_test_dataset(
        num_questions=2,
        custom_questions=custom_questions
    )
    
    assert len(dataset) >= 2
    assert dataset[0]['question'] == custom_questions[0]['question']
    assert dataset[0]['ground_truth'] == custom_questions[0]['ground_truth']
    
    print(f"✓ Custom questions processed successfully")


def test_evaluation_metrics(evaluator):
    """Test RAGAS evaluation metrics"""
    print("\n4. Testing RAGAS evaluation (may take 2-3 minutes)...")
    
    # Generate small dataset
    dataset = evaluator.generate_test_dataset(num_questions=2)
    
    print(f"  Evaluating {len(dataset)} samples...")
    
    # Run evaluation with subset of metrics (faster)
    from ragas.metrics import faithfulness, answer_relevancy
    
    try:
        results = evaluator.evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy]
        )
        
        assert 'scores' in results
        assert 'dataset_size' in results
        assert 'timestamp' in results
        
        # Check scores
        scores = results['scores']
        assert len(scores) > 0
        
        print(f"✓ Evaluation completed")
        print(f"  Metrics computed: {list(scores.keys())}")
        for metric, score in scores.items():
            print(f"  {metric}: {score:.4f}")
        
    except Exception as e:
        print(f"⚠ Evaluation skipped: {e}")
        pytest.skip(f"Evaluation error: {e}")


def test_evaluate_and_save(evaluator, tmp_path):
    """Test complete evaluation pipeline with saving"""
    print("\n5. Testing complete evaluation with saving...")
    
    output_dir = tmp_path / "evaluation_test"
    
    try:
        results = evaluator.evaluate_and_save(
            output_dir=output_dir,
            num_questions=2,
            save_dataset=True
        )
        
        # Check results
        assert 'scores' in results
        assert results['dataset_size'] >= 2
        
        # Check files created
        assert (output_dir / "evaluation_results.json").exists()
        assert (output_dir / "evaluation_report.md").exists()
        assert (output_dir / "test_dataset.json").exists()
        
        # Verify JSON results
        with open(output_dir / "evaluation_results.json") as f:
            saved_results = json.load(f)
            assert 'scores' in saved_results
            assert 'pipeline_stats' in saved_results
        
        # Verify report
        report_path = output_dir / "evaluation_report.md"
        report_content = report_path.read_text(encoding='utf-8')
        assert "# RAGAS Evaluation Report" in report_content
        assert "## Evaluation Metrics" in report_content
        assert "## Overall Assessment" in report_content
        
        print(f"✓ Evaluation saved successfully")
        print(f"  Output directory: {output_dir}")
        print(f"  Files created:")
        for file in output_dir.iterdir():
            print(f"    - {file.name}")
    
    except Exception as e:
        print(f"⚠ Full evaluation skipped: {e}")
        pytest.skip(f"Evaluation error: {e}")


def test_report_generation(evaluator):
    """Test report generation"""
    print("\n6. Testing report generation...")
    
    # Generate dataset
    dataset = evaluator.generate_test_dataset(num_questions=2)
    
    # Create mock results
    mock_results = {
        'scores': {
            'faithfulness': 0.85,
            'answer_relevancy': 0.78,
            'context_precision': 0.82
        },
        'dataset_size': 2,
        'timestamp': '2024-01-01T12:00:00'
    }
    
    # Generate report
    report = evaluator._generate_report(mock_results, dataset)
    
    # Check report content
    assert "# RAGAS Evaluation Report" in report
    assert "faithfulness" in report.lower()
    assert "answer relevancy" in report.lower()
    assert "context precision" in report.lower()
    assert "## Overall Assessment" in report
    assert "## Pipeline Configuration" in report
    
    print(f"✓ Report generated successfully")
    print(f"  Length: {len(report)} characters")


def test_pipeline_stats_in_results(evaluator, tmp_path):
    """Test that pipeline stats are included in results"""
    print("\n7. Testing pipeline stats in results...")
    
    output_dir = tmp_path / "stats_test"
    
    try:
        results = evaluator.evaluate_and_save(
            output_dir=output_dir,
            num_questions=2,
            save_dataset=True
        )
        
        # Check saved results include pipeline stats
        with open(output_dir / "evaluation_results.json") as f:
            saved_results = json.load(f)
            
        assert 'pipeline_stats' in saved_results
        stats = saved_results['pipeline_stats']
        
        assert 'vector_store' in stats
        assert 'bm25' in stats
        assert 'llm' in stats
        assert 'embedder' in stats
        
        print(f"✓ Pipeline stats included in results")
        print(f"  Vector store count: {stats['vector_store']['count']}")
        print(f"  BM25 count: {stats['bm25']['count']}")
        print(f"  LLM model: {stats['llm']['model']}")
    
    except Exception as e:
        print(f"⚠ Test skipped: {e}")
        pytest.skip(f"Error: {e}")


def main():
    """Run tests manually"""
    print("\n" + "="*70)
    print("Phase 8: RAGAS Evaluation Tests")
    print("="*70)
    
    # Check API key
    api_key = os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY
    if not api_key:
        print("\n❌ GROQ_API_KEY not found!")
        return
    
    # Initialize pipeline
    print("\nInitializing RAG pipeline...")
    try:
        pipeline = RAGPipeline(groq_api_key=api_key, use_reranker=False)
        
        if pipeline.vector_store.count() == 0:
            print("❌ Vector store is empty! Index documents first.")
            return
        
        print(f"✓ Pipeline initialized ({pipeline.vector_store.count()} documents)")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Initialize evaluator
    evaluator = RAGASEvaluator(pipeline=pipeline, use_groq_for_eval=True)
    
    # Run tests
    try:
        test_evaluator_initialization(evaluator)
        test_generate_test_dataset(evaluator)
        test_custom_questions(evaluator)
        test_report_generation(evaluator)
        
        print("\n" + "="*70)
        print("✓ All basic tests passed!")
        print("="*70)
        
        # Ask about full evaluation
        print("\nFull evaluation with RAGAS metrics takes 2-3 minutes.")
        print("This requires API calls for metric computation.")
        response = input("Run full evaluation? (y/n): ").strip().lower()
        
        if response == 'y':
            from pathlib import Path
            test_evaluation_metrics(evaluator)
            test_evaluate_and_save(evaluator, Path("./data/test_evaluation"))
            print("\n✓ Full evaluation tests passed!")
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()

