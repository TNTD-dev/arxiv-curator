"""
Generate Custom Evaluation Questions

This script helps create custom evaluation questions from your indexed papers.
It can extract key topics and generate questions with ground truth answers.

Usage:
    python scripts/generate_custom_questions.py --output custom_questions.json
"""

import argparse
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.rag.pipeline import RAGPipeline
from loguru import logger
import os


def generate_questions_from_papers(pipeline: RAGPipeline, num_questions: int = 15) -> list:
    """
    Generate evaluation questions based on indexed papers
    
    Args:
        pipeline: Initialized RAG pipeline
        num_questions: Number of questions to generate
    
    Returns:
        List of question dictionaries
    """
    print(f"\nGenerating {num_questions} questions from indexed papers...")
    
    # Topics common in AI/ML papers
    topics = [
        "transformer architecture and attention mechanisms",
        "neural network training and optimization",
        "transfer learning and fine-tuning",
        "embeddings and representation learning",
        "generative models and GANs",
        "reinforcement learning",
        "model evaluation metrics",
        "regularization techniques",
        "convolutional neural networks",
        "recurrent neural networks and LSTMs",
        "batch normalization and layer normalization",
        "activation functions",
        "loss functions and optimization",
        "data augmentation techniques",
        "model compression and efficiency"
    ]
    
    questions = []
    
    for i, topic in enumerate(topics[:num_questions]):
        # Generate question about the topic
        question = f"Explain {topic} based on the research papers."
        
        # Query pipeline to get relevant info
        try:
            result = pipeline.query(
                question=question,
                top_k=3,
                stream=False,
                return_context=True
            )
            
            answer = result.get('answer', '')
            contexts = result.get('contexts', [])
            
            # Use the generated answer as ground truth
            ground_truth = answer if answer else f"Information about {topic} from research papers."
            
            questions.append({
                'question': question,
                'ground_truth': ground_truth,
                'topic': topic,
                'num_contexts': len(contexts)
            })
            
            print(f"  {i+1}. Generated question about: {topic}")
            
        except Exception as e:
            logger.error(f"Error generating question for topic '{topic}': {e}")
            continue
    
    return questions


def create_template_questions() -> list:
    """
    Create a template of custom questions that users can fill in
    
    Returns:
        List of template question dictionaries
    """
    return [
        {
            'question': 'What are the main contributions of this research?',
            'ground_truth': '[Replace with expected answer about key contributions]',
            'category': 'Research Overview'
        },
        {
            'question': 'What methodology does the paper use?',
            'ground_truth': '[Replace with expected answer about methodology]',
            'category': 'Methodology'
        },
        {
            'question': 'What are the experimental results?',
            'ground_truth': '[Replace with expected answer about results]',
            'category': 'Results'
        },
        {
            'question': 'What are the limitations mentioned?',
            'ground_truth': '[Replace with expected answer about limitations]',
            'category': 'Limitations'
        },
        {
            'question': 'What future work is suggested?',
            'ground_truth': '[Replace with expected answer about future work]',
            'category': 'Future Work'
        }
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate custom evaluation questions"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='custom_questions.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--num-questions',
        type=int,
        default=10,
        help='Number of questions to generate'
    )
    parser.add_argument(
        '--template-only',
        action='store_true',
        help='Generate template questions only (no RAG queries)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Custom Question Generator")
    print("="*70)
    
    output_path = Path(args.output)
    
    if args.template_only:
        # Generate template
        print("\nGenerating template questions...")
        questions = create_template_questions()
        print(f"✓ Created {len(questions)} template questions")
        
    else:
        # Generate from indexed papers
        print("\nGenerating questions from indexed papers...")
        
        # Check API key
        api_key = os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY
        if not api_key:
            print("\n❌ GROQ_API_KEY not found!")
            return 1
        
        # Initialize pipeline
        try:
            print("Initializing RAG pipeline...")
            pipeline = RAGPipeline(groq_api_key=api_key, use_reranker=False)
            
            if pipeline.vector_store.count() == 0:
                print("\n❌ Vector store is empty!")
                print("Please index documents first.")
                return 1
            
            print(f"✓ Pipeline initialized ({pipeline.vector_store.count()} documents)")
            
        except Exception as e:
            print(f"\n❌ Failed to initialize pipeline: {e}")
            return 1
        
        # Generate questions
        questions = generate_questions_from_papers(pipeline, args.num_questions)
    
    # Save questions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Questions saved to: {output_path}")
    print(f"  Total questions: {len(questions)}")
    
    # Show sample
    if questions:
        print("\nSample question:")
        print("-"*70)
        sample = questions[0]
        print(f"Q: {sample['question']}")
        print(f"A: {sample['ground_truth'][:150]}...")
    
    print("\n" + "="*70)
    print("✓ Generation complete!")
    print("="*70)
    
    if not args.template_only:
        print("\nYou can now use these questions for evaluation:")
        print(f"  python scripts/run_evaluation.py --custom-questions {output_path}")
    else:
        print(f"\nEdit {output_path} to fill in your expected answers,")
        print("then use it for evaluation:")
        print(f"  python scripts/run_evaluation.py --custom-questions {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

