"""
Observable RAG Pipeline with Langfuse Integration

This module extends the base RAG pipeline with observability features.
"""

from pathlib import Path
from typing import Dict, Optional, Generator
import time
from loguru import logger

from src.rag.pipeline import RAGPipeline
from src.observability.langfuse_client import LangfuseObserver


class ObservableRAGPipeline(RAGPipeline):
    """
    RAG Pipeline with Langfuse observability
    """
    
    def __init__(
        self,
        groq_api_key: str,
        langfuse_public_key: Optional[str] = None,
        langfuse_secret_key: Optional[str] = None,
        langfuse_host: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize observable RAG pipeline
        
        Args:
            groq_api_key: Groq API key
            langfuse_public_key: Langfuse public key (optional)
            langfuse_secret_key: Langfuse secret key (optional)
            langfuse_host: Langfuse host URL (optional)
            **kwargs: Additional arguments for RAGPipeline
        """
        # Initialize base pipeline
        super().__init__(groq_api_key=groq_api_key, **kwargs)
        
        # Initialize observer
        self.observer = LangfuseObserver(
            public_key=langfuse_public_key,
            secret_key=langfuse_secret_key,
            host=langfuse_host
        )
        
        logger.info(f"Observable RAG Pipeline initialized (observability={'enabled' if self.observer.enabled else 'disabled'})")
    
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
        Query with observability tracking
        
        Args:
            question: User question
            top_k: Number of contexts to retrieve
            mode: System message mode
            stream: Enable streaming
            return_context: Include contexts in response
            rerank_top_k: Number after re-ranking
        
        Returns:
            Query result dictionary
        """
        # Start trace
        trace = self.observer.trace_query(
            query=question,
            metadata={
                "top_k": top_k,
                "mode": mode,
                "use_reranker": self.use_reranker,
                "model": self.groq_model_name
            }
        )
        
        overall_start = time.time()
        
        try:
            with trace:
                # Step 1: Retrieval
                retrieval_start = time.time()
                contexts = self.hybrid_retriever.retrieve(
                    question,
                    top_k=top_k,
                    rerank_top_k=rerank_top_k
                )
                retrieval_duration = time.time() - retrieval_start
                
                # Track retrieval
                self.observer.track_retrieval(
                    trace_id=trace.id,
                    query=question,
                    results=contexts,
                    duration=retrieval_duration,
                    metadata={
                        "alpha": self.alpha,
                        "use_reranker": self.use_reranker
                    }
                )
                
                if not contexts:
                    logger.warning("No contexts retrieved")
                    return {
                        "answer": "I don't have enough information to answer this question.",
                        "contexts": [],
                        "metadata": {"status": "no_context"}
                    }
                
                # Step 2: LLM Generation
                from src.llm.prompts import format_rag_prompt, get_system_message
                
                system_message = get_system_message(mode)
                llm_start = time.time()
                
                response = self.llm.generate_with_context(
                    query=question,
                    contexts=contexts,
                    system_message=system_message,
                    stream=stream
                )
                
                llm_duration = time.time() - llm_start
                
                # Track LLM generation
                self.observer.track_llm_generation(
                    trace_id=trace.id,
                    query=question,
                    contexts=contexts,
                    response=response if isinstance(response, str) else "",
                    duration=llm_duration,
                    model=self.groq_model_name,
                    tokens={
                        "model": self.groq_model_name,
                        "temperature": self.llm.temperature
                    }
                )
                
                # Build result
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
                        "use_reranker": self.use_reranker,
                        "retrieval_duration_ms": retrieval_duration * 1000,
                        "llm_duration_ms": llm_duration * 1000,
                        "total_duration_ms": (time.time() - overall_start) * 1000
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
            self.observer.track_error(trace.id, e, "query")
            return {
                "answer": f"Error processing query: {str(e)}",
                "contexts": [],
                "metadata": {"status": "error", "error": str(e)}
            }
        finally:
            self.observer.flush()


def main():
    """Example usage"""
    from src.config import settings
    import os
    
    print("\n" + "="*80)
    print("Observable RAG Pipeline Demo")
    print("="*80)
    print()
    
    # Check keys
    groq_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    langfuse_public = settings.LANGFUSE_PUBLIC_KEY or os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret = settings.LANGFUSE_SECRET_KEY or os.getenv("LANGFUSE_SECRET_KEY")
    
    if not groq_key:
        print("❌ GROQ_API_KEY not found!")
        return
    
    print(f"✓ Groq API key: {groq_key[:20]}...")
    
    if not langfuse_public or not langfuse_secret:
        print("⚠️  Langfuse keys not found (observability disabled)")
        langfuse_public = None
        langfuse_secret = None
    else:
        print(f"✓ Langfuse public key: {langfuse_public[:20]}...")
    
    print()
    
    # Initialize pipeline
    print("Initializing Observable RAG Pipeline...")
    rag = ObservableRAGPipeline(
        groq_api_key=groq_key,
        langfuse_public_key=langfuse_public,
        langfuse_secret_key=langfuse_secret
    )
    
    print("✓ Pipeline initialized")
    print()
    
    # Test query
    test_query = "What are the main innovations in transformer architecture?"
    print(f"Query: {test_query}")
    print()
    
    result = rag.query(test_query, top_k=5)
    
    print("Result:")
    print(f"  Answer: {result['answer'][:200]}...")
    print(f"  Contexts: {result['metadata']['num_contexts']}")
    print(f"  Retrieval: {result['metadata']['retrieval_duration_ms']:.2f}ms")
    print(f"  LLM: {result['metadata']['llm_duration_ms']:.2f}ms")
    print(f"  Total: {result['metadata']['total_duration_ms']:.2f}ms")
    
    if rag.observer.enabled:
        print()
        print("✓ Query tracked in Langfuse!")
        print("  Check dashboard: https://cloud.langfuse.com")
    
    print()
    print("="*80)


if __name__ == "__main__":
    main()

