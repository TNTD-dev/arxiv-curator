"""
Langfuse Integration for Observability

This module provides Langfuse integration for tracking RAG pipeline
performance, latency, and LLM usage.

Updated for Langfuse 3.9.1+ API
"""

from typing import Optional, Dict, Any, List
from loguru import logger
from datetime import datetime
import time
from functools import wraps

# Optional Langfuse imports - graceful fallback if not installed
try:
    from langfuse import Langfuse, observe
    from langfuse.types import TraceContext
    LANGFUSE_AVAILABLE = True
except ImportError:
    logger.warning("Langfuse not installed. Observability features will be disabled.")
    logger.info("Install with: pip install langfuse>=2.6.0")
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    observe = None
    TraceContext = None


class LangfuseObserver:
    """
    Langfuse observer for RAG pipeline monitoring
    
    Updated for Langfuse 3.9.1+ API which uses:
    - start_span() instead of span()
    - start_generation() instead of generation()
    - create_trace_id() to create traces
    """
    
    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize Langfuse observer
        
        Args:
            public_key: Langfuse public key
            secret_key: Langfuse secret key
            host: Langfuse host URL
            enabled: Enable/disable observability
        """
        self.enabled = enabled and public_key and secret_key and LANGFUSE_AVAILABLE
        self.client = None
        
        if not LANGFUSE_AVAILABLE:
            logger.info("Langfuse observer disabled (package not installed)")
            return
        
        if self.enabled:
            try:
                self.client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host or "https://cloud.langfuse.com"
                )
                logger.info("✓ Langfuse observer initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Langfuse: {e}")
                self.enabled = False
        else:
            logger.info("Langfuse observer disabled (no credentials)")
    
    def trace_query(
        self,
        query: str,
        metadata: Optional[Dict] = None
    ):
        """
        Create a trace for a query
        
        Args:
            query: User query
            metadata: Additional metadata
        
        Returns:
            TraceContext dict with trace_id
        """
        if not self.enabled:
            return DummyTrace()
        
        try:
            # Create a unique trace ID
            trace_id = Langfuse.create_trace_id()
            
            # Create trace context
            trace_context = {
                "trace_id": trace_id
            }
            
            # Create the initial span for the query
            span = self.client.start_span(
                name="rag_query",
                trace_context=trace_context,
                input={"query": query},
                metadata=metadata or {}
            )
            
            # Return a TraceWrapper that holds both context and span
            return TraceWrapper(trace_context, span)
            
        except Exception as e:
            logger.warning(f"Could not create trace: {e}")
            return DummyTrace()
    
    def track_retrieval(
        self,
        trace_id: str,
        query: str,
        results: List[Dict],
        duration: float,
        metadata: Optional[Dict] = None
    ):
        """
        Track retrieval step
        
        Args:
            trace_id: Parent trace ID
            query: Search query
            results: Retrieved results
            duration: Retrieval duration (seconds)
            metadata: Additional metadata
        """
        if not self.enabled:
            return
        
        try:
            trace_context = {"trace_id": trace_id}
            
            span = self.client.start_span(
                name="retrieval",
                trace_context=trace_context,
                input={"query": query},
                output={
                    "num_results": len(results),
                    "top_scores": [r.get('score', 0) for r in results[:5]]
                },
                metadata={
                    "duration_ms": duration * 1000,
                    "retrieval_method": "hybrid",
                    **(metadata or {})
                }
            )
            span.end()
            
        except Exception as e:
            logger.warning(f"Could not track retrieval: {e}")
    
    def track_reranking(
        self,
        trace_id: str,
        input_count: int,
        output_count: int,
        duration: float,
        score_changes: Optional[List[float]] = None
    ):
        """
        Track re-ranking step
        
        Args:
            trace_id: Parent trace ID
            input_count: Number of input results
            output_count: Number of output results
            duration: Re-ranking duration
            score_changes: Score improvements
        """
        if not self.enabled:
            return
        
        try:
            trace_context = {"trace_id": trace_id}
            
            span = self.client.start_span(
                name="reranking",
                trace_context=trace_context,
                input={"count": input_count},
                output={"count": output_count},
                metadata={
                    "duration_ms": duration * 1000,
                    "score_changes": score_changes or [],
                    "model": "cross-encoder"
                }
            )
            span.end()
            
        except Exception as e:
            logger.warning(f"Could not track reranking: {e}")
    
    def track_llm_generation(
        self,
        trace_id: str,
        query: str,
        contexts: List[Dict],
        response: str,
        duration: float,
        model: str,
        tokens: Optional[Dict] = None
    ):
        """
        Track LLM generation step
        
        Args:
            trace_id: Parent trace ID
            query: User query
            contexts: Retrieved contexts
            response: LLM response
            duration: Generation duration
            model: Model name
            tokens: Token usage info
        """
        if not self.enabled:
            return
        
        try:
            trace_context = {"trace_id": trace_id}
            
            generation = self.client.start_generation(
                name="llm_generation",
                trace_context=trace_context,
                input=[
                    {"role": "system", "content": "You are a research assistant"},
                    {"role": "user", "content": query}
                ],
                output=response,
                model=model,
                metadata={
                    "duration_ms": duration * 1000,
                    "num_contexts": len(contexts),
                    "response_length": len(response),
                    **(tokens or {})
                }
            )
            generation.end()
            
        except Exception as e:
            logger.warning(f"Could not track LLM generation: {e}")
    
    def track_error(
        self,
        trace_id: str,
        error: Exception,
        step: str
    ):
        """
        Track error
        
        Args:
            trace_id: Parent trace ID
            error: Exception that occurred
            step: Step where error occurred
        """
        if not self.enabled:
            return
        
        try:
            trace_context = {"trace_id": trace_id}
            
            span = self.client.start_span(
                name=f"error_{step}",
                trace_context=trace_context,
                level="ERROR",
                metadata={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "step": step
                }
            )
            span.end()
            
        except Exception as e:
            logger.warning(f"Could not track error: {e}")
    
    def flush(self):
        """Flush pending events"""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.warning(f"Could not flush Langfuse: {e}")


class TraceWrapper:
    """
    Wrapper for trace context and span to support context manager
    """
    
    def __init__(self, trace_context: Dict, span):
        self.trace_context = trace_context
        self.span = span
        self._trace_id = trace_context.get("trace_id", "dummy")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if self.span:
            try:
                self.span.end()
            except:
                pass
    
    @property
    def id(self):
        return self._trace_id


class DummyTrace:
    """Dummy trace for when Langfuse is disabled"""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    @property
    def id(self):
        return "dummy"


def observe_rag_query(func):
    """
    Decorator to observe RAG query function
    
    Usage:
        @observe_rag_query
        def query(self, question: str, ...):
            ...
    """
    @wraps(func)
    def wrapper(self, question: str, *args, **kwargs):
        # Check if observability is enabled
        observer = getattr(self, 'observer', None)
        
        if not observer or not observer.enabled:
            return func(self, question, *args, **kwargs)
        
        # Create trace
        trace = observer.trace_query(
            query=question,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "function": func.__name__
            }
        )
        
        start_time = time.time()
        
        try:
            with trace:
                result = func(self, question, *args, **kwargs)
                
                # Track overall duration
                duration = time.time() - start_time
                
                # Update the trace with final results
                if isinstance(result, dict) and observer.enabled:
                    try:
                        trace_context = {"trace_id": trace.id}
                        final_span = observer.client.start_span(
                            name="query_complete",
                            trace_context=trace_context,
                            output={
                                "answer_length": len(result.get('answer', '')),
                                "num_contexts": result.get('metadata', {}).get('num_contexts', 0),
                                "duration_ms": duration * 1000
                            }
                        )
                        final_span.end()
                    except:
                        pass
                
                return result
                
        except Exception as e:
            observer.track_error(trace.id, e, "query")
            raise
        finally:
            observer.flush()
    
    return wrapper


class MetricsCollector:
    """
    Collect and aggregate metrics from Langfuse
    """
    
    def __init__(self, langfuse_client: Langfuse):
        """
        Initialize metrics collector
        
        Args:
            langfuse_client: Langfuse client instance
        """
        self.client = langfuse_client
    
    def get_query_metrics(
        self,
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """
        Get query metrics
        
        Args:
            time_range: Time range (e.g., "24h", "7d", "30d")
        
        Returns:
            Metrics dictionary
        """
        try:
            # This would use Langfuse API to fetch metrics
            # For now, return placeholder
            return {
                "total_queries": 0,
                "avg_latency_ms": 0,
                "avg_contexts": 0,
                "error_rate": 0
            }
        except Exception as e:
            logger.error(f"Could not fetch metrics: {e}")
            return {}
    
    def get_llm_metrics(
        self,
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """
        Get LLM usage metrics
        
        Args:
            time_range: Time range
        
        Returns:
            LLM metrics
        """
        try:
            return {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "estimated_cost": 0.0
            }
        except Exception as e:
            logger.error(f"Could not fetch LLM metrics: {e}")
            return {}


def create_observer(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None
) -> LangfuseObserver:
    """
    Create Langfuse observer
    
    Args:
        public_key: Langfuse public key
        secret_key: Langfuse secret key
        host: Langfuse host
    
    Returns:
        LangfuseObserver instance
    """
    return LangfuseObserver(
        public_key=public_key,
        secret_key=secret_key,
        host=host
    )


def main():
    """Example usage"""
    import os
    
    print("\n" + "="*70)
    print("Langfuse Observer Demo (Langfuse 3.9.1+)")
    print("="*70)
    
    # Check if Langfuse is available
    if not LANGFUSE_AVAILABLE:
        print("\n❌ Langfuse not installed!")
        print("Install with: pip install langfuse>=2.6.0")
        return
    
    # Check credentials
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    
    if not public_key or not secret_key:
        print("\n⚠️  Langfuse credentials not found")
        print("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env")
        return
    
    # Create observer
    observer = create_observer(public_key, secret_key)
    
    if not observer.enabled:
        print("\n❌ Observer not enabled")
        return
    
    print("\n✓ Observer initialized")
    
    # Simulate query tracking
    print("\nSimulating query tracking...")
    trace = observer.trace_query(
        query="What are transformer models?",
        metadata={"user": "demo"}
    )
    
    with trace:
        # Simulate retrieval
        observer.track_retrieval(
            trace_id=trace.id,
            query="What are transformer models?",
            results=[{"score": 0.95}, {"score": 0.87}],
            duration=0.15,
            metadata={"method": "hybrid"}
        )
        
        # Simulate LLM generation
        observer.track_llm_generation(
            trace_id=trace.id,
            query="What are transformer models?",
            contexts=[],
            response="Transformers are...",
            duration=1.2,
            model="llama-3.3-70b",
            tokens={"input": 100, "output": 50}
        )
    
    observer.flush()
    
    print("✓ Query tracked successfully")
    print("\nCheck Langfuse dashboard for results!")
    print("  https://cloud.langfuse.com")
    print("="*70)


if __name__ == "__main__":
    main()
