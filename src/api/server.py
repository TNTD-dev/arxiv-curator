"""FastAPI server for RAG pipeline with streaming support"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import json
import time
from loguru import logger

from src.config import settings
from src.rag.observable_pipeline import ObservableRAGPipeline


# ==================== Models ====================

class QueryRequest(BaseModel):
    """Query request model"""
    question: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(20, ge=5, le=50, description="Number of contexts to retrieve")
    mode: str = Field("default", description="Response mode: default, technical, beginner_friendly")
    use_reranker: bool = Field(True, description="Enable cross-encoder re-ranking")
    stream: bool = Field(False, description="Enable streaming response")


class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    contexts: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    pipeline_initialized: bool
    groq_api_configured: bool
    num_indexed_papers: int


class StatsResponse(BaseModel):
    """System statistics response"""
    total_queries: int
    indexed_papers: int
    vector_store_size: int
    reranker_enabled: bool


# ==================== Application ====================

app = FastAPI(
    title="arXiv Paper Curator API",
    description="RAG-powered research paper Q&A system",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
rag_pipeline: Optional[ObservableRAGPipeline] = None
query_count = 0


# ==================== Lifecycle ====================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup"""
    global rag_pipeline
    
    logger.info("Starting FastAPI server...")
    
    api_key = settings.GROQ_API_KEY
    if not api_key:
        logger.warning("⚠️ GROQ_API_KEY not configured!")
        return
    
    try:
        # Load Langfuse keys from settings
        langfuse_public = settings.LANGFUSE_PUBLIC_KEY
        langfuse_secret = settings.LANGFUSE_SECRET_KEY
        langfuse_host = settings.LANGFUSE_HOST
        
        # Log Langfuse status
        if langfuse_public and langfuse_secret:
            logger.info(f"Langfuse enabled: {langfuse_host}")
        else:
            logger.warning("⚠️ Langfuse keys not configured (observability disabled)")
        
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = ObservableRAGPipeline(
            groq_api_key=api_key,
            langfuse_public_key=langfuse_public,
            langfuse_secret_key=langfuse_secret,
            langfuse_host=langfuse_host,
            use_reranker=True
        )
        logger.success("✅ RAG pipeline initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down FastAPI server...")


# ==================== Endpoints ====================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    num_papers = 0
    if rag_pipeline and rag_pipeline.hybrid_retriever:
        try:
            # Try to get collection info
            collection = rag_pipeline.hybrid_retriever.vector_store.client.get_collection(
                settings.QDRANT_COLLECTION_NAME
            )
            num_papers = collection.count()
        except:
            pass
    
    return HealthResponse(
        status="healthy" if rag_pipeline else "degraded",
        pipeline_initialized=rag_pipeline is not None,
        groq_api_configured=bool(settings.GROQ_API_KEY),
        num_indexed_papers=num_papers
    )


@app.post("/api/query")
async def query_rag(request: QueryRequest):
    """
    Query the RAG pipeline
    
    Supports both regular and streaming responses
    """
    global query_count
    
    if not rag_pipeline:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Check GROQ_API_KEY."
        )
    
    # Check if reranker state needs to change
    current_reranker = rag_pipeline.use_reranker
    if current_reranker != request.use_reranker:
        logger.info(f"Reinitializing pipeline with reranker={request.use_reranker}")
        try:
            rag_pipeline.use_reranker = request.use_reranker
            if request.use_reranker:
                from src.retrieval.reranker import CrossEncoderReranker
                rag_pipeline.hybrid_retriever.reranker = CrossEncoderReranker()
            else:
                rag_pipeline.hybrid_retriever.reranker = None
        except Exception as e:
            logger.error(f"Failed to reinitialize: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Streaming response
    if request.stream:
        return StreamingResponse(
            stream_query_response(request),
            media_type="text/event-stream"
        )
    
    # Regular response
    try:
        query_count += 1
        start_time = time.time()
        
        rerank_top_k = max(3, request.top_k // 4) if request.use_reranker else None
        
        result = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k,
            mode=request.mode,
            stream=False,
            return_context=True,
            rerank_top_k=rerank_top_k
        )
        
        duration = time.time() - start_time
        
        # Add timing to metadata
        result["metadata"]["query_time"] = round(duration * 1000, 2)  # ms
        
        return result
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_query_response(request: QueryRequest):
    """
    Stream query response using SSE (Server-Sent Events)
    
    Sends events in format:
    - event: status | data: {"status": "searching"}
    - event: context | data: {context_data}
    - event: answer | data: {text_chunk}
    - event: done | data: {metadata}
    """
    global query_count
    
    # Create Langfuse trace for streaming query
    trace = rag_pipeline.observer.trace_query(
        query=request.question,
        metadata={
            "top_k": request.top_k,
            "mode": request.mode,
            "use_reranker": request.use_reranker,
            "stream": True
        }
    )
    
    try:
        query_count += 1
        start_time = time.time()
        
        with trace:
            # Send searching status
            yield f"event: status\ndata: {json.dumps({'status': 'searching'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Retrieve contexts
            rerank_top_k = max(3, request.top_k // 4) if request.use_reranker else None
            
            retrieval_start = time.time()
            contexts = rag_pipeline.hybrid_retriever.retrieve(
                request.question,
                top_k=request.top_k,
                rerank_top_k=rerank_top_k
            )
            retrieval_duration = time.time() - retrieval_start
            
            # Track retrieval in Langfuse
            rag_pipeline.observer.track_retrieval(
                trace_id=trace.id,
                query=request.question,
                results=contexts,
                duration=retrieval_duration,
                metadata={
                    "alpha": rag_pipeline.alpha,
                    "use_reranker": request.use_reranker
                }
            )
        
            if not contexts:
                yield f"event: error\ndata: {json.dumps({'error': 'No contexts found'})}\n\n"
                return
            
            # Send contexts
            contexts_data = [
                {
                    "text": ctx['text'],
                    "score": ctx.get('rerank_score', ctx.get('combined_score', 0)),
                    "combined_score": ctx.get('combined_score', 0),
                    "rerank_score": ctx.get('rerank_score', None),
                    "paper_id": ctx.get('metadata', {}).get('paper_id', 'Unknown'),
                    "section": ctx.get('metadata', {}).get('section', 'Unknown')
                }
                for ctx in contexts
            ]
            
            yield f"event: contexts\ndata: {json.dumps(contexts_data)}\n\n"
            await asyncio.sleep(0.1)
            
            # Generate streaming answer
            yield f"event: status\ndata: {json.dumps({'status': 'generating'})}\n\n"
            
            from src.llm.prompts import get_system_message
            system_message = get_system_message(request.mode)
            
            # Get streaming response from Groq
            llm_start = time.time()
            response_generator = rag_pipeline.llm.generate_with_context(
                query=request.question,
                contexts=contexts,
                system_message=system_message,
                stream=True
            )
            
            # Stream answer chunks and collect full response
            full_response = ""
            for chunk in response_generator:
                full_response += chunk
                yield f"event: answer\ndata: {json.dumps({'chunk': chunk})}\n\n"
                await asyncio.sleep(0.01)
            
            llm_duration = time.time() - llm_start
            
            # Track LLM generation in Langfuse
            rag_pipeline.observer.track_llm_generation(
                trace_id=trace.id,
                query=request.question,
                contexts=contexts,
                response=full_response,
                duration=llm_duration,
                model=rag_pipeline.groq_model_name,
                tokens={
                    "model": rag_pipeline.groq_model_name,
                    "response_length": len(full_response)
                }
            )
        
        # Send completion metadata
        duration = time.time() - start_time
        
        if rag_pipeline.use_reranker and contexts and 'rerank_score' in contexts[0]:
            top_score = contexts[0].get('rerank_score', 0)
            score_type = "rerank_score"
        else:
            top_score = contexts[0].get('combined_score', 0)
            score_type = "combined_score"
        
        metadata = {
            "num_contexts": len(contexts),
            "top_score": top_score,
            "score_type": score_type,
            "model": rag_pipeline.groq_model_name,
            "mode": request.mode,
            "use_reranker": rag_pipeline.use_reranker,
            "query_time": round(duration * 1000, 2)
        }
        
        yield f"event: done\ndata: {json.dumps(metadata)}\n\n"
        
        # Flush Langfuse events
        rag_pipeline.observer.flush()
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        # Track error in Langfuse
        rag_pipeline.observer.track_error(trace.id, e, "streaming")
        rag_pipeline.observer.flush()
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    
    num_papers = 0
    vector_size = 0
    
    if rag_pipeline and rag_pipeline.hybrid_retriever:
        try:
            collection = rag_pipeline.hybrid_retriever.vector_store.client.get_collection(
                settings.QDRANT_COLLECTION_NAME
            )
            num_papers = collection.count()
            vector_size = num_papers  # Approximate
        except:
            pass
    
    return StatsResponse(
        total_queries=query_count,
        indexed_papers=num_papers,
        vector_store_size=vector_size,
        reranker_enabled=rag_pipeline.use_reranker if rag_pipeline else False
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "arXiv Paper Curator API",
        "version": "1.0.0",
        "docs": "/api/docs"
    }


# ==================== Main ====================

def main():
    """Launch FastAPI server"""
    import uvicorn
    
    logger.info("🚀 Starting arXiv Paper Curator API...")
    logger.info("📚 Docs available at: http://localhost:8000/api/docs")
    
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()