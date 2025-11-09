"""Configuration settings for AI Research Paper Navigator"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    TEXTS_DIR: Path = PROCESSED_DIR / "texts"
    TABLES_DIR: Path = PROCESSED_DIR / "tables"
    FIGURES_DIR: Path = PROCESSED_DIR / "figures"
    VECTOR_DB_DIR: Path = DATA_DIR / "vector_db"
    
    # arXiv Configuration
    ARXIV_MAX_RESULTS: int = 15  # Increased for better coverage
    ARXIV_QUERY: str = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL"  # AI, ML, NLP
    
    # Chunking Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # Embeddings Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Vector Store Configuration (Qdrant)
    VECTOR_STORE_TYPE: str = "qdrant"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "arxiv_papers"
    QDRANT_DISTANCE: str = "Cosine"  # or "Euclid", "Dot"
    
    # Retrieval Configuration
    TOP_K_RETRIEVAL: int = 20
    TOP_K_RERANK: int = 5
    
    # Groq LLM Configuration
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_MAX_TOKENS: int = 2048
    GROQ_TEMPERATURE: float = 0.1
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    
    # Alternative Groq models
    # - llama-3.3-70b-versatile: Latest, best quality (default)
    # - llama-3.1-70b-versatile: Previous version, still excellent
    # - mixtral-8x7b-32768: Good balance, large context
    # - llama-3.1-8b-instant: Fastest, smaller
    # - gemma2-9b-it: Good for reasoning
    
    # OpenAI Configuration (for RAGAS evaluation)
    OPENAI_API_KEY: Optional[str] = ""
    
    # Langfuse Observability Configuration
    LANGFUSE_PUBLIC_KEY: Optional[str] = ""
    LANGFUSE_SECRET_KEY: Optional[str] = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    
    # Re-ranking Configuration
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # BM25 Configuration
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Initialize settings
settings = Settings()

# Create necessary directories on import
def create_directories():
    """Create all required directories if they don't exist"""
    directories = [
        settings.DATA_DIR,
        settings.RAW_DIR,
        settings.PROCESSED_DIR,
        settings.TEXTS_DIR,
        settings.TABLES_DIR,
        settings.FIGURES_DIR,
        settings.VECTOR_DB_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Auto-create directories when config is imported
create_directories()

