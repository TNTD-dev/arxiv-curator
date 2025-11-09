"""
Fetch and Index Papers from arXiv

This script fetches recent papers from arXiv and indexes them into the vector store.
It's a complete pipeline: fetch → process → chunk → index.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.data.arxiv_client import ArxivClient
from src.data.docling_processor import DoclingProcessor
from src.data.chunker import SemanticChunker
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index
from loguru import logger
import arxiv
from typing import List, Dict
import json


def fetch_papers(
    query: str = None,
    max_results: int = None,
    sort_by: str = "submitted"
) -> List[Path]:
    """
    Fetch papers from arXiv
    
    Args:
        query: Search query (uses config default if None)
        max_results: Number of papers (uses config default if None)
        sort_by: Sort criterion (submitted, relevance, updated)
    
    Returns:
        List of downloaded PDF paths
    """
    print("\n" + "="*80)
    print("📥 FETCHING PAPERS FROM ARXIV")
    print("="*80)
    print()
    
    # Use config defaults if not specified
    query = query or settings.ARXIV_QUERY
    max_results = max_results or settings.ARXIV_MAX_RESULTS
    
    # Map string to enum
    sort_map = {
        "submitted": arxiv.SortCriterion.SubmittedDate,
        "relevance": arxiv.SortCriterion.Relevance,
        "updated": arxiv.SortCriterion.LastUpdatedDate
    }
    sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.SubmittedDate)
    
    print(f"Query: {query}")
    print(f"Max results: {max_results}")
    print(f"Sort by: {sort_by}")
    print()
    
    # Initialize client
    client = ArxivClient(
        download_dir=settings.RAW_DIR,
        max_results=max_results
    )
    
    # Search papers
    print("Searching arXiv...")
    papers = client.search(query, max_results=max_results, sort_by=sort_criterion)
    print(f"✓ Found {len(papers)} papers")
    print()
    
    # Display paper info
    print("Papers to download:")
    print("-"*80)
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper.title}")
        print(f"   ID: {paper.get_short_id()}")
        print(f"   Published: {paper.published.strftime('%Y-%m-%d')}")
        print(f"   Authors: {', '.join(a.name for a in paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        print()
    
    # Download
    print("Downloading papers...")
    print("-"*80)
    downloaded = client.download_papers(papers)
    
    # Extract PDF paths from metadata
    pdf_paths = [Path(metadata['filepath']) for metadata in downloaded]
    
    print()
    print(f"✓ Downloaded {len(pdf_paths)} papers to {settings.RAW_DIR}")
    print()
    
    return pdf_paths


def process_papers(pdf_paths: List[Path]) -> List[Dict]:
    """
    Process PDFs with Docling
    
    Args:
        pdf_paths: List of PDF file paths
    
    Returns:
        List of processed document metadata
    """
    print("="*80)
    print("📄 PROCESSING PAPERS WITH DOCLING")
    print("="*80)
    print()
    
    processor = DoclingProcessor(
        texts_dir=settings.TEXTS_DIR,
        tables_dir=settings.TABLES_DIR,
        figures_dir=settings.FIGURES_DIR
    )
    
    documents = processor.process_batch(pdf_paths)
    
    print(f"✓ Processed {len(documents)} papers")
    print()
    
    return documents


def index_papers(documents: List[Dict], force_rebuild: bool = False):
    """
    Index papers into vector store and BM25
    
    Args:
        documents: List of processed document metadata
        force_rebuild: If True, rebuild index from scratch
    """
    print("="*80)
    print("🔍 INDEXING PAPERS")
    print("="*80)
    print()
    
    # Initialize components
    print("Initializing indexing components...")
    embedder = Embedder(settings.EMBEDDING_MODEL)
    
    vector_store = VectorStore(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding_dim=embedder.get_dimension(),
        persist_dir=settings.VECTOR_DB_DIR,
        distance=settings.QDRANT_DISTANCE
    )
    
    bm25_index = BM25Index()
    
    # Check existing data
    existing_count = vector_store.count()
    
    if existing_count > 0 and not force_rebuild:
        print(f"⚠️  Vector store already has {existing_count} documents")
        response = input("Rebuild index? (y/n): ")
        if response.lower() != 'y':
            print("Keeping existing index")
            return
        
        # Delete collection
        print("Deleting existing collection...")
        vector_store.delete_collection()
        
        # IMPORTANT: Cleanup old instance to release lock
        del vector_store
        import gc
        gc.collect()
        import time
        time.sleep(1)  # Give OS time to release file locks
        
        # Create new instance
        vector_store = VectorStore(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embedding_dim=embedder.get_dimension(),
            persist_dir=settings.VECTOR_DB_DIR,
            distance=settings.QDRANT_DISTANCE
        )
    
    print("✓ Components initialized")
    print()
    
    # Chunk documents
    print("Chunking documents...")
    chunker = SemanticChunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    
    all_chunks = []
    for doc in documents:
        text_file = Path(doc['text_file'])
        tables_file = Path(doc['tables_file']) if doc.get('tables_file') else None
        chunks = chunker.chunk_document(text_file, doc, tables_file)
        all_chunks.extend(chunks)
    
    print(f"✓ Created {len(all_chunks)} chunks")
    print()
    
    # Generate embeddings
    print("Generating embeddings...")
    texts = [chunk['text'] for chunk in all_chunks]
    embeddings = embedder.embed_batch(texts, batch_size=32, show_progress=True)
    print(f"✓ Generated {len(embeddings)} embeddings")
    print()
    
    # Build vector store
    print("Building vector store...")
    metadatas = [
        {
            "chunk_id": chunk['chunk_id'],
            "paper_id": chunk['paper_id'],
            "section": chunk['section'],
            "has_table": chunk.get('has_table', False)
        }
        for chunk in all_chunks
    ]
    
    ids = [chunk['chunk_id'] for chunk in all_chunks]
    vector_store.add_documents(texts, embeddings, metadatas, ids)
    print(f"✓ Vector store built: {vector_store.count()} documents")
    print()
    
    # Build BM25 index
    print("Building BM25 index...")
    bm25_index.build_index(texts, metadatas)
    
    # Save BM25 index
    bm25_path = settings.VECTOR_DB_DIR / "bm25_index.pkl"
    bm25_index.save(bm25_path)
    print(f"✓ BM25 index saved: {bm25_index.get_document_count()} documents")
    print()


def main():
    """Main pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fetch and index papers from arXiv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch default (15 papers, AI+ML+NLP, newest)
  python scripts/fetch_and_index_papers.py
  
  # Fetch more papers
  python scripts/fetch_and_index_papers.py --max-results 20
  
  # Custom query (transformer papers)
  python scripts/fetch_and_index_papers.py --query "all:transformer" --max-results 10
  
  # Fetch only (no indexing)
  python scripts/fetch_and_index_papers.py --fetch-only
  
  # Index existing papers (skip download)
  python scripts/fetch_and_index_papers.py --index-only
"""
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="arXiv search query (default: from config)"
    )
    
    parser.add_argument(
        "--max-results",
        type=int,
        help="Number of papers to fetch (default: from config)"
    )
    
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=["submitted", "relevance", "updated"],
        default="submitted",
        help="Sort papers by (default: submitted = newest)"
    )
    
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Only fetch papers, don't process or index"
    )
    
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only index existing papers, don't fetch new ones"
    )
    
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild index (delete existing)"
    )
    
    args = parser.parse_args()
    
    try:
        # Step 1: Fetch papers (unless index-only)
        if not args.index_only:
            pdf_paths = fetch_papers(
                query=args.query,
                max_results=args.max_results,
                sort_by=args.sort_by
            )
            
            if args.fetch_only:
                print("="*80)
                print("✓ FETCH COMPLETE")
                print("="*80)
                return
        else:
            # Get existing PDFs
            pdf_paths = list(settings.RAW_DIR.glob("*.pdf"))
            print(f"Found {len(pdf_paths)} existing PDFs")
            print()
        
        # Step 2: Process papers
        documents = process_papers(pdf_paths)
        
        # Step 3: Index papers
        index_papers(documents, force_rebuild=args.force_rebuild)
        
        # Summary
        print("="*80)
        print("✓ PIPELINE COMPLETE")
        print("="*80)
        print()
        print("Summary:")
        print(f"  • Papers processed: {len(documents)}")
        print(f"  • PDFs: {settings.RAW_DIR}")
        print(f"  • Texts: {settings.TEXTS_DIR}")
        print(f"  • Tables: {settings.TABLES_DIR}")
        print(f"  • Vector DB: {settings.VECTOR_DB_DIR}")
        print()
        print("Next steps:")
        print("  1. Test retrieval: python tests/test_phase3_retrieval.py")
        print("  2. Test RAG: python tests/test_phase4_rag.py")
        print("  3. Launch UI: python demo_ui.py")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

