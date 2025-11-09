"""
Phase 2 Integration Test - Data Ingestion Pipeline

Test the complete flow: arXiv fetch → PDF download → extraction → chunking
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.data.arxiv_client import ArxivClient
from src.data.docling_processor import DoclingProcessor
from src.data.chunker import SemanticChunker
from loguru import logger
import json


def test_phase2_pipeline(num_papers: int = 3):
    """
    Run complete Phase 2 pipeline test
    
    Args:
        num_papers: Number of papers to test with
    """
    print("="*80)
    print("PHASE 2 INTEGRATION TEST - DATA INGESTION PIPELINE")
    print("="*80)
    print()
    
    # Step 1: Fetch and download papers
    print("Step 1: Fetching papers from arXiv...")
    print("-"*80)
    
    client = ArxivClient(settings.RAW_DIR, max_results=num_papers)
    
    # Use a focused query for AI/ML papers
    query = "cat:cs.AI AND (all:transformer OR all:attention)"
    
    try:
        papers = client.fetch_and_download(query, max_results=num_papers)
        
        if not papers:
            print("✗ Failed to download any papers (check SSL/network)")
            print()
            print("This is usually caused by SSL certificate issues on Windows.")
            print("The ArxivClient should have automatically configured SSL bypass.")
            print()
            return False
        
        print(f"✓ Successfully downloaded {len(papers)} papers")
        
        for i, paper in enumerate(papers, 1):
            print(f"  {i}. {paper['title'][:70]}...")
            print(f"     ID: {paper['paper_id']}")
        
    except Exception as e:
        logger.error(f"Failed to fetch papers: {e}")
        return False
    
    print()
    
    # Step 2: Extract content from PDFs
    print("Step 2: Extracting content from PDFs...")
    print("-"*80)
    
    processor = DoclingProcessor(
        texts_dir=settings.TEXTS_DIR,
        tables_dir=settings.TABLES_DIR,
        figures_dir=settings.FIGURES_DIR,
        use_ocr=False
    )
    
    pdf_paths = [Path(paper['filepath']) for paper in papers]
    
    try:
        documents = processor.process_batch(pdf_paths)
        print(f"✓ Successfully processed {len(documents)} PDFs")
        
        for doc in documents:
            print(f"  - {doc['paper_id']}")
            print(f"    Pages: {doc['num_pages']}, "
                  f"Tables: {doc['num_tables']}, "
                  f"Figures: {doc['num_figures']}")
            print(f"    Processor: {doc['processor']}")
        
    except Exception as e:
        logger.error(f"Failed to process PDFs: {e}")
        return False
    
    print()
    
    # Step 3: Chunk documents
    print("Step 3: Chunking documents...")
    print("-"*80)
    
    chunker = SemanticChunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    
    try:
        all_chunks = chunker.chunk_batch(documents)
        total_chunks = sum(len(chunks) for chunks in all_chunks.values())
        print(f"✓ Successfully created {total_chunks} chunks from {len(documents)} documents")
        
        for paper_id, chunks in all_chunks.items():
            print(f"  - {paper_id}: {len(chunks)} chunks")
            
            # Show statistics
            sections = set(chunk['section'] for chunk in chunks)
            chunks_with_tables = sum(1 for chunk in chunks if chunk['has_table'])
            avg_chars = sum(chunk['char_count'] for chunk in chunks) / len(chunks)
            
            print(f"    Sections: {len(sections)}")
            print(f"    Chunks with tables: {chunks_with_tables}")
            print(f"    Avg chunk size: {avg_chars:.0f} chars")
        
    except Exception as e:
        logger.error(f"Failed to chunk documents: {e}")
        return False
    
    print()
    
    # Step 4: Save results summary
    print("Step 4: Saving results...")
    print("-"*80)
    
    try:
        results_file = settings.DATA_DIR / "phase2_test_results.json"
        
        results = {
            "num_papers": len(papers),
            "num_documents_processed": len(documents),
            "total_chunks": total_chunks,
            "papers": [
                {
                    "paper_id": paper['paper_id'],
                    "title": paper['title'],
                    "num_chunks": len(all_chunks.get(paper['paper_id'], []))
                }
                for paper in papers
            ],
            "chunking_stats": {
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "avg_chunks_per_paper": round(total_chunks / len(documents), 2) if documents else 0
            }
        }
        
        results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"✓ Results saved to: {results_file}")
        
    except Exception as e:
        logger.warning(f"Could not save results: {e}")
    
    print()
    print("="*80)
    print("PHASE 2 TEST COMPLETED SUCCESSFULLY! ✓")
    print("="*80)
    print()
    print("Summary:")
    print(f"  • Papers downloaded: {len(papers)}")
    print(f"  • PDFs processed: {len(documents)}")
    print(f"  • Total chunks created: {total_chunks}")
    avg_chunks = total_chunks / len(documents) if documents else 0
    print(f"  • Average chunks per paper: {avg_chunks:.1f}")
    print()
    print("Next steps:")
    print("  1. Verify extracted content in data/processed/texts/")
    print("  2. Check tables in data/processed/tables/")
    print("  3. Ready to proceed to Phase 3 (Vector Store & Retrieval)")
    print()
    
    return True


def test_individual_components():
    """Test each component individually"""
    print("\n" + "="*80)
    print("INDIVIDUAL COMPONENT TESTS")
    print("="*80 + "\n")
    
    # Test 1: ArxivClient
    print("Test 1: ArxivClient")
    print("-"*80)
    try:
        client = ArxivClient(settings.RAW_DIR, max_results=1)
        papers = client.search("cat:cs.AI", max_results=1)
        print(f"✓ ArxivClient working: found {len(papers)} paper(s)")
    except Exception as e:
        print(f"✗ ArxivClient failed: {e}")
    
    print()
    
    # Test 2: DoclingProcessor
    print("Test 2: DoclingProcessor")
    print("-"*80)
    try:
        processor = DoclingProcessor(
            texts_dir=settings.TEXTS_DIR,
            tables_dir=settings.TABLES_DIR,
            figures_dir=settings.FIGURES_DIR
        )
        print(f"✓ DoclingProcessor initialized successfully")
        
        # Check if any PDFs exist to process
        pdf_files = list(settings.RAW_DIR.glob("*.pdf"))
        if pdf_files:
            print(f"  Found {len(pdf_files)} PDF(s) to process")
        else:
            print(f"  No PDFs found in {settings.RAW_DIR}")
    except Exception as e:
        print(f"✗ DoclingProcessor failed: {e}")
    
    print()
    
    # Test 3: SemanticChunker
    print("Test 3: SemanticChunker")
    print("-"*80)
    try:
        chunker = SemanticChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        print(f"✓ SemanticChunker initialized successfully")
        print(f"  Chunk size: {settings.CHUNK_SIZE}")
        print(f"  Overlap: {settings.CHUNK_OVERLAP}")
    except Exception as e:
        print(f"✗ SemanticChunker failed: {e}")
    
    print()


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Phase 2 Data Ingestion Pipeline")
    parser.add_argument(
        "--num-papers",
        type=int,
        default=3,
        help="Number of papers to test with (default: 3)"
    )
    parser.add_argument(
        "--components-only",
        action="store_true",
        help="Only test individual components, skip full pipeline"
    )
    
    args = parser.parse_args()
    
    if args.components_only:
        test_individual_components()
    else:
        success = test_phase2_pipeline(num_papers=args.num_papers)
        
        if not success:
            print("\n⚠️  Some tests failed. Check logs for details.")
            sys.exit(1)


if __name__ == "__main__":
    main()

