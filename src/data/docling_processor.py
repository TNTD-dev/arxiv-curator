"""
Docling PDF Processor for extracting structured content from research papers

This module uses Docling to extract text, tables, figures, and document structure
from PDF files, which is especially valuable for research papers.
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
from loguru import logger

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("Docling not available. Install with: pip install docling")

# Fallback to PyMuPDF if Docling not available
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class DoclingProcessor:
    """
    Process PDFs using Docling to extract structured content including
    text, tables, figures, and document hierarchy
    """
    
    def __init__(
        self,
        texts_dir: Path,
        tables_dir: Path,
        figures_dir: Path,
        use_ocr: bool = False
    ):
        """
        Initialize Docling processor
        
        Args:
            texts_dir: Directory to save extracted text
            tables_dir: Directory to save extracted tables
            figures_dir: Directory to save figure metadata
            use_ocr: Enable OCR for scanned PDFs (slower)
        """
        self.texts_dir = Path(texts_dir)
        self.tables_dir = Path(tables_dir)
        self.figures_dir = Path(figures_dir)
        
        # Create directories
        for dir_path in [self.texts_dir, self.tables_dir, self.figures_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        if DOCLING_AVAILABLE:
            try:
                # Initialize Docling converter with simplified API (v2.0+)
                # The new API is much simpler - no pipeline options needed
                self.converter = DocumentConverter()
                self.use_ocr = use_ocr
                logger.info("✓ DoclingProcessor initialized with Docling (simplified API)")
            except Exception as e:
                logger.warning(f"Could not initialize Docling: {e}")
                logger.warning("Falling back to PyMuPDF")
                self.converter = None
        else:
            self.converter = None
            logger.warning("DoclingProcessor initialized with PyMuPDF fallback")
    
    def process_pdf(self, pdf_path: Path) -> Dict:
        """
        Extract structured content from PDF
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Dictionary with metadata and paths to extracted content
        """
        if DOCLING_AVAILABLE and self.converter:
            return self._process_with_docling(pdf_path)
        elif PYMUPDF_AVAILABLE:
            return self._process_with_pymupdf(pdf_path)
        else:
            raise RuntimeError("No PDF processing library available. Install docling or PyMuPDF")
    
    def _process_with_docling(self, pdf_path: Path) -> Dict:
        """Process PDF using Docling (advanced extraction)"""
        logger.info(f"Processing {pdf_path.name} with Docling...")
        
        if not self.converter:
            logger.warning("Docling converter not available, using PyMuPDF")
            return self._process_with_pymupdf(pdf_path)
        
        try:
            # Convert document
            result = self.converter.convert(str(pdf_path))
            doc = result.document
            
            paper_id = pdf_path.stem
            
            # Extract text with structure (Markdown format)
            markdown_text = doc.export_to_markdown()
            text_file = self.texts_dir / f"{paper_id}.md"
            text_file.write_text(markdown_text, encoding="utf-8")
            
            # Extract tables
            tables = []
            if hasattr(doc, 'tables'):
                for table_idx, table in enumerate(doc.tables):
                    table_data = {
                        "table_id": f"{paper_id}_table_{table_idx}",
                        "caption": getattr(table, 'caption', ''),
                        "content": table.export_to_markdown() if hasattr(table, 'export_to_markdown') else str(table),
                        "page": getattr(table, 'page', None)
                    }
                    tables.append(table_data)
            
            if tables:
                tables_file = self.tables_dir / f"{paper_id}_tables.json"
                tables_file.write_text(json.dumps(tables, indent=2), encoding="utf-8")
            
            # Extract figures/images metadata
            figures = []
            if hasattr(doc, 'pictures'):
                for fig_idx, figure in enumerate(doc.pictures):
                    fig_data = {
                        "figure_id": f"{paper_id}_fig_{fig_idx}",
                        "caption": getattr(figure, 'caption', ''),
                        "page": getattr(figure, 'page', None),
                        "bbox": getattr(figure, 'bbox', None)
                    }
                    figures.append(fig_data)
            
            if figures:
                figures_file = self.figures_dir / f"{paper_id}_figures.json"
                figures_file.write_text(json.dumps(figures, indent=2), encoding="utf-8")
            
            # Extract section information
            sections = self._extract_sections(doc)
            
            # Build metadata
            metadata = {
                "paper_id": paper_id,
                "source_pdf": str(pdf_path),
                "num_pages": len(doc.pages) if hasattr(doc, 'pages') else 0,
                "num_tables": len(tables),
                "num_figures": len(figures),
                "sections": sections,
                "text_file": str(text_file),
                "tables_file": str(tables_file) if tables else None,
                "figures_file": str(figures_file) if figures else None,
                "processor": "docling"
            }
            
            logger.success(
                f"✓ Processed {paper_id}: "
                f"{metadata['num_pages']} pages, "
                f"{metadata['num_tables']} tables, "
                f"{metadata['num_figures']} figures"
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name} with Docling: {e}")
            # Fallback to PyMuPDF
            if PYMUPDF_AVAILABLE:
                logger.info("Falling back to PyMuPDF...")
                return self._process_with_pymupdf(pdf_path)
            raise
    
    def _process_with_pymupdf(self, pdf_path: Path) -> Dict:
        """Process PDF using PyMuPDF (fallback, simpler extraction)"""
        logger.info(f"Processing {pdf_path.name} with PyMuPDF...")
        
        try:
            doc = fitz.open(pdf_path)
            paper_id = pdf_path.stem
            
            # Extract text
            full_text = []
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                full_text.append(f"# Page {page_num}\n\n{text}")
            
            text_content = "\n\n".join(full_text)
            text_file = self.texts_dir / f"{paper_id}.txt"
            text_file.write_text(text_content, encoding="utf-8")
            
            # Build metadata (simplified)
            metadata = {
                "paper_id": paper_id,
                "source_pdf": str(pdf_path),
                "num_pages": len(doc),
                "num_tables": 0,  # PyMuPDF doesn't extract tables easily
                "num_figures": 0,
                "sections": [],
                "text_file": str(text_file),
                "tables_file": None,
                "figures_file": None,
                "processor": "pymupdf"
            }
            
            doc.close()
            
            logger.success(f"✓ Processed {paper_id} with PyMuPDF: {metadata['num_pages']} pages")
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name} with PyMuPDF: {e}")
            raise
    
    def _extract_sections(self, doc) -> List[str]:
        """Extract section headers from Docling document"""
        sections = []
        try:
            if hasattr(doc, 'texts'):
                for item in doc.texts:
                    if hasattr(item, 'label') and 'heading' in str(item.label).lower():
                        sections.append(item.text)
        except Exception as e:
            logger.warning(f"Could not extract sections: {e}")
        return sections
    
    def process_batch(self, pdf_paths: List[Path]) -> List[Dict]:
        """
        Process multiple PDFs
        
        Args:
            pdf_paths: List of PDF file paths
        
        Returns:
            List of metadata dictionaries
        """
        results = []
        
        for pdf_path in pdf_paths:
            try:
                metadata = self.process_pdf(pdf_path)
                results.append(metadata)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                continue
        
        logger.info(f"Processed {len(results)}/{len(pdf_paths)} PDFs successfully")
        return results


def main():
    """Example usage"""
    from src.config import settings
    
    # Initialize processor
    processor = DoclingProcessor(
        texts_dir=settings.TEXTS_DIR,
        tables_dir=settings.TABLES_DIR,
        figures_dir=settings.FIGURES_DIR
    )
    
    # Process all PDFs in raw directory
    pdf_files = list(settings.RAW_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in raw directory.")
        print(f"Download papers first using arxiv_client.py")
        return
    
    print(f"\nProcessing {len(pdf_files)} PDF files...")
    results = processor.process_batch(pdf_files)
    
    # Display results
    print("\n" + "="*70)
    print("Processing Results:")
    print("="*70)
    for result in results:
        print(f"\n✓ {result['paper_id']}")
        print(f"  Pages: {result['num_pages']}")
        print(f"  Tables: {result['num_tables']}")
        print(f"  Figures: {result['num_figures']}")
        print(f"  Processor: {result['processor']}")
        print(f"  Text file: {result['text_file']}")


if __name__ == "__main__":
    main()

