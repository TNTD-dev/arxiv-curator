"""
Semantic Chunking Module for Research Papers

This module provides intelligent chunking strategies that preserve document structure,
sections, and context for better retrieval performance.
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
import re
from loguru import logger


class SemanticChunker:
    """
    Chunk documents while preserving structure and context
    Optimized for research papers with sections, tables, and citations
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        """
        Initialize semantic chunker
        
        Args:
            chunk_size: Target size for each chunk (in characters)
            chunk_overlap: Overlap between chunks for context continuity
            min_chunk_size: Minimum chunk size (avoid tiny chunks)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        logger.info(
            f"SemanticChunker initialized: "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )
    
    def chunk_document(
        self,
        text_file: Path,
        metadata: Dict,
        tables_file: Optional[Path] = None
    ) -> List[Dict]:
        """
        Chunk a document while preserving structure
        
        Args:
            text_file: Path to extracted text file
            metadata: Document metadata from processor
            tables_file: Optional path to tables JSON file
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Read text
        text = text_file.read_text(encoding="utf-8")
        
        # Load tables if available
        tables = []
        if tables_file and tables_file.exists():
            try:
                tables = json.loads(tables_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Could not load tables: {e}")
        
        # Detect if it's markdown or plain text
        is_markdown = text_file.suffix == ".md"
        
        if is_markdown:
            chunks = self._chunk_markdown(text, metadata, tables)
        else:
            chunks = self._chunk_plain_text(text, metadata, tables)
        
        logger.info(f"Created {len(chunks)} chunks for {metadata['paper_id']}")
        return chunks
    
    def _chunk_markdown(
        self,
        text: str,
        metadata: Dict,
        tables: List[Dict]
    ) -> List[Dict]:
        """Chunk markdown text respecting headers and structure"""
        chunks = []
        lines = text.split('\n')
        
        current_section = "Introduction"
        current_chunk_lines = []
        current_length = 0
        chunk_id = 0
        
        for line in lines:
            # Detect section headers (# Header, ## Subheader, etc.)
            if line.strip().startswith('#'):
                # Save current chunk if it exists
                if current_chunk_lines and current_length >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        current_chunk_lines,
                        metadata,
                        current_section,
                        tables,
                        chunk_id
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    
                    # Keep overlap
                    overlap_lines = self._get_overlap_lines(current_chunk_lines)
                    current_chunk_lines = overlap_lines
                    current_length = sum(len(l) for l in overlap_lines)
                
                # Update section
                current_section = line.strip('#').strip()
                current_chunk_lines.append(line)
                current_length += len(line)
                continue
            
            # Add line to current chunk
            current_chunk_lines.append(line)
            current_length += len(line)
            
            # Split if chunk too large
            if current_length >= self.chunk_size:
                chunk = self._create_chunk(
                    current_chunk_lines,
                    metadata,
                    current_section,
                    tables,
                    chunk_id
                )
                chunks.append(chunk)
                chunk_id += 1
                
                # Keep overlap
                overlap_lines = self._get_overlap_lines(current_chunk_lines)
                current_chunk_lines = overlap_lines
                current_length = sum(len(l) for l in overlap_lines)
        
        # Add remaining chunk
        if current_chunk_lines and current_length >= self.min_chunk_size:
            chunk = self._create_chunk(
                current_chunk_lines,
                metadata,
                current_section,
                tables,
                chunk_id
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_plain_text(
        self,
        text: str,
        metadata: Dict,
        tables: List[Dict]
    ) -> List[Dict]:
        """Chunk plain text with paragraph detection"""
        chunks = []
        
        # Split by double newline (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = []
        current_length = 0
        current_section = "Content"
        chunk_id = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Try to detect section headers
            if self._is_section_header(para):
                # Save current chunk
                if current_chunk and current_length >= self.min_chunk_size:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk = self._create_chunk_dict(
                        chunk_text,
                        metadata,
                        current_section,
                        tables,
                        chunk_id
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    current_chunk = []
                    current_length = 0
                
                current_section = para
            
            # Add paragraph to chunk
            current_chunk.append(para)
            current_length += len(para)
            
            # Split if too large
            if current_length >= self.chunk_size:
                chunk_text = '\n\n'.join(current_chunk)
                chunk = self._create_chunk_dict(
                    chunk_text,
                    metadata,
                    current_section,
                    tables,
                    chunk_id
                )
                chunks.append(chunk)
                chunk_id += 1
                
                # Keep last paragraph for overlap
                if len(current_chunk) > 1:
                    current_chunk = [current_chunk[-1]]
                    current_length = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_length = 0
        
        # Add remaining
        if current_chunk and current_length >= self.min_chunk_size:
            chunk_text = '\n\n'.join(current_chunk)
            chunk = self._create_chunk_dict(
                chunk_text,
                metadata,
                current_section,
                tables,
                chunk_id
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(
        self,
        lines: List[str],
        metadata: Dict,
        section: str,
        tables: List[Dict],
        chunk_id: int
    ) -> Dict:
        """Create chunk dictionary from lines"""
        text = '\n'.join(lines)
        return self._create_chunk_dict(text, metadata, section, tables, chunk_id)
    
    def _create_chunk_dict(
        self,
        text: str,
        metadata: Dict,
        section: str,
        tables: List[Dict],
        chunk_id: int
    ) -> Dict:
        """Create chunk dictionary with metadata"""
        # Check if chunk references tables
        has_table = False
        table_refs = []
        for table in tables:
            if table['table_id'] in text or any(
                keyword in text.lower() 
                for keyword in ['table', 'figure', 'fig.']
            ):
                has_table = True
                table_refs.append(table['table_id'])
        
        return {
            "chunk_id": f"{metadata['paper_id']}_chunk_{chunk_id}",
            "text": text.strip(),
            "paper_id": metadata['paper_id'],
            "section": section,
            "has_table": has_table,
            "table_refs": table_refs,
            "num_tables": metadata.get('num_tables', 0),
            "num_figures": metadata.get('num_figures', 0),
            "processor": metadata.get('processor', 'unknown'),
            "char_count": len(text)
        }
    
    def _get_overlap_lines(self, lines: List[str]) -> List[str]:
        """Get lines for overlap based on chunk_overlap setting"""
        if not lines:
            return []
        
        overlap_chars = 0
        overlap_lines = []
        
        for line in reversed(lines):
            overlap_chars += len(line)
            overlap_lines.insert(0, line)
            if overlap_chars >= self.chunk_overlap:
                break
        
        return overlap_lines
    
    def _is_section_header(self, text: str) -> bool:
        """Detect if text is likely a section header"""
        # Common section patterns in research papers
        section_patterns = [
            r'^(abstract|introduction|background|related work|methodology|methods)',
            r'^(experiments|results|discussion|conclusion|references)',
            r'^(\d+\.?\s+[A-Z])',  # "1. Introduction" or "1 Introduction"
            r'^([A-Z][A-Z\s]{2,30}$)',  # ALL CAPS short titles
        ]
        
        text_lower = text.lower().strip()
        
        for pattern in section_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Short text in title case
        if len(text) < 100 and text.istitle():
            return True
        
        return False
    
    def chunk_batch(
        self,
        documents: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of document metadata from processor
        
        Returns:
            Dictionary mapping paper_id to list of chunks
        """
        all_chunks = {}
        
        for doc in documents:
            try:
                text_file = Path(doc['text_file'])
                tables_file = Path(doc['tables_file']) if doc.get('tables_file') else None
                
                chunks = self.chunk_document(text_file, doc, tables_file)
                all_chunks[doc['paper_id']] = chunks
                
            except Exception as e:
                logger.error(f"Error chunking {doc['paper_id']}: {e}")
                continue
        
        total_chunks = sum(len(chunks) for chunks in all_chunks.values())
        logger.info(f"Created {total_chunks} total chunks from {len(documents)} documents")
        
        return all_chunks


def main():
    """Example usage"""
    from src.config import settings
    from src.data.docling_processor import DoclingProcessor
    
    # Process PDFs first
    processor = DoclingProcessor(
        texts_dir=settings.TEXTS_DIR,
        tables_dir=settings.TABLES_DIR,
        figures_dir=settings.FIGURES_DIR
    )
    
    pdf_files = list(settings.RAW_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found. Run arxiv_client.py first.")
        return
    
    print(f"Processing {len(pdf_files)} PDFs...")
    documents = processor.process_batch(pdf_files)
    
    # Chunk documents
    chunker = SemanticChunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    
    print("\nChunking documents...")
    all_chunks = chunker.chunk_batch(documents)
    
    # Display results
    print("\n" + "="*70)
    print("Chunking Results:")
    print("="*70)
    for paper_id, chunks in all_chunks.items():
        print(f"\n📄 {paper_id}: {len(chunks)} chunks")
        
        # Show first chunk as example
        if chunks:
            first_chunk = chunks[0]
            print(f"   Section: {first_chunk['section']}")
            print(f"   Characters: {first_chunk['char_count']}")
            print(f"   Preview: {first_chunk['text'][:150]}...")


if __name__ == "__main__":
    main()

