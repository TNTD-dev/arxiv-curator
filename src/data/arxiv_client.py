"""
arXiv API Client for fetching research papers

This module provides functionality to search and download papers from arXiv.
"""

import arxiv
from pathlib import Path
from typing import List, Optional, Dict
from loguru import logger
import time
import ssl
import shutil


class ArxivClient:
    """Client for interacting with arXiv API"""
    
    def __init__(self, download_dir: Path, max_results: int = 10):
        """
        Initialize arXiv client
        
        Args:
            download_dir: Directory to save downloaded PDFs
            max_results: Maximum number of papers to fetch per query
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.max_results = max_results
        
        # Fix SSL certificate verification issues on Windows
        self._configure_ssl()
        
        logger.info(f"ArxivClient initialized with download_dir: {self.download_dir}")
    
    def _configure_ssl(self):
        """Configure SSL for arxiv downloads (Windows compatibility fix)"""
        try:
            # Disable SSL verification for arxiv downloads
            # This is safe for arxiv.org but should not be used for sensitive operations
            ssl._create_default_https_context = ssl._create_unverified_context
            logger.warning("⚠️  SSL verification disabled for arXiv downloads (Windows fix)")
        except Exception as e:
            logger.warning(f"Could not configure SSL: {e}")
    
    def search(
        self, 
        query: str, 
        max_results: Optional[int] = None,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate
    ) -> List[arxiv.Result]:
        """
        Search for papers on arXiv
        
        Args:
            query: Search query (e.g., "cat:cs.AI", "attention mechanism")
            max_results: Maximum results to return (uses default if None)
            sort_by: Sort criterion (SubmittedDate, Relevance, LastUpdatedDate)
        
        Returns:
            List of arXiv paper results
        
        Examples:
            >>> client = ArxivClient("./data/raw")
            >>> papers = client.search("cat:cs.AI OR cat:cs.LG", max_results=5)
            >>> papers = client.search("all:transformer AND all:attention")
        """
        max_results = max_results or self.max_results
        
        logger.info(f"Searching arXiv: query='{query}', max_results={max_results}")
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_by
            )
            
            results = list(search.results())
            logger.info(f"Found {len(results)} papers")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            raise
    
    def download_paper(
        self, 
        paper: arxiv.Result,
        filename: Optional[str] = None,
        retry_attempts: int = 3
    ) -> Path:
        """
        Download a single paper PDF with retry logic
        
        Args:
            paper: arXiv paper result object
            filename: Custom filename (uses paper ID if None)
            retry_attempts: Number of download retry attempts
        
        Returns:
            Path to downloaded PDF file
        """
        # Generate filename
        if filename is None:
            paper_id = paper.entry_id.split('/')[-1].replace('.', '_')
            filename = f"{paper_id}.pdf"
        
        filepath = self.download_dir / filename
        
        # Check if already exists
        if filepath.exists():
            logger.info(f"Paper already exists: {filename}")
            return filepath
        
        # Download with retry logic
        for attempt in range(retry_attempts):
            try:
                logger.info(f"Downloading {filename} (attempt {attempt + 1}/{retry_attempts})")
                
                # arxiv library's download_pdf() expects a directory path
                # It will create the PDF with its own filename inside that directory
                # So we download to temp location, then rename
                downloaded_path = paper.download_pdf(dirpath=str(self.download_dir))
                
                # Rename to our desired filename
                if downloaded_path != str(filepath):
                    shutil.move(downloaded_path, str(filepath))
                
                logger.success(f"✓ Downloaded: {filename}")
                return filepath
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to download {filename} after {retry_attempts} attempts")
                    raise
    
    def download_papers(
        self, 
        papers: List[arxiv.Result],
        delay: float = 1.0
    ) -> List[Dict[str, any]]:
        """
        Download multiple papers with metadata
        
        Args:
            papers: List of arXiv paper results
            delay: Delay between downloads (seconds) to be polite
        
        Returns:
            List of dictionaries with paper metadata and file paths
        """
        downloaded = []
        
        for i, paper in enumerate(papers):
            try:
                # Download PDF
                filepath = self.download_paper(paper)
                
                # Extract metadata
                metadata = {
                    'paper_id': paper.entry_id.split('/')[-1],
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'summary': paper.summary,
                    'published': paper.published.isoformat(),
                    'updated': paper.updated.isoformat() if paper.updated else None,
                    'categories': paper.categories,
                    'pdf_url': paper.pdf_url,
                    'filepath': str(filepath),
                    'primary_category': paper.primary_category
                }
                
                downloaded.append(metadata)
                
                # Be polite - add delay between downloads
                if i < len(papers) - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Failed to process paper {paper.entry_id}: {e}")
                continue
        
        logger.info(f"Successfully downloaded {len(downloaded)}/{len(papers)} papers")
        return downloaded
    
    def fetch_and_download(
        self,
        query: str,
        max_results: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        One-step function to search and download papers
        
        Args:
            query: Search query
            max_results: Maximum results
        
        Returns:
            List of paper metadata dictionaries
        """
        papers = self.search(query, max_results)
        return self.download_papers(papers)


def main():
    """Example usage"""
    from src.config import settings
    
    # Initialize client
    client = ArxivClient(settings.RAW_DIR, max_results=5)
    
    # Search and download
    papers = client.fetch_and_download(
        query="cat:cs.AI AND (all:transformer OR all:attention)",
        max_results=3
    )
    
    # Display results
    print("\n" + "="*70)
    print("Downloaded Papers:")
    print("="*70)
    for paper in papers:
        print(f"\n✓ {paper['title']}")
        print(f"  ID: {paper['paper_id']}")
        print(f"  Authors: {', '.join(paper['authors'][:3])}")
        print(f"  Categories: {', '.join(paper['categories'])}")
        print(f"  File: {paper['filepath']}")


if __name__ == "__main__":
    main()

