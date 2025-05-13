"""
Document processing module for the Financial RAG System
Handles extraction and chunking of various document formats (PDF, Word, Text)
"""

import os
import logging
from typing import List, Dict
from datetime import datetime

import PyPDF2
import docx
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)

class FinancialDocumentProcessor:
    """
    Handles the preprocessing of financial documents (PDFs, Word docs, text files)
    Extracts text and splits it into manageable chunks for embedding generation
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize document processor with chunking parameters
        
        Args:
            chunk_size: Maximum characters per chunk
            overlap: Character overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        logger.info(f"Initialized document processor with chunk_size={chunk_size}, overlap={overlap}")
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text content from PDF files
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                # Extract text from each page
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """
        Extract text content from Word documents
        
        Args:
            file_path: Path to Word document
            
        Returns:
            Extracted text content
        """
        try:
            doc = docx.Document(file_path)
            # Extract text from all paragraphs
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {str(e)}")
            return ""
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Determine file type and extract text accordingly
        
        Args:
            file_path: Path to document file
            
        Returns:
            Extracted text content
        """
        if file_path.endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            return self.extract_text_from_docx(file_path)
        elif file_path.endswith('.txt'):
            # Simple text file reading
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Error reading TXT {file_path}: {str(e)}")
                return ""
        else:
            logger.warning(f"Unsupported file format: {file_path}")
            return ""
    
    def create_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for better context preservation
        This method ensures that we don't cut words in half at chunk boundaries
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position with overlap consideration
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end]
            
            # Ensure we don't cut words in half at chunk boundaries
            # Look for the last space before the boundary
            if end < text_length and not text[end].isspace():
                last_space = chunk.rfind(' ')
                if last_space > start:
                    end = start + last_space
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            # Move start position back by overlap amount for next chunk
            start = end - self.overlap
        
        return chunks
    
    def process_documents(self, document_paths: List[str]) -> List[Dict]:
        """
        Process multiple documents and return chunks with metadata
        
        Args:
            document_paths: List of paths to documents
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        all_chunks = []
        
        # Process each document with progress bar
        for doc_path in tqdm(document_paths, desc="Processing documents"):
            logger.info(f"Processing document: {doc_path}")
            
            # Extract text from document
            text = self.extract_text_from_file(doc_path)
            
            if text:
                # Split text into chunks
                chunks = self.create_chunks(text)
                
                # Add metadata to each chunk
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        'text': chunk,
                        'source': doc_path,
                        'chunk_id': i,
                        'timestamp': datetime.now().isoformat(),
                        'length': len(chunk)
                    })
        
        logger.info(f"Processed {len(document_paths)} documents, created {len(all_chunks)} chunks")
        return all_chunks
    
    def get_document_stats(self, document_paths: List[str]) -> Dict:
        """
        Get statistics about the documents to be processed
        
        Args:
            document_paths: List of document paths
            
        Returns:
            Dictionary with document statistics
        """
        stats = {
            'total_documents': len(document_paths),
            'pdf_count': 0,
            'docx_count': 0,
            'txt_count': 0,
            'other_count': 0,
            'total_size_mb': 0
        }
        
        for path in document_paths:
            # Count file types
            if path.endswith('.pdf'):
                stats['pdf_count'] += 1
            elif path.endswith('.docx'):
                stats['docx_count'] += 1
            elif path.endswith('.txt'):
                stats['txt_count'] += 1
            else:
                stats['other_count'] += 1
            
            # Calculate total size
            if os.path.exists(path):
                stats['total_size_mb'] += os.path.getsize(path) / (1024 * 1024)
        
        return stats