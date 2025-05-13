"""
Test document processing functionality
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from document_processor import FinancialDocumentProcessor

class TestFinancialDocumentProcessor:
    """Test document processing functionality"""
    
    @pytest.fixture
    def processor(self):
        """Create a document processor instance"""
        return FinancialDocumentProcessor(chunk_size=512, overlap=50)
    
    @pytest.fixture
    def temp_text_file(self):
        """Create a temporary text file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document with financial information. " * 20)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)
    
    def test_initialization(self, processor):
        """Test processor initialization"""
        assert processor.chunk_size == 512
        assert processor.overlap == 50
    
    def test_extract_text_from_txt(self, processor, temp_text_file):
        """Test text extraction from text files"""
        text = processor.extract_text_from_file(temp_text_file)
        assert len(text) > 0
        assert "financial information" in text
    
    def test_create_chunks(self, processor):
        """Test text chunking"""
        # Create long text
        text = "This is a test sentence. " * 100
        chunks = processor.create_chunks(text)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Check chunk sizes
        for chunk in chunks:
            assert len(chunk) <= processor.chunk_size
        
        # Check overlap between consecutive chunks
        if len(chunks) > 1:
            # Find common text between first two chunks
            chunk1_end = chunks[0][-processor.overlap:]
            chunk2_start = chunks[1][:processor.overlap]
            # There should be some overlap (not exact due to word boundaries)
            assert len(set(chunk1_end.split()) & set(chunk2_start.split())) > 0
    
    def test_process_documents(self, processor, temp_text_file):
        """Test processing multiple documents"""
        documents = [temp_text_file]
        chunks = processor.process_documents(documents)
        
        # Should create at least one chunk
        assert len(chunks) >= 1
        
        # Check chunk metadata
        for chunk in chunks:
            assert 'text' in chunk
            assert 'source' in chunk
            assert 'chunk_id' in chunk
            assert 'timestamp' in chunk
            assert chunk['source'] == temp_text_file
    
    def test_empty_file_handling(self, processor):
        """Test handling of empty files"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name
        
        try:
            chunks = processor.process_documents([temp_path])
            # Should handle empty files gracefully
            assert len(chunks) == 0
        finally:
            os.unlink(temp_path)
    
    def test_unsupported_file_type(self, processor):
        """Test handling of unsupported file types"""
        with tempfile.NamedTemporaryFile(suffix='.xyz') as f:
            text = processor.extract_text_from_file(f.name)
            assert text == ""
    
    @patch('PyPDF2.PdfReader')
    def test_pdf_extraction(self, mock_pdf_reader, processor):
        """Test PDF text extraction (mocked)"""
        # Mock PDF reader
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Test PDF content"
        mock_pdf_reader.return_value.pages = [mock_page, mock_page]
        
        with tempfile.NamedTemporaryFile(suffix='.pdf') as f:
            text = processor.extract_text_from_pdf(f.name)
            assert text == "Test PDF contentTest PDF content"
    
    @patch('docx.Document')
    def test_docx_extraction(self, mock_docx, processor):
        """Test Word document extraction (mocked)"""
        # Mock Word document
        mock_paragraph = MagicMock()
        mock_paragraph.text = "Test Word content"
        mock_docx.return_value.paragraphs = [mock_paragraph, mock_paragraph]
        
        with tempfile.NamedTemporaryFile(suffix='.docx') as f:
            text = processor.extract_text_from_docx(f.name)
            assert text == "Test Word content\nTest Word content"
    
    def test_get_document_stats(self, processor, temp_text_file):
        """Test document statistics"""
        documents = [temp_text_file]
        stats = processor.get_document_stats(documents)
        
        assert stats['total_documents'] == 1
        assert stats['txt_count'] == 1
        assert stats['pdf_count'] == 0
        assert stats['docx_count'] == 0
        assert stats['total_size_mb'] > 0

class TestDocumentProcessorEdgeCases:
    """Test edge cases and error handling"""
    
    def test_very_short_chunks(self):
        """Test handling of very short chunks"""
        processor = FinancialDocumentProcessor(chunk_size=10, overlap=5)
        text = "Short text"
        chunks = processor.create_chunks(text)
        
        assert len(chunks) >= 1
        assert all(len(chunk) <= 10 for chunk in chunks)
    
    def test_chunk_overlap_larger_than_size(self):
        """Test when overlap is larger than chunk size"""
        processor = FinancialDocumentProcessor(chunk_size=50, overlap=100)
        text = "This is a test " * 20
        chunks = processor.create_chunks(text)
        
        # Should still work, though overlap will be limited
        assert len(chunks) >= 1
    
    def test_nonexistent_file(self):
        """Test handling of non-existent files"""
        processor = FinancialDocumentProcessor()
        chunks = processor.process_documents(['nonexistent_file.txt'])
        
        # Should handle gracefully and return empty list
        assert len(chunks) == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])