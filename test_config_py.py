"""
Test configuration for the Financial RAG System
"""

import pytest
import os
import tempfile
from config import FinRAGConfig

class TestFinRAGConfig:
    """Test configuration management"""
    
    def test_get_config(self):
        """Test getting default configuration"""
        config = FinRAGConfig.get_config()
        
        # Check required keys exist
        required_keys = ['chunk_size', 'overlap', 'finbert_model', 'mpnet_model']
        for key in required_keys:
            assert key in config
        
        # Check default values
        assert config['chunk_size'] == 512
        assert config['overlap'] == 50
        assert 'finbert' in config['finbert_model'].lower()
        assert 'mpnet' in config['mpnet_model'].lower()
    
    def test_validate_environment(self):
        """Test environment validation"""
        # Set mock environment variable
        os.environ['PINECONE_API_KEY'] = 'test_key'
        
        # Should return True when API key is set
        assert FinRAGConfig.validate_environment() == True
        
        # Clean up
        del os.environ['PINECONE_API_KEY']
    
    def test_config_values(self):
        """Test configuration value ranges"""
        config = FinRAGConfig.get_config()
        
        # Check reasonable ranges
        assert 100 <= config['chunk_size'] <= 2048
        assert 0 <= config['overlap'] <= 200
        assert config['fine_tune_epochs'] >= 1
        assert config['fine_tune_batch_size'] >= 1

class TestConfigIntegration:
    """Integration tests for configuration"""
    
    def test_config_in_system(self):
        """Test config integration with system components"""
        from document_processor import FinancialDocumentProcessor
        
        config = FinRAGConfig.get_config()
        
        # Test document processor uses config
        processor = FinancialDocumentProcessor(
            chunk_size=config['chunk_size'],
            overlap=config['overlap']
        )
        
        assert processor.chunk_size == config['chunk_size']
        assert processor.overlap == config['overlap']
    
    def test_config_validation(self):
        """Test configuration validation"""
        from utils import validate_config
        
        # Valid config
        valid_config = FinRAGConfig.get_config()
        assert validate_config(valid_config) == True
        
        # Invalid config (missing key)
        invalid_config = {'chunk_size': 512}
        assert validate_config(invalid_config) == False
        
        # Invalid config (bad value)
        invalid_config = FinRAGConfig.get_config().copy()
        invalid_config['chunk_size'] = 50  # Too small
        assert validate_config(invalid_config) == False

if __name__ == "__main__":
    pytest.main([__file__])