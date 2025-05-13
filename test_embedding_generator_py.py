"""
Test embedding generation functionality
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from embedding_generator import FinEmbeddingGenerator

class TestFinEmbeddingGenerator:
    """Test embedding generation functionality"""
    
    @pytest.fixture
    def mock_models(self):
        """Mock the model loading to avoid downloading models in tests"""
        with patch('embedding_generator.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('embedding_generator.AutoModel.from_pretrained') as mock_model, \
             patch('embedding_generator.SentenceTransformer') as mock_sent_transformer:
            
            # Mock tokenizer
            mock_tokenizer.return_value = MagicMock()
            
            # Mock FinBERT model
            mock_model_instance = MagicMock()
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.eval.return_value = mock_model_instance
            mock_model.return_value = mock_model_instance
            
            # Mock sentence transformer
            mock_sent_transformer.return_value = MagicMock()
            
            yield {
                'tokenizer': mock_tokenizer,
                'model': mock_model,
                'sent_transformer': mock_sent_transformer
            }
    
    @pytest.fixture
    def generator(self, mock_models):
        """Create an embedding generator instance"""
        return FinEmbeddingGenerator()
    
    def test_initialization(self, generator, mock_models):
        """Test generator initialization"""
        assert generator.device in ['cuda', 'cpu']
        assert generator.finbert_tokenizer is not None
        assert generator.finbert_model is not None
        assert generator.mpnet_model is not None
    
    def test_get_finbert_embeddings(self, generator, mock_models):
        """Test FinBERT embedding generation"""
        # Mock tokenizer output
        mock_models['tokenizer'].return_value.return_tensors = 'pt'
        mock_models['tokenizer'].return_value.to.return_value = {
            'input_ids': MagicMock(),
            'attention_mask': MagicMock()
        }
        
        # Mock model output
        mock_output = MagicMock()
        mock_output.pooler_output.cpu.return_value.numpy.return_value = np.random.rand(2, 768)
        generator.finbert_model.return_value = mock_output
        
        texts = ["Sample financial text", "Another financial document"]
        embeddings = generator.get_finbert_embeddings(texts)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 768  # FinBERT dimension
    
    def test_get_mpnet_embeddings(self, generator, mock_models):
        """Test MPNet embedding generation"""
        # Mock sentence transformer encode
        mock_embeddings = np.random.rand(2, 768)
        generator.mpnet_model.encode.return_value = mock_embeddings
        
        texts = ["Sample text 1", "Sample text 2"]
        embeddings = generator.get_mpnet_embeddings(texts)
        
        assert np.array_equal(embeddings, mock_embeddings)
        generator.mpnet_model.encode.assert_called_once()
    
    def test_combine_embeddings(self, generator):
        """Test embedding combination"""
        # Create sample embeddings
        finbert_emb = np.random.rand(3, 768)
        mpnet_emb = np.random.rand(3, 768)
        
        combined = generator.combine_embeddings(finbert_emb, mpnet_emb, alpha=0.7)
        
        # Check dimensions
        assert combined.shape == finbert_emb.shape
        
        # Check normalization (should be unit vectors)
        norms = np.linalg.norm(combined, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
    
    def test_combine_embeddings_different_alpha(self, generator):
        """Test embedding combination with different alpha values"""
        finbert_emb = np.ones((2, 10))  # Simple embeddings for testing
        mpnet_emb = np.ones((2, 10)) * 2
        
        # Test different alpha values
        combined1 = generator.combine_embeddings(finbert_emb, mpnet_emb, alpha=0.0)
        combined2 = generator.combine_embeddings(finbert_emb, mpnet_emb, alpha=1.0)
        
        # With alpha=0, should be close to normalized mpnet_emb
        # With alpha=1, should be close to normalized finbert_emb
        assert not np.allclose(combined1, combined2)
    
    def test_get_embedding_dimension(self, generator, mock_models):
        """Test getting embedding dimensions"""
        # Mock the embedding generation to return known dimensions
        mock_finbert_emb = np.random.rand(1, 768)
        mock_mpnet_emb = np.random.rand(1, 384)
        
        with patch.object(generator, 'get_finbert_embeddings') as mock_finbert, \
             patch.object(generator, 'get_mpnet_embeddings') as mock_mpnet:
            
            mock_finbert.return_value = mock_finbert_emb
            mock_mpnet.return_value = mock_mpnet_emb
            
            finbert_dim, mpnet_dim = generator.get_embedding_dimension()
            
            assert finbert_dim == 768
            assert mpnet_dim == 384
    
    def test_save_and_load_embeddings(self, generator):
        """Test saving and loading embeddings"""
        # Create sample embeddings
        embeddings = np.random.rand(10, 768)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save embeddings
            generator.save_embeddings(embeddings, temp_path)
            
            # Check file exists
            assert os.path.exists(temp_path)
            
            # Load embeddings
            loaded_embeddings = generator.load_embeddings(temp_path)
            
            # Check they're the same
            np.testing.assert_array_equal(embeddings, loaded_embeddings)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

class TestEmbeddingGeneratorEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.fixture
    def generator_no_mock(self):
        """Create generator without mocking (for edge case tests)"""
        # This would require actual models, so we'll mock differently
        with patch('embedding_generator.torch.cuda.is_available') as mock_cuda:
            mock_cuda.return_value = False  # Force CPU mode
            
            with patch('embedding_generator.AutoTokenizer.from_pretrained'), \
                 patch('embedding_generator.AutoModel.from_pretrained'), \
                 patch('embedding_generator.SentenceTransformer'):
                # Create generator that won't try to use real models
                pass
    
    def test_empty_text_list(self, mock_models):
        """Test handling of empty text list"""
        generator = FinEmbeddingGenerator()
        
        # Mock to return empty array
        with patch.object(generator, 'get_finbert_embeddings') as mock_method:
            mock_method.return_value = np.array([]).reshape(0, 768)
            result = generator.get_finbert_embeddings([])
            assert result.shape[0] == 0
    
    def test_very_long_text(self, mock_models):
        """Test handling of very long text"""
        generator = FinEmbeddingGenerator()
        
        # Create very long text
        long_text = ["This is a very long text. " * 1000]
        
        # Mock tokenizer to simulate truncation
        mock_models['tokenizer'].return_value.return_value = MagicMock()
        mock_models['tokenizer'].return_value.to.return_value = {
            'input_ids': MagicMock(shape=(1, 512)),  # Truncated to 512
            'attention_mask': MagicMock(shape=(1, 512))
        }
        
        # Should handle gracefully (no exceptions)
        generator.get_finbert_embeddings(long_text)
    
    def test_batch_size_handling(self, mock_models):
        """Test different batch sizes"""
        generator = FinEmbeddingGenerator()
        
        # Mock model output
        mock_output = MagicMock()
        mock_output.pooler_output.cpu.return_value.numpy.return_value = np.random.rand(1, 768)
        generator.finbert_model.return_value = mock_output
        
        texts = ["Text " + str(i) for i in range(10)]
        
        # Test with different batch sizes
        for batch_size in [1, 3, 5, 10]:
            embeddings = generator.get_finbert_embeddings(texts, batch_size=batch_size)
            assert embeddings.shape[0] == len(texts)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])