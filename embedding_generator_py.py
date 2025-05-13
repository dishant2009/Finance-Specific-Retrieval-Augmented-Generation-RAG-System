"""
Embedding generation module using FinBERT and MPNet
Handles creation of domain-specific financial embeddings
"""

import logging
from typing import List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)

class FinEmbeddingGenerator:
    """
    Generates embeddings for financial text using FinBERT and sentence transformers
    Handles both local FinBERT processing and MPNet for sentence similarity
    """
    
    def __init__(self, finbert_model: str = "ProsusAI/finbert", 
                 mpnet_model: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize embedding generators
        
        Args:
            finbert_model: FinBERT model name or path
            mpnet_model: MPNet model name or path
        """
        # Determine device (GPU if available, otherwise CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Load FinBERT for domain-specific financial embeddings
        logger.info(f"Loading FinBERT model: {finbert_model}")
        self.finbert_tokenizer = AutoTokenizer.from_pretrained(finbert_model)
        self.finbert_model = AutoModel.from_pretrained(finbert_model).to(self.device)
        self.finbert_model.eval()  # Set to evaluation mode
        
        # Load MPNet for general sentence similarity tasks
        logger.info(f"Loading MPNet model: {mpnet_model}")
        self.mpnet_model = SentenceTransformer(mpnet_model, device=self.device)
        
        logger.info("Embedding generators initialized successfully")
    
    def get_finbert_embeddings(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Generate embeddings using FinBERT model for financial domain specificity
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            NumPy array of embeddings
        """
        all_embeddings = []
        
        # Process texts in batches to manage memory usage
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating FinBERT embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize the batch with padding and truncation
            encoded = self.finbert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,  # FinBERT max sequence length
                return_tensors='pt'
            ).to(self.device)
            
            # Generate embeddings without computing gradients (faster)
            with torch.no_grad():
                outputs = self.finbert_model(**encoded)
                # Use the pooled output (CLS token) as sentence embedding
                embeddings = outputs.pooler_output.cpu().numpy()
                all_embeddings.append(embeddings)
        
        # Concatenate all batch embeddings
        final_embeddings = np.vstack(all_embeddings)
        logger.info(f"Generated {final_embeddings.shape[0]} FinBERT embeddings")
        return final_embeddings
    
    def get_mpnet_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using MPNet for sentence similarity
        MPNet is particularly good at capturing semantic similarity
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings
        """
        logger.info("Generating MPNet embeddings for sentence similarity")
        embeddings = self.mpnet_model.encode(texts, show_progress_bar=True)
        logger.info(f"Generated {embeddings.shape[0]} MPNet embeddings")
        return embeddings
    
    def combine_embeddings(self, finbert_emb: np.ndarray, mpnet_emb: np.ndarray, 
                          alpha: float = 0.7) -> np.ndarray:
        """
        Combine FinBERT and MPNet embeddings for enhanced representation
        This hybrid approach leverages both domain-specific knowledge and general semantics
        
        Args:
            finbert_emb: FinBERT embeddings
            mpnet_emb: MPNet embeddings
            alpha: Weight for FinBERT embeddings (1-alpha for MPNet)
            
        Returns:
            Combined embeddings
        """
        # Normalize embeddings to unit vectors before combining
        finbert_norm = finbert_emb / np.linalg.norm(finbert_emb, axis=1, keepdims=True)
        mpnet_norm = mpnet_emb / np.linalg.norm(mpnet_emb, axis=1, keepdims=True)
        
        # Weighted combination of the two embeddings
        combined = alpha * finbert_norm + (1 - alpha) * mpnet_norm
        
        # Normalize the combined embeddings
        combined_norm = combined / np.linalg.norm(combined, axis=1, keepdims=True)
        
        logger.info(f"Combined embeddings with alpha={alpha} (FinBERT weight)")
        return combined_norm
    
    def get_embedding_dimension(self) -> Tuple[int, int]:
        """
        Get the dimensionality of FinBERT and MPNet embeddings
        
        Returns:
            Tuple of (FinBERT dimension, MPNet dimension)
        """
        # Test embeddings to get dimensions
        test_text = ["Test sentence"]
        finbert_test = self.get_finbert_embeddings(test_text)
        mpnet_test = self.get_mpnet_embeddings(test_text)
        
        return finbert_test.shape[1], mpnet_test.shape[1]
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """
        Save embeddings to file for later use
        
        Args:
            embeddings: NumPy array of embeddings
            filepath: Path to save embeddings
        """
        np.save(filepath, embeddings)
        logger.info(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """
        Load previously saved embeddings
        
        Args:
            filepath: Path to load embeddings from
            
        Returns:
            NumPy array of embeddings
        """
        embeddings = np.load(filepath)
        logger.info(f"Loaded embeddings from {filepath}")
        return embeddings