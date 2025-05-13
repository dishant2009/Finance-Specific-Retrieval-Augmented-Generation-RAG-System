"""
Configuration settings for the Financial RAG System
Contains all configurable parameters and environment variables
"""

import os
from typing import Dict, Optional

class FinRAGConfig:
    """Configuration class for Financial RAG System"""
    
    # Document processing settings
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    
    # Model settings
    FINBERT_MODEL = "ProsusAI/finbert"
    MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"
    
    # Fine-tuning parameters
    FINE_TUNE_EPOCHS = 3
    FINE_TUNE_BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    
    # Retrieval settings
    TOP_K_RESULTS = 5
    EMBEDDING_COMBINATION_ALPHA = 0.7  # Weight for FinBERT vs MPNet
    
    # Vector store settings
    USE_PINECONE = True
    PINECONE_ENVIRONMENT = "us-east1-gcp"
    PINECONE_INDEX_NAME = "financial-rag-index"
    
    # Paths
    FINE_TUNED_MODEL_PATH = "./models/fine_tuned_finbert"
    LOGS_DIR = "./logs"
    DATA_DIR = "./data"
    
    # API Keys (should be set as environment variables)
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    
    @classmethod
    def get_config(cls) -> Dict:
        """Return configuration as dictionary"""
        return {
            'chunk_size': cls.CHUNK_SIZE,
            'overlap': cls.CHUNK_OVERLAP,
            'finbert_model': cls.FINBERT_MODEL,
            'mpnet_model': cls.MPNET_MODEL,
            'fine_tune_epochs': cls.FINE_TUNE_EPOCHS,
            'fine_tune_batch_size': cls.FINE_TUNE_BATCH_SIZE,
            'learning_rate': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            'top_k_results': cls.TOP_K_RESULTS,
            'embedding_alpha': cls.EMBEDDING_COMBINATION_ALPHA,
            'use_pinecone': cls.USE_PINECONE,
            'pinecone_env': cls.PINECONE_ENVIRONMENT,
            'pinecone_index': cls.PINECONE_INDEX_NAME,
            'fine_tuned_model_path': cls.FINE_TUNED_MODEL_PATH,
            'logs_dir': cls.LOGS_DIR,
            'data_dir': cls.DATA_DIR
        }
    
    @classmethod
    def validate_environment(cls) -> bool:
        """Validate that required environment variables are set"""
        if cls.USE_PINECONE and not cls.PINECONE_API_KEY:
            print("Warning: PINECONE_API_KEY not set. Pinecone features will be disabled.")
            return False
        return True