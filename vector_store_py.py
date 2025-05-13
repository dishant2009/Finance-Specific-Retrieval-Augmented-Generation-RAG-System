"""
Vector storage and retrieval module for the Financial RAG System
Manages both FAISS (local) and Pinecone (cloud) vector databases
"""

import os
import logging
from typing import List, Dict, Optional
import numpy as np
import faiss
import pickle

# Set up logging
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages both FAISS (local) and Pinecone (cloud) vector databases
    Provides unified interface for storing and retrieving embeddings
    """
    
    def __init__(self, dimension: int, use_pinecone: bool = True, 
                 pinecone_env: str = "us-east1-gcp"):
        """
        Initialize vector storage manager
        
        Args:
            dimension: Dimensionality of the embeddings
            use_pinecone: Whether to use Pinecone cloud storage
            pinecone_env: Pinecone environment name
        """
        self.dimension = dimension
        self.use_pinecone = use_pinecone
        
        # Initialize FAISS index for local retrieval
        # Using IndexFlatIP for inner product similarity (equivalent to cosine for normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_metadata = []
        
        # Initialize Pinecone if requested
        if use_pinecone:
            self.setup_pinecone(pinecone_env)
        
        logger.info(f"Initialized vector store manager with dimension={dimension}")
    
    def setup_pinecone(self, environment: str):
        """
        Setup Pinecone cloud vector database
        
        Args:
            environment: Pinecone environment name
        """
        try:
            # Import Pinecone here to handle cases where it's not installed
            import pinecone
            
            # Get API key from environment variables
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key:
                logger.warning("PINECONE_API_KEY not found. Pinecone features will be disabled.")
                self.use_pinecone = False
                return
            
            # Initialize Pinecone
            pinecone.init(api_key=api_key, environment=environment)
            
            # Create or connect to index
            index_name = "financial-rag-index"
            
            # Check if index exists, create if not
            if index_name not in pinecone.list_indexes():
                logger.info(f"Creating new Pinecone index: {index_name}")
                pinecone.create_index(
                    name=index_name,
                    dimension=self.dimension,
                    metric="cosine"  # Cosine similarity for semantic search
                )
            
            # Connect to the index
            self.pinecone_index = pinecone.Index(index_name)
            logger.info("Successfully connected to Pinecone")
        except ImportError:
            logger.error("Pinecone client not installed. Install with: pip install pinecone-client")
            self.use_pinecone = False
        except Exception as e:
            logger.error(f"Error setting up Pinecone: {str(e)}")
            self.use_pinecone = False
    
    def add_to_faiss(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Add embeddings to FAISS index
        
        Args:
            embeddings: NumPy array of embeddings
            metadata: List of metadata dictionaries for each embedding
        """
        # Ensure embeddings are in the correct format (float32 for FAISS)
        embeddings_float32 = embeddings.astype('float32')
        
        # Add to FAISS index
        self.faiss_index.add(embeddings_float32)
        self.faiss_metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} vectors to FAISS index")
    
    def add_to_pinecone(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Add embeddings to Pinecone index
        
        Args:
            embeddings: NumPy array of embeddings
            metadata: List of metadata dictionaries for each embedding
        """
        if not self.use_pinecone:
            return
        
        # Prepare vectors for Pinecone (requires specific format)
        vectors = []
        start_id = len(self.faiss_metadata)  # Use existing count as offset
        
        for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
            vector_id = str(start_id + i)  # Create unique ID
            vectors.append((vector_id, emb.tolist(), meta))
        
        # Batch upload to Pinecone
        try:
            self.pinecone_index.upsert(vectors=vectors)
            logger.info(f"Added {len(vectors)} vectors to Pinecone index")
        except Exception as e:
            logger.error(f"Error adding vectors to Pinecone: {str(e)}")
    
    def search_faiss(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search FAISS index for similar vectors
        
        Args:
            query_embedding: Query embedding to search for
            k: Number of nearest neighbors to return
            
        Returns:
            List of search results with metadata and similarity scores
        """
        # Ensure query is in correct format
        query_float32 = query_embedding.astype('float32')
        if len(query_float32.shape) == 1:
            query_float32 = query_float32.reshape(1, -1)
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_float32, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result (not empty)
                result = self.faiss_metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def search_pinecone(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search Pinecone index for similar vectors
        
        Args:
            query_embedding: Query embedding to search for
            k: Number of nearest neighbors to return
            
        Returns:
            List of search results with metadata and similarity scores
        """
        if not self.use_pinecone:
            return []
        
        try:
            # Query Pinecone index
            query_list = query_embedding.flatten().tolist()
            results = self.pinecone_index.query(
                vector=query_list,
                top_k=k,
                include_metadata=True
            )
            
            # Format results
            search_results = []
            for match in results.matches:
                result = match.metadata.copy()
                result['similarity_score'] = match.score
                search_results.append(result)
            
            return search_results
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            return []
    
    def hybrid_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Combine FAISS and Pinecone results for better retrieval
        This approach gives us the best of both local and cloud search
        
        Args:
            query_embedding: Query embedding to search for
            k: Number of results to return
            
        Returns:
            Combined and ranked search results
        """
        # Get results from both sources
        faiss_results = self.search_faiss(query_embedding, k)
        
        if self.use_pinecone:
            pinecone_results = self.search_pinecone(query_embedding, k)
            
            # Combine results and remove duplicates based on source + chunk_id
            all_results = faiss_results + pinecone_results
            unique_results = {}
            
            for result in all_results:
                # Create unique key from source and chunk_id
                key = f"{result.get('source', '')}_{result.get('chunk_id', '')}"
                
                # Keep the result with the highest similarity score
                if key not in unique_results or result['similarity_score'] > unique_results[key]['similarity_score']:
                    unique_results[key] = result
            
            # Convert back to list and sort by similarity score
            final_results = list(unique_results.values())
            final_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return final_results[:k]
        
        return faiss_results
    
    def save_faiss_index(self, filepath: str):
        """
        Save FAISS index and metadata to disk
        
        Args:
            filepath: Path to save the index (without extension)
        """
        # Save FAISS index
        faiss.write_index(self.faiss_index, f"{filepath}.index")
        
        # Save metadata
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(self.faiss_metadata, f)
        
        logger.info(f"Saved FAISS index to {filepath}")
    
    def load_faiss_index(self, filepath: str):
        """
        Load FAISS index and metadata from disk
        
        Args:
            filepath: Path to load the index from (without extension)
        """
        # Load FAISS index
        self.faiss_index = faiss.read_index(f"{filepath}.index")
        
        # Load metadata
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            self.faiss_metadata = pickle.load(f)
        
        logger.info(f"Loaded FAISS index from {filepath}")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store
        
        Returns:
            Dictionary with store statistics
        """
        stats = {
            'faiss_total_vectors': self.faiss_index.ntotal,
            'dimension': self.dimension,
            'use_pinecone': self.use_pinecone
        }
        
        if self.use_pinecone:
            try:
                # Get Pinecone index stats
                pinecone_stats = self.pinecone_index.describe_index_stats()
                stats['pinecone_total_vectors'] = pinecone_stats.total_vector_count
            except:
                stats['pinecone_total_vectors'] = "Unable to fetch"
        
        return stats