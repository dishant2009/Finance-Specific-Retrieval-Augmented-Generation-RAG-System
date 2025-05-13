"""
Main Financial RAG System that orchestrates all components
Combines document processing, embedding generation, retrieval, and response generation
"""

import logging
from typing import List, Dict, Optional
import os
from datetime import datetime

from config import FinRAGConfig
from document_processor import FinancialDocumentProcessor
from embedding_generator import FinEmbeddingGenerator
from vector_store import VectorStoreManager
from finbert_finetuner import FinBERTFineTuner
from evaluator import RAGEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialRAGSystem:
    """
    Main RAG system that combines all components for financial Q&A
    Orchestrates document processing, embedding generation, retrieval, and response generation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Financial RAG system with all components
        
        Args:
            config: Configuration dictionary (uses default if None)
        """
        # Use provided config or default
        self.config = config or FinRAGConfig.get_config()
        
        # Validate environment
        FinRAGConfig.validate_environment()
        
        # Initialize document processor
        self.doc_processor = FinancialDocumentProcessor(
            chunk_size=self.config.get('chunk_size', 512),
            overlap=self.config.get('overlap', 50)
        )
        
        # Initialize embedding generator
        self.embedding_generator = FinEmbeddingGenerator(
            finbert_model=self.config.get('finbert_model', 'ProsusAI/finbert'),
            mpnet_model=self.config.get('mpnet_model', 'sentence-transformers/all-mpnet-base-v2')
        )
        
        # Vector store will be initialized after determining embedding dimension
        self.vector_store = None
        
        # Initialize fine-tuner
        self.fine_tuner = FinBERTFineTuner(
            base_model=self.config.get('finbert_model', 'ProsusAI/finbert')
        )
        
        # Initialize evaluator
        self.evaluator = RAGEvaluator()
        
        # System state
        self.is_built = False
        self.fine_tuned = False
        
        logger.info("Financial RAG System initialized successfully")
    
    def build_knowledge_base(self, document_paths: List[str], save_embeddings: bool = True):
        """
        Process documents and build the searchable knowledge base
        
        Args:
            document_paths: List of paths to financial documents
            save_embeddings: Whether to save embeddings for later use
        """
        logger.info("Building knowledge base from financial documents")
        
        # Process documents into chunks
        chunks = self.doc_processor.process_documents(document_paths)
        
        if not chunks:
            logger.error("No chunks created from documents")
            raise ValueError("Failed to create chunks from provided documents")
        
        # Extract text for embedding generation
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings using both models
        logger.info("Generating FinBERT embeddings...")
        finbert_embeddings = self.embedding_generator.get_finbert_embeddings(texts)
        
        logger.info("Generating MPNet embeddings...")
        mpnet_embeddings = self.embedding_generator.get_mpnet_embeddings(texts)
        
        # Combine embeddings for enhanced representation
        logger.info("Combining embeddings...")
        combined_embeddings = self.embedding_generator.combine_embeddings(
            finbert_embeddings, 
            mpnet_embeddings,
            alpha=self.config.get('embedding_alpha', 0.7)
        )
        
        # Initialize vector store with correct dimension
        self.vector_store = VectorStoreManager(
            dimension=combined_embeddings.shape[1],
            use_pinecone=self.config.get('use_pinecone', True),
            pinecone_env=self.config.get('pinecone_env', 'us-east1-gcp')
        )
        
        # Store embeddings in both FAISS and Pinecone
        logger.info("Storing embeddings in vector databases...")
        self.vector_store.add_to_faiss(combined_embeddings, chunks)
        self.vector_store.add_to_pinecone(combined_embeddings, chunks)
        
        # Save embeddings if requested
        if save_embeddings:
            embeddings_dir = os.path.join(self.config.get('data_dir', './data'), 'embeddings')
            os.makedirs(embeddings_dir, exist_ok=True)
            
            # Save to files with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            finbert_path = os.path.join(embeddings_dir, f'finbert_embeddings_{timestamp}.npy')
            mpnet_path = os.path.join(embeddings_dir, f'mpnet_embeddings_{timestamp}.npy')
            combined_path = os.path.join(embeddings_dir, f'combined_embeddings_{timestamp}.npy')
            
            self.embedding_generator.save_embeddings(finbert_embeddings, finbert_path)
            self.embedding_generator.save_embeddings(mpnet_embeddings, mpnet_path)
            self.embedding_generator.save_embeddings(combined_embeddings, combined_path)
        
        self.is_built = True
        logger.info("Knowledge base construction completed successfully")
    
    def fine_tune_finbert(self, qa_pairs: List[Dict], force_retrain: bool = False):
        """
        Fine-tune FinBERT using Q&A pairs with LoRA for improved performance
        
        Args:
            qa_pairs: List of Q&A pairs with questions and contexts
            force_retrain: Whether to retrain even if a model exists
        """
        model_path = self.config.get('fine_tuned_model_path', './models/fine_tuned_finbert')
        
        # Check if model already exists
        if os.path.exists(model_path) and not force_retrain:
            logger.info(f"Fine-tuned model already exists at {model_path}. Skipping training.")
            logger.info("Set force_retrain=True to retrain the model.")
            self.fine_tuned = True
            return
        
        logger.info("Fine-tuning FinBERT with financial Q&A data")
        
        # Create directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Fine-tune the model
        fine_tuned_model = self.fine_tuner.fine_tune_model(
            qa_pairs,
            output_dir=model_path,
            epochs=self.config.get('fine_tune_epochs', 3),
            batch_size=self.config.get('fine_tune_batch_size', 8),
            learning_rate=self.config.get('learning_rate', 2e-5)
        )
        
        self.fine_tuned = True
        logger.info("FinBERT fine-tuning completed successfully")
    
    def retrieve_relevant_chunks(self, query: str, k: int = None) -> List[Dict]:
        """
        Retrieve most relevant document chunks for a query
        
        Args:
            query: User query
            k: Number of chunks to retrieve (uses config default if None)
            
        Returns:
            List of relevant chunks with metadata and similarity scores
        """
        if not self.is_built:
            raise ValueError("Knowledge base not built. Call build_knowledge_base() first.")
        
        k = k or self.config.get('top_k_results', 5)
        
        # Generate query embeddings
        query_finbert = self.embedding_generator.get_finbert_embeddings([query])
        query_mpnet = self.embedding_generator.get_mpnet_embeddings([query])
        query_combined = self.embedding_generator.combine_embeddings(
            query_finbert, 
            query_mpnet,
            alpha=self.config.get('embedding_alpha', 0.7)
        )
        
        # Search vector store
        results = self.vector_store.hybrid_search(query_combined, k)
        
        logger.info(f"Retrieved {len(results)} relevant chunks for query")
        return results
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict], 
                       use_llm: bool = False) -> str:
        """
        Generate answer based on query and retrieved context
        
        Args:
            query: User query
            retrieved_chunks: Retrieved relevant chunks
            use_llm: Whether to use a language model for generation
            
        Returns:
            Generated answer
        """
        if not retrieved_chunks:
            return "I couldn't find relevant information to answer your question. Please try reformulating your query."
        
        # Combine retrieved texts as context
        context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
        
        if use_llm:
            # In a production system, you would use an actual language model here
            # For example, using transformers library with GPT-3, GPT-4, or T5
            # This is a placeholder for LLM integration
            prompt = f"""
            Based on the following financial documents, please answer the question accurately and concisely.
            
            Context from financial documents:
            {context}
            
            Question: {query}
            
            Answer: """
            
            # Placeholder - integrate with your preferred language model
            answer = self._call_language_model(prompt)
        else:
            # Simple rule-based response for demonstration
            answer = f"""Based on the retrieved financial documents, here's information related to your query about '{query}':

The most relevant information I found discusses: {retrieved_chunks[0]['text'][:200]}...

This information comes from {len(retrieved_chunks)} relevant document(s) with an average similarity score of {np.mean([chunk['similarity_score'] for chunk in retrieved_chunks]):.3f}."""
        
        return answer
    
    def answer_question(self, query: str, k: int = None, use_llm: bool = False) -> Dict:
        """
        Main method to answer financial questions using RAG
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            use_llm: Whether to use language model for generation
            
        Returns:
            Response dictionary with answer and metadata
        """
        if not self.is_built:
            raise ValueError("Knowledge base not built. Call build_knowledge_base() first.")
        
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve_relevant_chunks(query, k)
        
        # Generate answer
        answer = self.generate_answer(query, retrieved_chunks, use_llm)
        
        # Prepare comprehensive response
        response = {
            'query': query,
            'answer': answer,
            'retrieved_chunks': len(retrieved_chunks),
            'sources': [chunk['source'] for chunk in retrieved_chunks],
            'similarity_scores': [chunk['similarity_score'] for chunk in retrieved_chunks],
            'chunk_details': retrieved_chunks,
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'finbert_model': self.config.get('finbert_model'),
                'fine_tuned': self.fine_tuned,
                'embeddings_combined': True
            }
        }
        
        return response
    
    def _call_language_model(self, prompt: str) -> str:
        """
        Placeholder method for language model integration
        Replace with actual LLM API calls (OpenAI, Hugging Face, etc.)
        
        Args:
            prompt: Input prompt for the language model
            
        Returns:
            Generated response
        """
        # This is a placeholder - integrate with your preferred LLM
        # Examples:
        # - OpenAI GPT-3/4 via API
        # - Hugging Face transformers
        # - Local language models
        # - Google T5, FLAN-T5, etc.
        
        logger.warning("Language model integration not implemented. Using placeholder response.")
        return "This would be the response from a language model like GPT-3/4."
    
    def get_system_stats(self) -> Dict:
        """
        Get comprehensive system statistics
        
        Returns:
            Dictionary with system statistics
        """
        stats = {
            'system_status': {
                'knowledge_base_built': self.is_built,
                'model_fine_tuned': self.fine_tuned,
                'config': self.config
            }
        }
        
        if self.vector_store:
            stats['vector_store'] = self.vector_store.get_stats()
        
        if self.is_built:
            # Get embedding dimensions
            finbert_dim, mpnet_dim = self.embedding_generator.get_embedding_dimension()
            stats['embeddings'] = {
                'finbert_dimension': finbert_dim,
                'mpnet_dimension': mpnet_dim,
                'combined_dimension': finbert_dim  # After normalization
            }
        
        return stats
    
    def save_system_state(self, filepath: str):
        """
        Save the system state for later loading
        
        Args:
            filepath: Path to save system state
        """
        if not self.vector_store:
            logger.warning("No vector store to save")
            return
        
        # Save FAISS index and metadata
        faiss_path = filepath.replace('.pkl', '_faiss')
        self.vector_store.save_faiss_index(faiss_path)
        
        # Save system configuration
        import pickle
        system_state = {
            'config': self.config,
            'is_built': self.is_built,
            'fine_tuned': self.fine_tuned,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(system_state, f)
        
        logger.info(f"System state saved to {filepath}")
    
    def load_system_state(self, filepath: str):
        """
        Load a previously saved system state
        
        Args:
            filepath: Path to load system state from
        """
        import pickle
        
        # Load system state
        with open(filepath, 'rb') as f:
            system_state = pickle.load(f)
        
        self.config = system_state.get('config', self.config)
        self.is_built = system_state.get('is_built', False)
        self.fine_tuned = system_state.get('fine_tuned', False)
        
        # Load FAISS index if system was built
        if self.is_built:
            faiss_path = filepath.replace('.pkl', '_faiss')
            
            # Reinitialize components with loaded config
            self.embedding_generator = FinEmbeddingGenerator(
                finbert_model=self.config.get('finbert_model'),
                mpnet_model=self.config.get('mpnet_model')
            )
            
            # Get embedding dimension and initialize vector store
            finbert_dim, mpnet_dim = self.embedding_generator.get_embedding_dimension()
            self.vector_store = VectorStoreManager(
                dimension=finbert_dim,
                use_pinecone=self.config.get('use_pinecone', True)
            )
            
            # Load FAISS index
            self.vector_store.load_faiss_index(faiss_path)
        
        logger.info(f"System state loaded from {filepath}")
    
    def batch_answer_questions(self, queries: List[str], k: int = None) -> List[Dict]:
        """
        Answer multiple questions in batch for efficiency
        
        Args:
            queries: List of queries to answer
            k: Number of chunks to retrieve per query
            
        Returns:
            List of response dictionaries
        """
        logger.info(f"Processing {len(queries)} queries in batch")
        
        responses = []
        for query in queries:
            try:
                response = self.answer_question(query, k)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing query '{query}': {str(e)}")
                responses.append({
                    'query': query,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return responses