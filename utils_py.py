"""
Utility functions for the Financial RAG System
Contains helper functions for data processing, system maintenance, and common operations
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import pandas as pd
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class FinRAGUtils:
    """Utility class with static methods for common operations"""
    
    @staticmethod
    def create_directory_structure(base_path: str = "./finrag_system"):
        """
        Create the recommended directory structure for the RAG system
        
        Args:
            base_path: Base directory path
        """
        directories = [
            'data/documents',
            'data/embeddings',
            'data/qa_pairs',
            'models/fine_tuned',
            'models/embeddings',
            'logs',
            'outputs/evaluations',
            'outputs/responses',
            'config'
        ]
        
        for directory in directories:
            full_path = os.path.join(base_path, directory)
            os.makedirs(full_path, exist_ok=True)
            logger.info(f"Created directory: {full_path}")
    
    @staticmethod
    def load_qa_pairs_from_file(filepath: str) -> List[Dict]:
        """
        Load Q&A pairs from various file formats (JSON, CSV, etc.)
        
        Args:
            filepath: Path to Q&A file
            
        Returns:
            List of Q&A pair dictionaries
        """
        file_extension = Path(filepath).suffix.lower()
        
        try:
            if file_extension == '.json':
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    # Handle both list of dicts and dict with list
                    if isinstance(data, list):
                        return data
                    elif 'qa_pairs' in data:
                        return data['qa_pairs']
                    else:
                        logger.warning(f"Unexpected JSON structure in {filepath}")
                        return []
            
            elif file_extension == '.csv':
                df = pd.read_csv(filepath)
                qa_pairs = []
                required_columns = ['question', 'positive_context']
                
                # Check if required columns exist
                if not all(col in df.columns for col in required_columns):
                    logger.error(f"CSV must contain columns: {required_columns}")
                    return []
                
                for _, row in df.iterrows():
                    qa_pair = {
                        'question': row['question'],
                        'positive_context': row['positive_context'],
                        'negative_context': row.get('negative_context', ''),
                        'answer': row.get('answer', '')
                    }
                    qa_pairs.append(qa_pair)
                
                return qa_pairs
            
            else:
                logger.error(f"Unsupported file format: {file_extension}")
                return []
        
        except Exception as e:
            logger.error(f"Error loading Q&A pairs from {filepath}: {str(e)}")
            return []
    
    @staticmethod
    def save_qa_pairs_to_file(qa_pairs: List[Dict], filepath: str):
        """
        Save Q&A pairs to file in JSON or CSV format
        
        Args:
            qa_pairs: List of Q&A pair dictionaries
            filepath: Path to save the file
        """
        file_extension = Path(filepath).suffix.lower()
        
        try:
            if file_extension == '.json':
                with open(filepath, 'w') as f:
                    json.dump(qa_pairs, f, indent=2)
            
            elif file_extension == '.csv':
                df = pd.DataFrame(qa_pairs)
                df.to_csv(filepath, index=False)
            
            else:
                logger.error(f"Unsupported file format: {file_extension}")
                return
            
            logger.info(f"Saved {len(qa_pairs)} Q&A pairs to {filepath}")
        
        except Exception as e:
            logger.error(f"Error saving Q&A pairs to {filepath}: {str(e)}")
    
    @staticmethod
    def validate_documents(document_paths: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate that document files exist and are readable
        
        Args:
            document_paths: List of document file paths
            
        Returns:
            Tuple of (valid_paths, invalid_paths)
        """
        valid_paths = []
        invalid_paths = []
        
        supported_extensions = {'.pdf', '.docx', '.txt'}
        
        for path in document_paths:
            if not os.path.exists(path):
                logger.warning(f"File does not exist: {path}")
                invalid_paths.append(path)
                continue
            
            if Path(path).suffix.lower() not in supported_extensions:
                logger.warning(f"Unsupported file type: {path}")
                invalid_paths.append(path)
                continue
            
            if not os.access(path, os.R_OK):
                logger.warning(f"Cannot read file: {path}")
                invalid_paths.append(path)
                continue
            
            valid_paths.append(path)
        
        logger.info(f"Validated {len(document_paths)} documents: {len(valid_paths)} valid, {len(invalid_paths)} invalid")
        return valid_paths, invalid_paths
    
    @staticmethod
    def monitor_system_resources():
        """
        Monitor system resources (memory, disk space, etc.)
        
        Returns:
            Dictionary with resource information
        """
        try:
            import psutil
            
            # Get memory information
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            resources = {
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_percent': memory.percent
                },
                'disk': {
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'used_percent': (disk.used / disk.total) * 100
                },
                'cpu_percent': psutil.cpu_percent(interval=1)
            }
            
            # Warning if resources are low
            if memory.percent > 90:
                logger.warning(f"High memory usage: {memory.percent}%")
            if (disk.used / disk.total) * 100 > 90:
                logger.warning(f"Low disk space: {(disk.free / (1024**3)):.2f} GB free")
            
            return resources
        
        except ImportError:
            logger.warning("psutil not installed. Cannot monitor system resources.")
            return {}
    
    @staticmethod
    def benchmark_search_performance(rag_system, test_queries: List[str], iterations: int = 5) -> Dict:
        """
        Benchmark search performance of the RAG system
        
        Args:
            rag_system: The RAG system instance
            test_queries: List of test queries
            iterations: Number of iterations to run
            
        Returns:
            Performance metrics
        """
        import time
        
        times = []
        
        for i in range(iterations):
            logger.info(f"Benchmark iteration {i+1}/{iterations}")
            iteration_times = []
            
            for query in test_queries:
                start_time = time.time()
                try:
                    response = rag_system.answer_question(query)
                    end_time = time.time()
                    iteration_times.append(end_time - start_time)
                except Exception as e:
                    logger.error(f"Error in benchmark query '{query}': {str(e)}")
                    continue
            
            times.extend(iteration_times)
        
        if not times:
            logger.error("No successful benchmark runs")
            return {}
        
        metrics = {
            'avg_response_time': np.mean(times),
            'min_response_time': np.min(times),
            'max_response_time': np.max(times),
            'std_response_time': np.std(times),
            'total_queries': len(times)
        }
        
        logger.info(f"Benchmark completed. Average response time: {metrics['avg_response_time']:.3f} seconds")
        return metrics
    
    @staticmethod
    def export_responses_to_csv(responses: List[Dict], filepath: str):
        """
        Export RAG system responses to CSV for analysis
        
        Args:
            responses: List of response dictionaries
            filepath: Path to save CSV file
        """
        # Flatten response data for CSV
        flattened_data = []
        
        for response in responses:
            flat_response = {
                'query': response.get('query', ''),
                'answer': response.get('answer', ''),
                'num_chunks': response.get('retrieved_chunks', 0),
                'avg_similarity': np.mean(response.get('similarity_scores', [0])),
                'sources': '; '.join(response.get('sources', [])),
                'timestamp': response.get('timestamp', '')
            }
            flattened_data.append(flat_response)
        
        df = pd.DataFrame(flattened_data)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(responses)} responses to {filepath}")
    
    @staticmethod
    def cleanup_old_files(directory: str, days_old: int = 7, pattern: str = "*"):
        """
        Clean up old files in a directory
        
        Args:
            directory: Directory to clean
            days_old: Delete files older than this many days
            pattern: File pattern to match (glob pattern)
        """
        from glob import glob
        import time
        
        file_pattern = os.path.join(directory, pattern)
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)  # Convert days to seconds
        
        deleted_count = 0
        for filepath in glob(file_pattern):
            if os.path.getmtime(filepath) < cutoff_time:
                try:
                    os.remove(filepath)
                    deleted_count += 1
                    logger.info(f"Deleted old file: {filepath}")
                except Exception as e:
                    logger.error(f"Error deleting {filepath}: {str(e)}")
        
        logger.info(f"Cleanup completed. Deleted {deleted_count} files older than {days_old} days")
    
    @staticmethod
    def validate_qa_pairs(qa_pairs: List[Dict]) -> List[Dict]:
        """
        Validate and clean Q&A pairs
        
        Args:
            qa_pairs: List of Q&A pair dictionaries
            
        Returns:
            List of validated Q&A pairs
        """
        valid_pairs = []
        required_fields = ['question', 'positive_context']
        
        for i, pair in enumerate(qa_pairs):
            # Check required fields
            if not all(field in pair and pair[field] for field in required_fields):
                logger.warning(f"Q&A pair {i} missing required fields: {required_fields}")
                continue
            
            # Clean and validate text fields
            pair['question'] = str(pair['question']).strip()
            pair['positive_context'] = str(pair['positive_context']).strip()
            
            # Ensure minimum length
            if len(pair['question']) < 10:
                logger.warning(f"Q&A pair {i} has very short question")
                continue
            
            if len(pair['positive_context']) < 20:
                logger.warning(f"Q&A pair {i} has very short context")
                continue
            
            valid_pairs.append(pair)
        
        logger.info(f"Validated Q&A pairs: {len(valid_pairs)} valid out of {len(qa_pairs)} total")
        return valid_pairs
    
    @staticmethod
    def calculate_embedding_similarities(embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarities between query and document embeddings
        
        Args:
            embeddings: Document embeddings (n x d)
            query_embedding: Query embedding (1 x d)
            
        Returns:
            Array of similarity scores
        """
        # Normalize embeddings
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        query_norm = query_embedding / np.linalg.norm(query_embedding, keepdims=True)
        
        # Calculate cosine similarity
        similarities = np.dot(embeddings_norm, query_norm.T).flatten()
        
        return similarities
    
    @staticmethod
    def create_evaluation_dataset(responses: List[Dict], ground_truth_file: str) -> List[Dict]:
        """
        Create evaluation dataset by matching responses with ground truth
        
        Args:
            responses: List of RAG system responses
            ground_truth_file: Path to ground truth file
            
        Returns:
            List of evaluation examples
        """
        try:
            # Load ground truth
            with open(ground_truth_file, 'r') as f:
                ground_truth = json.load(f)
            
            # Create lookup by query
            gt_lookup = {item['query']: item for item in ground_truth}
            
            eval_dataset = []
            for response in responses:
                query = response.get('query', '')
                if query in gt_lookup:
                    eval_item = {
                        'query': query,
                        'predicted_answer': response.get('answer', ''),
                        'reference_answer': gt_lookup[query].get('answer', ''),
                        'predicted_sources': response.get('sources', []),
                        'reference_sources': gt_lookup[query].get('relevant_docs', [])
                    }
                    eval_dataset.append(eval_item)
            
            logger.info(f"Created evaluation dataset with {len(eval_dataset)} examples")
            return eval_dataset
        
        except Exception as e:
            logger.error(f"Error creating evaluation dataset: {str(e)}")
            return []
    
    @staticmethod
    def log_system_stats(rag_system):
        """
        Log comprehensive system statistics
        
        Args:
            rag_system: The RAG system instance
        """
        stats = rag_system.get_system_stats()
        
        logger.info("=== RAG System Statistics ===")
        logger.info(f"Knowledge Base Built: {stats['system_status']['knowledge_base_built']}")
        logger.info(f"Model Fine-tuned: {stats['system_status']['model_fine_tuned']}")
        
        if 'vector_store' in stats:
            logger.info(f"FAISS Index Size: {stats['vector_store']['faiss_total_vectors']} vectors")
            if stats['vector_store']['use_pinecone']:
                logger.info(f"Pinecone Index Size: {stats['vector_store'].get('pinecone_total_vectors', 'N/A')} vectors")
        
        if 'embeddings' in stats:
            logger.info(f"FinBERT Dimension: {stats['embeddings']['finbert_dimension']}")
            logger.info(f"MPNet Dimension: {stats['embeddings']['mpnet_dimension']}")
        
        # System resources
        resources = FinRAGUtils.monitor_system_resources()
        if resources:
            logger.info(f"Memory Usage: {resources['memory']['used_percent']:.1f}%")
            logger.info(f"Disk Usage: {resources['disk']['used_percent']:.1f}%")
            logger.info(f"CPU Usage: {resources['cpu_percent']:.1f}%")
        
        logger.info("=== End Statistics ===")

# Standalone utility functions
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration for the entire system
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logging.info(f"Logging configured: level={log_level}, file={log_file}")

# Configuration validation
def validate_config(config: Dict) -> bool:
    """
    Validate configuration dictionary
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = [
        'finbert_model',
        'mpnet_model',
        'chunk_size',
        'overlap'
    ]
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            return False
    
    # Validate numeric values
    numeric_keys = {
        'chunk_size': (100, 2048),
        'overlap': (0, 200),
        'fine_tune_epochs': (1, 20),
        'fine_tune_batch_size': (1, 32)
    }
    
    for key, (min_val, max_val) in numeric_keys.items():
        if key in config:
            value = config[key]
            if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                logger.error(f"Invalid value for {key}: {value} (should be between {min_val} and {max_val})")
                return False
    
    logger.info("Configuration validation passed")
    return True