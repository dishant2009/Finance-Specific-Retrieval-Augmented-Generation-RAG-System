"""
Evaluation module for the Financial RAG System
Computes retrieval and generation metrics including F1-score, precision, recall, and ROUGE scores
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)

# Try to import ROUGE, install with: pip install rouge-score
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    logger.warning("ROUGE scorer not available. Install with: pip install rouge-score")
    ROUGE_AVAILABLE = False

class RAGEvaluator:
    """
    Evaluates the performance of the RAG system
    Computes metrics like F1-score, precision, recall, and ROUGE scores
    """
    
    def __init__(self):
        """Initialize the evaluator with necessary components"""
        if ROUGE_AVAILABLE:
            # Initialize ROUGE scorer with different n-gram types
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        
        logger.info("RAG Evaluator initialized")
    
    def evaluate_retrieval(self, queries: List[str], ground_truth_docs: List[List[str]], 
                          predicted_docs: List[List[str]]) -> Dict:
        """
        Evaluate retrieval performance using precision, recall, and F1-score
        
        Args:
            queries: List of input queries
            ground_truth_docs: List of lists containing ground truth document IDs for each query
            predicted_docs: List of lists containing predicted document IDs for each query
            
        Returns:
            Dictionary with retrieval metrics
        """
        total_queries = len(queries)
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        ndcg_scores = []
        
        for i in range(total_queries):
            gt_set = set(ground_truth_docs[i])
            pred_set = set(predicted_docs[i])
            
            # Calculate precision: How many retrieved docs are relevant?
            if pred_set:
                precision = len(gt_set.intersection(pred_set)) / len(pred_set)
            else:
                precision = 0
            
            # Calculate recall: How many relevant docs were retrieved?
            if gt_set:
                recall = len(gt_set.intersection(pred_set)) / len(gt_set)
            else:
                recall = 0
            
            # Calculate F1-score: Harmonic mean of precision and recall
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            
            # Calculate NCDG (Normalized Cumulative Discount Gain)
            ndcg = self._calculate_ndcg(ground_truth_docs[i], predicted_docs[i])
            ndcg_scores.append(ndcg)
        
        # Calculate average metrics
        metrics = {
            'precision': total_precision / total_queries,
            'recall': total_recall / total_queries,
            'f1_score': total_f1 / total_queries,
            'ndcg': np.mean(ndcg_scores) if ndcg_scores else 0
        }
        
        logger.info(f"Retrieval metrics - F1: {metrics['f1_score']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        return metrics
    
    def evaluate_generation(self, predicted_answers: List[str], 
                           reference_answers: List[str]) -> Dict:
        """
        Evaluate answer generation quality using ROUGE scores
        
        Args:
            predicted_answers: List of generated answers
            reference_answers: List of reference (ground truth) answers
            
        Returns:
            Dictionary with generation metrics
        """
        if not ROUGE_AVAILABLE:
            logger.warning("ROUGE not available. Returning empty metrics.")
            return {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
        
        rouge_scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        # Calculate ROUGE scores for each answer pair
        for pred, ref in zip(predicted_answers, reference_answers):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        # Calculate average ROUGE scores
        metrics = {
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL'])
        }
        
        logger.info(f"Generation metrics - ROUGE-1: {metrics['rouge1']:.4f}, ROUGE-2: {metrics['rouge2']:.4f}, ROUGE-L: {metrics['rougeL']:.4f}")
        return metrics
    
    def comprehensive_evaluation(self, test_data: List[Dict], rag_system) -> Dict:
        """
        Run comprehensive evaluation of the RAG system on test data
        
        Args:
            test_data: List of test cases with queries, answers, and relevant docs
            rag_system: The RAG system to evaluate
            
        Returns:
            Complete evaluation metrics
        """
        logger.info("Running comprehensive evaluation")
        
        # Extract test components
        queries = [item['query'] for item in test_data]
        ground_truth_answers = [item['answer'] for item in test_data]
        ground_truth_docs = [item.get('relevant_docs', []) for item in test_data]
        
        # Get predictions from RAG system
        predicted_answers = []
        predicted_docs = []
        retrieval_times = []
        generation_times = []
        
        for query in queries:
            import time
            
            # Measure retrieval time
            start_time = time.time()
            response = rag_system.answer_question(query)
            retrieval_time = time.time() - start_time
            
            predicted_answers.append(response['answer'])
            predicted_docs.append(response['sources'])
            retrieval_times.append(retrieval_time)
        
        # Evaluate retrieval performance
        retrieval_metrics = self.evaluate_retrieval(queries, ground_truth_docs, predicted_docs)
        
        # Evaluate generation performance
        generation_metrics = self.evaluate_generation(predicted_answers, ground_truth_answers)
        
        # Calculate additional metrics
        additional_metrics = {
            'average_retrieval_time': np.mean(retrieval_times),
            'total_queries': len(queries),
            'avg_retrieved_docs': np.mean([len(docs) for docs in predicted_docs])
        }
        
        # Combine all metrics
        all_metrics = {
            'retrieval': retrieval_metrics,
            'generation': generation_metrics,
            'performance': additional_metrics,
            'overall_f1': retrieval_metrics['f1_score']  # Main comparison metric
        }
        
        logger.info(f"Comprehensive evaluation completed. Overall F1-score: {all_metrics['overall_f1']:.4f}")
        return all_metrics
    
    def _calculate_ndcg(self, relevance_scores: List[str], predicted_docs: List[str], k: int = None) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG)
        
        Args:
            relevance_scores: Ground truth relevant documents
            predicted_docs: Predicted documents
            k: Consider only top k results
            
        Returns:
            NDCG score
        """
        if k is None:
            k = len(predicted_docs)
        
        # Create relevance judgments (1 if relevant, 0 if not)
        relevance_map = {doc: 1 for doc in relevance_scores}
        relevance_list = [relevance_map.get(doc, 0) for doc in predicted_docs[:k]]
        
        # Calculate DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_list))
        
        # Calculate ideal DCG (sort by relevance)
        ideal_relevance = sorted(relevance_list, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        # Calculate NDCG
        if idcg == 0:
            return 0
        return dcg / idcg
    
    def evaluate_financial_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """
        Evaluate financial-specific metrics like numerical accuracy
        
        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries
            
        Returns:
            Financial-specific evaluation metrics
        """
        metrics = {
            'numerical_accuracy': 0,
            'percentage_accuracy': 0,
            'currency_accuracy': 0,
            'date_accuracy': 0
        }
        
        # This would implement domain-specific evaluation
        # For example, checking if financial figures are within acceptable ranges
        
        logger.info("Financial-specific metrics calculated")
        return metrics
    
    def compare_with_baseline(self, current_metrics: Dict, baseline_metrics: Dict) -> Dict:
        """
        Compare current model performance with baseline (e.g., SBERT)
        
        Args:
            current_metrics: Current model metrics
            baseline_metrics: Baseline model metrics
            
        Returns:
            Comparison results
        """
        comparison = {}
        
        # Compare F1 scores
        if 'retrieval' in current_metrics and 'retrieval' in baseline_metrics:
            current_f1 = current_metrics['retrieval']['f1_score']
            baseline_f1 = baseline_metrics['retrieval']['f1_score']
            improvement = (current_f1 - baseline_f1) / baseline_f1 * 100
            
            comparison['f1_improvement'] = improvement
            comparison['f1_current'] = current_f1
            comparison['f1_baseline'] = baseline_f1
        
        # Compare ROUGE scores
        if 'generation' in current_metrics and 'generation' in baseline_metrics:
            for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                if rouge_type in current_metrics['generation'] and rouge_type in baseline_metrics['generation']:
                    current_score = current_metrics['generation'][rouge_type]
                    baseline_score = baseline_metrics['generation'][rouge_type]
                    improvement = (current_score - baseline_score) / baseline_score * 100
                    comparison[f'{rouge_type}_improvement'] = improvement
        
        logger.info(f"Model comparison completed. F1 improvement: {comparison.get('f1_improvement', 0):.2f}%")
        return comparison
    
    def generate_evaluation_report(self, metrics: Dict, output_path: str = "evaluation_report.json"):
        """
        Generate a detailed evaluation report
        
        Args:
            metrics: Evaluation metrics
            output_path: Path to save the report
        """
        import json
        from datetime import datetime
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'summary': {
                'overall_f1': metrics.get('overall_f1', 0),
                'retrieval_performance': metrics.get('retrieval', {}),
                'generation_performance': metrics.get('generation', {}),
                'system_performance': metrics.get('performance', {})
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_path}")
    
    def plot_metrics(self, metrics_history: List[Dict], save_path: str = "metrics_plot.png"):
        """
        Plot evaluation metrics over time/iterations
        
        Args:
            metrics_history: List of metrics from different evaluations
            save_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            # Extract F1 scores over time
            f1_scores = [m['overall_f1'] for m in metrics_history]
            iterations = range(1, len(f1_scores) + 1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, f1_scores, 'b-o', linewidth=2, markersize=8)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('F1-Score', fontsize=12)
            plt.title('Model Performance Over Time', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            
            # Add value labels on points
            for i, f1 in enumerate(f1_scores):
                plt.annotate(f'{f1:.3f}', (i+1, f1), textcoords="offset points", 
                           xytext=(0,10), ha='center')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Metrics plot saved to {save_path}")
        except ImportError:
            logger.warning("Matplotlib not available. Install with: pip install matplotlib")