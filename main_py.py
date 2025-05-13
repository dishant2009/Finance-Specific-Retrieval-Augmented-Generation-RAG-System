"""
Main execution script for the Financial RAG System
Demonstrates the complete workflow from setup to evaluation
"""

import os
import sys
import logging
from typing import List, Dict
import json
from datetime import datetime

# Import our custom modules
from config import FinRAGConfig
from rag_system import FinancialRAGSystem
from evaluator import RAGEvaluator
from utils import FinRAGUtils, setup_logging, validate_config

def main():
    """Main function demonstrating the complete RAG system usage"""
    
    # Setup logging
    setup_logging(log_level="INFO", log_file="./logs/finrag_system.log")
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Financial RAG System demonstration")
    
    # Create directory structure
    logger.info("Creating directory structure...")
    FinRAGUtils.create_directory_structure()
    
    # Get configuration
    config = FinRAGConfig.get_config()
    
    # Validate configuration
    if not validate_config(config):
        logger.error("Configuration validation failed")
        sys.exit(1)
    
    # Initialize the RAG system
    logger.info("Initializing Financial RAG System...")
    rag_system = FinancialRAGSystem(config)
    
    # Example document paths (you'll need to provide actual paths)
    # In a real scenario, these would be paths to your financial documents
    document_paths = [
        './data/documents/annual_report_2023.pdf',
        './data/documents/quarterly_earnings_q4.pdf',
        './data/documents/sec_filing_10k.pdf',
        './data/documents/investor_presentation.pdf',
        './data/documents/earnings_call_transcript.txt'
    ]
    
    # Validate documents
    valid_docs, invalid_docs = FinRAGUtils.validate_documents(document_paths)
    
    if invalid_docs:
        logger.warning(f"Found {len(invalid_docs)} invalid documents:")
        for doc in invalid_docs:
            logger.warning(f"  - {doc}")
    
    if not valid_docs:
        logger.error("No valid documents found. Please add financial documents to ./data/documents/")
        logger.info("For testing, you can add sample PDF or text files to continue.")
        
        # Create sample documents for demonstration
        create_sample_documents()
        # Re-validate after creating samples
        valid_docs, _ = FinRAGUtils.validate_documents(document_paths)
    
    if valid_docs:
        # Build knowledge base from documents
        logger.info("Building knowledge base...")
        try:
            rag_system.build_knowledge_base(valid_docs)
            logger.info("Knowledge base built successfully")
        except Exception as e:
            logger.error(f"Error building knowledge base: {str(e)}")
            logger.info("Continuing with limited functionality...")
    else:
        logger.warning("No valid documents available. Some features will be limited.")
    
    # Example Q&A pairs for fine-tuning
    # In practice, you would load these from a file
    example_qa_pairs = [
        {
            'question': 'What was the company\'s revenue in Q4?',
            'positive_context': 'The company reported revenue of $2.1 billion in Q4 2023, representing a 15% increase year-over-year.',
            'negative_context': 'The company experienced increased operational expenses during Q4.',
            'answer': 'The company\'s revenue in Q4 was $2.1 billion.'
        },
        {
            'question': 'How did debt levels change compared to last year?',
            'positive_context': 'Total debt decreased from $850 million in 2022 to $720 million in 2023, a reduction of $130 million.',
            'negative_context': 'The company increased its research and development spending significantly.',
            'answer': 'Debt levels decreased by $130 million, from $850 million to $720 million.'
        },
        {
            'question': 'What are the main risk factors?',
            'positive_context': 'Key risk factors include market volatility, regulatory changes, and competition from emerging technologies.',
            'negative_context': 'The company has strong cash reserves and multiple revenue streams.',
            'answer': 'Main risk factors include market volatility, regulatory changes, and competition from emerging technologies.'
        }
    ]
    
    # Save Q&A pairs for later use
    qa_pairs_file = './data/qa_pairs/financial_qa_pairs.json'
    FinRAGUtils.save_qa_pairs_to_file(example_qa_pairs, qa_pairs_file)
    
    # Fine-tune FinBERT (optional - can be time-consuming)
    fine_tune_model = input("Do you want to fine-tune FinBERT? (y/n): ").lower().strip() == 'y'
    
    if fine_tune_model:
        logger.info("Fine-tuning FinBERT with Q&A pairs...")
        try:
            # Load more Q&A pairs if available
            all_qa_pairs = FinRAGUtils.load_qa_pairs_from_file(qa_pairs_file)
            validated_pairs = FinRAGUtils.validate_qa_pairs(all_qa_pairs)
            
            if validated_pairs:
                rag_system.fine_tune_finbert(validated_pairs)
            else:
                logger.warning("No valid Q&A pairs found for fine-tuning")
        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")
    else:
        logger.info("Skipping FinBERT fine-tuning")
    
    # Example queries for testing
    example_queries = [
        "What was the company's revenue growth in the last quarter?",
        "How did the company's debt levels change compared to last year?",
        "What are the main risk factors mentioned in the latest filing?",
        "What was the dividend policy mentioned in the annual report?",
        "What were the key operational challenges faced in Q4?",
        "How does the company plan to expand into new markets?"
    ]
    
    # Test the system with example queries
    logger.info("Testing the RAG system with example queries")
    responses = []
    
    for query in example_queries:
        try:
            logger.info(f"Processing query: {query}")
            response = rag_system.answer_question(query)
            responses.append(response)
            
            # Display results
            print(f"\n{'='*60}")
            print(f"Query: {response['query']}")
            print(f"Answer: {response['answer']}")
            print(f"Sources: {response['sources']}")
            print(f"Retrieved chunks: {response['retrieved_chunks']}")
            print(f"Avg similarity: {np.mean(response['similarity_scores']):.3f}")
            print('='*60)
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
    
    # Export responses for analysis
    if responses:
        responses_file = f"./outputs/responses/responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        FinRAGUtils.export_responses_to_csv(responses, responses_file)
        logger.info(f"Responses exported to {responses_file}")
    
    # Run evaluation if test data is available
    evaluator = RAGEvaluator()
    
    # Example test data structure (in practice, load from file)
    test_data = [
        {
            'query': 'What was the revenue in Q4?',
            'answer': 'The revenue in Q4 was $2.1 billion, representing a 15% increase year-over-year.',
            'relevant_docs': ['annual_report_2023.pdf', 'quarterly_earnings_q4.pdf']
        },
        {
            'query': 'How did debt levels change?',
            'answer': 'Total debt decreased from $850 million to $720 million, a reduction of $130 million.',
            'relevant_docs': ['sec_filing_10k.pdf', 'annual_report_2023.pdf']
        }
    ]
    
    # Run evaluation if system is built
    if rag_system.is_built and test_data:
        try:
            logger.info("Running system evaluation...")
            metrics = evaluator.comprehensive_evaluation(test_data, rag_system)
            
            # Save evaluation results
            eval_file = f"./outputs/evaluations/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            evaluator.generate_evaluation_report(metrics, eval_file)
            
            # Display key metrics
            print(f"\n{'='*40} EVALUATION RESULTS {'='*40}")
            print(f"Overall F1-score: {metrics['overall_f1']:.4f}")
            print(f"Retrieval Precision: {metrics['retrieval']['precision']:.4f}")
            print(f"Retrieval Recall: {metrics['retrieval']['recall']:.4f}")
            print(f"Generation ROUGE-1: {metrics['generation']['rouge1']:.4f}")
            print(f"Generation ROUGE-L: {metrics['generation']['rougeL']:.4f}")
            print('='*88)
            
            # Compare with baseline if available
            # This would compare with SBERT or another baseline model
            baseline_metrics = {
                'retrieval': {'f1_score': 0.65},  # Example baseline F1
                'generation': {'rouge1': 0.55, 'rougeL': 0.50}
            }
            
            comparison = evaluator.compare_with_baseline(metrics, baseline_metrics)
            if comparison.get('f1_improvement'):
                print(f"\nImprovement over baseline:")
                print(f"F1-score improvement: {comparison['f1_improvement']:.2f}%")
                if comparison['f1_improvement'] >= 17:
                    print("✅ Achieved 17%+ improvement over baseline SBERT!")
                else:
                    print(f"Current improvement: {comparison['f1_improvement']:.2f}% (target: 17%)")
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
    
    # Log system statistics
    FinRAGUtils.log_system_stats(rag_system)
    
    # Benchmark performance
    run_benchmark = input("\nDo you want to run performance benchmark? (y/n): ").lower().strip() == 'y'
    
    if run_benchmark and rag_system.is_built:
        logger.info("Running performance benchmark...")
        benchmark_queries = example_queries[:3]  # Use subset for benchmark
        benchmark_results = FinRAGUtils.benchmark_search_performance(
            rag_system, 
            benchmark_queries, 
            iterations=3
        )
        
        if benchmark_results:
            print(f"\n{'='*30} BENCHMARK RESULTS {'='*30}")
            print(f"Average Response Time: {benchmark_results['avg_response_time']:.3f} seconds")
            print(f"Min Response Time: {benchmark_results['min_response_time']:.3f} seconds")
            print(f"Max Response Time: {benchmark_results['max_response_time']:.3f} seconds")
            print('='*70)
    
    # Save system state for later use
    save_state = input("\nDo you want to save the system state? (y/n): ").lower().strip() == 'y'
    
    if save_state:
        state_file = f"./data/system_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        rag_system.save_system_state(state_file)
        logger.info(f"System state saved to {state_file}")
    
    # Interactive query mode
    interactive_mode = input("\nEnter interactive query mode? (y/n): ").lower().strip() == 'y'
    
    if interactive_mode and rag_system.is_built:
        print("\n" + "="*50)
        print("Interactive Query Mode - Enter 'quit' to exit")
        print("="*50)
        
        while True:
            user_query = input("\nEnter your financial question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_query:
                continue
            
            try:
                response = rag_system.answer_question(user_query)
                print(f"\nAnswer: {response['answer']}")
                print(f"Sources: {', '.join(response['sources'])}")
                print(f"Confidence: {np.mean(response['similarity_scores']):.3f}")
            except Exception as e:
                print(f"Error: {str(e)}")
    
    logger.info("Financial RAG System demonstration completed successfully!")
    print("\n" + "="*60)
    print("Thank you for using the Financial RAG System!")
    print("="*60)

def create_sample_documents():
    """Create sample financial documents for demonstration"""
    logger = logging.getLogger(__name__)
    
    sample_docs = {
        './data/documents/sample_annual_report.txt': """
        Annual Report 2023
        
        Executive Summary:
        Our company achieved record revenue of $5.2 billion in 2023, representing a 12% increase from the previous year.
        Net income reached $850 million, with earnings per share of $3.42.
        
        Financial Highlights:
        - Revenue: $5.2 billion (2023) vs $4.6 billion (2022)
        - Net Income: $850 million (2023) vs $720 million (2022)
        - Total Assets: $8.5 billion
        - Debt-to-Equity Ratio: 0.45
        
        Key Achievements:
        - Launched three new product lines
        - Expanded operations to five new markets
        - Reduced operational costs by 8%
        
        Risk Factors:
        - Market volatility and economic uncertainty
        - Regulatory changes in key markets
        - Competition from emerging technologies
        - Supply chain disruptions
        """,
        
        './data/documents/sample_q4_earnings.txt': """
        Q4 2023 Earnings Report
        
        Fourth Quarter Performance:
        - Revenue: $1.35 billion (vs $1.15 billion Q4 2022)
        - Operating Income: $275 million
        - Net Income: $205 million
        - Earnings per Share: $0.85
        
        Operational Highlights:
        - Customer acquisition increased by 15%
        - Product margins improved to 45%
        - International sales grew by 22%
        
        Outlook for 2024:
        - Expected revenue growth of 10-15%
        - Planned expansion into emerging markets
        - Continued investment in R&D (15% of revenue)
        
        Key Metrics:
        - Customer Retention Rate: 92%
        - Employee Growth: 12%
        - Market Share: 23% (up from 21%)
        """
    }
    
    os.makedirs('./data/documents/', exist_ok=True)
    
    for filepath, content in sample_docs.items():
        with open(filepath, 'w') as f:
            f.write(content)
        logger.info(f"Created sample document: {filepath}")
    
    logger.info("Sample documents created successfully")

if __name__ == "__main__":
    # Ensure numpy is available for calculations
    try:
        import numpy as np
    except ImportError:
        print("NumPy not installed. Please install with: pip install numpy")
        sys.exit(1)
    
    main()