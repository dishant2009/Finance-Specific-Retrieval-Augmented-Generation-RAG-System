"""
FinBERT fine-tuning module using LoRA/QLoRA for efficient adaptation
Implements instruction tuning for improved context understanding
"""

import os
import logging
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

class FinBERTDataset(Dataset):
    """Custom dataset class for Q&A training data"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

class FinBERTFineTuner:
    """
    Fine-tunes FinBERT model using LoRA/QLoRA for efficient adaptation
    Implements instruction tuning for improved context understanding
    """
    
    def __init__(self, base_model: str = "ProsusAI/finbert"):
        """
        Initialize FinBERT fine-tuner
        
        Args:
            base_model: Base FinBERT model name or path
        """
        self.base_model = base_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Add pad token if it doesn't exist (some models don't have one)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Initialized FinBERT fine-tuner with {base_model}")
    
    def prepare_model_for_training(self, use_qlora: bool = True, num_labels: int = 2):
        """
        Setup model with LoRA/QLoRA configuration for efficient fine-tuning
        
        Args:
            use_qlora: Whether to use QLoRA (4-bit quantization)
            num_labels: Number of labels for classification
            
        Returns:
            Model ready for training
        """
        logger.info(f"Preparing model for training with {'QLoRA' if use_qlora else 'LoRA'}")
        
        # Configure quantization for QLoRA if requested
        if use_qlora:
            # QLoRA configuration for 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,              # Enable 4-bit loading
                bnb_4bit_quant_type="nf4",      # Use normalized float 4-bit quantization
                bnb_4bit_compute_dtype=torch.float16,  # Compute in float16
                bnb_4bit_use_double_quant=True   # Double quantization for better accuracy
            )
            
            # Load model with quantization
            model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                num_labels=num_labels,
                device_map="auto"  # Automatically distribute across GPUs if available
            )
        else:
            # Load model without quantization (standard LoRA)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model,
                num_labels=num_labels
            ).to(self.device)
        
        # LoRA configuration for efficient fine-tuning
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,      # Sequence classification task
            inference_mode=False,             # Training mode
            r=16,                            # LoRA rank (higher = more parameters)
            lora_alpha=32,                   # LoRA scaling parameter
            lora_dropout=0.1,                # Dropout for LoRA layers
            target_modules=["query", "value", "key", "dense"]  # Which layers to add LoRA to
        )
        
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters info
        model.print_trainable_parameters()
        
        return model
    
    def create_qa_dataset(self, qa_pairs: List[Dict]) -> Dict:
        """
        Create training dataset from Q&A pairs with positive and negative examples
        
        Args:
            qa_pairs: List of Q&A pairs with questions and contexts
            
        Returns:
            Dictionary with questions, contexts, and labels
        """
        questions = []
        contexts = []
        labels = []
        
        for qa_pair in qa_pairs:
            question = qa_pair.get('question', '')
            positive_context = qa_pair.get('positive_context', '')
            negative_context = qa_pair.get('negative_context', '')
            
            # Create positive example (relevant context)
            if positive_context:
                questions.append(question)
                contexts.append(positive_context)
                labels.append(1)  # Label 1 for relevant
            
            # Create negative example if available (irrelevant context)
            if negative_context:
                questions.append(question)
                contexts.append(negative_context)
                labels.append(0)  # Label 0 for irrelevant
        
        logger.info(f"Created dataset with {len(questions)} examples")
        return {
            'questions': questions,
            'contexts': contexts,
            'labels': labels
        }
    
    def tokenize_dataset(self, dataset: Dict, max_length: int = 512) -> FinBERTDataset:
        """
        Tokenize the dataset for training
        
        Args:
            dataset: Dataset dictionary with questions, contexts, and labels
            max_length: Maximum sequence length
            
        Returns:
            FinBERTDataset object
        """
        # Tokenize question-context pairs
        encodings = self.tokenizer(
            dataset['questions'],
            dataset['contexts'],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return FinBERTDataset(encodings, dataset['labels'])
    
    def fine_tune_model(self, qa_pairs: List[Dict], output_dir: str = "./fine_tuned_finbert",
                       epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5):
        """
        Fine-tune FinBERT with instruction tuning using Q&A pairs
        
        Args:
            qa_pairs: List of Q&A pairs for training
            output_dir: Directory to save the fine-tuned model
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
            
        Returns:
            Fine-tuned model
        """
        logger.info("Starting FinBERT fine-tuning process")
        
        # Prepare model with LoRA
        model = self.prepare_model_for_training(use_qlora=True)
        
        # Create dataset
        dataset_dict = self.create_qa_dataset(qa_pairs)
        train_dataset = self.tokenize_dataset(dataset_dict)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,                    # Warmup steps for learning rate
            weight_decay=0.01,                   # Weight decay for regularization
            logging_dir=f'{output_dir}/logs',    # TensorBoard logs
            logging_steps=100,                   # Log every 100 steps
            save_strategy="epoch",               # Save at the end of each epoch
            save_total_limit=2,                  # Keep only the best 2 checkpoints
            load_best_model_at_end=False,        # Load best model at end
            dataloader_pin_memory=False,         # Disable pin memory for QLoRA
            fp16=torch.cuda.is_available(),      # Use mixed precision if on GPU
            report_to=["tensorboard"],           # Report to TensorBoard
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        logger.info("Beginning fine-tuning...")
        trainer.train()
        
        # Save the fine-tuned model
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
        
        # Save training metrics
        trainer.save_metrics("train", trainer.state.log_history)
        
        return model
    
    def load_fine_tuned_model(self, model_path: str):
        """
        Load a previously fine-tuned model for inference
        
        Args:
            model_path: Path to the fine-tuned model
            
        Returns:
            Loaded model ready for inference
        """
        logger.info(f"Loading fine-tuned model from {model_path}")
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=2,
            device_map="auto"
        )
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info("Fine-tuned model loaded successfully")
        return model, tokenizer
    
    def evaluate_model(self, model, test_qa_pairs: List[Dict]) -> Dict:
        """
        Evaluate the fine-tuned model on test data
        
        Args:
            model: Model to evaluate
            test_qa_pairs: Test Q&A pairs
            
        Returns:
            Evaluation metrics
        """
        # Create test dataset
        test_dataset_dict = self.create_qa_dataset(test_qa_pairs)
        test_dataset = self.tokenize_dataset(test_dataset_dict)
        
        # Prepare for evaluation
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        # Evaluate on test data
        with torch.no_grad():
            for batch in DataLoader(test_dataset, batch_size=8):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                correct_predictions += (predictions == batch['labels']).sum().item()
                total_predictions += len(batch['labels'])
        
        accuracy = correct_predictions / total_predictions
        logger.info(f"Model accuracy on test set: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }
    
    def create_instruction_dataset(self, qa_pairs: List[Dict]) -> List[str]:
        """
        Create instruction-tuned training data for improved context understanding
        
        Args:
            qa_pairs: Q&A pairs
            
        Returns:
            List of formatted instruction strings
        """
        instructions = []
        
        for qa_pair in qa_pairs:
            question = qa_pair.get('question', '')
            answer = qa_pair.get('answer', '')
            context = qa_pair.get('positive_context', '')
            
            # Create instruction format
            instruction = f"""
            Based on the following financial context, answer the question accurately:
            
            Context: {context}
            
            Question: {question}
            
            Answer: {answer}
            """.strip()
            
            instructions.append(instruction)
        
        return instructions