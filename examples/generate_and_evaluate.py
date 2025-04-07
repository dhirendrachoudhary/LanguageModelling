# examples/generate_and_evaluate.py
import os
import math
import torch
import numpy as np
from datetime import datetime
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from src.models.lstm_model import LSTMLanguageModel
from src.models.transformer_model import TransformerLanguageModel
from src.data.preprocessor import TextPreprocessor
import argparse

class TextGeneratorEvaluator:
    def __init__(self, model, tokenizer, reference_text=None):
        self.model = model
        self.tokenizer = tokenizer
        self.reference_text = reference_text
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def generate_text(self, seed_text, max_length=100, temperature=1.0):
        """Generate text from seed and return metrics"""
        seed_tokens = self.tokenizer.tokenize(seed_text)
        generated_tokens = self.model.generate(
            seed_tokens, 
            self.tokenizer, 
            max_length=max_length, 
            temperature=temperature
        )
        generated_text = self.tokenizer.detokenize(generated_tokens)
        return generated_text, self.calculate_metrics(seed_text, generated_text)
    
    def calculate_metrics(self, seed_text, generated_text):
        """Calculate comprehensive text generation metrics"""
        metrics = {
            # Basic statistics
            'seed_length': len(seed_text.split()),
            'gen_length': len(generated_text.split()),
            
            # Diversity metrics
            'distinct_1': self.distinct_n(generated_text, 1),
            'distinct_2': self.distinct_n(generated_text, 2),
            'distinct_3': self.distinct_n(generated_text, 3),
            'repetition_2': self.repetition_score(generated_text, 2),
            
            # Model performance
            'perplexity': self.calculate_perplexity(generated_text),
        }

        if self.reference_text:
            # Quality metrics
            metrics.update({
                'bleu': sentence_bleu([self.reference_text.split()], generated_text.split()),
                # 'rouge_l': self.calculate_rouge_l(generated_text),
                'length_ratio': len(generated_text.split()) / len(self.reference_text.split()),
                'oov_rate': self.oov_rate(generated_text)
            })

        return metrics

    def distinct_n(self, text, n):
        """Calculate distinct-n score (unique n-grams ratio)"""
        tokens = text.split()
        if len(tokens) < n: return 0.0
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        return len(set(ngrams)) / len(ngrams) if ngrams else 0.0

    def repetition_score(self, text, n=2):
        """Calculate repetition percentage for n-grams"""
        tokens = text.split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        return 1 - (len(set(ngrams)) / len(ngrams)) if ngrams else 0.0

    def oov_rate(self, text):
        """Calculate out-of-vocabulary rate"""
        tokens = text.split()
        oov = sum(1 for t in tokens if t not in self.tokenizer.token_to_id)
        return oov / len(tokens) if tokens else 0.0

    def calculate_perplexity(self, text):
        """Calculate perplexity of generated text"""
        token_ids = self.tokenizer.tokenize(text)
        if len(token_ids) < 2: return float('inf')
        
        inputs = torch.tensor([token_ids[:-1]]).to(self.device)
        targets = torch.tensor([token_ids[1:]]).to(self.device)
        
        with torch.no_grad():
            outputs, _ = self.model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
        return math.exp(loss.item())

    def save_results(self, seed_text, generated_text, metrics, output_dir='generations'):
        """Save generated text and metrics to file"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(output_dir, f"generation_{timestamp}.txt")

        report = [
            f"Seed: {seed_text}",
            "\nGenerated Text:",
            generated_text,
            "\nMetrics:",
            *[f"- {k.replace('_', ' ').title()}: {v:.4f}" if isinstance(v, float) else f"- {k.replace('_', ' ').title()}: {v}"
              for k, v in metrics.items()]
        ]

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
            
        return filepath

def load_model(model_type):
        """Load model based on type with dynamic class handling"""
        model_classes = {
            'lstm': LSTMLanguageModel,
            'transformer': TransformerLanguageModel
        }
        
        checkpoint_path = f"models/{model_type}languagemodel_best.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found for {model_type} at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Initialize model with correct class and config
        model_class = model_classes[model_type]
        model = model_class(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text Generation Evaluator')
    parser.add_argument('--model', type=str, required=True,
                        choices=['lstm', 'transformer'],
                        help='Model type to evaluate (lstm/transformer)')
    parser.add_argument('--seed', type=str, default="Once upon a",
                        help='Seed text for generation')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum length of generated text')
    args = parser.parse_args()

    # Initialize components
    tokenizer = TextPreprocessor()
    tokenizer.load("data/preprocessor.pt")
    
    try:
        model, checkpoint = load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Evaluation setup
    reference = "Once upon a time there was a princess in a castle."
    evaluator = TextGeneratorEvaluator(model.to('cpu'), tokenizer, reference)
    
    # Generate and evaluate
    text, metrics = evaluator.generate_text(args.seed, args.max_length)
    
    # Display results
    print("\n" + "="*40)
    print(f"Generation using {args.model.upper()} model:")
    print(text)
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k.replace('_', ' ').title():<15}: {v:.4f}" if isinstance(v, float) else f"{k.replace('_', ' ').title():<15}: {v}")
    
    # Save results
    saved_path = evaluator.save_results(args.seed, text, metrics)
    print(f"\nReport saved to: {saved_path}")