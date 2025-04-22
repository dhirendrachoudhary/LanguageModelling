from generate_and_evaluate import TextGeneratorEvaluator

import os
import math
import torch
import numpy as np
import torch.nn as nn
from datetime import datetime
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from src.data.data_loader import TextDataset
from src.models.lstm_model import LSTMLanguageModel
from src.models.transformer_model import TransformerLanguageModel
from src.data.preprocessor import TextPreprocessor
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
def load_model(model_type, model_path, vocab_size):
        """Load model based on type with dynamic class handling"""
        model_classes = {
            'lstm': LSTMLanguageModel,
            'transformer': TransformerLanguageModel
        }
        checkpoint_path = model_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found for {model_type} at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Initialize model with correct class and config
        model_class = model_classes[model_type]
        keys_to_remove = ['model_name','exist_model','seq_length','batch_size','learning_rate','epochs','grad_clip']
        for key in keys_to_remove:
            checkpoint['config'].pop(key, None)
        checkpoint['config']['vocab_size'] = vocab_size
        model = model_class(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint

def compute_perplexity(model,dataloader):
    """Compute the perplexity of the model on a given dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            
        if hasattr(model, 'lstm'):
            outputs, _ = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            total_tokens += targets.size(0)

        else:
            outputs = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            total_tokens += targets.size(0)
            
        # Compute loss
        loss = nn.CrossEntropyLoss()(outputs, targets)
        total_loss += loss.item() 

    
    perplexity = math.exp(-(total_loss / total_tokens ))
    return perplexity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text Generation Evaluator')
    parser.add_argument('--model', type=str, required=True,
                        choices=['lstm', 'transformer'],
                        help='Model type to evaluate (lstm/transformer)')
    
    args = parser.parse_args()

    # Initialize components
    tokenizer = TextPreprocessor()
    tokenizer.load("data/preprocessor.pt")

    vocab_size = tokenizer.vocab_size
    #get paths of lstm models
    val_dataset = TextDataset("data/processed/val.txt", 100, tokenizer)


    model_dir = 'models'
    model_type = args.model
    #get all models in the directory
    models = [f for f in os.listdir(model_dir) if f.endswith('.pt') and model_type in f]
    if not models:
        print(f"No {model_type} models found in {model_dir}")
        exit(1)
    results = {}
    for model_ in models:
        try:
            
            model, checkpoint = load_model(args.model, os.path.join(model_dir, model_),vocab_size)
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)
        print(f"Calculating perplexity for {model_}")
        # Move model to device
        model.to(device)
        # Set model to evaluation mode
        model.eval()
        # Create a DataLoader for the validation dataset
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        # Compute perplexity
        perplexity = compute_perplexity(model, val_dataloader)
        print(f"Perplexity for {model_}: {perplexity}")
        results[model_] = perplexity
    #save results to a files
    with open(f'examples/{args.model}_perplexity_results.txt', 'w') as f:
        for model_, perplexity in results.items():
            f.write(f"{model_}: {perplexity}\n")
    print("Perplexity results saved to perplexity_results.txt")
