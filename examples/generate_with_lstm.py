import json
import torch
import os
from src.models.lstm_model import LSTMLanguageModel
from src.data.preprocessor import TextPreprocessor

def main():
    # Check files exist
    if not os.path.exists("data/preprocessor.pt"):
        print("Error: Tokenizer not found at data/preprocessor.pt")
        return

    # Load configuration
    if not os.path.exists("config/lstm_config.json"):
        print("Error: config/lstm_config.json not found")
        return
        
    with open("config/lstm_config.json", "r") as f:
        training_config = json.load(f)

    config =  training_config["1"]
    if not os.path.exists(f"models/{config['model_name']}_best.pt"):
        print(f"Error: Model checkpoint not found at models/{config['model_name']}_best.pt")
        return
    
    # Load the vocabulary
    tokenizer = TextPreprocessor()
    tokenizer.load("data/preprocessor.pt")
    print(f"Loaded tokenizer with vocabulary size: {tokenizer.vocab_size}")
    
    # Load the checkpoint
    try:
        checkpoint = torch.load(f"models/{config['model_name']}_best.pt", map_location='cuda' if torch.cuda.is_available() else 'cpu')
        # If using GPU, uncomment the next line

        print("Checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Check if checkpoint is dictionary with model_state_dict
    if not isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint:
        print("Warning: Checkpoint format not as expected, trying to load as state_dict directly")
        model_state = checkpoint
    else:
        model_state = checkpoint['model_state_dict']
        
    # Initialize model with the right vocabulary size
    vocab_size = tokenizer.vocab_size
    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout_rate=0.2
    )

    # Load the state dictionary
    try:
        model.load_state_dict(model_state)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model state: {e}")
        # If error occurs, it might be a vocabulary size mismatch
        if "size mismatch" in str(e):
            # Extract the correct vocab size from error message
            import re
            match = re.search(r'shape torch.Size\(\[(\d+), \d+\]\)', str(e))
            if match:
                correct_size = int(match.group(1))
                print(f"Attempting to reload with vocabulary size: {correct_size}")
                model = LSTMLanguageModel(
                    vocab_size=correct_size,
                    embedding_dim=256,
                    hidden_dim=512,
                    num_layers=2,
                    dropout_rate=0.2
                )
                model.load_state_dict(model_state)
                print("Model loaded successfully with adjusted vocabulary size")
            else:
                return
        else:
            return

    # Generate text
    seed_text = "Once upon a time"
    seed_tokens = tokenizer.tokenize(seed_text)
    print(f"Seed tokens: {seed_tokens}")
    
    generated_tokens = model.generate(seed_tokens, tokenizer, max_length=100)
    generated_text = tokenizer.detokenize(generated_tokens)
    
    print(f"Generated text:\n{generated_text}")

if __name__ == "__main__":
    main()
