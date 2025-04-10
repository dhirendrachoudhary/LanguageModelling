# examples/generate_with_transformer.py
import json
import torch
import os
import re
from src.models.transformer_model import TransformerLanguageModel
from src.data.preprocessor import TextPreprocessor

def main():
    # Check files exist
    if not os.path.exists("data/preprocessor.pt"):
        print("Error: Tokenizer not found at data/preprocessor.pt")
        return
        
    with open("config/transformer_config.json", "r") as f:
       training_config = json.load(f)
    

    config = training_config["1"]

    
    # Load the vocabulary
    tokenizer = TextPreprocessor()
    tokenizer.load("data/preprocessor.pt")
    print(f"Loaded tokenizer with vocabulary size: {tokenizer.vocab_size}")
    
    # Load the checkpoint
    # Load the checkpoint
    try:
        checkpoint = torch.load(f"models/{config['model_name']}_best.pt", map_location='cuda' if torch.cuda.is_available() else 'cpu')
        # If using GPU, uncomment the next line

        print("Checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Check if checkpoint is dictionary with model_state_dict
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        print("Warning: Checkpoint format not as expected, trying to load as state_dict directly")
        model_state = checkpoint
    else:
        model_state = checkpoint['model_state_dict']
        
    # Initialize model with the right vocabulary size
    vocab_size = tokenizer.vocab_size
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=256,
        nhead=4,
        dim_feedforward=512,
        num_layers=2,
        dropout_rate=0.1
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
            match = re.search(r'shape torch.Size\(\[(\d+), \d+\]\)', str(e))
            if match:
                correct_size = int(match.group(1))
                print(f"Attempting to reload with vocabulary size: {correct_size}")
                model = TransformerLanguageModel(
                    vocab_size=correct_size,
                    embedding_dim=256,
                    nhead=4,
                    dim_feedforward=512,
                    num_layers=2,
                    dropout_rate=0.1
                )
                model.load_state_dict(model_state)
                print("Model loaded successfully with adjusted vocabulary size")
            else:
                return
        else:
            return

    # Generate text
    seed_text = "Lovely Ilonka"
    seed_tokens = tokenizer.tokenize(seed_text)
    print(f"Seed tokens: {seed_tokens}")
    
    # Custom generation function that handles the token_to_id dictionary correctly
    def generate_text(model, seed_tokens, tokenizer, max_length=100, temperature=1.0):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        generated = seed_tokens.copy()
        input_ids = torch.tensor([generated]).to(device)
        
        # Get the end token ID
        end_token_id = tokenizer.token_to_id.get('<|endoftext|>', -1) 
                
        # Generate one token at a time
        model.eval()
        with torch.no_grad():
            for _ in range(max_length):
                output = model(input_ids)
                
                # Get the predictions for the last token in the sequence
                logits = output[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                
                # Sample from the distribution
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
                
                # Update input for next iteration
                input_ids = torch.tensor([generated]).to(device)
                
                # Stop if we generate the end token
                if next_token == end_token_id:
                    break
                    
        return generated
    
    # Generate text using our custom function instead of the model's method
    generated_tokens = generate_text(model, seed_tokens, tokenizer, max_length=100)
    generated_text = tokenizer.detokenize(generated_tokens)
    
    print(f"Generated text:\n{generated_text}")

if __name__ == "__main__":
    main()
