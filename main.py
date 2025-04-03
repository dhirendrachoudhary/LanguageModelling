import torch
import argparse
import os
from src.data_processing import load_data, preprocess_text, create_sequences
from src.lstm_model import LSTMTextGenerator
from src.transformer_model import TransformerTextGenerator
from src.train import train_model
from src.generate import generate_text
from src.utils import save_model, load_model
# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Text generation using LSTM/Transformer models")
    parser.add_argument("--mode", type=str, choices=["train", "generate"], default="train",
                        help="Operation mode: 'train' or 'generate'")
    parser.add_argument("--model_path", type=str, 
                        help="Path to load saved model (required for generate mode)")
    parser.add_argument("--start_text", type=str, default="once upon a",
                        help="Initial text for generation")
    parser.add_argument("--model_type", type=str, choices=["lstm", "transformer"], default="lstm",
                        help="Type of model to use: 'lstm' or 'transformer'")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model hyperparameters
    embed_size = 16
    hidden_size = 128
    num_layers = 2
    num_heads = 4
    MODEL_TYPE = args.model_type
    
    # Load and process data
    file_path = "data/fairy_tales.txt"
    text = load_data(file_path)
    encoded_text, char_to_idx, idx_to_char = preprocess_text(text)
    vocab_size = len(char_to_idx)
    
    if args.mode == "train":
        # Prepare training data
        seq_length = 10
        X, y = create_sequences(encoded_text, seq_length)
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # Initialize model
        if MODEL_TYPE == "lstm":
            model = LSTMTextGenerator(vocab_size, embed_size, hidden_size, num_layers).to(device)
        else:
            model = TransformerTextGenerator(vocab_size, embed_size, num_heads, hidden_size, num_layers).to(device)
        
        # Train model
        model = train_model(model, X_tensor, y_tensor, device, model_type=MODEL_TYPE)
        
        # Save trained model check for path if not create one
        experiment_folder = "experiments/transformer_experiment_1" if MODEL_TYPE == "transformer" else "experiments/lstm_experiment_1"
        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)

        save_model(model, f"{experiment_folder}/model_checkpoint.pth")
        
    elif args.mode == "generate":
        if not args.model_path:
            raise ValueError("Model path is required for generate mode. Use --model_path to specify.")
        
        # Load the appropriate model
        if MODEL_TYPE == "lstm":
            model_class = LSTMTextGenerator
        else:
            model_class = TransformerTextGenerator
        
        # Load saved model with the correct parameters
        model = load_model(
            model_class=model_class,
            filepath=args.model_path,
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,  # Used as hidden_size for LSTM and hidden_dim for Transformer
            num_layers=num_layers,
            device=device,
            num_heads=num_heads if MODEL_TYPE == "transformer" else None  # Only needed for transformer
        )
        
        # Generate text
        generated_text = generate_text(model, args.start_text, char_to_idx, idx_to_char, device, model_type=MODEL_TYPE)
        print(generated_text)

if __name__ == "__main__":
    main()
