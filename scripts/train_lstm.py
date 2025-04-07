import json
import torch
import os
from torch.utils.data import DataLoader
from src.data.data_loader import TextDataset
from src.models.lstm_model import LSTMLanguageModel
from src.training.trainer import Trainer

def main():
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    
    # Load configuration
    if not os.path.exists("config/lstm_config.json"):
        print("Error: config/lstm_config.json not found")
        return
        
    with open("config/lstm_config.json", "r") as f:
        config = json.load(f)
    
    # Load dataset
    train_dataset = TextDataset("data/processed/train.txt", config["seq_length"])
    val_dataset = TextDataset("data/processed/val.txt", config["seq_length"])
    
    print(f"Training with vocabulary size: {train_dataset.vocab_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    
    # Initialize model
    model = LSTMLanguageModel(
        vocab_size=train_dataset.vocab_size,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout_rate=config["dropout_rate"]
    )
    
    # Initialize trainer
    trainer = Trainer(model, config)
    
    # Train model
    trainer.train(train_loader, val_loader, config["epochs"])

if __name__ == "__main__":
    main()
