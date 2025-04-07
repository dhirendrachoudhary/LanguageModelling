# scripts/train_transformer.py
import json
import torch
from torch.utils.data import DataLoader
from src.data.data_loader import TextDataset
from src.models.transformer_model import TransformerLanguageModel
from src.data.preprocessor import TextPreprocessor
from src.training.trainer import Trainer

def main():
    # Load configuration
    with open("config/transformer_config.json", "r") as f:
        config = json.load(f)

    tokenizer = TextPreprocessor()
    tokenizer.load("data/preprocessor.pt")

    train_dataset = TextDataset("data/processed/train.txt", config["seq_length"], tokenizer)
    val_dataset = TextDataset("data/processed/val.txt", config["seq_length"], tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    
    # Initialize model
    model = TransformerLanguageModel(
        vocab_size=train_dataset.vocab_size,
        embedding_dim=config["embedding_dim"],
        nhead=config["nhead"],
        dim_feedforward=config["dim_feedforward"],
        num_layers=config["num_layers"],
        dropout_rate=config["dropout_rate"]
    )
    
    # Initialize trainer
    trainer = Trainer(model, config)
    
    # Train model
    trainer.train(train_loader, val_loader, config["epochs"])

if __name__ == "__main__":
    main()
