# src/training/trainer.py
import torch
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.model_name = config['model_name']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        # Initialize lists to store loss values for graphing
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass (different for LSTM and Transformer)
            if hasattr(self.model, 'lstm'):
                outputs, _ = self.model(inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
            else:
                outputs = self.model(inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass (different for LSTM and Transformer)
                if hasattr(self.model, 'lstm'):
                    outputs, _ = self.model(inputs)
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)
                else:
                    outputs = self.model(inputs)
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    # In the train method where you save the model
    def train(self, train_dataloader, val_dataloader, epochs):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_dataloader)
            val_loss = self.evaluate(val_dataloader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
            
            # Save best model with additional info
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs('models', exist_ok=True)  # Ensure directory exists
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'vocab_size': self.model.embedding.weight.size(0),
                    'config': self.config
                }, f"models/{self.model_name}_best.pt")
                print(f"Model saved with validation loss: {val_loss:.4f}")
            # Save intermediate model
            if (epoch + 1) % 2 == 0:
                os.makedirs('models', exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'vocab_size': self.model.embedding.weight.size(0),
                    'config': self.config
                }, f"models/{self.model_name}_epoch_{epoch+1}.pt")
                print(f"Model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.model.embedding.weight.size(0),
            'config': self.config
        }, f"models/{self.model_name}_final.pt")
        # Generate training report
        self.generate_training_report()

    def generate_training_report(self):
        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss Over {self.config["epochs"]} Epochs')
        plt.legend()
        plt.grid(True)
        
        # Save the plot as an image
        os.makedirs('reports', exist_ok=True)
        plt.savefig(f'reports/{self.model_name}_training_results.png')
        plt.close()
        
