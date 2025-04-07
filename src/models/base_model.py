# src/models/base_model.py
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def generate(self, seed_text, max_length, temperature=1.0):
        """Generate text based on a seed string"""
        raise NotImplementedError("Subclasses must implement generate method")
