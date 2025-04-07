# src/data/data_loader.py
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text_file, seq_length, tokenizer=None):
        with open(text_file, 'r', encoding='utf-8') as f:
            self.data = f.read()
        
        # Initialize tokenizer if not provided
        if tokenizer is None:
            from .preprocessor import TextPreprocessor
            self.tokenizer = TextPreprocessor()
            # Build vocabulary from text
            self.tokenizer.build_vocab([self.data])
        else:
            self.tokenizer = tokenizer
            
        self.vocab_size = self.tokenizer.vocab_size
        self.seq_length = seq_length
        
        # Tokenize the entire text
        self.tokenized_text = self.tokenizer.tokenize(self.data)
        
    def __len__(self):
        return len(self.tokenized_text) - self.seq_length
        
    def __getitem__(self, idx):
        inputs = torch.tensor(self.tokenized_text[idx:idx+self.seq_length])
        targets = torch.tensor(self.tokenized_text[idx+1:idx+self.seq_length+1])
        return inputs, targets
