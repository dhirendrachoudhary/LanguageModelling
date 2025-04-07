# src/data/preprocessor.py
import re
import torch
from collections import Counter

class TextPreprocessor:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.token_to_id = {'<|pad|>': 0, '<|unk|>': 1, '<|endoftext|>': 2}
        self.id_to_token = {0: '<|pad|>', 1: '<|unk|>', 2: '<|endoftext|>'}
        self.vocab_size = len(self.token_to_id)
    
    def load(self, path):
        """Load tokenizer state from a saved file"""
        state = torch.load(path, weights_only=False)
        self.token_to_id = state['token_to_id']
        self.id_to_token = state['id_to_token']
        self.vocab_size = state['vocab_size']
        self.min_freq = state['min_freq']
        return self

    def build_vocab(self, texts):
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())
            
        # Add words that appear frequently enough
        for word, count in word_counts.items():
            if count >= self.min_freq and word not in self.token_to_id:
                self.token_to_id[word] = len(self.token_to_id)
                self.id_to_token[len(self.id_to_token)] = word
                
        self.vocab_size = len(self.token_to_id)
        return self.vocab_size
    
    def tokenize(self, text):
        return [self.token_to_id.get(word, self.token_to_id['<|unk|>']) 
                for word in text.split()]
    
    def detokenize(self, token_ids):
        return ' '.join([self.id_to_token.get(token_id, '<|unk|>') 
                        for token_id in token_ids])
