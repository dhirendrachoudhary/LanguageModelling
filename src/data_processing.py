import numpy as np

def load_data(file_path):
    """Load text data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    return text

def preprocess_text(text):
    """Convert text into character indices."""
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    encoded_text = [char_to_idx[ch] for ch in text]
    return encoded_text, char_to_idx, idx_to_char

def create_sequences(encoded_text, seq_length=10):
    """Generate input-target sequences from encoded text."""
    sequences, targets = [], []
    for i in range(len(encoded_text) - seq_length):
        sequences.append(encoded_text[i:i+seq_length])
        targets.append(encoded_text[i+seq_length])
    return np.array(sequences), np.array(targets)
