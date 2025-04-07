from src.data.preprocessor import TextPreprocessor
import os
import torch

# Create directories if they don't exist
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)  # Make sure raw directory exists

# Check if file exists
if not os.path.exists('data/raw/fairy_tales.txt'):
    print("Error: fairy_tales.txt not found in data/raw/ directory")
    exit(1)

# Read the fairy tales
with open('data/raw/fairy_tales.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create and train a tokenizer
tokenizer = TextPreprocessor(min_freq=2)
tokenizer.build_vocab([text])

# Split into train/validation (90/10 split)
split_idx = int(len(text) * 0.9)
train_text = text[:split_idx]
val_text = text[split_idx:]

# Prepare tokenizer state for saving
tokenizer_state = {
    'token_to_id': tokenizer.token_to_id,
    'id_to_token': tokenizer.id_to_token,
    'vocab_size': tokenizer.vocab_size,
    'min_freq': tokenizer.min_freq
}

# Save processed data
with open('data/processed/train.txt', 'w', encoding='utf-8') as f:
    f.write(train_text)
    
with open('data/processed/val.txt', 'w', encoding='utf-8') as f:
    f.write(val_text)

# Save tokenizer
torch.save(tokenizer_state, 'data/preprocessor.pt')

print(f"Preprocessing complete! Vocabulary size: {tokenizer.vocab_size}")
