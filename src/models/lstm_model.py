# src/models/lstm_model.py
import torch
import torch.nn as nn
from .base_model import BaseModel

class LSTMLanguageModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden

    def generate(self, seed_tokens, tokenizer, max_length=100, temperature=0.1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        input_ids = torch.tensor([seed_tokens]).to(device)
        generated = seed_tokens.copy()
        
        # Generate one token at a time
        self.eval()
        with torch.no_grad():
            hidden = None
            for _ in range(max_length):
                output, hidden = self(input_ids, hidden)
                
                # Get the predictions for the last token in the sequence
                logits = output[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                
                # Sample from the distribution
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
                
                # Update input for next iteration
                input_ids = torch.tensor([[next_token]]).to(device)
                
                # Stop if we generate the end token
                if next_token == tokenizer.token_to_id['<|endoftext|>']:
                    break
                    
        return generated
