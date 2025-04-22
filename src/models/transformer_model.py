# src/models/transformer_model.py
import torch
import torch.nn as nn
import math
from .base_model import BaseModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerLanguageModel(BaseModel):
    def __init__(self, vocab_size, embedding_dim, nhead, dim_feedforward, num_layers, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout_rate)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        self.embedding_dim = embedding_dim
    
    def _generate_square_subsequent_mask(self, sz):
        # Create a square mask for the transformer
        # The mask is used to prevent attending to future tokens
        # in the sequence during training
        # The mask is of size (sz, sz) and has 0s in the upper triangle
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src):
        src_mask = self._generate_square_subsequent_mask(src.size(1)).to(src.device)
        src = self.embedding(src) * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.output_layer(output)
        return output
    
    def generate(self, seed_tokens, tokenizer, max_length=50, temperature=0.1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        generated = seed_tokens.copy()
        input_ids = torch.tensor([generated]).to(device)
        
        # Generate one token at a time
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                output = self(input_ids)
                
                # Get the predictions for the last token in the sequence
                logits = output[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                
                # Sample from the distribution
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
                
                # Update input for next iteration
                input_ids = torch.tensor([generated]).to(device)
                
                # Stop if we generate the end token
                if next_token == tokenizer.token_to_id['<|endoftext|>']:
                    break
                    
        return generated
    



     