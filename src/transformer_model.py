import torch
import torch.nn as nn

class TransformerTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers):
        super(TransformerTextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, embed_size))  # Positional encoding
        self.transformer = nn.Transformer(
            d_model=embed_size, 
            nhead=num_heads, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers, 
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x, tgt, src_mask=None, tgt_mask=None):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        out = self.transformer(x, tgt, src_mask, tgt_mask)
        out = self.fc(out[:, -1, :])  # Take output from last timestep
        return out
