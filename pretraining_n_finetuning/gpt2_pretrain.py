import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer, models, pre_tokenizers, trainers


batch_size    = 16
block_size    = 64
learning_rate = 3e-4
epochs        = 5
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 1
eval_iters    = 200
n_embd        = 256
n_head        = 3
n_layer       = 4
dropout       = 0.2
bpe_vocab     = 30000

print(device)



with open('/home/susmita-roy/Downloads/NLP/NLP Project/fairytale.txt', 'r', encoding='utf-8') as f:
    text = f.read()



if not os.path.isfile('tokenizer.json'):

    with open('data.txt', 'w', encoding='utf-8') as f:
        f.write(text)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=bpe_vocab,
                                  special_tokens=["<pad>", "<unk>"])
    tokenizer.train(files=["data.txt"], trainer=trainer)
    tokenizer.save("tokenizer.json")
else:
    tokenizer = Tokenizer.from_file("tokenizer.json")


encode = lambda s: tokenizer.encode(s).ids
decode = lambda ids: tokenizer.decode(ids)

vocab_size = tokenizer.get_vocab_size()
stoi       = tokenizer.get_vocab()                # token -> id
itos       = {i: tok for tok, i in stoi.items()}  # id    -> token


data      = torch.tensor(encode(text), dtype=torch.long)
n_train   = int(0.9 * len(data))
train_data= data[:n_train]
val_data  = data[n_train:]

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.bs   = block_size
    def __len__(self):
        return len(self.data) - self.bs
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.bs]
        y = self.data[idx + 1 : idx + 1 + self.bs]
        return x, y

train_ds = CharDataset(train_data, block_size)
val_ds   = CharDataset(val_data,   block_size)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        for i, (xb, yb) in enumerate(loader):
            if i >= eval_iters: break
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.size()
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B,T, n_embd)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa    = MultiHeadAttention(n_head, head_size)
        self.ffwd  = FeedForward(n_embd)
        self.ln1   = nn.LayerNorm(n_embd)
        self.ln2   = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_embd)
        self.lm_head= nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        token_emb = self.token_embedding_table(idx)
        pos_emb   = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            probs     = F.softmax(logits[:, -1, :], dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


model     = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")


for epoch in range(1, epochs+1):
    print(f"\n=== Epoch {epoch}/{epochs} ===")
    # training
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # evaluation
    losses = estimate_loss()
    print(f"Train loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}")


model.eval()
start = torch.zeros((1,1), dtype=torch.long, device=device)
out   = model.generate(start, max_new_tokens=200)[0].tolist()
print("\n--- Generated Text ---\n")
print(decode(out))
