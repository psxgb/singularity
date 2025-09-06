import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTConfig:
    def __init__(self, vocab_size, n_layer=3, n_head=8, d_model=256, d_ff=256, max_seq_len=256):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.head(x)

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(config.d_model, config.n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model)
        )
        mask = torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        h = self.ln1(x)
        T = h.size(1)
        attn_out, _ = self.attn(h, h, h, attn_mask=self.causal_mask[:T, :T], need_weights=False)
        x = x + attn_out
        h = self.ln2(x)
        x = x + self.mlp(h)
        return x
