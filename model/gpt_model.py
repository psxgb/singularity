import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Config
# -------------------------------
@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    d_model: int
    d_ff: int
    max_seq_len: int
    dropout: float
    tie_word_embeddings: bool
    use_rope: bool

# -------------------------------
# Rotary Positional Embeddings
# -------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb  # [T, head_dim]

# -------------------------------
# Multi-Head Attention with RoPE
# -------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        assert cfg.d_model % cfg.n_head == 0, "d_model must be divisible by n_head"

        self.qkv = nn.Linear(cfg.d_model, cfg.d_model * 3, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(self.head_dim) if cfg.use_rope else None

    def _apply_rope(self, x, rope_emb):
        # x: [B, T, n_head, head_dim]
        B, T, n_head, head_dim = x.shape
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        cos = rope_emb[:, ::2].unsqueeze(0).unsqueeze(2)
        sin = rope_emb[:, 1::2].unsqueeze(0).unsqueeze(2)
        x_rot = torch.empty_like(x)
        x_rot[..., ::2] = x1 * cos - x2 * sin
        x_rot[..., 1::2] = x1 * sin + x2 * cos
        return x_rot

    def forward(self, x, attn_mask=None, past_kv=None):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        # reshape to [B, T, n_head, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        # Apply RoPE if enabled
        if self.rope is not None:
            rope_emb = self.rope(T, x.device)  # [T, head_dim]
            q = self._apply_rope(q, rope_emb)
            k = self._apply_rope(k, rope_emb)

        # Transpose to [B, n_head, T, head_dim] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Concatenate past key/value along sequence dimension
        if past_kv is not None:
            past_k, past_v = past_kv
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)  # concat along sequence
                v = torch.cat([past_v, v], dim=2)

        # Attention
        k_t = k.transpose(-2, -1)  # [B, n_head, head_dim, Tp+T]
        att = torch.matmul(q, k_t) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            att = att.masked_fill(attn_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = torch.matmul(att, v)  # [B, n_head, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out(out)
        out = self.dropout(out)

        # Return new past key/value
        new_past_kv = (k, v)
        return out, new_past_kv

# -------------------------------
# FeedForward
# -------------------------------
class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# -------------------------------
# Transformer Block
# -------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x, attn_mask=None, past_kv=None):
        y = self.ln1(x)
        y, new_kv = self.attn(y, attn_mask=attn_mask, past_kv=past_kv)
        x = x + y
        y = self.ln2(x)
        y = self.ff(y)
        x = x + y
        return x, new_kv

# -------------------------------
# GPT Model
# -------------------------------
class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -------------------------------
    # Causal mask for autoregressive attention
    # -------------------------------
    def _causal_mask(self, T, device, past_len=0):
        total_len = past_len + T
        mask = torch.tril(torch.ones((total_len, total_len), device=device)).unsqueeze(0).unsqueeze(0)
        if past_len > 0:
            mask = mask[:, :, past_len:, :]
        return mask  # [1,1,T,total_len]

    # -------------------------------
    # Forward
    # -------------------------------
    def forward(self, input_ids, past_key_values=None, attn_mask=None):
        B, T = input_ids.size()

        # Determine past length
        past_len = 0
        if past_key_values is not None:
            for kv in past_key_values:
                if kv is not None:
                    past_len = kv[0].size(2)  # kv[0] is key [B, n_head, Tp, head_dim]
                    break

        # causal mask
        if attn_mask is None:
            attn_mask = self._causal_mask(T, input_ids.device, past_len=past_len)

        x = self.tok_emb(input_ids) + self.pos_emb[:, :T, :]
        x = self.drop(x)

        new_past_key_values = []
        for i, block in enumerate(self.blocks):
            kv = past_key_values[i] if past_key_values is not None else None
            x, new_kv = block(x, attn_mask=attn_mask, past_kv=kv)
            new_past_key_values.append(new_kv)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_past_key_values

# -------------------------------
# Utility to create model config
# -------------------------------
def get_model_config(cfg, vocab_size):
    model_config = GPTConfig(
        vocab_size=vocab_size,
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        d_model=cfg["model"]["d_model"],
        d_ff=cfg["model"]["d_ff"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"]['dropout'],
        tie_word_embeddings=cfg["model"]['tie_word_embeddings'],
        use_rope=cfg["model"]['use_rope'],
    )
    return model_config