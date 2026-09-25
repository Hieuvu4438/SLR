"""Minimal DistilBERT encoder (HF `DistilBertModel` semantics) so the seds env (torch 2.3) can run
the multilingual CLIP-aligned text tower without upgrading transformers. Weights load from the HF
safetensors file with identical parameter names. Verified against HF transformers 5.5 / torch 2.11
in the base env by tests/test_distilbert_min.py (max |diff| reported there)."""
import math

import torch
import torch.nn.functional as F
from torch import nn


class _Attn(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.h = h
        self.q_lin, self.k_lin, self.v_lin, self.out_lin = (nn.Linear(d, d) for _ in range(4))
        self.drop = nn.Dropout(p)

    def forward(self, x, mask):
        B, L, D = x.shape
        dh = D // self.h
        sp = lambda t: t.view(B, L, self.h, dh).transpose(1, 2)
        q, k, v = sp(self.q_lin(x)) / math.sqrt(dh), sp(self.k_lin(x)), sp(self.v_lin(x))
        s = q @ k.transpose(-1, -2)
        s = s.masked_fill(~mask[:, None, None, :], torch.finfo(s.dtype).min)
        a = self.drop(F.softmax(s, dim=-1))
        return self.out_lin((a @ v).transpose(1, 2).reshape(B, L, D))


class _FFN(nn.Module):
    def __init__(self, d, hd, p):
        super().__init__()
        self.lin1, self.lin2, self.drop = nn.Linear(d, hd), nn.Linear(hd, d), nn.Dropout(p)

    def forward(self, x):
        return self.drop(self.lin2(F.gelu(self.lin1(x))))


class _Block(nn.Module):
    def __init__(self, d, h, hd, p, ap):
        super().__init__()
        self.attention = _Attn(d, h, ap)
        self.sa_layer_norm = nn.LayerNorm(d, eps=1e-12)
        self.ffn = _FFN(d, hd, p)
        self.output_layer_norm = nn.LayerNorm(d, eps=1e-12)

    def forward(self, x, mask):
        x = self.sa_layer_norm(self.attention(x, mask) + x)
        return self.output_layer_norm(self.ffn(x) + x)


class _Emb(nn.Module):
    def __init__(self, vocab, d, maxpos, p):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab, d, padding_idx=0)
        self.position_embeddings = nn.Embedding(maxpos, d)
        self.LayerNorm = nn.LayerNorm(d, eps=1e-12)
        self.dropout = nn.Dropout(p)

    def forward(self, ids):
        pos = torch.arange(ids.shape[1], device=ids.device)[None]
        return self.dropout(self.LayerNorm(self.word_embeddings(ids) + self.position_embeddings(pos)))


class _Tr(nn.Module):
    def __init__(self, n, *a):
        super().__init__()
        self.layer = nn.ModuleList([_Block(*a) for _ in range(n)])


class DistilBertMin(nn.Module):
    def __init__(self, vocab_size=119547, dim=768, n_layers=6, n_heads=12, hidden_dim=3072,
                 max_position_embeddings=512, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.embeddings = _Emb(vocab_size, dim, max_position_embeddings, dropout)
        self.transformer = _Tr(n_layers, dim, n_heads, hidden_dim, dropout, attention_dropout)

    @classmethod
    def from_dir(cls, path):
        import json, os
        from safetensors.torch import load_file
        c = json.load(open(os.path.join(path, "config.json")))
        m = cls(c["vocab_size"], c["dim"], c["n_layers"], c["n_heads"], c["hidden_dim"],
                c["max_position_embeddings"], c["dropout"], c["attention_dropout"])
        missing, unexpected = m.load_state_dict(load_file(os.path.join(path, "model.safetensors")), strict=False)
        assert not missing, missing
        assert all("pooler" in k or "vocab" in k for k in unexpected), unexpected
        return m

    def forward(self, input_ids, attention_mask):
        mask = attention_mask.bool()
        x = self.embeddings(input_ids)
        for blk in self.transformer.layer:
            x = blk(x, mask)
        return x  # last_hidden_state [B, L, dim]
