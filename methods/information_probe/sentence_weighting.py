"""Prior-inspired CiCo weighting probe; not a claimed novel method."""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


def eot_vectors(tokens, mask):
    valid = mask == 1
    if not valid.any(-1).all():
        raise ValueError('Each text needs a valid token')
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = positions.masked_fill(~valid, -1).max(-1).values
    return F.normalize(tokens[torch.arange(len(tokens), device=tokens.device), last].float(), dim=-1)


def random_conditioning(query, bank, seed=42):
    """Input-only deterministic control; bank must contain TRAIN descriptors only."""
    bank_bytes = [x.tobytes() for x in bank.detach().float().cpu().numpy()]
    if len(set(bank_bytes)) < 2:
        raise ValueError('Conditioning bank needs distinct descriptors')
    indexes = []
    for row in query.detach().float().cpu().numpy():
        raw = row.tobytes()
        index = int.from_bytes(hashlib.sha256(str(seed).encode() + raw).digest()[:8], 'big') % len(bank)
        while bank_bytes[index] == raw:
            index = (index + 1) % len(bank)
        indexes.append(index)
    index = torch.tensor(indexes, device=bank.device)
    return bank[index], indexes


@torch.no_grad()
def frozen_channels_and_values(v, t, vm, tm, scale):
    v, t = F.normalize(v.float(), dim=-1), F.normalize(t.float(), dim=-1)
    affinity = torch.einsum('ifd,jld->ijfl', v, t)
    vv, tv = vm == 0, tm == 1
    if not vv.any(-1).all() or not tv.any(-1).all():
        raise ValueError('All-padding example')
    av = (affinity * (affinity / .07).softmax(-1)).sum(-1)
    at = (affinity * (affinity / .07).softmax(-2)).sum(-2)
    a = (av * vv[:, None]).sum(-1) / vv.sum(-1)[:, None]
    b = (at * tv[None]).sum(-1) / tv.sum(-1)[None]
    return scale * a, scale * b, av


class SentenceWeighting(nn.Module):
    def __init__(self, dim=512, hidden=32):
        super().__init__()
        self.key = nn.Linear(dim, hidden)
        self.query = nn.Linear(dim, hidden, bias=False)
        nn.init.zeros_(self.query.weight)
        self.input_scale = math.sqrt(dim)
        self.attention_scale = math.sqrt(hidden)

    def weights(self, video, video_mask, conditioning):
        valid = video_mask == 0
        if not valid.any(-1).all():
            raise ValueError('All-padding video')
        k = torch.tanh(self.key(F.normalize(video.float(), dim=-1) * self.input_scale))
        q = torch.tanh(self.query(conditioning.float() * self.input_scale))
        logits = torch.einsum('ifh,jh->ijf', k, q) / self.attention_scale
        logits = logits.masked_fill(~valid[:, None], -torch.inf)
        uniform = torch.zeros_like(logits).masked_fill(~valid[:, None], -torch.inf).softmax(-1)
        return logits.softmax(-1), uniform

    def forward(self, video, video_mask, conditioning, values, scale):
        weights, uniform = self.weights(video, video_mask, conditioning)
        return scale * ((weights - uniform) * values).sum(-1)
