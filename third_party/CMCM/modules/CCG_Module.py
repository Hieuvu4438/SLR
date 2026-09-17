import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.register_buffer("causal_mask", torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024))

    def forward(self, x):
        B, T, D = x.shape
        assert D == self.embed_dim, "Feature dimension mismatch"

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D/H)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D/H)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D/H)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T, T)
        attn_scores = attn_scores.masked_fill(self.causal_mask[:T, :T] == 0, -1e9)  # 因果掩码
        attn_probs = F.softmax(attn_scores, dim=-1)  # (B, H, T, T)

        attn_out = attn_probs @ v  # (B, H, T, D/H)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)

        return self.out_proj(attn_out) + x


class GaussianParameterization(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mu_proj = nn.Linear(input_dim, output_dim)
        self.sigma_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Softplus()
        )

    def forward(self, x):
        mu = self.mu_proj(x)
        sigma_sq = self.sigma_proj(x)
        return mu, sigma_sq


class GaussianAlignmentModule(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, align_dim=128):
        super().__init__()
        self.causal_attn = CausalAttention(embed_dim, num_heads)
        self.video_gaussian = GaussianParameterization(embed_dim, align_dim)
        self.cross_modal_gaussian = GaussianParameterization(embed_dim, align_dim)

    def forward(self, video_feat, cross_modal_feat):

        cross_modal_attn = self.causal_attn(cross_modal_feat)  # (B, T, D)
        mu_v, sigma_sq_v = self.video_gaussian(video_feat)  # (B, T, K)
        mu_t, sigma_sq_t = self.cross_modal_gaussian(cross_modal_attn)  # (B, T, K)
        kl_loss = 0.5 * (
                (sigma_sq_t / (sigma_sq_v + 1e-8)).log() +
                (sigma_sq_v + (mu_v - mu_t) ** 2) / (sigma_sq_t + 1e-8) - 1
        )
        kl_loss = kl_loss.mean(dim=1)
        aligned_loss = kl_loss.mean()

        return aligned_loss
