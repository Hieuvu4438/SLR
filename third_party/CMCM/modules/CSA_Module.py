import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalBackdoorAdjuster(nn.Module):
    def __init__(self, adjust_dim=2048):
        super().__init__()
        self.adjust_net = nn.Sequential(
            nn.Linear(adjust_dim * 2, adjust_dim),
            nn.ReLU(),
            nn.LayerNorm(adjust_dim),
            nn.Linear(adjust_dim, adjust_dim)
        ).to(DEVICE)  # 显式上移 GPU

    def forward(self, original_feat, augmented_feat):
        combined = torch.cat([original_feat, augmented_feat], dim=1)  # (B, 2D)
        adjust_factor = self.adjust_net(combined)  # (B, D)
        stable_feat = augmented_feat - adjust_factor * (augmented_feat - original_feat)
        return stable_feat
