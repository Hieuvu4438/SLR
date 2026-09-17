import torch
import torch.nn as nn
import torchvision.models as models


class VideoEncoder(nn.Module):
    def __init__(self, pretrained_i3d_path=None):
        super().__init__()

        self.i3d_encoder = models.video.r2plus1d_18(pretrained=True)
        self.trainable_encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.AdaptiveAvgPool3d((None, 1, 1))
        )
        self.trainable_proj = nn.Linear(128, 2048)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, video):
        i3d_feat = self.i3d_encoder(video)
        trainable_feat = self.trainable_encoder(video)
        trainable_feat = trainable_feat.flatten(2).squeeze(-1).squeeze(-1)
        trainable_feat = self.trainable_proj(trainable_feat)

        fused_feat = self.alpha * i3d_feat + self.beta * trainable_feat
        return fused_feat