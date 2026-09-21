"""Ordinary label smoothing for native square paired contrastive objectives.

Engineering candidate C05. No changed positives, mining, memory or teacher.
The coefficient applies only during training; scorer and evaluator are untouched.
"""
import torch
from torch import nn


class SmoothedCrossEn(nn.Module):
    def __init__(self, epsilon=.05):
        super().__init__()
        if not 0 <= epsilon < 1:
            raise ValueError('Smoothing must lie in [0,1)')
        self.epsilon = epsilon

    def forward(self, scores):
        if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
            raise ValueError('Native paired loss requires a square score matrix')
        log_probs = torch.log_softmax(scores,dim=-1)
        paired = -torch.diag(log_probs).mean()
        if self.epsilon == 0:
            return paired
        uniform = -log_probs.mean()
        return (1-self.epsilon)*paired+self.epsilon*uniform
