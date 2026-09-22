"""Attach a training-only gloss CTC head to native SEDS visual tokens.

Gloss targets are supplied only from TRAIN annotations. The native video/text
score matrix and the evaluation path are unchanged.
"""

from __future__ import annotations

from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


class GlossCTC(nn.Module):
    def __init__(self, width: int, vocab_size: int) -> None:
        super().__init__()
        if width < 1 or vocab_size < 1:
            raise ValueError("invalid gloss-head shape")
        self.classifier = nn.Linear(width, vocab_size + 1)  # index 0 is CTC blank
        self.targets: list[list[int]] | None = None
        self.loss: torch.Tensor | None = None
        self.stats: dict[str, float | int] = {}

    def set_targets(self, targets: list[list[int]]) -> None:
        if not targets or any(not row for row in targets):
            raise ValueError("every training row needs a non-empty gloss sequence")
        self.targets = targets

    def forward(self, pose_tokens: torch.Tensor, video_mask: torch.Tensor) -> torch.Tensor:
        if pose_tokens.ndim != 3 or video_mask.shape != pose_tokens.shape[:2]:
            raise ValueError("pose token/mask shape mismatch")
        if self.targets is None or len(self.targets) != pose_tokens.shape[0]:
            raise ValueError("gloss targets missing or batch size mismatch")
        # Native SEDS mask includes a CLS token at index zero; zero means valid.
        valid = (video_mask[:, 1:] == 0)
        if not torch.all(valid[:, 0]) or torch.any(valid[:, 1:] & ~valid[:, :-1]):
            raise ValueError("expected left-aligned nonempty valid sign windows")
        lengths = valid.sum(-1).long()
        target_lengths = torch.tensor([len(row) for row in self.targets],
                                      device=pose_tokens.device, dtype=torch.long)
        repeats = torch.tensor([sum(a == b for a, b in zip(row, row[1:]))
                                for row in self.targets], device=pose_tokens.device)
        eligible = lengths >= target_lengths + repeats
        if not torch.all(eligible):
            raise ValueError(f"CTC-ineligible TRAIN examples: {int((~eligible).sum())}")
        target = torch.tensor([item for row in self.targets for item in row],
                              device=pose_tokens.device, dtype=torch.long)
        if target.min() < 1 or target.max() >= self.classifier.out_features:
            raise ValueError("gloss target outside TRAIN vocabulary")
        logits = self.classifier(pose_tokens[:, 1:]).float().log_softmax(-1).transpose(0, 1)
        per_example = F.ctc_loss(logits, target, lengths, target_lengths, blank=0,
                                 reduction="none", zero_infinity=False)
        if not torch.isfinite(per_example).all():
            raise FloatingPointError("non-finite gloss CTC")
        self.loss = (per_example / target_lengths).mean()
        self.stats = {"ctc_loss": float(self.loss.detach()),
                      "mean_input_length": float(lengths.float().mean()),
                      "mean_target_length": float(target_lengths.float().mean()),
                      "eligible": int(eligible.sum())}
        return self.loss


def attach_gloss_ctc(model: nn.Module, vocab_size: int) -> GlossCTC:
    prototype = next(model.fusion.parameters())
    head = GlossCTC(width=512, vocab_size=vocab_size).to(prototype.device)
    model.add_module("gloss_ctc", head)
    original_visual = model.get_visual_output
    original_forward = model.forward

    def visual_with_gloss(self, *args, **kwargs):
        output = original_visual(*args, **kwargs)
        if self.training and head.targets is not None:
            video_mask, pose_tokens, _rgb_tokens = output
            head(pose_tokens, video_mask)
        return output

    def forward_with_gloss(self, *args, **kwargs):
        head.loss = None
        result = original_forward(*args, **kwargs)
        if self.training:
            if not isinstance(result, tuple) or len(result) != 7 or head.loss is None:
                raise ValueError("native seven losses or gloss auxiliary missing")
            aux = head.loss
            head.loss = None
            head.targets = None
            return (*result, aux)
        return result

    model.get_visual_output = MethodType(visual_with_gloss, model)
    model.forward = MethodType(forward_with_gloss, model)
    return head
