"""C20: annealed train-only dropout of the stronger RGB fusion stream.

This is a narrow adaptation of modality dropout, not an inference module.  A
private RNG prevents the intervention from changing native dropout masks.  The
probability decays linearly to zero so the final update uses complete inputs.
"""
from types import MethodType

import torch
from torch import nn


class AnnealedRGBDropout(nn.Module):
    def __init__(self, start_probability=.2, seed=42):
        super().__init__()
        if not 0 < start_probability <= 1:
            raise ValueError('Start probability must be in (0,1]')
        self.start_probability = float(start_probability)
        self.seed = int(seed)
        self.probability = self.start_probability
        self.step = 1
        self.total_steps = 1
        self.generator = None
        self.generator_device = None
        self.dropped_total = 0
        self.seen_total = 0
        self.stats = {}
        self.eval_identity_passed = False

    def set_step(self, step, total_steps):
        if total_steps < 2 or not 1 <= step <= total_steps:
            raise ValueError('Expected 1 <= step <= total_steps with at least two steps')
        self.step = int(step)
        self.total_steps = int(total_steps)
        progress = (step - 1) / (total_steps - 1)
        self.probability = self.start_probability * (1 - progress)

    def _generator(self, device):
        key = str(device)
        if self.generator is None:
            self.generator = torch.Generator(device=device).manual_seed(self.seed)
            self.generator_device = key
        elif self.generator_device != key:
            raise ValueError('C20 generator device changed within one run')
        return self.generator

    def forward(self, rgb):
        if rgb.ndim != 3:
            raise ValueError('Expected RGB fusion features [B,T,D]')
        if not self.training:
            self.eval_identity_passed = True
            return rgb
        probability = float(self.probability)
        if not 0 <= probability <= self.start_probability:
            raise ValueError('Invalid annealed probability')
        if probability == 0:
            dropped = torch.zeros(rgb.shape[0], dtype=torch.bool, device=rgb.device)
        else:
            dropped = torch.rand(rgb.shape[0], device=rgb.device,
                                 generator=self._generator(rgb.device)) < probability
        result = rgb.masked_fill(dropped[:, None, None], 0)
        count = int(dropped.sum())
        self.dropped_total += count
        self.seen_total += rgb.shape[0]
        self.stats = dict(rgb_drop_probability=probability,
            rgb_dropped_samples=count,rgb_batch_samples=int(rgb.shape[0]),
            rgb_drop_fraction=count / max(1, rgb.shape[0]),
            rgb_dropped_total=self.dropped_total,rgb_seen_total=self.seen_total,
            rgb_moddrop_step=self.step,rgb_moddrop_total_steps=self.total_steps)
        return result


def attach_annealed_rgb_dropout(model, start_probability=.2, seed=42):
    fusion = model.fusion
    if hasattr(fusion, 'rgb_moddrop'):
        raise ValueError('C20 already attached')
    module = AnnealedRGBDropout(start_probability, seed)
    module.train(fusion.training)
    fusion.add_module('rgb_moddrop', module)
    original = fusion.forward

    def forward(self, pose, rgb, mask):
        return original(pose, self.rgb_moddrop(rgb), mask)

    fusion.forward = MethodType(forward, fusion)
    # The module has no parameters or buffers, so resulting checkpoints remain
    # directly loadable by the unmodified inference model.
    return module
