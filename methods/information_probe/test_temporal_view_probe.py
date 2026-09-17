import numpy as np
import torch

from slr_common.data.views import uniform_indices, jittered_view
from methods.information_probe.temporal_view_probe import validate_indices


def test_registered_views_preserve_count_order_and_bounded_displacement():
    for n in (1, 32, 64, 65, 71, 128, 500):
        a = uniform_indices(n, 64)
        for seed in (42, 1337, 2026):
            b, independent = jittered_view(n, 64, global_seed=seed, video_id='test-id')
            va, vb = np.full(64, -1), np.full(64, -1)
            va[:len(a)], vb[:len(a)] = a, b.indices
            aa, bb = torch.tensor(va)[None], torch.tensor(vb)[None]
            changes = validate_indices(aa, bb, aa >= 0)
            assert bool(changes[0] > 0) == independent
            if n <= 64:
                assert not independent
