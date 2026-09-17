import numpy as np
from .stream_fusion_probe import mix_stream_scores


def test_stream_mix_uses_fixed_weights_and_preserves_orientation():
    a = np.array([[1, 3], [5, 7]], dtype=np.float32)
    b = np.array([[4, 2], [8, 6]], dtype=np.float32)
    assert np.allclose(mix_stream_scores(a, b), [[3.7, 2.1], [7.7, 6.1]])
    assert np.array_equal(mix_stream_scores(a, a), a)
    shifted = mix_stream_scores(np.roll(a, 1, axis=0), b)
    assert not np.array_equal(shifted, mix_stream_scores(a, b))
