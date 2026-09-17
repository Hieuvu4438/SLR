import numpy as np

from .common import metrics
from .seed_ensemble_probe import bootstrap_delta, mean_r1_from_counts, query_counts, uniform_mean


def test_repeated_mean_exact_and_elementwise():
    rng = np.random.default_rng(4)
    a, b = (rng.normal(size=(8, 8)).astype(np.float32) for _ in range(2))
    assert np.array_equal(uniform_mean([a]*3), a)
    expected = ((a.astype(float)*2+b)/3).astype(np.float32)
    assert np.array_equal(uniform_mean([a, b, a]), expected)
    assert uniform_mean([a, b, a])[2, 3] == uniform_mean([a[2:3, 3:4], b[2:3, 3:4], a[2:3, 3:4]])[0, 0]


def test_official_tie_counts_and_bootstrap():
    s = np.array([[2, 2, 0], [2, 2, 3], [0, 1, 2]], dtype=np.float32)
    assert abs(mean_r1_from_counts(query_counts(s).sum(0))-metrics(s)['official_mean_R1']) < 1e-12
    ci = bootstrap_delta(s, s, ['a', 'a', 'b'], draws=50)
    assert ci['mean_pp'] == ci['lower'] == ci['upper'] == 0
