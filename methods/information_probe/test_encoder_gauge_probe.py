import torch
from .encoder_gauge_probe import fit_orthogonal
from .scoring import channels


def test_procrustes_recovery_and_shared_score_invariance():
    torch.manual_seed(4)
    x = torch.randn(100, 8, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
    fit, audit = fit_orthogonal(x, x@q)
    assert torch.allclose(fit, q, atol=1e-12, rtol=0)
    assert audit['orthogonality_max_error'] < 1e-12
    v, t = torch.randn(3, 5, 8), torch.randn(3, 4, 8)
    vm, tm = torch.zeros(3, 5), torch.ones(3, 4)
    a, b = channels(v, t, vm, tm, 100.)
    aa, bb = channels(v@fit.float(), t@fit.float(), vm, tm, 100.)
    assert torch.allclose(a, aa, atol=3e-5, rtol=0)
    assert torch.allclose(b, bb, atol=3e-5, rtol=0)
