"""Unit tests for SignMatchHead + rerank (CPU)."""
import sys
import numpy as np
import torch
sys.path[:0] = ["/home/haipd/SLR/third_party/SEDS", "/home/haipd/SLR/methods/sota_slret/src"]
from method_impls import SignMatchHead, rerank

def test_shapes_grad_padding_invariance():
    torch.manual_seed(0)
    h = SignMatchHead(layers=2).eval()
    B = 4
    t = torch.randn(B, 32, 512, requires_grad=True); tv = torch.zeros(B, 32, dtype=torch.bool); tv[:, :10] = True
    vp = torch.randn(B, 65, 512, requires_grad=True); vr = torch.randn(B, 65, 512)
    vv = torch.zeros(B, 65, dtype=torch.bool); vv[:, :40] = True
    mem, mv = h.build_memory(vp, vr, None, vv)
    assert mem.shape == (B, 130, 512) and mv.shape == (B, 130)
    y = h(t, tv, mem, mv)
    assert y.shape == (B,) and torch.isfinite(y).all()
    y.sum().backward()
    assert t.grad.abs().sum() > 0 and vp.grad.abs().sum() > 0
    assert t.grad[:, 10:].abs().max() == 0 and vp.grad[:, 40:].abs().max() == 0  # padded tokens get no grad
    t2 = t.detach().clone(); t2[:, 10:] = 1e3 * torch.randn_like(t2[:, 10:])
    vp2 = vp.detach().clone(); vp2[:, 40:] = 1e3
    mem2, _ = h.build_memory(vp2, vr, None, vv)
    assert torch.allclose(h(t2, tv, mem2, mv), y.detach(), atol=1e-4)

def test_rerank_places_topk_first_and_reorders():
    S = np.array([[3.0, 0.0], [2.0, 1.0], [0.0, 2.0]])  # 3 videos x 2 texts
    art = {"top_v": np.array([[0, 1], [2, 1]]), "l_t2v": np.array([[-10.0, 10.0], [0.0, 0.0]]),
           "top_t": np.array([[0, 1], [0, 1], [1, 0]]), "l_v2t": np.zeros((3, 2))}
    St, Sv = rerank(S, art, k=2, w=1.0)
    assert np.argmax(St[:, 0]) == 1           # text0: itm flips video0 -> video1
    assert St[2, 0] < St[0, 0]                # non-top-k stays below top-k
    St0, _ = rerank(S, art, k=0, w=1.0)
    assert np.array_equal(St0, S)
