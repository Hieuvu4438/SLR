import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'research/slret_goal/tools'), str(ROOT/'third_party/SEDS')]
from seds_fusion_layout import double_permutation_counterexample
from modules.module_fusionencoder import DeformableMultiHeadedAttention


def explicit_reference(layer, k, v, q, mask):
    b, length, _ = k.shape
    h, d, n = layer.num_heads, layer.head_size, layer.num_keys
    k, v, q = layer.k_layer(k), layer.v_layer(v), layer.q_layer(q)
    offsets = layer.sample_offsets(q).reshape(b, -1, h, n).transpose(1, 2)
    left = -n//2
    reference = torch.arange(left, n+left, dtype=q.dtype)
    centers = torch.arange(q.shape[1], dtype=q.dtype).unsqueeze(-1)
    lengths = (1-mask).sum(-1).squeeze().float()-1
    locations = (reference+offsets+centers) % lengths[:, None, None, None]
    locations = locations/(length-1)*2-1
    grid = torch.stack([locations, torch.ones_like(locations)],-1).reshape(b*h,q.shape[1],n,2)
    # Explicit axes: [B,T,H,Dh] -> [B,H,Dh,1,T]. All other native conventions retained.
    def sample(x):
        image = x.reshape(b,length,h,d).permute(0,2,3,1).reshape(b*h,d,1,length)
        return F.grid_sample(image,grid,mode='bilinear',padding_mode='zeros',align_corners=False
                             ).reshape(b,h,d,q.shape[1],n).permute(0,1,3,4,2)
    sk, sv = sample(k), sample(v)
    qh = q.reshape(b,q.shape[1],h,d).transpose(1,2)/(d**.5)
    attn = layer.dropout(torch.softmax((qh.unsqueeze(3)*sk).sum(-1),dim=-1))
    context = (attn.unsqueeze(-1)*sv).sum(-2).transpose(1,2).reshape(b,q.shape[1],h*d)
    return layer.output_layer(context)


class FusionLayoutTests(unittest.TestCase):
    def test_tagged_axes(self):
        x = torch.arange(2*7*8).reshape(2,7,8)
        fixed = double_permutation_counterexample(x,2).reshape(2,2,1,7,4)
        expected = x.reshape(2,7,2,4).permute(0,2,1,3).unsqueeze(2)
        self.assertTrue(torch.equal(fixed,expected))
        self.assertFalse(torch.equal(x.reshape(2,2,1,7,4),expected))

    def test_attention_and_gradients(self):
        torch.manual_seed(42)
        layer = DeformableMultiHeadedAttention(2,8,dropout=0,num_keys=3).eval()
        inputs = [torch.randn(2,7,8,requires_grad=True) for _ in range(3)]
        mask = torch.zeros(2,7); mask[:,0] = 1
        expected = explicit_reference(layer,*inputs,mask)
        reference_grad = torch.autograd.grad(expected.sum(),inputs,retain_graph=True)
        native = layer(*inputs,mask)
        native_grad = torch.autograd.grad(native.sum(),inputs,retain_graph=True)
        torch.testing.assert_close(native,expected,atol=1e-6,rtol=1e-6)
        for a,b in zip(native_grad,reference_grad):
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        hooks = [p.register_forward_hook(lambda _m,_a,out: double_permutation_counterexample(out,2))
                 for p in [layer.k_layer,layer.v_layer]]
        actual = layer(*inputs,mask)
        for handle in hooks: handle.remove()
        self.assertFalse(torch.allclose(actual,expected,atol=1e-5,rtol=1e-5))

    def test_single_head_identity_and_invalid(self):
        x = torch.randn(2,7,8)
        self.assertTrue(torch.equal(double_permutation_counterexample(x,1),x))
        with self.assertRaises(ValueError): double_permutation_counterexample(x,3)


if __name__ == '__main__': unittest.main()
