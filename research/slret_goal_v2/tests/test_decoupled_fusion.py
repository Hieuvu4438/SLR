import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from methods.seds_adaptation.decoupled_fusion import (
    decoupled_cross_entropy,paired_loss,native_cross_entropy,
    registered_native_subset,attach_decoupled_fusion,blend_fused_loss)


class NativeToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))
        self.dual_mix = .3
        self.freeze_exfusion = False
        self.rgb_pose_kl = False

    def get_similarity_logits(self,x):
        return self.scale*x,self.scale*x.T+.2*torch.eye(len(x))

    def forward(self,x):
        a,b = self.get_similarity_logits(x)
        if not self.training:
            return a,b
        fused = paired_loss(a,b,self.dual_mix,native_cross_entropy)
        branches = (a.square().mean(),b.square().mean(),0.,0.,a.mean())
        return fused+sum(branches),fused,*branches


class DCLTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_equation_and_singleton(self):
        x = torch.randn(4,4)
        expected = torch.stack([torch.logsumexp(torch.cat([x[i,:i],x[i,i+1:]]),0)-x[i,i]
                                for i in range(4)]).mean()
        torch.testing.assert_close(decoupled_cross_entropy(x),expected)
        with self.assertRaises(ValueError):decoupled_cross_entropy(torch.ones(1,1))

    def test_gradient_coupling_identity(self):
        x = (torch.randn(4,4)+3*torch.eye(4)).requires_grad_()
        ce = native_cross_entropy(x)
        dcl = decoupled_cross_entropy(x)
        g_ce, = torch.autograd.grad(ce,x,retain_graph=True)
        g_dcl, = torch.autograd.grad(dcl,x)
        q = -torch.expm1(torch.log_softmax(x,dim=1).diag())
        torch.testing.assert_close(g_ce,q[:,None]*g_dcl,atol=1e-7,rtol=1e-5)
        self.assertGreater((g_dcl-g_ce).abs().sum().item(),0)

    def test_extreme_easy_positive_is_finite_and_not_vanishing(self):
        x = (100*torch.eye(3)).requires_grad_()
        loss = decoupled_cross_entropy(x)
        self.assertLess(loss.item(),0)  # Negative DCL is normal, not failure.
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        torch.testing.assert_close(x.grad.diag(),torch.full((3,),-1/3))

    def test_native_inference_and_parameters_unchanged(self):
        m = NativeToy().eval()
        x = torch.randn(4,4)
        expected = m(x)
        params = copy.deepcopy(m.state_dict())
        rng = torch.get_rng_state().clone()
        attach_decoupled_fusion(m)
        for a,b in zip(m(x),expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(params.keys(),m.state_dict().keys())
        for k in params:torch.testing.assert_close(params[k],m.state_dict()[k],rtol=0,atol=0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertIsNone(m._dcl_pending)

    def test_only_fused_loss_replaced_and_gradients_match(self):
        m = NativeToy()
        x = torch.randn(4,4)
        old = m(x)
        a,b = m.get_similarity_logits(x)
        expected = paired_loss(a,b,m.dual_mix,decoupled_cross_entropy)+sum(old[2:])
        expected_grad, = torch.autograd.grad(expected,m.scale)
        attach_decoupled_fusion(m)
        actual = m(x)
        for a,b in zip(actual[2:],old[2:]):
            torch.testing.assert_close(torch.as_tensor(a),torch.as_tensor(b),rtol=0,atol=0)
        torch.testing.assert_close(actual[0],expected)
        actual[0].backward()
        torch.testing.assert_close(m.scale.grad,expected_grad)
        self.assertTrue(m.dcl_stats['native_fusion_reconstruction_passed'])
        self.assertIsNone(m._dcl_pending)
        self.assertFalse(m._dcl_in_forward)

    def test_subset_loads_only_native_samples_in_registered_order(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'reference.json'
            ids = [f'train_{i}' for i in range(600)]
            selected = ids[5:517][::-1]
            p.write_text(json.dumps(dict(status='completed',signrep_data=dict(train_ids=selected))))
            subset,actual = registered_native_subset(list(range(600)),ids,p)
            self.assertEqual(actual,selected)
            self.assertEqual(subset[0],516)
            self.assertEqual(subset[-1],5)

    def test_blend_gradient_and_endpoints(self):
        x = (torch.randn(4,4)+3*torch.eye(4)).requires_grad_()
        ce,dcl = native_cross_entropy(x),decoupled_cross_entropy(x)
        self.assertIs(blend_fused_loss(ce,dcl,0),ce)
        self.assertIs(blend_fused_loss(ce,dcl,1),dcl)
        actual, = torch.autograd.grad(blend_fused_loss(ce,dcl,.05),x,retain_graph=True)
        gdcl, = torch.autograd.grad(dcl,x)
        q = -torch.expm1(torch.log_softmax(x,dim=1).diag())
        torch.testing.assert_close(actual,(.05+.95*q[:,None])*gdcl,atol=1e-7,rtol=1e-5)
        for bad in [-1,2,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):blend_fused_loss(ce,dcl,bad)

    def test_partial_blend_preserves_native_branches(self):
        m = NativeToy()
        x = torch.randn(4,4)
        old = m(x)
        a,b = m.get_similarity_logits(x)
        expected = .95*old[1]+.05*paired_loss(a,b,m.dual_mix,decoupled_cross_entropy)
        attach_decoupled_fusion(m,weight=.05)
        actual = m(x)
        torch.testing.assert_close(actual[1],expected)
        torch.testing.assert_close(actual[0],expected+sum(old[2:]))
        for a,b in zip(actual[2:],old[2:]):
            torch.testing.assert_close(torch.as_tensor(a),torch.as_tensor(b),rtol=0,atol=0)
        self.assertEqual(m.dcl_stats['dcl_weight'],.05)
        self.assertGreaterEqual(m.dcl_stats['effective_q_mean'],.05)


if __name__=='__main__':unittest.main()
