import copy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'third_party/SEDS')]
from modules.optimization import BertAdam
from methods.seds_adaptation.joint_bilinear import JointBilinear, attach_encoder_bilinear
from methods.seds_adaptation.masked_pose import masked_learning_rates


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph = SimpleNamespace(seqlen=1, part=[[0, 1, 2], [0, 3]])
        self.fc_out = 256
        self.project = nn.Linear(2, 256)

    def graph_max_pool(self, x, p, stride=None):
        return F.max_pool2d(x, p, stride)

    def forward(self, xy):
        x = self.project(xy).flatten(0, 1).transpose(1, 2).unsqueeze(2)
        return torch.cat([self.graph_max_pool(x[..., p], (1, len(p)))
                          for p in self.graph.part], -1)


class BilinearTest(unittest.TestCase):
    def test_centered_moment_and_missing_singleton(self):
        torch.manual_seed(42)
        m = JointBilinear(8,4,centered=True)
        z = torch.randn(3,5,4,requires_grad=True)
        valid = torch.tensor([[True]*5,[False]*5,[True,False,False,False,False]])
        moment = m.second_moment(z,valid)
        centered = z[0]-z[0].mean(0)
        torch.testing.assert_close(moment[0],centered.T @ centered / 5)
        self.assertEqual(torch.count_nonzero(moment[1:]).item(),0)
        torch.testing.assert_close(moment,m.second_moment(z+5,valid),atol=1e-6,rtol=1e-5)
        moment.square().sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        x = torch.randn(3,8,1,5)
        with torch.no_grad():m.output.weight.normal_(std=.01)
        m(x,valid).square().sum().backward()
        self.assertGreater(m.input.weight.grad.abs().sum().item(),0)
        self.assertTrue(torch.isfinite(m.input.weight.grad).all())
        self.assertEqual(torch.count_nonzero(m(x,valid)[1:]).item(),0)

    def test_centered_attachment_identity_roundtrip(self):
        torch.manual_seed(42)
        e = Encoder();clone = copy.deepcopy(e)
        xy = torch.rand(2,3,4,2)*100+1
        expected = e(xy).detach()
        attach_encoder_bilinear(e,centered=True)
        self.assertTrue(torch.equal(expected,e(xy)))
        e(xy).square().mean().backward()
        with torch.no_grad():e.joint_bilinear.output.weight.add_(-1e-4*e.joint_bilinear.output.weight.grad)
        attach_encoder_bilinear(clone,centered=True);clone.load_state_dict(e.state_dict(),strict=True)
        self.assertTrue(torch.equal(clone(xy),e(xy)))

    def test_native_warmup_delays_input_activation_until_three(self):
        torch.manual_seed(42)
        m = JointBilinear(8,4)
        x = torch.randn(3,8,1,5)
        valid = torch.ones(3,5,dtype=torch.bool)
        initial = m.input.weight.detach().clone()
        opt = BertAdam(m.parameters(),lr=1e-4,warmup=.1,t_total=666,weight_decay=0.)
        for step in range(1,4):
            opt.zero_grad();(m(x,valid)+1).square().mean().backward()
            if step <= 2:self.assertEqual(torch.count_nonzero(m.input.weight.grad).item(),0)
            else:self.assertGreater(m.input.weight.grad.abs().sum().item(),0)
            opt.step()
            if step == 1:self.assertEqual(torch.count_nonzero(m.output.weight).item(),0)
            else:self.assertGreater(m.output.weight.abs().sum().item(),0)
            if step <= 2:self.assertTrue(torch.equal(m.input.weight,initial))
            else:self.assertFalse(torch.equal(m.input.weight,initial))

    def test_zero_identity_missing_permutation_and_finite_gradient(self):
        torch.manual_seed(42)
        m = JointBilinear(8, 4)
        x = torch.randn(3, 8, 1, 5, requires_grad=True)
        valid = torch.tensor([[True]*5, [False]*5, [True, False, True, False, True]])
        self.assertEqual(torch.count_nonzero(m(x, valid)).item(), 0)
        desc = m.descriptor(x, valid)
        self.assertEqual(torch.count_nonzero(desc[1]).item(), 0)
        perm = torch.tensor([4, 1, 3, 0, 2])
        torch.testing.assert_close(desc, m.descriptor(x[..., perm], valid[:, perm]))
        altered = x.detach().clone().permute(0, 3, 2, 1)
        altered[~valid] = 10000
        torch.testing.assert_close(desc, m.descriptor(altered.permute(0, 3, 2, 1), valid))
        with torch.no_grad():m.output.weight.normal_(std=.01)
        m(x, valid).square().sum().backward()
        for p in m.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_captures_joint_coactivation_not_only_max(self):
        m = JointBilinear(4, 4)
        with torch.no_grad():m.input.weight.copy_(torch.eye(4))
        x = torch.tensor([[[[1.,0.]],[[0.,1.]],[[0.,0.]],[[0.,0.]]]])
        y = torch.tensor([[[[1.,0.]],[[1.,0.]],[[0.,0.]],[[0.,0.]]]])
        self.assertTrue(torch.equal(x.amax(-1), y.amax(-1)))
        self.assertFalse(torch.allclose(m.descriptor(x, torch.ones(1,2,dtype=torch.bool)),
                                        m.descriptor(y, torch.ones(1,2,dtype=torch.bool))))

    def test_attachment_activation_roundtrip_and_optimizer(self):
        torch.manual_seed(42)
        e = Encoder(); clone = copy.deepcopy(e)
        xy = torch.rand(2,3,4,2)*100+1
        expected = e(xy).detach()
        attach_encoder_bilinear(e)
        self.assertTrue(torch.equal(expected, e(xy)))
        opt = torch.optim.Adam(e.parameters(), lr=1e-4)
        for _ in range(2):
            opt.zero_grad();e(xy).square().mean().backward();opt.step()
        self.assertGreater(e.joint_bilinear.input.weight.grad.abs().sum().item(), 0)
        attach_encoder_bilinear(clone);clone.load_state_dict(e.state_dict(), strict=True)
        self.assertTrue(torch.equal(clone(xy), e(xy)))
        self.assertTrue(torch.equal(clone(xy+3), e(xy+3)))
        m = nn.Module();m.signbert = nn.Module();m.signbert.embed = e;m.fusion = nn.Linear(8,2)
        groups = masked_learning_rates(torch.optim.Adam(m.parameters()),m,control=True,joint_bilinear=True)
        self.assertEqual({g['adaptation_group']:g['lr'] for g in groups},
                         dict(encoder=1e-6,fusion=1e-5,bilinear=1e-4))


if __name__ == '__main__':unittest.main()
