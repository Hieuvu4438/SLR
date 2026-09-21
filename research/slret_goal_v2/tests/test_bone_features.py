import copy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'third_party/SEDS')]
from modules.modeling_gcn import ConvTemporalGraphical
from modules.modeling_graph import Graph
from methods.seds_adaptation.bone_features import BoneFeatures, attach_encoder_bones
from methods.seds_adaptation.masked_pose import masked_learning_rates


class Encoder(nn.Module):
    def __init__(self, layout):
        super().__init__()
        self.graph = Graph(layout, 'spatial', pad=0)
        self.register_buffer('A',torch.tensor(self.graph.A,dtype=torch.float32))
        block = nn.Module()
        block.gcn = ConvTemporalGraphical(2,8,self.A.shape[0])
        self.st_gcn_networks = nn.ModuleList([block])

    def forward(self, xy):
        b,t,v,_ = xy.shape
        x=xy.reshape(b*t,v,2).permute(0,2,1).unsqueeze(2)
        return self.st_gcn_networks[0].gcn(x,self.A)[0]


class BoneTest(unittest.TestCase):
    def test_descriptors_and_padding(self):
        m=BoneFeatures([0,0,1],8)
        xy=torch.tensor([[[[10.,10.],[13.,14.],[13.,14.]],
                          [[0.,0.],[13.,14.],[0.,0.]]]])
        d=m.descriptors(xy)
        self.assertTrue(torch.equal(d[0,0,0],torch.zeros(5)))
        torch.testing.assert_close(d[0,0,1],torch.tensor([3/256,4/256,5/256,.6,.8]))
        self.assertTrue(torch.equal(d[0,0,2],torch.zeros(5)))
        self.assertEqual(int(torch.count_nonzero(d[0,1])),0)
        self.assertTrue(torch.isfinite(m.descriptors(torch.zeros_like(xy))).all())
        # Translation changes native XY but not valid bone descriptors.
        torch.testing.assert_close(m.descriptors(xy[:,:1]+12),d[:,:1])
        bad=xy.clone();bad[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):m.descriptors(bad)

    def test_native_attachment_gradient_roundtrip(self):
        torch.manual_seed(42)
        for layout,v in (('stb',21),('body',7)):
            e=Encoder(layout); clone=copy.deepcopy(e)
            xy=torch.rand(2,3,v,2)*100+1
            expected=e(xy).detach()
            parents=attach_encoder_bones(e)
            self.assertEqual(parents[0],0)
            self.assertEqual(len(parents),v)
            if layout=='stb':self.assertEqual([parents[i] for i in (1,5,9,13,17)],[0]*5)
            else:self.assertEqual(parents,[0,0,1,2,0,4,5])
            self.assertTrue(torch.equal(e(xy),expected))
            e(xy).square().mean().backward()
            grad=e.bone_features.output.weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()),0)
            with torch.no_grad():e.bone_features.output.weight.add_(-1e-5*grad)
            actual=e(xy).detach()
            self.assertFalse(torch.equal(actual,expected))
            attach_encoder_bones(clone);clone.load_state_dict(e.state_dict(),strict=True)
            self.assertTrue(torch.equal(clone(xy),actual))
            # A second input cannot accidentally reuse the preceding descriptors.
            self.assertTrue(torch.equal(clone(xy+7),e(xy+7)))

    def test_learning_rates(self):
        m=nn.Module();m.signbert=nn.Module();m.signbert.embed=Encoder('body')
        attach_encoder_bones(m.signbert.embed);m.fusion=nn.Linear(8,2)
        opt=torch.optim.Adam(m.parameters())
        rates=masked_learning_rates(opt,m,control=True,bone_features=True)
        self.assertEqual({g['adaptation_group']:g['lr'] for g in rates},
                         dict(encoder=1e-6,fusion=1e-5,bone=1e-5))
        faster=torch.optim.Adam(m.parameters())
        rates=masked_learning_rates(faster,m,control=True,bone_features=True,bone_lr=1e-4)
        self.assertEqual({g['adaptation_group']:g['lr'] for g in rates},
                         dict(encoder=1e-6,fusion=1e-5,bone=1e-4))


if __name__=='__main__':unittest.main()
