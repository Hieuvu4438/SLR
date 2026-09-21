import copy
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from methods.seds_adaptation.global_exchange import (GlobalExchange,
    attach_global_exchange, configure_exchange_trainability, exchange_learning_rates)
from methods.seds_adaptation.signrep_transfer import transfer_state, load_transfer_state


class ToyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(512))

    def forward(self, pose, rgb, mask):
        return (pose + rgb) * self.weight


class ExchangeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.module = GlobalExchange(width=8, rank=4, slots=2)
        self.p = torch.randn(2, 6, 8)
        self.r = torch.randn(2, 6, 8)
        self.mask = torch.tensor([[0,0,0,0,1,1],[0,0,0,0,0,1]],dtype=torch.bool)

    def activate(self, module=None):
        module = module or self.module
        with torch.no_grad():
            for d in (module.to_pose, module.to_rgb):
                d.output.weight.normal_(std=.1)

    def test_identity_and_no_input_mutation(self):
        p, r = self.module(self.p, self.r, self.mask)
        self.assertTrue(torch.equal(p, self.p) and torch.equal(r, self.r))
        self.assertTrue(self.module.initial_identity_passed)
        self.assertIsNot(p, self.p)

    def test_padding_and_cls_do_not_influence_clips(self):
        self.activate()
        a = self.module(self.p, self.r, self.mask)
        p, r = self.p.clone(), self.r.clone()
        excluded = self.mask.clone(); excluded[:,0] = True
        p[excluded] = 900; r[excluded] = -700
        b = self.module(p, r, self.mask)
        for before, after, original in zip(a,b,(p,r)):
            torch.testing.assert_close(before[~excluded],after[~excluded],rtol=0,atol=0)
            self.assertTrue(torch.equal(after[excluded],original[excluded]))

    def test_empty_and_singleton_clips_finite(self):
        self.activate()
        mask = torch.ones_like(self.mask); mask[:,0] = False; mask[1,1] = False
        outputs = self.module(self.p,self.r,mask)
        self.assertTrue(all(torch.isfinite(x).all() for x in outputs))
        self.assertTrue(torch.equal(outputs[0][0],self.p[0]))

    def test_cross_source_changes_target_but_within_does_not(self):
        self.activate()
        other = self.r.clone(); other[:,3,:] += torch.arange(8)*2
        a = self.module(self.p,self.r,self.mask)[0]
        b = self.module(self.p,other,self.mask)[0]
        self.assertGreater(float((a[:,1]-b[:,1]).abs().sum()),0)
        self.module.mode = 'within'
        a = self.module(self.p,self.r,self.mask)[0]
        b = self.module(self.p,other,self.mask)[0]
        self.assertTrue(torch.equal(a,b))

    def test_pose_to_rgb_preserves_pose_and_freezes_reverse_direction(self):
        module = GlobalExchange(width=8,rank=4,slots=2,mode='pose_to_rgb')
        self.activate(module)
        pose, rgb = module(self.p,self.r,self.mask)
        self.assertTrue(torch.equal(pose,self.p))
        self.assertGreater(float((rgb-self.r).abs().sum()),0)
        self.assertTrue(all(not p.requires_grad for p in module.to_pose.parameters()))
        self.assertTrue(all(p.requires_grad for p in module.to_rgb.parameters()))

    def test_independent_video_encoding(self):
        self.activate()
        a = self.module(self.p,self.r,self.mask)
        b = self.module(self.p[:1],self.r[:1],self.mask[:1])
        for x,y in zip(a,b):torch.testing.assert_close(x[:1],y)

    def test_all_parameters_receive_gradients_after_output_activation(self):
        target = torch.randn_like(self.p)
        opt = torch.optim.SGD(self.module.parameters(), lr=.1)
        for step in range(2):
            p,r = self.module(self.p,self.r,self.mask)
            ((p-target).square().mean()+(r+target).square().mean()).backward()
            if step == 1:
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    and p.grad.abs().sum()>0 for p in self.module.parameters()))
            opt.step(); opt.zero_grad()

    def test_attachment_and_fresh_checkpoint_reconstruction(self):
        model = nn.Module(); model.fusion = ToyFusion()
        restored = copy.deepcopy(model)
        p,r = torch.randn(2,6,512),torch.randn(2,6,512)
        native = model.fusion(p,r,self.mask)
        attach_global_exchange(model)
        self.assertTrue(torch.equal(model.fusion(p,r,self.mask),native))
        self.activate(model.fusion.global_exchange)
        delta = transfer_state(model)
        attach_global_exchange(restored)
        load_transfer_state(restored,delta)
        torch.testing.assert_close(model.fusion(p,r,self.mask),restored.fusion(p,r,self.mask),rtol=0,atol=0)
        self.assertFalse(restored.fusion.global_exchange.initial_identity_passed)
        with self.assertRaises(ValueError):attach_global_exchange(model)

    def test_learning_rates_preserve_old_parameters_and_options(self):
        model = nn.Module(); model.fusion = ToyFusion(); attach_global_exchange(model)
        group = dict(params=list(model.parameters()),lr=1e-5,weight_decay=.2,adaptation_group='fusion')
        opt = SimpleNamespace(param_groups=[group])
        exchange_learning_rates(opt,model)
        self.assertEqual({g['lr'] for g in opt.param_groups},{1e-5,1e-4})
        params = [p for g in opt.param_groups for p in g['params']]
        self.assertEqual(len(params),len(set(map(id,params))))
        self.assertEqual(set(map(id,params)),set(map(id,model.parameters())))
        self.assertTrue(all(g['weight_decay']==.2 for g in opt.param_groups))

    def test_asymmetric_learning_rates_exclude_frozen_direction(self):
        model = nn.Module(); model.fusion = ToyFusion(); attach_global_exchange(model,'pose_to_rgb')
        # Reproduce the generic fusion policy that caused the first C19-R1 run
        # to re-enable the deliberately inactive reverse direction.
        model.fusion.requires_grad_(True)
        count = configure_exchange_trainability(model)
        self.assertEqual(count,131328)
        trainable = [p for p in model.parameters() if p.requires_grad]
        group = dict(params=trainable,lr=1e-5,weight_decay=.2,adaptation_group='fusion')
        opt = SimpleNamespace(param_groups=[group])
        exchange_learning_rates(opt,model)
        params = [p for g in opt.param_groups for p in g['params']]
        self.assertEqual(set(map(id,params)),set(map(id,trainable)))
        frozen = set(map(id,model.fusion.global_exchange.to_pose.parameters()))
        self.assertTrue(frozen.isdisjoint(map(id,params)))

    def test_asymmetric_trainable_parameters_all_receive_gradients(self):
        module = GlobalExchange(width=8,rank=4,slots=2,mode='pose_to_rgb')
        opt = torch.optim.SGD((p for p in module.parameters() if p.requires_grad),lr=.1)
        target = torch.randn_like(self.r)
        for step in range(2):
            _, rgb = module(self.p,self.r,self.mask)
            (rgb-target).square().mean().backward()
            if step == 1:
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    and p.grad.abs().sum()>0
                                    for p in module.parameters() if p.requires_grad))
            opt.step(); opt.zero_grad()

    def test_invalid_shapes_rejected(self):
        with self.assertRaises(ValueError):self.module(self.p,self.r[:1],self.mask)


if __name__=='__main__':unittest.main()
