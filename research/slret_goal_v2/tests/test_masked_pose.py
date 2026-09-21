import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from methods.seds_adaptation.masked_pose import MaskedPose, attach_masked_pose, configure_masked_pose, masked_learning_rates
from methods.seds_adaptation.train_policies import training_modes


class Sign(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(98, 1536)

    def gcn_emb(self, pose):
        pose['feat'] = self.embed(torch.cat([pose[k] for k in ('left','right','body')], 2).flatten(2))
        return pose


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.signbert = Sign()
        self.fusion = nn.Linear(1536, 2)
        self.frozen = nn.Linear(2, 2)
        self.task_config = SimpleNamespace(slide_windows=2)

    def get_sign_output(self, right, left, body):
        features = self.signbert.gcn_emb(dict(left=left['pose'], right=right['pose'], body=body['pose']))['feat']
        if hasattr(self.signbert, 'GCN_Conv'):
            features = self.signbert.GCN_Conv(features.transpose(1,2)).transpose(1,2)
        return features

    def forward(self, right, left, body):
        features = self.get_sign_output(right, left, body)
        loss = self.fusion(features).square().mean()
        return (loss,) * 7 if self.training else features


def batch():
    return [dict(pose=torch.full((2, 5, n, 2), 128.),
                 clips_start=torch.tensor([[1,-1],[0,2]])) for n in (21,21,7)]


class Tests(unittest.TestCase):
    def test_mask_target_padding_and_rng(self):
        m = MaskedPose(ratio=.5)
        pose = {k: torch.full((2,5,n,2),128.) for k,n in zip(('left','right','body'),(21,21,7))}
        pose['left'][:,0] = 0
        covered = torch.ones(2,5,dtype=torch.bool); covered[:,-1] = False
        rng = torch.get_rng_state().clone()
        corrupted, target, selected = m.corrupt(pose, covered)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(selected[:,-1].any())
        self.assertFalse(selected[:,0,:21].any())
        self.assertEqual(float(target[:,1].abs().sum()), 0.)
        original = torch.cat([pose[k] for k in ('left','right','body')],2)
        actual = torch.cat([corrupted[k] for k in ('left','right','body')],2)
        self.assertTrue(torch.equal(actual[~selected], original[~selected]))
        self.assertEqual(float(actual[selected].abs().sum()), 0.)
        self.assertEqual(float(pose['body'][0,0,0,0]), 128.)

    def test_eval_identity_and_training_gradient(self):
        m = Model(); data = batch()
        expected = m.get_sign_output(*data).detach()
        attach_masked_pose(m)
        m.eval()
        self.assertTrue(torch.equal(expected, m.get_sign_output(*data)))
        self.assertIsNone(m.masked_pose.loss)
        active = configure_masked_pose(m); training_modes(m,active)
        self.assertFalse(m.signbert.embed.training)
        m.get_sign_output(*data)
        m.masked_pose.loss.backward()
        self.assertGreater(float(m.signbert.embed.weight.grad.abs().sum()),0.)
        self.assertIsNone(m.frozen.weight.grad)
        self.assertEqual(m.masked_pose.stats['valid_joints'], 2*49+4*49)
        m.eval(); m.get_sign_output(*data)
        self.assertIsNone(m.masked_pose.loss)

    def test_control_and_rates(self):
        m = Model(); data=batch(); expected=m.get_sign_output(*data).detach()
        attach_masked_pose(m,control=True)
        active=configure_masked_pose(m,control=True); training_modes(m,active)
        self.assertTrue(torch.equal(expected,m.get_sign_output(*data)))
        self.assertFalse(any(p.requires_grad for p in m.masked_pose.parameters()))
        optim=torch.optim.Adam(m.parameters())
        groups=masked_learning_rates(optim,m,control=True)
        self.assertEqual({g['adaptation_group']:g['lr'] for g in groups}, {'encoder':1e-6,'fusion':1e-5})

    def test_empty_target(self):
        m=MaskedPose(width=4)
        loss=m.reconstruct(torch.randn(2,3,4),torch.zeros(2,3,49,2),torch.zeros(2,3,49,dtype=torch.bool))
        self.assertEqual(float(loss),0.)
        loss.backward()

    def test_fusion_ablation_frozen_encoder(self):
        m=Model(); attach_masked_pose(m,control=True)
        active=configure_masked_pose(m,control=True,fusion_only=True)
        training_modes(m,active)
        before={n:p.detach().clone() for n,p in m.named_parameters()}
        optim=torch.optim.Adam(m.parameters())
        groups=masked_learning_rates(optim,m,control=True,fusion_only=True)
        self.assertEqual({g['adaptation_group']:g['lr'] for g in groups},{'fusion':1e-5})
        m(*batch())[0].backward(); optim.step()
        changed=[n for n,p in m.named_parameters() if not torch.equal(p,before[n])]
        self.assertTrue(changed and all(n.startswith('fusion.') for n in changed))
        self.assertFalse(m.signbert.embed.training)
        self.assertIsNone(m.masked_pose.loss)

    def test_clip_temporal_policy_updates_and_frozen_bn(self):
        m=Model()
        m.signbert.GCN_Conv=nn.Sequential(nn.Conv1d(1536,1536,3,padding=1,groups=1536),nn.BatchNorm1d(1536))
        attach_masked_pose(m,control=True)
        active=configure_masked_pose(m,control=True,clip_temporal=True)
        training_modes(m,active)
        before={n:p.detach().clone() for n,p in m.named_parameters()}
        buffers={n:b.clone() for n,b in m.named_buffers()}
        optim=torch.optim.Adam(m.parameters())
        groups=masked_learning_rates(optim,m,control=True,clip_temporal=True)
        self.assertEqual({g['adaptation_group']:g['lr'] for g in groups},
                         {'fusion':1e-5,'encoder':1e-6,'clip_temporal':1e-6})
        m(*batch())[0].backward();optim.step()
        changed=[n for n,p in m.named_parameters() if not torch.equal(p,before[n])]
        for prefix in ('signbert.embed.','signbert.GCN_Conv.','fusion.'):
            self.assertTrue(any(n.startswith(prefix) for n in changed))
        self.assertTrue(all(torch.equal(buffers[n],b) for n,b in m.named_buffers()))
        self.assertFalse(m.signbert.GCN_Conv.training)
        self.assertFalse(any(n.startswith('masked_pose.') for n in changed))

    def test_ddp_two_steps_and_eval(self):
        # Same find_unused_parameters setting as native SEDS. This regression
        # exercises the reducer, which the original plain-module tests missed.
        with tempfile.TemporaryDirectory(prefix='c08-ddp-') as temp:
            torch.distributed.init_process_group('gloo',
                init_method=Path(temp, 'rendezvous').as_uri(), rank=0, world_size=1)
            try:
                for control in (False, True):
                    m = Model(); data = batch()
                    m.eval(); expected = m(*data).detach()
                    attach_masked_pose(m, control=control)
                    self.assertTrue(torch.equal(expected, m(*data)))
                    active = configure_masked_pose(m, control=control)
                    wrapped = nn.parallel.DistributedDataParallel(m, find_unused_parameters=True)
                    optimizer = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1e-7)
                    for _ in range(2):
                        training_modes(m, active); optimizer.zero_grad()
                        losses = wrapped(*data)
                        self.assertEqual(len(losses), 7 if control else 8)
                        loss = losses[0] if control else losses[0] + .05*losses[7]
                        loss.backward()
                        self.assertIsNone(m.masked_pose.loss)
                        if not control:
                            self.assertTrue(m.masked_pose.stats['auxiliary_feature_gradient_finite'])
                            self.assertGreater(m.masked_pose.stats['auxiliary_feature_gradient_norm'],0.)
                            self.assertIsNotNone(m.masked_pose.decoder[-1].bias.grad)
                        self.assertGreater(float(m.signbert.embed.weight.grad.abs().sum()),0.)
                        optimizer.step()
                    m.eval()
                    self.assertTrue(torch.equal(m(*data), m.get_sign_output(*data)))
                    del wrapped
            finally:
                torch.distributed.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
