import tempfile
import unittest
from pathlib import Path
import torch
from methods.seds_adaptation.gcn_freeze import freeze_gcn_keep_optimizer
from methods.seds_adaptation.masked_pose import attach_masked_pose, configure_masked_pose
from methods.seds_adaptation.train_policies import training_modes
from research.slret_goal_v2.tests.test_masked_pose import Model, batch


class TestFreeze(unittest.TestCase):
    def test_ddp_transition_keeps_optimizer_and_freezes_gcn(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            torch.distributed.init_process_group('gloo',init_method=Path(tmp,'rdzv').as_uri(),rank=0,world_size=1)
            try:
                model=Model(); attach_masked_pose(model,control=True)
                active=configure_masked_pose(model,control=True)
                optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-3)
                wrapped=torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
                def step():
                    wrapped.train(); training_modes(model,active)
                    optimizer.zero_grad(); wrapped(*batch())[0].backward(); optimizer.step()
                step(); step()
                parameter=next(model.fusion.parameters())
                momentum=optimizer.state[parameter]['exp_avg'].clone()
                del wrapped
                result=freeze_gcn_keep_optimizer(model,optimizer)
                self.assertEqual(result['fusion_optimizer_steps'],[2])
                self.assertFalse(result['optimizer_reset'])
                self.assertTrue(torch.equal(momentum,optimizer.state[parameter]['exp_avg']))
                anchor={n:p.clone() for n,p in model.signbert.embed.named_parameters()}
                fusion=parameter.detach().clone()
                active=[model.fusion]
                wrapped=torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
                step(); step()
                self.assertEqual(int(optimizer.state[parameter]['step']),4)
                self.assertFalse(torch.equal(parameter,fusion))
                for n,p in model.signbert.embed.named_parameters():
                    self.assertTrue(torch.equal(p,anchor[n])); self.assertIsNone(p.grad)
            finally:
                torch.distributed.destroy_process_group()
