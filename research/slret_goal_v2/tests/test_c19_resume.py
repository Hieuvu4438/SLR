import copy
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/slret_goal_v2/tools'))
sys.path.insert(0,str(ROOT/'third_party/SEDS'))
from modules.optimization import BertAdam
from c19_resume import restore_bertadam, check_config, inherit_progress, GATES


class ResumeTests(unittest.TestCase):
    def optimizer(self, parameter):
        return BertAdam([dict(params=[parameter],lr=1e-4,adaptation_group='test')],
                        t_total=10,warmup=.1,max_grad_norm=1.)

    def test_bertadam_next_update_matches_uninterrupted(self):
        torch.manual_seed(10)
        p = torch.nn.Parameter(torch.randn(8))
        opt = self.optimizer(p)
        for _ in range(4):
            (p*torch.rand(8)).square().sum().backward();opt.step();opt.zero_grad()
        saved = copy.deepcopy(opt.state_dict())
        rng = torch.get_rng_state()
        q = torch.nn.Parameter(p.detach().clone())
        resumed = self.optimizer(q)
        restore_bertadam(resumed,saved,4)
        for parameter, optimizer in ((p,opt),(q,resumed)):
            torch.set_rng_state(rng)
            (parameter*torch.rand(8)).square().sum().backward()
            optimizer.step();optimizer.zero_grad()
        self.assertTrue(torch.equal(p,q))
        for key in ('next_m','next_v'):
            self.assertTrue(torch.equal(opt.state[p][key],resumed.state[q][key]))
        self.assertEqual(resumed.state[q]['step'],5)

    def test_fp16_parameter_retains_exact_fp32_moments(self):
        p = torch.nn.Parameter(torch.ones(2,dtype=torch.float16))
        opt = self.optimizer(p)
        state = opt.state_dict()
        state['state'] = {0:dict(step=4,next_m=torch.tensor([.1234567,.7654321]),
                                next_v=torch.tensor([.0246813,.1357924]))}
        restore_bertadam(opt,state,4)
        self.assertEqual(opt.state[p]['next_m'].dtype,torch.float32)
        self.assertTrue(torch.equal(opt.state[p]['next_m'],state['state'][0]['next_m']))

    def test_wrong_moment_precision_rejected(self):
        p = torch.nn.Parameter(torch.ones(2,dtype=torch.float16)); opt = self.optimizer(p)
        saved = opt.state_dict()
        saved['state'] = {0:dict(step=4,next_m=torch.ones(2,dtype=torch.float16),next_v=torch.ones(2))}
        with self.assertRaisesRegex(ValueError,'FP32'):restore_bertadam(opt,saved,4)

    def test_schedule_mismatch_rejected(self):
        p = torch.nn.Parameter(torch.ones(2)); opt = self.optimizer(p)
        saved = copy.deepcopy(opt.state_dict());saved['param_groups'][0]['t_total']=20
        with self.assertRaisesRegex(ValueError,'schedule'):restore_bertadam(opt,saved,4)

    def test_config_allows_only_recovery_identity(self):
        saved = dict(run_id='parent',seed=42,epochs=10)
        check_config(saved,dict(run_id='child',seed=42,epochs=10,resume_from='last.pt'))
        with self.assertRaises(ValueError):check_config(saved,dict(run_id='child',seed=43,epochs=10))

    def test_inherited_metrics_exclude_future_and_label_provenance(self):
        parent = dict(steps=121,criterion='native',evaluations={'0':{},'80':{},'160':{}},
                      **{k:True for k in GATES})
        report = {}
        inherit_progress(report,parent,112)
        self.assertEqual(set(report['evaluations']),{'0','80'})
        self.assertEqual(report['resume']['discarded_unsaved_updates'],9)
        self.assertEqual(report['steps'],112)


if __name__=='__main__':unittest.main()
