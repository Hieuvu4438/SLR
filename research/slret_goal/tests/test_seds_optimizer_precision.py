"""Numerical controls against native BertAdam; no benchmark claim."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/slret_goal/tools'))
from seds_optimizer_precision import ensure_fp32_moments
spec=importlib.util.spec_from_file_location('seds_native_optimization',ROOT/'third_party/SEDS/modules/optimization.py')
native=importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class MomentPrecisionTests(unittest.TestCase):
    def optimizer(self,p):
        return native.BertAdam([p],lr=.001,t_total=-1,weight_decay=0,max_grad_norm=0)

    def test_nonzero_second_moment_and_reference_update(self):
        p=torch.nn.Parameter(torch.ones(4,dtype=torch.float16))
        reference=torch.nn.Parameter(p.float())
        opt,fp32=self.optimizer(p),self.optimizer(reference)
        for _ in range(2):
            p.grad=torch.full_like(p,1e-4)
            reference.grad=p.grad.float()
            ensure_fp32_moments(opt)
            opt.step(); fp32.step()
            self.assertTrue(torch.equal(p,reference.half()))
            self.assertTrue(torch.equal(opt.state[p]['next_v'],fp32.state[reference]['next_v']))
            # Mirror native parameter rounding; isolate moment dtype only.
            reference.data.copy_(p.float())
        self.assertTrue((opt.state[p]['next_v']>0).all())
        self.assertEqual(p.dtype,torch.float16)

    def test_native_half_underflows_for_same_nonzero_gradient(self):
        p=torch.nn.Parameter(torch.ones(4,dtype=torch.float16)); opt=self.optimizer(p)
        p.grad=torch.full_like(p,1e-4); opt.step()
        self.assertTrue((opt.state[p]['next_m']!=0).all())
        self.assertTrue((opt.state[p]['next_v']==0).all())
        with self.assertRaises(ValueError): ensure_fp32_moments(opt)

    def test_float32_unchanged_and_inactive_not_initialized(self):
        p=torch.nn.Parameter(torch.ones(4)); opt=self.optimizer(p)
        p.grad=torch.ones_like(p)
        self.assertEqual(ensure_fp32_moments(opt),0)
        self.assertEqual(len(opt.state),0)
        q=torch.nn.Parameter(torch.ones(4,dtype=torch.float16)); inactive=self.optimizer(q)
        self.assertEqual(ensure_fp32_moments(inactive),0)
        self.assertEqual(len(inactive.state),0)


if __name__=='__main__': unittest.main()
