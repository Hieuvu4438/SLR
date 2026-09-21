from types import SimpleNamespace
import unittest

import numpy as np

from research.slret_goal_v2.tools.extract_h4w_sample import person_box, projection_to_pixels
from research.slret_goal_v2.tools.extract_h4w_sample import set_inference_flags


def box(cls, confidence, corners):
    return SimpleNamespace(cls=np.array(cls),conf=np.array(confidence),xyxy=np.array([corners]))


class SampleTests(unittest.TestCase):
    def test_inference_does_not_invoke_overloaded_train(self):
        import torch
        class WorkflowWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__();self.bn=torch.nn.BatchNorm1d(3)
            def train(self,*args,**kwargs):
                raise AssertionError('Must never invoke training workflow')
        model=torch.nn.Sequential(WorkflowWrapper(),torch.nn.Dropout(.5))
        before=model[0].bn.running_mean.clone()
        set_inference_flags(model)
        self.assertTrue(all(not m.training for m in model.modules()))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        model[0].bn(torch.ones(2,3))
        self.assertTrue(torch.equal(before,model[0].bn.running_mean))

    def test_native_person_selection_and_fallback(self):
        boxes=[box(1,.99,[1,2,3,4]),box(0,.6,[10,20,110,220]),box(0,.9,[20,30,100,180])]
        self.assertEqual(person_box(boxes,210,260),([20,30,80,150],False))
        self.assertEqual(person_box([],210,260),([0.,0.,210.,260.],True))
        with self.assertRaises(ValueError):
            person_box([box(0,.9,[20,30,10,180])],210,260)


    def test_projection_scale_inverse_affine_no_input_mutation(self):
        uv=np.array([[0.,0.],[6.,8.],[12.,16.]],np.float32)
        old=uv.copy()
        # A crop-to-original affine: half scale and a known offset.
        affine=np.array([[.5,0,10],[0,.5,-5]])
        actual=projection_to_pixels(uv,(512,384),(16,12),affine)
        np.testing.assert_allclose(actual,[[10,-5],[106,123],[202,251]])
        np.testing.assert_array_equal(uv,old)
