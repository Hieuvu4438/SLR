import copy
import unittest

import numpy as np
import torch

from methods.seds_adaptation.geometry_cache import clip_centers, aligned_geometry, geometry_collate
from methods.seds_adaptation.pose3d_branch import JOINT_NAMES, Pose3DBranch


class GeometryCacheTests(unittest.TestCase):
    def metadata(self):
        retained=np.arange(0,40,2)
        starts=[0,1,4]
        windows=retained[np.array(starts)[:,None]+np.arange(16)]
        return dict(retained_frame_indices=retained.tolist(),clip_starts=starts,
                    original_frame_indices_per_window=windows.tolist(),decoded_frames=40)

    def cache(self):
        centers,starts=clip_centers(self.metadata())
        return dict(frame_ids=centers,clip_frame_ids=centers,clip_starts=starts,
                    joint_names=np.asarray(JOINT_NAMES),xyz=np.ones((3,49,3),np.float32))

    def test_exact_raw_centers_and_tail_repeat(self):
        centers,starts=clip_centers(self.metadata())
        np.testing.assert_array_equal(centers,[16,18,24])
        m=dict(retained_frame_indices=[2,5],clip_starts=[0],
               original_frame_indices_per_window=[[2]+[5]*15],decoded_frames=6)
        np.testing.assert_array_equal(clip_centers(m)[0],[5])
        broken=self.metadata();broken['original_frame_indices_per_window'][0][0]=3
        with self.assertRaises(ValueError):clip_centers(broken)

    def test_clip_alignment_and_zero_padding(self):
        g,v=aligned_geometry(self.cache(),[0,1,4,-1],[0,0,0,0,1])
        self.assertEqual(g.shape,(4,1,49,3))
        self.assertEqual(v[:,0].tolist(),[True,True,True,False])
        self.assertEqual(g[-1].count_nonzero(),0)
        with self.assertRaises(ValueError):
            aligned_geometry(self.cache(),[1,0,4,-1],[0,0,0,0,1])
        bad=self.cache();bad['frame_ids']=np.array([16,18,25])
        with self.assertRaises(ValueError):
            aligned_geometry(bad,[0,1,4,-1],[0,0,0,0,1])

    def test_collate_keeps_native_tensors(self):
        samples=[dict(geometry_windows=torch.ones(4,1,49,3),geometry_valid=torch.ones(4,1,dtype=torch.bool))]*2
        original=torch.randn(2,10,7,2)
        batch=geometry_collate(lambda samples:dict(body_pose=original),samples)
        self.assertIs(batch['body_pose'],original)
        self.assertEqual(batch['geometry_windows'].shape,(2,4,1,49,3))

    def test_clip_temporal_depth_control_and_padding(self):
        torch.manual_seed(42)
        x=torch.randn(2,4,1,49,3);valid=torch.ones(2,4,1,dtype=torch.bool)
        valid[:,3]=False
        model=Pose3DBranch(8,4,True,clip_temporal=True)
        self.assertEqual(model(x,valid).count_nonzero(),0)
        model(x,valid).sum().backward()
        self.assertGreater(model.output.weight.grad.abs().sum(),0)
        with torch.no_grad():model.output.weight.normal_()
        missing=x.clone();missing[~valid]=float('nan')
        self.assertTrue(torch.equal(model(x,valid),model(missing,valid)))
        self.assertEqual(model(x,valid)[:,3].count_nonzero(),0)
        xy=copy.deepcopy(model);xy.use_depth=False
        other=x.clone();other[...,2]+=torch.randn_like(other[...,2])
        self.assertTrue(torch.equal(xy(x,valid),xy(other,valid)))
        self.assertFalse(torch.allclose(model(x,valid),model(other,valid)))
