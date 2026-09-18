import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from rgb_gradient_replay import window_batch


class RGBReplayTests(unittest.TestCase):
    def test_windows_and_padding(self):
        # Constant planes isolate temporal/channel indexing from spatial resampling.
        recipe = SimpleNamespace(clip_frames=2,resize_short_side=4,crop_size=2)
        frames = torch.arange(3.).reshape(3,1,1,1).expand(3,3,4,4)/4
        clips = window_batch(frames,[0,1],recipe,'cpu')
        self.assertEqual(tuple(clips.shape),(2,3,2,2,2))
        for batch,start in enumerate([0,1]):
            for t in range(2):
                torch.testing.assert_close(clips[batch,:,t],torch.full((3,2,2),(start+t)/4-.5))
        one = window_batch(frames[:1],[0],recipe,'cpu')
        self.assertTrue(torch.equal(one[:,:,0],one[:,:,1]))

    def test_gradient_cache_chain_rule(self):
        torch.manual_seed(0)
        layer = torch.nn.Linear(3,2).double()
        x = torch.randn(8,3,dtype=torch.double)
        target = torch.randn(8,2,dtype=torch.double)
        loss = ((layer(x)-target)**2).sum()
        expected = torch.autograd.grad(loss,tuple(layer.parameters()))
        leaf = layer(x).detach().requires_grad_(True)
        grad, = torch.autograd.grad(((leaf-target)**2).sum(),leaf)
        layer.zero_grad()
        for start in range(0,8,2):
            layer(x[start:start+2]).backward(grad[start:start+2])
        for p,g in zip(layer.parameters(),expected):
            torch.testing.assert_close(p.grad,g,atol=1e-12,rtol=1e-12)


if __name__ == '__main__': unittest.main()
