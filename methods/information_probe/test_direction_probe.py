import numpy as np
import torch

from .direction_probe import fit,paired
from .scoring import channels
from .test_scoring import inputs


def test_paired_agrees_with_matrix_diagonal():
    v,t,vm,tm=inputs()
    a,b=channels(v,t,vm,tm,13.)
    torch.testing.assert_close(paired(v,t,vm,tm,13.),torch.stack((a.diag(),b.diag()),-1))


def test_convex_fit_chooses_dominant_channel():
    assert fit(np.tile([2.,-2.],(12,1)))['eta']==1.
    assert fit(np.tile([-2.,2.],(12,1)))['eta']==0.
