import copy
import torch
from methods.seds_adaptation.pose3d_branch import (Pose3DBranch, normalize_geometry,
    canonicalize_geometry, attach_pose3d_branch, joint_indices, JOINT_NAMES)


def test_xy_depth_isolation_mask_and_gradient():
    torch.manual_seed(13)
    x = torch.randn(2,3,4,49,3)
    valid = torch.ones(2,3,4,dtype=torch.bool); valid[1,2] = False
    a = normalize_geometry(x,valid,True); b = normalize_geometry(x,valid,False)
    assert torch.equal(a[...,:2],b[...,:2]) and torch.count_nonzero(b[...,2]) == 0
    model = Pose3DBranch(8,4)
    assert torch.count_nonzero(model(x,valid)) == 0
    model(x,valid).sum().backward()
    assert model.output.weight.grad.abs().sum() > 0
    with torch.no_grad(): model.output.weight.add_(.01*model.output.weight.grad)
    altered=x.clone(); altered[~valid] = float('nan')
    assert torch.equal(model(x,valid),model(altered,valid))
    assert torch.count_nonzero(model(x,valid)[1,2]) == 0
    xy = copy.deepcopy(model); xy.use_depth = False
    altered=x.clone(); altered[...,2] += torch.randn_like(altered[...,2])
    assert torch.equal(xy(x,valid),xy(altered,valid))
    assert not torch.allclose(model(x,valid),model(altered,valid))


class Fake(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.clip=torch.nn.Module(); self.clip.visual=torch.nn.Module()
        self.clip.visual.proj=torch.nn.Parameter(torch.eye(8))
    def get_visual_output(self,right,left,body):
        return body['mask'],body['pose'],body['rgb']


def test_keep_original_streams_identity_cls_padding_checkpoint():
    model=Fake(); rebuilt=copy.deepcopy(model)
    body=dict(mask=torch.tensor([[0,0,0,1]]),pose=torch.randn(1,4,8),rgb=torch.randn(1,4,8),
              geometry_windows=torch.randn(1,3,4,49,3),geometry_valid=torch.ones(1,3,4,dtype=torch.bool))
    original={k:v.clone() for k,v in body.items()}
    attach_pose3d_branch(model,hidden=4)
    mask,pose,rgb=model.get_visual_output({}, {},body)
    assert torch.equal(pose,body['pose']) and torch.equal(rgb,body['rgb'])
    with torch.no_grad(): model.pose3d_branch.output.weight.normal_()
    _,pose,rgb=model.get_visual_output({}, {},body)
    assert torch.equal(pose[:,0],body['pose'][:,0]) and torch.equal(pose[:,3],body['pose'][:,3])
    assert torch.equal(rgb,body['rgb']) and all(torch.equal(body[k],original[k]) for k in body)
    assert not torch.equal(pose[:,1:3],body['pose'][:,1:3])
    attach_pose3d_branch(rebuilt,hidden=4); rebuilt.load_state_dict(model.state_dict(),strict=True)
    assert torch.equal(rebuilt.get_visual_output({}, {},body)[1],pose)


def test_mapping_uses_names_not_positions():
    names=list(dict.fromkeys(JOINT_NAMES))[::-1]
    assert [names[i] for i in joint_indices(names)] == list(JOINT_NAMES)


def test_canonical_geometry_is_translation_scale_rotation_invariant():
    torch.manual_seed(23)
    x=torch.randn(2,3,1,49,3)
    # Make the anatomical axes comfortably non-degenerate.
    x[...,5,:]+=torch.tensor((2.,0.,0.));x[...,17,:]-=torch.tensor((2.,0.,0.))
    x[...,26,:]+=torch.tensor((2.,0.,0.));x[...,38,:]-=torch.tensor((2.,0.,0.))
    x[...,43,:]-=torch.tensor((2.,0.,0.));x[...,46,:]+=torch.tensor((2.,0.,0.))
    x[...,9,:]+=torch.tensor((0.,2.,0.));x[...,30,:]+=torch.tensor((0.,2.,0.))
    x[...,42,:]+=torch.tensor((0.,2.,0.))
    valid=torch.ones(2,3,1,dtype=torch.bool)
    q,_=torch.linalg.qr(torch.randn(3,3))
    if torch.linalg.det(q)<0:q[:,0]*=-1
    transformed=torch.einsum('...jc,cd->...jd',x,q)*3.7+torch.tensor((11.,-8.,4.))
    torch.testing.assert_close(canonicalize_geometry(x,valid),
                               canonicalize_geometry(transformed,valid),rtol=1e-5,atol=2e-5)


def test_canonical_motion_identity_gradient_and_mask():
    torch.manual_seed(29)
    x=torch.randn(2,5,1,49,3)
    valid=torch.ones(2,5,1,dtype=torch.bool);valid[1,4]=False
    module=Pose3DBranch(8,4,use_depth=True,clip_temporal=True,
                        representation='canonical_motion')
    assert torch.count_nonzero(module(x,valid))==0
    module(x,valid).sum().backward()
    assert module.output.weight.grad.abs().sum()>0
    with torch.no_grad():module.output.weight.add_(module.output.weight.grad*.01)
    altered=x.clone();altered[~valid]=float('nan')
    assert torch.equal(module(x,valid),module(altered,valid))
    assert torch.count_nonzero(module(x,valid)[1,4])==0


def test_canonical_motion_contract_rejects_xy_or_non_temporal():
    for kwargs in ({'use_depth':False,'clip_temporal':True},
                   {'use_depth':True,'clip_temporal':False}):
        try:Pose3DBranch(8,4,representation='canonical_motion',**kwargs)
        except ValueError:pass
        else:raise AssertionError('Invalid canonical-motion contract accepted')
