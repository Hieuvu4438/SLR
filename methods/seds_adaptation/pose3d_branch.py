"""Additional geometric branch: original RGB and pose2D encoders are retained.

Input is explicitly aligned H4W windows [B,L,S,49,3], not a replacement for
native 2D keypoints. XYZ vs XY share parameters and XY-only normalization.
No confidence weighting, camera calibration claim, or implicit interpolation.
"""
from types import MethodType
import torch
from torch import nn


BODY_NAMES = ('Nose','L_Shoulder','L_Elbow','L_Wrist','R_Shoulder','R_Elbow','R_Wrist')
FINGER_NAMES = tuple(f'{finger}_{j}' for finger in ('Thumb','Index','Middle','Ring','Pinky') for j in range(1,5))
JOINT_NAMES = tuple(['L_Wrist'] + ['L_'+n for n in FINGER_NAMES]
                    + ['R_Wrist'] + ['R_'+n for n in FINGER_NAMES] + list(BODY_NAMES))


def joint_indices(source_names):
    if len(set(source_names)) != len(source_names):
        raise ValueError('Ambiguous source keypoint names')
    return [source_names.index(n) for n in JOINT_NAMES]


def normalize_geometry(x, valid, use_depth=True):
    if x.ndim != 5 or x.shape[-2:] != (49,3) or valid.shape != x.shape[:3]:
        raise ValueError('Expected [B,L,S,49,3] windows and [B,L,S] validity')
    if valid.dtype != torch.bool:
        raise ValueError('Frame validity must be boolean')
    if not torch.isfinite(x[valid]).all():
        raise ValueError('Nonfinite geometry in a valid frame')
    x = torch.where(valid[...,None,None], x, torch.zeros_like(x))
    if not use_depth:
        x = x * x.new_tensor([1.,1.,0.])
    # Both arms use exactly the same XY scale, so XY does not inherit depth.
    scale = (x[...,43,:2]-x[...,46,:2]).norm(dim=-1).clamp_min(1e-4)
    center = (x[...,43:44,:]+x[...,46:47,:])/2
    parts = [x[...,:21,:]-x[...,:1,:], x[...,21:42,:]-x[...,21:22,:], x[...,42:,:]-center]
    return torch.cat(parts,-2) / scale[...,None,None]


def _unit(vector, fallback, eps=1e-5):
    norm = vector.norm(dim=-1,keepdim=True)
    fallback = (fallback.to(device=vector.device,dtype=vector.dtype) if torch.is_tensor(fallback)
                else vector.new_tensor(fallback)).expand_as(vector)
    return torch.where(norm > eps, vector/norm.clamp_min(eps), fallback)


def _anatomical_basis(horizontal, vertical):
    """Return right-handed local axes as rows, with stable rare fallbacks."""
    x = _unit(horizontal,(1.,0.,0.))
    y = vertical-(vertical*x).sum(-1,keepdim=True)*x
    fallback = x.new_tensor((0.,1.,0.)).expand_as(x)
    alternate = x.new_tensor((0.,0.,1.)).expand_as(x)
    fallback = torch.where(((fallback*x).sum(-1,keepdim=True).abs()>.9),alternate,fallback)
    fallback = fallback-(fallback*x).sum(-1,keepdim=True)*x
    y = _unit(y,_unit(fallback,(0.,1.,0.)))
    z = _unit(torch.cross(x,y,dim=-1),(0.,0.,1.))
    # Recompute y so numerical errors do not make the frame non-orthogonal.
    y = _unit(torch.cross(z,x,dim=-1),(0.,1.,0.))
    return torch.stack((x,y,z),dim=-2)


def canonicalize_geometry(x, valid):
    """Map H4W camera-axis XYZ to body/palm-local, scale-normalized XYZ.

    Hands use wrist origin, index-to-pinky palm axis and projected middle-MCP
    axis. Body uses shoulder center, left-to-right shoulder axis and nose axis.
    Left/right hands remain separate downstream, so handedness is not collapsed.
    """
    if x.ndim != 5 or x.shape[-2:] != (49,3) or valid.shape != x.shape[:3]:
        raise ValueError('Expected [B,L,S,49,3] windows and [B,L,S] validity')
    if valid.dtype != torch.bool or not torch.isfinite(x[valid]).all():
        raise ValueError('Canonical geometry requires finite valid XYZ and bool mask')
    x = torch.where(valid[...,None,None],x,torch.zeros_like(x))

    def hand(part):
        relative = part-part[...,:1,:]
        horizontal = relative[...,5,:]-relative[...,17,:]   # index MCP - pinky MCP
        vertical = relative[...,9,:]                         # wrist -> middle MCP
        basis = _anatomical_basis(horizontal,vertical)
        scale = horizontal.norm(dim=-1).clamp_min(1e-4)
        return torch.einsum('...jc,...kc->...jk',relative,basis)/scale[...,None,None]

    body = x[...,42:,:]
    center = (body[...,1:2,:]+body[...,4:5,:])/2
    relative = body-center
    horizontal = body[...,4,:]-body[...,1,:]                 # left -> right shoulder
    vertical = body[...,0,:]-center[...,0,:]                 # shoulder center -> nose
    basis = _anatomical_basis(horizontal,vertical)
    scale = horizontal.norm(dim=-1).clamp_min(1e-4)
    body_local = torch.einsum('...jc,...kc->...jk',relative,basis)/scale[...,None,None]
    result = torch.cat((hand(x[...,:21,:]),hand(x[...,21:42,:]),body_local),dim=-2)
    return result*valid[...,None,None]


class Pose3DBranch(nn.Module):
    def __init__(self, output_width=512, hidden=64, use_depth=True, clip_temporal=False,
                 representation='raw'):
        super().__init__()
        if representation not in ('raw','canonical_motion'):
            raise ValueError('Unknown 3D representation')
        if representation == 'canonical_motion' and (not use_depth or not clip_temporal):
            raise ValueError('Canonical motion requires XYZ and clip-temporal mode')
        self.use_depth = use_depth
        self.clip_temporal = clip_temporal
        self.representation = representation
        multiplier = 2 if representation == 'canonical_motion' else 1
        self.hand = nn.Sequential(nn.Linear(63*multiplier,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.body = nn.Sequential(nn.Linear(21*multiplier,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.temporal = nn.Conv1d(hidden*3,hidden*3,3,padding=1)
        self.output = nn.Linear(hidden*3,output_width,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, geometry, valid):
        g = (canonicalize_geometry(geometry,valid) if self.representation == 'canonical_motion'
             else normalize_geometry(geometry,valid,self.use_depth))
        b,l,s = g.shape[:3]
        clip_valid = valid.any(2)
        if self.representation == 'canonical_motion':
            # Cache currently supplies one exact center per native clip. Average
            # generally so the representation remains correct for S>1.
            g = (g*valid[...,None,None]).sum(2)/valid.sum(2).clamp_min(1)[...,None,None]
            motion = torch.zeros_like(g)
            pair_valid = clip_valid[:,1:] & clip_valid[:,:-1]
            motion[:,1:] = (g[:,1:]-g[:,:-1])*pair_valid[...,None,None]
            g = torch.cat((g,motion),dim=-1).unsqueeze(2)
            valid = clip_valid.unsqueeze(-1)
            s = 1
        h = torch.cat([self.hand(g[...,:21,:].flatten(-2)),
                       self.hand(g[...,21:42,:].flatten(-2)),
                       self.body(g[...,42:,:].flatten(-2))],-1)
        h = h * valid[...,None]
        if self.clip_temporal:
            # One exact center per native clip: temporal convolution runs across
            # clip tokens, NOT across invented/interpolated within-clip frames.
            h = h.sum(2)/valid.sum(2).clamp_min(1)[...,None]
            clip_valid = valid.any(2)
            h = torch.nn.functional.gelu(self.temporal(h.transpose(1,2))).transpose(1,2)
            return self.output(h) * clip_valid[...,None]
        h = h.reshape(b*l,s,-1).transpose(1,2)
        h = torch.nn.functional.gelu(self.temporal(h)).transpose(1,2).reshape(b,l,s,-1)
        h = (h*valid[...,None]).sum(2)/valid.sum(2).clamp_min(1)[...,None]
        return self.output(h)


def attach_pose3d_branch(model, use_depth=True, hidden=64, clip_temporal=False,
                         representation='raw'):
    if hasattr(model,'pose3d_branch'):
        raise ValueError('Already attached')
    prototype = model.clip.visual.proj
    branch = Pose3DBranch(prototype.shape[-1],hidden,use_depth,clip_temporal,representation).to(
        device=prototype.device,dtype=torch.float32)
    model.add_module('pose3d_branch',branch)
    original = model.get_visual_output

    def augmented(self,right_batch,left_batch,body_batch,*args,**kwargs):
        # Existing computation runs unchanged; additional fields are not read by
        # native SEDS. Do not remove or replace body_batch['pose'] or RGB data.
        mask,pose,rgb = original(right_batch,left_batch,body_batch,*args,**kwargs)
        delta = self.pose3d_branch(body_batch['geometry_windows'],body_batch['geometry_valid'])
        if delta.shape[:2] != (pose.shape[0],pose.shape[1]-1):
            raise ValueError('3D clip alignment mismatch: CLS must not have a 3D sample')
        delta = torch.cat([torch.zeros_like(delta[:,:1]),delta],dim=1)
        delta = delta.masked_fill(mask.bool()[...,None],0)
        return mask,pose+delta.to(pose.dtype),rgb

    model.get_visual_output = MethodType(augmented,model)
    return branch
