"""Differentiable window collation matching the unchanged shared I3D extractor."""
import torch
import torch.nn.functional as F


def window_batch(frames, starts, recipe, device):
    if not starts:
        raise ValueError('Nonempty window starts required')
    frames = frames.to(device)
    if len(frames) < recipe.clip_frames:
        frames = torch.cat([frames,frames[-1:].expand(recipe.clip_frames-len(frames),-1,-1,-1)])
    index = torch.as_tensor(starts,device=device)[:,None]+torch.arange(recipe.clip_frames,device=device)[None,:]
    clips = frames.index_select(0,index.flatten()).reshape(len(starts),recipe.clip_frames,3,
                                       recipe.resize_short_side,recipe.resize_short_side)
    clips = clips.permute(0,2,1,3,4).contiguous()
    ticks = torch.linspace(-recipe.crop_size/recipe.resize_short_side,
                            recipe.crop_size/recipe.resize_short_side,recipe.crop_size,
                            device=device,dtype=clips.dtype)
    gy,gx = torch.meshgrid(ticks,ticks,indexing='ij')
    grid = torch.stack([gx,gy],dim=2).unsqueeze(0).expand(len(starts),-1,-1,-1)
    clips = F.grid_sample(clips.reshape(len(starts),3*recipe.clip_frames,recipe.resize_short_side,
                         recipe.resize_short_side),grid,mode='bilinear',align_corners=False,
                         padding_mode='zeros').reshape(len(starts),3,recipe.clip_frames,
                                                      recipe.crop_size,recipe.crop_size)
    return clips.sub_(.5)


def unfreeze_i3d_tail(model):
    model.eval().requires_grad_(False)
    selected = {}
    for name,p in model.named_parameters():
        if name.startswith(('Mixed_5b.','Mixed_5c.')):
            p.requires_grad_(True)
            selected[name] = p
    if not selected:
        raise ValueError('No expected I3D tail parameters')
    return selected
