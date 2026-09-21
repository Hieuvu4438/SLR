"""Exact native-clip-center H4W cache; no replacement of native RGB/pose2D."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from methods.seds_adaptation.pose3d_branch import JOINT_NAMES

SELECTION_SALT = 'pose3d-pilot-v1:'


def selected_metadata(root, count):
    files = sorted(Path(root).glob('*.json'), key=lambda p:
                   hashlib.sha256((SELECTION_SALT+p.stem).encode()).hexdigest())
    if len(files) < count:
        raise ValueError('Not enough metadata for registered subset')
    return files[:count]


def clip_centers(metadata):
    windows = np.asarray(metadata['original_frame_indices_per_window'], dtype=np.int64)
    starts = np.asarray(metadata['clip_starts'], dtype=np.int64)
    retained = np.asarray(metadata['retained_frame_indices'], dtype=np.int64)
    if windows.shape != (len(starts), 16) or not 1 <= len(starts) <= 64:
        raise ValueError('Expected 1..64 native windows of 16 frames')
    if len(retained) == 0 or np.any(np.diff(retained) <= 0) or np.any(starts < 0):
        raise ValueError('Invalid retained frame sequence or clip starts')
    expected = retained[np.minimum(starts[:, None]+np.arange(16),len(retained)-1)]
    if not np.array_equal(windows,expected):
        raise ValueError('Native temporal metadata mismatch')
    if windows.min() < 0 or windows.max() >= metadata['decoded_frames']:
        raise ValueError('Frame IDs must be zero-based decoded video indices')
    # Upper middle (offset8 of16) is registered, not a sampled/averaged pose.
    return windows[:,8].copy(), starts.copy()


def aligned_geometry(cache, starts, mask):
    """Materialize [64,1,49,3], rejecting reordered or incompatible clips."""
    frame_ids = np.asarray(cache['frame_ids'])
    centers = np.asarray(cache['clip_frame_ids'])
    xyz = np.asarray(cache['xyz'])
    n = len(centers)
    if (xyz.shape != (len(frame_ids),49,3) or not np.isfinite(xyz).all()
            or np.any(np.diff(frame_ids) <= 0) or len(frame_ids) == 0
            or list(cache['joint_names']) != list(JOINT_NAMES)):
        raise ValueError('Invalid geometry cache contents/joint mapping')
    starts = torch.as_tensor(starts).cpu()
    mask = torch.as_tensor(mask).bool().cpu()
    if starts.ndim != 1 or mask.shape != (len(starts)+1,) or mask[0] or n > len(starts):
        raise ValueError('Invalid native clip/CLS dimensions')
    if (not np.array_equal(starts[:n].numpy(),cache['clip_starts'])
            or not torch.all(starts[n:] == -1)
            or mask[1:1+n].any() or not mask[1+n:].all()):
        raise ValueError('Geometry/native clip alignment mismatch')
    positions = np.searchsorted(frame_ids,centers)
    if np.any(positions >= len(frame_ids)) or not np.array_equal(frame_ids[positions],centers):
        raise ValueError('Missing exact center-frame geometry')
    geometry = torch.zeros(len(starts),1,49,3,dtype=torch.float32)
    valid = torch.zeros(len(starts),1,dtype=torch.bool)
    geometry[:n,0] = torch.from_numpy(xyz[positions].astype(np.float32))
    valid[:n] = True  # Inferred finite geometry, NOT annotated pose confidence.
    return geometry,valid


class GeometryDataset(Dataset):
    """Wrap an existing native dataset without changing its samples or RNG."""
    def __init__(self, native, native_ids, cache_root, split):
        if split not in ('train','dev'):
            raise ValueError('TEST is outside geometry pilot scope')
        self.native = native
        self.cache_root = Path(cache_root)
        report = json.loads((self.cache_root/'run.json').read_text())
        if (report['status'] != 'completed' or report['completed']!=report['total_videos']
                or report['completed_frames']!=report['total_frames']):
            raise ValueError('Incomplete geometry extraction cannot be used for training')
        plan = json.loads((self.cache_root/'plan.json').read_text())
        planned = {x['id']:x for x in plan['items'] if x['split'] == split}
        self.planned=planned
        if len(set(native_ids)) != len(native_ids) or not set(planned).issubset(native_ids):
            raise ValueError('Ambiguous or missing native video IDs')
        self.ids = [vid for vid in native_ids if vid in planned]
        self.indices = [i for i,vid in enumerate(native_ids) if vid in planned]
        if split == 'dev' and self.ids != list(native_ids):
            raise ValueError('Must preserve entire DEV gallery in native order')
        self.split = split
        self.multi_sentence_per_video = True
        self.cut_off_points = list(range(1,len(self.ids)+1))
        self.sentence_num = self.video_num = len(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self,index):
        sample = self.native[self.indices[index]]
        with np.load(self.cache_root/self.split/(self.ids[index]+'.npz'),allow_pickle=False) as cache:
            item=self.planned[self.ids[index]]
            if any(not np.array_equal(cache[key],item[key]) for key in ('frame_ids','clip_frame_ids','clip_starts')):
                raise ValueError('Geometry file differs from registered video/clip plan')
            geometry,valid = aligned_geometry(cache,sample['body']['clips_start'],sample['body']['pose_mask'])
        sample['geometry_windows'],sample['geometry_valid'] = geometry,valid
        return sample


def geometry_collate(native_collate, samples):
    batch = native_collate(samples)
    batch['geometry_windows'] = torch.stack([x['geometry_windows'] for x in samples])
    batch['geometry_valid'] = torch.stack([x['geometry_valid'] for x in samples])
    return batch
