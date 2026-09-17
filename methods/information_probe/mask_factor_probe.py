"""Separate three historical inner-softmax mask factors, with fixed outer masks."""
import itertools
import json
import os
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from .common import ART, ROOT, dump, metrics, ranks, sha
from .scoring import channels

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'


def factor_channels(v, t, vm, tm, scale, *, exclude_cls=False, exclude_vpad=False, exclude_tpad=False):
    v, t = F.normalize(v, dim=-1), F.normalize(t, dim=-1)
    a = torch.einsum('ifd,jld->ijfl', v, t)
    vv, tv = vm == 0, tm == 1
    assert vv.any(-1).all() and tv.any(-1).all() and not vv[:, 0].any()
    allowed_video = torch.ones_like(vv)
    if exclude_cls:
        allowed_video[:, 0] = False
    if exclude_vpad:
        allowed_video[:, 1:] = vv[:, 1:]
    lv, lt = a/.07, a/.07
    if exclude_tpad:
        lv = lv.masked_fill(~tv[None, :, None, :], -torch.inf)
    if exclude_cls or exclude_vpad:
        lt = lt.masked_fill(~allowed_video[:, None, :, None], -torch.inf)
    av = (a*lv.softmax(-1)).sum(-1)
    at = (a*lt.softmax(-2)).sum(-2)
    return scale*(av*vv[:, None]).sum(-1)/vv.sum(-1)[:, None], scale*(at*tv[None]).sum(-1)/tv.sum(-1)[None]


@torch.inference_mode()
def main():
    path = OUT/'AS-C14-MASK_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    start = time.time()
    result = {'experiment_id': 'AS-C14-MASK', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C14_protocol.md'),
              'scorer_sha256': sha(ROOT/'methods/information_probe/scoring.py'),
              'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'selector': None, 'updates': 0, 'trainable_parameters': 0, 'method_go': False, 'variants': {}}
    dump(path, result)
    try:
        dev = torch.load(ART/'frozen_dev.pt', weights_only=True)
        assert dev['manifest_sha256'] == sha(ROOT/'artifacts/manifests/ph_dev.jsonl')
        result.update(cache_sha256=sha(ART/'frozen_dev.pt'), checkpoint_sha256=dev['checkpoint_sha256'],
                      manifest_sha256=dev['manifest_sha256'])
        v, t, vm, tm = (dev[k].cuda() for k in ('video_tokens', 'text_tokens', 'video_mask', 'text_mask'))
        base, base_ch = np.load(ART/'baseline_dev_scores.npy'), np.load(ART/'baseline_dev_channels.npy')
        result['validity_counts'] = {'video_cls': len(v), 'video_padding': int((vm[:, 1:] != 0).sum()),
                                     'text_padding': int((tm == 0).sum())}
        for c, vp, tp in itertools.product((False, True), repeat=3):
            name = f'c{int(c)}_vp{int(vp)}_tp{int(tp)}'
            scores_ch = np.empty_like(base_ch)
            masked_delta = 0.
            for i in range(0, len(v), 64):
                for j in range(0, len(t), 64):
                    args = (v[i:i+64], t[j:j+64], vm[i:i+64], tm[j:j+64], dev['logit_scale'])
                    ab = factor_channels(*args, exclude_cls=c, exclude_vpad=vp, exclude_tpad=tp)
                    if c and vp and tp:
                        ref = channels(*args, masked=True)
                        masked_delta = max(masked_delta, max(float((a-b).abs().max()) for a, b in zip(ab, ref)))
                    scores_ch[:, i:i+64, j:j+64] = torch.stack(ab).cpu().numpy()
            score = scores_ch.mean(0)
            if not any((c, vp, tp)):
                result['identity_max_channel_delta'] = float(np.abs(scores_ch-base_ch).max())
                assert result['identity_max_channel_delta'] <= 2e-5
                assert all(np.array_equal(ranks(score)[d], ranks(base)[d]) for d in ('T2V', 'V2T'))
            if c and vp and tp:
                assert masked_delta <= 2e-5
                result['joint_mask_independent_channel_delta'] = masked_delta
            m = metrics(score, base)
            dump(OUT/f'AS-C14-{name}_metrics.json', m)
            np.save(ART/f'AS-C14-{name}_scores.npy', score)
            entry = {'exclude_video_cls': c, 'exclude_video_pad': vp, 'exclude_text_pad': tp,
                     'mean_R1': m['official_mean_R1'],
                     'max_channel_delta_vs_historical': [float(np.abs(scores_ch[k]-base_ch[k]).max()) for k in (0, 1)],
                     'directions': {d: {**{k: m[official][k] for k in ('R1', 'R5', 'R10')},
                                        'persistent_mean_rank_delta': m[d]['persistent_mean_rank_delta']}
                                    for d, official in [('T2V', 'official_T2V'), ('V2T', 'V2T')]}}
            result['variants'][name] = entry
            dump(path, result)
            print(json.dumps({'variant': name, **entry}), flush=True)
        result.update(status='completed', wall_seconds=time.time()-start,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                      limits=['Inference-only fixed-backbone ablation, not training-consistent comparison.',
                              'All8 cells reported; repeated PH-dev exploration, no method selection.',
                              'Mask correction and global/local mixtures are not automatically novel.'])
        dump(path, result)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-start)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
