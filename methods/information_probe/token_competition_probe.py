"""Fixed inner-competition endpoints, keeping legacy inner/outer mask roles."""
import json
import os
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from .common import ART, ROOT, dump, metrics, ranks, sha

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'


def reduce_channels(a, vm, tm, scale, mode):
    vv, tv = vm == 0, tm == 1
    assert vv.any(-1).all() and tv.any(-1).all()
    if mode == 'soft':
        av = (a*(a/.07).softmax(-1)).sum(-1)
        at = (a*(a/.07).softmax(-2)).sum(-2)
    elif mode == 'hard':
        av, at = a.max(-1).values, a.max(-2).values
    elif mode == 'mean':
        av, at = a.mean(-1), a.mean(-2)
    else:
        raise ValueError(mode)
    return (scale*(av*vv[:, None]).sum(-1)/vv.sum(-1)[:, None],
            scale*(at*tv[None]).sum(-1)/tv.sum(-1)[None])


@torch.inference_mode()
def main():
    path = OUT/'AS-C24-ENDPOINTS_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    started = time.time()
    result = {'experiment_id': 'AS-C24-ENDPOINTS', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C24_protocol.md'),
              'method_go': False, 'updates': 0, 'test_loaded': False, 'variants': {}}
    dump(path, result)
    try:
        cache = torch.load(ART/'frozen_dev.pt', weights_only=True)
        assert cache['manifest_sha256'] == sha(ROOT/'artifacts/manifests/ph_dev.jsonl')
        assert cache['checkpoint_sha256'] == sha(ROOT/'runs/ph_base_b512_s42/checkpoints/best_dev.pt')
        result.update(cache_sha256=sha(ART/'frozen_dev.pt'), manifest_sha256=cache['manifest_sha256'], checkpoint_sha256=cache['checkpoint_sha256'])
        base, base_ch = np.load(ART/'baseline_dev_scores.npy'), np.load(ART/'baseline_dev_channels.npy')
        v, t = [F.normalize(cache[k].cuda(), dim=-1) for k in ('video_tokens', 'text_tokens')]
        vm, tm = [cache[k].cuda() for k in ('video_mask', 'text_mask')]
        ch = {m: np.empty((2, 519, 519), np.float32) for m in ('soft', 'hard', 'mean')}
        for i in range(0, 519, 64):
            if time.time()-started > 300:
                raise TimeoutError('AS-C24 timeout300s')
            for j in range(0, 519, 64):
                a = torch.einsum('ifd,jld->ijfl', v[i:i+64], t[j:j+64])
                for mode in ch:
                    values = reduce_channels(a, vm[i:i+64], tm[j:j+64], cache['logit_scale'], mode)
                    ch[mode][:, i:i+64, j:j+64] = torch.stack(values).cpu().numpy()
        error = float(np.abs(ch['soft']-base_ch).max())
        assert error <= 2e-5, error
        assert all(np.array_equal(ranks(ch['soft'].mean(0))[d], ranks(base)[d]) for d in ('T2V', 'V2T'))
        assert (ch['mean'] <= ch['soft']+2e-5).all() and (ch['soft'] <= ch['hard']+2e-5).all()
        result['identity'] = {'max_channel_error': error, 'rank_parity': True, 'endpoint_order': True}
        for mode, value in ch.items():
            np.save(ART/f'AS-C24-{mode}_channels.npy', value)
        scores = {'soft': ch['soft'].mean(0)}
        for mode in ('hard', 'mean'):
            scores[f'{mode}_both'] = ch[mode].mean(0)
            scores[f'{mode}_A'] = (ch[mode][0]+ch['soft'][1])/2
            scores[f'{mode}_B'] = (ch['soft'][0]+ch[mode][1])/2
        bm = metrics(base, base)
        for name, s in scores.items():
            m = metrics(s, base)
            np.save(ART/f'AS-C24-{name}_scores.npy', s)
            dump(OUT/f'AS-C24-{name}_metrics.json', m)
            entry = {'mean_R1': m['official_mean_R1'], 'directions': {}}
            for d, o in [('T2V', 'official_T2V'), ('V2T', 'V2T')]:
                delta = np.array(m[d]['rank_delta'])
                entry['directions'][d] = {**{k: m[o][k] for k in ('R1', 'R5', 'R10')},
                    'persistent_mean_rank_delta': m[d]['persistent_mean_rank_delta'],
                    'improved_rank_n': int((delta < 0).sum()), 'worsened_rank_n': int((delta > 0).sum()),
                    'mean_fixed_margin_change': float(np.mean(np.array(m[d]['fixed_hard_margin'])-bm[d]['fixed_hard_margin']))}
            entry['lead_gate'] = (m['official_mean_R1']-bm['official_mean_R1'] >= .5
                and all(m[o]['R1']-bm[o]['R1'] >= -.25 and all(m[o][k]-bm[o][k] >= -.5 for k in ('R5', 'R10'))
                        and m[d]['persistent_mean_rank_delta'] < 0 for d, o in [('T2V', 'official_T2V'), ('V2T', 'V2T')]))
            result['variants'][name] = entry
        result.update(status='completed', wall_seconds=time.time()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated())
        dump(path, result)
        print(json.dumps(result, indent=2))
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
