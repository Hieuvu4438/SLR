"""Decompose the frozen expectation scorer's entropy term; no learned method."""
import json
import math
import os
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from .common import ART, ROOT, dump, metrics, ranks, sha
from .token_competition_probe import OUT, reduce_channels


def inner_terms(a, dim, tau=.07):
    logp = (a/tau).log_softmax(dim)
    p = logp.exp()
    expectation = (p*a).sum(dim)
    entropy = -(p*logp).sum(dim)
    lme = tau*(torch.logsumexp(a/tau, dim)-math.log(a.shape[dim]))
    derivative = p*(1+(a-expectation.unsqueeze(dim))/tau)
    return expectation, entropy, lme, derivative


def outer(av, at, vm, tm, scale):
    vv, tv = vm == 0, tm == 1
    return torch.stack((scale*((av*vv[:, None]).sum(-1)/vv.sum(-1)[:, None]),
                        scale*((at*tv[None]).sum(-1)/tv.sum(-1)[None])))


@torch.inference_mode()
def main():
    path = OUT/'AS-C25-ENTROPY_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    started = time.time()
    result = {'experiment_id': 'AS-C25-ENTROPY', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C25_protocol.md'),
              'method_go': False, 'updates': 0, 'test_loaded': False, 'variants': {}}
    dump(path, result)
    try:
        c = torch.load(ART/'frozen_dev.pt', weights_only=True)
        assert c['manifest_sha256'] == sha(ROOT/'artifacts/manifests/ph_dev.jsonl')
        assert c['checkpoint_sha256'] == sha(ROOT/'runs/ph_base_b512_s42/checkpoints/best_dev.pt')
        result.update(cache_sha256=sha(ART/'frozen_dev.pt'), manifest_sha256=c['manifest_sha256'], checkpoint_sha256=c['checkpoint_sha256'])
        v, t = [F.normalize(c[k].cuda(), dim=-1) for k in ('video_tokens', 'text_tokens')]
        vm, tm = [c[k].cuda() for k in ('video_mask', 'text_mask')]
        base, base_ch = np.load(ART/'baseline_dev_scores.npy'), np.load(ART/'baseline_dev_channels.npy')
        matrices = {k: np.empty_like(base_ch) for k in ('soft', 'lme', 'entropy_added')}
        max_algebra = 0.
        for i in range(0, 519, 64):
            if time.time()-started > 300:
                raise TimeoutError('AS-C25 timeout300s')
            for j in range(0, 519, 64):
                a = torch.einsum('ifd,jld->ijfl', v[i:i+64], t[j:j+64])
                e1, h1, u1, _ = inner_terms(a, -1)
                e2, h2, u2, _ = inner_terms(a, -2)
                x = outer(u1, u2, vm[i:i+64], tm[j:j+64], c['logit_scale'])
                z = outer(e1+.07*h1, e2+.07*h2, vm[i:i+64], tm[j:j+64], c['logit_scale'])
                constant = torch.tensor([math.log(32), math.log(65)], device='cuda')[:, None, None]*.07*c['logit_scale']
                max_algebra = max(max_algebra, float((x-(z-constant)).abs().max()))
                for k, val in [('lme', x), ('entropy_added', z),
                               ('soft', torch.stack(reduce_channels(a, vm[i:i+64], tm[j:j+64], c['logit_scale'], 'soft')))]:
                    matrices[k][:, i:i+64, j:j+64] = val.cpu().numpy()
        assert max_algebra <= 5e-5, max_algebra
        assert np.abs(matrices['soft']-base_ch).max() <= 2e-5
        assert all(np.array_equal(ranks(matrices['soft'].mean(0))[d], ranks(base)[d]) for d in ('T2V', 'V2T'))
        result['identity'] = {'rank_parity': True, 'max_soft_channel_error': float(np.abs(matrices['soft']-base_ch).max()),
                              'max_scaled_algebra_error': max_algebra}
        result['algebraic_rank_disagreements'] = {f'{channel}_{d}': int(np.count_nonzero(ranks(matrices['lme'][channel])[d] != ranks(matrices['entropy_added'][channel])[d]))
                                                 for channel in (0, 1) for d in ('T2V', 'V2T')}
        wrong = base.copy()
        np.fill_diagonal(wrong, -np.inf)
        result['negative_similarity_derivatives'] = {}
        for name, vi, ti in [('positive', np.arange(519), np.arange(519)),
                              ('T2V_confuser', wrong.argmax(0), np.arange(519)),
                              ('V2T_confuser', np.arange(519), wrong.argmax(1))]:
            a = torch.einsum('bfd,bld->bfl', v[vi], t[ti])
            entry = {}
            for d, dim, active in [('A', -1, (vm[vi] == 0)[:, :, None]), ('B', -2, (tm[ti] == 1)[:, None, :])]:
                _, _, _, derivative = inner_terms(a, dim)
                active = active.expand_as(derivative)
                entry[d] = {'negative_n': int(((derivative < 0)&active).sum()), 'active_coordinate_n': int(active.sum()),
                            'negative_fraction': float(((derivative < 0)&active).sum()/active.sum())}
            result['negative_similarity_derivatives'][name] = entry
        bm = metrics(base, base)
        scores = {'lme_both': matrices['lme'].mean(0),
                  'lme_A': (matrices['lme'][0]+matrices['soft'][1])/2,
                  'lme_B': (matrices['soft'][0]+matrices['lme'][1])/2}
        for name, s in scores.items():
            m = metrics(s, base)
            dump(OUT/f'AS-C25-{name}_metrics.json', m)
            np.save(ART/f'AS-C25-{name}_scores.npy', s)
            entry = {'mean_R1': m['official_mean_R1'], 'directions': {
                d: {**{k: m[o][k] for k in ('R1', 'R5', 'R10')}, 'persistent_mean_rank_delta': m[d]['persistent_mean_rank_delta']}
                for d, o in [('T2V', 'official_T2V'), ('V2T', 'V2T')]}}
            entry['lead_gate'] = (m['official_mean_R1']-bm['official_mean_R1'] >= .5
                and all(m[o]['R1']-bm[o]['R1'] >= -.25 and all(m[o][k]-bm[o][k] >= -.5 for k in ('R5', 'R10'))
                        and m[d]['persistent_mean_rank_delta'] < 0 for d, o in [('T2V', 'official_T2V'), ('V2T', 'V2T')]))
            result['variants'][name] = entry
        for k, x in matrices.items():
            np.save(ART/f'AS-C25-{k}_channels.npy', x)
        result.update(status='completed', wall_seconds=time.time()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated())
        dump(path, result)
        print(json.dumps(result, indent=2))
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
