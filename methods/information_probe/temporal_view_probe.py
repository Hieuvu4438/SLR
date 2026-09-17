"""Frozen nearby-view acquisition diagnosis, not sampling-consistency training."""
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from slr_common.data.cico_dataset import CiCoFeatureDataset
from slr_common.upstream.cico_bridge import CiCoBridge
from slr_common.upstream.factory import _load_cico_core_from_state
from .common import ART, ROOT, dump, metrics, ranks, sha
from .scoring import channels

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'


def validate_indices(a, b, valid):
    assert torch.equal(a >= 0, valid) and torch.equal(b >= 0, valid)
    assert ((b-a)[valid].abs() <= 1).all()
    for x in b:
        x = x[x >= 0]
        assert (x[1:] > x[:-1]).all()
    return (a != b).sum(1)


@torch.inference_mode()
def score_video(video, cache):
    v, t = video.cuda(), cache['text_tokens'].cuda()
    vm, tm = cache['video_mask'].cuda(), cache['text_mask'].cuda()
    result = np.empty((2, len(v), len(t)), dtype=np.float32)
    for i in range(0, len(v), 64):
        for j in range(0, len(t), 64):
            ab = channels(v[i:i+64], t[j:j+64], vm[i:i+64], tm[j:j+64], cache['logit_scale'])
            result[:, i:i+64, j:j+64] = torch.stack(ab).cpu().numpy()
    return result


@torch.inference_mode()
def main():
    path = OUT/'AS-C17-VIEWS_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    start = time.time()
    report = {'experiment_id': 'AS-C17-VIEWS', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C17_protocol.md'),
              'view_code_sha256': sha(ROOT/'shared/slr_common/data/views.py'),
              'scorer_sha256': sha(ROOT/'methods/information_probe/scoring.py'),
              'method_go': False, 'training_updates': 0, 'selector': None, 'views': {}, 'variants': {}}
    dump(path, report)
    try:
        checkpoint = ROOT/'runs/ph_base_b512_s42/checkpoints/best_dev.pt'
        config = yaml.safe_load((checkpoint.parents[1]/'resolved_config.yaml').read_text())
        cache = torch.load(ART/'frozen_dev.pt', weights_only=True)
        assert cache['checkpoint_sha256'] == sha(checkpoint)
        assert cache['manifest_sha256'] == sha(ROOT/'artifacts/manifests/ph_dev.jsonl')
        report.update(checkpoint_sha256=sha(checkpoint), cache_sha256=sha(ART/'frozen_dev.pt'),
                      manifest_sha256=cache['manifest_sha256'])
        cico_root = ROOT/config['upstream']['cico_root']
        sys.path.insert(0, str(cico_root))
        raw = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
        state = {k.removeprefix('core.'): v for k, v in raw['model'].items() if k.startswith('core.')}
        core = _load_cico_core_from_state(config, state, cico_root=cico_root, device='cuda')
        core.eval().requires_grad_(False)
        bridge = CiCoBridge(core)
        del raw, state
        base, base_ch = np.load(ART/'baseline_dev_scores.npy'), np.load(ART/'baseline_dev_channels.npy')
        scores_by_name = {}
        for seed in (42, 1337, 2026):
            dataset = CiCoFeatureDataset(ROOT/'artifacts/manifests/ph_dev.jsonl', feature_len=64,
                                        alpha=.9, split='dev', include_jittered_view=True, seed=seed)
            original, jitter, changes, cosine = [], [], [], []
            offset = 0
            for batch in DataLoader(dataset, batch_size=128, num_workers=4, shuffle=False):
                n = len(batch['pair_id'])
                assert batch['pair_id'] == cache['ids'][offset:offset+n]
                valid = batch['valid'].bool()
                assert torch.equal(valid, batch['valid_b'].bool())
                changed = validate_indices(batch['dense_index'], batch['dense_index_b'], valid)
                assert torch.equal(changed > 0, batch['view_independent'].bool())
                a = bridge.encode_video(batch['h'].cuda(), valid.cuda())
                b = bridge.encode_video(batch['h_b'].cuda(), valid.cuda())
                assert torch.equal(a.tokens.cpu(), cache['video_tokens'][offset:offset+n])
                assert torch.equal(a.mask.cpu(), cache['video_mask'][offset:offset+n])
                assert torch.equal(b.mask, a.mask)
                identical = changed == 0
                assert torch.equal(b.tokens.cpu()[identical], a.tokens.cpu()[identical])
                distance = 1-F.cosine_similarity(a.tokens[:, 1:], b.tokens[:, 1:], dim=-1)
                cosine.extend(((distance.cpu()*valid).sum(1)/valid.sum(1)).tolist())
                changes.extend(changed.tolist())
                original.append(a.tokens.cpu()); jitter.append(b.tokens.cpu())
                offset += n
            assert offset == 519
            for name, encoded in [(f'canonical_repeat_{seed}', original), (f'jitter_{seed}', jitter)]:
                scored = score_video(torch.cat(encoded), cache)
                scores_by_name[name] = scored.mean(0)
                if name.startswith('canonical'):
                    delta = float(np.abs(scored-base_ch).max())
                    assert delta <= 2e-5
                    assert all(np.array_equal(ranks(scored.mean(0))[d], ranks(base)[d]) for d in ('T2V', 'V2T'))
                    report.setdefault('identity_max_channel_deltas', []).append(delta)
            report['views'][str(seed)] = {'changed_rows': int(np.count_nonzero(changes)),
                                          'changed_slots_per_row': changes,
                                          'mean_valid_token_cosine_distance': float(np.mean(cosine)),
                                          'per_sequence_cosine_distance': cosine}
            dump(path, report)
            print(json.dumps({'view_seed': seed, 'changed_rows': np.count_nonzero(changes).item(),
                              'mean_context_cosine_distance': float(np.mean(cosine))}), flush=True)
        canonical = [scores_by_name[f'canonical_repeat_{s}'] for s in (42, 1337, 2026)]
        assert all(np.array_equal(canonical[0], x) for x in canonical[1:])
        # Accumulate in double so averaging identical float32 controls is exact.
        scores_by_name['canonical_mean3'] = np.mean(np.stack(canonical).astype(np.float64), axis=0)
        scores_by_name['jitter_mean3'] = np.mean(np.stack([scores_by_name[f'jitter_{s}'] for s in (42, 1337, 2026)]).astype(np.float64), axis=0)
        assert np.array_equal(scores_by_name['canonical_mean3'], canonical[0])
        for name, score in scores_by_name.items():
            m = metrics(score, base)
            dump(OUT/f'AS-C17-{name}_metrics.json', m)
            np.save(ART/f'AS-C17-{name}_scores.npy', score)
            report['variants'][name] = {'mean_R1': m['official_mean_R1'],
                'directions': {d: {**{k: m[o][k] for k in ('R1', 'R5', 'R10')},
                                    'persistent_mean_rank_delta': m[d]['persistent_mean_rank_delta']}
                               for d, o in [('T2V', 'official_T2V'), ('V2T', 'V2T')]}}
            print(json.dumps({'variant': name, **report['variants'][name]}), flush=True)
        control, mean = (report['variants'][x] for x in ('canonical_mean3', 'jitter_mean3'))
        report['exploratory_lead_gate'] = (mean['mean_R1']-control['mean_R1'] >= .5 and
                                          all(mean['directions'][d]['R1']-control['directions'][d]['R1'] >= -.25 for d in ('T2V', 'V2T')))
        report.update(status='completed', wall_seconds=time.time()-start, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                      limits=['One trained checkpoint, not three independent models.',
                              'Fixed-view averaging is not novel; no sampling-consistency training.',
                              'Nearby I3D index changes are not validated semantic/signing-rate perturbations.',
                              'No dev-selected seed, frame count, mixture, or modified positive.'])
        dump(path, report)
    except Exception:
        report.update(status='failed', wall_seconds=time.time()-start, traceback=traceback.format_exc())
        dump(path, report)
        raise


if __name__ == '__main__':
    main()
