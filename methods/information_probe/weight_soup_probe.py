"""AS-C33 fixed uniform weight-average control, explicitly existing prior art."""
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import yaml

from slr_common.data.cico_dataset import CiCoFeatureDataset
from slr_common.data.tokenize import CiCoCollator
from slr_common.upstream.cico_bridge import CiCoBridge
from slr_common.upstream.factory import _load_cico_core_from_state, load_cico_tokenizer
from .cache_baseline import subset
from .common import ART, ROOT, dump, metrics, rows, sha
from .seed_ensemble_probe import OUT, SEEDS, bootstrap_delta


def average_states(states):
    assert states and all(set(x) == set(states[0]) for x in states)
    out = {}
    for key, ref in states[0].items():
        vals = [x[key] for x in states]
        assert all(x.shape == ref.shape and x.dtype == ref.dtype for x in vals)
        if ref.is_floating_point():
            mean = sum((x.double() for x in vals), torch.zeros_like(ref, dtype=torch.float64))/len(vals)
            out[key] = mean.to(ref.dtype)
        else:
            assert all(torch.equal(x, ref) for x in vals)
            out[key] = ref.clone()
    return out


def state_digest(state):
    h = hashlib.sha256()
    for key in sorted(state):
        v = state[key].detach().cpu().contiguous()
        h.update(json.dumps([key, str(v.dtype), list(v.shape)]).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def main():
    path = OUT/'AS-C33-SOUP_run.json'
    if path.exists():
        raise FileExistsError(path)
    started = time.time()
    torch.set_num_threads(8)
    result = {'experiment_id': 'AS-C33-SOUP', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C33_protocol.md'),
              'method_go': False, 'updates': 0, 'test_loaded': False, 'variants': {}}
    dump(path, result)
    try:
        parent_path = OUT/'AS-C32-ENSEMBLE_run.json'
        parent = json.loads(parent_path.read_text())
        assert parent['status'] == 'completed'
        result['parent_run_sha256'] = sha(parent_path)
        states = []
        for seed in SEEDS:
            cp = ROOT/f'runs/ph_base_b512_s{seed}/checkpoints/best_dev.pt'
            assert sha(cp) == parent['inputs'][str(seed)]['checkpoint_sha256']
            saved = torch.load(cp, map_location='cpu', weights_only=True, mmap=True)
            states.append({k.removeprefix('core.'): v for k, v in saved['model'].items() if k.startswith('core.')})
        identity = average_states([states[0]]*3)
        assert all(torch.equal(identity[k], v) for k, v in states[0].items())
        average = average_states(states)
        result['weight_audit'] = {'state_tensor_n': len(average), 'repeat3_all_tensors_exact': True,
                                  'identity_state_sha256': state_digest(identity),
                                  'uniform3_state_sha256': state_digest(average),
                                  'uniform3_tensors_different_from_seed42': sum(not torch.equal(v, states[0][k]) for k, v in average.items())}
        del states, saved
        config = yaml.safe_load((ROOT/'runs/ph_base_b512_s42/resolved_config.yaml').read_text())
        cico_root = ROOT/config['upstream']['cico_root']
        sys.path.insert(0, str(cico_root))
        tok = load_cico_tokenizer(config)
        records = rows('dev')
        manifest = ROOT/'artifacts/manifests/ph_dev.jsonl'
        assert sha(manifest) == parent['manifest_sha256']
        result['manifest_sha256'] = sha(manifest)
        dataset = CiCoFeatureDataset(manifest, feature_len=64, alpha=.9, split='dev')
        items = [dataset[i] for i in range(len(dataset))]
        collator = CiCoCollator(tok, 32, augment=False)
        baseline_cache = torch.load(ART/'frozen_dev.pt', weights_only=True)
        assert baseline_cache['manifest_sha256'] == result['manifest_sha256']
        assert baseline_cache['checkpoint_sha256'] == parent['inputs']['42']['checkpoint_sha256']
        base = np.load(ART/'baseline_dev_scores.npy')
        ensemble = np.load(ART/'AS-C32-uniform3_scores.npy')
        assert sha(ART/'AS-C32-uniform3_scores.npy') == parent['ensemble_score_sha256']
        labels = [r['video_id'].rsplit('-', 1)[0] for r in records]
        for name, state in [('repeat3_42', identity), ('uniform3_weights', average)]:
            if time.time()-started > 300:
                raise TimeoutError('AS-C33 timeout300s')
            core = _load_cico_core_from_state(config, state, cico_root=cico_root, device='cuda')
            core.eval().requires_grad_(False)
            bridge = CiCoBridge(core)
            parts = {f'{p}_{k}': [] for p in ('video', 'text') for k in ('mask', 'tokens', 'cls')}
            ids = []
            for i in range(0, len(items), 128):
                batch = collator(items[i:i+128])
                v = bridge.encode_video(batch['h'].cuda(), batch['valid'].cuda())
                t = bridge.encode_text(*(x.cuda() for x in batch['clean_text']))
                for prefix, enc in [('video', v), ('text', t)]:
                    for field in ('mask', 'tokens', 'cls'):
                        parts[f'{prefix}_{field}'].append(getattr(enc, field).cpu())
                ids.extend(batch['pair_id'])
            cache = {k: torch.cat(v) for k, v in parts.items()}
            assert ids == baseline_cache['ids'] == [r['pair_id'] for r in records]
            if name == 'repeat3_42':
                assert all(torch.equal(v, baseline_cache[k]) for k, v in cache.items())
            n = len(ids)
            ch = np.empty((2, n, n), np.float32)
            for i in range(0, n, 128):
                for j in range(0, n, 128):
                    a, b = bridge.score(subset(cache, 'video', i, i+128), subset(cache, 'text', j, j+128))
                    ch[:, i:i+128, j:j+128] = torch.stack((a, b)).cpu().numpy()
            score = ch.mean(0)
            if name == 'repeat3_42':
                assert np.array_equal(score, base)
                result['identity_cache_and_scores_exact'] = True
            p = ART/f'AS-C33-{name}_scores.npy'
            if p.exists():
                raise FileExistsError(p)
            np.save(p, score)
            m = metrics(score, base)
            result['variants'][name] = {'metrics': m, 'score_sha256': sha(p),
                                         'parameter_n': sum(x.numel() for x in core.parameters()),
                                         'logit_scale': float(core.clip.logit_scale.exp()),
                                         'bootstrap_vs_single42': bootstrap_delta(score, base, labels),
                                         'bootstrap_vs_score_ensemble': bootstrap_delta(score, ensemble, labels)}
            dump(path, result)
            print(json.dumps({'variant': name, 'mean_R1': m['official_mean_R1'],
                              'vs_single42': result['variants'][name]['bootstrap_vs_single42'],
                              'vs_score_ensemble': result['variants'][name]['bootstrap_vs_score_ensemble']}), flush=True)
            del core, bridge, cache, parts
        m = result['variants']['uniform3_weights']['metrics']
        em = parent['variants']['uniform3']
        result['ensemble_retention_gate'] = (m['official_mean_R1']-em['official_mean_R1'] >= -.25
            and all(m[d]['R1']-em[d]['R1'] >= -.25 and all(m[d][f'R{k}']-em[d][f'R{k}'] >= -.5 for k in (5, 10)) for d in ('official_T2V', 'V2T')))
        result.update(status='completed', wall_seconds=time.time()-started,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(), inference_models=1)
        dump(path, result)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
