"""AS-C34 fixed crossed encoder sources; no best-cell selection or learned routing."""
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from slr_common.data.cico_dataset import CiCoFeatureDataset
from slr_common.data.tokenize import CiCoCollator
from slr_common.upstream.cico_bridge import CiCoBridge
from slr_common.upstream.factory import _load_cico_core_from_state, load_cico_tokenizer
from .checkpoint_cross_probe import encode, score
from .common import ART, ROOT, dump, metrics, rows, sha
from .seed_ensemble_probe import OUT, SEEDS, bootstrap_delta, uniform_mean


def grid_aggregates(grid):
    n = len(grid)
    assert n == 3 and all(len(row) == n for row in grid)
    out = {'matched3': uniform_mean([grid[i][i] for i in range(n)]),
           'mismatched6': uniform_mean([grid[i][j] for i in range(n) for j in range(n) if i != j]),
           'all9': uniform_mean([s for row in grid for s in row])}
    for i, seed in enumerate(SEEDS):
        out[f'fixed_video_{seed}_mean_text3'] = uniform_mean(grid[i])
        out[f'fixed_text_{seed}_mean_video3'] = uniform_mean([grid[j][i] for j in range(n)])
    return out


def comparison(candidate, reference, labels, label):
    value = bootstrap_delta(candidate, reference, labels)
    value['scope'] = f'{label}; conditional on fixed full519 gallery, not training/selection uncertainty'
    return value


@torch.inference_mode()
def main():
    path = OUT/'AS-C34-ENCODER-GRID_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    started = time.time()
    result = {'experiment_id': 'AS-C34-ENCODER-GRID', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C34_protocol.md'),
              'updates': 0, 'method_go': False, 'test_loaded': False, 'encoders': {}, 'cells': {}, 'aggregates': {}}
    dump(path, result)
    try:
        parent_path = OUT/'AS-C32-ENSEMBLE_run.json'
        parent = json.loads(parent_path.read_text())
        assert parent['status'] == 'completed'
        result['parent_run_sha256'] = sha(parent_path)
        rec = rows('dev')
        ids = [r['pair_id'] for r in rec]
        labels = [r['video_id'].rsplit('-', 1)[0] for r in rec]
        manifest = ROOT/'artifacts/manifests/ph_dev.jsonl'
        assert sha(manifest) == parent['manifest_sha256']
        result['manifest_sha256'] = sha(manifest)
        encoders, bridges = [], []
        for seed in SEEDS:
            if time.time()-started > 300:
                raise TimeoutError('AS-C34 timeout300s')
            root = ROOT/f'runs/ph_base_b512_s{seed}'
            cp = root/'checkpoints/best_dev.pt'
            assert sha(cp) == parent['inputs'][str(seed)]['checkpoint_sha256']
            config = yaml.safe_load((root/'resolved_config.yaml').read_text())
            cico_root = ROOT/config['upstream']['cico_root']
            sys.path.insert(0, str(cico_root))
            saved = torch.load(cp, weights_only=True, map_location='cpu', mmap=True)
            state = {k.removeprefix('core.'): v for k, v in saved['model'].items() if k.startswith('core.')}
            core = _load_cico_core_from_state(config, state, cico_root=cico_root, device='cuda')
            core.eval().requires_grad_(False)
            dataset = CiCoFeatureDataset(manifest, feature_len=64, alpha=.9, split='dev')
            loader = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=False,
                                collate_fn=CiCoCollator(load_cico_tokenizer(config), 32, augment=False))
            video, text, actual_ids = encode(core, loader)
            assert actual_ids == ids
            encoders.append((video, text))
            bridges.append(CiCoBridge(core))
            result['encoders'][str(seed)] = {'checkpoint_sha256': sha(cp), 'config_sha256': sha(root/'resolved_config.yaml'),
                                             'logit_scale': float(core.clip.logit_scale.exp()), 'ids_exact': True}
            del saved, state
        assert all(torch.equal(encoders[0][k].mask, e[k].mask) for e in encoders for k in (0, 1))
        base = np.load(ART/'baseline_dev_scores.npy')
        grid = [[None]*3 for _ in range(3)]
        repeats = [[], [], []]
        for i, vi in enumerate(SEEDS):
            for j, ti in enumerate(SEEDS):
                if time.time()-started > 300:
                    raise TimeoutError('AS-C34 timeout300s')
                s = score(bridges[i], encoders[i][0], encoders[j][1])
                if i == j:
                    original = np.load(ROOT/f'runs/ph_base_b512_s{vi}/evaluation/dev/scores_video_x_text.npy')
                    assert sha(ROOT/f'runs/ph_base_b512_s{vi}/evaluation/dev/scores_video_x_text.npy') == parent['inputs'][str(vi)]['score_sha256']
                    assert np.array_equal(s, original), f'diagonal{vi} must replay exactly'
                    repeats[i].append(s)
                    for _ in range(2):
                        again = score(bridges[i], encoders[i][0], encoders[i][1])
                        assert np.array_equal(again, original)
                        repeats[i].append(again)
                grid[i][j] = s
                name = f'video_{vi}_text_{ti}'
                target = ART/f'AS-C34-{name}_scores.npy'
                if target.exists():
                    raise FileExistsError(target)
                np.save(target, s)
                m = metrics(s, base)
                result['cells'][name] = {'metrics': m, 'score_sha256': sha(target),
                                         'scorer_and_scale_source': vi, 'diagonal_exact': i == j}
                dump(path, result)
                print(json.dumps({'cell': name, 'mean_R1': m['official_mean_R1']}), flush=True)
        agg = grid_aggregates(grid)
        original_ensemble_path = ART/'AS-C32-uniform3_scores.npy'
        assert sha(original_ensemble_path) == parent['ensemble_score_sha256']
        assert np.array_equal(agg['matched3'], np.load(original_ensemble_path))
        control6 = uniform_mean([x for group in repeats for x in group[:2]])
        control9 = uniform_mean([x for group in repeats for x in group])
        assert np.array_equal(control6, agg['matched3']) and np.array_equal(control9, agg['matched3'])
        result['repeated_scoring_controls'] = {'six_pass_exact': True, 'nine_pass_exact': True,
                                               'actual_total_full_gallery_score_calls': 15}
        for name, s in agg.items():
            m = metrics(s, base)
            target = ART/f'AS-C34-{name}_scores.npy'
            if target.exists():
                raise FileExistsError(target)
            np.save(target, s)
            result['aggregates'][name] = {'metrics': m, 'score_sha256': sha(target),
                'versus_matched3': comparison(s, agg['matched3'], labels, f'{name} minus matched3')}
            print(json.dumps({'aggregate': name, 'mean_R1': m['official_mean_R1'],
                              'vs_matched3': result['aggregates'][name]['versus_matched3']}), flush=True)
        ci = result['aggregates']['mismatched6']['versus_matched3']
        result['pairing_dependence_signal'] = ci['mean_pp'] <= -.5 and ci['upper'] < 0
        result.update(status='completed', wall_seconds=time.time()-started,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(), selected_cell=None)
        dump(path, result)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
