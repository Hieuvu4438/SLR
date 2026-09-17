"""Fixed two-stream fusion-location diagnosis; no fitting or novelty claim."""
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
from slr_common.upstream.cico_bridge import CiCoBridge
from slr_common.upstream.factory import _load_cico_core_from_state
from .common import ART, ROOT, dump, metrics, ranks, sha
from .temporal_view_probe import OUT, score_video


def mix_stream_scores(aware, agnostic):
    return .1*np.asarray(aware, dtype=np.float64) + .9*np.asarray(agnostic, dtype=np.float64)


@torch.inference_mode()
def main():
    path = OUT/'AS-C21-STREAMS_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    started = time.time()
    report = {'experiment_id': 'AS-C21-STREAMS', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C21_protocol.md'),
              'method_go': False, 'optimizer_updates': 0, 'test_loaded': False, 'variants': {}}
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
        base = np.load(ART/'baseline_dev_scores.npy')
        base_ch = np.load(ART/'baseline_dev_channels.npy')
        loaders = [DataLoader(CiCoFeatureDataset(ROOT/'artifacts/manifests/ph_dev.jsonl', feature_len=64,
                             alpha=a, split='dev'), batch_size=128, num_workers=0, shuffle=False) for a in (.9, 0., 1.)]
        enc = {k: [] for k in ('early', 'repeat', 'aware', 'agnostic')}
        offset, max_fusion_error = 0, 0.
        for early, aware, agnostic in zip(*loaders, strict=True):
            if time.time()-started > 300:
                raise TimeoutError('AS-C21 hard timeout300s')
            n = len(early['pair_id'])
            assert early['pair_id'] == aware['pair_id'] == agnostic['pair_id'] == cache['ids'][offset:offset+n]
            for key in ('valid', 'dense_index'):
                assert torch.equal(early[key], aware[key]) and torch.equal(early[key], agnostic[key])
            delta = float((early['h'] - (.1*aware['h']+.9*agnostic['h'])).abs().max())
            max_fusion_error = max(max_fusion_error, delta)
            assert delta <= 1e-6
            for name, batch in [('early', early), ('repeat', early), ('aware', aware), ('agnostic', agnostic)]:
                v = bridge.encode_video(batch['h'].cuda(), batch['valid'].cuda())
                assert torch.equal(v.mask.cpu(), cache['video_mask'][offset:offset+n])
                assert torch.isfinite(v.tokens).all()
                if name in ('early', 'repeat'):
                    assert torch.equal(v.tokens.cpu(), cache['video_tokens'][offset:offset+n])
                enc[name].append(v.tokens.cpu())
            offset += n
        assert offset == 519
        ch = {k: score_video(torch.cat(v), cache) for k, v in enc.items()}
        assert np.array_equal(ch['early'], ch['repeat'])
        assert float(np.abs(ch['early']-base_ch).max()) <= 2e-5
        assert all(np.array_equal(ranks(ch['early'].mean(0))[d], ranks(base)[d]) for d in ('T2V', 'V2T'))
        report['identity'] = {'tokens_exact': True, 'repeat_channels_exact': True,
                              'max_channel_error': float(np.abs(ch['early']-base_ch).max()),
                              'max_input_fusion_error': max_fusion_error, 'rank_parity': True}
        scores = {k: v.mean(0) for k, v in ch.items()}
        scores['early_two_pass'] = (scores['early'].astype(np.float64)+scores['repeat'])/2
        scores['late_fixed'] = mix_stream_scores(scores['aware'], scores['agnostic'])
        scores['late_shifted_aware'] = mix_stream_scores(np.roll(scores['aware'], 1, axis=0), scores['agnostic'])
        for name, score in scores.items():
            m = metrics(score, base)
            np.save(ART/f'AS-C21-{name}_scores.npy', score)
            dump(OUT/f'AS-C21-{name}_metrics.json', m)
            report['variants'][name] = {'mean_R1': m['official_mean_R1'],
                'directions': {d: {**{k: m[o][k] for k in ('R1', 'R5', 'R10')},
                                    'persistent_mean_rank_delta': m[d]['persistent_mean_rank_delta']}
                               for d, o in [('T2V', 'official_T2V'), ('V2T', 'V2T')]}}
        rs = {k: ranks(v) for k, v in scores.items()}
        report['component_correct_overlap'] = {d: {
            'aware_correct_n': int((rs['aware'][d] == 0).sum()),
            'agnostic_correct_n': int((rs['agnostic'][d] == 0).sum()),
            'both_correct_n': int(((rs['aware'][d] == 0)&(rs['agnostic'][d] == 0)).sum()),
            'union_correct_n_NONDEPLOYABLE': int(((rs['aware'][d] == 0)|(rs['agnostic'][d] == 0)).sum()),
            'either_component_correct_early_wrong_n': int((((rs['aware'][d] == 0)|(rs['agnostic'][d] == 0))&(rs['early'][d] > 0)).sum())}
            for d in ('T2V', 'V2T')}
        control, late, shifted = [report['variants'][k] for k in ('early_two_pass', 'late_fixed', 'late_shifted_aware')]
        report['lead_gate'] = (late['mean_R1']-control['mean_R1'] >= .5 and late['mean_R1']-shifted['mean_R1'] >= .5
            and all(late['directions'][d]['R1']-control['directions'][d]['R1'] >= -.25
                    and all(late['directions'][d][k]-control['directions'][d][k] >= -.5 for k in ('R5', 'R10'))
                    and late['directions'][d]['persistent_mean_rank_delta'] < 0 for d in ('T2V', 'V2T')))
        report.update(status='completed', wall_seconds=time.time()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated())
        dump(path, report)
        print(json.dumps(report, indent=2))
    except Exception:
        report.update(status='failed', wall_seconds=time.time()-started, traceback=traceback.format_exc())
        dump(path, report)
        raise


if __name__ == '__main__':
    main()
