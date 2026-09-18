"""Fresh full-gallery PH DEV replay using established shared infrastructure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'shared'))
from inventory import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--checkpoint', default='runs/ph_base_b512_s42/checkpoints/best_dev.pt')
    p.add_argument('--historical', default='runs/ph_base_b512_s42/evaluation/dev/scores_video_x_text.npy')
    cli = p.parse_args()
    out = ROOT / 'artifacts/slret_goal' / cli.run_id
    out.mkdir(parents=True, exist_ok=False)
    report = {'run_id': cli.run_id, 'status': 'running', 'pid': os.getpid(), 'start_unix': time.time(),
        'command': sys.argv, 'selection_split': 'historical_PH_dev_only_no_new_selection',
        'evaluation_split': 'official_dev_519', 'optimizer_updates': 0,
        'runner_sha256': sha(__file__), 'artifacts': str(out),
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT)).hexdigest()}
    def record():
        (out / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        with (ROOT / 'research/slret_goal/experiments.jsonl').open('a') as f:
            f.write(json.dumps(report, allow_nan=False) + '\n')
    record()
    try:
        import numpy as np
        import torch
        import yaml
        from torch.utils.data import DataLoader
        from slr_common.data.cico_dataset import CiCoFeatureDataset
        from slr_common.data.tokenize import CiCoCollator
        from slr_common.upstream.cico_bridge import CiCoBridge, VideoEncoding, TextEncoding
        from slr_common.upstream.factory import _load_cico_core_from_state, load_cico_tokenizer
        from slr_common.evaluation.cico_eval import evaluate_score_matrix
        torch.set_num_threads(8)
        torch.manual_seed(42)
        np.random.seed(42)
        config_path = ROOT / 'runs/ph_base_b512_s42/resolved_config.yaml'
        config = yaml.safe_load(config_path.read_text())
        checkpoint = ROOT / cli.checkpoint
        digest = sha(checkpoint)
        if checkpoint.name == 'best_dev.pt':
            selection_path = checkpoint.parents[1] / 'selection.json'
            selection = json.loads(selection_path.read_text())
            assert selection['checkpoint_sha256'] == digest
            report['selection_sha256'] = sha(selection_path)
            # Trusted local training checkpoint contains RNG/optimizer objects
            # unsupported by PyTorch2.3's restricted unpickler. Hash-verify first.
            raw = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        else:
            raw = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if 'model' in raw:
            state = {k.removeprefix('core.'): v for k, v in raw['model'].items() if k.startswith('core.')}
        else:
            state = raw
        cico_root = ROOT / config['upstream']['cico_root']
        sys.path.insert(0, str(cico_root))
        core = _load_cico_core_from_state(config, state, cico_root=cico_root, device='cuda')
        core.eval().requires_grad_(False)
        del raw, state
        bridge = CiCoBridge(core)
        manifest = ROOT / 'artifacts/manifests/ph_dev.jsonl'
        data = CiCoFeatureDataset(manifest, feature_len=64, alpha=.9, split='dev')
        assert len(data) == 519
        tokenizer = load_cico_tokenizer(config)
        loader = DataLoader(data, batch_size=128, num_workers=0, shuffle=False,
            collate_fn=CiCoCollator(tokenizer, 32, augment=False))
        report.update(hardware=torch.cuda.get_device_name(), torch=torch.__version__, seed=42,
            checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), config_sha256=sha(config_path),
            manifest_sha256=sha(manifest), feature_regime='BSL5K + H2S-transfer-aware locally re-extracted',
            encoder_batch=128, score_block=128,
            precision='checkpoint-native encoder, FP32 scores',
            source_sha256=sha(cico_root / 'modules/modeling.py'))
        parts = {f'{prefix}_{field}': [] for prefix in ['video', 'text'] for field in ['mask', 'tokens', 'cls']}
        ids = []
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for batch in loader:
                encs = {'video': bridge.encode_video(batch['h'].cuda(), batch['valid'].cuda()),
                    'text': bridge.encode_text(*(x.cuda() for x in batch['clean_text']))}
                for prefix, enc in encs.items():
                    for field in ['mask', 'tokens', 'cls']:
                        parts[f'{prefix}_{field}'].append(getattr(enc, field))
                ids.extend(batch['pair_id'])
                print(json.dumps({'encoded': len(ids), 'total': len(data)}), flush=True)
            cache = {k: torch.cat(v) for k, v in parts.items()}
            assert ids == [json.loads(x)['pair_id'] for x in manifest.read_text().splitlines()]
            channels = np.empty((2, len(ids), len(ids)), np.float32)
            def subset(prefix, start, end):
                cls = VideoEncoding if prefix == 'video' else TextEncoding
                return cls(*(cache[f'{prefix}_{f}'][start:end] for f in ['mask', 'tokens', 'cls']))
            for i in range(0, len(ids), 128):
                for j in range(0, len(ids), 128):
                    a, b = bridge.score(subset('video', i, i+128), subset('text', j, j+128))
                    channels[0, i:i+128, j:j+128] = a.cpu().numpy()
                    channels[1, i:i+128, j:j+128] = b.cpu().numpy()
            scores = channels.mean(0)
        np.save(out / 'scores_video_x_text.npy', scores)
        mapping = {x: [x] for x in ids}
        metrics = evaluate_score_matrix(scores, video_ids=ids, text_ids=ids, video_to_text=mapping, text_to_video=mapping)
        (out / 'metrics.json').write_text(json.dumps(metrics) + '\n')
        (out / 'ids.json').write_text(json.dumps(ids) + '\n')
        report['metrics'] = {d: {k: metrics[d][k] for k in ['R1', 'R5', 'R10', 'MedianR', 'MeanR']} for d in ['T2V', 'V2T']}
        report['mean_R1'] = (metrics['T2V']['R1'] + metrics['V2T']['R1']) / 2
        report['score_sha256'] = sha(out / 'scores_video_x_text.npy')
        report['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
        if cli.historical:
            old = np.load(ROOT / cli.historical)
            old_metrics = evaluate_score_matrix(old, video_ids=ids, text_ids=ids, video_to_text=mapping, text_to_video=mapping)
            report['parity'] = {'historical_sha256': sha(ROOT / cli.historical),
                'max_absolute_score_delta': float(np.max(np.abs(old - scores))),
                'rank_changes': {d: sum(a != b for a, b in zip(metrics[d]['cols'], old_metrics[d]['cols'])) for d in ['T2V', 'V2T']}}
            if report['parity']['max_absolute_score_delta'] > 1e-4 or any(report['parity']['rank_changes'].values()):
                report['parity_gate'] = 'failed_investigate_no_training'
            else:
                report['parity_gate'] = 'passed'
        report.update(status='completed', exit_status=0)
        print(json.dumps({k: report.get(k) for k in ['mean_R1', 'metrics', 'parity', 'parity_gate']}), flush=True)
    except Exception:
        report.update(status='failed', exit_status=1, error=traceback.format_exc())
        raise
    finally:
        report['end_unix'] = time.time()
        report['wall_seconds'] = report['end_unix'] - report['start_unix']
        record()


if __name__ == '__main__':
    main()
