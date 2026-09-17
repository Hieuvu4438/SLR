"""Exact baseline augmentation stress; no fitting or new-method claim."""
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import yaml

from slr_common.data.tokenize import augment_caption, encode_cico_text
from slr_common.upstream.cico_bridge import CiCoBridge, TextEncoding
from slr_common.upstream.factory import _load_cico_core_from_state, load_cico_tokenizer
from .cache_baseline import subset
from .common import ART, ROOT, dump, metrics, rows, sha
from .direction_probe import paired

OUT = ROOT / 'docs/proposal7/evidence/autonomous_search'


def captions(records, seed):
    return [r['caption_model'] if seed is None else augment_caption(
        r['caption_model'], r['pair_id'], seed=seed, epoch=0) for r in records]


@torch.inference_mode()
def encode(bridge, tokenizer, texts):
    parts = []
    for i in range(0, len(texts), 128):
        triples = [encode_cico_text(t, tokenizer, 32) for t in texts[i:i+128]]
        enc = bridge.encode_text(*(torch.stack([x[k] for x in triples]).cuda() for k in range(3)))
        parts.append(enc)
    return TextEncoding(*(torch.cat([getattr(x, k) for x in parts]) for k in ('mask', 'tokens', 'cls')))


@torch.inference_mode()
def train_margins(cache, enc, hard):
    v, vm = cache['video_tokens'].cuda(), cache['video_mask'].cuda()
    result = {}
    for direction, key in [('T2V', 'text_confuser'), ('V2T', 'video_confuser')]:
        chunks = []
        for i in range(0, len(v), 128):
            idx = torch.arange(i, min(i+128, len(v)), device='cuda')
            other = hard[key][i:i+128].cuda()
            vi, ti = (other, idx) if direction == 'T2V' else (idx, other)
            pos = paired(v[idx], enc.tokens[idx], vm[idx], enc.mask[idx], cache['logit_scale'])
            neg = paired(v[vi], enc.tokens[ti], vm[vi], enc.mask[ti], cache['logit_scale'])
            chunks.append((pos-neg).cpu().numpy())
        result[direction] = np.concatenate(chunks)
    return result


@torch.inference_mode()
def dev_channels(bridge, cache, enc):
    n = len(cache['ids'])
    result = np.empty((2, n, n), dtype=np.float32)
    for i in range(0, n, 128):
        for j in range(0, n, 128):
            text = TextEncoding(*(getattr(enc, k)[j:j+128] for k in ('mask', 'tokens', 'cls')))
            a, b = bridge.score(subset(cache, 'video', i, i+128), text)
            result[:, i:i+128, j:j+128] = torch.stack((a, b)).cpu().numpy()
    return result


def main():
    report_path = OUT / 'AS-C08-TEXT-AUG_run.json'
    if report_path.exists():
        raise FileExistsError(report_path)
    torch.set_num_threads(8)
    start = time.time()
    checkpoint = ROOT / 'runs/ph_base_b512_s42/checkpoints/best_dev.pt'
    config = yaml.safe_load((checkpoint.parents[1] / 'resolved_config.yaml').read_text())
    report = {'experiment_id': 'AS-C08-TEXT-AUG', 'status': 'running', 'pid': os.getpid(),
              'start_unix': start, 'command': sys.argv, 'checkpoint_sha256': sha(checkpoint),
              'code_sha256': sha(__file__), 'tokenize_sha256': sha(ROOT/'shared/slr_common/data/tokenize.py'),
              'paired_kernel_sha256': sha(ROOT/'methods/information_probe/direction_probe.py'),
              'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'trainable_parameters': 0, 'updates': 0, 'selector': None,
              'augmentation_seeds': [42, 1337, 2026], 'epoch': 0,
              'gpu': torch.cuda.get_device_name(), 'splits': {}, 'method_go': False}
    dump(report_path, report)
    try:
        cico_root = ROOT / config['upstream']['cico_root']
        sys.path.insert(0, str(cico_root))
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
        state = {k.removeprefix('core.'): v for k, v in saved['model'].items() if k.startswith('core.')}
        core = _load_cico_core_from_state(config, state, cico_root=cico_root, device='cuda')
        core.eval().requires_grad_(False)
        bridge, tokenizer = CiCoBridge(core), load_cico_tokenizer(config)
        for split in ('dev', 'train'):
            records, cache = rows(split), torch.load(ART/f'frozen_{split}.pt', weights_only=True)
            assert cache['ids'] == [r['pair_id'] for r in records]
            assert cache['checkpoint_sha256'] == report['checkpoint_sha256']
            assert cache['manifest_sha256'] == sha(ROOT/f'artifacts/manifests/ph_{split}.jsonl')
            clean_texts = captions(records, None)
            clean = encode(bridge, tokenizer, clean_texts)
            assert torch.equal(clean.tokens.cpu(), cache['text_tokens']), 'clean token parity failed'
            assert torch.equal(clean.mask.cpu(), cache['text_mask'])
            keys = [tuple(encode_cico_text(t, tokenizer, 32)[0].tolist()) for t in clean_texts]
            entry = {'manifest_sha256': cache['manifest_sha256'], 'clean_token_parity': True, 'variants': {}}
            if split == 'dev':
                base = np.load(ART/'baseline_dev_scores.npy')
                clean_ch = dev_channels(bridge, cache, clean)
                assert np.array_equal(clean_ch.mean(0), base), 'clean score parity failed'
            else:
                hard = torch.load(ART/'train_confusers.pt', weights_only=True)
                assert hard['manifest_sha256'] == cache['manifest_sha256']
                clean_delta = train_margins(cache, clean, hard)
            for seed in report['augmentation_seeds']:
                texts = captions(records, seed)
                enc = encode(bridge, tokenizer, texts)
                aug_keys = [tuple(encode_cico_text(t, tokenizer, 32)[0].tolist()) for t in texts]
                changed = np.array([a != b for a, b in zip(keys, aug_keys)])
                result = {'changed_deployed_inputs': int(changed.sum())}
                if split == 'dev':
                    aug_ch = dev_channels(bridge, cache, enc)
                    for mode, scores in [('both_aug', aug_ch.mean(0)),
                                         ('A_clean_B_aug', (clean_ch[0]+aug_ch[1])/2)]:
                        m = metrics(scores, base)
                        name = f'AS-C08-{split}-s{seed}-{mode}'
                        np.save(ART/f'{name}_scores.npy', scores)
                        dump(OUT/f'{name}_metrics.json', m)
                        result[mode] = {'T2V_R1': m['official_T2V']['R1'], 'V2T_R1': m['V2T']['R1'],
                                        'mean_R1': m['official_mean_R1']}
                else:
                    augmented = train_margins(cache, enc, hard)
                    for direction, key in [('T2V', 'text_confuser'), ('V2T', 'video_confuser')]:
                        same = np.array([keys[i] == keys[int(j)] for i, j in enumerate(hard[key])])
                        baseline_margin = clean_delta[direction].mean(1)
                        for mode, margin in [('clean', baseline_margin),
                                             ('both_aug', augmented[direction].mean(1)),
                                             ('A_clean_B_aug', (clean_delta[direction][:, 0]+augmented[direction][:, 1])/2)]:
                            wrong, tied = margin < -1e-4, np.abs(margin) <= 1e-4
                            result[f'{direction}_{mode}'] = {
                                'negative_n': int(wrong.sum()), 'near_tie_n': int(tied.sum()),
                                'different_clean_input_nonpositive_n': int(((wrong|tied)&~same).sum()),
                                'same_clean_input_nonpositive_n': int(((wrong|tied)&same).sum()),
                                'lost_positive_margin_n': int(((baseline_margin>1e-4)&(margin<=1e-4)).sum()),
                                'gained_positive_margin_n': int(((baseline_margin<=1e-4)&(margin>1e-4)).sum()),
                                'mean_margin': float(margin.mean())}
                            np.save(ART/f'AS-C08-{split}-s{seed}-{direction}-{mode}_margins.npy', margin)
                entry['variants'][str(seed)] = result
                report['splits'][split] = entry
                dump(report_path, report)
                print(json.dumps({'split': split, 'seed': seed, **result}), flush=True)
            del cache, clean, enc
        report.update(status='completed', wall_seconds=time.time()-start,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                      limits=['Frozen perturbation, not training ablation or semantic minimal pair.',
                              'Train fixed confusers are not full-gallery retrieval.',
                              'Seeds vary augmentation only, not backbone training.',
                              'A clean / B augmented matches scorer channel usage, not the full training objective.'])
        dump(report_path, report)
    except Exception:
        report.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-start)
        dump(report_path, report)
        raise


if __name__ == '__main__':
    main()
