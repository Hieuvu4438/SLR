"""AS-C01: measured scorer restrictions, not claims of linguistic causality."""
import json
import subprocess
import time

import numpy as np
import torch

from .common import ART, ROOT, dump, metrics, sha
from .scoring import channels, pooled

OUT = ROOT / 'docs/proposal7/evidence/autonomous_search'


@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    start = time.time()
    cache = torch.load(ART / 'frozen_dev.pt', weights_only=True)
    baseline = np.load(ART / 'baseline_dev_scores.npy')
    original_channels = np.load(ART / 'baseline_dev_channels.npy')
    result = {'experiment_id': 'AS-C01-SCORER', 'status': 'running',
              'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'checkpoint_sha256': cache['checkpoint_sha256'],
              'manifest_sha256': cache['manifest_sha256'],
              'code_sha256': sha(__file__), 'scoring_code_sha256': sha(ROOT / 'methods/information_probe/scoring.py'),
              'seed': 42, 'updates': 0, 'trainable_parameters': 0,
              'selector': None, 'precision': 'float32', 'variants': {}}
    scale = cache['logit_scale']
    gpu = {k: v.cuda() for k, v in cache.items() if torch.is_tensor(v)}
    for mode in ('legacy', 'masked', 'reverse_context_tokens', 'pooled'):
        v, t = gpu['video_tokens'], gpu['text_tokens']
        vm, tm = gpu['video_mask'], gpu['text_mask']
        if mode == 'reverse_context_tokens':
            v, vm = v.flip(1), vm.flip(1)
            t, tm = t.flip(1), tm.flip(1)
        if mode == 'pooled':
            v = pooled(v, vm == 0)[:, None].expand_as(v)
            t = pooled(t, tm == 1)[:, None].expand_as(t)
        matrix = np.empty_like(original_channels)
        for i in range(0, len(v), 64):
            for j in range(0, len(t), 64):
                a, b = channels(v[i:i+64], t[j:j+64], vm[i:i+64], tm[j:j+64],
                                scale, masked=mode == 'masked')
                matrix[:, i:i+64, j:j+64] = torch.stack((a, b)).cpu().numpy()
        scores = matrix.mean(0)
        m = metrics(scores, baseline)
        dump(OUT / f'AS-C01-SCORER_{mode}_metrics.json', m)
        np.save(ART / f'scorer_{mode}_scores.npy', scores)
        result['variants'][mode] = {
            'max_score_delta': float(np.max(np.abs(scores-baseline))),
            'max_channel_delta': float(np.max(np.abs(matrix-original_channels))),
            'official_mean_R1': m['official_mean_R1'],
            'T2V_R1': m['official_T2V']['R1'], 'V2T_R1': m['V2T']['R1'],
            'changed_ranks': {d: int(np.count_nonzero(m[d]['rank_delta'])) for d in ('T2V', 'V2T')},
        }
        print(json.dumps({'mode': mode, **result['variants'][mode]}), flush=True)
    # Use each *existing* channel alone; no dev-selected mixture or training.
    for index, name in enumerate(('video_to_text_channel', 'text_to_video_channel')):
        m = metrics(original_channels[index], baseline)
        dump(OUT / f'AS-C01-SCORER_{name}_metrics.json', m)
        result['variants'][name] = {'T2V_R1': m['official_T2V']['R1'],
                                    'V2T_R1': m['V2T']['R1'], 'official_mean_R1': m['official_mean_R1']}
    result['status'] = 'completed'
    result['wall_seconds'] = time.time()-start
    result['interpretation_limits'] = [
        'Post-context permutation invariance does not establish absence of temporal information in tokens.',
        'Mask correction is an implementation-policy intervention, not a novel method.',
        'Directional channel substitutions are descriptive, not dev-selected method results.',
        'Pooled comparison changes the scoring functional and is not a controlled training result.']
    dump(OUT / 'AS-C01-SCORER.json', result)


if __name__ == '__main__':
    main()
