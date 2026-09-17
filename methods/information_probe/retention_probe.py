"""Train-only aligned ridge predictability; never an information-loss bound."""
import json
import os
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from slr_common.data.cico_dataset import CiCoFeatureDataset
from .common import ART, ROOT, dump, metrics, ranks, sha
from .scoring import channels

OUT = ROOT / 'docs/proposal7/evidence/autonomous_search'


def paired_moments(x, y, weight, chunk=8192):
    """Weighted centered moments, accumulated in double without giant copies."""
    if len(x) != len(y) or weight.shape != (len(x),) or (weight < 0).any():
        raise ValueError('aligned nonnegative weights required')
    if not torch.isclose(weight.sum(), weight.new_tensor(1.), atol=1e-7):
        raise ValueError('weights must sum to one')
    mx, my = torch.zeros(x.shape[1], dtype=torch.double), torch.zeros(y.shape[1], dtype=torch.double)
    xx = torch.zeros(x.shape[1], x.shape[1], dtype=torch.double)
    yy = torch.zeros(y.shape[1], y.shape[1], dtype=torch.double)
    xy = torch.zeros(x.shape[1], y.shape[1], dtype=torch.double)
    for i in range(0, len(x), chunk):
        a, b, w = x[i:i+chunk].double(), y[i:i+chunk].double(), weight[i:i+chunk].double()
        aw, bw = a*w[:, None], b*w[:, None]
        mx += aw.sum(0); my += bw.sum(0)
        xx += a.T@aw; yy += b.T@bw; xy += a.T@bw
    return mx, my, xx-mx[:, None]*mx, yy-my[:, None]*my, xy-mx[:, None]*my


def ridge_map(mx, my, covariance, cross, penalty=.01):
    covariance = (covariance+covariance.T)/2
    ridge = penalty*covariance.trace()/len(mx)
    matrix = torch.linalg.solve(covariance+ridge*torch.eye(len(mx), dtype=torch.double), cross)
    return {'input_mean': mx, 'target_mean': my, 'matrix': matrix, 'ridge': float(ridge)}


def predict(x, fitted):
    return (x-fitted['input_mean'].to(x))@fitted['matrix'].to(x)+fitted['target_mean'].to(x)


def aligned_inputs(split, cache):
    dataset = CiCoFeatureDataset(ROOT/f'artifacts/manifests/ph_{split}.jsonl',
                                feature_len=64, alpha=.9, split=split)
    values = torch.empty(len(dataset), 64, 1024)
    count = 0
    for batch in DataLoader(dataset, batch_size=128, num_workers=4):
        n = len(batch['pair_id'])
        assert batch['pair_id'] == cache['ids'][count:count+n]
        assert torch.equal(batch['valid'].bool(), cache['video_mask'][count:count+n, 1:] == 0)
        values[count:count+n] = F.normalize(batch['h'], dim=-1)
        count += n
        if count % 1024 == 0 or count == len(dataset):
            print(json.dumps({'loading': split, 'rows': count}), flush=True)
    assert count == len(cache['ids'])
    return values


def reconstruction(x, y, valid, fitted, target_mean):
    errors, means, cosines = [], [], []
    for i in range(0, len(x), 128):
        a, b, mask = x[i:i+128].double(), y[i:i+128].double(), valid[i:i+128]
        p = target_mean.expand_as(b) if fitted is None else predict(a, fitted)
        w = mask.double()/mask.sum(1)[:, None]
        errors.extend((((p-b).square().sum(-1))*w).sum(1).tolist())
        means.extend((((target_mean-b).square().sum(-1))*w).sum(1).tolist())
        cosines.extend((F.cosine_similarity(p, b, dim=-1)*w).sum(1).tolist())
    return {'weighted_mse': float(np.mean(errors)), 'mean_predictor_mse': float(np.mean(means)),
            'explained_fraction': float(1-np.mean(errors)/np.mean(means)),
            'mean_cosine': float(np.mean(cosines)),
            'per_sequence_mse': errors, 'per_sequence_mean_mse': means,
            'per_sequence_cosine': cosines}


def replace_valid(original, prediction, valid):
    result = original.clone()
    result[:, 1:] = torch.where(valid[..., None], prediction.to(result), original[:, 1:])
    return result


@torch.inference_mode()
def main():
    path = OUT/'AS-C13-RETENTION_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    started = time.time()
    result = {'experiment_id': 'AS-C13-RETENTION', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C13_protocol.md'),
              'scorer_sha256': sha(ROOT/'methods/information_probe/scoring.py'),
              'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'fit_precision': 'CPU float64', 'gpu': torch.cuda.get_device_name(),
              'control_seed': 42, 'penalty_fraction': .01, 'selector': None,
              'trainable_encoder_parameters': 0, 'method_go': False, 'reconstruction': {}, 'variants': {}}
    dump(path, result)
    try:
        train = torch.load(ART/'frozen_train.pt', weights_only=True)
        dev = torch.load(ART/'frozen_dev.pt', weights_only=True)
        assert train['checkpoint_sha256'] == dev['checkpoint_sha256']
        result['checkpoint_sha256'] = train['checkpoint_sha256']
        result['provenance'] = {}
        for split, cache in [('train', train), ('dev', dev)]:
            assert cache['manifest_sha256'] == sha(ROOT/f'artifacts/manifests/ph_{split}.jsonl')
            result['provenance'][split] = {'cache_sha256': sha(ART/f'frozen_{split}.pt'),
                                            'manifest_sha256': cache['manifest_sha256'], 'n': len(cache['ids'])}
        xt = aligned_inputs('train', train)
        yt = F.normalize(train['video_tokens'][:, 1:], dim=-1)
        vt = train['video_mask'][:, 1:] == 0
        x, y = xt[vt], yt[vt]
        weight = (vt.double()/vt.sum(1)[:, None]/len(vt))[vt]
        maps = {}
        for name, target in [('aligned', y), ('shuffled', y[torch.randperm(len(y), generator=torch.Generator().manual_seed(42))])]:
            mx, my, xx, yy, xy = paired_moments(x, target, weight)
            maps[name+'_forward'] = ridge_map(mx, my, xx, xy)
            maps[name+'_reverse'] = ridge_map(my, mx, yy, xy.T)
            print(json.dumps({'fitted': name, 'valid_slots': len(x)}), flush=True)
        del x, y
        torch.save({'maps': maps, 'provenance': result['provenance'],
                    'checkpoint_sha256': result['checkpoint_sha256']}, ART/'AS-C13-maps.pt')
        xd = aligned_inputs('dev', dev)
        yd = F.normalize(dev['video_tokens'][:, 1:], dim=-1)
        vd = dev['video_mask'][:, 1:] == 0
        for split, a, b, valid in [('train', xt, yt, vt), ('dev', xd, yd, vd)]:
            result['reconstruction'][split] = {}
            for direction, source, target in [('forward', a, b), ('reverse', b, a)]:
                mean = maps['aligned_'+direction]['target_mean']
                for control in ('aligned', 'shuffled', 'mean'):
                    fitted = None if control == 'mean' else maps[control+'_'+direction]
                    measured = reconstruction(source, target, valid, fitted, mean)
                    result['reconstruction'][split][control+'_'+direction] = measured
                    print(json.dumps({'split': split, 'map': control+'_'+direction,
                                      **{k: v for k, v in measured.items() if not k.startswith('per_sequence')}}), flush=True)
            dump(path, result)
        del train, xt, yt
        original, text = dev['video_tokens'].cuda(), dev['text_tokens'].cuda()
        vm, tm = dev['video_mask'].cuda(), dev['text_mask'].cuda()
        base = np.load(ART/'baseline_dev_scores.npy')
        base_ch = np.load(ART/'baseline_dev_channels.npy')
        for name in ('identity', 'aligned_forward', 'shuffled_forward', 'mean', 'roundtrip'):
            if name == 'identity':
                video = original
            else:
                if name == 'mean':
                    pred = maps['aligned_forward']['target_mean'].float().expand_as(yd)
                elif name == 'roundtrip':
                    pred = predict(F.normalize(predict(yd, maps['aligned_reverse']), dim=-1), maps['aligned_forward'])
                else:
                    pred = predict(xd, maps[name])
                video = replace_valid(original, pred.cuda(), vd.cuda())
            scored = np.empty_like(base_ch)
            for i in range(0, len(video), 64):
                for j in range(0, len(text), 64):
                    ab = channels(video[i:i+64], text[j:j+64], vm[i:i+64], tm[j:j+64], dev['logit_scale'])
                    scored[:, i:i+64, j:j+64] = torch.stack(ab).cpu().numpy()
            scores = scored.mean(0)
            if name == 'identity':
                result['identity_max_channel_delta'] = float(np.abs(scored-base_ch).max())
                assert result['identity_max_channel_delta'] <= 2e-5
                assert all(np.array_equal(ranks(scores)[d], ranks(base)[d]) for d in ('T2V', 'V2T'))
            measured = metrics(scores, base)
            dump(OUT/f'AS-C13-{name}_metrics.json', measured)
            np.save(ART/f'AS-C13-{name}_scores.npy', scores)
            result['variants'][name] = {'T2V_R1': measured['official_T2V']['R1'],
                                        'V2T_R1': measured['V2T']['R1'], 'mean_R1': measured['official_mean_R1']}
            print(json.dumps({'variant': name, **result['variants'][name]}), flush=True)
            dump(path, result)
        result.update(status='completed', wall_seconds=time.time()-started,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                      limits=['Linear predictability is not semantic sufficiency or information loss.',
                              'One frozen backbone; no dev tuning; all fixed variants disclosed.',
                              'Original CLS/pad retained: hybrid diagnostic, not encoder removal.',
                              'Repeated PH-dev exploration, not independent confirmation.'])
        dump(path, result)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
