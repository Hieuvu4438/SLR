"""CICO-REOPEN-01: locked three-control frozen-CiCo pilot, TRAIN/DEV only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import hashlib
import json
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from .common import ART, ROOT, dump, metrics, rows, sha
from .scoring import balanced_loss
from .sentence_weighting import SentenceWeighting, eot_vectors, frozen_channels_and_values, random_conditioning

RUN = ROOT / 'artifacts/proposal7/CICO-REOPEN-01'
REPORT = ROOT / 'docs/proposal7/evidence/autonomous_search/CICO-REOPEN-01_run.json'
PROTOCOL = ROOT / 'docs/proposal7/evidence/autonomous_search/CICO-REOPEN-01_protocol.md'


def event(**data):
    print(json.dumps(data), flush=True)


def load_cache(split):
    if split not in ('train', 'dev'):
        raise ValueError('TRAIN/DEV only')
    c = torch.load(ART / f'frozen_{split}.pt', map_location='cpu', weights_only=True)
    assert c['manifest_sha256'] == sha(ROOT / f'artifacts/manifests/ph_{split}.jsonl')
    assert c['ids'] == [r['pair_id'] for r in rows(split)]
    assert c['video_tokens'].shape[-1] == c['text_tokens'].shape[-1] == 512
    return {k: v.float().cuda() if k in ('video_tokens', 'text_tokens') else
            v.cuda() if torch.is_tensor(v) else v for k, v in c.items()}


def state_sha(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def batch_schedule(train):
    confusers = torch.load(ART / 'train_confusers.pt', map_location='cpu', weights_only=True)
    assert confusers['checkpoint_sha256'] == train['checkpoint_sha256']
    assert confusers['manifest_sha256'] == train['manifest_sha256']
    assert confusers['split'] == 'train' and confusers['candidate_n'] == len(train['ids'])
    hard = [confusers['video_confuser'], confusers['text_confuser']]
    rng, batches = np.random.default_rng(42), []
    for step in range(260):
        anchors = rng.choice(len(train['ids']), 32, replace=False)
        indexes = list(dict.fromkeys(np.concatenate((anchors, hard[step % 2][anchors].numpy())).tolist()))
        while len(indexes) < 64:
            candidate = int(rng.integers(len(train['ids'])))
            if candidate not in indexes:
                indexes.append(candidate)
        assert len(indexes) == 64 and len(set(indexes)) == 64
        batches.append(indexes)
    return np.asarray(batches, dtype=np.int64)


@torch.no_grad()
def evaluate(model, c, conditioning, base_channels):
    model.eval()
    n = len(c['ids'])
    delta = np.empty((n, n), dtype=np.float32)
    parity_error = 0.
    weight_l1, pairs = 0., 0
    for i in range(0, n, 64):
        for j in range(0, n, 64):
            vi, tj = slice(i, i+64), slice(j, j+64)
            a, b, av = frozen_channels_and_values(c['video_tokens'][vi], c['text_tokens'][tj],
                                                 c['video_mask'][vi], c['text_mask'][tj], c['logit_scale'])
            fresh = np.stack((a.cpu().numpy(), b.cpu().numpy()))
            parity_error = max(parity_error, float(np.max(np.abs(fresh - base_channels[:, vi, tj]))))
            assert parity_error <= 5e-5, f'Baseline channel mismatch: {parity_error}'
            change = model(c['video_tokens'][vi], c['video_mask'][vi], conditioning[tj], av, c['logit_scale'])
            delta[vi, tj] = change.cpu().numpy()
            w, u = model.weights(c['video_tokens'][vi], c['video_mask'][vi], conditioning[tj])
            weight_l1 += float((w-u).abs().sum())
            pairs += w.shape[0] * w.shape[1]
    scores = base_channels.mean(0) + .5 * delta
    assert np.isfinite(scores).all()
    return scores, {'baseline_channel_maxabs': parity_error,
                    'mean_weight_l1_from_uniform': weight_l1 / pairs,
                    'maxabs_A_correction': float(np.abs(delta).max())}


def metric_key(m):
    return m['official_mean_R1'], .5 * (m['official_T2V']['R5'] + m['V2T']['R5'])


def run_arm(mode, train, dev, cond, base_channels, schedule, report):
    torch.manual_seed(42)
    model = SentenceWeighting().cuda()
    directory = RUN / mode
    directory.mkdir()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=260)
    arm = {'status': 'running', 'parameters': sum(p.numel() for p in model.parameters()),
           'initial_state_sha256': state_sha(model), 'history': [], 'train_trace': []}
    report['arms'][mode] = arm
    best = (-float('inf'), -float('inf'))
    started = time.monotonic()
    for step in range(261):
        if step in (0, 130, 260):
            scores, checks = evaluate(model, dev, cond['dev'], base_channels)
            if step == 0:
                assert np.array_equal(scores, base_channels.mean(0)), 'Initialization changed baseline'
            m = metrics(scores, base_channels.mean(0))
            torch.save(model.state_dict(), directory / f'head_step{step}.pt')
            np.save(directory / f'scores_step{step}.npy', scores)
            dump(directory / f'metrics_step{step}.json', m)
            record = {'step': step, 'mean_R1': m['official_mean_R1'],
                      'T2V_R1': m['official_T2V']['R1'], 'V2T_R1': m['V2T']['R1'],
                      'elapsed_seconds': time.monotonic()-started, **checks}
            arm['history'].append(record)
            if metric_key(m) > best:
                best = metric_key(m)
                arm['selected_step'] = step
                arm['selected_metrics'] = m
                torch.save(model.state_dict(), directory / 'selected.pt')
                np.save(directory / 'selected_scores.npy', scores)
            dump(REPORT, report)
            event(event='dev', arm=mode, **record)
        if step == 260:
            break
        model.train()
        idx = torch.tensor(schedule[step], device='cuda')
        a, b, av = frozen_channels_and_values(train['video_tokens'][idx], train['text_tokens'][idx],
                                             train['video_mask'][idx], train['text_mask'][idx], train['logit_scale'])
        delta = model(train['video_tokens'][idx], train['video_mask'][idx], cond['train'][idx], av, train['logit_scale'])
        loss = balanced_loss(a+delta, b)
        assert torch.isfinite(loss), 'Nonfinite loss'
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        scheduler.step()
        arm['train_trace'].append({'step': step+1, 'loss': float(loss.detach()),
                                   'grad_norm_before_clip': float(grad_norm)})
        if (step+1) % 20 == 0:
            dump(REPORT, report)
            event(event='train', arm=mode, **arm['train_trace'][-1])
    arm.update(status='completed', updates=260, elapsed_seconds=time.monotonic()-started,
               selected_checkpoint_sha256=sha(directory / 'selected.pt'),
               final_state_sha256=state_sha(model))
    dump(REPORT, report)


def main():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError('Refusing to overwrite CICO-REOPEN-01')
    RUN.mkdir(parents=True)
    started = time.monotonic()
    report = {'experiment_id': 'CICO-REOPEN-01', 'status': 'initializing',
              'pid': os.getpid(), 'started_unix': time.time(), 'seed': 42, 'arms': {},
              'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'protocol_sha256': sha(PROTOCOL),
              'code_sha256': {p: sha(ROOT / p) for p in (
                  'methods/information_probe/sentence_weight_pilot.py',
                  'methods/information_probe/sentence_weighting.py',
                  'methods/information_probe/scoring.py', 'methods/information_probe/common.py')},
              'optimizer': 'AdamW lr=.001 betas=(.9,.999) eps=1e-8 wd=.001 clip=1',
              'schedule': 'cosine260, no warmup', 'batch': 64, 'updates_per_arm': 260,
              'precision': 'FP32, TF32 disabled', 'encoder_updates': 0,
              'augmentation': 'clean-only frozen contextual cache, all arms',
              'dev_selector': 'meanR1 then meanR5, earliest tie; steps0,130,260',
              'feature_regime': 'Existing CiCo R0 contextual caches; BSL5K agnostic/H2S-transfer-aware inputs',
              'scope': 'Exploratory single-seed trained weighting screen; no method GO',
              'test_accessed': False, 'seds_assets_used': False}
    dump(REPORT, report)
    try:
        torch.set_num_threads(8)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report['torch'] = torch.__version__
        report['gpu'] = torch.cuda.get_device_name()
        event(event='loading', pid=os.getpid())
        train, dev = load_cache('train'), load_cache('dev')
        assert len(train['ids']) == 7096 and len(dev['ids']) == 519
        assert train['checkpoint_sha256'] == dev['checkpoint_sha256']
        report['checkpoint_sha256'] = train['checkpoint_sha256']
        report['manifest_hashes'] = {'train': train['manifest_sha256'], 'dev': dev['manifest_sha256']}
        input_files = [ART/'frozen_train.pt', ART/'frozen_dev.pt', ART/'train_confusers.pt', ART/'baseline_dev_channels.npy']
        report['input_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in input_files}
        base_channels = np.load(ART / 'baseline_dev_channels.npy')
        assert base_channels.shape == (2, 519, 519)
        baseline_metrics = metrics(base_channels.mean(0))
        report['baseline_metrics'] = baseline_metrics
        schedule = batch_schedule(train)
        np.save(RUN/'train_batches.npy', schedule)
        report['batch_schedule_sha256'] = sha(RUN/'train_batches.npy')
        query = {s: eot_vectors(c['text_tokens'], c['text_mask']) for s, c in [('train', train), ('dev', dev)]}
        mean = F.normalize(query['train'].mean(0), dim=0)
        conditions = {'sentence': query, 'blind': {s: mean[None].expand_as(q) for s, q in query.items()}, 'shuffled': {}}
        report['shuffled_bank_indices'] = {}
        for split, q in query.items():
            conditions['shuffled'][split], idx = random_conditioning(q, query['train'])
            report['shuffled_bank_indices'][split] = idx
        report['status'] = 'running'
        dump(REPORT, report)
        for mode in ('blind', 'shuffled', 'sentence'):
            run_arm(mode, train, dev, conditions[mode], base_channels, schedule, report)
        assert len({a['initial_state_sha256'] for a in report['arms'].values()}) == 1
        target = report['arms']['sentence']['selected_metrics']
        controls = [baseline_metrics] + [report['arms'][x]['selected_metrics'] for x in ('blind', 'shuffled')]
        lead_margin = target['official_mean_R1'] - max(x['official_mean_R1'] for x in controls)
        direction_ok = all(target[d]['R1'] >= baseline_metrics[d]['R1']-.25 for d in ('official_T2V', 'V2T'))
        recall_ok = all(target[d][k] >= baseline_metrics[d][k]-.5 for d in ('official_T2V', 'V2T') for k in ('R5', 'R10'))
        persistent_ok = all(target[d]['persistent_mean_rank_delta'] < 0 for d in ('T2V', 'V2T'))
        report['decision'] = {'lead_margin_over_strongest_screen_control_pp': lead_margin,
                              'direction_ok': direction_ok, 'R5_R10_ok': recall_ok, 'persistent_ok': persistent_ok,
                              'screen_lead': lead_margin >= .5 and direction_ok and recall_ok and persistent_ok,
                              'delta_to_three_model_ensemble_pp': target['official_mean_R1']-77.263969,
                              'method_go': False}
        report['status'] = 'completed'
        event(event='completed', **report['decision'])
    except Exception:
        report['status'] = 'failed'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            report['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
        report['wall_seconds'] = time.monotonic()-started
        dump(REPORT, report)


if __name__ == '__main__':
    main()
