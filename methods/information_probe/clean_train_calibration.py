"""AS-C20: generic-initialized, TRAIN-only calibration; never a method pilot."""
import hashlib
import json
import math
import os
import time
import traceback

import numpy as np
import torch
import yaml

from slr_common.data.cico_dataset import CiCoFeatureDataset
from slr_common.data.tokenize import CiCoCollator, encode_cico_text
from slr_common.upstream.cico_bridge import CiCoBridge
from slr_common.upstream.factory import load_cico_tokenizer
from .clean_initialization_audit import GENERIC_SHA, OUT, RANDOM_KEYS, initialize
from .common import ART, ROOT, dump, rows, sha
from .scoring import balanced_loss, channels


def lr_factor(step, total=1000, warmup=100):
    if step <= warmup:
        return step / warmup
    return .5 * (1 + math.cos(math.pi * (step-warmup)/(total-warmup)))


def compact_metrics(scores, text_keys):
    """Official asymmetric singleton tie kernels without dev-dependent helpers."""
    s = np.asarray(scores)
    assert s.shape == (len(text_keys), len(text_keys)) and np.isfinite(s).all()
    n = len(s)
    v = torch.argsort(torch.argsort(torch.from_numpy(s), dim=1, descending=True), dim=1).diagonal().numpy()
    t = (s > s.diagonal()[None, :]).sum(0)
    expanded = np.concatenate([np.flatnonzero(np.sort(s[:, i])[::-1] == s[i, i]) for i in range(n)])
    different = np.asarray([[a != b for b in text_keys] for a in text_keys])
    residual = np.where(different, s, -np.inf)
    output = {}
    for direction, primary, per_query, axis in [('T2V', expanded, t, 0), ('V2T', v, v, 1)]:
        margin = s.diagonal() - residual.max(axis=axis)
        output[direction] = {**{f'R{k}': float(100*np.mean(primary < k)) for k in (1, 5, 10)},
                             'ranks': per_query.tolist(), 'primary_rank_entries': len(primary),
                             'strict_different_text_error_n': int((margin < 0).sum()),
                             'strict_different_text_error_indexes': np.flatnonzero(margin < 0).tolist()}
    output['mean_R1'] = (output['T2V']['R1'] + output['V2T']['R1']) / 2
    return output


def adequacy(initial, final):
    learning = (all(final['fit'][d]['R1'] >= 80 and final['held'][d]['R1'] >= 50 for d in ('T2V', 'V2T'))
                and final['held']['mean_R1'] - initial['held']['mean_R1'] >= 5)
    residual = all(final['held'][d]['strict_different_text_error_n'] >= 50 for d in ('T2V', 'V2T'))
    return {'learning_gate': learning, 'residual_count_gate': residual, 'diagnostic_adequacy': learning and residual,
            'method_go': False}


@torch.no_grad()
def evaluate(core, bridge, items, indexes, collator, keys):
    core.eval()
    enc = {k: [] for k in ('v', 't', 'vm', 'tm')}
    for start in range(0, len(indexes), 128):
        b = collator([items[i] for i in indexes[start:start+128]])
        v = bridge.encode_video(b['h'].cuda(), b['valid'].cuda())
        t = bridge.encode_text(*(x.cuda() for x in b['clean_text']))
        for k, value in zip(enc, (v.tokens, t.tokens, v.mask, t.mask)):
            enc[k].append(value.cpu())
    enc = {k: torch.cat(v).cuda() for k, v in enc.items()}
    s = np.empty((len(indexes), len(indexes)), np.float32)
    for i in range(0, len(indexes), 64):
        for j in range(0, len(indexes), 64):
            a, b = channels(enc['v'][i:i+64], enc['t'][j:j+64], enc['vm'][i:i+64], enc['tm'][j:j+64], core.clip.logit_scale.exp())
            s[i:i+64, j:j+64] = ((a+b)/2).cpu().numpy()
    return s, compact_metrics(s, [keys[i] for i in indexes])


def main():
    path = OUT/'AS-C20-TRAIN_run.json'
    dest = ART/'AS-C20'
    if path.exists() or dest.exists():
        raise FileExistsError('AS-C20 outputs already exist; never overwrite a prior attempt')
    dest.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    result = {'experiment_id': 'AS-C20-TRAIN', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C20_protocol.md'),
              'initialization_code_sha256': sha(ROOT/'methods/information_probe/clean_initialization_audit.py'),
              'dev_or_test_loaded': False, 'ph_retrieval_checkpoint_loaded': False,
              'method_go': False, 'updates': 0, 'training': [], 'evaluations': []}
    def record():
        result['wall_seconds'] = time.time()-started
        dump(path, result)
    record()
    try:
        partition_path = OUT/'AS-C19-TRAIN-partition.json'
        partition = json.loads(partition_path.read_text())
        assert partition['manifest_sha256'] == sha(ROOT/'artifacts/manifests/ph_train.jsonl')
        result['partition_sha256'] = sha(partition_path)
        fit, held = partition['fit_indexes'], partition['held_indexes']
        assert len(fit) == 5721 and len(held) == 1375 and not set(fit)&set(held)
        config_path = ROOT/'runs/ph_base_b512_s42/resolved_config.yaml'
        config = yaml.safe_load(config_path.read_text())
        result['architecture_config_sha256'] = sha(config_path)
        generic_path = ROOT/'artifacts/pretrained/ViT-B-32.pt'
        assert sha(generic_path) == GENERIC_SHA
        result['generic_clip_sha256'] = GENERIC_SHA
        generic = torch.jit.load(str(generic_path), map_location='cpu').state_dict()
        core, result['initialization'] = initialize(config, generic, 42)
        del generic
        core.float().cuda()
        bridge = CiCoBridge(core)
        tok = load_cico_tokenizer(config)
        dataset = CiCoFeatureDataset(ROOT/'artifacts/manifests/ph_train.jsonl', feature_len=64, alpha=.9, split='train')
        items = [dataset[i] for i in range(len(dataset))]
        keys = [tuple(encode_cico_text(x['caption'], tok, 32)[0].tolist()) for x in items]
        fit_eval = sorted(sorted(fit, key=lambda i: hashlib.sha256(items[i]['pair_id'].encode()).digest())[:len(held)])
        result['fit_eval_indexes'] = fit_eval
        train_collator = CiCoCollator(tok, 32, augment=True, seed=42)
        eval_collator = CiCoCollator(tok, 32, augment=False, seed=42)
        groups = {}
        for name, p in core.named_parameters():
            base_lr = 1e-4 if name in RANDOM_KEYS else 1e-5
            decay = 0. if p.ndim <= 1 or name.endswith('.bias') else .001
            groups.setdefault((base_lr, decay), []).append(p)
        optimizer = torch.optim.AdamW([{'params': ps, 'lr': lr, 'base_lr': lr, 'weight_decay': wd}
                                      for (lr, wd), ps in groups.items()], betas=(.9, .98), eps=1e-6)
        rng = torch.Generator().manual_seed(42)
        def check_timeout():
            if time.time()-started > 3600:
                raise TimeoutError('preregistered 3600-second hard timeout')
        def eval_at(step):
            check_timeout()
            entry = {'step': step}
            for name, indexes in [('fit', fit_eval), ('held', held)]:
                score, entry[name] = evaluate(core, bridge, items, indexes, eval_collator, keys)
                p = dest/f'{name}_step{step}.npy'
                np.save(p, score)
                entry[name]['score_sha256'] = sha(p)
            result['evaluations'].append(entry)
            record()
            print(json.dumps({'step': step, 'fit_R1': entry['fit']['mean_R1'], 'held_R1': entry['held']['mean_R1'], 'wall_seconds': result['wall_seconds']}), flush=True)
        eval_at(0)
        epoch, losses = 0, []
        while result['updates'] < 1000:
            order = [fit[i] for i in torch.randperm(len(fit), generator=rng).tolist()]
            train_collator.set_epoch(epoch)
            for start in range(0, len(order)-127, 128):
                check_timeout()
                core.train()
                step = result['updates']+1
                for group in optimizer.param_groups:
                    group['lr'] = group['base_lr']*lr_factor(step)
                batch = train_collator([items[i] for i in order[start:start+128]])
                optimizer.zero_grad(set_to_none=True)
                v = bridge.encode_video(batch['h'].cuda(), batch['valid'].cuda())
                t = bridge.encode_text(*(x.cuda() for x in batch['clean_text']))
                ta = bridge.encode_text(*(x.cuda() for x in batch['aug_text']))
                a, _ = channels(v.tokens, t.tokens, v.mask, t.mask, core.clip.logit_scale.exp())
                _, b = channels(v.tokens, ta.tokens, v.mask, ta.mask, core.clip.logit_scale.exp())
                loss = balanced_loss(a, b)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'nonfinite loss at {step}')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(core.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                result['updates'] = step
                if step % 25 == 0:
                    entry = {'step': step, 'epoch': epoch, 'mean_loss_last25': sum(losses)/len(losses),
                             'gradient_norm_preclip': float(norm), 'logit_scale': float(core.clip.logit_scale.exp().detach())}
                    losses = []
                    result['training'].append(entry)
                    record()
                    print(json.dumps(entry), flush=True)
                # Release training graphs before the full-gallery evaluation.
                del v, t, ta, a, b, loss
                if step in (250, 500, 1000):
                    eval_at(step)
                if step == 1000:
                    break
            epoch += 1
        checkpoint = dest/'final_step1000.pt'
        torch.save({'state_dict': {k: v.detach().cpu() for k, v in core.state_dict().items()},
                    'step': 1000, 'partition_sha256': result['partition_sha256'],
                    'protocol_sha256': result['protocol_sha256']}, checkpoint)
        result['checkpoint'] = {'path': str(checkpoint), 'sha256': sha(checkpoint)}
        result['adequacy'] = adequacy(result['evaluations'][0], result['evaluations'][-1])
        result.update(status='completed', peak_gpu_bytes=torch.cuda.max_memory_allocated())
        record()
        print(json.dumps({'status': result['status'], 'adequacy': result['adequacy'], 'wall_seconds': result['wall_seconds']}), flush=True)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc())
        record()
        raise


if __name__ == '__main__':
    main()
