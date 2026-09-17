"""AS-C40 fixed end-of-warmup augmentation on/off training intervention."""
import argparse
import hashlib
import json
import os
import time
import traceback

import numpy as np
import torch
import yaml

from .common import ART, ROOT, dump, sha
from .historical_training_replay import OUT, EpochComplete, check_sources, exact_compare, historical_function


def tensor_digest(values):
    digest = hashlib.sha256()
    for name, value in values:
        x = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(x.dtype), list(x.shape)]).encode())
        digest.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def batch_record(batch, number):
    clean, aug = batch['clean_text'], batch['aug_text']
    changed = torch.stack([(a != b).reshape(len(a), -1).any(1) for a, b in zip(clean, aug)]).any(0)
    return {'batch_number': number, 'pair_ids': batch['pair_id'],
            'clean_sha256': tensor_digest([(str(i), x) for i, x in enumerate(clean)]),
            'aug_sha256': tensor_digest([(str(i), x) for i, x in enumerate(aug)]),
            'visual_sha256': tensor_digest([('h', batch['h']), ('valid', batch['valid'])]),
            'changed_text_row_n': int(changed.sum())}


def main():
    import elsc.train as trainer
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(42, 1337, 2026), required=True)
    parser.add_argument('--condition', choices=('on', 'off'), required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    name = f'AS-C40-s{args.seed}-{args.condition}'
    assert args.attempt >= 1
    if args.attempt > 1:
        name += f'-attempt{args.attempt}'
    path, run = OUT/f'{name}_run.json', ART/name
    if path.exists() or run.exists():
        raise FileExistsError(name)
    torch.set_num_threads(24)
    started = time.time()
    result = {'experiment_id': name, 'status': 'running', 'pid': os.getpid(),
              'seed': args.seed, 'condition': args.condition, 'method_go': False, 'test_loaded': False,
              'attempt': args.attempt,
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C40_protocol.md'),
              'epochs': [], 'training_batches': [], 'checkpoint_writes': 0}
    dump(path, result)
    try:
        parent_path = OUT/'AS-C39-TRAIN-REPLAY_run.json'
        parent = json.loads(parent_path.read_text())
        assert parent['status'] == 'completed' and parent['replay_gate']
        result['parent_replay_sha256'] = sha(parent_path)
        result['source_checks'] = check_sources()
        config = yaml.safe_load((ROOT/'runs/ph_base_b512_s42/resolved_config.yaml').read_text())
        config['seed'] = args.seed
        if args.condition == 'off':
            config['data']['text_augmentation'] = None
        assert config['method'] == 'baseline' and config['train']['epochs'] == 200
        assert config['train']['per_device_batch'] == 512 and config['train']['num_workers'] == 8
        for filename, digest in parent['inputs'].items():
            assert sha(ROOT/filename) == digest
        assert sha(ROOT/config['model']['init_checkpoint']) == config['model']['init_checkpoint_sha256']
        namespace = dict(vars(trainer))
        original_build, original_loader = trainer.build_retriever_from_checkpoint, trainer.DataLoader
        original_eval = trainer.evaluate_model

        def build(*a, **kw):
            model, raw = original_build(*a, **kw)
            assert 'initial_state_sha256' not in result
            result['initial_state_sha256'] = tensor_digest(model.state_dict().items())
            return model, raw

        class TracedLoader:
            def __init__(self, *a, **kw):
                self.inner = original_loader(*a, **kw)
                assert len(self.inner) == 13

            def __len__(self):
                return len(self.inner)

            def __iter__(self):
                for batch in self.inner:
                    result['training_batches'].append(batch_record(batch, len(result['training_batches'])))
                    yield batch

        def evaluate(model, cfg, split, device):
            assert split == 'dev'
            epoch = len(result['epochs'])
            assert 0 <= epoch < 20
            s, m = original_eval(model, cfg, split, device)
            entry = {'epoch': epoch, 'step': (epoch+1)*13,
                     'metrics': {d: {k: m[d][k] for k in ('R1', 'R5', 'R10', 'MeanR', 'MedianR')} for d in ('T2V', 'V2T')}}
            if epoch in (0, 19):
                target = ART/f'{name}-epoch{epoch}_scores.npy'
                if target.exists():
                    raise FileExistsError(target)
                np.save(target, s)
                entry['score_sha256'] = sha(target)
            result['epochs'].append(entry)
            result['wall_seconds'] = time.time()-started
            dump(path, result)
            print(json.dumps({'run': name, **entry, 'wall_seconds': result['wall_seconds']}), flush=True)
            return s, m

        def save_callback(save_path, model, optimizer, scheduler, scaler, generator, epoch, step, cfg, best, provenance):
            assert 0 <= epoch < 20 and step == (epoch+1)*13
            if args.seed == 42 and args.condition == 'on' and epoch == 0 and save_path.name == 'last.pt':
                cp = ROOT/'runs/ph_base_b512_s42/checkpoints/best_dev.pt'
                old = torch.load(cp, map_location='cpu', weights_only=True, mmap=True)
                actual = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict(), 'amp_scaler': scaler.state_dict(),
                          'rng': trainer.capture_rng_state(), 'sampler_generator_state': generator.get_state()}
                mismatches = {k: exact_compare(v, old[k], k) for k, v in actual.items()}
                score_exact = np.array_equal(np.load(ART/f'{name}-epoch0_scores.npy'), np.load(ART/'AS-C39-epoch0_scores.npy'))
                result['seed42_epoch0_replay'] = {'score_exact': score_exact, 'mismatches': mismatches}
                assert score_exact and not any(mismatches.values())
            if epoch == 19 and save_path.name == 'last.pt':
                result['endpoint_model_sha256'] = tensor_digest(model.state_dict().items())
                result['endpoint_scheduler'] = scheduler.state_dict()
                result['updates'] = step
                raise EpochComplete()

        namespace.update(build_retriever_from_checkpoint=build, DataLoader=TracedLoader,
                         evaluate_model=evaluate, _save_checkpoint=save_callback)
        try:
            historical_function(namespace)(config, run, device=torch.device('cuda:0'))
        except EpochComplete:
            pass
        else:
            raise AssertionError('Expected fixed epoch19 boundary')
        assert len(result['training_batches']) == 260 and len(result['epochs']) == 20
        # Training function frame is gone. A separate initialization evaluation
        # cannot consume training RNG or affect the already completed updates.
        torch.cuda.empty_cache()
        trainer._seed_everything(args.seed)
        initial, raw = original_build(config, ROOT/config['model']['init_checkpoint'], device=torch.device('cuda:0'))
        assert tensor_digest(initial.state_dict().items()) == result['initial_state_sha256']
        s, m = original_eval(initial, config, 'dev', torch.device('cuda:0'))
        target = ART/f'{name}-initialization_scores.npy'
        if target.exists():
            raise FileExistsError(target)
        np.save(target, s)
        result['initialization'] = {'score_sha256': sha(target), 'R1': {d: m[d]['R1'] for d in ('T2V', 'V2T')}}
        result.update(status='completed', wall_seconds=time.time()-started,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(), full_200epoch_run=False)
        dump(path, result)
        print(json.dumps({'run': name, 'status': 'completed', 'wall_seconds': result['wall_seconds'],
                          'endpoint': result['epochs'][-1], 'initialization': result['initialization']}), flush=True)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
