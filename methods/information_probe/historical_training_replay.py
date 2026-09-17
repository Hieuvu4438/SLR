"""AS-C39 replay archived baseline train function through its first epoch."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch
import yaml

from .common import ART, ROOT, dump, sha

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'
COMMIT = '39449e18def6b154944ceeaed39dfd5c570882a3'


def archived(path):
    return subprocess.check_output(['git', 'show', f'{COMMIT}:{path}'], cwd=ROOT, text=True)


def definitions(source):
    tree = ast.parse(source) if isinstance(source, str) else source
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def check_sources():
    pairs = [('elsc/train.py', 'methods/elsc/elsc/train.py',
              ['_seed_everything', '_amp_settings', '_optimizer', '_scheduler']),
             ('elsc/data/tokenize.py', 'shared/slr_common/data/tokenize.py', None),
             ('elsc/data/cico_dataset.py', 'shared/slr_common/data/cico_dataset.py', None),
             ('elsc/data/views.py', 'shared/slr_common/data/views.py', None),
             ('elsc/models/retriever.py', 'methods/elsc/elsc/models/retriever.py', None),
             ('elsc/models/adapter.py', 'methods/elsc/elsc/models/adapter.py', None),
             ('elsc/losses/coarse.py', 'methods/elsc/elsc/losses/coarse.py', None),
             ('elsc/evaluation/runtime.py', 'shared/slr_common/evaluation/runtime.py', None),
             ('elsc/utils.py', 'shared/slr_common/utils.py', None),
             ('elsc/evaluate.py', 'methods/elsc/elsc/evaluate.py', ['evaluate_model']),
             ('elsc/upstream/factory.py', 'shared/slr_common/upstream/factory.py',
              ['_state_dict', 'cico_task_config', '_load_cico_core_from_state', 'load_cico_tokenizer', 'load_cico_tokenizer_components'])]
    checked = {}
    for old, new, names in pairs:
        source = archived(old)
        a, b = definitions(source), definitions((ROOT/new).read_text())
        names = list(a) if names is None else names
        assert all(ast.dump(a[k]) == ast.dump(b[k]) for k in names), (old, new)
        checked[new] = {'current_sha256': sha(ROOT/new), 'archived_sha256': hashlib.sha256(source.encode()).hexdigest(),
                        'exact_ast_definitions': names}
    a = definitions(archived('elsc/upstream/cico_bridge.py'))['CiCoBridge']
    b = definitions((ROOT/'shared/slr_common/upstream/cico_bridge.py').read_text())['CiCoBridge']
    names = ['__init__', 'upstream_video', 'upstream_video_mask', 'encode_video', 'encode_text', 'score', 'mixed_score']
    assert all(ast.dump(definitions(a)[k]) == ast.dump(definitions(b)[k]) for k in names)
    checked['shared/slr_common/upstream/cico_bridge.py'] = {'current_sha256': sha(ROOT/'shared/slr_common/upstream/cico_bridge.py'),
                                                         'exact_ast_members': names, 'paired_score_not_used': True}
    a = definitions(archived('elsc/upstream/factory.py'))['build_retriever_from_checkpoint']
    b = definitions((ROOT/'methods/elsc/elsc/upstream/factory.py').read_text())['build_retriever_from_checkpoint']
    # Only normalize documented docstrings and the moved class import.
    for n in (a, b):
        n.body = [x for x in n.body if not (isinstance(x, ast.Expr) and isinstance(x.value, ast.Constant)
                  and isinstance(x.value.value, str)) and not (isinstance(x, ast.ImportFrom) and x.module == 'elsc.models.retriever')]
    assert ast.dump(a) == ast.dump(b)
    checked['methods/elsc/elsc/upstream/factory.py'] = {'current_sha256': sha(ROOT/'methods/elsc/elsc/upstream/factory.py'),
                                                     'builder_normalized_ast_exact': True}
    return checked


def exact_compare(a, b, path='root'):
    """Return leaf mismatches; never replace exact equality with tolerance."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        ok = torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype
        if ok and torch.equal(a.detach().cpu(), b.detach().cpu()):
            return []
        delta = None
        if ok and a.numel() and a.dtype != torch.bool:
            delta = float((a.detach().cpu().double()-b.detach().cpu().double()).abs().max())
        return [{'path': path, 'kind': 'tensor', 'max_abs_delta': delta}]
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        ok = isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
        return [] if ok else [{'path': path, 'kind': 'ndarray'}]
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for k in a.keys() | b.keys():
            if k not in a or k not in b:
                result.append({'path': f'{path}/{k}', 'kind': 'missing_key'})
            else:
                result.extend(exact_compare(a[k], b[k], f'{path}/{k}'))
        return result
    if isinstance(a, (list, tuple)) and isinstance(b, type(a)):
        if len(a) != len(b):
            return [{'path': path, 'kind': 'sequence_length'}]
        return [r for i, (x, y) in enumerate(zip(a, b)) for r in exact_compare(x, y, f'{path}/{i}')]
    return [] if type(a) == type(b) and a == b else [{'path': path, 'kind': 'scalar_or_type', 'actual': str(a), 'expected': str(b)}]


class EpochComplete(Exception):
    """Intentional callback boundary after all13updates and epoch0evaluation."""


def historical_function(namespace):
    node = definitions(archived('elsc/train.py'))['train']
    module = ast.Module(body=[node], type_ignores=[])
    code = compile(ast.fix_missing_locations(module), f'git:{COMMIT}:elsc/train.py', 'exec')
    exec(code, namespace)
    return namespace['train']


def main():
    import elsc.train as trainer
    path = OUT/'AS-C39-TRAIN-REPLAY_run.json'
    run = ART/'AS-C39-TRAIN-REPLAY'
    target = ART/'AS-C39-epoch0_scores.npy'
    if path.exists() or run.exists() or target.exists():
        raise FileExistsError('AS-C39 outputs already exist')
    started = time.time()
    result = {'experiment_id': 'AS-C39-TRAIN-REPLAY', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C39_protocol.md'),
              'method_go': False, 'test_loaded': False, 'historical_commit': COMMIT,
              'environment': {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                              'gpu': torch.cuda.get_device_name(), 'threads': torch.get_num_threads()}}
    dump(path, result)
    try:
        result['source_checks'] = check_sources()
        original_root = ROOT/'runs/ph_base_b512_s42'
        config = yaml.safe_load((original_root/'resolved_config.yaml').read_text())
        assert config['method'] == 'baseline' and config['seed'] == 42
        assert not config['model']['adapter']['enabled'] and not config['evidence']['enabled']
        assert all(config[k].get(w, 0) == 0 for k, w in [('loss', 'lexical_weight'), ('caption', 'weight'), ('keep', 'weight')])
        assert config['train']['epochs'] == 200 and config['train']['per_device_batch'] == 512
        cp = original_root/'checkpoints/best_dev.pt'
        assert sha(cp) == 'ee45bd2dcad57ff4237eaeb2fc326d288698400c13aac0e51d9df01496b36351'
        assert sha(ROOT/config['model']['init_checkpoint']) == config['model']['init_checkpoint_sha256']
        result['inputs'] = {str(p.relative_to(ROOT)): sha(p) for p in (
            cp, original_root/'resolved_config.yaml', original_root/'train.jsonl',
            original_root/'evaluation/dev/scores_video_x_text.npy',
            ROOT/config['data']['train_manifest'], ROOT/config['data']['dev_manifest'])}
        result['archived_train_source_sha256'] = hashlib.sha256(archived('elsc/train.py').encode()).hexdigest()
        namespace = dict(vars(trainer))
        original_evaluate = trainer.evaluate_model

        def evaluate(model, cfg, split, device):
            assert split == 'dev' and not target.exists()
            scores, met = original_evaluate(model, cfg, split, device)
            np.save(target, scores)
            result['score_exact'] = np.array_equal(scores, np.load(original_root/'evaluation/dev/scores_video_x_text.npy'))
            result['score_sha256'] = sha(target)
            result['R1'] = {d: met[d]['R1'] for d in ('T2V', 'V2T')}
            return scores, met

        def finish(save_path, model, optimizer, scheduler, scaler, generator, epoch, step, cfg, best, provenance):
            assert epoch == 0 and step == 13 and save_path.name == 'last.pt'
            old = torch.load(cp, map_location='cpu', weights_only=True, mmap=True)
            assert old['epoch'] == epoch and old['step'] == step
            actual = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(), 'amp_scaler': scaler.state_dict(),
                      'rng': trainer.capture_rng_state(), 'sampler_generator_state': generator.get_state(),
                      'trainable_parameter_names': [k for k, v in model.named_parameters() if v.requires_grad]}
            result['state_checks'] = {k: {'exact': not (m := exact_compare(v, old[k], k)), 'mismatches': m}
                                      for k, v in actual.items()}
            result['model_tensor_n'] = len(actual['model'])
            result['optimizer_state_parameter_n'] = len(actual['optimizer']['state'])
            result['scheduler_total_steps'] = 2600
            result['epoch'], result['updates'] = epoch, step
            print(json.dumps({'boundary': 'epoch0step13', 'score_exact': result['score_exact'],
                              'state_exact': {k: x['exact'] for k, x in result['state_checks'].items()}}), flush=True)
            raise EpochComplete()

        namespace['evaluate_model'] = evaluate
        namespace['_save_checkpoint'] = finish
        train = historical_function(namespace)
        try:
            train(config, run, device=torch.device('cuda:0'))
        except EpochComplete:
            pass
        else:
            raise AssertionError('Expected first-epoch stop callback')
        old_logs = [json.loads(x) for x in (original_root/'train.jsonl').read_text().splitlines()]
        new_logs = [json.loads(x) for x in (run/'train.jsonl').read_text().splitlines()]
        old_logs = [x for x in old_logs if x['epoch'] == 0]
        strip = lambda x: {k: v for k, v in x.items() if k not in ('elapsed_seconds', 'peak_gpu_memory_bytes')}
        result['log_mismatches'] = exact_compare(list(map(strip, new_logs)), list(map(strip, old_logs)), 'epoch0logs')
        result['log_exact'] = not result['log_mismatches']
        result['log_record_n'] = len(new_logs)
        result['replay_gate'] = result['score_exact'] and result['log_exact'] and all(x['exact'] for x in result['state_checks'].values())
        result.update(status='completed', wall_seconds=time.time()-started,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(), checkpoint_writes=0)
        dump(path, result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('source_checks', 'state_checks', 'inputs')}, indent=2), flush=True)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
