"""CPU-only paired-control integrity audit; never loads TEST or executes models."""
import argparse
import json
import time

import numpy as np
import torch

from extraction_resume import atomic_json
from inventory import ROOT, sha


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    cli = parser.parse_args()
    start = time.time()
    roots = [ROOT/'artifacts/slret_goal'/n for n in
             ['seds-continuation-control-001', 'seds-moment-control-001']]
    runs = [json.loads((p/'run.json').read_text()) for p in roots]
    assert all(r['status'] == 'completed' and r['exit_status'] == 0 for r in runs)
    out = ROOT/'artifacts/slret_goal'/cli.run_id
    out.mkdir(exist_ok=False)
    report = dict(run_id=cli.run_id, status='running', test_loaded=False, gpu_used=False,
                  command=__import__('sys').argv, script_sha256=sha(__file__),
                  sources={str(p/'run.json'): sha(p/'run.json') for p in roots})
    try:
        keys = ['checkpoint_sha256', 'train_assets_digest', 'dev_assets_digest',
                'batch_order_sha256', 'source_sha256', 'runtime_sha256', 'seed',
                'batch_size', 'optimizer_updates', 'examples_seen', 'step0_parity']
        report['matched_fields'] = {k: runs[0][k] == runs[1][k] for k in keys}
        assert all(report['matched_fields'].values())
        configs = [r['config'] for r in runs]
        differences = {k: [c.get(k) for c in configs] for k in configs[0].keys() | configs[1].keys()
                       if configs[0].get(k) != configs[1].get(k)}
        report['config_differences'] = differences
        assert set(differences) == {'output_dir'}
        steps = [[json.loads(line) for line in (p/'train_steps.jsonl').read_text().splitlines()]
                 for p in roots]
        assert all(len(rows) == 222 for rows in steps)
        assert all(a['step'] == b['step'] and a['ids'] == b['ids'] for a, b in zip(*steps))
        report['matched_batch_id_steps'] = 222
        report['first_two_losses_and_gradients_exact'] = all(
            a['loss'] == b['loss'] and a['gradient_norm'] == b['gradient_norm']
            for a, b in zip(steps[0][:2], steps[1][:2]))
        assert report['first_two_losses_and_gradients_exact']
        report['initial_score_max_abs_delta'] = {}
        for stream in ['fusion', 'rgb', 'pose']:
            arrays = [np.load(p/'eval_step000'/f'{stream}_video_x_text.npy') for p in roots]
            delta = float(np.max(np.abs(arrays[0]-arrays[1])))
            report['initial_score_max_abs_delta'][stream] = delta
            assert delta == 0
        torch.set_num_threads(4)
        report['checkpoint_checks'] = {}
        for step in [111, 222]:
            paths = [p/f'checkpoint_step{step:03d}.pt' for p in roots]
            ckpts = [torch.load(p, map_location='cpu', weights_only=False, mmap=True) for p in paths]
            models = [c['model'] for c in ckpts]
            assert models[0].keys() == models[1].keys()
            assert all(models[0][k].dtype == models[1][k].dtype and
                       models[0][k].shape == models[1][k].shape for k in models[0])
            rng_equal = same_tree(ckpts[0]['rng'], ckpts[1]['rng'])
            assert rng_equal, 'Paired RNG trajectories differ at checkpoint'
            assert same_tree(ckpts[0]['batches'], ckpts[1]['batches'])
            report['checkpoint_checks'][str(step)] = dict(
                paths=[str(p) for p in paths], sha256=[sha(p) for p in paths],
                rng_exact=rng_equal, model_shapes_and_dtypes_exact=True,
                changed_state_tensors=sum(not torch.equal(models[0][k], models[1][k]) for k in models[0]),
                optimizer_state_saved=['optimizer' in c for c in ckpts])
            del models, ckpts
        report.update(status='completed', exit_status=0,
                      decision='Matched control integrity passed; efficacy and generalization are separate questions.')
    except Exception:
        report.update(status='failed', exit_status=1, error=__import__('traceback').format_exc())
        raise
    finally:
        report['wall_seconds'] = time.time()-start
        atomic_json(out/'run.json', report)
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:
            f.write(json.dumps(report)+'\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
