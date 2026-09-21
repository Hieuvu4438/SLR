"""Registered C13/C14/C15 smoke then one pilot; fail closed, no retry loop."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'research/slret_goal/tools'))
from extraction_resume import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slower-graph', action='store_true')
    parser.add_argument('--bone-features', action='store_true')
    parser.add_argument('--faster-bone', action='store_true')
    parser.add_argument('--joint-bilinear', action='store_true')
    parser.add_argument('--centered-bilinear', action='store_true')
    cli = parser.parse_args()
    if cli.slower_graph and cli.bone_features:
        raise ValueError('Choose one registered candidate')
    if cli.faster_bone and not cli.bone_features:
        raise ValueError('Faster bone requires bone features')
    if cli.joint_bilinear and (cli.bone_features or cli.slower_graph or cli.faster_bone):
        raise ValueError('Joint bilinear is a separate candidate')
    if cli.centered_bilinear and not cli.joint_bilinear:
        raise ValueError('Centered bilinear requires the bilinear candidate')
    bone_lr = 1e-4 if cli.faster_bone else 1e-5
    graph_lr = 1e-5 if cli.slower_graph else 1e-4
    base = ROOT/'artifacts/slret_goal_v2'
    out = base/'c13-adaptive-graph-pilot-001'
    jobs = [('seds-adaptive-graph-smoke-001', 90, True),
            ('seds-adaptive-graph-001', 2300, False)]
    if cli.slower_graph:
        out = base/'c13-adaptive-graph-lr1e5-pilot-001'
        jobs = [('seds-adaptive-graph-lr1e5-smoke-001',90,True),
                ('seds-adaptive-graph-lr1e5-001',2300,False)]
    if cli.bone_features:
        out = base/'c14-bone-features-pilot-001'
        jobs = [('seds-bone-features-smoke-001',90,True),
                ('seds-bone-features-001',2300,False)]
        if cli.faster_bone:
            out = base/'c14-bone-features-lr1e4-pilot-001'
            jobs = [('seds-bone-features-lr1e4-smoke-001',90,True),
                    ('seds-bone-features-lr1e4-001',2300,False)]
    if cli.joint_bilinear:
        out = base/'c15-joint-bilinear-pilot-002'
        jobs = [('seds-joint-bilinear-smoke-002',120,True),
                ('seds-joint-bilinear-001',2500,False)]
        if cli.centered_bilinear:
            out = base/'c15-joint-covariance-pilot-001'
            jobs = [('seds-joint-covariance-smoke-001',120,True),
                    ('seds-joint-covariance-001',2500,False)]
    for name, _, _ in jobs:
        if (base/name).exists():
            raise FileExistsError(name)
    out.mkdir(exist_ok=False)
    sources = [Path(__file__), ROOT/'research/slret_goal_v2/tools/train_seds_extended.py',
               *sorted((ROOT/'methods/seds_adaptation').glob('*.py')),
               ROOT/'research/slret_goal/tools/seds_runtime.py',
               ROOT/'research/slret_goal/tools/seds_optimizer_precision.py',
               ROOT/'third_party/SEDS/main_task_retrieval.py',
               ROOT/'third_party/SEDS/modules/modeling_gcn.py',
               ROOT/'third_party/SEDS/modules/modeling_graph.py']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report = dict(run_id=out.name, status='running', pid=os.getpid(), completed=[],
                  test_used=False, source_sha256=hashes, reservation_seconds=2660 if cli.joint_bilinear else 2430,
                  graph_lr=graph_lr, bone_features=cli.bone_features, bone_lr=bone_lr,
                  joint_bilinear=cli.joint_bilinear, centered_bilinear=cli.centered_bilinear)
    start = time.time()
    child = None

    def record():
        report['wall_seconds'] = time.time()-start
        atomic_json(out/'run.json', report)

    def terminate(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
        raise TimeoutError('Bounded parent received SIGTERM')

    signal.signal(signal.SIGTERM, terminate)
    record()
    try:
        anchor = json.loads((base/'seds-gcn-horizon3-001/run.json').read_text())
        incumbent = json.loads((base/'seds-gcn-horizon3-seed1337-001/run.json').read_text())
        for name, cap, smoke in jobs:
            if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
                raise ValueError('Registered source changed')
            command = ['timeout', '--signal=TERM', '--kill-after=10s', f'{cap}s',
                '/home/haipd/miniconda3/envs/seds/bin/python', '-m', 'torch.distributed.run',
                '--standalone', '--nproc_per_node=1',
                str(ROOT/'research/slret_goal_v2/tools/train_seds_extended.py'),
                '--run-id', name, '--masked-pose', 'control', '--policy', 'fusion',
                '--aux-weight', '1', '--batch-size', '32', '--seed', '42',
                '--lr', '1e-5', '--sign-lr', '1e-6', '--epochs', '3',
                '--gcn-long-horizon', '--early-stop-drop-pp', '2']
            if cli.joint_bilinear:
                command.append('--joint-bilinear')
                if cli.centered_bilinear:
                    command.append('--centered-bilinear')
            elif cli.bone_features:
                command.extend(['--bone-features','--bone-lr',str(bone_lr)])
            else:
                command.extend(['--adaptive-graph','--graph-lr', str(graph_lr)])
            if smoke:
                command.append('--smoke')
            with (out/(name+'.log')).open('xb') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                report['current'] = dict(run_id=name, pid=child.pid, command=command)
                record()
                code = child.wait()
            path = base/name/'run.json'
            result = json.loads(path.read_text()) if path.exists() else {}
            report['completed'].append(dict(run_id=name, returncode=code,
                status=result.get('status'), wall_seconds=result.get('wall_seconds'),
                steps=result.get('steps'), selection=result.get('selection')))
            record()
            if code or result.get('status') != 'completed':
                raise RuntimeError(f'{name} failed; no automatic retry')
            update_key = ('joint_bilinear_updated' if cli.joint_bilinear else
                          'bone_features_updated' if cli.bone_features else 'adaptive_graph_updated')
            if not result.get(update_key) or not result.get('masked_update_checks_passed'):
                raise ValueError('Adaptive graph/update/buffer gates failed')
            expected_parity = ('zero-initialized parameterization fullDEV score parity passed' if smoke
                               else 'inherited C13 smoke fullDEV zero-init parity')
            if cli.bone_features and not smoke:
                expected_parity = 'inherited C14 smoke fullDEV zero-init parity'
            if cli.joint_bilinear and not smoke:
                expected_parity = 'inherited C15 smoke fullDEV zero-init parity'
            if result['step0'] != expected_parity:
                raise ValueError('Initial parity missing')
            rates = {g['adaptation_group']: g['lr'] for g in result['actual_optimizer_groups']}
            expected_rates = dict(encoder=1e-6, fusion=1e-5, bone=bone_lr) if cli.bone_features else dict(encoder=1e-6, fusion=1e-5, graph=graph_lr)
            if cli.joint_bilinear:
                expected_rates = dict(encoder=1e-6, fusion=1e-5, bilinear=1e-4)
            if rates != expected_rates:
                raise ValueError('Wrong learning rates')
            if not smoke:
                for key in ('seed','batch_size','epochs','aux_weight','lr','sign_lr',
                            'masked_pose','policy','early_stop_drop_pp','gcn_long_horizon'):
                    if result['config'][key] != anchor['config'][key]:
                        raise ValueError(f'Recipe mismatch: {key}')
                for key in ('batch_order_sha256','checkpoint_sha256_inherited','inherited_asset_digests'):
                    if result[key] != anchor[key]:
                        raise ValueError(f'Base/data/order mismatch: {key}')
                report['recipe_matches_historical_R1'] = True
                if cli.centered_bilinear:
                    prior = json.loads((base/'seds-joint-bilinear-001/run.json').read_text())
                    for key, value in prior['config'].items():
                        if key != 'run_id' and result['config'][key] != value:
                            raise ValueError(f'C15 refinement changed another setting: {key}')
                    report['selected_minus_C15_original'] = result['selection']['mean_R1']-prior['selection']['mean_R1']
                if cli.faster_bone:
                    prior = json.loads((base/'seds-bone-features-001/run.json').read_text())
                    for key,value in prior['config'].items():
                        if key not in ('run_id','bone_lr') and result['config'][key] != value:
                            raise ValueError(f'C14 refinement changed another setting: {key}')
                    report['selected_minus_C14_original'] = result['selection']['mean_R1']-prior['selection']['mean_R1']
                if cli.slower_graph:
                    prior = json.loads((base/'seds-adaptive-graph-001/run.json').read_text())
                    for key, value in prior['config'].items():
                        if key not in ('run_id','graph_lr') and result['config'][key] != value:
                            raise ValueError(f'C13 refinement changed another setting: {key}')
                    report['selected_minus_C13_original'] = result['selection']['mean_R1']-prior['selection']['mean_R1']
                report['selected_minus_R1_seed42'] = result['selection']['mean_R1']-anchor['selection']['mean_R1']
                report['selected_minus_incumbent'] = result['selection']['mean_R1']-incumbent['selection']['mean_R1']
                report['comparison_caveat'] = 'Historical control; R2 revealed unresolved training replay mismatch. Replicate a positive lead before causal claims.'
        report.update(status='completed', exit_status=0,
                      decision='await_user_return; no automatic new configurations')
    except Exception:
        report.update(status='failed', exit_status=1, error=traceback.format_exc())
        raise
    finally:
        record()


if __name__ == '__main__':
    main()
