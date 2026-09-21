"""Registered smoke/pilot profiles, fail closed; historical C05 profile retained."""
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
sys.path.insert(0, str(ROOT/'research/slret_goal/tools'))
from extraction_resume import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['c05', 'lora-depth3', 'lora-text', 'interaction-frozen', 'temporal-delta', 'joint-cross', 'joint-cross-lrfix'], default='c05')
    args = parser.parse_args()
    depth3 = args.profile == 'lora-depth3'
    text_lora = args.profile == 'lora-text'
    interaction = args.profile == 'interaction-frozen'
    temporal = args.profile == 'temporal-delta'
    joint = args.profile in ('joint-cross', 'joint-cross-lrfix')
    joint_lrfix = args.profile == 'joint-cross-lrfix'
    peft = depth3 or text_lora
    name = 'c04-depth3-smoke-pilot-001' if depth3 else 'c05-smoke-pilot-001'
    if text_lora:
        name = 'c04-text-smoke-pilot-001'
    if interaction:
        name = 'c03-frozen-smoke-pilot-001'
    if temporal:
        name = 'c06-temporal-smoke-pilot-001'
    if joint:
        name = 'c07-joint-smoke-pilot-001'
    if joint_lrfix:
        name = 'c07-joint-lrfix-smoke-pilot-001'
    out = ROOT/'artifacts/slret_goal_v2'/name
    prefix = 'seds-lora-depth3' if depth3 else 'seds-smooth005'
    if text_lora:
        prefix = 'seds-lora-text'
    if interaction:
        prefix = 'seds-articulator-frozen'
    if temporal:
        prefix = 'seds-temporal-delta'
    if joint:
        prefix = 'seds-joint-cross'
    if joint_lrfix:
        prefix = 'seds-joint-cross-lrfix'
    jobs = [(prefix+'-smoke-001', 240 if joint else 180, True),
            (prefix+'-001', 1800 if joint else (1500 if interaction else (1800 if peft or temporal else 1200)), False)]
    recipe = ['--aux-weight','1','--batch-size','32','--seed','42']
    recipe += (['--policy','lora','--sign-lr','1e-5','--lora-lr','1e-4',
                '--epochs','3','--lora-upper-blocks','3' if depth3 else '1','--early-stop-drop-pp','5']
               if peft else ['--policy','all','--lr','1e-5','--sign-lr','1e-4',
                              '--epochs','1','--label-smoothing','.05'])
    if text_lora:
        recipe.append('--text-lora')
    if interaction:
        recipe = ['--aux-weight','1','--batch-size','32','--seed','42',
                  '--policy','fusion','--interaction','product','--lr','1e-5',
                  '--sign-lr','1e-5','--epochs','3','--early-stop-drop-pp','5']
    if temporal:
        recipe = ['--aux-weight','1','--batch-size','32','--seed','42',
                  '--policy','fusion','--temporal-delta','--lr','1e-5',
                  '--sign-lr','1e-5','--epochs','3','--early-stop-drop-pp','5']
    if joint:
        recipe = ['--aux-weight','1','--batch-size','32','--seed','42',
                  '--policy','fusion','--joint-exchange','cross','--lr','1e-5',
                  '--sign-lr','1e-4','--epochs','1','--early-stop-drop-pp','2']
    for run, _, _ in jobs:
        if (out.parent/run).exists():
            raise FileExistsError(f'Refuse duplicate {run}')
    out.mkdir(parents=True, exist_ok=False)
    runner = ROOT/'research/slret_goal_v2/tools/train_seds_extended.py'
    sources = [Path(__file__), runner, *sorted((ROOT/'methods/seds_adaptation').glob('*.py'))]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    started = time.time()
    report = dict(run_id=out.name, status='running', pid=os.getpid(),
                  start_unix=started, source_sha256=hashes, completed=[], test_used=False)

    def record(terminal=False):
        report['wall_seconds'] = time.time()-started
        atomic_json(out/'run.json', report)
        if terminal:
            with (ROOT/'research/slret_goal_v2/EXPERIMENTS.jsonl').open('a') as f:
                f.write(json.dumps(report)+'\n')

    record(True)
    try:
        for run, seconds, smoke in jobs:
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                       for p, h in hashes.items()), 'Live source drift'
            command = ['timeout', '--signal=TERM', '--kill-after=10s', f'{seconds}s',
                       '/home/haipd/miniconda3/envs/seds/bin/python', '-m',
                       'torch.distributed.run', '--standalone', '--nproc_per_node=1',
                       str(runner), '--run-id', run, *recipe]
            if smoke:
                command.append('--smoke')
            with (out/f'{run}.log').open('xb') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                report['current'] = dict(run_id=run, child_pid=child.pid,
                                         command=command, start_unix=time.time())
                record()
                code = child.wait()
            path = out.parent/run/'run.json'
            result = json.loads(path.read_text()) if path.exists() else {}
            report['completed'].append(dict(run_id=run, returncode=code,
                status=result.get('status'), wall_seconds=result.get('wall_seconds'),
                selection=result.get('selection')))
            record()
            assert code == 0 and result.get('status') == 'completed', f'{run} failed; no retry'
            if smoke:
                assert result['steps'] == 2 and result['criterion'] == ('native' if peft or interaction or temporal or joint else 'label_smoothed')
                assert result['changed_parameter_tensors_step2'] > 0
                expected = {'clip','clip_rgb','fusion'} if peft else {'clip','clip_rgb','fusion','signbert'}
                if interaction:
                    expected = {'fusion','signbert'}
                    assert result['interaction_output_updated']
                if temporal:
                    expected = {'fusion','temporal_delta'}
                    assert result['temporal_delta_updated']
                if joint:
                    expected = {'fusion','signbert'}
                    assert result['joint_exchange_updated']
                    observed = {(g['adaptation_group'],g['lr']) for g in result['actual_optimizer_groups']}
                    assert observed == {('joint',1e-4),('fusion',1e-5)}, observed
                assert set(result['changed_parameter_groups_step2']) == expected
                if depth3:
                    assert result['lora_updated_B_tensors'] == 12
                if text_lora:
                    assert result['lora_updated_B_tensors'] == 6
        report.update(status='completed', exit_status=0, current=None)
    except Exception:
        report.update(status='failed', exit_status=1, error=traceback.format_exc())
        raise
    finally:
        record(True)


if __name__ == '__main__':
    main()
