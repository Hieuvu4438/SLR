"""Disclosed output-only AS-C42 attempt2; compare every available interrupted prefix."""
import argparse
import ast
import copy
import json
import subprocess
import sys
import time

from .common import ART, ROOT, dump, sha
from .clean_initialization_audit import OUT

HARNESS = ROOT/'methods/information_probe/extended_budget_calibration.py'
LABEL = 'AS-C42-BUDGET-attempt2'


def transform(source):
    tree = ast.parse(source)
    replacements = {'AS-C42-BUDGET': LABEL, 'AS-C42-BUDGET_run.json': LABEL+'_run.json',
                    'AS-C42-BUDGET-console.log': LABEL+'-console.log'}
    changes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in replacements:
            changes.append({'line': node.lineno, 'column': node.col_offset,
                            'old': node.value, 'new': replacements[node.value]})
            node.value = replacements[node.value]
    assert len(changes) == 6, changes
    restored = copy.deepcopy(tree)
    originals = {(x['line'], x['column']): x['old'] for x in changes}
    for node in ast.walk(restored):
        if isinstance(node, ast.Constant) and (node.lineno, node.col_offset) in originals:
            node.value = originals[node.lineno, node.col_offset]
    assert ast.dump(restored) == ast.dump(ast.parse(source))
    return tree, changes


def worker():
    old_path = OUT/'AS-C42-BUDGET_run.json'
    old = json.loads(old_path.read_text())
    assert old['updates'] == 6600 and len(old['training']) == 264 and len(old['evaluations']) == 3
    assert sha(HARNESS) == old['harness_sha256']
    tree, changes = transform(HARNESS.read_text())
    scope = {'__name__': 'methods.information_probe._asc42_attempt2',
             '__package__': 'methods.information_probe', '__file__': str(HARNESS)}
    exec(compile(tree, str(HARNESS), 'exec'), scope)

    def checked_dump(path, result):
        n = min(len(old['training']), len(result['training']))
        m = min(len(old['evaluations']), len(result['evaluations']))
        valid = result['training'][:n] == old['training'][:n] and result['evaluations'][:m] == old['evaluations'][:m]
        result['recovery'] = {'attempt': 2, 'wrapper_sha256': sha(__file__),
            'interrupted_run_sha256': sha(old_path), 'output_literal_changes': changes,
            'checked_training_records': n, 'checked_evaluation_records': m,
            'prefix_exact': valid, 'original_cause': 'unknown termination; process absent',
            'full_interrupted_prefix_checked': n == 264 and m == 3 and valid}
        dump(path, result)
        assert valid, 'Attempt2 differs from interrupted numerical prefix'

    scope['dump'] = checked_dump
    scope['main']()


def launch():
    path = OUT/(LABEL+'-launch.json')
    if path.exists() or (OUT/(LABEL+'_run.json')).exists() or (ART/LABEL).exists():
        raise FileExistsError('Inspect existing attempt2; never auto-relaunch')
    bootstrap = ART/(LABEL+'-bootstrap.log')
    command = ['timeout', '7300', sys.executable, '-m',
               'methods.information_probe.recover_extended_budget', '--supervise']
    with bootstrap.open('xb') as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    dump(path, {'status': 'launched', 'pid': child.pid, 'command': command,
                'launcher_sha256': sha(__file__), 'started_unix': time.time(),
                'bootstrap_log': str(bootstrap)})
    print(json.dumps({'pid': child.pid, 'launch_record': str(path)}))


def supervise():
    started = time.time()
    child = subprocess.Popen([sys.executable, '-m', 'methods.information_probe.recover_extended_budget', '--worker'], cwd=ROOT)
    dump(OUT/(LABEL+'-worker.json'), {'worker_pid': child.pid, 'started_unix': started})
    code = child.wait()
    dump(OUT/(LABEL+'-exit.json'), {'returncode': code, 'wall_seconds': time.time()-started})
    raise SystemExit(code)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true')
    p.add_argument('--supervise', action='store_true')
    args = p.parse_args()
    worker() if args.worker else supervise() if args.supervise else launch()
