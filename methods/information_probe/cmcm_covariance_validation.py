"""Posthoc isolated epsilon-consistency control; no upstream file changes."""
import ast
import copy
import hashlib
import json
import subprocess
import time

import torch

from .cmcm_covariance_audit import COMMIT, OUT, reference
from .common import ROOT, dump, sha


def corrected_classes():
    source = subprocess.check_output(['git', '-C', str(ROOT/'third_party/CMCM'),
                                      'show', COMMIT+':MPNCOV/MPNCOV.py']).decode()
    tree = ast.parse(source)
    classes = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name in {'Covpool', 'Sqrtm', 'Triuvec'}]
    sqrt = next(n for n in classes if n.name == 'Sqrtm')
    backward = next(n for n in sqrt.body if isinstance(n, ast.FunctionDef) and n.name == 'backward')
    replacements = 0
    for node in ast.walk(backward):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'grad_input' for t in node.targets):
            if ast.unparse(node.value).startswith('der_NSiter.div('):
                node.value = ast.parse('der_NSiter.div((normA+1e-5).view(batchSize,1,1).expand_as(x))', mode='eval').body
                replacements += 1
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if ast.unparse(node.right) == 'normA[i] * normA[i]':
                node.right = ast.parse('(normA[i]+1e-5)**2', mode='eval').body
                replacements += 1
    assert replacements == 2
    scope = {'torch': torch, 'Function': torch.autograd.Function}
    module = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
    exec(compile(module, '<cmcm-epsilon-control>', 'exec'), scope)
    return scope, hashlib.sha256(ast.dump(module).encode()).hexdigest()


def main():
    path = OUT/'CMCM-COVARIANCE_validation.json'
    if path.exists():
        raise FileExistsError(path)
    started = time.monotonic()
    torch.set_num_threads(2)
    run_path = OUT/'CMCM-COVARIANCE_run.json'
    run = json.loads(run_path.read_text())
    scope, ast_sha = corrected_classes()
    weight = torch.tensor(run['weight'], dtype=torch.float64)
    errors = []
    for case in run['cases']:
        x = torch.tensor(case['features'], dtype=torch.float64, requires_grad=True)
        y = scope['Triuvec'].apply(scope['Sqrtm'].apply(scope['Covpool'].apply(x), 3))
        g, = torch.autograd.grad((y*weight).sum(), x)
        independent = reference(x)
        reference_g, = torch.autograd.grad((independent*weight).sum(), x)
        saved = torch.tensor(case['reference_gradient'], dtype=torch.float64)
        assert torch.equal(saved, reference_g)
        error = float((g-reference_g).abs().max())
        assert error < 1e-12
        errors.append({'trace': case['target_trace'], 'corrected_gradient_max_error': error})
    result = {'status':'completed', 'posthoc': True, 'control_only_not_upstream_edit': True,
              'code_sha256':sha(__file__), 'original_run_sha256':sha(run_path),
              'controlled_ast_sha256':ast_sha, 'cases':errors,
              'dataset_loaded':False, 'test_loaded':False, 'method_go':False,
              'wall_seconds':time.monotonic()-started}
    dump(path,result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
