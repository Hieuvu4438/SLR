"""Parse released training recipes without importing SEDS or accessing assets."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import patch


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / 'third_party/SEDS/main_task_retrieval.py'
    wanted = {'rgb_pose_kl', 'rgb_pose_match', 'rgb_pose_match_loss',
              'kl_pose_loss', 'kl_rgb_loss', 'kl_logit'}
    tree = ast.parse(source.read_bytes())
    fn, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_args']
    scope = {'argparse': argparse}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), scope)
    def parse(tokens):
        with patch.object(sys, 'argv', ['source-fixture'] + tokens):
            return scope['get_args']()
    defaults = {k: v for k,v in vars(parse(['--do_train'])).items() if k in wanted}
    assert defaults['rgb_pose_kl'] is False
    assert parse(['--do_train', '--rgb_pose_kl']).rgb_pose_kl is True
    targeted = argparse.ArgumentParser(add_help=False)
    selected = [ast.Expr(value=n) for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'add_argument' and n.args
                and isinstance(n.args[0], ast.Constant)
                and n.args[0].value in {'--' + k for k in wanted}]
    assert len(selected) == len(wanted)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                 str(source), 'exec'), {'parser': targeted, 'float': float})
    records = []
    files = [source, root / 'third_party/SEDS/modules/modeling.py', Path(__file__)]
    for dataset in ('ph', 'h2s', 'csl'):
        path = root / f'third_party/SEDS/scripts/train_{dataset}.sh'
        command = path.read_text().replace('\\\n', ' ')
        # Tokenization only; never execute shell substitutions or launch training.
        tokens = shlex.split(command, comments=True)
        start = tokens.index('main_task_retrieval.py') + 1
        error = io.StringIO()
        try:
            with contextlib.redirect_stderr(error):
                parse(tokens[start:])
        except SystemExit as exc:
            full_parse = {'exit_code': exc.code, 'last_error_line': error.getvalue().splitlines()[-1]}
        else:
            full_parse = {'exit_code': 0}
        args, _ = targeted.parse_known_args(tokens[start:])
        assert not args.rgb_pose_kl and args.rgb_pose_match
        assert args.rgb_pose_match_loss == .4
        full_defaults = parse(['--do_train'])
        assert not hasattr(full_defaults, 'freeze_exfusion')
        model_path = root / 'third_party/SEDS/modules/modeling.py'
        model_tree = ast.parse(model_path.read_bytes())
        assignment, = [n for n in ast.walk(model_tree) if isinstance(n, ast.Assign)
                       and isinstance(n.value, ast.Attribute)
                       and isinstance(n.value.value, ast.Name)
                       and n.value.value.id == 'task_config'
                       and n.value.attr == 'freeze_exfusion']
        try:
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(model_path), 'exec'),
                 {'self': SimpleNamespace(), 'task_config': full_defaults})
        except AttributeError:
            failure = 'AttributeError: parsed namespace lacks freeze_exfusion'
        else:
            raise AssertionError('Expected missing constructor attribute')
        records.append({'dataset': dataset, 'flags': {k:v for k,v in vars(args).items() if k in wanted},
                        'full_parser_current_python': full_parse,
                        'freeze_exfusion_present_in_full_default_namespace': False,
                        'isolated_constructor_assignment': failure})
        files.append(path)
    print(json.dumps({'status': 'completed', 'scope': 'released recipe flag parsing only',
                      'python_version': sys.version,
                      'default_flags': defaults, 'explicit_kl_flag_positive_control': True,
                      'recipes': records,
                      'sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in files},
                      'upstream_imports': False, 'data_or_checkpoint_access': False,
                      'training_launched': False, 'method_go': False}, indent=2))


if __name__ == '__main__':
    main()
