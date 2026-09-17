"""AS-C30: immutable AS-C20 replay plus a two-table freeze intervention."""
import argparse
import ast
from collections import Counter
import hashlib
import json

import numpy as np
import torch

from .common import ART, ROOT, dump, sha
from .clean_initialization_audit import OUT

SOURCE = ROOT/'methods/information_probe/clean_train_calibration.py'
SOURCE_SHA = 'e812d98277cd7f7d00c02600722ba53a5d2dafc767f3db44df6a26f572d7dc71'
FROZEN = ('clip.positional_embedding', 'clip.token_embedding.weight')


def transformed_tree(source, arm):
    """Change ONLY four output/provenance literals, never training operations."""
    assert arm in ('control', 'freeze')
    label = 'AS-C30-' + arm.upper()
    replacements = {'AS-C20-TRAIN_run.json': label+'_run.json',
                    'AS-C20': label, 'AS-C20-TRAIN': label,
                    'AS-C20_protocol.md': 'AS-C30_protocol.md'}
    tree = ast.parse(source)
    counts = Counter(n.value for n in ast.walk(tree)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str)
                     and n.value in replacements)
    assert counts == Counter({k: 1 for k in replacements}), counts
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in replacements:
            node.value = replacements[node.value]
    return tree, replacements


def freeze_tables(core, arm):
    params = dict(core.named_parameters())
    assert all(p.requires_grad for p in params.values())
    assert all(k in params for k in FROZEN)
    if arm == 'freeze':
        for key in FROZEN:
            params[key].requires_grad_(False)
    frozen = [k for k, p in params.items() if not p.requires_grad]
    assert set(frozen) == (set(FROZEN) if arm == 'freeze' else set())
    return {'frozen_keys': frozen,
            'frozen_parameter_n': sum(params[k].numel() for k in frozen),
            'trainable_parameter_n': sum(p.numel() for p in params.values() if p.requires_grad)}


def compare_control(result):
    original = json.loads((OUT/'AS-C20-TRAIN_run.json').read_text())
    score_checks = {}
    for step in (0, 250, 500, 1000):
        for split in ('fit', 'held'):
            name = f'{split}_step{step}.npy'
            a, b = ART/'AS-C20'/name, ART/'AS-C30-CONTROL'/name
            score_checks[name] = {'file_sha256_equal': sha(a) == sha(b),
                                 'array_equal': bool(np.array_equal(np.load(a), np.load(b)))}
    old = torch.load(original['checkpoint']['path'], map_location='cpu', weights_only=False)['state_dict']
    new = torch.load(result['checkpoint']['path'], map_location='cpu', weights_only=False)['state_dict']
    mismatches = [k for k in old if k not in new or not torch.equal(old[k], new[k])]
    extra = sorted(set(new)-set(old))
    exact = (not mismatches and not extra and result['training'] == original['training']
             and result['evaluations'] == original['evaluations']
             and all(all(v.values()) for v in score_checks.values()))
    return {'exact_replay': exact, 'score_checks': score_checks,
            'training_log_equal': result['training'] == original['training'],
            'evaluation_records_equal': result['evaluations'] == original['evaluations'],
            'state_tensor_n': len(old), 'state_tensor_mismatches': mismatches,
            'extra_state_keys': extra,
            'checkpoint_container_bytes_not_compared': 'protocol metadata intentionally differs'}


def main(arm):
    assert sha(SOURCE) == SOURCE_SHA, 'immutable calibration source changed'
    if arm == 'freeze':
        control = json.loads((OUT/'AS-C30-CONTROL_run.json').read_text())
        assert control['status'] == 'completed' and control['comparison']['exact_replay']
        assert control['harness_sha256'] == sha(__file__)
        assert control['protocol_sha256'] == sha(OUT/'AS-C30_protocol.md')
    tree, replacements = transformed_tree(SOURCE.read_text(), arm)
    namespace = {'__name__': 'methods.information_probe._asc30_replay',
                 '__package__': 'methods.information_probe', '__file__': str(SOURCE)}
    exec(compile(tree, str(SOURCE), 'exec'), namespace)
    original_initialize = namespace['initialize']
    held = {}

    def initialize(*args, **kwargs):
        core, report = original_initialize(*args, **kwargs)
        held['freeze_audit'] = freeze_tables(core, arm)
        held['core'] = core
        held['initial_tables'] = {k: core.state_dict()[k].detach().float().cpu().clone() for k in FROZEN}
        return core, report

    def record(path, result):
        result.update(arm=arm, harness_sha256=sha(__file__),
                      literal_replacements=replacements,
                      transformed_ast_sha256=hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
                      environment={'torch': torch.__version__, 'cuda': torch.version.cuda,
                                   'gpu': torch.cuda.get_device_name(0)})
        if 'freeze_audit' in held:
            result['freeze_audit'] = held['freeze_audit']
        if result['status'] == 'completed':
            result['final_table_equals_initial'] = {
                k: torch.equal(held['core'].state_dict()[k].detach().cpu(), v)
                for k, v in held['initial_tables'].items()}
            if arm == 'freeze':
                assert all(result['final_table_equals_initial'].values())
            else:
                result['comparison'] = compare_control(result)
        dump(path, result)

    namespace['initialize'] = initialize
    namespace['dump'] = record
    namespace['main']()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('control', 'freeze'), required=True)
    main(parser.parse_args().arm)
