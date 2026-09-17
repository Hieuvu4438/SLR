"""Post-hoc AS-C31 numerical clarification; no score or gate changes."""
from collections import defaultdict
import json
import os
import sys
import time
import traceback

import torch
import yaml

from slr_common.data.tokenize import encode_cico_text
from slr_common.upstream.cico_bridge import CiCoBridge
from slr_common.upstream.factory import _load_cico_core_from_state, load_cico_tokenizer
from .common import ART, ROOT, dump, rows, sha
from .text_augmentation_probe import OUT


@torch.inference_mode()
def main():
    path = OUT/'AS-C31-PREFIX-NUMERIC_run.json'
    if path.exists():
        raise FileExistsError(path)
    started = time.time()
    torch.set_num_threads(8)
    result = {'status': 'running', 'pid': os.getpid(), 'code_sha256': sha(__file__),
              'parent_run_sha256': sha(OUT/'AS-C31-PREFIX_run.json'), 'post_hoc': True,
              'method_go': False, 'updates': 0, 'test_loaded': False, 'splits': {}}
    dump(path, result)
    try:
        parent = json.loads((OUT/'AS-C31-PREFIX_run.json').read_text())
        assert parent['status'] == 'completed'
        cp = ROOT/'runs/ph_base_b512_s42/checkpoints/best_dev.pt'
        assert sha(cp) == parent['checkpoint_sha256']
        config = yaml.safe_load((cp.parents[1]/'resolved_config.yaml').read_text())
        cico_root = ROOT/config['upstream']['cico_root']
        sys.path.insert(0, str(cico_root))
        saved = torch.load(cp, weights_only=True, map_location='cpu', mmap=True)
        state = {k.removeprefix('core.'): v for k, v in saved['model'].items() if k.startswith('core.')}
        core = _load_cico_core_from_state(config, state, cico_root=cico_root, device='cuda')
        del saved, state
        core.eval().requires_grad_(False)
        bridge, tok = CiCoBridge(core), load_cico_tokenizer(config)
        for split in ('train', 'dev'):
            cache_path = ART/f'frozen_{split}.pt'
            assert sha(cache_path) == parent['splits'][split]['cache_sha256']
            cache, rec = torch.load(cache_path, weights_only=True), rows(split)
            triples = [encode_cico_text(r['caption_model'], tok, 32) for r in rec]
            seqs = [tuple(x[0][1:int(x[2].sum())-1].tolist()) for x in triples]
            groups = defaultdict(list)
            for i, seq in enumerate(seqs):
                for k in range(1, len(seq)+1):
                    groups[seq[:k]].append(i)
            tail_start = len(rec)//128*128
            counts = {'equal_128_vs_128': 0, 'different_128_vs_128': 0,
                      'equal_tail_vs_tail': 0, 'different_tail_vs_tail': 0,
                      'equal_128_vs_tail': 0, 'different_128_vs_tail': 0}
            examples = []
            for prefix, ids in sorted(groups.items()):
                if len({seqs[i] for i in ids}) <= 1:
                    continue
                a, pos = ids[0], len(prefix)
                for b in ids[1:]:
                    equal = torch.equal(cache['text_tokens'][a, pos], cache['text_tokens'][b, pos])
                    bucket = ('128_vs_128' if a < tail_start and b < tail_start else
                              'tail_vs_tail' if a >= tail_start and b >= tail_start else '128_vs_tail')
                    counts[('equal_' if equal else 'different_')+bucket] += 1
                    if not equal and seqs[a] != seqs[b] and len(examples) < 16:
                        examples.append((a, b, pos))
            checks = []
            for a, b, pos in examples:
                if time.time()-started > 300:
                    raise TimeoutError('AS-C31 numeric timeout300s')
                inputs = [torch.stack([triples[a][k]]*128).cuda() for k in range(3)]
                first = bridge.encode_text(*inputs)
                tail = bridge.encode_text(*(x[:len(rec)-tail_start] for x in inputs))
                changed = [x.clone() for x in inputs]
                for k in range(3):
                    changed[k][0] = triples[b][k].cuda()
                assert torch.equal(inputs[0][0, :pos+1], changed[0][0, :pos+1])
                second = bridge.encode_text(*changed)
                delta = float((first.tokens[0, :pos+1]-second.tokens[0, :pos+1]).abs().max())
                checks.append({'a': a, 'b': b, 'common_prefix_at_least': pos,
                               'fixed_shape_suffix_intervention_prefix_max_delta': delta,
                               'fixed_shape_suffix_intervention_eot_max_delta': float((first.cls[0]-second.cls[0]).abs().max()),
                               'same_input_full_vs_tail_shape_prefix_max_delta': float((first.tokens[0, :pos+1]-tail.tokens[0, :pos+1]).abs().max())})
            result['splits'][split] = {'tail_batch_size': len(rec)-tail_start,
                                      'representative_comparisons': counts, 'fixed_first16_checks': checks}
            dump(path, result)
            print(json.dumps({'split': split, 'counts': counts, 'check_n': len(checks),
                              'max_suffix_prefix_delta': max((x['fixed_shape_suffix_intervention_prefix_max_delta'] for x in checks), default=None)}), flush=True)
        result.update(status='completed', wall_seconds=time.time()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated())
        dump(path, result)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
