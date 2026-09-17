"""Uniform content-token omission inventory; no training or intervention."""
from collections import Counter, defaultdict
import json
import os
import time
import traceback

import numpy as np
import torch
import yaml

from slr_common.data.tokenize import encode_cico_text
from slr_common.upstream.factory import load_cico_tokenizer
from .common import ROOT, dump, ranks, rows, sha

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'


def selected_indexes(n, capacity=30):
    return list(range(n)) if n <= capacity else np.linspace(0, n-1, capacity, dtype=int).tolist()


def new_collisions(full, deployed):
    groups = defaultdict(list)
    for i, key in enumerate(deployed):
        groups[tuple(key)].append(i)
    return [idx for idx in groups.values() if len({tuple(full[i]) for i in idx}) > 1]


def main():
    path = OUT/'AS-C22-OMISSIONS_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    started = time.time()
    result = {'experiment_id': 'AS-C22-OMISSIONS', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C22_protocol.md'),
              'tokenizer_contract_sha256': sha(ROOT/'shared/slr_common/data/tokenize.py'),
              'method_go': False, 'test_loaded': False, 'splits': {}}
    dump(path, result)
    try:
        config = yaml.safe_load((ROOT/'runs/ph_base_b512_s42/resolved_config.yaml').read_text())
        tok = load_cico_tokenizer(config)
        for split in ('train', 'dev'):
            records, audit, full, deployed = rows(split), [], [], []
            for i, r in enumerate(records):
                if time.time()-started > 300:
                    raise TimeoutError('AS-C22 timeout300s')
                tokens = tok.tokenize(r['caption_model'])
                selected = selected_indexes(len(tokens))
                omitted = sorted(set(range(len(tokens)))-set(selected))
                constructed = tok.convert_tokens_to_ids(['<|startoftext|>', *[tokens[j] for j in selected], '<|endoftext|>'])
                constructed += [0]*(32-len(constructed))
                actual = encode_cico_text(r['caption_model'], tok, 32)[0].tolist()
                assert constructed == actual, (split, i)
                full.append(tok.convert_tokens_to_ids(tokens))
                deployed.append(actual)
                audit.append({'index': i, 'pair_id': r['pair_id'], 'content_token_n': len(tokens),
                              'selected_indexes': selected, 'omitted_indexes': omitted,
                              'omitted_tokens': [tokens[j] for j in omitted]})
            lengths = np.array([r['content_token_n'] for r in audit])
            result['splits'][split] = {'manifest_sha256': sha(ROOT/f'artifacts/manifests/ph_{split}.jsonl'),
                'n': len(audit), 'all_deployed_token_ids_exact': True,
                'affected_n': int((lengths > 30).sum()), 'beyond75_n': int((lengths > 75).sum()),
                'total_omitted_tokens': int(np.maximum(lengths-30, 0).sum()),
                'length_min_median_p90_p95_p99_max': np.quantile(lengths, [0, .5, .9, .95, .99, 1]).tolist(),
                'new_collision_groups': new_collisions(full, deployed), 'rows': audit}
            if split == 'dev':
                affected = lengths > 30
                by_seed = {}
                for seed in (42, 1337, 2026):
                    p = ROOT/f'runs/ph_base_b512_s{seed}/evaluation/dev/scores_video_x_text.npy'
                    s = np.load(p)
                    r = ranks(s)
                    wrong = s.copy()
                    np.fill_diagonal(wrong, -np.inf)
                    by_seed[str(seed)] = {'score_sha256': sha(p), 'directions': {}}
                    for d, axis in [('T2V', 0), ('V2T', 1)]:
                        hard = wrong.argmax(axis=axis)
                        by_seed[str(seed)]['directions'][d] = {
                            'ranks': r[d].tolist(), 'hard_candidate': hard.tolist(),
                            'affected_query_wrong_n': int((affected & (r[d] > 0)).sum()),
                            'unaffected_query_wrong_n': int((~affected & (r[d] > 0)).sum()),
                            'affected_query_R1': float(100*np.mean(r[d][affected] == 0)) if affected.any() else None,
                            'unaffected_query_R1': float(100*np.mean(r[d][~affected] == 0)) if (~affected).any() else None}
                support = {}
                for d in ('T2V', 'V2T'):
                    rr = np.array([by_seed[str(s)]['directions'][d]['ranks'] for s in (42, 1337, 2026)])
                    pp = (rr > 0).all(0)
                    hard = np.array(by_seed['42']['directions'][d]['hard_candidate'])
                    relevant = affected if d == 'T2V' else affected | affected[hard]
                    support[d] = {'persistent_n': int(pp.sum()), 'affected_positive_persistent_n': int((pp & affected).sum()),
                        'involved_persistent_n': int((pp & relevant).sum()),
                        'involved_persistent_indexes': np.flatnonzero(pp & relevant).tolist(),
                        'involved_top10_persistent_n': int((pp & relevant & (rr[0] < 10)).sum()),
                        'affected_seed42_confuser_persistent_n': int((pp & affected[hard]).sum()) if d == 'V2T' else None}
                result['historical_dev'] = by_seed
                result['support'] = support
                result['restoration_diagnostic_feasible'] = sum(support[d]['involved_persistent_n'] for d in support) >= 3
        result.update(status='completed', wall_seconds=time.time()-started)
        dump(path, result)
        print(json.dumps({'status': result['status'], 'splits': {k: {n: v for n, v in x.items() if n != 'rows'} for k, x in result['splits'].items()},
                          'support': result['support'], 'restoration_diagnostic_feasible': result['restoration_diagnostic_feasible'],
                          'wall_seconds': result['wall_seconds']}, indent=2))
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
