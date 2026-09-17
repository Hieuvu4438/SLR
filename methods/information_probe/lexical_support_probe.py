"""Input-only common-reference support, never a cross-model causal comparison."""
import json
import os
import time
import traceback

import numpy as np
from scipy.sparse import csr_matrix
import torch
import yaml

from slr_common.data.tokenize import encode_cico_text
from slr_common.upstream.factory import load_cico_tokenizer
from .common import ART, ROOT, dump, ranks, rows, sha

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'


def nearest_jaccard(queries, reference, query_sources, reference_sources):
    vocab = {v: i for i, v in enumerate(sorted(set().union(*queries, *reference)))}
    def matrix(values):
        rr, cc = [], []
        for i, s in enumerate(values):
            for value in s:
                rr.append(i); cc.append(vocab[value])
        return csr_matrix((np.ones(len(rr), dtype=np.int32), (rr, cc)), shape=(len(values), len(vocab)))
    q, r = matrix(queries), matrix(reference)
    qn, rn = np.array([len(x) for x in queries]), np.array([len(x) for x in reference])
    output = []
    for start in range(0, len(qn), 128):
        intersect = (q[start:start+128]@r.T).toarray()
        union = qn[start:start+128, None]+rn[None]-intersect
        score = np.divide(intersect, union, out=np.ones_like(intersect, dtype=np.float64), where=union != 0)
        for j, s in enumerate(score):
            i = start+j
            best = int(s.argmax())
            valid = np.array([x != query_sources[i] for x in reference_sources])
            other = int(np.where(valid, s, -np.inf).argmax()) if valid.any() else None
            output.append({'best_jaccard': float(s[best]), 'reference_index': best,
                           'cross_source_best_jaccard': float(s[other]) if other is not None else None,
                           'cross_source_reference_index': other})
    return output


def summarize(values):
    x = np.asarray(values, dtype=float)
    return {'mean': float(x.mean()), 'quantiles_0_25_50_75_100': np.quantile(x, [0, .25, .5, .75, 1]).tolist()}


def main():
    path = OUT/'AS-C26-SUPPORT_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    started = time.time()
    result = {'experiment_id': 'AS-C26-SUPPORT', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C26_protocol.md'),
              'method_go': False, 'test_loaded': False, 'regimes': {}}
    dump(path, result)
    try:
        train, dev = rows('train'), rows('dev')
        pp = OUT/'AS-C19-TRAIN-partition.json'
        partition = json.loads(pp.read_text())
        assert partition['manifest_sha256'] == sha(ROOT/'artifacts/manifests/ph_train.jsonl')
        result.update(partition_sha256=sha(pp), train_manifest_sha256=partition['manifest_sha256'], dev_manifest_sha256=sha(ROOT/'artifacts/manifests/ph_dev.jsonl'))
        config = yaml.safe_load((ROOT/'runs/ph_base_b512_s42/resolved_config.yaml').read_text())
        tok = load_cico_tokenizer(config)
        def describe(rec):
            tokens = [tok.convert_tokens_to_ids(tok.tokenize(r['caption_model'])) for r in rec]
            return {'unigram': [set(x) for x in tokens], 'bigram': [set(zip(x, x[1:])) for x in tokens],
                    'length': [len(x) for x in tokens], 'source': [r['video_id'].rsplit('-', 1)[0] for r in rec],
                    'key': [tuple(encode_cico_text(r['caption_model'], tok, 32)[0].tolist()) for r in rec]}
        td, dd = describe(train), describe(dev)
        fit, held = partition['fit_indexes'], partition['held_indexes']
        calibration_path = OUT/'AS-C20-TRAIN_run.json'
        calibration = json.loads(calibration_path.read_text())
        assert calibration['status'] == 'completed' and calibration['updates'] == 1000
        result['calibration_run_sha256'] = sha(calibration_path)
        hs = ART/'AS-C20/held_step1000.npy'
        ds = ROOT/'runs/ph_base_b512_s42/evaluation/dev/scores_video_x_text.npy'
        assert sha(hs) == calibration['evaluations'][-1]['held']['score_sha256']
        result['score_sha256'] = {'held': sha(hs), 'dev': sha(ds)}
        hr, dr = ranks(np.load(hs)), ranks(np.load(ds))
        for name, desc, records, qi, ri, rr in [
            ('held_vs_fit', td, train, held, fit, hr), ('dev_vs_fit', dd, dev, list(range(519)), fit, dr),
            ('dev_vs_all_train', dd, dev, list(range(519)), list(range(7096)), dr)]:
            if time.time()-started > 300:
                raise TimeoutError('AS-C26 timeout300s')
            qs, rs = [desc['source'][i] for i in qi], [td['source'][i] for i in ri]
            nearest = {k: nearest_jaccard([desc[k][i] for i in qi], [td[k][i] for i in ri], qs, rs) for k in ('unigram', 'bigram')}
            refkeys = {td['key'][i] for i in ri}
            counts = {s: rs.count(s) for s in set(rs)}
            audit = []
            for j, i in enumerate(qi):
                x = {'index': i, 'pair_id': records[i]['pair_id'], 'source': qs[j], 'content_length': desc['length'][i],
                     'same_source_reference_rows': counts.get(qs[j], 0), 'exact_deployed_match': desc['key'][i] in refkeys,
                     'ranks': {d: int(rr[d][j]) for d in rr}}
                for k in nearest:
                    y = dict(nearest[k][j])
                    for field in ('reference_index', 'cross_source_reference_index'):
                        if y[field] is not None:
                            y[field] = ri[y[field]]
                    x[k] = y
                audit.append(x)
            bins = np.digitize([x['unigram']['best_jaccard'] for x in audit], [.25, .5, .75, 1.])
            summary = {'query_n': len(qi), 'reference_n': len(ri),
                'same_source_query_n': sum(x['same_source_reference_rows'] > 0 for x in audit),
                'exact_deployed_match_n': sum(x['exact_deployed_match'] for x in audit),
                'content_length': summarize([x['content_length'] for x in audit]),
                **{f'{k}_{field}': summarize([x[k][field] for x in audit]) for k in nearest for field in ('best_jaccard', 'cross_source_best_jaccard')},
                'fixed_unigram_bins': []}
            for b, label in enumerate(('[0,.25)', '[.25,.5)', '[.5,.75)', '[.75,1)', '{1}')):
                mask = bins == b
                summary['fixed_unigram_bins'].append({'bin': label, 'n': int(mask.sum()),
                    'per_query_R1': {d: float(100*np.mean(rr[d][mask] == 0)) if mask.any() else None for d in rr}})
            result['regimes'][name] = {'summary': summary, 'rows': audit}
            dump(path, result)
            print(json.dumps({'regime': name, **summary}), flush=True)
        result.update(status='completed', wall_seconds=time.time()-started)
        dump(path, result)
        print(json.dumps({'status': result['status'], 'wall_seconds': result['wall_seconds']}), flush=True)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
