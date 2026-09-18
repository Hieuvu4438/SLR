"""CPU replay of existing CSL scores; not a model/feature reproduction."""
import json
import sys
import time
import numpy as np

from inventory import ROOT, sha
from extraction_resume import atomic_json


def main():
    started = time.time()
    sys.path.insert(0,str(ROOT/'shared'))
    from slr_common.data.manifest import load_manifest
    from slr_common.evaluation.cico_eval import evaluate_score_matrix
    from slr_common.utils import ordered_hash
    out = ROOT/'artifacts/slret_goal/csl-score-contract-001'
    out.mkdir(exist_ok=False)
    source = ROOT/'runs/csl_base_b512_s42'
    old = json.loads((source/'evaluation/dev/metrics.json').read_text())
    selection = json.loads((source/'selection.json').read_text())
    manifest = ROOT/'artifacts/manifests/csl_dev.jsonl'
    assert sha(manifest) == selection['dev_manifest_sha256']
    records = load_manifest(manifest,expected_split='dev')
    videos = list(dict.fromkeys(r.video_id for r in records))
    texts = list(dict.fromkeys(r.caption_id for r in records))
    assert videos == [r['query_id'] for r in old['per_query']['V2T']]
    assert texts == [r['query_id'] for r in old['per_query']['T2V']]
    assert ordered_hash(videos) == old['id_hashes']['videos']
    assert ordered_hash(texts) == old['id_hashes']['texts']
    v2t = {r.video_id:[r.caption_id] for r in records}
    t2v = {t:[r.video_id for r in records if r.caption_id==t] for t in texts}
    score_path = source/'evaluation/dev/scores_video_x_text.npy'
    matrix = np.load(score_path)
    actual = evaluate_score_matrix(matrix,video_ids=videos,text_ids=texts,video_to_text=v2t,text_to_video=t2v)
    for direction in ['T2V','V2T']:
        assert actual[direction] == old[direction], direction
    mean = sum(actual[d]['R1'] for d in ['T2V','V2T'])/2
    assert abs(mean-selection['best_value']) < 1e-10
    report = dict(run_id=out.name,status='completed',exit_status=0,test_loaded=False,
                  kind='cached_score_metric_replay_not_fresh_inference',script_sha256=sha(__file__),
                  manifest_sha256=sha(manifest),scores_sha256=sha(score_path),
                  historical_metrics_sha256=sha(source/'evaluation/dev/metrics.json'),
                  selection_sha256=sha(source/'selection.json'),gallery=actual['gallery'],
                  metric_kernel=actual['metric_kernel'],mean_R1=mean,
                  metrics={d:{k:v for k,v in actual[d].items() if k!='cols'} for d in ['T2V','V2T']},
                  rank_parity='exact_both_directions',wall_seconds=time.time()-started,
                  decision='Use grouped CSL protocol; historical DEV is exposed, not fresh confirmation.')
    atomic_json(out/'run.json',report)
    with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:
        f.write(json.dumps(report)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
