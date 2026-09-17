"""Read-only head/score replay; never repeats training or opens TEST."""
import json
import time

import numpy as np
import torch

from .common import ART, ROOT, dump, metrics, sha
from .sentence_weight_pilot import REPORT, RUN, evaluate, load_cache, state_sha
from .sentence_weighting import SentenceWeighting, eot_vectors, random_conditioning


def main():
    start = time.monotonic()
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = json.loads(REPORT.read_text())
    assert report['status'] == 'completed'
    assert all(sha(ROOT / p) == h for p, h in report['code_sha256'].items())
    train, dev = load_cache('train'), load_cache('dev')
    qtrain = eot_vectors(train['text_tokens'], train['text_mask'])
    qdev = eot_vectors(dev['text_tokens'], dev['text_mask'])
    mean = torch.nn.functional.normalize(qtrain.mean(0), dim=0)
    shuffled, indexes = random_conditioning(qdev, qtrain)
    assert indexes == report['shuffled_bank_indices']['dev']
    conditions = {'sentence': qdev, 'blind': mean[None].expand_as(qdev), 'shuffled': shuffled}
    base = np.load(ART/'baseline_dev_channels.npy')
    outputs = []
    for mode, cond in conditions.items():
        arm = report['arms'][mode]
        assert arm['updates'] == 260 and len(arm['train_trace']) == 260
        assert arm['initial_state_sha256'] != arm['final_state_sha256']
        assert arm['selected_step'] == 0
        for step in (0, 130, 260):
            model = SentenceWeighting().cuda()
            path = RUN / mode / f'head_step{step}.pt'
            model.load_state_dict(torch.load(path, weights_only=True, map_location='cuda'))
            saved = np.load(RUN/mode/f'scores_step{step}.npy')
            replay, checks = evaluate(model, dev, cond, base)
            assert np.array_equal(saved, replay), (mode, step, float(np.abs(saved-replay).max()))
            m = metrics(replay, base.mean(0))
            original_m = json.loads((RUN/mode/f'metrics_step{step}.json').read_text())
            assert m == original_m
            if step == 260:
                assert state_sha(model) == arm['final_state_sha256']
            outputs.append({'arm': mode, 'step': step, 'score_replay_exact': True,
                            'all_metrics_and_ranks_exact': True,
                            'mean_R1': m['official_mean_R1'],
                            'T2V': {k:m['official_T2V'][k] for k in ('R1','R5','R10','MnR')},
                            'V2T': {k:m['V2T'][k] for k in ('R1','R5','R10','MnR')},
                            'persistent_rank_delta': {d:m[d]['persistent_mean_rank_delta'] for d in ('T2V','V2T')},
                            'head_sha256': sha(path), **checks})
    result = {'status': 'completed', 'run_sha256': sha(REPORT), 'outputs': outputs,
              'optimizer_updates_in_validation': 0, 'wall_seconds': time.monotonic()-start,
              'script_sha256': sha(__file__), 'independent_training_replication': False,
              'scope': 'Same-environment deterministic saved-head replay; not independent implementation'}
    dump(ROOT/'docs/proposal7/evidence/autonomous_search/CICO-REOPEN-01_validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
