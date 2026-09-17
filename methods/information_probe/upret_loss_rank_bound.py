"""Posthoc analytic inference from saved losses, not a fresh model evaluation."""
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    record_path = ROOT / 'docs/proposal7/evidence/autonomous_search/UPRET-REAL-GRADIENT_run.json'
    raw = record_path.read_bytes()
    record = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == '9a9d34bafd5fcbd7702b098dc77e83e2c40bc5517b92a73346ce1973c54be25f'
    config_path = ROOT / 'methods/sssc/configs/method1/ph_seed42_base_initial.yaml'
    config = config_path.read_bytes()
    assert hashlib.sha256(config).hexdigest() == record['config_sha256']
    assert b'dual_mix: 0.5' in config and b'mix_design: balance' in config
    baseline_path = ROOT / 'methods/sssc/method1/baseline.py'
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == record['implementation']['files']['method1/baseline.py']
    assert len(record['batch_evidence']) == 4
    assert all(len(batch) == 32 for batch in record['batch_evidence'])
    results = []
    for state in record['states']:
        for batch in state['screen_batches']:
            loss = batch['arms']['no_ot']['loss']
            # Four equally weighted CEs, each averaged over 32 queries.
            total_ce = 4 * 32 * loss
            margin_bound = -math.log(math.expm1(total_ce))
            wrong_bound = min(128, math.floor(total_ce / math.log(2)))
            results.append({
                'state': state['label'], 'batch': batch['batch'],
                'saved_no_ot_loss': loss,
                'maximum_single_query_ce_bound': total_ce,
                'every_pair_logit_margin_lower_bound': margin_bound,
                'nonpositive_margin_query_count_upper_bound_of_128': wrong_bound,
                'all_four_directional_channels_strictly_correct_implied': margin_bound > 0,
            })
    print(json.dumps({
        'scope': 'Real-arithmetic consequences of saved FP32 loss values; not recomputed ranks',
        'source_record_sha256': hashlib.sha256(raw).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'one_wrong_query_total_loss_floor': math.log(2) / 128,
        'source_default_ot_score_bounds_unit_mass': [-0.25, 0.5],
        'source_default_ot_margin_perturbation_bound': 0.75,
        'results': results,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
