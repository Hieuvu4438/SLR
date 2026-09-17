"""Bounded AS-C18 interpretation; path partitions are not loss attribution."""
import json

import numpy as np

from .common import ROOT, dump, sha

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'


def main():
    source = OUT/'AS-C18-PARAMETER_run.json'
    run = json.loads(source.read_text())
    assert run['status'] == 'completed' and len(run['conditions']) == 6
    assert {(x['seed'], x['condition']) for x in run['conditions']} == {
        (s, c) for s in (42, 1337, 2026) for c in ('clean', 'deployed_aug')}
    report = {'experiment_id': 'AS-C18-SUMMARY', 'status': 'ANALYZED', 'method_go': False,
              'source_sha256': sha(source), 'code_sha256': sha(__file__),
              'gradient_parameter_n': run['gradient_parameter_n'], 'smoke': run['smoke'],
              'conditions': {}, 'records': []}
    for condition in ('clean', 'deployed_aug'):
        selected = [x for x in run['conditions'] if x['condition'] == condition]
        fields = {
            'valid_interface_cosine': [x['valid_interface']['global_cosine'] for x in selected],
            'all_interface_cosine': [x['all_slot_interface_cosine'] for x in selected],
            'parameter_cosine': [x['parameter_directional_cosine'] for x in selected],
            'duplicate_row_fraction': [x['valid_interface']['duplicate_example_fraction'] for x in selected],
            'valid_interface_duplicate_squared_norm_fraction': [x['valid_interface']['duplicate_squared_norm_mass_fraction'] for x in selected],
            'parameter_duplicate_signed_projection': [x['duplicate_video_pathways']['part_signed_projection_onto_total'] for x in selected],
            'parameter_invalid_signed_projection': [x['invalid_slot_pathways']['part_signed_projection_onto_total'] for x in selected]}
        report['conditions'][condition] = {k: {'mean': float(np.mean(v)), 'min': min(v), 'max': max(v)} for k, v in fields.items()}
        for x in selected:
            report['records'].append({'seed': x['seed'], 'condition': condition,
                'valid_interface_cosine': x['valid_interface']['global_cosine'],
                'all_interface_cosine': x['all_slot_interface_cosine'],
                'parameter_cosine': x['parameter_directional_cosine'],
                'duplicate_row_n': x['valid_interface']['duplicate_n'],
                'duplicate_projection': x['duplicate_video_pathways']['part_signed_projection_onto_total'],
                'invalid_projection': x['invalid_slot_pathways']['part_signed_projection_onto_total']})
    report['numerical_limits'] = [
        'Native FP16 VJPs with fixed scaling; no scale-sweep or full-float64 encoder verification.',
        'Eight-example direct-versus-chain smoke exact; not an independent full512-example backward verification.',
        'Complementary pathway vectors computed as total minus selected pathway; squared-norm identity is an algebra check, NOT an independent VJP closure test.',
        'Synthetic double test checks chunk additivity; mixed-precision rounding can make separately differentiated pathways imperfectly additive.',
        'One inactive conv2_trans.weight tensor included as zero; no claim that all declared parameters are used.']
    report['fallacy_scan_coverage'] = '11/11'
    report['fallacy_checks'] = {
        'Simpson': 'All six cells and three spaces reported; direction agreement not inferred solely from pooled means.',
        'Ecological': 'Batchwise shared-parameter gradients not claims about each signer or linguistic contrast.',
        'Berkson': 'Fixed PH-trained R0 and three prespecified batches; no all-training-stage generalization.',
        'Collider': 'Duplicate grouping selects input identity, not a causal intervention or adjusted effect.',
        'Base_rate': 'Duplicate row fractions retained alongside signed projections; projections not prevalence.',
        'Regression_to_mean': 'No actual treatment or downstream performance gain reported.',
        'Survivorship': 'All6 conditions completed and retained; no failed cells or negative cosines discarded.',
        'Look_elsewhere': 'Exploratory six-cell audit, no significance, model selection or GO claim.',
        'Forking_paths': 'Fixed scale, batches and tolerances; complementary residual validation limitations explicit.',
        'Correlation_causation': 'Projection and gradient alignment cannot establish harmful optimizer updates.',
        'Reverse_causality': 'One checkpoint sensitivity does not explain how the model arrived at its representation.'}
    report['decision'] = 'Simple opposing-direction story unsupported in this video-parameter scope. Pathway concentration is heterogeneous; no harmful-update evidence or novel PMGR/RPCA-free mechanism.'
    dump(OUT/'AS-C18-SUMMARY.json', report)
    print(json.dumps(report['conditions'], indent=2))


if __name__ == '__main__':
    main()
