"""Descriptive summaries and explicit inference limits for AS-C11/C12."""
import json
import numpy as np

from .common import ROOT,dump,sha

OUT=ROOT/'docs/proposal7/evidence/autonomous_search'


def main():
    geometry=json.loads((OUT/'AS-C11-GEOMETRY_run.json').read_text())
    gradient=json.loads((OUT/'AS-C12-GRADIENT_run.json').read_text())
    assert geometry['status']==gradient['status']=='completed'
    assert len(gradient['batches'])==24
    summary={}
    fields=['global_cosine','negative_example_cosine_fraction','duplicate_example_fraction',
            'duplicate_squared_norm_mass_fraction','top10_examples_squared_norm_mass_fraction',
            'V2T_gradient_norm','T2V_gradient_norm']
    for condition in gradient['conditions']:
        rows=[r for r in gradient['batches'] if r['condition']==condition]
        summary[condition]={'n_batches':len(rows)}
        for field in fields:
            x=np.array([r[field] for r in rows])
            summary[condition][field]={'mean':float(x.mean()),'min':float(x.min()),'max':float(x.max())}
    report={'experiment_id':'AS-C11-C12-SUMMARY','status':'ANALYZED',
            'sources':{name:sha(OUT/name) for name in ('AS-C11-GEOMETRY_run.json','AS-C12-GRADIENT_run.json')},
            'code_sha256':sha(__file__), 'gradient_summary':summary,
            'geometry_deltas_vs_identity_pp':{name:x['mean_R1']-geometry['variants']['identity']['mean_R1']
                                            for name,x in geometry['variants'].items()},
            'gradient_space_clarification':'Reported cosine/norms mask video_mask!=0; VALID video-token interface only. CLS/padding gradients and shared encoder parameter Jacobians are not measured.',
            'attribution_limit':'Squared gradient norm across separate per-example token coordinates is not additive attribution to one shared parameter update.',
            'fallacy_scan_coverage':'11/11',
            'fallacy_checks':{
                'Simpson':'Both directions and clean/aug strata retained; no pooled directional improvement claim.',
                'Ecological':'Batch-mean summaries not claims about each caption or individual signer.',
                'Berkson':'Fixed trained-backbone and duplicate strata are selected; no population generalization.',
                'Collider':'No regression adjustment or causal conditioned association claimed.',
                'Base_rate':'Duplicate example fractions reported alongside gradient squared-norm fractions.',
                'Regression_to_mean':'No treatment improvement claimed for selected persistent errors.',
                'Survivorship':'All7 geometry variants and24 gradient conditions retained; terminal reports checked.',
                'Look_elsewhere':'Exploratory repeated dev reuse; no p-value, significance or method selection.',
                'Forking_paths':'Fixed protocols precede runs; random controls and precision/mask conventions disclosed.',
                'Correlation_causation':'Interface concentration is not harmful parameter-update attribution.',
                'Reverse_causality':'No learning-direction claim from one checkpoint; geometry manipulations scoped to fixed inference.'},
            'decision':'No candidate GO. Geometry controls fail; measured interface gradient concentration intersects closed PMGR/RPCA mechanisms, not permission to reopen them.'}
    dump(OUT/'AS-C11-C12-SUMMARY.json',report)
    print(json.dumps({'gradient_summary':summary,'geometry_deltas':report['geometry_deltas_vs_identity_pp']},indent=2))


if __name__=='__main__':
    main()
