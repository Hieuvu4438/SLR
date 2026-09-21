"""Single C17 DCL pilot, using native TRAIN512 without loading teacher features."""
import json
from pathlib import Path
import sys
import run_c16_control_v4 as runner

RUN = 'seds-fused-dcl-offload-001'
REFERENCE = 'seds-signrep-control-offload-001'
JOB = 'v4-c17-dcl-001'
DCL_WEIGHT = 1.
ARTIFACT_CAP = 38


def validate(result):
    base = runner.BASE
    if result.get('status') != 'completed' or not (base/RUN/'last.pt').is_file():
        raise ValueError('Missing successful report or last checkpoint')
    anchor = json.loads((base/REFERENCE/'run.json').read_text())
    for key in ('batch_order_sha256','checkpoint_sha256_inherited','inherited_asset_digests'):
        if result[key] != anchor[key]:
            raise ValueError('Matched control mismatch: '+key)
    for key in ('train_ids','dev_ids'):
        if result['subset_data'][key] != anchor['signrep_data'][key]:
            raise ValueError('Subset mismatch: '+key)
    if result['subset_data']['teacher_features_loaded'] or result['test_loaded']:
        raise ValueError('Unexpected teacher/TEST data')
    config = result['config']
    if not config['native_subset'] or config['fusion_objective'] != 'dcl' or config['signrep'] is not None:
        raise ValueError('Wrong C17 intervention')
    if config.get('dcl_weight',1.) != DCL_WEIGHT:
        raise ValueError('Wrong registered DCL weight')
    for key,value in anchor['config'].items():
        if key not in ('run_id','signrep','artifact_cap_gib') and config[key] != value:
            raise ValueError('Recipe mismatch: '+key)
    for stream in ('fusion','pose','rgb'):
        for direction in ('T2V','V2T'):
            for metric in ('R1','R5','R10'):
                if abs(result['evaluations']['0'][stream][direction][metric] -
                       anchor['evaluations']['0'][stream][direction][metric]) > 1e-10:
                    raise ValueError('Initial recall mismatch')
    for key in ('masked_update_checks_passed','delta_roundtrip_passed','dcl_loss_replacement_passed'):
        if not result.get(key):
            raise ValueError('Missing gate: '+key)
    criterion = 'fused_dcl_native_branches' if DCL_WEIGHT == 1 else 'fused_dcl_blend_native_branches'
    if result['criterion'] != criterion:
        raise ValueError('Wrong loss')
    best = Path(result['selection']['checkpoint'])
    if not best.is_file():
        raise ValueError('Selected checkpoint absent')
    expected = result['selection'].get('checkpoint_sha256',result['checkpoint_sha256_inherited'])
    if runner.sha(best) != expected:
        raise ValueError('Selected checksum mismatch')
    return result['selection']['mean_R1']-anchor['selection']['mean_R1']


def configure():
    runner.RUN = RUN
    runner.JOB = JOB
    runner.OUT = runner.ROOT/'artifacts/slret_goal/jobs'/runner.JOB
    runner.MODE = 'native_subset'
    runner.REFERENCE = REFERENCE
    runner.ARTIFACT_CAP = ARTIFACT_CAP
    runner.EXTRA_SOURCES = [Path(__file__).resolve()]
    runner.INPUT_FLAGS = ['--native-subset','--fusion-objective','dcl','--dcl-weight',str(DCL_WEIGHT)]
    runner.validate = validate


if __name__ == '__main__':
    configure()
    if sys.argv[1:] == ['--launch']:
        runner.launch(__file__)
    elif sys.argv[1:]:
        raise SystemExit('Only --launch is accepted')
    else:
        runner.main()
