"""One native B64 pilot: same TRAIN exposures as B32, no DCL/teacher/new scorer."""
import json
from pathlib import Path
import sys
import run_c16_control_v4 as runner

RUN = 'seds-native-batch64-offload-001'
REFERENCE = 'seds-signrep-control-offload-001'


def check_exposure(candidate,control):
    if len(candidate)!=80 or len(control)!=160:
        raise ValueError('Expected80 B64 and160 B32 updates')
    if any(len(b)!=64 for b in candidate) or any(len(b)!=32 for b in control):
        raise ValueError('Wrong contrastive batch size')
    for i,b in enumerate(candidate):
        if b!=control[2*i]+control[2*i+1]:
            raise ValueError('Training order/exposure mismatch')


def validate(result):
    base=runner.BASE
    if result.get('status')!='completed' or not (base/RUN/'last.pt').is_file():
        raise ValueError('Missing successful result or last checkpoint')
    anchor=json.loads((base/REFERENCE/'run.json').read_text())
    config=result['config']
    if not config['native_subset'] or config['batch_size']!=64 or config['signrep'] is not None:
        raise ValueError('Wrong C18 inputs')
    if config['fusion_objective']!='native' or result['criterion']!='native':
        raise ValueError('C18 must use native losses; DCL is user-closed')
    for key,value in anchor['config'].items():
        if key not in ('run_id','signrep','artifact_cap_gib','batch_size') and config[key]!=value:
            raise ValueError('Recipe mismatch: '+key)
    for key in ('checkpoint_sha256_inherited','inherited_asset_digests'):
        if result[key]!=anchor[key]:raise ValueError('Asset mismatch: '+key)
    for key in ('train_ids','dev_ids'):
        if result['subset_data'][key]!=anchor['signrep_data'][key]:raise ValueError('ID mismatch')
    if result['test_loaded'] or result['subset_data']['teacher_features_loaded']:
        raise ValueError('Unexpected TEST/teacher inputs')
    check_exposure(json.loads((base/RUN/'batch_indices.json').read_text()),
                   json.loads((base/REFERENCE/'batch_indices.json').read_text()))
    for stream in ('fusion','pose','rgb'):
        for direction in ('T2V','V2T'):
            for metric in ('R1','R5','R10'):
                if abs(result['evaluations']['0'][stream][direction][metric]-
                       anchor['evaluations']['0'][stream][direction][metric])>1e-10:
                    raise ValueError('Initial recall mismatch')
    for key in ('masked_update_checks_passed','delta_roundtrip_passed'):
        if not result.get(key):raise ValueError('Missing real update/load gate: '+key)
    selected=Path(result['selection']['checkpoint'])
    expected=result['selection'].get('checkpoint_sha256',result['checkpoint_sha256_inherited'])
    if not selected.is_file() or runner.sha(selected)!=expected:
        raise ValueError('Selected checkpoint missing or changed')
    return result['selection']['mean_R1']-anchor['selection']['mean_R1']


def configure():
    runner.RUN=RUN
    runner.JOB='v4-c18-batch64-001'
    runner.OUT=runner.ROOT/'artifacts/slret_goal/jobs'/runner.JOB
    runner.MODE='native_batch64'
    runner.REFERENCE=REFERENCE
    runner.TOTAL_UNITS=80
    runner.HARD_SECONDS=2400
    runner.OUTER_SECONDS=2450
    runner.ESTIMATED_SECONDS=1800
    runner.COMMAND_OVERRIDE=[sys.executable,'-u','-m','torch.distributed.run','--standalone',
        '--nproc_per_node=1',str(runner.ROOT/'research/slret_goal_v2/tools/train_seds_extended.py'),
        '--run-id',RUN,'--native-subset','--fusion-objective','native','--batch-size','64',
        '--masked-pose','control','--policy','fusion','--aux-weight','1','--seed','42',
        '--epochs','10','--lr','1e-5','--sign-lr','1e-6','--early-stop-drop-pp','2',
        '--activation-offload','--allocator-cap-gib','16','--artifact-cap-gib','40']
    runner.validate=validate


if __name__=='__main__':
    configure()
    if sys.argv[1:]==['--launch']:runner.launch(__file__)
    elif sys.argv[1:]:raise SystemExit('Only --launch is accepted')
    else:runner.main()
