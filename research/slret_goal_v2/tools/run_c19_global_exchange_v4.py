"""One C19 TRAIN512 pilot; no dependent experiment selected automatically."""
import json
from pathlib import Path
import sys
import run_c16_control_v4 as runner

RUN = 'seds-global-exchange-cross-001'
REFERENCE = 'seds-signrep-control-offload-001'


def validate_mode(result, mode='cross'):
    base = runner.BASE
    run_id = result.get('run_id')
    if result.get('status') != 'completed' or not run_id or not (base/run_id/'last.pt').is_file():
        raise ValueError('Missing successful result or last checkpoint')
    anchor = json.loads((base/REFERENCE/'run.json').read_text())
    config = result['config']
    expected = dict(native_subset=True,global_exchange=mode,batch_size=32,
                    signrep=None,fusion_objective='native',activation_offload=True)
    if any(config.get(k)!=v for k,v in expected.items()) or result['criterion']!='native':
        raise ValueError('Wrong C19 recipe/objective')
    for key,value in anchor['config'].items():
        if key not in ('run_id','signrep','artifact_cap_gib') and config[key]!=value:
            raise ValueError('Control recipe mismatch: '+key)
    for key in ('checkpoint_sha256_inherited','inherited_asset_digests','batch_order_sha256'):
        if result[key]!=anchor[key]:raise ValueError('Asset/order mismatch: '+key)
    for key in ('train_ids','dev_ids'):
        if result['subset_data'][key]!=anchor['signrep_data'][key]:raise ValueError('ID mismatch')
    if result['test_loaded'] or result['subset_data']['teacher_features_loaded']:
        raise ValueError('Unexpected TEST/teacher inputs')
    gates = ('masked_update_checks_passed','delta_roundtrip_passed',
             'global_exchange_initial_identity_passed','global_exchange_outputs_updated',
             'global_exchange_all_gradients_passed','global_exchange_trained_delta_roundtrip_passed')
    if not all(result.get(k) for k in gates):raise ValueError('Missing actual C19 gate')
    if set(result['evaluations'])!={'0','80','160'} and not result.get('stopping_reason'):
        raise ValueError('Missing scheduled fullDEV evaluation')
    for stream in ('fusion','pose','rgb'):
        for direction in ('T2V','V2T'):
            for metric in ('R1','R5','R10'):
                if abs(result['evaluations']['0'][stream][direction][metric]-
                       anchor['evaluations']['0'][stream][direction][metric])>1e-10:
                    raise ValueError('Initial recall mismatch')
    rates = {g['adaptation_group']:g['lr'] for g in result['actual_optimizer_groups']}
    if rates != dict(encoder=1e-6,fusion=1e-5,global_exchange=1e-4):
        raise ValueError('Unexpected optimizer rates')
    selected = Path(result['selection']['checkpoint'])
    expected_sha = result['selection'].get('checkpoint_sha256',result['checkpoint_sha256_inherited'])
    if not selected.is_file() or runner.sha(selected)!=expected_sha:
        raise ValueError('Selected checkpoint missing or changed')
    return result['selection']['mean_R1']-anchor['selection']['mean_R1']


def validate(result):
    return validate_mode(result,'cross')


def configure():
    runner.RUN = RUN
    runner.JOB = 'v4-c19-global-exchange-001'
    runner.OUT = runner.ROOT/'artifacts/slret_goal/jobs'/runner.JOB
    runner.MODE = 'global_exchange'
    runner.REFERENCE = REFERENCE
    runner.TOTAL_UNITS = 160
    runner.HARD_SECONDS = 2400
    runner.OUTER_SECONDS = 2450
    runner.ESTIMATED_SECONDS = 1700
    runner.GPU_QUERY_TIMEOUT_SECONDS = 20
    runner.COMMAND_OVERRIDE = [sys.executable,'-u','-m','torch.distributed.run','--standalone',
        '--nproc_per_node=1',str(runner.ROOT/'research/slret_goal_v2/tools/train_seds_extended.py'),
        '--run-id',RUN,'--native-subset','--global-exchange','cross',
        '--fusion-objective','native','--batch-size','32','--masked-pose','control',
        '--policy','fusion','--aux-weight','1','--seed','42','--epochs','10',
        '--lr','1e-5','--sign-lr','1e-6','--early-stop-drop-pp','2',
        '--activation-offload','--allocator-cap-gib','16','--artifact-cap-gib','42']
    runner.validate = validate


if __name__=='__main__':
    configure()
    if sys.argv[1:]==['--launch']:runner.launch(__file__)
    elif sys.argv[1:]:raise SystemExit('Only --launch is accepted')
    else:runner.main()
