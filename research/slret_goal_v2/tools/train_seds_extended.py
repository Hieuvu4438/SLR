"""Extended V2 runner, versioned while C03 uses the previous entrypoint.

No TEST construction, extraction, corpus rehash or baseline replay. Existing
Reuse release scores for loss-only changes; new parameterizations check step0.
"""
import argparse
from contextlib import nullcontext
import resource
from functools import partial
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'shared'), str(ROOT/'research/slret_goal/tools')]
from extraction_resume import atomic_json
from inventory import sha
from seds_continue import rng_state, restore_rng
from seds_optimizer_precision import ensure_fp32_moments
from seds_runtime import eval_dataset, evaluate, model_inputs, native_kwargs, patch_pickle
from methods.seds_adaptation.objectives import fused_priority_loss
from methods.seds_adaptation.train_policies import configure_trainable_stage, training_modes, configure_interaction_fusion
from methods.seds_adaptation.articulator_interaction import attach_articulator_interaction
from methods.seds_adaptation.peft_setup import configure_lora_fusion, configure_peft_learning_rates, enable_gcn_adaptation
from methods.seds_adaptation.smoothed_contrastive import SmoothedCrossEn
from methods.seds_adaptation.temporal_delta import attach_temporal_delta, configure_temporal_fusion
from methods.seds_adaptation.joint_exchange import attach_joint_exchange, configure_joint_fusion, configure_joint_learning_rates
from methods.seds_adaptation.pose3d_branch import attach_pose3d_branch
from methods.seds_adaptation.geometry_cache import GeometryDataset, geometry_collate
from methods.seds_adaptation.geometry_training import (configure_geometry_fusion,
    geometry_learning_rates, adaptation_state, check_adaptation_roundtrip)
from methods.seds_adaptation.masked_pose import attach_masked_pose, configure_masked_pose, masked_learning_rates
from methods.seds_adaptation.gcn_freeze import freeze_gcn_keep_optimizer
from methods.seds_adaptation.adaptive_graph import attach_adaptive_graph, graph_statistics
from methods.seds_adaptation.bone_features import attach_bone_features
from methods.seds_adaptation.joint_bilinear import attach_joint_bilinear
from methods.seds_adaptation.signrep_transfer import (SignRepDataset, signrep_collate,
    attach_signrep_transfer, transfer_learning_rates, transfer_state, load_transfer_state)
from methods.seds_adaptation.decoupled_fusion import attach_decoupled_fusion, registered_native_subset
from methods.seds_adaptation.global_exchange import (attach_global_exchange,
    configure_exchange_trainability, exchange_learning_rates)
from methods.seds_adaptation.dominant_stream_dropout import attach_annealed_rgb_dropout
from methods.seds_adaptation.phase_modulation import attach_phase_modulation, configure_phase_modulation
from methods.seds_adaptation.graph_bottleneck_adapter import (attach_graph_bottleneck_adapters,
    configure_graph_bottleneck_adapters, graph_adapter_learning_rates)
from c19_resume import read_recovery, restore_bertadam, inherit_progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--aux-weight', type=float, default=.25)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--policy', choices=['all','staged','lora','fusion'], default='all')
    parser.add_argument('--fusion-steps', type=int, default=222)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--sign-lr', type=float)
    parser.add_argument('--interaction', choices=['none','product','additive'], default='none')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--label-smoothing', type=float, default=0.)
    parser.add_argument('--lora-lr', type=float, default=1e-4)
    parser.add_argument('--lora-upper-blocks', type=int, default=1)
    parser.add_argument('--text-lora', action='store_true')
    parser.add_argument('--temporal-delta', action='store_true')
    parser.add_argument('--joint-exchange', choices=['cross', 'within'])
    parser.add_argument('--early-stop-drop-pp', type=float)
    parser.add_argument('--geometry', choices=['xy','xyz'])
    parser.add_argument('--geometry-representation',choices=['raw','canonical_motion'],default='raw')
    parser.add_argument('--geometry-control', action='store_true',
                        help='Matched TRAIN512 fusion-only control; geometry output stays frozen at zero')
    parser.add_argument('--geometry-freeze-fusion', action='store_true',
                        help='C09 refinement: train only geometric branch through frozen fusion')
    parser.add_argument('--geometry-cache', default=str(ROOT/'artifacts/slret_goal_v2/h4w-ph-center-pilot-002'))
    parser.add_argument('--masked-pose', choices=['reconstruct', 'control'])
    parser.add_argument('--gcn-lora', action='store_true')
    parser.add_argument('--clean-fusion-control', action='store_true')
    parser.add_argument('--gcn-clip-temporal', action='store_true')
    parser.add_argument('--gcn-long-horizon', action='store_true',
                        help='Registered clean GCN refinement: three epochs, no new module')
    parser.add_argument('--gcn-freeze-after',type=int,choices=[2,222])
    parser.add_argument('--adaptive-graph', action='store_true')
    parser.add_argument('--graph-lr', type=float, choices=[1e-4,1e-5], default=1e-4)
    parser.add_argument('--bone-features', action='store_true')
    parser.add_argument('--bone-lr', type=float, choices=[1e-5,1e-4], default=1e-5)
    parser.add_argument('--joint-bilinear', action='store_true')
    parser.add_argument('--centered-bilinear', action='store_true')
    parser.add_argument('--signrep', choices=['transfer','control'])
    parser.add_argument('--native-subset',action='store_true',help='Reuse matched TRAIN512 IDs without external teacher features')
    parser.add_argument('--global-exchange',choices=['cross','within','pose_to_rgb'])
    parser.add_argument('--rgb-moddrop-start',type=float,choices=[0.,.2],default=0.,
                        help='C20 train-only RGB stream dropout, linearly annealed to zero')
    parser.add_argument('--temporal-modulation',action='store_true')
    parser.add_argument('--temporal-modulation-lr',type=float,choices=[1e-4],default=1e-4)
    parser.add_argument('--graph-bottleneck-adapter',action='store_true')
    parser.add_argument('--resume-from',type=str)
    parser.add_argument('--fusion-objective',choices=['native','dcl'],default='native')
    parser.add_argument('--dcl-weight',type=float,choices=[1.,.05],default=1.)
    parser.add_argument('--signrep-loss',choices=['pointwise','relational'],default='pointwise')
    parser.add_argument('--signrep-weight',type=float,choices=[.1,1.],default=.1,
                        help='Registered C16-R2 changes only relational auxiliary strength')
    parser.add_argument('--artifact-cap-gib',type=int,choices=[36,38,40,42],default=36)
    parser.add_argument('--activation-offload', action='store_true')
    parser.add_argument('--allocator-cap-gib',type=float)
    parser.add_argument('--signrep-cache', default=str(ROOT/'artifacts/slret_goal_v2/signrep-native-window-pilot-001'))
    cli = parser.parse_args()
    subset_pilot = bool(cli.signrep or cli.native_subset)
    assert not cli.resume_from or (cli.global_exchange == 'cross' and not cli.smoke)
    assert not cli.global_exchange or (cli.native_subset and cli.batch_size == 32
        and cli.fusion_objective == 'native' and cli.activation_offload)
    assert not cli.rgb_moddrop_start or (cli.gcn_long_horizon and cli.masked_pose == 'control'
        and cli.policy == 'fusion' and cli.batch_size == 32 and cli.epochs == 3
        and cli.seed == 42 and not cli.native_subset and not cli.clean_fusion_control
        and not any((cli.global_exchange,cli.gcn_freeze_after,cli.adaptive_graph,
                     cli.bone_features,cli.joint_bilinear,cli.gcn_clip_temporal)))
    assert not cli.temporal_modulation or (cli.masked_pose == 'control'
        and cli.policy == 'fusion' and cli.batch_size == 32 and cli.epochs == 1
        and cli.seed == 42 and cli.aux_weight == 1 and not any((cli.gcn_long_horizon,
        cli.gcn_clip_temporal,cli.clean_fusion_control,cli.adaptive_graph,
        cli.bone_features,cli.joint_bilinear,cli.global_exchange,cli.rgb_moddrop_start)))
    assert not cli.graph_bottleneck_adapter or (cli.masked_pose == 'control'
        and cli.policy == 'fusion' and cli.batch_size == 32 and cli.epochs == 1
        and cli.seed == 42 and cli.aux_weight == 1 and not any((cli.gcn_long_horizon,
        cli.gcn_clip_temporal,cli.clean_fusion_control,cli.adaptive_graph,
        cli.bone_features,cli.joint_bilinear,cli.global_exchange,cli.rgb_moddrop_start,
        cli.temporal_modulation)))
    assert not cli.activation_offload or subset_pilot
    assert cli.fusion_objective == 'native' or cli.native_subset
    assert cli.dcl_weight == 1 or cli.fusion_objective == 'dcl'
    assert not cli.native_subset or (not cli.signrep and cli.masked_pose == 'control'
        and cli.epochs == 10 and cli.batch_size in (32,64) and cli.aux_weight == 1
        and not any((cli.gcn_long_horizon,cli.gcn_clip_temporal,cli.clean_fusion_control,
                     cli.adaptive_graph,cli.bone_features,cli.joint_bilinear)))
    assert not (cli.native_subset and cli.batch_size == 64) or cli.fusion_objective == 'native'
    assert cli.signrep_loss == 'pointwise' or cli.signrep == 'transfer'
    assert cli.signrep_weight == .1 or (cli.signrep == 'transfer' and cli.signrep_loss == 'relational')
    assert cli.allocator_cap_gib is None or 1 <= cli.allocator_cap_gib <= 16
    assert 1 <= cli.epochs <= 10 and 0 <= cli.aux_weight <= 2
    assert cli.batch_size in (32,64,128,256) and 0 <= cli.label_smoothing < 1
    assert cli.interaction == 'none' or cli.policy in ('all','fusion')
    assert 1 <= cli.lora_upper_blocks <= 12
    assert not cli.text_lora or cli.policy == 'lora'
    assert not cli.temporal_delta or (cli.policy == 'fusion' and cli.interaction == 'none' and not cli.label_smoothing)
    assert not cli.joint_exchange or (cli.policy == 'fusion' and cli.interaction == 'none' and not cli.label_smoothing and not cli.temporal_delta)
    assert cli.early_stop_drop_pp is None or 0 < cli.early_stop_drop_pp <= 100
    assert not cli.geometry or (cli.policy == 'fusion' and cli.interaction == 'none'
        and not cli.temporal_delta and not cli.joint_exchange and not cli.label_smoothing)
    assert cli.geometry_representation == 'raw' or (cli.geometry == 'xyz'
        and not cli.geometry_control and not cli.geometry_freeze_fusion
        and not cli.masked_pose and cli.batch_size == 32 and cli.epochs == 8
        and cli.seed == 42 and cli.aux_weight == 1)
    assert not cli.geometry_control or cli.geometry == 'xy'
    assert not cli.geometry_freeze_fusion or (cli.geometry and not cli.geometry_control)
    assert not cli.masked_pose or (cli.policy == 'fusion' and not cli.geometry
        and not cli.joint_exchange and not cli.temporal_delta and cli.interaction == 'none'
        and not cli.label_smoothing and cli.aux_weight == 1)
    assert not cli.gcn_lora or (cli.policy == 'lora' and not cli.masked_pose
        and not cli.geometry and not cli.joint_exchange and not cli.temporal_delta
        and cli.interaction == 'none' and not cli.label_smoothing and cli.aux_weight == 1
        and cli.lora_upper_blocks == 1 and not cli.text_lora)
    assert not cli.clean_fusion_control or cli.masked_pose == 'control'
    assert not cli.gcn_clip_temporal or (cli.masked_pose == 'control' and not cli.clean_fusion_control)
    assert not cli.gcn_long_horizon or (cli.masked_pose == 'control' and cli.epochs == 3
                                      and not cli.gcn_clip_temporal)
    assert not cli.gcn_freeze_after or (cli.gcn_long_horizon and not cli.clean_fusion_control
                                      and cli.gcn_freeze_after == (2 if cli.smoke else 222))
    assert not cli.adaptive_graph or (cli.gcn_long_horizon and not cli.clean_fusion_control
                                     and not cli.gcn_freeze_after)
    assert cli.adaptive_graph or cli.graph_lr == 1e-4
    assert not cli.bone_features or (cli.gcn_long_horizon and not cli.clean_fusion_control
                                    and not cli.adaptive_graph and not cli.gcn_freeze_after)
    assert cli.bone_features or cli.bone_lr == 1e-5
    assert not cli.joint_bilinear or (cli.gcn_long_horizon and not cli.clean_fusion_control
        and not cli.adaptive_graph and not cli.bone_features and not cli.gcn_freeze_after)
    assert not cli.centered_bilinear or cli.joint_bilinear
    assert not cli.signrep or (cli.masked_pose == 'control' and cli.epochs == 10
        and cli.batch_size == 32 and not any((cli.gcn_long_horizon,cli.gcn_clip_temporal,
        cli.clean_fusion_control,cli.adaptive_graph,cli.bone_features,cli.joint_bilinear)))
    out = ROOT/'artifacts/slret_goal_v2'/cli.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    ledger = ROOT/'research/slret_goal_v2/EXPERIMENTS.jsonl'
    candidate = 'C03' if cli.interaction != 'none' else ('C02' if cli.policy == 'staged' else 'C01')
    if cli.policy == 'lora':
        candidate = 'C04'
    elif cli.policy == 'fusion' and cli.interaction == 'none':
        candidate = 'C04_control_fusion'
    elif cli.label_smoothing:
        candidate = 'C05'
    if cli.temporal_delta:
        candidate = 'C06'
    if cli.joint_exchange:
        candidate = 'C07' if cli.joint_exchange == 'cross' else 'C07_control_within'
    if cli.geometry:
        candidate = 'C09_'+cli.geometry
        if cli.geometry_control:
            candidate = 'C09_control_fusion512'
        elif cli.geometry_freeze_fusion:
            candidate += '_frozen_fusion'
        if cli.geometry_representation == 'canonical_motion':
            candidate = 'C21_anatomical_canonical_3D_motion'
    if cli.masked_pose:
        candidate = 'C08' if cli.masked_pose == 'reconstruct' else 'C08_control_continuation'
    if cli.gcn_lora:
        candidate = 'C11_GCN_LoRA'
    if cli.clean_fusion_control:
        candidate = 'GCN_ablation_fusion_only'
    if cli.gcn_clip_temporal:
        candidate = 'C12_GCN_clip_temporal'
    if cli.gcn_long_horizon:
        candidate = 'GCN_R1_horizon3_control_fusion' if cli.clean_fusion_control else 'GCN_R1_horizon3'
    if cli.gcn_freeze_after:
        candidate = 'GCN_R2_freeze_after222'
    if cli.adaptive_graph:
        candidate = 'C13_adaptive_joint_graph'
        if cli.graph_lr == 1e-5:
            candidate = 'C13_R1_slower_graph'
    if cli.bone_features:
        candidate = 'C14_explicit_bone_features'
        if cli.bone_lr == 1e-4:
            candidate = 'C14_R1_faster_bone_projection'
    if cli.joint_bilinear:
        candidate = 'C15_joint_bilinear_pooling'
        if cli.centered_bilinear:
            candidate = 'C15_R1_within_part_covariance'
    if cli.signrep:
        candidate = 'C16_SignRep_'+cli.signrep
        if cli.signrep_loss == 'relational':
            candidate = 'C16_R1_SignRep_within_video_RKD_D'
            if cli.signrep_weight == 1.:
                candidate = 'C16_R2_SignRep_RKD_D_weight1'
    if cli.native_subset:
        candidate = 'C17_fused_DCL' if cli.fusion_objective == 'dcl' else 'C17_native_subset_control'
        if cli.dcl_weight == .05:
            candidate = 'C17_R1_fused_DCL_blend005'
        if cli.batch_size == 64:
            candidate = 'C18_native_larger_negative_batch64'
    if cli.global_exchange:
        candidate = 'C19_global_bottleneck_'+cli.global_exchange
    if cli.rgb_moddrop_start:
        candidate = 'C20_annealed_RGB_modality_dropout'
    if cli.temporal_modulation:
        candidate = 'C22_shared_phase_low_rank_modulation'
    if cli.graph_bottleneck_adapter:
        candidate = 'C23_frozen_hierarchical_GCN_bottleneck_adapters'
    report = dict(run_id=cli.run_id, candidate=candidate, status='running', pid=os.getpid(),
                  start_unix=started, command=sys.argv, config=vars(cli), steps=0,
                  selection_split='PH_adapted_DEV519' if not cli.smoke else 'none_smoke',
                  test_loaded=False, artifact_dir=str(out),
                  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [
                      Path(__file__), ROOT/'methods/seds_adaptation/objectives.py',
                      ROOT/'methods/seds_adaptation/train_policies.py',
                      ROOT/'methods/seds_adaptation/articulator_interaction.py',
                      ROOT/'methods/seds_adaptation/low_rank_attention.py',
                      ROOT/'methods/seds_adaptation/peft_setup.py',
                      ROOT/'methods/seds_adaptation/smoothed_contrastive.py',
                      ROOT/'methods/seds_adaptation/temporal_delta.py',
                      ROOT/'methods/seds_adaptation/joint_exchange.py',
                      ROOT/'methods/seds_adaptation/pose3d_branch.py',
                      ROOT/'methods/seds_adaptation/geometry_cache.py',
                      ROOT/'methods/seds_adaptation/geometry_training.py',
                      ROOT/'methods/seds_adaptation/masked_pose.py',
                      ROOT/'methods/seds_adaptation/gcn_freeze.py',
                      ROOT/'methods/seds_adaptation/adaptive_graph.py',
                      ROOT/'methods/seds_adaptation/bone_features.py',
                      ROOT/'methods/seds_adaptation/joint_bilinear.py',
                      ROOT/'methods/seds_adaptation/signrep_transfer.py',
                      ROOT/'methods/seds_adaptation/decoupled_fusion.py',
                      ROOT/'methods/seds_adaptation/global_exchange.py',
                      ROOT/'methods/seds_adaptation/dominant_stream_dropout.py',
                      ROOT/'methods/seds_adaptation/phase_modulation.py',
                      ROOT/'methods/seds_adaptation/graph_bottleneck_adapter.py',
                      ROOT/'research/slret_goal_v2/tools/c19_resume.py',
                      ROOT/'third_party/SEDS/modules/modeling_gcn.py',
                      ROOT/'third_party/SEDS/modules/modeling_graph.py',
                      ROOT/'third_party/SEDS/main_task_retrieval.py',
                      ROOT/'research/slret_goal/tools/seds_runtime.py',
                      ROOT/'research/slret_goal/tools/seds_optimizer_precision.py']})

    def record(terminal=False):
        report['wall_seconds'] = time.time()-started
        atomic_json(out/'run.json', report)
        if terminal or report['steps'] == 0:
            with ledger.open('a') as f:
                f.write(json.dumps(report, allow_nan=False)+'\n')

    record()
    try:
        recovery = read_recovery(ROOT,cli.resume_from,vars(cli)) if cli.resume_from else None
        for relative in report['code_sha256']:
            destination = out/'source'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/relative,destination)
        reserve = 15*1024**3
        # C13 measured total ~2.1GiB including best/last/scores; 3GiB covers
        # the observed output plus temporary checkpoint replacement headroom.
        planned = 128*1024**2 if cli.smoke else (3 if cli.adaptive_graph or cli.bone_features or cli.joint_bilinear else
            4 if cli.policy == 'all' or cli.masked_pose or cli.gcn_lora else 2)*1024**3
        if subset_pilot and not cli.smoke:
            # Measured transfer best162MB + last465MB + atomic replacement465MB,
            # with >240MB additional headroom. No change to retained outputs.
            planned = int((1.25 if cli.activation_offload else 1)*1024**3)
        used = sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())
        assert used+planned <= cli.artifact_cap_gib*1024**3, 'Campaign artifact allowance'
        assert shutil.disk_usage(out).free-planned >= reserve, 'Free disk reserve'
        torch.set_num_threads(4)
        base = ROOT/'third_party/SEDS'
        sys.path.insert(0, str(base))
        os.chdir(base)
        import main_task_retrieval as native
        from dataloaders.dataloader_ph_retrieval_pose import ph_DataLoader_pose, ph_pose_collate_fn
        from dataloaders.dataloader_ph_retrieval_train_pose import ph_DataLoader_train_pose, ph_train_pose_collate_fn
        patch_pickle(ph_DataLoader_pose, ph_DataLoader_train_pose)
        reference_root = ROOT/'artifacts/slret_goal/seds-adapted-dev-eval-002'
        reference = json.loads((reference_root/'run.json').read_text())
        previous = json.loads((ROOT/'artifacts/slret_goal/seds-moment-control-001/run.json').read_text())
        args = argparse.Namespace(**previous['config'])
        args.output_dir, args.seed, args.epochs = str(out), cli.seed, cli.epochs
        args.batch_size = cli.batch_size
        if cli.masked_pose:
            assert args.crop_size == 256, 'C08 target scale requires native 256px crops'
        if cli.lr is not None:
            args.lr = cli.lr
        if cli.sign_lr is not None:
            args.sign_lr = cli.sign_lr
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        assert not args.freeze_exfusion and not args.rgb_pose_kl
        train_root = ROOT/'artifacts/slret_goal/seds-adapted-train-001'
        dev_root = ROOT/'artifacts/slret_goal/seds-adapted-dev-001'
        for root, split, count in [(train_root,'train',7096), (dev_root,'dev',519)]:
            manifest = json.loads((root/'run.json').read_text())
            assert manifest['status'] == 'completed' and manifest['completed'] == count
            assert (root/'labels'/f'{split}.pkl').is_file()
        args = native.set_seed_logger(args)
        device, n_gpu = native.init_device(args, args.local_rank)
        if cli.allocator_cap_gib is not None:
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(cli.allocator_cap_gib*1024**3/total,device)
            report['allocator_cap_bytes'] = int(cli.allocator_cap_gib*1024**3)
        assert n_gpu == 1 and torch.distributed.get_world_size() == 1
        tokenizer = native.ClipTokenizer()
        train = ph_DataLoader_train_pose(**native_kwargs(args,tokenizer,train_root,'train'))
        dev, dev_ids = eval_dataset(ph_DataLoader_pose,args,tokenizer,dev_root)
        with (train_root/'labels/train.pkl').open('rb') as f:
            ids = list(pickle.load(f))
        assert len(ids) == 7096 and not set(ids)&set(dev_ids)
        train_collate,dev_collate=ph_train_pose_collate_fn,ph_pose_collate_fn
        if cli.native_subset:
            subset_reference = ROOT/'artifacts/slret_goal_v2/seds-signrep-control-offload-001/run.json'
            train,ids = registered_native_subset(train,ids,subset_reference)
            report['subset_data'] = dict(train_ids=ids,dev_ids=dev_ids,
                reference_report_sha256=sha(subset_reference),teacher_features_loaded=False)
        if cli.signrep:
            train = SignRepDataset(train,ids,cli.signrep_cache)
            ids = train.ids
            assert len(ids) == 512 and len(dev) == 519
            train_collate = partial(signrep_collate,ph_train_pose_collate_fn)
            report['signrep_data'] = dict(train_ids=ids,dev_ids=dev_ids,
                cache=cli.signrep_cache,report_sha256=sha(Path(cli.signrep_cache)/'run.json'),
                plan_sha256=sha(Path(cli.signrep_cache)/'plan.json'),dev_teacher_used=False)
        if cli.geometry:
            train=GeometryDataset(train,ids,cli.geometry_cache,'train')
            dev=GeometryDataset(dev,dev_ids,cli.geometry_cache,'dev')
            ids=train.ids
            assert len(ids)==512 and len(dev)==519
            train_collate=partial(geometry_collate,ph_train_pose_collate_fn)
            dev_collate=partial(geometry_collate,ph_pose_collate_fn)
            report['geometry_data']=dict(cache=cli.geometry_cache,train_ids=ids,dev_ids=dev_ids,
                report_sha256=sha(Path(cli.geometry_cache)/'run.json'),
                plan_sha256=sha(Path(cli.geometry_cache)/'plan.json'))
        loader = torch.utils.data.DataLoader(dev,batch_size=32,shuffle=False,num_workers=0,
                                             collate_fn=dev_collate)
        model = native.init_model(args,device)
        active = None
        if cli.policy == 'lora':
            with torch.random.fork_rng():
                active = configure_lora_fusion(model,rank=8,alpha=8,
                    upper_blocks=cli.lora_upper_blocks,text_lora=cli.text_lora)
            if cli.gcn_lora:
                enable_gcn_adaptation(model)
        if cli.label_smoothing:
            model.loss_fct = SmoothedCrossEn(cli.label_smoothing)
        if cli.temporal_delta:
            with torch.random.fork_rng():
                attach_temporal_delta(model)
        if cli.joint_exchange:
            with torch.random.fork_rng():
                attach_joint_exchange(model.signbert, mode=cli.joint_exchange)
        if cli.geometry:
            with torch.random.fork_rng():
                geometry_branch = attach_pose3d_branch(model,use_depth=cli.geometry=='xyz',clip_temporal=True,
                                                       representation=cli.geometry_representation)
            assert torch.count_nonzero(geometry_branch.output.weight).item() == 0
            report['geometry_branch_parameters'] = sum(p.numel() for p in geometry_branch.parameters())
            report['geometry_representation'] = cli.geometry_representation
            report['geometry_zero_output_weight_passed'] = True
        if cli.masked_pose:
            with torch.random.fork_rng():
                attach_masked_pose(model, control=cli.masked_pose=='control', seed=cli.seed)
        if cli.signrep:
            with torch.random.fork_rng():
                attach_signrep_transfer(model,control=cli.signrep=='control',loss_kind=cli.signrep_loss)
        if cli.fusion_objective == 'dcl':
            attach_decoupled_fusion(model,weight=cli.dcl_weight)
        if cli.global_exchange:
            with torch.random.fork_rng():
                module = attach_global_exchange(model,mode=cli.global_exchange)
            report['global_exchange_parameters'] = sum(p.numel() for p in module.parameters())
            report['global_exchange_trainable_parameters'] = sum(
                p.numel() for p in module.parameters() if p.requires_grad)
        if cli.rgb_moddrop_start:
            module = attach_annealed_rgb_dropout(model,cli.rgb_moddrop_start,cli.seed)
            module.eval()
            probe = torch.zeros(2,3,4,device=device)
            assert module(probe) is probe and module.eval_identity_passed
            module.train(model.fusion.training)
            report.update(rgb_moddrop_parameters=0,rgb_moddrop_eval_identity_passed=True,
                          rgb_moddrop_rng='private_seeded_generator')
        if cli.temporal_modulation:
            module = attach_phase_modulation(model,rank=3,anchors=32)
            assert all(torch.count_nonzero(p).item()==0 for values in
                       (module.scale,module.shift) for p in values.values())
            report.update(temporal_modulation_parameters=sum(p.numel() for p in module.parameters()),
                          temporal_modulation_rank=3,temporal_modulation_anchors=32,
                          temporal_modulation_zero_identity_passed=True)
        if cli.graph_bottleneck_adapter:
            module = attach_graph_bottleneck_adapters(model,rank=16)
            assert all(torch.count_nonzero(adapter.output.weight).item()==0
                       for adapter in module.values())
            report.update(graph_bottleneck_adapter_parameters=sum(p.numel() for p in module.parameters()),
                          graph_bottleneck_adapter_rank=16,
                          graph_bottleneck_adapter_count=len(module),
                          graph_bottleneck_adapter_zero_identity_passed=True)
        if cli.adaptive_graph:
            report['adaptive_graph_parameters'] = attach_adaptive_graph(model)
        if cli.bone_features:
            with torch.random.fork_rng():
                report['bone_parents'] = attach_bone_features(model)
        if cli.joint_bilinear:
            with torch.random.fork_rng():
                report['joint_bilinear_layout'] = attach_joint_bilinear(model,centered=cli.centered_bilinear)
        if cli.interaction != 'none':
            with torch.random.fork_rng():
                attach_articulator_interaction(model.signbert,512,32,cli.interaction)
        report.update(native_config=vars(args), hardware=torch.cuda.get_device_name(),
                      torch=torch.__version__, checkpoint=args.init_model,
                      checkpoint_sha256_inherited=previous['checkpoint_sha256'],
                      inherited_asset_digests={k:previous[k] for k in ['train_assets_digest','dev_assets_digest']},
                      data_validation='reuse V1 validated features; no full corpus rehash',
                      reference_report_sha256=sha(reference_root/'run.json'))
        initial = previous['evaluations']['0']
        initial_r1 = {d:initial['fusion'][d]['R1'] for d in ['T2V','V2T']}
        reference_mean = sum(initial_r1.values())/2
        incumbent_mean = previous['selection']['mean_R1']
        if cli.geometry or cli.masked_pose:
            incumbent_mean = json.loads((ROOT/'artifacts/slret_goal_v2/seds-lora-001/selection.json').read_text())['mean_R1']
        if cli.gcn_lora or (cli.masked_pose == 'control' and cli.run_id.startswith('seds-gcn-clean-seed')):
            control_path = ROOT/'artifacts/slret_goal_v2/seds-masked-control-002/run.json'
            control_report = json.loads(control_path.read_text())
            assert control_report['status'] == 'completed'
            incumbent_mean = control_report['selection']['mean_R1']
            report['matched_control_report_sha256'] = sha(control_path)
        if cli.clean_fusion_control or cli.gcn_clip_temporal or cli.gcn_long_horizon:
            incumbent_path = ROOT/'artifacts/slret_goal_v2/seds-gcn-clean-seed1337-001/run.json'
            if cli.gcn_long_horizon and (cli.clean_fusion_control or cli.seed != 42):
                incumbent_path = ROOT/'artifacts/slret_goal_v2/seds-gcn-horizon3-001/run.json'
            if cli.gcn_freeze_after or cli.adaptive_graph or cli.bone_features or cli.joint_bilinear:
                incumbent_path = ROOT/'artifacts/slret_goal_v2/seds-gcn-horizon3-seed1337-001/run.json'
            incumbent_mean = json.loads(incumbent_path.read_text())['selection']['mean_R1']
            report['incumbent_report_sha256'] = sha(incumbent_path)
        if subset_pilot:
            incumbent_path = ROOT/'artifacts/slret_goal_v2/seds-gcn-horizon3-seed1337-001/run.json'
            incumbent_mean = json.loads(incumbent_path.read_text())['selection']['mean_R1']
            report['incumbent_report_sha256'] = sha(incumbent_path)
        selection = dict(step=0,mean_R1=reference_mean,R1=initial_r1,checkpoint=args.init_model)
        report.update(reference_mean_R1=reference_mean,incumbent_mean_R1=incumbent_mean,
                      evaluations={'0':initial},step0='reused unchanged inference checkpoint; loss-only intervention')
        if recovery:
            _, parent_report, selection, _ = recovery
            initial = parent_report['evaluations']['0']
            initial_r1 = {d:initial['fusion'][d]['R1'] for d in ('T2V','V2T')}
            reference_mean = sum(initial_r1.values())/2
            inherit_progress(report,parent_report,recovery[0]['next_batch_index'])
        elif subset_pilot and cli.activation_offload and cli.smoke:
            # Offload changes saved-tensor storage only during training. Smoke
            # checks the failed TRAIN batch, not another full DEV score replay.
            report.update(evaluations={},step0='training-memory smoke; no DEV evaluation',
                          reference_metrics_inherited=True)
        elif subset_pilot:
            if cli.signrep:
                proof_path = ROOT/'artifacts/slret_goal_v2/signrep-hook-check-002/run.json'
                proof = json.loads(proof_path.read_text())
                assert proof['status'] == 'completed' and proof['hook_exact']
                assert all(x['exact'] for values in proof['comparisons'].values() for x in values)
                report['signrep_hook_proof_sha256'] = sha(proof_path)
            saved = rng_state()
            zero = evaluate(native,args,model,loader,device,dev_ids,out/'eval_step0000')
            restore_rng(saved)
            report['historical_initial_score_max_differences'] = {}
            for stream in ('fusion','pose','rgb'):
                actual = np.load(out/'eval_step0000'/f'{stream}_video_x_text.npy')
                anchor = np.load(reference_root/f'{stream}_video_x_text.npy')
                report['historical_initial_score_max_differences'][stream] = float(np.max(np.abs(actual-anchor)))
                for direction in ('T2V','V2T'):
                    for key in ('R1','R5','R10'):
                        assert abs(zero[stream][direction][key]-initial[stream][direction][key]) < 1e-5
            initial = zero
            if cli.global_exchange:
                assert model.fusion.global_exchange.initial_identity_passed
                report['global_exchange_initial_identity_passed'] = True
            initial_r1 = {d:zero['fusion'][d]['R1'] for d in ('T2V','V2T')}
            reference_mean = sum(initial_r1.values())/2
            selection = dict(step=0,mean_R1=reference_mean,R1=initial_r1,checkpoint=args.init_model)
            report.update(evaluations={'0':zero},reference_mean_R1=reference_mean,
                step0='fresh fullDEV initialization; historical recall matched; '+('same-batch SignRep hook proof' if cli.signrep else 'identity C19 inputs' if cli.global_exchange else 'native inference; DCL training-only'),
                replay_caveat='Cross-process pose score differences remain unresolved; no score parity claim')
        elif (cli.interaction != 'none' or cli.policy == 'lora' or cli.temporal_delta
              or cli.joint_exchange or cli.geometry or cli.temporal_modulation
              or cli.graph_bottleneck_adapter or (cli.masked_pose and cli.smoke)):
            saved = rng_state()
            zero = evaluate(native,args,model,loader,device,dev_ids,out/'eval_step0000')
            report['initial_score_max_differences'] = {}
            report['initial_metric_differences'] = {}
            for stream in ('fusion','pose','rgb'):
                actual = np.load(out/'eval_step0000'/f'{stream}_video_x_text.npy')
                anchor = np.load(reference_root/f'{stream}_video_x_text.npy')
                difference = float(np.max(np.abs(actual-anchor)))
                report['initial_score_max_differences'][stream] = difference
                assert actual.shape == anchor.shape and np.isfinite(actual).all()
                # The canonical branch is structurally zero-initialized, but
                # its extra CUDA work can change otherwise equivalent floating
                # scheduling. Require every reported retrieval metric to match
                # instead of rejecting on immaterial score-array roundoff.
                relaxed_identity = (cli.geometry_representation == 'canonical_motion'
                                    or cli.temporal_modulation or cli.graph_bottleneck_adapter)
                if not relaxed_identity:
                    assert difference <= 1e-4
                report['initial_metric_differences'][stream] = {}
                for direction in ('T2V','V2T'):
                    report['initial_metric_differences'][stream][direction] = {}
                    for metric in ('R1','R5','R10','MedianR','MeanR'):
                        metric_difference = abs(zero[stream][direction][metric]-initial[stream][direction][metric])
                        report['initial_metric_differences'][stream][direction][metric] = metric_difference
                        if relaxed_identity and metric == 'MeanR':
                            # One rank position among 519 examples is the maximum
                            # admitted effect of backend score scheduling; primary
                            # R@K and MedianR below must remain exactly unchanged.
                            assert metric_difference <= 1/len(dev_ids)+1e-10
                        else:
                            assert metric_difference < 1e-10
            restore_rng(saved)
            report['step0'] = ('zero-initialized parameterization fullDEV metric parity passed; '
                'finite score differences recorded' if relaxed_identity
                else 'zero-initialized parameterization fullDEV score parity passed')
        elif cli.adaptive_graph or cli.bone_features or cli.joint_bilinear:
            proof_name = ('seds-adaptive-graph-lr1e5-smoke-001' if cli.graph_lr == 1e-5
                          else 'seds-adaptive-graph-smoke-001')
            if cli.bone_features:
                proof_name = ('seds-bone-features-lr1e4-smoke-001' if cli.bone_lr == 1e-4
                              else 'seds-bone-features-smoke-001')
            if cli.joint_bilinear:
                proof_name = ('seds-joint-covariance-smoke-001' if cli.centered_bilinear
                              else 'seds-joint-bilinear-smoke-002')
            proof_path = ROOT/'artifacts/slret_goal_v2'/proof_name/'run.json'
            proof = json.loads(proof_path.read_text())
            update_key = ('joint_bilinear_updated' if cli.joint_bilinear else
                          'bone_features_updated' if cli.bone_features else 'adaptive_graph_updated')
            assert proof['status'] == 'completed' and proof[update_key]
            assert proof['step0'] == 'zero-initialized parameterization fullDEV score parity passed'
            for key in ('code_sha256','checkpoint_sha256_inherited','inherited_asset_digests'):
                assert proof[key] == report[key], 'C13 initial parity source/base/data changed'
            for key in vars(cli):
                if key not in ('run_id','smoke'):
                    assert proof['config'][key] == vars(cli)[key], 'C13 smoke recipe changed'
            report['step0'] = 'inherited C14 smoke fullDEV zero-init parity' if cli.bone_features else 'inherited C13 smoke fullDEV zero-init parity'
            if cli.joint_bilinear:
                report['step0'] = 'inherited C15 smoke fullDEV zero-init parity'
            report['step0_proof_sha256'] = sha(proof_path)
        elif cli.policy in ('staged','fusion'):
            report['step0'] = 'inherited unchanged checkpoint/scorer; only trainable subset changes'
        if cli.rgb_moddrop_start:
            report['step0'] = 'inference unchanged; C20 is training-only and disabled in eval'
        atomic_json(out/'selection.json',selection)
        batches = []
        steps_per_epoch = (len(ids)+cli.batch_size-1)//cli.batch_size
        for epoch in range(cli.epochs):
            order = torch.randperm(len(ids),generator=torch.Generator().manual_seed(cli.seed+epoch)).tolist()
            batches.extend([order[i:i+cli.batch_size] for i in range(0,len(order),cli.batch_size)])
        schedule_steps = len(batches)
        if cli.policy == 'staged':
            assert 1 < cli.fusion_steps < schedule_steps
        if cli.smoke:
            batches = [list(range(cli.batch_size)) for _ in range(
                6 if cli.policy == 'staged' else 3 if cli.joint_bilinear or cli.global_exchange else 2)]
            if cli.activation_offload and cli.signrep:
                # Reproduce the inputs that failed at step23, not just batch0.
                failed = ROOT/'artifacts/slret_goal_v2/seds-signrep-transfer-001/batch_indices.json'
                failed_batch = json.loads(failed.read_text())[22]
                batches = [failed_batch,failed_batch]
                report['offload_smoke_batch'] = dict(source=str(failed),sha256=sha(failed),index=22)
            if cli.gcn_freeze_after:
                batches = [list(range(cli.batch_size)) for _ in range(4)]
            if cli.policy == 'staged':
                assert cli.fusion_steps == 2, 'Stage smoke tests transition after two steps'
                schedule_steps = 6
        atomic_json(out/'batch_indices.json',batches)
        report['batch_order_sha256'] = sha(out/'batch_indices.json')
        if cli.policy in ('staged','fusion'):
            active = (configure_interaction_fusion(model) if cli.interaction != 'none'
                      else configure_trainable_stage(model,'fusion'))
        if cli.temporal_delta:
            active = configure_temporal_fusion(model)
        if cli.joint_exchange:
            active = configure_joint_fusion(model)
        if cli.geometry:
            active = configure_geometry_fusion(model,fusion_only=cli.geometry_control,
                                               freeze_fusion=cli.geometry_freeze_fusion)
        if cli.masked_pose:
            active = configure_masked_pose(model, control=cli.masked_pose=='control',fusion_only=cli.clean_fusion_control,
                                           clip_temporal=cli.gcn_clip_temporal)
        if cli.temporal_modulation:
            active = configure_phase_modulation(model,active)
        if cli.graph_bottleneck_adapter:
            active = configure_graph_bottleneck_adapters(model,active)
        if cli.global_exchange:
            # Generic fusion configuration above enables every child parameter;
            # restore the registered C19 direction before optimizer construction.
            report['global_exchange_trainable_parameters'] = configure_exchange_trainability(model)
        if cli.signrep == 'transfer':
            model.signrep_transfer.requires_grad_(True)
            active.append(model.signrep_transfer)
        first_schedule = cli.fusion_steps if cli.policy == 'staged' else schedule_steps
        optimizer, scheduler, wrapped = native.prep_optimizer(args,model,first_schedule,device,1,
                                                              args.local_rank,coef_lr=args.coef_lr)
        if cli.policy == 'lora':
            report['actual_optimizer_groups'] = configure_peft_learning_rates(
                optimizer,model,cli.lora_lr,args.lr if cli.gcn_lora else args.sign_lr,
                encoder_lr=args.sign_lr if cli.gcn_lora else None)
        if cli.joint_exchange:
            report['actual_optimizer_groups'] = configure_joint_learning_rates(
                optimizer, model, joint_lr=args.sign_lr, fusion_lr=args.lr)
            print(json.dumps(dict(event='optimizer_groups', groups=report['actual_optimizer_groups'])),flush=True)
        if cli.masked_pose and not cli.signrep:
            report['actual_optimizer_groups'] = (graph_adapter_learning_rates(optimizer,model)
                if cli.graph_bottleneck_adapter else masked_learning_rates(optimizer, model, control=cli.masked_pose=='control',fusion_only=cli.clean_fusion_control,
                                                                      clip_temporal=cli.gcn_clip_temporal,
                                                                      adaptive_graph=cli.adaptive_graph, graph_lr=cli.graph_lr,
                                                                      bone_features=cli.bone_features, bone_lr=cli.bone_lr,
                                                                      joint_bilinear=cli.joint_bilinear,
                                                                      temporal_modulation=cli.temporal_modulation,
                                                                      temporal_modulation_lr=cli.temporal_modulation_lr))
            report['masked_recipe'] = dict(ratio=.2, weight=.05, target='XY/256-.5',
                encoder_mode='eval_with_grad', control=cli.masked_pose=='control')
        if subset_pilot:
            report['actual_optimizer_groups'] = transfer_learning_rates(optimizer,model,cli.signrep!='transfer')
            if cli.global_exchange:
                report['actual_optimizer_groups'] = exchange_learning_rates(optimizer,model)
            report['checkpoint_format'] = ('GCN_fusion_transfer_delta_requires_native_release' if cli.signrep
                                           else 'GCN_fusion_delta_requires_native_release')
            report['base_checkpoint'] = dict(path=args.init_model,sha256=previous['checkpoint_sha256'])
            state = transfer_state(model)
            load_transfer_state(model,state)
            assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in state.items())
            report['delta_roundtrip_passed'] = True
            del state
        if cli.geometry:
            report['actual_optimizer_groups']=geometry_learning_rates(
                optimizer,model,args.sign_lr,args.lr,fusion_only=cli.geometry_control,
                freeze_fusion=cli.geometry_freeze_fusion)
            report['initial_adaptation_norm']=float(model.pose3d_branch.output.weight.norm())
            report['checkpoint_format']='adaptation_only_requires_base_checkpoint'
            report['base_checkpoint']=dict(path=args.init_model,sha256=previous['checkpoint_sha256'])
            print(json.dumps(dict(event='optimizer_groups',groups=report['actual_optimizer_groups'])),flush=True)
        torch.cuda.reset_peak_memory_stats()
        tracked = dict(model.named_parameters())
        frozen_buffers = {n:b.detach().clone() for n,b in model.named_buffers()
                          if not n.startswith(('pose3d_branch.',) if cli.geometry_freeze_fusion
                                              else ('fusion.','pose3d_branch.'))} if cli.geometry or cli.masked_pose or cli.gcn_lora else {}
        fusion_anchor = {k:v.detach().cpu().clone() for k,v in model.fusion.state_dict().items()} if cli.geometry_freeze_fusion else {}
        evaluation_steps = {steps_per_epoch//2,*range(steps_per_epoch,schedule_steps+1,steps_per_epoch),schedule_steps}
        if cli.geometry:
            evaluation_steps={schedule_steps//2,schedule_steps}
        if subset_pilot:
            evaluation_steps={schedule_steps//2,schedule_steps}
        frozen_gcn_anchor = None

        def save_offload_resume():
            if cli.activation_offload and not cli.smoke:
                torch.save(dict(model=transfer_state(model),optimizer=optimizer.state_dict(),scheduler=None,
                    checkpoint_format=report['checkpoint_format'],base_checkpoint=report['base_checkpoint'],
                    rng=rng_state(),batches=batches,next_batch_index=report['steps'],config=vars(args),
                    adaptation=vars(cli),code_sha256=report['code_sha256'],
                    optimizer_precision='fp32_moments_native_parameters'),out/'last.tmp')
                os.replace(out/'last.tmp',out/'last.pt')

        start_step = 0
        if recovery:
            checkpoint, parent_report, selection, previous_rows = recovery
            if checkpoint['batches'] != batches or checkpoint['base_checkpoint'] != report['base_checkpoint']:
                raise ValueError('Recovery batch/base mismatch')
            if parent_report['inherited_asset_digests'] != report['inherited_asset_digests'] or parent_report['subset_data'] != report['subset_data']:
                raise ValueError('Recovery data contract mismatch')
            start_step = checkpoint['next_batch_index']
            load_transfer_state(model,checkpoint['model'])
            assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in checkpoint['model'].items())
            restore_bertadam(optimizer,checkpoint['optimizer'],start_step)
            model.fusion.global_exchange._identity_checked = True
            report['resume']['model_optimizer_restored'] = True
            with (out/'train_steps.jsonl').open('x') as f:
                for row in previous_rows:
                    f.write(json.dumps(row,allow_nan=False)+'\n')
            report['resume']['inherited_train_rows'] = len(previous_rows)
            record()
            print(json.dumps(dict(event='resume_restored',**report['resume'])),flush=True)
            restore_rng(checkpoint['rng'])
            del checkpoint, recovery, previous_rows

        for step,indices in enumerate(batches[start_step:],start_step+1):
            tick = time.time()
            if cli.rgb_moddrop_start:
                model.fusion.rgb_moddrop.set_step(step,schedule_steps)
            if cli.gcn_freeze_after and step == cli.gcn_freeze_after+1:
                del wrapped
                transition=freeze_gcn_keep_optimizer(model,optimizer)
                active=[model.fusion]
                frozen_gcn_anchor={n:p.detach().clone() for n,p in model.signbert.embed.named_parameters()}
                wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],
                    output_device=args.local_rank,find_unused_parameters=True)
                report['gcn_freeze_transition']=dict(step=step,**transition)
                print(json.dumps(dict(event='gcn_freeze',**report['gcn_freeze_transition'])),flush=True)
            if cli.policy == 'staged' and step == cli.fusion_steps+1:
                del wrapped, optimizer
                active = configure_trainable_stage(model,'upper')
                optimizer,scheduler,wrapped = native.prep_optimizer(args,model,
                    schedule_steps-cli.fusion_steps,device,1,args.local_rank,coef_lr=args.coef_lr)
                report['stage_transition'] = dict(step=step,optimizer_reset=True,
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
            wrapped.train()
            if active is not None:
                training_modes(model,active)
            batch = train_collate([train[i] for i in indices])
            optimizer.zero_grad()
            # Lossless CPU storage for tensors saved by autograd. Keep native
            # batch32 negatives/objective; no microbatch contrastive approximation.
            # CPU packing may densify expanded attention masks (65x65 head
            # stride4225), which the fused efficient-SDPA backward rejects.
            # Math SDPA supports these layouts; same attention, no fused kernel.
            with (torch.autograd.graph.save_on_cpu(pin_memory=True) if cli.activation_offload else nullcontext()), \
                 (torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,
                    enable_mem_efficient=False,enable_cudnn=False) if cli.activation_offload else nullcontext()):
                losses = wrapped(*model_inputs(batch,device))
            reconstruction = None
            transfer_loss = None
            if cli.signrep == 'transfer':
                assert len(losses) == 8
                losses, transfer_loss = losses[:7], losses[7]
            if cli.masked_pose == 'reconstruct':
                assert len(losses) == 8, 'DDP must see auxiliary loss in forward output'
                losses, reconstruction = losses[:7], losses[7]
            assert all(torch.isfinite(torch.as_tensor(v)).all() for v in losses)
            if step == 1:
                assert torch.allclose(fused_priority_loss(losses,1),losses[0],atol=1e-6,rtol=1e-6)
                report['returned_loss_components_consistent'] = True
                report['criterion'] = (('fused_dcl_native_branches' if cli.dcl_weight == 1 else
                                        'fused_dcl_blend_native_branches') if cli.fusion_objective == 'dcl'
                                       else 'label_smoothed' if cli.label_smoothing else 'native')
            loss = fused_priority_loss(losses,cli.aux_weight)
            masked_stats = {}
            if cli.rgb_moddrop_start:
                masked_stats.update(model.fusion.rgb_moddrop.stats)
                if step == 2:
                    assert masked_stats['rgb_dropped_total'] > 0
                    assert masked_stats['rgb_seen_total'] == 2 * cli.batch_size
                    report['rgb_moddrop_training_passed'] = True
                if step == schedule_steps:
                    assert masked_stats['rgb_drop_probability'] == 0
                    assert masked_stats['rgb_dropped_samples'] == 0
                    report['rgb_moddrop_zero_terminal_passed'] = True
            if cli.fusion_objective == 'dcl':
                masked_stats.update(model.dcl_stats)
                assert masked_stats['native_fusion_reconstruction_passed']
                report['dcl_loss_replacement_passed'] = True
            if transfer_loss is not None:
                assert torch.isfinite(transfer_loss)
                loss = loss + cli.signrep_weight * transfer_loss
                loss_key = 'signrep_cosine_loss' if cli.signrep_loss == 'pointwise' else 'signrep_relational_loss'
                masked_stats[loss_key] = float(transfer_loss.detach())
                masked_stats['weighted_signrep_loss'] = float(cli.signrep_weight * transfer_loss.detach())
            if cli.masked_pose == 'reconstruct':
                assert reconstruction is not None and torch.isfinite(reconstruction)
                loss = loss + .05 * reconstruction
                masked_stats = dict(model.masked_pose.stats, reconstruction_loss=float(reconstruction.detach()),
                                    weighted_reconstruction_loss=float(.05*reconstruction.detach()))
                model.masked_pose.loss = None
            elif cli.masked_pose:
                assert model.masked_pose.loss is None
            if cli.global_exchange:
                masked_stats.update(model.fusion.global_exchange.stats)
            loss.backward()
            if cli.global_exchange and step == 3:
                gradients = [p.grad for p in model.fusion.global_exchange.parameters() if p.requires_grad]
                assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
                report['global_exchange_all_gradients_passed'] = True
            if cli.signrep == 'transfer':
                stats = model.signrep_transfer.stats
                assert stats['auxiliary_gradient_finite'] and stats['auxiliary_gradient_norm'] > 0
                report['signrep_gradient_passed'] = True
                masked_stats.update(stats)
                model.signrep_transfer.loss = None
            if cli.masked_pose == 'reconstruct':
                stats = model.masked_pose.stats
                assert stats['auxiliary_feature_gradient_finite'] and stats['auxiliary_feature_gradient_norm'] > 0
                encoder_gradients = [p.grad for p in model.signbert.embed.parameters() if p.grad is not None]
                assert encoder_gradients and all(torch.isfinite(g).all() for g in encoder_gradients)
                assert any(g.abs().sum() > 0 for g in encoder_gradients)
                report['reconstruction_encoder_gradient_passed'] = True
                report['auxiliary_returned_in_forward_single_backward'] = True
                masked_stats.update(stats)
            grad_norm = torch.nn.utils.clip_grad_norm_(wrapped.parameters(),1.)
            assert torch.isfinite(grad_norm)
            ensure_fp32_moments(optimizer)
            check_update = step in (2,cli.fusion_steps+2) if cli.policy == 'staged' else step == 2
            if cli.resume_from and step == start_step+1:
                check_update = True
            if cli.joint_bilinear and step == 3:
                check_update = True
                for s in ('st_gcn_hand','st_gcn_body'):
                    gradient = getattr(model.signbert.embed,s).joint_bilinear.input.weight.grad
                    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
                report['joint_bilinear_input_gradient_passed'] = True
            if cli.temporal_modulation and step == 3:
                check_update = True
            if cli.graph_bottleneck_adapter and step == 3:
                check_update = True
            if cli.gcn_freeze_after and step==cli.gcn_freeze_after+2:
                check_update=True
            before = {n:p.detach().clone() for n,p in tracked.items()} if check_update else None
            optimizer.step()
            optimizer.zero_grad()
            if cli.global_exchange and step == 3:
                state = transfer_state(model)
                load_transfer_state(model,state)
                assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in state.items())
                report['global_exchange_trained_delta_roundtrip_passed'] = True
                del state
            torch.clamp_(model.clip.logit_scale.data,max=np.log(100))
            if before is not None:
                changed = [n for n,p in tracked.items() if not torch.equal(before[n],p)]
                assert changed, 'No parameter updated at step2'
                report['changed_parameter_tensors_step2'] = len(changed)
                report['changed_parameter_groups_step2'] = sorted({n.split('.')[0] for n in changed})
                report.setdefault('stage_update_checks',{})[str(step)] = dict(
                    changed=len(changed),groups=sorted({n.split('.')[0] for n in changed}))
                if active is not None:
                    assert all(tracked[n].requires_grad for n in changed), 'Frozen weight drift'
                if cli.interaction != 'none':
                    assert 'signbert.articulator_interaction.output.weight' in changed, 'Interaction did not activate'
                    report['interaction_output_updated'] = True
                if cli.temporal_delta:
                    assert all(f'temporal_delta.{stream}.output.weight' in changed for stream in ('pose','rgb'))
                    report['temporal_delta_updated'] = True
                if cli.joint_exchange:
                    assert 'signbert.joint_exchange.output.weight' in changed, 'Joint exchange did not activate'
                    report['joint_exchange_updated'] = True
                if cli.masked_pose:
                    if cli.graph_bottleneck_adapter:
                        output_names=[f'graph_bottleneck_adapters.{stream}_{index}.output.weight'
                                      for stream in ('hand','body') for index in range(5)]
                        if step == 2:
                            assert all(name in changed for name in output_names), 'C23 adapter outputs inactive'
                            report['graph_bottleneck_adapter_outputs_updated']=True
                        if step == 3:
                            input_names=[f'graph_bottleneck_adapters.{stream}_{index}.input.weight'
                                         for stream in ('hand','body') for index in range(5)]
                            assert all(name in changed for name in input_names), 'C23 adapter bottlenecks inactive'
                            report['graph_bottleneck_adapters_updated']=True
                    if cli.temporal_modulation:
                        factors=[f'temporal_modulation.{kind}.{stream}'
                                 for kind in ('scale','shift') for stream in ('pose','rgb')]
                        if step == 2:
                            assert all(n in changed for n in factors), 'C22 stream factors inactive'
                            report['temporal_modulation_stream_factors_updated']=True
                        if step == 3:
                            assert 'temporal_modulation.phase' in changed, 'C22 shared phase inactive'
                            report['temporal_modulation_shared_phase_updated']=True
                    if cli.global_exchange:
                        directions = ('to_rgb',) if cli.global_exchange == 'pose_to_rgb' else ('to_pose','to_rgb')
                        assert all(f'fusion.global_exchange.{d}.output.weight' in changed
                                   for d in directions), 'C19 did not activate'
                        if cli.global_exchange == 'pose_to_rgb':
                            assert not any(n.startswith('fusion.global_exchange.to_pose.') for n in changed), \
                                'Frozen RGB-to-pose direction changed'
                        report['global_exchange_outputs_updated'] = True
                    if cli.signrep == 'transfer':
                        assert 'signrep_transfer.project.1.weight' in changed
                        report['signrep_head_updated'] = True
                    if cli.joint_bilinear:
                        # Warmup update1 has LR0; output first moves at update2.
                        # Only backward3 can propagate through its nonzero weight.
                        layers = ('output',) if step == 2 else ('input','output')
                        assert all(f'signbert.embed.{s}.joint_bilinear.{layer}.weight' in changed
                                   for s in ('st_gcn_hand','st_gcn_body') for layer in layers), 'Bilinear branch inactive'
                        report['joint_bilinear_output_updated'] = True
                        if step == 3:
                            report['joint_bilinear_updated'] = True
                    if cli.bone_features:
                        assert all(f'signbert.embed.{s}.bone_features.output.weight' in changed
                                   for s in ('st_gcn_hand','st_gcn_body')), 'Bone projections inactive'
                        report['bone_features_updated'] = True
                    if cli.adaptive_graph:
                        assert all(n in changed for n in report['adaptive_graph_parameters']), 'Graph delta did not activate'
                        report['adaptive_graph_updated'] = True
                    if cli.clean_fusion_control or (cli.gcn_freeze_after and step>cli.gcn_freeze_after):
                        assert all(n.startswith('fusion.') for n in changed), 'Non-fusion tensor updated'
                        report['only_fusion_updated'] = True
                    elif not cli.graph_bottleneck_adapter:
                        assert any(n.startswith('signbert.embed.') for n in changed), 'GCN did not update'
                    if cli.graph_bottleneck_adapter:
                        assert not any(n.startswith('signbert.embed.') for n in changed), 'Frozen native GCN changed'
                    if cli.gcn_clip_temporal:
                        assert any(n.startswith('signbert.GCN_Conv.') for n in changed), 'Clip temporal convolution did not update'
                        assert not model.signbert.GCN_Conv.training
                        report['clip_temporal_updated'] = True
                    assert any(n.startswith('fusion.') for n in changed), 'Fusion did not update'
                    if cli.masked_pose == 'reconstruct':
                        assert any(n.startswith('masked_pose.') for n in changed), 'Decoder did not update'
                    buffers = dict(model.named_buffers())
                    assert all(torch.equal(buffers[n],v) for n,v in frozen_buffers.items()), 'Frozen buffer drift'
                    report['masked_update_checks_passed'] = True
                if cli.geometry:
                    if cli.geometry_control:
                        assert all(n.startswith('fusion.') for n in changed), 'Control changed non-fusion parameters'
                        assert torch.count_nonzero(model.pose3d_branch.output.weight)==0
                        report['control_geometry_zero']=True
                    else:
                        assert 'pose3d_branch.output.weight' in changed, 'Geometry branch did not activate'
                        report['geometry_output_updated']=True
                    if cli.geometry_freeze_fusion:
                        assert all(n.startswith('pose3d_branch.') for n in changed)
                        report['only_geometry_updated']=True
                    buffers=dict(model.named_buffers())
                    assert all(torch.equal(buffers[n],v) for n,v in frozen_buffers.items()), 'Frozen buffer drift'
                    report['frozen_buffers_unchanged']=True
                    check_adaptation_roundtrip(model)
                    report['adaptation_roundtrip_passed']=True
                if cli.policy == 'lora':
                    expected_b = [n for n in tracked if '.parametrizations.' in n and n.endswith('.B')]
                    assert len(expected_b) == 4*cli.lora_upper_blocks + 2*int(cli.text_lora)
                    assert all(n in changed for n in expected_b), 'Not all LoRA projections activated'
                    report['lora_updated'] = True
                    report['lora_updated_B_tensors'] = len(expected_b)
                    if cli.gcn_lora:
                        assert any(n.startswith('signbert.embed.') for n in changed), 'GCN did not update'
                        assert not model.signbert.embed.training
                        buffers = dict(model.named_buffers())
                        assert all(torch.equal(buffers[n],v) for n,v in frozen_buffers.items()), 'Frozen buffer drift'
                        report['gcn_lora_update_checks_passed'] = True
                del before
                if cli.resume_from and step == start_step+1:
                    report['resume']['first_resumed_update_passed'] = True
            torch.cuda.synchronize()
            row = dict(step=step,epoch=(step-1)//steps_per_epoch,loss=float(loss),
                       native_components=[float(v) for v in losses],gradient_norm=float(grad_norm),
                       seconds=time.time()-tick)
            row.update(masked_stats)
            if cli.activation_offload:
                row.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(),
                    cuda_peak_bytes=torch.cuda.max_memory_allocated(),
                    host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if cli.joint_bilinear and (step <= 2 or step % 10 == 0):
                row['bilinear_output_norms'] = {s:float(getattr(model.signbert.embed,s).joint_bilinear.output.weight.detach().norm())
                                                for s in ('st_gcn_hand','st_gcn_body')}
            if cli.adaptive_graph and (step <= 2 or step % 10 == 0):
                row['adaptive_graph'] = graph_statistics(model)
            if cli.bone_features and (step <= 2 or step % 10 == 0):
                row['bone_output_norms'] = {s:float(getattr(model.signbert.embed,s).bone_features.output.weight.detach().norm())
                                            for s in ('st_gcn_hand','st_gcn_body')}
            with (out/'train_steps.jsonl').open('a') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n')
            report['steps'] = step
            if cli.activation_offload and step % 16 == 0:
                save_offload_resume()
            if not cli.smoke and step in evaluation_steps:
                saved = rng_state()
                metrics = evaluate(native,args,model,loader,device,dev_ids,out/f'eval_step{step:04d}')
                restore_rng(saved)
                report['evaluations'][str(step)] = metrics
                r1 = {d:metrics['fusion'][d]['R1'] for d in ['T2V','V2T']}
                mean = sum(r1.values())/2
                eligible = all(r1[d] >= initial_r1[d]-.5 for d in r1)
                print(json.dumps(dict(event='dev',step=step,mean_R1=mean,R1=r1,
                    delta_reference=mean-reference_mean,delta_incumbent=mean-incumbent_mean,
                    guardrail_pass=eligible)),flush=True)
                if mean > selection['mean_R1'] + 1e-8 and eligible:
                    path = out/'best.pt'
                    torch.save(dict(model=transfer_state(model) if subset_pilot else adaptation_state(model) if cli.geometry else model.state_dict(),
                                    step=step,config=vars(args),adaptation=vars(cli),
                                    checkpoint_format=report.get('checkpoint_format','full_model'),
                                    base_checkpoint=report.get('base_checkpoint')),out/'best.tmp')
                    os.replace(out/'best.tmp',path)
                    selection = dict(step=step,mean_R1=mean,R1=r1,checkpoint=str(path),checkpoint_sha256=sha(path))
                    atomic_json(out/'selection.json',selection)
                if cli.early_stop_drop_pp is not None and reference_mean-mean > cli.early_stop_drop_pp:
                    report['stopping_reason'] = dict(kind='dev_degradation',step=step,
                        drop_pp=reference_mean-mean,threshold_pp=cli.early_stop_drop_pp)
                    print(json.dumps(dict(event='early_stop',**report['stopping_reason'])),flush=True)
                    record()
                    break
            if step % 10 == 0 or cli.smoke:
                print(json.dumps(row),flush=True)
            record()
        if cli.gcn_freeze_after:
            if frozen_gcn_anchor is not None:
                assert all(torch.equal(p,frozen_gcn_anchor[n]) for n,p in model.signbert.embed.named_parameters())
                assert all(torch.equal(dict(model.named_buffers())[n],v) for n,v in frozen_buffers.items())
                report['frozen_gcn_unchanged_final']=True
            else:
                assert 'stopping_reason' in report, 'Missing scheduled GCN freeze'
        if cli.geometry_freeze_fusion:
            assert all(torch.equal(model.fusion.state_dict()[k].cpu(),v) for k,v in fusion_anchor.items())
            report['fusion_state_unchanged_final']=True
        if not cli.smoke:
            if cli.geometry_control:
                assert torch.count_nonzero(model.pose3d_branch.output.weight)==0
                report['control_geometry_zero_final']=True
            torch.save(dict(model=transfer_state(model) if subset_pilot else adaptation_state(model) if cli.geometry else model.state_dict(),
                            optimizer=optimizer.state_dict(),scheduler=None,
                            checkpoint_format=report.get('checkpoint_format','full_model'),
                            base_checkpoint=report.get('base_checkpoint'),
                            rng=rng_state(),batches=batches,next_batch_index=report['steps'],
                            config=vars(args),adaptation=vars(cli),code_sha256=report['code_sha256'],
                            optimizer_precision='fp32_moments_native_parameters'),out/'last.pt')
        report.update(status='completed',exit_status=0,selection=selection,
                      stopped_early='stopping_reason' in report,
                      delta_reference=selection['mean_R1']-reference_mean,
                      delta_incumbent=selection['mean_R1']-incumbent_mean,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                      decision='smoke_only' if cli.smoke else 'await_DEV_driven_refine_promote_drop')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        if cli.activation_offload and torch.cuda.is_initialized():
            report['cuda_failure_memory'] = dict(allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(),peak=torch.cuda.max_memory_allocated(),
                free=torch.cuda.mem_get_info()[0])
        raise
    finally:
        record(True)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
