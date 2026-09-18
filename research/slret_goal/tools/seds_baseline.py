"""Run unchanged SEDS on TRAIN diagnostics or explicitly adapted full PH DEV.

Native metric function names differ from video-row/text-column semantics;
the native log labels correctly swap them. No test dataset is constructed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'shared'))
from inventory import sha


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--backward-batch-size', type=int, default=2)
    parser.add_argument('--adapted-root', type=Path)
    parser.add_argument('--evaluation-split', choices=['train', 'dev'], default='train')
    cli = parser.parse_args()
    if not 2 <= cli.limit <= 7096:
        raise ValueError('Explicit train diagnostic size required')
    if cli.evaluation_split == 'dev' and (cli.adapted_root is None or cli.limit != 519 or cli.backward):
        raise ValueError('DEV requires explicit adapted root, full519 gallery, and no backward')
    if cli.adapted_root:
        cli.adapted_root = cli.adapted_root.resolve()
        extraction = json.loads((cli.adapted_root / 'run.json').read_text())
        if extraction['status'] != 'completed' or extraction['completed'] != extraction['videos']:
            raise ValueError('Adapted extraction is incomplete')
        if extraction['split'] != cli.evaluation_split or extraction['videos'] < cli.limit:
            raise ValueError('Adapted split/size mismatch')
    command = list(sys.argv)
    out = ROOT / 'artifacts/slret_goal' / cli.run_id
    out.mkdir(parents=True, exist_ok=False)
    ledger = ROOT / 'research/slret_goal/experiments.jsonl'
    report = {'run_id': cli.run_id, 'status': 'running', 'pid': os.getpid(), 'start_unix': time.time(),
        'command': command, 'selection_split': 'none', 'evaluation_split': 'train_resubstitution',
        'limit': cli.limit, 'seed': 42, 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT)).hexdigest(),
        'runner_sha256': sha(__file__), 'artifacts': str(out), 'optimizer_updates': 0}
    if cli.adapted_root:
        report.update(evaluation_split=cli.evaluation_split + '_adapted_input_transfer',
                      preprocessing='adapted_not_B_release', adapted_root=str(cli.adapted_root),
                      extraction_report_sha256=sha(cli.adapted_root / 'run.json'))
    def record():
        write(out / 'run.json', report)
        with ledger.open('a') as f:
            f.write(json.dumps(report, allow_nan=False) + '\n')
    record()
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        from torch.utils.data import DataLoader
        from slr_common.evaluation.cico_eval import evaluate_score_matrix
        base = ROOT / 'third_party/SEDS'
        sys.path.insert(0, str(base))
        os.chdir(base)
        # Upstream import initializes NCCL using torchrun environment.
        import main_task_retrieval as native
        from dataloaders.dataloader_ph_retrieval_pose import ph_DataLoader_pose, ph_pose_collate_fn
        from dataloaders.dataloader_ph_retrieval_train_pose import ph_DataLoader_train_pose, ph_train_pose_collate_fn
        if cli.adapted_root and int(np.__version__.split('.')[0]) < 2:
            # Keep NumPy's module registry untouched: aliasing numpy._core
            # globally interferes with SciPy C-extension initialization.
            class CompatibleUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module.startswith('numpy._core'):
                        module = module.replace('numpy._core', 'numpy.core', 1)
                    return super().find_class(module, name)
            def compatible_load(file, **kw):
                return CompatibleUnpickler(file, **kw).load()
            for cls in [ph_DataLoader_pose, ph_DataLoader_train_pose]:
                sys.modules[cls.__module__].pkl = SimpleNamespace(load=compatible_load)
            report['numpy_pickle_compatibility'] = 'scoped PH-loader unpickler maps NumPy2 reconstruction names to NumPy1'
        sys.argv = [str(base / 'main_task_retrieval.py'), '--do_eval', '--signbert',
            '--init_model', str(base / 'ckpt/ph_best_model.bin'), '--fusion_type', 'gloss_atten',
            '--rgb_pose_match', '--rgb_pose_match_loss', '0.4', '--lr', '1e-5', '--sign_lr', '1e-4',
            '--max_words', '32', '--feature_len', '64', '--max_length_frames', '300',
            '--slide_windows', '16', '--windows_stride', '1', '--crop_size', '256',
            '--frames_threshold', '0.1', '--threshold', '0.4', '--batch_size_val', str(cli.batch_size),
            '--datatype', 'ph_pose', '--coef_lr', '1.', '--freeze_layer_num', '0', '--linear_patch', '2d',
            '--sim_header', 'Filip', '--pretrained_clip_name', 'ViT-B/32', '--num_thread_reader', '0',
            '--data_path', str(base / 'data_ph'), '--features_path', str(base / 'PHOENIX-2014-T/RTM_Keypoints'),
            '--features_RGB_path', str(base / 'PHOENIX-2014-T/I3D_features'), '--output_dir', str(out)]
        args = native.get_args()
        if cli.adapted_root:
            args.data_path = str(cli.adapted_root / 'labels')
            args.features_path = str(cli.adapted_root / 'pose')
            args.features_RGB_path = str(cli.adapted_root / 'rgb')
        args = native.set_seed_logger(args)
        device, n_gpu = native.init_device(args, args.local_rank)
        report.update(hardware=torch.cuda.get_device_name(), torch=torch.__version__, config=vars(args),
            checkpoint_sha256=sha(args.init_model), source_hashes={str(p.relative_to(ROOT)): sha(p) for p in
            [base / 'main_task_retrieval.py', base / 'modules/modeling.py', base / 'metrics.py',
             base / 'dataloaders/dataloader_ph_retrieval_pose.py']})
        tokenizer = native.ClipTokenizer()
        model = native.init_model(args, device)
        # Loading silently with missing keys would invalidate release parity.
        expected = torch.load(args.init_model, map_location='cpu')
        state = model.state_dict()
        report['checkpoint_contract'] = {'missing': sorted(set(state) - set(expected)),
            'unexpected': sorted(set(expected) - set(state)),
            'mismatched_loaded_tensors': [k for k in expected if k in state and not torch.equal(expected[k].cpu(), state[k].cpu())]}
        del expected, state
        if any(report['checkpoint_contract'].values()):
            raise AssertionError('Loaded checkpoint tensor mismatch')
        kwargs = dict(subset='train', data_path=args.data_path, features_path=args.features_path,
            features_RGB_path=args.features_RGB_path, tokenizer=tokenizer, max_words=args.max_words,
            feature_len=args.feature_len, max_length_frames=args.max_length_frames,
            slide_windows=args.slide_windows, windows_stride=args.windows_stride, args=args)
        if cli.evaluation_split == 'train':
            dataset = ph_DataLoader_pose(**kwargs)
        else:
            # Native constructor has no dev caption-path entry. Initialize its
            # data contract explicitly without ever constructing a test loader.
            labels = pickle.load((cli.adapted_root / 'labels/dev.pkl').open('rb'))
            manifest = ROOT / 'artifacts/manifests/ph_dev.jsonl'
            canonical = [json.loads(x)['video_id'] for x in manifest.read_text().splitlines()]
            assert len(canonical) == 519 and set(labels) == set(canonical)
            dataset = object.__new__(ph_DataLoader_pose)
            for key, value in kwargs.items():
                if key != 'args':
                    setattr(dataset, key, value)
            dataset.subset = 'dev'
            dataset.interval, dataset.threshold, dataset.frames_threshold = args.interval, args.threshold, args.frames_threshold
            dataset.crop_img_size = np.array([[args.crop_size, args.crop_size]], dtype=np.float32)
            dataset.captions = labels
            dataset.sentences_dict = {vid: labels[vid]['text'] for vid in canonical}
            dataset.video_dict = {i: (vid, str(cli.adapted_root / 'pose' / (vid + '.pkl'))) for i, vid in enumerate(canonical)}
            dataset.video_RGB_dict = {i: (vid, str(cli.adapted_root / 'rgb/dev' / (vid + '.pkl'))) for i, vid in enumerate(canonical)}
            dataset.cut_off_points = list(range(1, len(canonical) + 1))
            dataset.multi_sentence_per_video = True
            dataset.SPECIAL_TOKEN = {'CLS_TOKEN': '<|startoftext|>', 'SEP_TOKEN': '<|endoftext|>', 'MASK_TOKEN': '[MASK]', 'UNK_TOKEN': '[UNK]', 'PAD_TOKEN': '[PAD]'}
            report['canonical_manifest_sha256'] = sha(manifest)
        dataset.sample_len = cli.limit
        dataset.cut_off_points = dataset.cut_off_points[:cli.limit]
        dataset.sentence_num = dataset.video_num = cli.limit
        ids = [dataset.video_dict[i][0] for i in range(cli.limit)]
        assert len(set(ids)) == len(ids)
        write(out / 'ids.json', ids)
        report['ids_sha256'] = sha(out / 'ids.json')
        report['asset_hashes'] = {p: sha(p) for i in range(cli.limit) for p in
            [dataset.video_dict[i][1], dataset.video_RGB_dict[i][1]]}
        if cli.adapted_root:
            for i, vid in enumerate(ids):
                metadata = json.loads((cli.adapted_root / 'metadata' / (vid + '.json')).read_text())
                assert report['asset_hashes'][dataset.video_dict[i][1]] == metadata['pose_sha256']
                assert report['asset_hashes'][dataset.video_RGB_dict[i][1]] == metadata['rgb_sha256']
        loader = DataLoader(dataset, batch_size=cli.batch_size, shuffle=False, num_workers=0, collate_fn=ph_pose_collate_fn)
        captures = {}
        original = native._run_on_single_gpu_new_mix
        def capture(*a, **kw):
            result = original(*a, **kw)
            for name, index in [('fusion', 0), ('pose', 2), ('rgb', 4)]:
                captures[name] = np.concatenate(result[index], axis=0)
            return result
        native._run_on_single_gpu_new_mix = capture
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        native.eval_epoch(args, model, loader, device, n_gpu, False)
        torch.cuda.synchronize()
        report['eval_wall_seconds'] = time.time() - start
        report['streams'] = {}
        for name, matrix in captures.items():
            assert matrix.shape == (cli.limit, cli.limit) and np.isfinite(matrix).all()
            np.save(out / f'{name}_video_x_text.npy', matrix)
            mapping = {x: [x] for x in ids}
            result = evaluate_score_matrix(matrix, video_ids=ids, text_ids=ids,
                video_to_text=mapping, text_to_video=mapping)
            write(out / f'{name}_metrics.json', result)
            native_row = native.tensor_text_to_video_metrics(matrix[:, None, :])
            native_col = native.compute_metrics(native.tensor_video_to_text_sim(matrix[:, None, :]))
            for direction, reference in [('V2T', native_row), ('T2V', native_col)]:
                for metric in ['R1', 'R5', 'R10', 'MeanR']:
                    assert abs(result[direction][metric] - reference[metric]) < 1e-5, (name, direction, metric)
            report['streams'][name] = {'native_kernel_row_V2T': native_row,
                'native_kernel_column_T2V': native_col,
                'score_sha256': sha(out / f'{name}_video_x_text.npy')}
        if cli.backward:
            train_data = ph_DataLoader_train_pose(**kwargs)
            batch = ph_train_pose_collate_fn([train_data[i] for i in range(cli.backward_batch_size)])
            batch = {k: v.to(device) for k, v in batch.items()}
            report['batch_shapes'] = {k: list(v.shape) for k, v in batch.items()}
            right, left = {'pose': batch['right_pose']}, {'pose': batch['left_pose']}
            body = {'pose': batch['body_pose'], 'clips_start': batch['body_clips_start'],
                    'mask': batch['body_mask'], 'rgb': batch['RGB_feature']}
            model.train()
            start = time.time()
            losses = model(batch['pairs_text'], batch['pairs_segment'], batch['pairs_mask'], right, left, body,
                batch['pairs_text_aug'], batch['pairs_mask_aug'])
            assert torch.isfinite(losses[0])
            losses[0].backward()
            torch.cuda.synchronize()
            grads = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None}
            assert grads and all(np.isfinite(x) for x in grads.values())
            report['backward'] = {'wall_seconds': time.time() - start, 'batch_size': cli.backward_batch_size,
                'losses_total_fusion_pose_rgb_klpose_klrgb_match': [float(x) for x in losses],
                'gradient_parameter_count': len(grads), 'nonzero_gradient_parameter_count': sum(x > 0 for x in grads.values())}
            write(out / 'gradient_norms.json', grads)
            model.zero_grad(set_to_none=True)
        report['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
        report['status'] = 'completed'
        report['exit_status'] = 0
    except Exception:
        report['status'] = 'failed'
        report['exit_status'] = 1
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['end_unix'] = time.time()
        report['wall_seconds'] = report['end_unix'] - report['start_unix']
        record()
        if 'torch' in locals() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
