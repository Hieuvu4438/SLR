"""Native SEDS adapters: explicit split contracts, never construct TEST."""
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from inventory import ROOT, sha


class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core') and int(np.__version__.split('.')[0]) < 2:
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def compatible_load(file, **kwargs):
    return CompatibleUnpickler(file, **kwargs).load()


def patch_pickle(*classes):
    for cls in classes:
        sys.modules[cls.__module__].pkl = SimpleNamespace(load=compatible_load)


def native_kwargs(args, tokenizer, root, split):
    root = Path(root)
    return dict(subset=split, data_path=str(root / 'labels'), features_path=str(root / 'pose'),
                features_RGB_path=str(root / 'rgb'), tokenizer=tokenizer, max_words=args.max_words,
                feature_len=args.feature_len, max_length_frames=args.max_length_frames,
                slide_windows=args.slide_windows, windows_stride=args.windows_stride, args=args)


def verify_assets(root, split, ids, require_complete=True):
    root = Path(root)
    report = json.loads((root / 'run.json').read_text())
    if report['split'] != split:
        raise ValueError('Wrong extraction split')
    if require_complete and (report['status'] != 'completed' or report['completed'] != report['videos']):
        raise ValueError('Incomplete extraction')
    # Historical report field hashes SOURCE labels, before DEV subsetting and
    # protocol-4 reserialization. Do not confuse it with the output file hash.
    source_path = ROOT / 'third_party/SEDS/data_ph' / (split + '.pkl')
    if sha(source_path) != report['labels_sha256']:
        raise ValueError('Source labels changed')
    with source_path.open('rb') as handle:
        source = pickle.load(handle)
    label_path = root / 'labels' / (split + '.pkl')
    with label_path.open('rb') as handle:
        labels = pickle.load(handle)
    if any(vid not in labels or labels[vid] != source[vid] for vid in ids):
        raise ValueError('Adapted labels differ from source')
    result = {str(label_path):sha(label_path)}
    for vid in ids:
        metadata = json.loads((root / 'metadata' / (vid + '.json')).read_text())
        for folder, field in [('pose', 'pose_sha256'), ('rgb/' + split, 'rgb_sha256')]:
            path = root / folder / (vid + '.pkl')
            actual = sha(path)
            if actual != metadata[field] or not metadata['native_loader_smoke_pass']:
                raise ValueError(f'Feature verification failed: {path}')
            result[str(path)] = actual
    return report, result


def eval_dataset(cls, args, tokenizer, root):
    root = Path(root)
    with (root / 'labels/dev.pkl').open('rb') as handle:
        labels = pickle.load(handle)
    manifest = ROOT / 'artifacts/manifests/ph_dev.jsonl'
    ids = [json.loads(x)['video_id'] for x in manifest.read_text().splitlines()]
    if len(ids) != 519 or set(labels) != set(ids):
        raise ValueError('Canonical DEV519 contract failed')
    data = object.__new__(cls)
    for key, value in native_kwargs(args, tokenizer, root, 'dev').items():
        if key != 'args':
            setattr(data, key, value)
    data.interval, data.threshold, data.frames_threshold = args.interval, args.threshold, args.frames_threshold
    data.crop_img_size = np.array([[args.crop_size, args.crop_size]], dtype=np.float32)
    data.captions = labels
    data.sentences_dict = {vid: labels[vid]['text'] for vid in ids}
    data.video_dict = {i: (vid, str(root / 'pose' / (vid + '.pkl'))) for i, vid in enumerate(ids)}
    data.video_RGB_dict = {i: (vid, str(root / 'rgb/dev' / (vid + '.pkl'))) for i, vid in enumerate(ids)}
    data.cut_off_points = list(range(1, len(ids) + 1))
    data.multi_sentence_per_video = True
    data.SPECIAL_TOKEN = {'CLS_TOKEN': '<|startoftext|>', 'SEP_TOKEN': '<|endoftext|>',
                         'MASK_TOKEN': '[MASK]', 'UNK_TOKEN': '[UNK]', 'PAD_TOKEN': '[PAD]'}
    data.sample_len = data.sentence_num = data.video_num = len(ids)
    return data, ids


def model_inputs(batch, device):
    batch = {k:v.to(device) for k,v in batch.items()}
    right, left = {'pose':batch['right_pose']}, {'pose':batch['left_pose']}
    body = dict(pose=batch['body_pose'], clips_start=batch['body_clips_start'],
                mask=batch['body_mask'], rgb=batch['RGB_feature'])
    return (batch['pairs_text'], batch['pairs_segment'], batch['pairs_mask'],
            right, left, body, batch['pairs_text_aug'], batch['pairs_mask_aug'])


def evaluate(native, args, model, data_loader, device, ids, out):
    from slr_common.evaluation.cico_eval import evaluate_score_matrix
    out.mkdir(parents=True, exist_ok=False)
    capture = {}
    original = native._run_on_single_gpu_new_mix
    previous_output = args.output_dir
    def collect(*a, **kw):
        result = original(*a, **kw)
        for name, index in [('fusion',0), ('pose',2), ('rgb',4)]:
            capture[name] = np.concatenate(result[index], axis=0)
        return result
    native._run_on_single_gpu_new_mix = collect
    args.output_dir = str(out)
    try:
        native.eval_epoch(args, model, data_loader, device, 1, False)
    finally:
        native._run_on_single_gpu_new_mix = original
        args.output_dir = previous_output
    if set(capture) != {'fusion', 'pose', 'rgb'}:
        raise ValueError('Native evaluator did not return all three streams')
    summary = {}
    mapping = {x:[x] for x in ids}
    for name, matrix in capture.items():
        np.save(out / (name + '_video_x_text.npy'), matrix)
        metric = evaluate_score_matrix(matrix, video_ids=ids, text_ids=ids,
                                       video_to_text=mapping, text_to_video=mapping)
        reference = {'V2T':native.tensor_text_to_video_metrics(matrix[:,None,:]),
                     'T2V':native.compute_metrics(native.tensor_video_to_text_sim(matrix[:,None,:]))}
        for direction in reference:
            for key in ['R1','R5','R10','MeanR']:
                assert abs(metric[direction][key] - reference[direction][key]) < 1e-5
        (out / (name + '_metrics.json')).write_text(json.dumps(metric) + '\n')
        summary[name] = {d:{k:v for k,v in metric[d].items() if k != 'cols'} for d in ['T2V','V2T']}
    return summary
