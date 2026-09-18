"""CPU-only real-data contract check before adapted SEDS continuation."""
import argparse
import json
import pickle
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from inventory import ROOT, sha
from extraction_resume import atomic_json, digest
from seds_runtime import compatible_load, eval_dataset, native_kwargs, patch_pickle, verify_assets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    cli = parser.parse_args()
    out = ROOT / 'artifacts/slret_goal' / cli.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    report = dict(run_id=cli.run_id, status='running', kind='cpu_data_contract_not_retrieval',
                  command=sys.argv, test_loaded=False, seed=42, script_sha256=sha(__file__),
                  runtime_sha256=sha(ROOT/'research/slret_goal/tools/seds_runtime.py'),
                  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    try:
        assert not torch.cuda.is_available(), 'Run with CUDA_VISIBLE_DEVICES empty'
        torch.set_num_threads(2)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        base = ROOT/'third_party/SEDS'
        sys.path.insert(0, str(base))
        from modules.tokenization_clip import SimpleTokenizer
        from dataloaders.dataloader_ph_retrieval_pose import ph_DataLoader_pose, ph_pose_collate_fn
        from dataloaders.dataloader_ph_retrieval_train_pose import ph_DataLoader_train_pose, ph_train_pose_collate_fn
        patch_pickle(ph_DataLoader_pose, ph_DataLoader_train_pose)
        args = argparse.Namespace(**json.loads((ROOT/'artifacts/slret_goal/seds-adapted-dev-eval-002/run.json').read_text())['config'])
        train_root = ROOT/'artifacts/slret_goal/seds-adapted-train-001'
        dev_root = ROOT/'artifacts/slret_goal/seds-adapted-dev-001'
        tokenizer = SimpleTokenizer()
        train = ph_DataLoader_train_pose(**native_kwargs(args,tokenizer,train_root,'train'))
        dev, dev_ids = eval_dataset(ph_DataLoader_pose,args,tokenizer,dev_root)
        original = pickle.load((base/'data_ph/train.pkl').open('rb'))
        train_ids = [train.sentences_dict[i][0] for i in range(len(train))]
        assert train_ids == list(original) and len(train_ids) == 7096
        assert not set(train_ids) & set(dev_ids)
        # Include already committed long clips to exercise >300-frame subsampling.
        long_ids = []
        for vid in train_ids:
            path = train_root/'metadata'/(vid+'.json')
            if path.exists() and json.loads(path.read_text())['decoded_frames'] > 300:
                long_ids.append(vid)
                if len(long_ids) == 5:
                    break
        selected_ids = list(dict.fromkeys(train_ids[:32]+long_ids))
        _, train_hashes = verify_assets(train_root,'train',selected_ids,False)
        _, dev_hashes = verify_assets(dev_root,'dev',dev_ids)
        report.update(train_ids_checked=selected_ids, long_train_ids_checked=long_ids,
                      dev_count=len(dev_ids), train_assets_digest=digest(train_hashes),
                      dev_assets_digest=digest(dev_hashes))
        for vid in selected_ids:
            pose = compatible_load((train_root/'pose'/(vid+'.pkl')).open('rb'))
            meta = json.loads((train_root/'metadata'/(vid+'.json')).read_text())
            names = list(pose['img_list'])
            kept = train.GetTotalFrameList(pose, np.array([210,260],dtype=np.float32))
            assert [names.index(x) for x in kept] == meta['retained_frame_indices'], vid
        indices = [train_ids.index(vid) for vid in selected_ids]
        random_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        samples = [train[i] for i in indices]
        batch = ph_train_pose_collate_fn(samples)
        assert all(torch.isfinite(v).all() for v in batch.values())
        report['train_batch_shapes'] = {k:list(v.shape) for k,v in batch.items()}
        report['augmented_captions'] = int((batch['pairs_text'] != batch['pairs_text_aug']).flatten(1).any(1).sum())
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        repeated = ph_train_pose_collate_fn([train[i] for i in indices])
        assert all(torch.equal(v,repeated[k]) for k,v in batch.items()), 'RNG replay differs'
        for offset in range(0,len(dev),32):
            batch_dev = ph_pose_collate_fn([dev[i] for i in range(offset,min(offset+32,len(dev)))])
            assert all(torch.isfinite(v).all() for v in batch_dev.values())
        report.update(status='completed',exit_status=0,frame_indices='exact_metadata_match',
                      rng_replay='bit_exact_all_batch_tensors',all_dev_loader_samples='passed',
                      decision='CPU contracts pass only; GPU step0/optimizer smoke still required.')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        report['wall_seconds'] = time.time()-started
        atomic_json(out/'run.json',report)
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:
            f.write(json.dumps(report)+'\n')
        print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
