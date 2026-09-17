"""Post-training table intervention on inadequate clean model, not frozen training."""
import json
import os
import time
import traceback

import numpy as np
import torch
import yaml

from slr_common.data.cico_dataset import CiCoFeatureDataset
from slr_common.data.tokenize import CiCoCollator, encode_cico_text
from slr_common.upstream.cico_bridge import CiCoBridge
from slr_common.upstream.factory import load_cico_tokenizer
from .clean_initialization_audit import OUT, initialize
from .clean_train_calibration import evaluate
from .common import ART, ROOT, dump, rows, sha

KEYS = ('clip.positional_embedding', 'clip.token_embedding.weight')


@torch.no_grad()
def restore_tables(core, trained, initial, reset):
    assert set(reset) <= set(KEYS)
    params = dict(core.named_parameters())
    for k in KEYS:
        params[k].copy_((initial if k in reset else trained)[k])


def main():
    path = OUT/'AS-C29-ROLLBACK_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    result = {'experiment_id':'AS-C29-ROLLBACK','status':'running','pid':os.getpid(),
              'code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'AS-C29_protocol.md'),
              'method_go':False,'updates':0,'dev_or_test_loaded':False,'variants':{}}
    dump(path,result)
    try:
        run_path = OUT/'AS-C20-TRAIN_run.json'
        run = json.loads(run_path.read_text())
        assert run['status'] == 'completed' and run['updates'] == 1000
        cp = run['checkpoint']['path']
        assert sha(cp) == run['checkpoint']['sha256']
        partition_path = OUT/'AS-C19-TRAIN-partition.json'
        assert sha(partition_path) == run['partition_sha256']
        partition = json.loads(partition_path.read_text())
        assert sha(ROOT/'artifacts/manifests/ph_train.jsonl') == partition['manifest_sha256']
        result.update(checkpoint_sha256=run['checkpoint']['sha256'],partition_sha256=run['partition_sha256'],
                      manifest_sha256=partition['manifest_sha256'],calibration_run_sha256=sha(run_path))
        config = yaml.safe_load((ROOT/'runs/ph_base_b512_s42/resolved_config.yaml').read_text())
        generic_path = ROOT/'artifacts/pretrained/ViT-B-32.pt'
        assert sha(generic_path) == run['generic_clip_sha256']
        generic = torch.jit.load(str(generic_path),map_location='cpu').state_dict()
        core,_ = initialize(config,generic,42)
        del generic
        core.float()
        initial = {k:core.state_dict()[k].clone() for k in KEYS}
        saved = torch.load(cp,weights_only=True,map_location='cpu')
        core.load_state_dict(saved['state_dict'],strict=True)
        trained = {k:saved['state_dict'][k].clone() for k in KEYS}
        del saved
        core.cuda().eval().requires_grad_(False)
        # Snapshot non-target tensors for exact post-intervention integrity.
        other = {k:v.detach().cpu().clone() for k,v in core.state_dict().items() if k not in KEYS}
        tok = load_cico_tokenizer(config)
        records = rows('train')
        keys = [tuple(encode_cico_text(r['caption_model'],tok,32)[0].tolist()) for r in records]
        held = partition['held_indexes']
        dataset = CiCoFeatureDataset(ROOT/'artifacts/manifests/ph_train.jsonl',feature_len=64,alpha=.9,split='train')
        items = {i:dataset[i] for i in held}
        original_path = ART/'AS-C20/held_step1000.npy'
        assert sha(original_path) == run['evaluations'][-1]['held']['score_sha256']
        original = np.load(original_path)
        bm = run['evaluations'][-1]['held']
        for name,reset in [('identity',()),('position',(KEYS[0],)),('token',(KEYS[1],)),('both',KEYS)]:
            if time.time()-started > 300:
                raise TimeoutError('AS-C29 timeout300s')
            restore_tables(core,trained,initial,reset)
            s,m = evaluate(core,CiCoBridge(core),items,held,CiCoCollator(tok,32),keys)
            if name == 'identity':
                assert np.array_equal(s,original)
                result['identity_score_exact'] = True
            assert all(torch.equal(core.state_dict()[k].cpu(),v) for k,v in other.items())
            p = ART/f'AS-C29-{name}_held_scores.npy'
            np.save(p,s)
            lead = (m['mean_R1']-bm['mean_R1'] >= .5 and all(m[d]['R1']-bm[d]['R1'] >= -.25
                and all(m[d][k]-bm[d][k] >= -.5 for k in ('R5','R10')) for d in ('T2V','V2T')))
            result['variants'][name] = {'metrics':m,'score_sha256':sha(p),'non_target_tensors_exact':True,
                                        'diagnostic_lead':lead,'held_50_each_direction':all(m[d]['R1'] >= 50 for d in ('T2V','V2T'))}
            dump(path,result)
            print(json.dumps({'variant':name,'mean_R1':m['mean_R1'],'T2V_R1':m['T2V']['R1'],
                              'V2T_R1':m['V2T']['R1'],'diagnostic_lead':lead}),flush=True)
        result.update(status='completed',wall_seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        dump(path,result)
        print(json.dumps({'status':result['status'],'wall_seconds':result['wall_seconds']}),flush=True)
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc(),wall_seconds=time.time()-started)
        dump(path,result)
        raise


if __name__ == '__main__':
    main()
