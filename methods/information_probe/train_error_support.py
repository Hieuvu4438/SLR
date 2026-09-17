"""Determine whether train hard-pair errors supply distinguishable supervision.

Uses exact deployed token IDs, not caption equality or first-30-token shortcuts.
Input collision is a mathematical limitation of paired-ID supervision, not an
expert claim that two videos have equivalent signed meaning.
"""
from collections import Counter
import json

import numpy as np
import torch
import yaml

from slr_common.data.tokenize import encode_cico_text
from slr_common.upstream.factory import load_cico_tokenizer
from .common import ART,ROOT,dump,rows,sha


def main():
    torch.set_num_threads(4)
    config=yaml.safe_load((ROOT/'runs/ph_base_b512_s42/resolved_config.yaml').read_text())
    tok=load_cico_tokenizer(config)
    result={'experiment_id':'AS-C04-TRAIN-SUPPORT','status':'measured',
            'code_sha256':sha(__file__),'tokenizer_scope':'exact deployed encode_cico_text max_words32',
            'splits':{},'directions':{}}
    keys_by_split={}
    for split in ('train','dev'):
        records=rows(split)
        keys=[tuple(encode_cico_text(r['caption_model'],tok,32)[0].tolist()) for r in records]
        keys_by_split[split]=keys
        counts=Counter(keys)
        old=[tuple(tok.tokenize(r['caption_model'])[:30]) for r in records]
        old_counts=Counter(old)
        changed_partition=sum((counts[k]>1)!=(old_counts[o]>1) for k,o in zip(keys,old))
        result['splits'][split]={
            'n':len(records),'manifest_sha256':sha(ROOT/f'artifacts/manifests/ph_{split}.jsonl'),
            'distinct_deployed_inputs':len(counts),
            'duplicate_input_rows':sum(n for n in counts.values() if n>1),
            'duplicate_input_classes':sum(n>1 for n in counts.values()),
            'long_caption_rows':sum(len(tok.tokenize(r['caption_model']))>30 for r in records),
            'duplicate_membership_changes_vs_first30_shortcut':changed_partition,
            'pair_id_top1_identifiability_bound_pct':100*len(counts)/len(records),
            'bound_limit':'one fixed top1 prediction per input; not optimistic tie-expanded recall and not a semantic ceiling',
        }
    hard=torch.load(ART/'train_confusers.pt',weights_only=True)
    keys=keys_by_split['train']
    counts=Counter(keys)
    for d,key in [('T2V','text_confuser'),('V2T','video_confuser')]:
        channels=np.load(ART/f'train_pair_channel_deltas_{d}.npy')
        margin=channels.mean(1)
        same=np.array([keys[i]==keys[int(j)] for i,j in enumerate(hard[key])])
        duplicated=np.array([counts[k]>1 for k in keys])
        wrong=margin < -1e-4
        tied=np.abs(margin)<=1e-4
        entry={
            'n':len(margin),'strict_negative_margin_n':int(wrong.sum()),'near_tie_n':int(tied.sum()),
            'negative_margin_same_deployed_input':int((wrong&same).sum()),
            'near_tie_same_deployed_input':int((tied&same).sum()),
            'negative_margin_unique_input':int((wrong&~duplicated).sum()),
            'near_tie_unique_input':int((tied&~duplicated).sum()),
            'different_input_nonpositive_n':int(((wrong|tied)&~same).sum()),
            'different_input_positive_n':int((~(wrong|tied)&~same).sum()),
            'nonpositive_ids_different_input':[
                {'index':int(i),'pair_id':rows('train')[i]['pair_id'],'hard_index':int(hard[key][i]),
                 'margin':float(margin[i])} for i in np.flatnonzero((wrong|tied)&~same)],
        }
        result['directions'][d]=entry
    result['limits']=[
        'These are train errors from an in-sample trained frozen backbone, not out-of-fold generalization errors.',
        'Identical text inputs do not establish identical signed meaning.',
        'Identifiability limits do not authorize changing official positives or reopening PMGR.',
        'Few distinguishable train errors would weaken negative-readout interpretation, not establish a new method.',
        'The tokenizer uses uniform content-token sampling for long captions, not first-30 truncation.']
    dump(ROOT/'docs/proposal7/evidence/autonomous_search/AS-C04-TRAIN-SUPPORT.json',result)
    print(json.dumps({**result,'directions':{d:{k:v for k,v in x.items() if k!='nonpositive_ids_different_input'}
                                             for d,x in result['directions'].items()}},indent=2))


if __name__=='__main__':
    main()
