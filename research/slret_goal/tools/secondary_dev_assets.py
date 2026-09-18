"""Read-only DEV asset availability and label contracts; never load TEST."""
import csv
import json
from pathlib import Path
import pickle
import time

from inventory import ROOT, sha
from extraction_resume import atomic_json


def main():
    started = time.time()
    out = ROOT/'artifacts/slret_goal/secondary-dev-assets-001'
    out.mkdir(exist_ok=False)
    base = ROOT/'third_party/SEDS'
    csl_manifest = ROOT/'artifacts/manifests/csl_dev.jsonl'
    csl_csv = Path('/home/dongvk/datasets/CSL_Daily_Sentence_Crop/dev_data_with_num_frames.csv')
    h2_csv = Path('/home/shared_data/sign_language/How2Sign/eval/how2sign_realigned_val.csv')
    h2_json = h2_csv.parent/'eval_label/labels.dev.json'
    csl = [json.loads(line) for line in csl_manifest.read_text().splitlines()]
    with csl_csv.open() as f:
        csl_official = list(csv.DictReader(f))
    with h2_csv.open() as f:
        h2 = list(csv.DictReader(f,delimiter='\t'))
    h2_subset = json.loads(h2_json.read_text())
    csl_ids = {r['video_id'] for r in csl}
    assert csl_ids == {r['name'] for r in csl_official}
    results = {}
    for name, rows, pose_base, rgb_base, raw_base in [
        ('csl',csl,base/'datasets/CSL/RTM_Keypoints',base/'datasets/CSL/I3D_features',csl_csv.parent/'videos'),
        ('h2s',h2,base/'datasets/How2Sign/RTMpose/Pose_all_24rates',base/'datasets/How2Sign/I3D_features',h2_csv.parent/'raw_videos')]:
        with (base/('data_csl' if name=='csl' else 'data_h2')/'train.pkl').open('rb') as f:
            train = pickle.load(f)
        train_video_ids = {item['video_name' if name=='csl' else 'new_video_name'] for group in train.values() for item in group}
        items = []
        for row in rows:
            vid = row['video_id' if name=='csl' else 'SENTENCE_NAME']
            item = dict(id=vid,raw=(raw_base/(vid+'.mp4')).is_file(),
                        release_pose=(pose_base/(vid+'.pkl')).is_file(),
                        release_rgb_dev=(rgb_base/'dev'/(vid+'.pkl')).is_file(),
                        release_rgb_train=(rgb_base/'train'/(vid+'.pkl')).is_file(),
                        train_id_overlap=vid in train_video_ids)
            if name=='csl':
                for field in ['feature_agnostic','feature_aware','temporal_metadata']:
                    item[field] = Path(row[field]).is_file()
            else:
                item['subset_json_present'] = row['SENTENCE_ID'] in h2_subset
            items.append(item)
        fields = [k for k in items[0] if k!='id']
        results[name] = dict(rows=len(items),unique_video_ids=len({x['id'] for x in items}),
                             train_caption_groups=len(train),train_unique_videos=len(train_video_ids),
                             coverage={k:sum(x[k] for x in items) for k in fields})
        if name=='csl':
            results[name]['dev_caption_groups'] = len({r['caption_id'] for r in rows})
        atomic_json(out/(name+'_availability.json'),items)
    report = dict(run_id=out.name,status='completed',exit_status=0,kind='secondary_dev_asset_census',
                  test_loaded=False,feature_contents_loaded=False,script_sha256=sha(__file__),
                  source_hashes={str(p):sha(p) for p in [csl_manifest,csl_csv,h2_csv,h2_json]},
                  datasets=results,wall_seconds=time.time()-started,
                  decision='Availability only; no release preprocessing parity or fresh confirmation implied.')
    atomic_json(out/'run.json',report)
    with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:
        f.write(json.dumps(report)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
