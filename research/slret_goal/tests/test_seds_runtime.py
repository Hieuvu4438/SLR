"""Regression checks for source-versus-export label hashes and fail-closed assets."""
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import seds_runtime
from inventory import sha


class AssetContractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'third_party/SEDS/data_ph/train.pkl'
        self.source.parent.mkdir(parents=True)
        self.out = self.root/'out'
        for folder in ['labels','pose','rgb/train','metadata']:
            (self.out/folder).mkdir(parents=True)
        self.labels = {'a':{'video_name':'a','text':'one'},'b':{'video_name':'b','text':'two'}}
        self.source.write_bytes(pickle.dumps(self.labels,protocol=3))
        self.export = self.out/'labels/train.pkl'
        self.export.write_bytes(pickle.dumps(self.labels,protocol=4))
        pose, rgb = self.out/'pose/a.pkl', self.out/'rgb/train/a.pkl'
        pose.write_bytes(b'pose fixture')
        rgb.write_bytes(b'rgb fixture')
        (self.out/'metadata/a.json').write_text(json.dumps(dict(pose_sha256=sha(pose),rgb_sha256=sha(rgb),native_loader_smoke_pass=True)))
        (self.out/'run.json').write_text(json.dumps(dict(split='train',status='running',completed=1,videos=2,labels_sha256=sha(self.source))))
        self.scope = patch.object(seds_runtime,'ROOT',self.root)
        self.scope.start()
        self.addCleanup(self.scope.stop)

    def test_reserialized_labels_accepted_and_hashed(self):
        self.assertNotEqual(sha(self.source),sha(self.export))
        _, hashes = seds_runtime.verify_assets(self.out,'train',['a'],False)
        self.assertEqual(hashes[str(self.export)],sha(self.export))

    def test_modified_export_rejected(self):
        self.labels['a']['text'] = 'changed'
        self.export.write_bytes(pickle.dumps(self.labels))
        with self.assertRaisesRegex(ValueError,'Adapted labels'):
            seds_runtime.verify_assets(self.out,'train',['a'],False)

    def test_source_change_rejected(self):
        self.source.write_bytes(pickle.dumps({}))
        with self.assertRaisesRegex(ValueError,'Source labels'):
            seds_runtime.verify_assets(self.out,'train',['a'],False)

    def test_corrupted_feature_rejected(self):
        (self.out/'pose/a.pkl').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'Feature verification'):
            seds_runtime.verify_assets(self.out,'train',['a'],False)

    def test_full_run_requires_complete_extraction(self):
        with self.assertRaisesRegex(ValueError,'Incomplete extraction'):
            seds_runtime.verify_assets(self.out,'train',['a'])


if __name__ == '__main__':
    unittest.main()
