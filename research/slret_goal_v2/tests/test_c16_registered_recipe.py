"""CPU-only recipe/collection checks; synthetic reports are not retrieval evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_c16_control_v4 as runner


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        for name in ('candidate','control'):
            (self.base/name).mkdir()
        (self.base/'candidate/last.pt').write_bytes(b'fixture')
        best = self.base/'candidate/best.pt'
        best.write_bytes(b'fixture')
        metrics = {s:{d:{m:70. for m in ('R1','R5','R10')} for d in ('T2V','V2T')}
                   for s in ('fusion','pose','rgb')}
        self.result = dict(status='completed',batch_order_sha256='same-order',
            checkpoint_sha256_inherited='same-base',signrep_data={'split':'TRAIN-only'},
            config=dict(run_id='candidate',signrep='transfer',signrep_loss='relational',
                        signrep_weight=1.,batch_size=32,seed=42),
            evaluations={'0':metrics},masked_update_checks_passed=True,
            delta_roundtrip_passed=True,signrep_gradient_passed=True,signrep_head_updated=True,
            selection=dict(checkpoint=str(best),checkpoint_sha256=runner.sha(best),mean_R1=78.))
        anchor = copy.deepcopy(self.result)
        anchor['config'].update(run_id='control',signrep='control',signrep_loss='pointwise')
        del anchor['config']['signrep_weight']  # Historical control predates this flag.
        anchor['selection']['mean_R1']=77.
        (self.base/'control/run.json').write_text(json.dumps(anchor))
        self.patch = patch.multiple(runner,BASE=self.base,RUN='candidate',REFERENCE='control',
                                   MODE='transfer',LOSS='relational',WEIGHT=1.)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_legacy_control_accepts_only_registered_intervention(self):
        self.assertEqual(runner.validate(self.result),1.)

    def test_wrong_weight_rejected(self):
        self.result['config']['signrep_weight']=.1
        with self.assertRaisesRegex(ValueError,'intervention mismatch: signrep_weight'):
            runner.validate(self.result)

    def test_changed_batch_rejected(self):
        self.result['config']['batch_size']=64
        with self.assertRaisesRegex(ValueError,'Recipe mismatch: batch_size'):
            runner.validate(self.result)

    def test_missing_update_gate_rejected(self):
        self.result['signrep_head_updated']=False
        with self.assertRaisesRegex(ValueError,'gradient/update gate'):
            runner.validate(self.result)

    def test_missing_selected_checkpoint_rejected(self):
        self.result['selection']['checkpoint']=str(self.base/'absent.pt')
        with self.assertRaisesRegex(ValueError,'checkpoint absent'):
            runner.validate(self.result)

if __name__=='__main__':
    unittest.main()
