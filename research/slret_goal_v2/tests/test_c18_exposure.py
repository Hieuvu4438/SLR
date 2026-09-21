import json
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_c18_batch64_v4 import check_exposure


class ExposureTests(unittest.TestCase):
    def setUp(self):
        self.small=[]
        self.large=[]
        for epoch in range(10):
            order=torch.randperm(512,generator=torch.Generator().manual_seed(42+epoch)).tolist()
            self.small.extend(order[i:i+32] for i in range(0,512,32))
            self.large.extend(order[i:i+64] for i in range(0,512,64))

    def test_same_exposure(self):
        check_exposure(self.large,self.small)
        self.assertEqual(sum(map(len,self.large)),5120)

    def test_wrong_order_rejected(self):
        self.large[0]=self.large[0][::-1]
        with self.assertRaisesRegex(ValueError,'order/exposure'):check_exposure(self.large,self.small)

    def test_wrong_size_rejected(self):
        self.large[0]=self.large[0][:-1]
        with self.assertRaisesRegex(ValueError,'batch size'):check_exposure(self.large,self.small)

    def test_native_control_record_matches_plan(self):
        root=Path(__file__).resolve().parents[3]
        source=root/'artifacts/slret_goal_v2/seds-signrep-control-offload-001/batch_indices.json'
        if not source.is_file():self.skipTest('Local control artifact absent')
        check_exposure(self.large,json.loads(source.read_text()))


if __name__=='__main__':unittest.main()
