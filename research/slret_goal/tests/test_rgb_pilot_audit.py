import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from rgb_pilot_audit import select_checkpoint,check_pair_rows,audit_scores,ROOT
sys.path.insert(0,str(ROOT/'shared'))
from slr_common.evaluation.cico_eval import evaluate_score_matrix


def metric(t,v):
    return dict(fusion=dict(T2V=dict(R1=t),V2T=dict(R1=v)))


class PilotAuditTests(unittest.TestCase):
    def test_score_audit_preserves_ties_and_rejects_corruption(self):
        ids=['a','b','c']
        matrix=np.array([[1,1,0],[0,1,2],[0,0,1]],dtype=np.float32)
        mapping={x:[x] for x in ids}
        saved=evaluate_score_matrix(matrix,video_ids=ids,text_ids=ids,
                                    video_to_text=mapping,text_to_video=mapping)
        summary={d:{k:v for k,v in saved[d].items() if k!='cols'} for d in ['T2V','V2T']}
        self.assertEqual(audit_scores(matrix,ids,saved,summary),summary)
        wrong=copy.deepcopy(saved)
        wrong['T2V']['cols'][0]+=1
        with self.assertRaises(ValueError): audit_scores(matrix,ids,wrong,summary)
        wrong=copy.deepcopy(summary)
        wrong['V2T']['R1']+=.1
        with self.assertRaises(ValueError): audit_scores(matrix,ids,saved,wrong)
        wrong=copy.deepcopy(saved)
        wrong['per_query']['V2T'][0]['ranked_candidate_ids'].reverse()
        with self.assertRaises(ValueError): audit_scores(matrix,ids,wrong,summary)

    def test_selector_keeps_initial_on_decline_or_tie(self):
        self.assertEqual(select_checkpoint({'0':metric(70,72),'111':metric(70.25,71.75),'222':metric(69,70)})['step'],0)

    def test_directional_failure_excludes_higher_mean(self):
        self.assertEqual(select_checkpoint({'0':metric(70,72),'111':metric(73,71.4),'222':metric(70.2,72.2)})['step'],222)

    def test_eligible_earlier_best_not_final(self):
        self.assertEqual(select_checkpoint({'0':metric(70,72),'111':metric(71,73),'222':metric(70.5,72.5)})['step'],111)

    def test_order_and_full_exposure(self):
        control=[dict(step=i+1,ids=list(range(i*32,min((i+1)*32,7096))),loss=[1.],gradient_norm=.2) for i in range(222)]
        pilot=[dict(**r,head_gradient_norm=.2) for r in control]
        check_pair_rows(pilot,control)
        wrong=copy.deepcopy(pilot)
        wrong[9]['ids'].reverse()
        with self.assertRaises(ValueError): check_pair_rows(wrong,control)
        with self.assertRaises(ValueError): check_pair_rows(pilot[:-1],control)
        wrong=copy.deepcopy(pilot)
        wrong[1]['loss']=[1.1]
        with self.assertRaises(ValueError): check_pair_rows(wrong,control)


if __name__=='__main__': unittest.main()
