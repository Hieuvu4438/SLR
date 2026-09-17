"""Algebraic falsification of two cohort-to-independent candidate mechanisms."""
import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

from .common import ROOT,dump,sha


def assignment_value(s):
    i,j=linear_sum_assignment(s,maximize=True)
    return float(s[i,j].sum())


def permutation_loss(s):
    values=[sum(s[i,p[i]] for i in range(len(s))) for p in itertools.permutations(range(len(s)))]
    return float(logsumexp(values)-np.trace(s))


def main():
    rng=np.random.default_rng(42)
    bank=rng.normal(size=(3,4))
    priors=np.array([assignment_value(np.delete(bank,j,axis=1)) for j in range(4)])
    max_error=0.
    for _ in range(20):
        q=rng.normal(size=4)
        forced=[]
        for target in range(4):
            scores=np.vstack((q,bank))
            values=[sum(scores[i,p[i]] for i in range(4)) for p in itertools.permutations(range(4)) if p[0]==target]
            forced.append(max(values))
        max_error=max(max_error,float(np.abs(np.array(forced)-(q+priors)).max()))
    s=np.eye(4)*10
    row=np.array([30.,0,0,0]);col=np.array([30.,0,0,0])
    altered=s+row[:,None]+col[None,:]
    a=linear_sum_assignment(s,maximize=True)[1]
    b=linear_sum_assignment(altered,maximize=True)[1]
    losses=[permutation_loss(x) for x in (s,altered)]
    recall=lambda x:{'T2V':float(100*np.mean(x.argmax(0)==np.arange(4))),
                     'V2T':float(100*np.mean(x.argmax(1)==np.arange(4)))}
    result={'experiment_id':'AS-C07-ALGEBRA','code_sha256':sha(__file__),
            'fixed_reference_bank_forced_score_additive_max_error':max_error,
            'assignment_unchanged_under_row_column_bias':bool(np.array_equal(a,b)),
            'permutation_loss_before_after':losses,
            'independent_recall_before':recall(s),'independent_recall_after':recall(altered),
            'interpretation':[
                'Fixed-bank forced assignment score is query score plus candidate-only potential, not a new conditional interaction.',
                'Permutation-only training leaves row/column score potentials unidentified despite their effect on independent ranking.',
                'These eliminate standalone candidate formulations, not every structured-retrieval mechanism.']}
    assert max_error<1e-12
    assert np.array_equal(a,b) and abs(losses[0]-losses[1])<1e-12
    dump(ROOT/'docs/proposal7/evidence/autonomous_search/AS-C07-ALGEBRA.json',result)
    print(result)


if __name__=='__main__':
    main()
