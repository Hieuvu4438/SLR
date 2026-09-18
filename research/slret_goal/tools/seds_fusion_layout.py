"""Rejected-layout diagnostic fixture. NEVER install this on a retrieval model.

The native attention already transposes K/V BEFORE its later reshape. Applying
this additional permutation would corrupt correct routing. Retained only as a
negative-control fixture for the independent attention/gradient regression test.
"""


def double_permutation_counterexample(x, heads):
    if x.ndim != 3 or heads < 1 or x.shape[-1] % heads:
        raise ValueError('Expected [batch,time,heads*head_dim] projection')
    b, t, d = x.shape
    return x.reshape(b, t, heads, d//heads).transpose(1, 2).contiguous().reshape(b, t, d)
