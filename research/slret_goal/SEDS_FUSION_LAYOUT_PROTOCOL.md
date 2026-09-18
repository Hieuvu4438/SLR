# SEDS K/V head–time layout diagnostic — preregistered 2026-09-18

## Outcome: hypothesis falsified before GPU

The independent CPU attention reference matches native outputs AND input
gradients. Source reinspection confirms K/V already execute
`view(batch_size,-1,num_heads,head_size).transpose(1,2)` before the later
reshape. The initial hypothesis overlooked this earlier transformation.
Original test failed its expected-native/reference-disagreement assertion;
this was evidence AGAINST the hypothesis, not a reason to change native code.

The proposed installation API was removed before any model invocation. Retained
fixture is explicitly a double-permutation COUNTEREXAMPLE, never a correction.
Regression tests require native/reference equivalence and show that the proposed
extra permutation breaks it. No GPU diagnostic, training, checkpoint, or vendor
modification follows. Label: FALSIFIED_SOURCE_HYPOTHESIS, not a research-method
failure or general statement about attention. The planned screen below is void.

## Original preregistration (retained, not active authorization)

HYPOTHESIS (subsequently falsified): `DeformableMultiHeadedAttention.forward` projects
K/V as[B,T,H*Dh],then reshapes directly to[B,H,1,T,Dh], unlike Q's explicit
[B,T,H,Dh]→[B,H,T,Dh]. This routes different time/head elements to the grid sampler
than the standard per-head layout. It is not yet evidence of recall harm:
the released weights were trained with the native routing and may compensate.

Scope: eight K/V projection hooks inside the four active fusion attention layers
pre-permute storage so the unchanged native downstream reshape implements the
explicit per-head reference. Keep weights,query projections,offsets,modulo,
masking,y coordinate,align_corners,scale and losses unchanged. No bundled repairs.
CPU axis-tagged tests and independently expressed sampling/gradient reference
must pass first. Wrapper only; no vendored edits.

Runtime screen: same adapted PH DEV519,release checkpoint,no optimizer andnoTEST.
Encode once,score native once with raw pose/RGB/text cache; instrument actual K/V
projections for shape/nonzero exposure and relative discrepancy. Then score with
the isolated layout correction and restore hooks. Preserve fullnative matrix,
corrected matrix,ranks,bothdirections R1/5/10,hashes andall timing. Require native
matrix parity<=1e-4 andexactranks against DEV002 before interpreting correction.
No inference masks/pooling/precision changed. <=300s,reserve32MiB output.

Decision registered before runtime: if source/reference equivalence fails, stop.
If corrected step0 is below native by>2pp meanR1,do not assume that a short
continuation is a fair test of the intended architecture; report transfer shock
and defer retraining-from-initialization pending measured feasible budget. If
within2pp and exposure ismaterial(>1% projected elements routed differently),
one matched222update FP32-moment continuation may be considered with its own
storage preregistration and initialization-eligible control,not automatic launch.
An immediate gain must clear+0.5pp and directionalguardrail before efficacy claims.
Zero-shot regression does not show the head-layout convention is universally bad.

Collision screen: currentNO_GO plus fullhistorical conceptual registry read;
targeted search found no prior SEDS head/time-layout intervention. This is a
specific source-conformance correction,not global order loss,temporal consistency,
reliability gating,raw-grid readout,CMCM Gaussian repair or hard-negative mining.
Ordinary correction is B_corrected,not a novel retrieval method. No excluded
mechanism reopened; no paper contribution follows from one bug or unit test.
