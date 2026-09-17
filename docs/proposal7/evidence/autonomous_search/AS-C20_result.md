# AS-C20 — clean learning works, held residual adequacy fails

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (metric and checkpoint replay checked; training not replicated)
- Version Label: AS-C20-result-v1

## Outcome

The fixed 1,000-update calibration completes and learns fitting examples, but
fails its prespecified held-learning gate. Do not promote its abundant errors
to useful correction targets, claim an information ceiling, or compare its gains
from initialization to R0. It supplies no method candidate or GO.

| Update | Fit subset T2V R1 | Fit subset V2T R1 | Held T2V R1 | Held V2T R1 |
|---|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0000 | 0.0000 | 0.0727 |
| 250 | 75.3455 | 79.6364 | 11.9273 | 13.1636 |
| 500 | 90.6182 | 93.8909 | 15.2000 | 15.2000 |
| 1000 | 95.2727 | 96.2909 | 17.0909 | 17.8909 |

Both galleries contain1,375 rows. The fit gallery is the hash-selected subset of
the5,721 fitting rows, not full-training-gallery performance. Held sources are
the fixed125 inferred prefixes, not a new benchmark or independently verified
recording split. Official singleton positives and direction-specific tie rules
are unchanged. No dev/test data, PH retrieval checkpoint or R0 features loaded.

Final held mean R1=17.490909%, up17.454545pp from initialization but far below
the required50% in EACH direction. Fit thresholds pass; overall learning gate
fails. Strict different-text-input held errors:1120 T2V,1102 V2T, so the count
gate passes but adequacy does not. Fit-subset strict errors:13 T2V,1 V2T. This
is evidence of a fit/held gap in this regime, not its causal explanation.
Final held R5=41.309091/42.109091 and R10=54.109091/54.836364(T/V).
Fit R5=98.181818/98.109091 and R10=99.418182/99.345455.

Descriptive post-run slice (not preregistered inference):72 held rows share a
fit exact token sequence;1,303 do not. One-rank-per-query R1 is12.5/13.888889
for seen text and17.344589/18.112049 for unseen text(T/V). T2V here explicitly
uses optimistic per-query ranks, not its primary tie-expanded denominator.
Unequal, unbalanced groups do not identify a text-novelty effect.

## Execution and verification

`AS-C20_protocol.md` was written before execution. Exact commands and output
scope recorded there. Batch128, seed42, float32, both encoders trainable,100-step
warmup/cosine, shared augmentation, four-CE objective. All1,000 updates completed
in446.31s on RTX5880Ada, peak allocated GPU14,128,786,944bytes. Final25-step
mean loss=.022614. No numerical failure or timeout. Training session handle was
unavailable on later continuation; terminal JSON and absence of the process
confirmed completion. Exit code was NOT independently recovered; no rerun.

Final checkpoint SHA256:
`3d1a135f8a6b93c69989acb821ea719e268fdf147e15fecbaff05fc8ffd1799f`.
Saved update1000 regardless of metrics; no best-held selection. Scores for all
four evaluations and the checkpoint remain under`artifacts/proposal7/phase2/AS-C20/`.

Separate `AS-C20-VALIDATION_run.json` completed exit0 in9.90s. It checks hashes,
all eight score matrices against the shared official evaluator (all R1/R5/R10
exact), and reloads update1000 to reproduce the full held score matrix exactly,
maximum absolute difference0. This is artifact/evaluation replay, NOT a second
training run or a three-seed result. Environment: Python3.13.5, torch2.11.0+cu128,
CUDA12.8. Focused suite35passed; metric/gate boundary tests2passed separately.

## Decision and next branch

No threshold reduction, unregistered continuation, correction teacher or frozen
readout on these held errors. A later adequate PH-unfitted regime would need its
own controlled protocol; this single budget does not prove one impossible.
Generic pretraining overlap remains unknown. Different initialization, fit data,
negative pool and precision prevent a causal comparison to historical R0.

Move from clean-model learning calibration to the strong baseline's two-stream
acquisition/representation interface. Existing tests fused I3D streams before
contextualization; they did not isolate whether stream-specific evidence survives
separate encoding. A fixed early-versus-late fusion diagnostic can test this
without training a new weak model. Ordinary fusion is not a novel method; even a
positive screen would require a measured mechanism and new candidates/controls.

## Statistical/integrity scan (11/11)

1. Simpson: both directions, fixed galleries and seen-text slices reported;
   no source-wise reversal ruled out.
2. Ecological: aggregate fit/held gap not signer-level behavior.
3. Berkson: source-held and fit subsets are different populations; descriptive.
4. Collider: no error-conditioned causal estimate or correction gain.
5. Base rate:1,375 rows/gallery, known72-row overlap and error counts retained.
6. Regression to mean: gains from near-chance initialization not method efficacy.
7. Survivorship: all scheduled steps saved, no best-checkpoint filter.
8. Look elsewhere: four prespecified evaluations, one seed; no significance claim.
9. Forking paths: gates unchanged; extra seen-text slice labeled post-run.
10. Correlation/causation: gap does not distinguish limited data, source shift,
    initialization, objective or optimization causes.
11. Reverse causality: no claim that observed errors cause poor generalization.

ARS discipline kept numerical execution success separate from research adequacy.
Previous/current goal work is progress. Goal remains active; no global exhaustion.
