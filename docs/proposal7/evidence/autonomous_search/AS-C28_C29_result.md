# AS-C28/C29 — recipe differences verified; table rollback does not rescue calibration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate / code audit
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (AST, formulas and identity verified; no training replication)
- Version Label: AS-C28-C29-result-v1

## Outcome

AS-C20 differs measurably from the inspected upstream DEFAULT training path.
This was a deliberately distinct calibration, not an upstream replication, so
the audit does not retroactively relabel it as one. Resetting the two drifted
text tables at the fixed final model fails to rescue held retrieval. No new
method, ordinary-freezing novelty, GO or information-ceiling claim follows.

## Exact frozen parameters and drift

Executed ONLY the actual upstream freeze-policy AST block, with
freeze_layer_num0 and linear_patch2d, on the exact AS-C19 initialization. No
main-program training/evaluation side effects or upstream dataloader execution.

| Key | Shape | Parameters | AS-C20 relative L2 drift | Changed rows |
|---|---|---:|---:|---:|
| clip.positional_embedding | 77×512 | 39,424 | .0241012 | 32/77 |
| clip.token_embedding.weight | 49,408×512 | 25,296,896 | .00394980 | 2106/49408 |

Exactly these two keys are frozen under that default. Total frozen25,336,320;
remaining trainable124,387,843 (includes inactive optional tensors). AS-C20
trained both. Max absolute drift.001099874position/.001297610token. Drift does
not prove semantic forgetting, and changed rows need not imply meaningful
gradient information; update arithmetic and weight decay can affect them.

Upstream name-based decay and AS-C20 all-1D exclusion differ on55tensors/33,539
parameters, including layernorm scales, class embedding and logit-scale scalars.
Their names and flags are retained in`AS-C28-RECIPE_run.json`. The upstream
predicate's literal`LayerNorm` substrings do not match the actual CLIP`ln_*`
parameter names. Do not substitute an intended policy for the executed one.

Upstream coef_lr1 applies defaultlr1e-5 across CLIP tensors. AS-C20 applies1e-4
to its three random tensors and1e-5 to copied generic tensors. The local README
default training example is How2Sign, NOT the established exact PH release
recipe. No claim of historical PH command reconstruction is made.

## Optimizer semantics: actual execution and independent formula

Actual local BertAdam matches an independently written uncorrected-moment
formula to5.55e-17 max absolute parameter difference over100fixed-gradient
float64 CPU steps. Constantlr1e-4, beta(.9,.98),epsilon1e-6, no decay/clipping:
its first coordinate change is−.00007070818 versus AdamW−.00009999950. By100
steps changes are−.01261952 versus−.00999995. Identical advertised beta/epsilon
and lr therefore do NOT make the optimizers equivalent; there is no simple
constant scalar correction across steps in this example.

| Applied update | Upstream schedule factor | AS-C20 schedule factor |
|---|---:|---:|
| 1 | 0 | .01 |
| 2 | .01 | .02 |
| 10 | .09 | .10 |
| 100 | .99 | 1 |
| 500 | .501570794 | .586824089 |
| 1000 | .00000246740 | 0 |

Upstream schedule reads each parameter's prior update count, beginning0. Its
post-warmup cosine is a function of total progress. AS-C20 starts at update1
and uses post-warmup-rescaled progress. Moment state still accumulates during
the upstream first zero-lr step. The matched synthetic scheduled/decay experiment
ends with max parameter difference.003666221; that total has multiple known
factors and is not attributed to bias correction alone.

The upstream CALLER globally clips model gradients before optimizer.step, which
then also clips each parameter. In the synthetic caller-consistent experiment,
extra internal clipping changes gradients by exactly0 at all recorded steps.
It would be misleading to describe upstream training as per-parameter-only
clipping. This is not a guarantee of bitwise identity on every real tensor shape.

Upstream caller clamps exp(logit_scale)<=100 after updates. AS-C20 has no clamp
and finishes at100.440804. A common positive scale alone does not change exact
inference ranks; its TRAINING effect is separate and unmeasured here.

## AS-C29 held-table intervention

Same clean update1000 model; full1375internal-held TRAIN gallery. No dev/test
input. Restore exact generic-loaded/target-cast AS-C19 tables, one or both, with
all other learned tensors checked bitwise unchanged after EVERY cell. Reload
original tables between cells. No persisted model modification or interpolation.

| Post-training table reset | T2V R1 | V2T R1 | Mean R1 | Delta mean pp |
|---|---:|---:|---:|---:|
| None, exact replay | 17.090909 | 17.890909 | 17.490909 | 0 |
| Position | 17.018182 | 17.818182 | 17.418182 | −.072727 |
| Token | 17.381818 | 18.327273 | 17.854545 | +.363636 |
| Both | 17.163636 | 18.254545 | 17.709091 | +.218182 |

All three nonidentity cells fail the fixed.5pp diagnostic threshold. Token-only
also loses.872727pp T2V R5; both loses1.018182pp T2V R5. Original AS-C20 held
>=50% EACH direction adequacy remains failed in every cell. All R5/R10 and
per-query ranks saved; none of the small improvements licenses a method pilot.

Crucially, post-training rollback is NOT training with frozen tables. Downstream
parameters co-adapted during learning. These results bound direct table effects
at one checkpoint but cannot reject or establish a training-time freezing effect.
No exact PH-recipe claim or harmful-forgetting story is supported.

## Execution and validation

AS-C28 terminal JSON completed in3.89s. Session handle expired before its output
was collected; terminal status and absent process confirmed completion. Exit
code not independently recovered; no restart. Source hashes match previously
recorded main/optimizer files. Two AST/formula tests subsequently passed.

AS-C29 completed exit0 in13.33s, peak allocated GPU1,115,479,552bytes. Full held
identity score matrix matches saved AS-C20 scores EXACTLY. Non-target tensors
unchanged. Small score files/JSON only; original checkpoint untouched. Target-
restore test confirms no accumulation between cells. Combined focused suite
49passed in1.12s; git diff --check passed. No numerical failures or retries.
These checks are not an independent full-training replication.

## Next decision

The audit establishes a concrete training-contract difference, not its causal
retrieval effect. If pursuing clean residual adequacy further, the next useful
experiment is a single-factor training comparison: AS-C20 recipe replay versus
the SAME recipe with only the two tables frozen from initialization, same seed,
data/order/augmentation/batch/update schedule and all1,375held rows. Do not bundle
BertAdam, lr, schedule, clamp or decay changes into the frozen-table arm. A new
preregistered protocol must preserve AS-C20 adequacy and distinguish ordinary
baseline calibration from research-method survival. Reuse of a weak baseline
cannot become the endpoint; a passed calibration still needs a controlled bridge
to strong R0 and the original novelty/B0-B3/three-seed/second-dataset gates.

No new training campaign has been launched by AS-C28/C29. No automatic optimizer
sweep, teacher or closed-family revival. Previous/current goal work is progress;
goal remains active, no global exhaustion or completion claim.

## Statistical/integrity scan (11/11)

1. Simpson: both directions/full-gallery outcomes retained; no every-source claim.
2. Ecological: table drift and synthetic updates not individual-sign behavior.
3. Berkson: fixed internal-held population and checkpoint, no selected-error fit.
4. Collider: no causal estimate conditioned on successful rollback queries.
5. Base rate: exact table/row counts and full1375gallery disclosed.
6. Regression to mean: small selected-cell gains not treated as method efficacy.
7. Survivorship: all synthetic checkpoints and all rollback cells retained.
8. Look elsewhere: no winning reset coefficient or significance inference.
9. Forking paths: separate protocols precede both audits, thresholds unchanged.
10. Correlation/causation: recipe difference/drift do not establish retrieval harm;
    rollback differs from training-time freezing and cannot answer that intervention.
11. Reverse causality: no claim that drift or optimizer choice caused the weak
    generalization observed earlier; prospective single-factor training needed.

ARS discipline kept executable recipe differences separate from efficacy claims.
