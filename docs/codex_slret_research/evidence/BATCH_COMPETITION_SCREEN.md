# Batch competition: existing evidence and attribution boundary

2026-09-23, Cycle16. AI-assisted academic-research-suite source verification
and adversarial review, performed inline; no independent reviewer. This is a
bounded hypothesis screen and historical-artifact check, not a new training
experiment or a demonstrated bottleneck.

## Question

Does limited in-batch competition justify a new native-loss, larger-negative-set
method, independently of the closed DCL, hard-mining and PMGR mechanisms?

[H] Some rank-critical TRAIN competitors might receive insufficient joint
exposure. The required prediction would be a retrieval benefit from changing
competitor exposure at controlled update/exposure budgets, not merely a lower
loss or a greater count of compared pairs. Persistent DEV errors alone do not
establish this hypothesis. No DEV confusers may become training targets.

## Local evidence omitted from the current reconstructed blacklist

The [C18 result](../../../research/slret_goal_v2/RESULTS.md:110) already records
the exact physical-batch enlargement experiment. Current static artifacts:

| Run under `artifacts/slret_goal_v2/` | Batch | Updates | TRAIN sample exposures | Selected step | Selected mean DEV R1 |
|---|---:|---:|---:|---:|---:|
| `seds-signrep-control-offload-001` | 32 | 160 | 5120 | 160 | 78.32369942196533 |
| `seds-native-batch64-offload-002` | 64 | 80 | 5120 | 40 | 78.22736030828517 |

[V] Both run records say completed, seed42, ten epochs, and `test_loaded=false`.
Ordered TRAIN512 and DEV519 ID lists are equal. Recorded inherited checkpoint
and feature digests are equal; the underlying feature/checkpoint bytes were
not rehashed this turn. This is not a claim about present process liveness.

[M] Both saved batch-order hashes match their run records. Every B64 batch
equals the concatenation of the corresponding two consecutive B32 batches.
Thus the complete schedules have identical sample order/exposure, but different
numbers of optimizer updates. The selected checkpoints have different exposure
counts too: 2560 for B64 versus5120 for B32. These are selected recipe outcomes,
not an isolated causal effect of negative cardinality.

[R, saved-score replay only] The two selected `fusion_video_x_text.npy` arrays
are finite519×519 matrices. Their ordered video/text IDs agree within and across
runs. Calling `shared/slr_common/evaluation/cico_eval.py:evaluate_score_matrix`
with the recorded paired-ID mapping exactly reproduces R1/R5/R10/MedianR/MeanR
and every per-query record in both directions. No model forward was executed.
T2V R1 is77.07129094412332 in both; V2T is79.57610789980733 for B32 and
79.38342967244701 for B64. Mean delta is−0.09633911368016 percentage points.
No significance or general batch-size-harm claim follows from one short seed.

The current [launcher](../../../research/slret_goal_v2/tools/run_c18_batch64_v4.py)
explicitly checks nested batches. Its retry changes infrastructure timeout,
not the training recipe. Current trainer lines595–599 use `seed+epoch` when
generating permutations: a source-level repeated-identical-epoch explanation
is not supported. Source generations/config schemas differ between the saved
runs (`signrep=control` versus `native_subset` among them); no blanket
byte-identical implementation claim is made. Historical integration/update gates
are inherited, not newly rerun.

## Adjacent prior art: bounded verification

Read official publication metadata and abstracts, not full methods/results:

- Gao et al., [Scaling Deep Contrastive Learning Batch Size under Memory
  Limited Setup](https://aclanthology.org/2021.repl4nlp-1.31/), RepL4NLP2021:
  gradient caching separates encoder backpropagation from the batch-coupled
  loss. Generic memory-efficient batch enlargement is therefore not a new
  SLRet mechanism merely because the scorer has contextual tokens.
- Chen et al., [Why do We Need Large Batchsizes in Contrastive Learning?
  A Gradient-Bias Perspective](https://papers.nips.cc/paper_files/paper/2022/hash/db174d373133dcc6bf83bc98e4b681f8-Abstract-Conference.html),
  NeurIPS2022: authors analyze minibatch gradient bias and propose Bayesian
  augmentation of the objective. This is a direct collision target for a
  generic minibatch-bias correction claim, not evidence of harmful SEDS updates.

Both are computational LevelIII publications; provisional GradeC for transfer
to the present hypothesis: fitnessC, reviewB (official proceedings), methodC
and dataC (abstract-only inspection), currencyA for historical priority,
conflictsC (not audited). No predatory-venue signal in the official records;
no independent funding/affiliation/retraction or indexing audit was performed.
No S2/DOI programmatic verification or cross-model verification is claimed.
The Cross-Batch Memory CVPR page returned403; its search excerpt is not used as
verified method evidence. SigLIP and other search hits remain unreviewed leads.

Searches: `contrastive retrieval batch size gradient bias full gallery gradient
cache cross batch negative sampling`; `sign language retrieval contrastive batch
size false negatives sigmoid loss`. This is not an exhaustive novelty search.

## Adversarial decision

1. Novelty: a bigger batch, cached negatives or gradient caching alone is an
   ordinary baseline/engineering option, not a distinct research contribution.
2. Experiment: exposure equality does not imply update equality or identical
   stochastic transformations; C18 cannot settle the broad causal hypothesis.
3. Sign-language validity: official unpaired captions are not independently
   verified semantic negatives. No human or AI relevance labels are inferred.
4. Retrieval: a training loss change is not deployed full-gallery improvement;
   the selected B64 recipe did not improve mean R1 over its local control.

Decision: **REJECTED as proposed** for generic batch enlargement/caching as the
new primary method. Retain the exact C18 recipe's defer/no-sweep boundary. Do
not reinterpret C17's closure as a ban on all native batch-composition research,
or C18's result as proof that competition can never matter. The strongest
counterargument—different updates confound the comparison—is valid, but does
not supply a new bottleneck or novel intervention and does not authorize a
repair sweep. Guide section41 gates1,3–5 are not established for a new method.

This changes the next action: do not launch a supposedly untried larger-batch
or cached-batch rescue. A future lead needs a distinct observed failure and
separating intervention, not another negative-count statistic without an open
decision consequence. No method is promoted; no training, GPU, TEST access,
deferred-job polling, data acquisition, annotation or third-party edit occurred.

## Current artifact fingerprints

SHA256, freshly computed this turn. Relative to `artifacts/slret_goal_v2/`:

| Run/file | SHA256 |
|---|---|
| `seds-signrep-control-offload-001/run.json` | `4b98c4b5874d3ac1a0e62114dcdedf6d9ff86279a42b2d0aeed747e7cdfb5869` |
| `seds-signrep-control-offload-001/batch_indices.json` | `3a0dae39c1ffc4193eb10eac4b13323ecb5b1c2a8c04a1972017510fa2e88c1b` |
| `seds-signrep-control-offload-001/eval_step0160/fusion_metrics.json` | `7201cf9f2c985807082d20000b3e3742f771e4c4d346445d75cd92f3385eb8c4` |
| `seds-signrep-control-offload-001/eval_step0160/fusion_video_x_text.npy` | `93d8b740016ba7f0e27ba81fcc2f8cb073959caad51239b7c2548be79317a3e8` |
| `seds-native-batch64-offload-002/run.json` | `2094e465518a789e0513613c31776688d075c437c92d3965d599d03b5f468f1c` |
| `seds-native-batch64-offload-002/batch_indices.json` | `25352bedd3a417f7b8742260318fa848c1c16961a9da04348e982c664ae582dd` |
| `seds-native-batch64-offload-002/eval_step0040/fusion_metrics.json` | `5b0e07de4c146057ab313773e4d10a4e8c31ca0de702d9359ca2e71c805078d1` |
| `seds-native-batch64-offload-002/eval_step0040/fusion_video_x_text.npy` | `30907a88bb7e06b08c69b4d82205fed07826887d6a48af0e895a073b77a0ed77` |
