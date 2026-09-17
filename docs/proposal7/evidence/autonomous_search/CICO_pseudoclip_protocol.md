# CiCo pseudo-clip exposure census — locked protocol

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, inline execution
- Origin Date: 2026-09-15
- Verification Status: PLANNED before aggregate exposure measurement
- Version Label: cico-pseudoclip-v1
- Authority: autonomous research loop §41; no upstream edits or training

## Question and decision

The nonlocal first-anchor merge in CiCo is **already documented and tested** in
the existing OCEM port. This census does not rediscover it or reopen OCEM.
Question: in the existing TRAIN-derived pseudoindex, how often can the actual
uniform random temporal sampler select frames outside the union of the merged
accepted windows recorded for that segment?

Primary endpoint: equal-segment mean probability of at least one such frame in
a training crop, on the 9,854 adaptation-training segments. An exposure lead
requires >=10%. This is a declared prioritization threshold, not a significance
test or retrieval GO gate. A passing result permits documenting a material
data-construction prerequisite for future reproduction; it does NOT authorize
an OCEM/pseudo-label repair campaign as a novel method. A failing result
deprioritizes this route. Either outcome requires changing research question
before a new efficacy experiment.

Secondary descriptive endpoints: nonlocal-merge prevalence (a merged start more
than 3 frames from its own anchor), positive-gap prevalence, probability of an
entirely uncovered crop, expected uncovered-frame fraction, and source-video
coverage. Report all segments and train/holdout separately. The holdout is an
internal split of official TRAIN, not DEV. No threshold tuning/subset selection.

## Inputs and safeguards

Only open the existing pseudoindex, current official TRAIN manifest, and named
source/lock/protocol files. Never follow video or feature paths in their rows.
No raw videos, features, checkpoints, classifier, DEV/TEST content, SEDS assets,
network uploads, or optimizer/model changes.

- Pseudoindex: `methods/ocem/runs/wp04_phoenix/pseudolabel_index.jsonl`, SHA256
  `f247de98d1917dee969d71dfef971debcd57e7002425dd8313fa3f4a2c07c432`.
- TRAIN: `artifacts/manifests/ph_train.jsonl`, SHA256
  `f032260cf21578876fcff997bef1df0d46094635d30032edb6b01c80693ace53`.
- Assert 7,096 unique TRAIN video IDs; all 10,992 pseudo rows must resolve to them,
  with 9,854 train / 1,138 holdout rows and unique pseudo IDs.
- Record hashes of grouping, crop sampler, pinned upstream source and completed
  adaptation lock; verify all input hashes unchanged after execution.

## Exact calculation and independent check

Accepted window at start s covers integer frames [s,s+16). The materialized
segment covers [min(s),max(s)+15), preserving the upstream exclusive-end
behavior. Uniform training crop starts range inclusively from segment start to
start+max(0,duration-16). A shorter segment repeats its final decoded frame.

For every row and every possible start, enumerate the 16 sampled frame indices
(clipping to the final materialized frame), and count indices outside the union.
Validate all integer numerators using a separate interval calculation: merge
accepted intervals, take their complement, count crop-start intervals intersecting
gaps and fitting inside gaps, and count per-frame inclusion multiplicities. Check
the actual local sampler's isolated AST for every offset using a deterministic
randint provider. No upstream dataset/module imports.

Synthetic fixtures must cover a lone 15-frame materialization with repeated last
frame, local merged windows without gaps, and a nonlocal gap. Census must fail
closed on schema, support, identity, hash, or independent-count mismatch.

Run command from repo root:

```bash
timeout 120s /home/haipd/miniconda3/bin/python -m methods.information_probe.cico_pseudoclip_audit --output docs/proposal7/evidence/autonomous_search/CICO-PSEUDOCLIP_run.json
```

Deterministic replay to a separate `_replay.json` must match all content except
wall time and process ID. Keep failed attempts. No inferential p-values or CIs:
this is a complete census of the fixed index and exact sampler expectation, not
a random sample of sign-language semantics or realized optimizer trajectories.

## Claim limits

"Outside recorded accepted-window support" does not mean an incorrect sign
label: suppressed windows are absent from this index, and semantic annotations
are not inspected. Window confidence is not a frame-level correctness oracle.
This local P14T adaptation is not the stronger H2S-transfer-aware CiCo control,
nor evidence about the authors' released feature files. No observed retrieval
harm, repair gain, publication novelty, or SOTA claim follows from this census.
