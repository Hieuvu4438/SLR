# AS-C19 — initialization feasible, useful held-out supervision unproven

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / provenance and feasibility audit
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (mapping/determinism/forward-backward verified)
- Version Label: AS-C19-result-v2

## Outcome

A standalone CiCo initialization using local generic CLIP and explicit new
I3D/temporal tensors is executable WITHOUT loading a deliberately PH-fitted
retrieval checkpoint. This resolves an implementation/provenance prerequisite,
not the adequacy of held-out train errors, model generalization or method novelty.
No fold-training, optimizer update, teacher-target creation, dev/test evaluation,
official split/positive change or new benchmark occurred.

Protocol `AS-C19_protocol.md`; complete audit `AS-C19-INITIALIZATION-V2_run.json`;
TRAIN-only internal indices `AS-C19-TRAIN-partition.json`. V2 completed exit0
in10.79s, peak allocated GPU3434233344bytes. Two new grouping tests; focused
suite33passed in1.21s. Compileall/git diff --check pass. No large model/cache
artifact saved. About31GB disk was available before this audit.

## Initialization provenance

Generic CLIP hash verified against the existing resource lock:
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
No web lookup or download; this is LOCAL lineage verification, not a new review
of CLIP's pretraining corpus. Train-only extraction reports for BSL5K agnostic
I3D and H2S-aware I3D each report7096 completed/0failed; their checkpoint hashes
match actual files and existing locks. Individual feature-pickle hashes were
not reverified; reports include resumed work rather than a fresh extraction.

Architecture metadata read from existing baseline config, but its PH checkpoint
path was NOT loaded. Generic CLIP supplies300 copied tensors/148,879,617 scalar
parameters, checked exactly after casting to the target dtype. New random
tensors total844,544 parameters:

|Tensor|Generic source shape|CiCo target shape|Initialization|
|---|---|---|---|
|Video input projection|768×3×32×32|768×1024×1×1|Seeded constructor|
|Video positions|50×768|65×768|Seeded constructor|
|Video optional compression|Absent|64×128×1×1|Seeded constructor; inactive in smoke|

Two additional scalar parameters, `logit_scale_first_softmax` and
`logit_scale_sec_softmax`, are explicitly constructor-initialized to1, bringing
total parameters to149,724,163. All other copied tensors—including text weights,
video transformer blocks and cross-modal projections—equal the generic source.
Metadata-only input_resolution/context_length/vocab_size entries are excluded.
Seeds42/42 yield identical records; seed1337 changes all three random tensors
while copied tensors remain exactly source-equal. No unknown key accepted.

This explicit shape-filtered mapping is a diagnostic specification, not a claim
to reproduce the official release's complete original training recipe. The
upstream private `_load_from_state_dict` path collects size errors and logs them;
our audit uses an exact missing-key allowlist so that a apparently successful
constructor cannot conceal unmapped tensors.

### Preserved failure

Attempt1 exited1 at the strict missing-key gate, before split creation or smoke:
the initial allowlist omitted the two scalar-one parameters. Failed
`AS-C19-INITIALIZATION_run.json` and source hash retained. Source inspection
located both declarations at module_clip.py:521–522 and no other references in
the inspected modeling/module_clip files. A recorded V2 amendment accepts ONLY
these two additional constants and asserts value1. It does not turn them into
random keys, skip a shape check, alter tolerances or weaken source provenance.
The V2 smoke confirms they have no gradient in this forward path. No hidden retry.

## Internal TRAIN partition

Fixed first8 SHA256 bytes of inferred source prefix modulo5; held fold0. Fit:
5721rows/518prefixes; held:1375rows/125prefixes. No prefix overlaps. All7096
TRAIN rows accounted for. Fold counts0/1/2/3/4 =1375/1512/1529/1414/1266.
Source prefix is not independently verified recording identity.

32 deployed-text keys occur on both sides;72/1375 held rows have exact fit-text
matches. This is NOT evidence that their held videos leaked, but a source-held
split is not a new-caption split. The alternative source+exact-text connected
component inventory has445components; largest1007,745,145,130,69. These counts
describe potential grouping tradeoffs; no new component partition was selected
after seeing them. Neither official positives nor train/dev/test membership
was changed; the new JSON only references internal TRAIN indices.

## Execution smoke and inference boundaries

First32 fit rows, clean and seed42/epoch0 augmented captions, existing fused
I3D features, explicitfloat32/eval, historical four-CE objective. Encodings,
losses and augmented-loss parameter backward all finite. Clean loss3.629602,
augmented3.644119; gradient norm23.464052 across149,715,969 gradient-bearing
parameters. Inactive keys: the optional compression8192weights plus2scalars.
No optimizer step or persisted learned state. This smoke does not estimate
batch512 training memory, full-gallery holdout accuracy or training runtime.

The strongest supported provenance statement is **no deliberately PH-fitted
retrieval initialization in this loader**. It does not establish that generic
CLIP/BSL5K/H2S pretraining excludes every PH clip, speaker or derivative. No
claim of absolutely contamination-free pretraining is made.

## Decision for the research loop

Next meaningful test is a prespecified, bounded TRAIN-only learning/calibration
run with the frozen internal partition. It must demonstrate learned retrieval,
not merely abundant errors from an untrained projection. Holdout labels must
not become teacher/correction targets while also serving as confirmation; any
subsequent readout requires a separate internal fit/validation contract.
No PH-dev tuning is needed for this adequacy decision.

Even successful generic-initialized holdout probes would concern a DIFFERENT
backbone state and training-data regime than R0. They cannot retroactively prove
R0 information sufficiency or justify a method gain against a weaker baseline.
A measured mechanism would need a controlled bridge to the original strong
baseline and unchanged GO gates. AS-C07's OTTER-like cohort distillation remains
rejected; this audit supplies no permission to relabel it as an OOF method.

## Statistical/integrity scan (11/11)

1. Simpson: per-fold counts and exact-text overlap retained, no pooled accuracy.
2. Ecological: source-prefix counts not claims about signer-level independence.
3. Berkson: one fixed diagnostic partition, no whole-dataset generalization.
4. Collider: no conditioned causal estimate; component inventory descriptive.
5. Base rate: all7096 rows accounted for,72 held exact-text overlaps disclosed.
6. Regression to mean: no before/after learning or correction gain claimed.
7. Survivorship: failed attempt preserved and successful attempt linked.
8. Look elsewhere: fixed hash-fold rule; no chosen result/seed winner.
9. Forking paths: scalar-key correction explicit and source-justified before V2.
10. Correlation/causation: finite gradients do not establish useful supervision.
11. Reverse causality: deterministic initialization says nothing about why R0
    performs well or whether any new learner will improve retrieval.

No GO/Proposal8 or global exhaustion claim. Previous and current turns progress;
goal remains active.
