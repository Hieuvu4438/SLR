# First selectively reopened CiCo pilot: no useful weighting gain

## Material Passport

2026-09-16; academic-research-suite, inline execution and validation. AI-assisted.
Status: completed controlled single-seed screen; deterministic output replay
verified, no independent training replication or method GO.

## Outcome

The first user-authorized reopening produced an actual implementation and780
optimizer updates: three matched32800-parameter heads,260updates each.
**Learned sentence-conditioned outer clip weighting did not beat unchanged CiCo.**
All three selectors chose step0. No seed/horizon/LR/weight rescue is justified
by this fixed screen; no larger campaign will be launched for this specification.

[Registered protocol](CICO-REOPEN-01_protocol.md),
[run and complete traces](CICO-REOPEN-01_run.json),
[saved-head validation](CICO-REOPEN-01_validation.json),
[weighting implementation](../../../../methods/information_probe/sentence_weighting.py),
[pilot runner](../../../../methods/information_probe/sentence_weight_pilot.py).

| Arm / step | T2V R1 | V2T R1 | Mean R1 | Mean delta vs CiCo |
|---|---:|---:|---:|---:|
| Unchanged CiCo / all initial heads | 74.181118 | 76.300578 | 75.240848 | 0 |
| Query-independent /130 | 73.217726 | 75.337187 | 74.277457 | −.963391 |
| Query-independent /260 | 73.410405 | 76.300578 | 74.855491 | −.385356 |
| Random TRAIN conditioning /130 | 72.639692 | 75.337187 | 73.988439 | −1.252408 |
| Random TRAIN conditioning /260 | 73.025048 | 76.107900 | 74.566474 | −.674374 |
| Sentence conditioning /130 | 72.832370 | 75.529865 | 74.181118 | −1.059730 |
| Sentence conditioning /260 | 73.410405 | 76.107900 | 74.759152 | −.481696 |

Official PH DEV519 full-gallery evaluation, unchanged positives/ties. Final
sentence-conditioned scores also worsen persistent mean ranks by.673913 T2V
and.494253 V2T (positive is worse). Final sentence T2V R5 is90.366089 versus
91.136802 baseline. Some higher recalls improve: final sentence T2V R10 and
V2T R5/R10 increase, so this is not an all-metric failure. Complete R5/R10,
MedR/MnR and per-query arrays are retained for every checkpoint.

The selected checkpoints equal unchanged CiCo and remain2.023121pp below the
existing77.263969 three-model control. The run's directional/R5/R10 gate flags
are true because they evaluate the selected **initial** checkpoint, not because
the trained sentence head passed those criteria. Lead margin0 and no persistent
gain fail the registered screen. No novel mechanism or SEDS superiority follows.

## What was actually trained

Only query/key weighting heads were updated; contextual CiCo encoders, source
features, text tokens, inner matching and the second channel remained frozen.
The controls share initial parameter bytes,260identical B64 batches, optimizer,
schedule and evaluation selection. Random conditioning uses only a fixed TRAIN
bank and a query-input hash, never another DEV query or relevance labels.
The blind head has equal parameter count but less effective conditioning capacity;
this limitation was declared before execution.

The heads changed: all final parameter hashes differ from initialization. Final
average weight L1 displacement from uniform is.202315 sentence/.202725 blind/
.208238 shuffled; max absolute sentence A-channel change is11.280282 logit units.
Thus failure is not explained by an identically disabled or zero-output head.
It does not establish the optimum of the function class or language-semantic
validity of the learned weights.

## Verification and resource accounting

- Initial test:1failed/2passed due9.54e−7 floating-point multiply/divide order.
  Implementation ordering was corrected before training; tolerance not relaxed.
  Focused final suite:7passed in.87s.
- All three initial mixed-score matrices exactly equal stored CiCo scores.
  Fresh channel maxabs discrepancy1.525879e−5, below locked5e−5 tolerance.
- Training exit0, no retry; reported elapsed6.922318s, peak GPU1816695296bytes
  on RTX5880Ada. Frozen-cache/head-only timing is not end-to-end throughput.
- All nine saved head/score/metric/rank outputs replay exactly in the same
  environment. Validation exit0,2.974778s, zero optimizer updates.
- Run SHA256:e0c3440c4c8fa769c244453f2ccddc892f3d24a85b2a9382c2003ce8344ca96e.
  Input/cache/manifest/checkpoint/source/protocol/batch hashes are in the run;
  every sampled batch and checkpoint remains under
  `artifacts/proposal7/CICO-REOPEN-01/`.
- No TEST data/evaluation, SEDS assets, new annotations/positives, upstream source
  modifications or external uploads. No worker remains active.

## Interpretation safeguards: 11/11 considered

1. Simpson: both directions and persistent results shown; unmeasured subgroup
   reversals are not ruled out.
2. Ecological: no individual sign/linguistic competence inference from recall.
3. Berkson: historical persistent errors and TRAIN confusers are selected sets;
   primary evaluation remains the complete DEV gallery.
4. Collider: no new error-conditioned fitting or nuisance adjustment.
5. Base rate:519queries per direction; persistent92/87, not new independent data.
6. Regression to mean: unchanged baseline and matched heads retained; no gain
   claimed from selected difficult instances.
7. Survivorship: all three completed arms/all checkpoints reported; initial unit
   failure disclosed, no crashed training omitted.
8. Look-elsewhere: one seed, three heads, fixed checkpoints; no significance claim
   or posthoc favorable subset/metric selection.
9. Forking paths: protocol preceded execution; no outcome-driven changes or retries.
10. Causation: intervention effect bounded to this frozen-cache specification;
    controls do not identify general semantic importance or full model potential.
11. Reverse causality: no retrospective linguistic mechanism assigned to errors.

## Next decision

This adds evidence for one previously untested learned-weighting specification;
it neither resets the old failures nor closes all permitted reopened ideas.
Do not reweight another channel, change hidden width/temperature, or add seeds
merely to rescue it. Next prioritize an encoder-side mechanism that changes
representations, with its own explicit reopening justification and matched
CiCo continuation. Inspect historical implementation and evidence first; this
note does not approve an unspecified decoder, pose or larger-backbone campaign.
The overall method-discovery goal remains active.
