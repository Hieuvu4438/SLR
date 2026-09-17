# CiCo pseudo-clip census: measured exposure is small

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, inline
- Date: 2026-09-15
- Verification Status: VERIFIED for deterministic census/replay; efficacy untested
- Disclosure: AI-assisted analysis; no independent human or linguistic review

## Outcome

[M] The already-known first-anchor merge creates gaps in 68/9,854 adaptation-
training segments (0.690075%). The equal-segment probability that a uniform
random training crop contains any frame outside its recorded accepted-window
union is **0.615417%**, below the locked 10% exposure gate. The exposure lead is
false. No retrieval effect was measured, and this is not a method candidate.

## Evidence and provenance

[V] The [CiCo paper, arXivv1 §3.2](https://arxiv.org/pdf/2303.12793)
describes 16-frame, stride-one pseudo-labeling and cross-entropy adaptation,
then combining adapted and unadapted encoders. Read scope refreshed here: full
§3.2, not verification of the final CVPR version or its complete experiments.

[V] Pinned SLRT revision: 38a4f7b00da7a858d59b7fabe5093876a84db8e0.
In `CiCo/I3D_feature_extractor/epoch_pseudo.py`, proximity to any retained anchor
can append frames to the first anchor. Materialization uses the maximum frame
index as an exclusive end. The existing local port explicitly preserves both
behaviors, with an existing synthetic regression test. Neither is newly discovered
here. New evidence is the complete index exposure census.

The completed OCEM adaptation lock records a 15-epoch TRAIN-only P14T adaptation
with 9,854 optimizer-training and 1,138 internal holdout segments. This census
uses that saved index, not regenerated labels or a newly trained encoder. Every
source ID in all 10,992 rows belongs to the current official TRAIN manifest's
7,096 IDs; 5,070 distinct source videos occur in the index. No raw paths were
followed. The local feature regime differs from the strongest H2S-transfer-aware
CiCo control and from unverified author-distributed feature provenance.

## Complete descriptive results

"Gap" below means outside the union of accepted windows **recorded for that
segment**, not semantically incorrect or unobserved video content.

| Metric | Adaptation TRAIN | Internal holdout | All index rows |
|---|---:|---:|---:|
| Segments | 9,854 | 1,138 | 10,992 |
| Source videos | 4,538 | 532 | 5,070 |
| Nonlocal-merge segments | 68 | 7 | 75 |
| Segments with uncovered gaps | 68 | 7 | 75 |
| Enumerated random-crop starts | 22,644 | 2,539 | 25,183 |
| Mean probability of any gap frame | 0.615417% | 0.562791% | 0.609969% |
| Mean probability of entirely gap crop | 0.241376% | 0.199359% | 0.237026% |
| Mean uncovered-frame fraction | 0.413895% | 0.356887% | 0.407993% |
| Longest materialized segment, frames | 175 | 199 | 199 |
| Largest uncovered gap-frame total | 138 | 168 | 168 |
| Single-window segments (15-frame materialization) | 4,878 | 564 | 5,442 |

Probabilities are equal-segment averages, not an average over the pooled set of
crop starts (which would over-weight long segments). Holdout/all columns describe
the same hypothetical uniform-crop calculation for comparison; actual holdout
evaluation uses centered crops. Center-start selection was checked, but no
holdout-center exposure endpoint or accuracy evaluation is claimed.

## Validation and execution

[Locked protocol](CICO_pseudoclip_protocol.md),
[first run](CICO-PSEUDOCLIP_run.json),
[deterministic replay](CICO-PSEUDOCLIP_replay.json).

- First execution: exit 0, PID 476522, 0.169077 seconds measured inside script.
- Replay: exit 0, PID 476731, 0.167691 seconds; exact full-payload match excluding
  PID and wall time. Both processes are terminal; no failed census attempt.
- Per-crop explicit frame enumeration equals an independent merged-interval /
  per-frame multiplicity computation for every integer endpoint in every row.
- The actual local temporal sampler was isolated by AST, with every possible
  random offset and every row's center start checked. No dataset import.
- Fixtures include hand-calculated 32-start nonlocal case: 31 crops touch a gap,
  one is entirely gap, and 256/512 total frame draws lie in the gap. Short-segment
  repeat-last padding and local/multiple-gap interval cases also pass.
- New fixtures: 3 passed in 0.02 seconds. Entire `methods/information_probe`
  directory suite: 90 passed in 1.76 seconds, exit 0. This suite scope differs
  from earlier combined baseline-fixture/probe counts; no removed-test claim.
- All declared input hashes match before/after. No videos, feature arrays,
  checkpoints, DEV/TEST content, SEDS assets, model updates, or upstream edits.

Hashes:

- Protocol: 74b864e2b676aa105e296c36822473fb1e1df93b41f7c344426eccad620ee2c3
- Probe: 667e8d9578915a9cd51a605dca8d8d976a3d1269f61b60a4c5a3099cbd5a5809
- First run: 5d8da4c0cdac8e57bc07f3176f79975e4213cc3468b912da55045955e6907d0c
- Replay: 29359c251a6a06dcd933111a5eab6d1ff840a6f623e738fa7ba0e0463e990c9b

## Interpretation safeguards and next decision

This census cannot reconstruct suppressed accepted windows, class correctness,
actual sampled crops in the historical run, per-example gradient magnitudes, or
effects of repairing grouping. Small sampling exposure does not prove no harm;
large gaps in the affected minority do not prove broad retrieval harm. The
15-frame exclusive-end behavior is counted, not tested for efficacy. Do not
rescue this route by adjusting the threshold, NMS, crop length, teacher, or loss.

Statistical fallacy scan, 11/11:

1. Simpson: train/holdout/all reported separately; no treatment trend asserted.
2. Ecological: segment expectation is not an individual retrieval prediction.
3. Berkson: index is confidence-selected; no inference to all sign frames.
4. Collider: no outcome-conditioned regression or selected performance slice.
5. Base rates: full segment and source-video denominators reported.
6. Regression to mean: no selected-extreme before/after comparison.
7. Survivorship: complete fixed index; omitted/rejected windows remain unknown.
8. Look-elsewhere: all registered endpoints reported; earlier negative probes kept.
9. Forking paths: endpoint/gate locked before census, exact independent check and
   replay; no threshold or subgroup rescue after observing results.
10. Correlation/causation: no label-noise, training-harm or retrieval-gain claim.
11. Reverse causality: no performance mechanism inferred from exposure alone.

Decision: deprioritize this specific source-repair route. OCEM and generic
pseudo-label/teacher/fusion families remain CLOSED. Next work must change the
research question and establish a distinct, internally open decision consequence
before spending GPU compute. The broader autonomous goal remains active; no GO,
Proposal 8, global exhaustion, or blocker is claimed.
