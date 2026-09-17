# SignRep comparator check and remaining-question audit handoff

## Material Passport

- Origin Skill: academic-research-suite / deep-research, inline
- Date: 2026-09-16
- Verification Status: ANALYZED; source inspection, no execution or efficacy claim
- Disclosure: AI-assisted research; no independent expert review

## Decision

Do not treat a SignRep-for-I3D replacement as the next novel method or as a
controlled answer to Q01 (information absent versus present but unused).
The comparator is relevant, but its resources and task differ. This check
advances an existing question; it does not manufacture another numbered method.

## Verified primary-source scope

The [SignRep preprint v1](https://arxiv.org/html/2503.08529v1), §§3–7.2, describes
Hiera-B initialized from Kinetics MAE and pretrained on YouTube-SL-25. Its targets
include geometric sign priors; regularization includes feature variance,
covariance and adversarial style terms. Its retrieval protocol is video-query
sign-dictionary lookup, not paired sentence text↔video retrieval. Thus its
published retrieval results do not establish superiority on our task. Read scope:
complete §§3–7.2 and the opened §7.4 excerpt; not complete final-paper verification.

Publication and author affiliation are corroborated by the
[University of Surrey record](https://openresearch.surrey.ac.uk/esploro/outputs/conferencePaper/SignRep-Enhancing-Self-Supervised-Sign-Representations/991013164802346),
which identifies ICCV 2025 and lists public grants. This is the author's
institutional provenance, not independent performance replication. No complete
conflict-of-interest or retraction-database check was performed.

Provisional source grade: B, primary controlled ML method evidence (approximately
Level III) for its own task; not a verified transfer effect in this project.

## Pinned public code inspected

[Official repository](https://github.com/ryanwongsa/SignRep), revision
`06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2` (commit API resolved `main` at inspection).
The recursive tree response has `truncated=false` and 12 tracked files. Full
contents read for:

- `example_usage.py`
- `models/final_models/FINAL_hiera_latent_model_head_v25_active.py`
- `models/head_models/dict_head_v6_FINAL_thoughtful.py`
- `augmentation/video/base_video_aug.py`

The example performs 16-frame, stride-two feature extraction and exposes both
`features` and `latent`. In the wrapper, these differ: `features` is the result
of a LayerNorm/linear projection of `latent`. A comparison must specify which
representation it uses. The forward also computes prior/activity heads; its
`extract_features` argument does not activate the commented-out shortcut. This
is a source-level execution/cost observation, not a measured numerical bug or
retrieval defect. No head-pruning or feature-choice sweep was performed.

The inspected tree contains model, head and augmentation implementations and an
extraction example, but no named pretraining driver/configuration or sentence-
retrieval trainer. This bounded tree observation does not prove the authors never
had such code or that another release cannot contain it. No checkpoint/release
assets were downloaded, no repository cloned, no dependency installed, no code
executed and no upstream files changed.

## Internal collision and attribution check

1. A pretrained-backbone substitution changes architecture, input representation
   and pretraining history simultaneously. Any improvement alone would not
   attribute a mechanism-specific gain beyond the matched resource control.
2. Generic pose-derived supervision, nuisance/style removal, extra streams and
   temporal self-supervision cannot be revived as new proposals; consult the
   binding [closure registry](../../Negative_Results_Registry.md).
3. A new source's positive result does not turn our negative probe into proof
   that frozen CiCo features lack information. Neither source establishes a
   current-project rank-critical information-loss signature.
4. Conversely, the source differences are not evidence that SignRep would fail
   in sentence retrieval. No such experiment was done.

## Existing-controls recheck: do not repeat AS-C27

The current files were read again, not merely recalled:

- [AS-C26/C27 result](AS-C26_C27_result.md): common 5,721-row fit reference;
  exact model held fixed; official 519 DEV compared with five fixed 519-row
  internal-held galleries. Primary held mean R1 27.649326%, DEV 32.273603%.
  Both were weak; no uniform seen-prefix advantage across directions. This
  already tests the obvious gallery/model confound at that checkpoint.
- [AS-C42 result](AS-C42_result.md): completed 8,800-update calibration, held
  mean 22.472727% in its 1,375-gallery protocol; original 50%-per-direction
  adequacy requirement failed. Its different gallery is not directly comparable
  with the preceding numbers. Source explicitly disallows duration/schedule/
  optimizer/freeze/seed rescue and does not establish information insufficiency.

Accordingly, neither another checkpoint's version of AS-C27 nor a pretrained
backbone swap supplies the missing new causal mechanism. No new DEV evaluation
or training was launched.

## Next mandatory work: audit the existing map, not declare exhaustion

The guide's §§10 and 43 allow a research-barrier conclusion only after an explicit
exhaustion table covering all high-priority feasible questions, evidence for
each rejection and reasons remaining branches cannot be pursued. The current
state's old sentence referring to "24 branches" is stale: the map now has Q01–Q37.
That historical sentence is not an exhaustion audit.

Required columns for the next audit:

1. Existing Q identifier and causal layer; retain every current entry.
2. Exact hypothesis tested versus broader question left unresolved.
3. Current artifact supporting the decision; distinguish protocol, run and result.
4. Adequacy and attribution limitations; missing or indirect evidence stays open.
5. Concrete next permissible intervention, if one exists; internal collision and
   resource requirements for that intervention.
6. Decision: feasible/high-information, unsupported bounded specification,
   prior-art collision, resource/authority dependency, or unresolved audit gap.

No terminal research barrier has been verified yet. In particular, unresolved
Q01 recoverability, Q02 geometry and Q17 relations cannot be relabeled impossible
solely because their previous narrow probes failed. If the audit finds an open,
feasible high-information intervention, execute that branch. If it instead
establishes a genuine scope/resource barrier, report the evidence and required
user decision without claiming a GO method or universal scientific impossibility.

No research worker is active. The previous subword/source investigation made
progress by correcting availability and novelty assumptions. The full goal
remains active; no completion or blocked status is requested here.
