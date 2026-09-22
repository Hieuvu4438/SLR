# Absolute score drift versus relative retrieval margins

2026-09-22. Written before computing the new endpoint counts. AI-assisted
diagnostic, not a method proposal, semantic review, or training admission.

Question (Q15/Q16): can a rise in an off-diagonal compatibility score alone be
used as evidence that training made that competitor more harmful to ranking?
The existing three-seed *selected-checkpoint* comparison cannot answer this
within-run question. This uses retained chronological endpoints instead.

Motivation: [SCL-SLT's abstract and introductory analysis](https://aclanthology.org/2026.acl-long.2116.pdf)
discuss negative similarity trajectories. The paper was already in proposal5/7;
it is not newly discovered, and this is not a reproduction/refutation of its
selection algorithm or translation results. Only pp1–2/abstract were inspected
in this cycle; a subsequent request for the methods text timed out.

## Fixed scope and decision

- Existing PH DEV fusion scores only: the three `seds-gcn-horizon3*` runs used
  in Cycle2, comparing step111 to step666 for **every** seed (42,1337,2026).
  Do not select endpoints using results or substitute selected steps.
- Verify completed run metadata, no TEST, seed, exact official DEV ID order,
  finite 519×519 video×text matrices, and replay stored direction-specific ranks.
- Analyze all 519×518 off-diagonal pairs per direction and, separately, each
  query's highest-scoring off-diagonal competitor at step111, fixed thereafter.
  These are official unpaired items, **not certified semantic negatives**.
- Let `n` be competitor score, `p` the paired score, `m=p-n`.
  Report signs of `delta(n)` and `delta(m)`, including discordant quadrants
  (`delta(n)>0, delta(m)>0`; `delta(n)<0, delta(m)<0`). Strict movements must
  exceed a fixed 1e-7 in float64 arithmetic on the stored arrays. Report small
  or zero changes separately. This is a numerical guard, not a fitted threshold.
- Also report fixed-competitor strict crossing counts and official query-rank
  win/loss/tie counts. Rank and margin change are different quantities.
- Reject the *sufficiency* of raw-score drift as a rank-harm indicator if any
  discordant pair exists. No prevalence or intervention-effect claim follows
  from this logical counterexample criterion. Report every seed/direction.
- Algebraic control: adding the same scalar to all scores preserves paired
  margins, both directional softmax losses, and rankings, while changing raw
  score trajectories. This is a score-level invariance, not a claim that the
  fixed normalized encoder can realize an arbitrary shift.

No fitting, new checkpoint, new labels, semantic-equivalence inference, data
download, GPU, TEST access, negative miner, calibration, or gradient protection.
No p-values/independence claim: pairs share queries and gallery candidates.
Endpoint movement cannot identify the causal contribution of a training example.
Any future intervention still needs a distinct mechanism and prior-art screen;
the closed negative-mining, score-calibration and RPCA families remain closed.

## Reproduction

Run `tools/diagnose_score_drift.py --output <new-report.json>` from this research
directory (or its absolute path from repository root). Output includes source,
protocol, run, annotation, evaluator, score and metric hashes. It refuses to
overwrite an existing report. A deterministic second execution must produce
byte-identical JSON; wall-clock timing is excluded from that comparison.
