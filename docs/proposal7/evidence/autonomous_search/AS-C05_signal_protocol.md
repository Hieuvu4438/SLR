# AS-C05: probe-training signal intervention, before outcomes

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED
- Version Label: signal-v1

Observed issue: AS-C04 found that every negative T2V train hard-pair margin
involves identical deployed text input. Only one V2T different-input margin is
nonpositive, and it is only about -0.0006. Train residual errors are therefore
not representative of the dev persistent population. This is NOT evidence of
irreducible dev errors or permission to reopen grouped-positive methods.

Hypothesis: the frozen baseline term in residual contrastive training suppresses
learning in the diagnostic readout; the previous negative screen may be a
training-signal failure, not a representation-information failure.

Intervention: same readout, features, train anchors/confusers, seeds, optimizer,
and inference rule R0 + readout. Cross the baseline coefficient INSIDE TRAINING
LOSS only: 1 (ordinary residual training) versus 0 (readout-only contrastive
training). Zero-initialized readout means identical R0 inference at step0 for
both. No dev labels enter the loss. The loss/inference mismatch in coefficient0
is deliberate for this diagnostic; it is not claimed as a publication method
or a baseline-equivalent final objective.

Matched screen: modes pooled / interaction / zero; seeds42/1337/2026; 2,400
updates EACH condition; batch64 with same train-only hard-confuser sampling;
AdamW lr3e-4, cosine, wd.001; float32; clean text as before. Longer exposure is
matched across coefficient conditions, not compared only against 600-step runs.
Eighteen runs total. Save initial, selected, last and every100-step dev scores.

Falsification: if coefficient0 trains successfully but yields no attributable
full-gallery improvement beyond matched controls, reject suppression as a
sufficient explanation for the failed readout. If the standalone readout does
not learn train discrimination, classify the probe as inadequately powered or
broken, not as proof of information absence. Record train loss and post-training
train hard-pair discrimination in addition to dev metrics.

The method-survival thresholds remain unchanged: >=.5pp beyond the strongest
matched control in >=2/3 seeds, positive source-cluster bootstrap lower bound,
direction/R5/R10 nonregression, persistent-rank improvement, beyond-init gain.
Even a positive diagnostic cannot create Proposal8 without localization,
materially distinct candidates, targeted novelty checks, and further controls.
