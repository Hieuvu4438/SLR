# GCN-R1 retained-score diagnostic protocol

Registered 2026-09-22 before calculating error intersections or DEV gloss-order
burden. Read-only CPU analysis of completed, saved model outputs. No feature
recovery dependency, model forward, training, active-job checks or TEST access.

Use the already DEV-selected GCN-R1 checkpoints, without selecting them anew:
seed42 step666; seed1337 step222; seed2026 step222. Replay all three stream
metrics with the shared evaluator and immutable pair-ID positives, verifying
ordered gallery IDs, stored ranks and all R1/5/10/MedianR/MeanR. Fail on mismatch.
This certifies saved-score evaluation, not model inference or checkpoint bytes.

Measure (separately by T2V and V2T): per-seed errors, all-three persistent
errors, rank2–10 among persistent errors, common strict confusers, same top
wrong candidate, and whether each persistent error also fails in both pose and
RGB branches. Report exact ties separately; do not replace official ranking.

Join official PH DEV annotations by exact sample ID. Define the narrow order
slice by equal gloss **multisets**, different ordered gloss sequences, and
different whitespace-normalized native translations. These are candidate
contrasts, not new positive/negative semantic labels. Count their availability
and strict higher-score burden among persistent errors. A shared strict
candidate must outrank the positive in all three seeds.

Decision rule: this exact-order-confuser route supports a *material descriptive
lead* only if shared strict candidate burden reaches 10% of persistent errors
in at least one direction. Below that, do not use this slice as the main
justification for C27 or relax to fuzzy gloss matching. Even above 10%, no
linguistic validity, causal attribution or method efficacy follows automatically.
This rule does not reject all sequence supervision or all order information.

Gloss-length strata use ≤7 versus >7 (TRAIN median measured in Cycle1), not a
DEV-optimized cutoff. Signer strata are descriptive, not nuisance attribution.
Across-seed mean/std are descriptive selected-run variability, not uncertainty
for a new method or a publishable significance test.
