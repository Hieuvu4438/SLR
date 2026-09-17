# AS-C21: strong-baseline two-stream fusion-location diagnostic

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (before execution)
- Version Label: AS-C21-v1

Question: does retaining existing I3D streams through frozen contextualization
recover ranking evidence obscured by baseline early feature fusion? This is a
fixed acquisition/representation-interface diagnostic after AS-C20's failed
clean learning calibration, NOT a new fusion method or active-acquisition revival.

Use strong R0 seed42, full519 PH dev gallery and its frozen text cache. No test,
training, targets, changed positives or new feature extraction. Stream assets
are exactly the existing H2S-aware and BSL5K-agnostic dense I3D files. Baseline
fuses .1aware+.9agnostic BEFORE the video transformer. Compare:

1. Exact canonical early fusion, repeated twice (two-pass compute control).
2. Aware-only encoding, retaining the canonical valid/mask/index contract.
3. Agnostic-only encoding, retaining the same contract.
4. Fixed late-score fusion: .1*aware-score+.9*agnostic-score. No weight sweep.
5. Late-score alignment-negative control: cyclically shift aware-score VIDEO
   rows by1, then use the same .1/.9 mix. This deliberately destroys row alignment;
   it is not a semantic perturbation or perfectly distribution-matched intervention.

Native checkpoint precision; batches128, scoring blocks64, unchanged .5 dual
channel mix. Canonical token equality to frozen cache required; channel difference
<=2e-5 and exact directional rank parity required before interpreting variants.
Repeat-control channels exact. Assert the .9 input equals separately loaded
.1aware+.9agnostic within1e-6 and all masks/indices identical.

Record R1/R5/R10, persistent ranks, per-query ranks and component correct-set
overlap. Any union/oracle statistic is descriptive nondeployable headroom, not
a method. Lead gate: late fusion beats two-pass canonical by>=.5pp mean R1,
neither direction loses>.25pp, neither R5/R10 loses>.5pp, both persistent mean
ranks improve, and late fusion beats row-shift control by>=.5pp. A passed
one-checkpoint lead would still NOT satisfy three-seed novelty/method GO gates.
Failed interventions only reject this frozen fixed-fusion specification; the
encoder was trained on fused inputs, so distribution shift confounds a claim
that separate streams lack useful information. Ordinary fusion is established
machinery, not a novelty claim, regardless of diagnostic result.

Command: `PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.stream_fusion_probe`
Working directory `/home/haipd/SLR`. Hard timeout300s. Monitor process and
`AS-C21-STREAMS_run.json`; numerical/parity failures retained, never silently
retried. Save only small score matrices and metrics under existing task folders.
