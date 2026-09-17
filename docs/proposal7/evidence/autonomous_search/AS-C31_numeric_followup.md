## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (post-hoc follow-up, specified before execution)
- Version Label: AS-C31-numeric-v1

AS-C31 completed successfully and FAILED its broad-prefix signature. Cache
reencoding was exact, but some shared-prefix vectors differ by up to.00390625.
Do not assume invariance or change the parent result/gate to hide this.

Read-only follow-up: classify every representative-to-member shared-prefix
comparison by original encoder batch sizes128 versus final tail (TRAIN56,DEV7).
Report exact equality counts separately within full batches, within the tail,
and across full/tail batches. This isolates a plausible numerical source without
asserting it in advance.

For the first16 nonidentical-full-input/nonexact-prefix pairs per split in
lexicographically sorted prefix order, test two unchanged-shape128-row forwards:
all rows initially inputA, then replace row0 with inputB sharing the prefix.
Measure prefix output difference and EOT difference. Separately forward identical
inputA with the corresponding tail batch size and compare prefix states.
This is a numerical/causal-mask check, not a retrieval intervention or semantic
minimal-pair test. No score, checkpoint, mask, threshold or dataset modification.
Record all deltas, do not tune precision or tolerance to obtain equality.

Command, timeout300s, expected tens of seconds, under4GB GPU:

    PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.text_prefix_numerics

Output AS-C31-PREFIX-NUMERIC_run.json. Parent result retained; no retry or overwrite.
