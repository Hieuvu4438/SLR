# Next Action

Evaluate the H1b pilots (E005/E006 seed 43; E007 detach; E008 random negatives) once queue B finishes them.

Why: E002 showed +2.31 PrimaryDev for ITM-auxiliary training (1 seed, warm-start); must separate seed luck,
generic regularisation and the hypothesised representation-shaping mechanism before spending 2.7 h+ per full run.

Falsification: kill H1b if E006 − E005 ≤ 0.5 PrimaryDev (seed 43). Mechanism refuted if E007 (detach) retains
the gain; hard-negative claim refuted if E008 (random) ≈ E002.

Prerequisites: E004 finished (queue B waits on its DONE.json); GPU free enough for 12 GB.
