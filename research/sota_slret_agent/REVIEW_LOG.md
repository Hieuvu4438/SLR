# Review Log

## Cycle 1 (2026-09-23 12:55) — H1 cross-encoder reranker pilot

**Researcher.** Strongest evidence: joint ITM training lifted the bi-encoder by +2.31 PrimaryDev over the matched
warm-start continuation (paired CI [0.29, 4.43]); both directions improved; also +1.6 over E001's best.
Mechanism believed: hard-negative token-level cross-attention gradients sharpen word↔clip token features.

**Reviewer #2.** Novelty: ITM with hard negatives as an auxiliary loss is ALBEF (2021); porting it to SLRet is
incremental unless the sign-specific parts (typed pose/RGB memory, sign-visual negatives) are shown to matter.
Experimental: warm-start pilot, one seed, selection on the same val used for reporting (E002 best chosen by the
reranked metric, which is actually adverse to the bi-encoder claim; last-epoch comparison is selection-free).
Alternative explanation: any extra loss that perturbs the converged model (regularisation) could help —
random-negative and detach controls are required.

**Statistician.** Val = 519 groups ⇒ 1 query = 0.19 pts. CI excludes 0 at 95% but seed variance of the pilot
protocol is unknown (E003 vs E001 already differs by −0.7…−1.0). Needs seed 43 before scaling.

**Engineer.** Cost: +8.4 M params (head, training only), +~5% step time; no inference cost for H1b. Reproducible
via queue files; resume + watchdog in place. Risk: GPU shared with foreign jobs (timing noise only).

Decision: **REVISE** (H1 → H1b) → REPEAT + ABLATE (queue B: E005–E009).
