# Method 1 experiment log

This file records executed runs only. Engineering checks are not reportable retrieval results.

## 2026-09-13 — PH real-data engineering smoke

- Revision under test: `8f16fa6` plus the subsequently committed output-compaction helper.
- Command equivalent: `python methods/sssc/tools/run_ph_engineering_smoke.py --device cpu --max-steps 1`.
- Deliberate non-main regime: world size 1, global batch 2, one optimizer step, float32,
  no checkpoint saved. The main protocol remains global batch 512 and 200 epochs.
- Real inputs: audited PH train features/captions and the complete official 519-example dev pool.
- Train evidence: base loss `1.0128757953643799`; pre-outer-clip gradient norm
  `80.7323226928711`; batch identity SHA-256
  `0b95ee50009a6148d964b61b1c7756777f271360ad0a6b9ef387aa70c19d9061`.
- Dev path: `runs/method1/ph/base_engineering_smoke/seed42/dev_step_1.json` (ignored run
  artifact). Both directions produced R@1 `0.192678%`, R@5 `1.156069%`, and R@10
  `2.312139%` on the 519×519 pool.
- Interpretation: PASS as a real data/model/loss/evaluator smoke. Recall is effectively random
  after one step from initialization and must not be cited as a baseline, ablation, or SOTA
  result. The run is labeled `pilot_complete` and cannot satisfy the reference-teacher gate.
