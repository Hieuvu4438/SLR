# Method 1 experiment log

This file records executed runs only. Engineering checks are not reportable retrieval results.

## 2026-09-13 — baseline augmentation and released-feature audits

- The pinned UPRet PH loader calls `textaugment.EDA.random_swap` with probability `.5`.
  Source distributions `textaugment==1.3.4` and `2.0.0` have the same one-swap path used
  here: whitespace split, two inclusive `randint` draws with at most four retries for a
  distinct second position, one swap, then whitespace join. Method 1 retains those semantics
  and applies the specification-required stateless per-sample/epoch RNG namespace.
- The complete official CiCo feature ZIP passes `unzip -t` but contains test only. Across all
  642 PH test videos and 54,997 aligned rows, local/released agnostic cosine averages
  `0.99319`; local P14T-aware/released-aware cosine averages `0.45116`; and the configured
  mixed-feature cosine averages `0.97124`. The mixed per-video relative L2 error averages
  `0.23802`. This confirms a material resource-regime discrepancy without authorizing a
  test-only feature substitution.
- These are source/resource audits, not retrieval results. They do not invalidate the active
  S1 implementation, but they preclude an exact published-resource or broad SOTA claim.

## 2026-09-13 — PH seed-42 S1 attempt invalidated by stochastic-parity audit

- The first full S1 attempt reached optimizer step 870/2600 before it was stopped.
- Its best observed dev checkpoint was step 533 with mean bidirectional R@1
  `19.749518%`; training loss later approached zero while dev retrieval remained
  implausibly low for a baseline reproduction.
- A direct real-PH B=3 comparison against the pinned UPRet scorer found maximum
  I2T/T2I logit disagreement `0.0073204041`. The local distribution path sampled
  both distribution modules before either Gaussian noise tensor, while UPRet
  samples text noise before invoking the video distribution module. Dropout makes
  those RNG schedules observably different.
- After restoring source RNG order, the same source comparison has exact maximum
  logit difference `0.0`. The run is scientifically invalid and is excluded from
  teacher selection, reference caching and all comparisons. Its preserved artifacts
  are under `runs/method1/ph/base/seed42_invalid_rng_order_20260913/`.

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

## 2026-09-13 — PH real-batch contract inspection

- Config: `configs/method1/ph_seed42_base_initial.yaml`.
- Artifact: `artifacts/method1/audit/ph_real_batch.json` (ignored runtime artifact).
- Short caption: 2 full BPE tokens, fully retained, 24 valid sampled feature rows.
- Overlong caption: 57 full BPE tokens, ordinary baseline input is exactly 32 positions and
  auxiliary eligibility is disabled, 64 valid sampled feature rows.
- Both agnostic/aware files for both records were re-hashed against the canonical manifest;
  the `[2,1024,64,1]` batch was finite and its class position was ignored.
- This is an engineering/data gate, not a retrieval result.
