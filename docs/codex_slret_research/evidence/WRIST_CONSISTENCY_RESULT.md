# Duplicate-wrist consistency: registered TRAIN screen is NO_GO

## Material Passport

- Origin: academic-research-suite / experiment-agent, inline run + validate.
- Date:2026-09-22; Cycle15. AI-assisted; no human or independent reviewer.
- Verification: VERIFIED for deterministic census replay; ANALYZED for its
  scientific interpretation. Neither detector accuracy nor efficacy is verified.
- Version: wrist-consistency-v1; [pre-count protocol](WRIST_CONSISTENCY_PROTOCOL.md).
- Finalized2026-09-23 local time; the registered20260922 artifact names are retained.

## Result

**NO_GO** for frequent, sustained, high-confidence disagreement between hand-root
and body-wrist identity under the registered geometry criterion. No frame meets
that criterion. This does not establish correct left/right identity: both sets
of points can be wrong together, or errors can occur outside eligible frames.

| Population | Frames | Eligible frames | Flagged frames | Clips with >=16 eligible frames | Clips with >=3-frame flagged run |
|---|---:|---:|---:|---:|---:|
| Raw TRAIN | 827354 | 657030 | 0 | 6944/7096 | 0/7096 |
| Retained TRAIN, primary | 823485 | 653774 | 0 | 6944/7096 | 0/7096 |

Clip coverage is97.857948%, passing the locked80% screen-coverage requirement.
Eligible retained-frame coverage is79.391124%; this is a different denominator.
The5% affected-clip burden requirement fails. No threshold, run-length or
confidence sweep follows. No DEV error association or retrieval-effect claim.

## Execution and verification

In `/home/haipd/SLR`, both commands completed with exit0 before hard60s:

```bash
timeout 60s /home/haipd/miniconda3/bin/python docs/codex_slret_research/tools/diagnose_wrist_consistency.py --output docs/codex_slret_research/evidence/wrist_consistency_20260922.json
timeout 60s /home/haipd/miniconda3/bin/python docs/codex_slret_research/tools/diagnose_wrist_consistency.py --output docs/codex_slret_research/evidence/wrist_consistency_20260922_replay.json
```

The execution handles were34006 and94318; both terminal, no pending wait.
Elapsed runtime was not instrumented, so no precise timing is claimed.
`cmp` exits0: reports are byte-identical (2195118bytes each), SHA256
`369f9911f9ee80058afab8a61461b13becb6d884892cc4361c4c34b5882c4a52`.
All8 report source hashes were revalidated. Every TRAIN pose hash and the
ordered metadata inventory match Cycle7. RGB hashes are inherited metadata,
not newly recomputed RGB-file hashes. No raw video was decoded.

The checkpoint's actual `meta.dataset_meta.keypoint_id2name` independently
matches the six inspected config names/indices. Its SHA256 remains
`13ce77ad08808333e2d4f850c632ac6068c837108e15f1c3902dcbcf30842db9`.
Loading this local checkpoint on CPU only inspected metadata, not model inference.

Eight synthetic tests passed before the census; two additional boundary tests
were added afterward without changing the protocol or census implementation.
Fixtures cover aligned and consistently crossed hands, root-only swaps,
confidence/coverage thresholds, temporal gaps, transient flags, degenerate
geometry, translation/scale behavior, invalid input and the absolute-distance
gate. Synthetic controls certify the operator, not natural detection sensitivity.
Final focused suite:10/10 passed, exit0. This is not a full repository test run.

## Validation cautions —11/11 checked

1. Simpson: no causal group comparison; this is a TRAIN census, not pooled
   TRAIN/DEV evidence. No signer-specific generalization claim.
2. Ecological: aggregate consistency is not proof of any sign's meaning or accuracy.
3. Berkson: high-confidence/separated wrists are selected; low-confidence and
   overlapping-hand frames may contain precisely the difficult cases.
4. Collider: confidence filtering does not permit causal error attribution.
5. Base rate: all7096 clips retained, all eligible and total frame counts shown;
   no calibrated false-positive/false-negative rates or expert ground truth.
6. Regression to mean: no pre/post gain, selected-confuser fit or worst-case repair.
7. Survivorship: no clip dropped;152 clips fail the16-eligible-frame coverage
   count but remain in the7096 primary denominator.
8. Look elsewhere: one locked heuristic, no p-values or favorable threshold search.
9. Forking paths: protocol preceded census; post-census tests and checkpoint-name
   inspection are disclosed, not retroactive changes to the decision rule.
10. Correlation/causation: even flagged disagreement would not identify which
    estimate was wrong or prove retrieval harm. No flags is narrower still.
11. Reverse causality: no inferred direction between geometry and retrieval;
    no retrieval scores were read for this diagnostic.

## Decision consequence

Stop this specific duplicate-wrist disagreement lead. Do not implement a hand
swap repair, marginalization module, confidence gate or new pose stream from it.
This is a new bounded input diagnostic, distinct from earlier frame-coverage
and crop-normalization checks, not a novel retrieval method. It does not reopen
closed C07/C09/C21 or reliability-gating recipes. Wider pose errors remain unknown.

ARS discipline supplied pre-count gates, deterministic replay and claim limits.
No new annotation, AI semantic labels, training, GPU, DEV/TEST content, deferred
job polling or external contact. The main method/SOTA goal remains incomplete.
