# C27 recovery verification and actual-window eligibility

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-22
- Verification Status: ANALYZED — artifact checks, not a repeat extraction
- Version Label: c27_recovery_validation_v2

## Outcome

[V] `c27-ph-feature-recovery-001` is terminal COMPLETED with exit0. The user
released the handoff with `xong`; this is no longer an unverified job wait.
The terminal record verifies resumption of the exact paused UniFormerV2 worker
at completion. It does not establish that worker's present liveness.

[M] All519 DEV and7096 TRAIN IDs match their official split CSVs and restored
label dictionaries. All15,230 saved pose/RGB files match their per-item hashes,
have expected shapes and finite values. Checkpoint and extractor hashes match
the launch record. All recorded clip starts and original-frame windows pass
the native window-construction check. Label values match the original native
label dictionaries; no relevance labels were changed.

[M] **13/7096 TRAIN examples fail the necessary CTC length condition**
`valid_windows >= gloss_tokens + adjacent_repeats`. All13 were additionally
loaded through the real native hand/body/RGB loader and retain the reported
ineligible window counts. There are16 native TRAIN checks in total and4 DEV
checks. Minimum TRAIN slack is−4. IDs and counts are retained in the
[final machine-readable report](c27_recovery_verification_20260922_v2.json).

[V/I] `GlossCTC.forward` explicitly raises on any such example; its zero mask
convention and CLS exclusion agree with the native returned mask. Therefore
the unchanged all-TRAIN C27 recipe cannot complete an epoch that includes these
examples. This is a technical feasibility finding, not a retrieval result,
semantic diagnosis, or rejection of CTC generally. No silent exclusion,
zero-infinity suppression, resampling, or alternate temporal head was added.

## Resource accounting and limits

The recorded supervisor interval is6339.263507 seconds (105.654 minutes), within
the7200-second cap. Charge this once against the prior17,876.913-second tranche:
remaining **11,537.649493 seconds**. Extraction-stage wall totals6303.997995
seconds; do not add it again to the supervisor charge. The maximum recorded
CUDA allocation peak is1,155,879,936 bytes, below20GB. This is not an independent
continuous VRAM-reservation measurement. No new GPU work ran during this audit.

No TEST data were opened by extraction or this verification according to the
recorded extraction scope and inspected auditor. Raw MP4 hashes were not
recomputed; all native pose preprocessing was not rerun; no complete feature
extraction was repeated. The report does not establish numerical equivalence
to the deleted cache or official SEDS release features. These remain the
registered **adapted** features, with their original resource limitations.

## Reproducibility and validation

Commands, working directory `/home/haipd/SLR`:

```bash
/home/haipd/miniconda3/bin/python docs/codex_slret_research/tools/verify_c27_recovery.py --output docs/codex_slret_research/evidence/c27_recovery_verification_20260922_v2.json
/home/haipd/miniconda3/bin/python -m pytest -q docs/codex_slret_research/tools/test_verify_c27_recovery.py research/slret_dataset_first/tests/test_c27_gloss_ctc.py
```

The final audit exits0; seven focused fixtures pass. The earlier
[v1 result](c27_recovery_verification_20260922.json) is retained with its original
script hash and fewer native checks. An intermediate audit incorrectly compared
re-serialized pickle bytes, producing a false label-change alarm. Direct
dictionary/ordered-key checks found identical values; the auditor now compares
content, retains source-file hashes, and tests changed/unknown labels. This was
an audit-code correction, not a dataset correction. The failed intermediate
attempt wrote no v2 result. No extraction retry occurred.

Reproducibility verdict for full extraction: **CANNOT_VERIFY by rerun — not
repeated**. The narrower artifact integrity and native spot checks are directly
measured. No full-reproduction claim follows from their success.

Statistical fallacy scan:11/11 applicability checks. Simpson, ecological,
Berkson, collider, base-rate, regression-to-mean and reverse-causality analyses
are N/A: no sampled association or intervention effect was estimated.
Survivorship: all official TRAIN/DEV IDs retained, including failures.
Look-elsewhere: a prespecified necessary feasibility inequality, no p-values.
Forking paths: expanded native checks and the corrected audit comparison are
disclosed above. Causality: structural head rejection is not interpreted as a
cause of retrieval errors. No significance, power or generalization claim.

## Decision

Accept the recovered artifacts for scoped diagnostics. Do not launch C27
training: broader C27 remains **OPEN**, not SUPPORTED-FOR-PILOT, with both this
technical incompatibility and the previously unresolved scientific admission
gates. Fixing13 length cases would not establish novelty or efficacy.
Human annotation remains excluded. The useful result of this checkpoint is
restored, checked inputs plus a falsified assumption about all-TRAIN CTC
eligibility—not a new method or a SOTA result.
