# Local SLRet research: reproduce and resume

This bounded empirical campaign is closed with scientific outcome
INCONCLUSIVE_OR_BLOCKED: no validated improved retriever or novel alternative
contribution. Start with FINAL_HANDOFF.md, STATE.md, RESULTS.md (measurements),
PAPER_CASE.md (claim limits), RUN_BUDGET.md and NO_GO_REGISTRY.md.
Goal specification: `docs/guide/ASTRA6_SLRET_RESEARCH_GOAL.md`.

## Environment and retained evidence

Work from `/home/haipd/SLR`. CiCo and CPU tests use
`/home/haipd/miniconda3/bin/python` (Python3.13, torch2.11.0+cu128 in recorded
runs). Native SEDS uses `/home/haipd/miniconda3/envs/seds/bin/python`
(Python3.10, torch2.3.1+cu121); do not interchange these interpreters casually.
The RGB training bridge uses both environments explicitly. Runs were measured
on the RTX5880Ada. Version changes require parity validation, not relaxed checks.

Initial commit/dirty patch, package inventory and baseline hashes are retained
in `artifacts/slret_goal/inventory/`. These are an initial snapshot, not a claim
that every later dependency is identically locked. Per-run source/config/model
hashes and runtime versions supplement it. Existing user UPRet edits are not
this campaign's patches; do not reset the worktree to reproduce the snapshot.

`environment_snapshot.json` records both interpreters and installed package
versions on2026-09-19 (510base/98nativeSEDS packages). It is a later inventory,
not proof of the exact initial environment or a tested fresh-install lockfile.
COMPLETION_AUDIT.md records the final requirement audit and unresolved scientific
limitations; it does not certify SOTA, publication novelty or paper reproduction.

Artifacts live under `artifacts/slret_goal/<run-id>/`. Preserve run.json, logs,
score matrices, ranks, configs and checkpoints. `experiments.jsonl` is append-only
history, including failures and intermediate entries, not one row per run.
Use the final run.json together with the latest ledger entry; a status string
alone does not prove a process is still alive. Raw data paths are catalogued in
`docs/proposal1/datasets.md`; they are not permission to redistribute the data.

## Safe CPU checks

```bash
/home/haipd/miniconda3/bin/python -m pytest research/slret_goal/tests -q
```

51 tests are expected after the record-audit addition; inspect actual output.
Tests check contracts/mechanics, not benchmark gains. The primary audit scripts
also compare real retained artifacts. Each audit below requires a **new** run ID
and writes its own report/ledger entry; never reuse an existing directory:

```bash
/home/haipd/miniconda3/bin/python research/slret_goal/tools/campaign_record_audit.py --run-id campaign-record-audit-review-001
/home/haipd/miniconda3/bin/python research/slret_goal/tools/cico_numeric_audit.py --run-id cico-pair-review-001 --native cico-numeric-native-001 --fp32 cico-numeric-fp32-001
```

The second command is CPU artifact verification, not fresh model execution.
The completed fresh-process reconstruction of all12 endpoints is recorded in
`cico-checkpoint-replay-001`. Do not relaunch GPU checks while another campaign
or user has the GPU occupied. Source identity drift must be investigated.

## CSL confirmation and replay commands

Update2026-09-19: evaluation and analysis commands below have now completed.
Their named output IDs already exist: DO NOT rerun them or claim TEST remains
unopened. See RESULTS.md and the actual run.json for outcomes. CPU recomputation
with a fresh analysis ID is possible; further inference needs a documented
reproducibility purpose, not selection/tuning.

Input preparation `csl-test-assets-001` completed, all1176 videos in both streams,
fixed batch128 and all feature/provenance/finiteness checks passed. ManifestSHA:
`68e5b7456d10daec1bb332b7d8b6d22deb44571a588a992e9a6e68f1458078ce`.
CPU lock `cico-csl-test-lock-001` completed, binding all10 distinct models.
LockSHA: `88151788467a2130a2b0c2f2b79650dc9e8986d226ec442bcda7704ff25b74c6`.

Before evaluation, read current STATE, verify no existing evaluation run/job,
GPU availability and disk reserve; inspect the launch/run reports if present.
Do not duplicate a job just because a prior tool session expired. The registered
evaluation command below is retained as historical provenance, **already executed**:

```bash
/home/haipd/miniconda3/bin/python research/slret_goal/tools/launch_bounded.py --name cico-csl-locked-test-001 --seconds 300 -- /home/haipd/miniconda3/bin/python research/slret_goal/tools/cico_locked_csl_test.py --run-id cico-csl-locked-test-001 --mode evaluate --lock /home/haipd/SLR/artifacts/slret_goal/cico-csl-test-lock-001/lock.json --lock-sha256 88151788467a2130a2b0c2f2b79650dc9e8986d226ec442bcda7704ff25b74c6
```

Only after all10 models finish successfully:

```bash
/home/haipd/miniconda3/bin/python research/slret_goal/tools/cico_csl_test_analysis.py --run-id cico-csl-test-analysis-001
```

No TEST-selected seed/checkpoint, retuning, altered translation or relevance.
795/798 CSL TEST caption IDs overlap DEV; this is held-out-video evidence, not
novel-caption generalization. Full protocol is CICO_LOCKED_CSL_TEST_PROTOCOL.md.
The PH TEST was already opened once under its own immutable lock. Its selected
accuracy delta is0 despite the positive endpoint precision effect.

## Historical record caveats

Final `campaign-record-audit-002` inspected 81 preceding reports: 71 completed,
10 failed, zero running, zero terminal-status mismatches with the ledger. It
retains the same nine schema/summary warnings and six absent command fields
described below. The audit excludes its own report by construction.

`campaign-record-audit-001` inspected77 run reports:66completed,10failed,
1then-running. All76 terminal statuses agreed with the ledger. Nine records had
schema/summary differences. Two failed startup records use different explanatory
wording (`pose-parity-l384-001`, `seds-adapted-dev-eval-001`); inspection confirms
both are pre-inference failures, not contradictory successful results. Seven
other differences are artifact-location fields. Preserve both histories.

Six reports omit a command field. Reproduction routes below are reconstructed
from present entrypoints, **not claims of recovered original invocations**:

| Report | Present executable route / caution |
|---|---|
| cico-ph-test-analysis-001 | tools/cico_ph_test_analysis.py --run-id NEW; hashes frozen inputs and rereads already-opened PH scores |
| seds-moment-audit-001 | tools/seds_moment_audit.py --run-id NEW; CPU stored-state census |
| train-extraction-stop-audit-001 | tools/check_extraction_stop.py --run-id NEW; current extraction is complete, so an earlier partial-state result cannot be recreated from present files |
| csl-score-contract-001 | tools/csl_score_audit.py has a fixed existing output ID; do not rerun as-is |
| secondary-dev-assets-001 | tools/secondary_dev_assets.py has a fixed existing output ID; do not rerun as-is |
| seds-fusion-layout-cpu-001 | native SEDS Python tests/test_seds_fusion_layout.py executes the retained independent reference/gradient tests; original inline report command was not recorded |

## Retention and claim boundaries

No artifact deletion is required. Campaign cap44GiB, free reserve15GiB; do not
start checkpoint-producing jobs without explicit admission. Some intermediate
SEDS/RGB selected checkpoints are inference-only; final-state retention does
not by itself implement or authorize resume. CiCo final.pt stores optimizer,
scheduler, RNG and sampler state; this study starts its paired arms with fresh
optimizer states from the same historically selected model.

Do not call standard FP32 moments a new optimizer, endpoint recovery a selected
model gain, or these adapted/transfer-aware protocols author-paper reproduction.
Final scope decision and handoff are in COMPLETION_AUDIT.md and FINAL_HANDOFF.md.
