# Repository census and current evidence boundaries

2026-09-22. Phase A: read-only reconstruction plus isolated audit utilities.

## Census

[M] All requested top-level directories exist: `docs`, `methods`, `shared`,
`configs`, `scripts`, `tests`, `patches`, `artifacts`, `runs`, `exports`,
`third_party`. The [machine census](evidence/state_audit_20260922.json) records
immediate source-directory contents and selected source hashes. It deliberately
does not traverse active job directories or inspect processes.

| Area | Inspected role |
|---|---|
| `docs/proposal1`–`proposal6` | Historical ELSC, DIVE, PLEL, OCEM, SSSC/partial alignment, PMGR/RPCA families; reconstructed via proposal7 registry and surviving gates. Complete direct rereading remains pending. |
| `docs/proposal7` | Extensive audits, 37-question map, AS-C experiments, negative registry and source/protocol reviews. Old final NO-GO is not completion of the present objective. |
| `docs/proposal8`, `docs/proposal9` | Present; proposal9 supplies current authority. Proposal8 semantic review pending. |
| `methods/` | `elsc`, `dive`, `ocem`, `sssc`, `pmgr`, `information_probe`, `seds_adaptation`, `sl_mvr`, `translation_retrieval`, `gloss_bridge` |
| `shared/slr_common/` | Data, feature provenance, upstream bridges, gallery encoding, evaluator and resource helpers |
| `research/slret_goal/` | Baseline replay/continuation, published-results CSV, exclusion registry, bounded runner |
| `research/slret_goal_v2/` | Later pose/graph/fusion controls and candidates; retained metric artifacts under `artifacts/slret_goal_v2/` |
| `research/slret_dataset_first/` | Current C25–C27 work, structure analysis, donor bridge, recovery supervisor |
| `third_party/` | SLRT/CiCo, SEDS, UPRet, SAN, CMCM, Uni-Sign, UniFormerV2. No top-level official C²RL checkout found. |

No applicable `AGENTS.md` was found at the checked ancestor paths or in the
queried proposal trees. This is not a claim that every filesystem path was read.

## Existing changes preserved

At entry, modified files were dataset-first `STATE.md`, V1 `experiments.jsonl`,
and `seds_adapted_features.py`; untracked work included proposal9, gloss bridge,
C27 decision/tests/recovery supervisor. No training code or third-party source
was changed by this cycle. New files are confined to this audit directory plus
an additive C27 decision note.

## Directly traced interfaces

- `shared/slr_common/evaluation/cico_eval.py`: video×text orientation;
  singleton T2V tie expansion and V2T double-argsort differ. Flat best-positive
  relevance does not automatically equal max-over-caption-group T2V. A CSL
  rectangular gallery cannot use a square-diagonal-only evaluator.
- `third_party/SEDS/modules/modeling.py`: ordered pose windows → contextual
  pose/RGB tokens → fusion → two directional token scores → matrix losses.
- `third_party/SEDS/dataloaders/dataloader_ph_retrieval_train_pose.py`:
  `_get_text` supplies caption ID; `_get_rawvideo` chooses a video path;
  `__getitem__` returns tensors without an explicit sample-ID field. A CTC
  adapter must join the **chosen video** to its TRAIN gloss, not assume batch
  position identifies a video. PH's current single-video entries reduce the
  immediate risk; CSL grouping makes the general contract important.
- `methods/gloss_bridge/ctc.py`: TRAIN-only linear head over contextual pose
  tokens; excludes CLS, checks adjacent-repeat CTC feasibility, removes targets
  after training forward. No complete trainer or loss-weight integration is
  present in this namespace. Existing unit tests cover finite loss/gradient,
  invalid length and out-of-vocabulary cases, not real-model baseline parity.

## New utility and verification

`tools/audit_research_state.py` hashes retained historical gate files and selected
code, measures PH TRAIN gloss structure, and writes an exclusive-create JSON
report. It never loads checkpoints, features, DEV/TEST annotations or active
job files. Two fixtures test multiset multiplicity/order distinction and invalid
annotations. These plus the three existing CTC tests passed: **5 passed**.

Cycle 2 additionally read the complete proposal7 main proposal, Phase2 empirical
bottleneck report and 37-question feasibility audit. Cycle3 completed the long
autonomous state (1,985 lines), research loop (1,344) and literature log (481),
plus complete candidate record, progress ledger, implementation audit, final
review and negative registry. Truncated sections were reread in smaller chunks.
All ten proposal9-named proposal7 history files have now been read completely
across these cycles. Earlier proposals1–6 remain reconstructed from the fully
read registry/progress ledger, their recorded source audit and surviving gates;
do not claim a fresh complete rereading of every original proposal/specification.
Historical instructions/status snapshots are not current job authority.

The new CPU-only residual diagnostic replays nine retained score matrices and
checks exact ranks/metrics; three synthetic fixtures pass. See
[protocol](evidence/GCN_RESIDUAL_PROTOCOL.md) and
[results](evidence/GCN_RESIDUAL_RESULT.md). It does not inspect the active job.

Missing verification remains explicit: broader original-specification coverage, current
complete baseline manifests, all upstream ancestry pins, checkpoint bytes and
data restoration, fresh-inference failures of the strongest control, and a
method that survives controlled falsification.
