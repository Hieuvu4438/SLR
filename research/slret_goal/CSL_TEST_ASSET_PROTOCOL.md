# CSL TEST input preparation: fixed-recipe feasibility

2026-09-18, after PH locked TEST was analyzed, before any CSL TEST score.
This is a second-dataset confirmation feasibility phase, not a PH TEST-informed
training/configuration change. The existing six CSL endpoint models (240 steps)
and their DEV-selected checkpoints are immutable. No new training is planned.

Read-only census found all 1176 official TEST raw videos, but neither stream's
features, translated queries nor a manifest. The user-provided dataset catalogue
identifies their location. Existing train/dev extraction used BSL5K and H2S
transfer-aware I3D weights, dense 16-frame/stride1 windows and batch128; DEV
English queries used pinned Helsinki-NLP/opus-mt-zh-en with beam4. Reuse these
exact processing choices, without manual translation or quality-based exclusions.

First bounded probe (<=300s, <=16MiB, charged to confirmation):

1. Hash all translator weights against the existing DEV translation artifact.
2. Re-extract the first official DEV-manifest video with each existing I3D
   checkpoint, batch128, same extractor/recipe. Require bit-exact features and
   no OOM-driven batch change; compare temporal metadata and provenance.
3. Re-translate the first 64 DEV caption groups as one original-size batch.
   Require exact string equality with stored DEV translations.
4. Census TEST IDs, caption groups, video overlaps with TRAIN/DEV, total windows
   and projected feature storage. No TEST translation, feature forward or
   retrieval scores in this probe. Caption overlap is reported, not filtered.

First DEV item/batch is predetermined by manifest/translation order, not chosen
from outcome. Failure is retained and investigated; no tolerance relaxation or
alternative translation recipe. Success admits input-preparation planning only,
not accuracy or contribution claims. The full TEST extraction requires a new
explicit wall/storage admission based on the measured projection. Historical
agnostic train/dev throughput was 48210.308s/19478 videos with concurrent stream
extraction; this is only a planning reference, not a guaranteed TEST runtime.

Keep all new files inside a new artifacts/slret_goal run directory. Preserve all
existing train/dev/PH TEST inputs. Campaign cap44GiB, free reserve15GiB. A later
CSL TEST score lock must fix all endpoints, initialization and all distinct
DEV-selected models, grouped gallery/relevance, complete metrics and bootstrap
before scoring. No best-TEST model/seed selection. Do not describe official
video-ID disjointness as caption/source/signer/pretraining disjointness.

## Full input-preparation admission (after probe, before TEST forward)

Probe001 passed in 11.955257s: both193-window DEV feature arrays bit-exact,
batch128 unchanged; all64 translations exact. TEST census:1176 videos,
798 caption groups,185718 dense windows. There is zero TRAIN/DEV video-ID
overlap and zero TRAIN caption-ID overlap, but **795/798 TEST caption groups
also occur in DEV**. Thus this extension can confirm held-out-video behavior
under the native split, NOT novel-caption generalization or an independent
query dataset. Preserve the official split; do not remove shared captions.

Admit csl-test-assets-001 with a9000s hard wall bound, charged to the6h
confirmation reserve. Before it, confirmation used38.543181s; full bound
would stay below9038.543181s/21600s. Dense dual-stream float32 payload is
1521401856 bytes (1.417GiB),1.559GiB with10% overhead; reserve2GiB total.
Campaign~40GiB+2GiB<44GiB; filesystem125GiBfree,15GiBreserve. No new checkpoints
or deletion. Historical dual-stream window-scaled wall projection~6210s,
with9000s hard margin. Smoke kernel times are faster but exclude decoding/I/O;
do not use them alone for admission. Separate GPU streams run sequentially.

Use the exact shared extractor, batch128 and checkpoint hashes verified by
probe. Any OOM-induced batch reduction makes preparation invalid for score
admission; do not silently accept a changed arithmetic path. Translate all798
groups with the pinned DEV translator/generation settings in official first-seen
order, with no manual repair or filtering. Save source/translation hashes and
all feature sidecars/temporal metadata. Build a new manifest under this run only.
Validate all1176 records, both stream shapes/finiteness/hashes/provenance and
temporal shape/recipe agreement. No retrieval evaluation in preparation.

On timeout/failure retain every partial artifact and report the exact cause.
No automatic retries, bounds extension or TEST retrieval until final preparation
audit and the separate fixed-model evaluation/analysis lock are complete.
