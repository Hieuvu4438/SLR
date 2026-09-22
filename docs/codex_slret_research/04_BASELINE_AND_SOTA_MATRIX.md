# Baseline and frontier evidence — provisional

Updated2026-09-23. Never compare a local DEV score to published TEST as a method gain.
No current SOTA certification is made. This matrix is incomplete until protocol,
source, resource and current-literature audits are finished.

## Locally recorded results

| Configuration | Dataset/gallery | T2V R1 | V2T R1 | Mean R1 | Evidence and fair-use boundary |
|---|---|---:|---:|---:|---|
| CiCo local replay | PH DEV519 | 74.181118 | 76.300578 | 75.240848 | Historical reproduced result in V1 state; agnostic + H2S-transfer-aware features. Old replay artifacts not revalidated here. |
| CiCo local replay | CSL DEV1077 videos /797 caption groups | 68.883312 | 65.923863 | 67.403588 | Historical V1 result; English text and specific grouped/flat relevance contract require named evaluator. |
| SEDS adapted release initialization | PH DEV519 | 76.493256 | 78.612717 | 77.552987 | Historical V1 result; adapted RTMPose/RGB inputs, not released-preprocessing parity. |
| SEDS GCN-R1, seed1337, selected step222 | PH DEV519 | 77.456647 | 79.961464 | 78.709056 | [V] Current retained run and metric JSON read; strongest recorded practical incumbent, not paired C27 causal control. |
| C26-A frozen Uni-Sign global head | CSL DEV1077/797 | 44.040151 | 43.361188 | 43.700670 | Historical dataset-first result; old job artifacts reported removed. Donor/text resources differ from CiCo. |

Incumbent sources:
`artifacts/slret_goal_v2/seds-gcn-horizon3-seed1337-001/run.json` and
`eval_step0222/fusion_metrics.json`. Retained run reports completed 666 updates,
seed1337, B32, GCN LR 1e-6, fusion LR 1e-5, native objective, DEV-only selection,
`test_loaded=false`. Metric file reports T2V R5/R10 93.256262/96.146435 and
V2T R5/R10 93.256262/95.183044. Reading these files is not fresh replay or
rehashing the checkpoint/features. Multiple DEV selections make this exploratory.

## Published frontier

Cycle 2 saved-score verification: all nine selected fusion/pose/RGB matrices
from seeds42/1337/2026 reproduce stored recalls, rank summaries and query ranks
exactly. Selected bidirectional mean R1 across seeds is **78.612717**, sample
SD **0.096339** percentage points. Selected steps differ (666/222/222), so this
is descriptive incumbent variation, not a new method's controlled multi-seed
gain. No checkpoint forward or TEST access occurred. See
[residual report](evidence/GCN_RESIDUAL_RESULT.md).

The existing detailed machine table is
[`published_results.csv`](../../research/slret_goal/published_results.csv).
Its CiCo/SEDS/UPRet/C²RL/SAN/earlier retrieval rows are **[A]** and retain paper
versions, source URLs, all six recalls and comparability notes. Do not silently
promote them to locally reproduced results or current frontier certification.
Current-turn verification is limited to the source passages listed in
[collision audit](08_PRIOR_ART_COLLISIONS.md). Required full frontier audit:

| Family | Material comparison boundary |
|---|---|
| Free-form retrieval / SPOT-ALIGN | Video backbone, recognition-score ensemble versus embedding-only scores, language/text encoder |
| CiCo | Target-domain versus transfer-aware feature ancestry and pseudo-label resources |
| SEDS | Extra pose stream/pretraining, adapted versus released coordinates, H2S retained populations |
| UPRet | Distribution/PDE objective; a partial local checkpoint cannot represent paper strength |
| C²RL | Different visual/text pretraining and translation supervision; official code availability unresolved |
| SAN | Coarse full-gallery versus constructed fine-grained stress protocol; mining resources and test selection |
| CMCM | Published source verified to exist; trained integration/resources unresolved; no invented missing metrics |
| Newer work | September 2026 SignSeek is dictionary retrieval, not a sentence-level frontier replacement. CSLR² is a relevant joint-training prior with BOBSL protocol, not PH/CSL comparator. |

For each eventual numeric frontier row, complete: venue/year, all recalls,
backbone/feature source, pretraining/extra data, pose/gloss/teacher, text encoder,
loss, gallery/positives, DEV selection, code/checkpoint availability, reproduction
status and fair-comparison class. Missing fields remain [U]; table existence is
not proof this requirement is finished.

## Cycle17: versioned numerical frontier and resource contracts

The [current primary-source snapshot](evidence/published_frontier_20260923.csv)
verifies11 published rows against their own result tables, including both SAN
backbones. [Resource cards and read scope](evidence/FRONTIER_CONTRACTS_20260923.md)
record architectures, pretraining, extra supervision, objectives, data counts,
availability, selection uncertainty and fair-comparison classes. This replaces
generic `pending_full_resource_protocol_lock` with specific known/unknown fields
for these rows; it does not declare those contracts fully matched.

All values below are **author-reported TEST percentages**, not local results.
Each triplet is R1/R5/R10. Sources and versions are in the snapshot and cards.

| Method | Dataset | T2V | V2T |
|---|---|---|---|
| SEDS | PH | 76.8/91.7/95.3 | 78.7/92.5/95.2 |
| C2RL | PH | 78.7/92.2/94.9 | 77.6/91.3/94.2 |
| UPRet | PH | 72.0/89.1/94.1 | 72.0/89.4/93.3 |
| SAN-CiCo, coarse | PH | 68.1/87.4/91.7 | 67.8/87.4/91.7 |
| SAN-GFSLT, coarse | PH | 70.2/89.3/94.4 | 67.4/85.4/90.5 |
| SEDS | CSL | 85.8/94.4/95.6 | 85.4/93.8/95.8 |
| C2RL | CSL | 90.3/96.4/97.7 | 88.4/95.7/97.1 |
| UPRet | CSL | 78.4/89.1/92.0 | 77.0/89.2/92.7 |
| SEDS | H2S | 62.5/75.1/80.1 | 57.9/70.4/74.9 |
| C2RL | H2S | 62.4/75.9/80.1 | 57.5/68.4/73.0 |
| UPRet | H2S | 59.1/71.5/75.7 | 53.4/65.4/70.0 |

These are reference rows, not a certified complete leaderboard. CMCM remains
unranked without verified numeric tables. CiCo/free-form/scaling/VAP rows retain
their historical evidence status in the original CSV. A componentwise envelope
must not be presented as one trained system. The matched local SEDS control
remains usable without waiting for a final frontier certification.

## C27 comparison contract (unchanged)

Cycle3 source clarification: UPRet's current native entry point and SAN's
released trainer select via configured test loaders; neither should be run
unchanged for this DEV-only program. C²RL derivative SLT code and incomplete
CMCM components are not verified trained retrieval controls. These are bounded
release-code findings, not allegations about the authors' actual experiments.
The [mechanism audit](evidence/BASELINE_MECHANISM_AUDIT.md) also preserves
SAN's constructed-stress versus standard-gallery distinction and the current
vendor ancestry limitation. Missing comparator assets do not invalidate the
usable local SEDS control or justify a replacement reproduction campaign.

CTC uses expert TRAIN gloss annotations. Therefore compare first to an otherwise
identical continuation with zero auxiliary weight, and then to a same-gloss,
same-parameter order-free auxiliary. The existing best GCN-R1 is a practical
target. Published gloss-free methods are resource-mismatched references. Beating
the release initialization alone would not establish improvement over the
strongest fair local baseline.
