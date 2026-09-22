# Published comparison contracts — Cycle17

2026-09-23. AI-assisted academic-research-suite fact-checking, inline. No
independent review or model reproduction. Scope: primary-source verification
of11 sentence-retrieval result rows and the resource boundaries necessary to
interpret them. This does not certify a complete September2026 leaderboard.

The [machine table](published_frontier_20260923.csv) contains all six recalls,
source versions, table locators, split, evidence class and the contract IDs
below. All results remain **[A] AUTHOR CLAIM**. Reading a published TEST result
does not mean evaluating or opening local TEST data. No local TEST was accessed.
The older project-wide CSV is preserved unchanged.

Structural validation passed:11 unique method/version/dataset/split keys,
bounded monotone recalls, explicit AUTHOR_CLAIM/TEST labels;10 numeric rows
match the historical CSV and the added SAN-GFSLT row matches the inspected
Table1 transcription. This checks transcription/schema, not scientific validity
or model reproducibility. `git diff --check` also passed.

## Resource cards

**SEDS.** ACM MM2024; author v1, Tables1–3 and §4.1. Frozen I3D/BSL-1K RGB,
online GCN pose, SignBERT initialization, CLIP ViT-B/32 interaction/text encoders.
Text-video and pose-RGB contrastive objectives. B128,200 epochs. Reported
TRAIN/DEV/TEST pairs: PH7096/519/642; CSL18401/1077/1176;
H2S31019/1738/2348. Pose:yes; supervised gloss targets in the described
retrieval objective:no; donor label ancestry remains separate. English rendering,
multi-positive implementation and checkpoint selector:U in these passages.
[Paper](https://arxiv.org/html/2407.16394v1).

**C2RL.** Author v1,2024; §III-A–C, IV-A–C, TableVI. ImageNet ResNet18/temporal
convolution, contrastive+autoregressive pretraining; offline visual features,
two independent mBART-large-cc25 encoders and CLCL retrieval. No pose/gloss
targets described. Downstream:80 epochs, eight3090s, B16/process. Reported pairs:
PH7098/519/642; CSL18401/1077/1176; filtered H2S31085/1739/2348.
Multi-positive handling, final selector and exact language-rendering parity:U.
TEST ablations are reported; this alone does not establish the actual selector.
[Paper](https://arxiv.org/html/2408.09949v1).

**UPRet.** Author v1,2024; §3.6, §4.1, Tables1–3. I3D/BSL-1K domain-agnostic
plus adapted encoder, CLIP ViT-B/32. Distribution modeling/transport auxiliary;
transport is training-only. Adam1e-5,B512,200 epochs,fourA100s. Pose/gloss
targets absent from the described retrieval objective; donor ancestry separate.
Reported pairs: PH7096/519/642; CSL18401/1077/1176; H2S31164/1740/2356.
Actual retained H2S gallery IDs, positive grouping, text-rendering parity and
published selector:U. [Paper](https://arxiv.org/html/2405.19689v1).

**SAN.** ACL2026; §4.1 and Table1. Both CSV rows are **coarse** retrieval,
not the separate original-caption-plus40-generated-negatives stress task.
PH7096/519/642. CiCo and GFSLT-VLP variants use CLCL plus a hard-caption loss;
the miner uses a trained GFSLT-VLP retriever. SGD.01,100 epochs; B256(CiCo),
B32(GFSLT). Pose/gloss-target use and complete donor ancestry:U for comparison.
Published selector:U. Current local release uses German BERT/mBART and test-loader
selection; it is not a verified exact implementation of both paper rows.
[Paper](https://arxiv.org/html/2607.09263v1),
[existing release audit](BASELINE_MECHANISM_AUDIT.md).

### Availability and fair-comparison classes

| Contract | Code/checkpoints | Local reproduction status | Class relative to current adapted SEDS |
|---|---|---|---|
| SEDS | [Official README](https://github.com/longtaojiang/SEDS) advertises code, RTM/I3D assets and checkpoints; linked payloads not downloaded/rechecked | Adapted local DEV path verified earlier; published numbers not freshly reproduced | Nearest baseline family, but paper preprocessing/resources not automatically matched |
| C2RL | Official SLRet release/checkpoint not located in bounded search; derivative SLT repository is not equivalent | Unverified retrieval reproduction | Different learned representation and language-model resources; frontier reference |
| UPRet | Existing local source audit; partial historical checkpoint only, current availability not re-audited | Not a complete competitive-model reproduction | Resource/protocol matching required; frontier reference |
| SAN | Existing local release audit; miner table provenance/competitive checkpoint unverified | No matched local reproduction | Different supervision/training/stress protocol; coarse reference only |
| CMCM | [Official repository](https://github.com/vddong-zjut/CMCM) landing page lists components, no numeric results table | Complete trained pipeline unverified | Unranked: inaccessible full paper tables must not become zeros or presumed inferiority |

The C2RL journal DOI10.1109/TCSVT.2025.3553052 again failed to resolve through
the browser. Thus these are explicitly author-v1 numbers, not a certified
final-journal transcription. Lack of a located release is search-bounded,
not proof no release exists. No requests were sent to authors.

## Comparison consequences [I], not new model results

- PH: the two leading inspected rows cross by direction. Combining the best
  T2V from one and V2T from the other fabricates a nonexistent configuration.
- CSL: C2RL exceeds SEDS in all six reported recalls, but the resource cards
  do not justify treating that as a same-input causal comparison.
- H2S: SEDS leads C2RL at R1 in both directions; C2RL leads at T2V R5. Equal
  reported TEST counts do not prove equal gallery membership or relevance maps.
- Different reported TRAIN/DEV counts are a protocol warning, not proof of
  leakage or actual sample mismatch. Resolve IDs before a final comparison;
  a paper typo remains possible, particularly PH7098 versus7096.
- A gain on local DEV cannot be subtracted from these published TEST values.
  The local causal control remains a matched adapted-SEDS run, not a paper row.

Do not launch a baseline reproduction merely to fill the table. A concrete
candidate may proceed with the usable local control under the admission gates;
these unresolved external contracts limit future SOTA claims, not all discovery.

## Current-paper inclusion screen

[SignMatch](https://arxiv.org/html/2609.01886v1) abstract and task definition
confirm dictionary-to-continuous visual sign matching, not sentence text-video
retrieval. Its September date does not make its scores the current task's
frontier. This corroborates prior local exclusion, not a newly discovered paper.
SignSeek's earlier task-scope exclusion remains inherited, not reverified here.

Queries executed: `"sign language retrieval" "2026" SignMatch`;
`"sign language retrieval" "2026" benchmark state art`;
`"2609.01886"`; `"10.1109/TCSVT.2025.3553052" C2RL`;
`"C2RL" "github" retrieval Chen Zhou`; and a2026 SLRet query excluding
SAN/SignMatch/SignSeek/Causality title terms. No additional verified standard
sentence-retrieval result entered this bounded screen. A search result on
aiXiv uses a colliding local identifier; authoritative arXiv resolves SignMatch.
No conclusions rely on the unrelated aggregator record or search snippets alone.

## Verification limits and source assessment

Main sources: four primary computational studies (LevelIII), GradeC for
cross-resource numerical comparison: fitnessC, methodC, dataC, currencyA for
versioned records; peer reviewB for official SEDS/ACL publication provenance,
U for the exact UPRet/C2RL version status; conflicts not fully audited. This is
a provisional claim-fitness assessment, not a paper-quality ranking. Official
publication/repository records supply no specific predatory-source signal;
no independent indexing, funding, retraction or COI clearance was performed.
SignMatch is used only for task definition, not efficacy. No S2 API, external
model, full systematic review, PDF-figure inspection or exhaustive search.

Reading scope is the sections/tables above; not whole papers. Seeds,
confidence intervals, exact gallery/duplicate policies, final version parity
and trained checkpoint identity remain unresolved where marked. CMCM full-table
access remains missing. The frontier audit is therefore **partial**; neither
the new table nor an empty search result certifies SOTA or method novelty.
