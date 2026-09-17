# CMCM: trained-activation resource gate

## Verification scope

2026-09-16; academic-research-suite / deep-research fact-check, inline.
AI-assisted primary repository inspection. This checks the Q35 prerequisite,
not retrieval efficacy, publication quality or global research exhaustion.
The preceding Q01 turn was progress, but supplied no method candidate.

## Finding

**Neither currently advertised CMCM branch supplies the trained checkpoint and
retrieval integration needed for the proposed activation-exposure measurement.**
The alternate `main` branch is not a fuller implementation: it contains only a
97-byte README naming the paper. The default `master` branch remains the exact
revision used in the numerical covariance audit. The repository's public tags
and releases endpoints return empty lists.

This closes a concrete resource lookup, not the hypothesis that the covariance
discrepancy might matter. It does not prove that compatible assets do not exist
privately, in deleted history, or at an unlinked host.

## Primary-source observations

Live, unauthenticated read-only GitHub API requests all returned HTTP200:

| Endpoint under `https://api.github.com/repos/vddong-zjut/CMCM` | Observed content | Response SHA256 |
|---|---|---|
| Repository metadata | default `master`; homepage null; not archived; pushed_at `2025-09-12T03:06:52Z` | `dc21be6832f8e1e75f7f8426d052b5b3382b9e4c0598a9ad89811a59a85bc9a7` |
| `/branches?per_page=100` | `main` at `92ec8470d5c62c44518f3e9825ba54cc837026fb`; `master` at `5d458719d1da2f082e188cc44705003d919e7e97`; no pagination Link header | `89dc01e0cff087173ffacbbdfc9881b835bc33d6cd500d26e78a3fa5896f678f` |
| `/tags?per_page=100` | empty list; no pagination Link header | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| `/releases?per_page=100` | empty list; no pagination Link header | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| `/git/trees/5d458719d1da2f082e188cc44705003d919e7e97?recursive=1` | 20 blobs, truncated=false; Python modules/datasets/utilities and six tracked dataset bytecode files; no README, checkpoint, trainer/config or evaluator file | `a7bdf03b1479e2d839d65efb7a40161cc59cbf4ed1ffadc1038ea4036e5ca2e7` |
| `/git/trees/92ec8470d5c62c44518f3e9825ba54cc837026fb?recursive=1` | only README.md, truncated=false | `5a9b94e6a50b01304cec52863cbad60eb9c0c6ef78189d3d2bab18b27bf909f5` |

The [pinned main README](https://github.com/vddong-zjut/CMCM/blob/92ec8470d5c62c44518f3e9825ba54cc837026fb/README.md)
was read completely through raw.githubusercontent.com, HTTP200, 97 bytes,
SHA256 `01895e6cd94cdeb30d908b95d93126068c212044576e88bf201eafcb34e8d8fa`.
It contains the repository/paper identity and no resource link.

The [repository browser page](https://github.com/vddong-zjut/CMCM) agrees on the
default branch and module directories. Its branch-page rendering exposed only
the default entry, so it was not used to infer that only one branch exists.
The browser release fetch failed with a cache miss; this was resolved by the
successful API request, not interpreted as an empty release list by itself.
The local `git ls-tree` agrees with the pinned master file inventory. API/browser/
local Git are cross-checks of the same author-controlled source, not independent
evidence of scientific performance.

Two targeted searches were used: the repository's paper-title string plus
`checkpoint`, and `"vddong-zjut" "CMCM" weights`. The
[publisher preview](https://www.sciencedirect.com/science/article/pii/S1077314225003546)
points to this same repository. No separately verified checkpoint location was
found in those results. This is a bounded search, not a web-wide absence proof.
Secondary search hits were not used to establish resources or efficacy.

## Why available generic weights are not the missing resource

The full [Encoder.py](../../../../third_party/CMCM/modules/Encoder.py) and
[TMCP_Module.py](../../../../third_party/CMCM/modules/TMCP_Module.py) were read.
`VideoEncoder(pretrained_i3d_path=None)` does not consume that argument; its
named `i3d_encoder` is instantiated as torchvision `r2plus1d_18(pretrained=True)`.
This source-level naming/loading fact is not evidence that a CMCM-trained
checkpoint has been supplied or loaded. No torchvision import, download or
execution was performed here.

TMCP constructs learned reduction convolutions, batch-normalization modules and
attention before covariance pooling. Its forward calls covariance, three-step
matrix square root and triangular vectorization. Measuring raw CiCo features,
generic torchvision outputs or newly initialized TMCP tensors therefore would
not measure the requested **trained CMCM pre-covariance activation distribution**.
The existing [covariance result](CMCM_source_result.md) remains a synthetic
derivative check, not measured exposure in the published model.

A targeted source search for URLs, checkpoint/load calls, pretrained paths,
optimizer or main entry points found the generic encoder constructor and an
MPN-COV paper URL, but no CMCM checkpoint loader. This is corroborating code
inspection, not a claim that lexical search proves all runtime behavior.
Previously recorded missing imports/shape/mask concerns were not reclassified
as new findings or repaired.

## Resource requirement and decision

To make Q35's next measurement attributable, a supplied package would need:

1. CMCM-trained state including the pre-covariance transformations and BN state,
   with architecture/config and checkpoint provenance.
2. The actual input preprocessing and integration path identifying which tensors
   enter TMCP and how training/evaluation mode is used.
3. A permitted TRAIN-only invocation or recorded TRAIN activations linked to that
   checkpoint. Activation scale measurements alone would still not establish
   retrieval harm; a matched control would be a later gate.

None is verified by the current repository branches/releases. Do not acquire
generic weights as a substitute, assemble a replacement CMCM pipeline, or start
a repair-training campaign. No author contact or external write was made or
authorized. A resource request to the authors would require the user's direction.

Q35 is now specifically **resource-dependent after branch/release verification**.
Do not repeat this check without a changed release or supplied artifact. All other
open questions retain the narrower statuses in the all37 audit; this missing
resource does not make the entire GO search blocked. The optional question about
closed-family reconsideration remains unanswered, and all closures remain binding.

## Evidence grade and limits

Repository/API metadata and source: Level VI descriptive primary evidence,
high fitness for these exact public-resource observations; no peer-reviewed
performance inference. Author control of the resource is explicit. Publisher
preview is used only for the repository pointer; full paper, detailed numerical
results, venue indexes, author profiles, retraction databases and independent
performance replication were not rechecked. No misconduct or venue-quality
claim follows from an incomplete public package.

No checkpoint/assets, data/TEST files, issue bodies, model execution, DEV scores,
training, dependency installation, upstream edits, pull/reset/commit/push.
Existing untracked module caches were preserved. No new candidate, Q38 or GO.
