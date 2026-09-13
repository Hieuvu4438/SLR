# Method 1 benchmark and claim ledger

Audit date: 2026-09-13. Values below are published **test** R@1 percentages, not
results produced by this repository. They are historical context and broad-system
hurdles; they must not be mixed with Method 1 dev metrics or treated as matched
comparisons.

## Published full-pool results

| System | Resource regime | PH T2V / V2T | How2Sign T2V / V2T | CSL-Daily T2V / V2T | Source status |
|---|---|---:|---:|---:|---|
| CiCo | Domain-agnostic plus domain-aware I3D, CLIP text | 69.5 / 70.2 | 56.6 / 51.6 | 75.3 / 74.7 | Values transcribed from the later primary papers' comparison tables; original [CiCo paper](https://arxiv.org/abs/2303.12793) and [source](https://github.com/FangyunWei/SLRT/tree/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo) pinned |
| UPRet | CiCo feature regime plus probabilistic training | 72.0 / 72.0 | 59.1 / 53.4 | 78.4 / 77.0 | [ECCV 2024 paper, Tables 1–3](https://arxiv.org/abs/2405.19689); repository pinned at [`0463662`](https://github.com/xua222/UPRet/tree/046366227417e1d8ec14145965403462df345984) |
| SEDS | Offline RGB plus trainable pose/SignBERT stream | 76.8 / 78.7 | 62.5 / 57.9 | 85.8 / 85.4 | [ACM MM 2024 paper, Tables 1–3](https://arxiv.org/abs/2407.16394); different modality and initialization resources |
| C²RL | Learned frame-level visual representation with ICL/ECL pretraining and mBART downstream model | 78.7 / 77.6 | 62.4 / 57.5 | 90.3 / 88.4 | [2024 arXiv v1, Table VI](https://arxiv.org/abs/2408.09949); exact equivalence to final TCSVT tables and released retrieval pipeline remains unresolved |
| CMCM | Causal augmentation/alignment and multi-grained pooling | — | — | — | Paper identity and a minimal [author repository](https://github.com/vddong-zjut/CMCM) are verified, but accessible primary material does not expose auditable numerical tables; keep values **UNVERIFIED** ([publisher page](https://doi.org/10.1016/j.cviu.2025.104631)) |

The largest verified R@1 in this audited set is therefore 78.7 in both PH
directions, split across C²RL T2V and SEDS V2T; 62.5/57.9 on How2Sign from
SEDS; and 90.3/88.4 on CSL-Daily from C²RL. This is a bounded audited frontier,
not proof that no newer or inaccessible result is stronger.

## Fine-grained SAN result is a separate protocol

[SAN](https://aclanthology.org/2026.acl-long.1302/) evaluates PH with both the
ordinary full candidate pool and a 41-candidate, automatically perturbed-caption
stress test. In its CiCo branch, SAN changes ordinary T2V/V2T R@1 from
69.2/70.1 to 68.1/67.8 while stress-test V2T R@1 changes from 17.9 to 39.4.
Those stress numbers cannot be inserted into the full-pool table. They motivate
the Method 1 trade-off hypothesis; they do not establish shared support as its
cause.

## Claim gates for this repository

1. **Matched-resource improvement:** compare `span_shared` against the paired
   `span_independent`, `base_continuation`, `caption_hn`, `fsc_local`, and
   `fsc_local_caption_hn` arms using the same seed, manifests, baseline teacher,
   miner, update budget and corrected evaluator.
2. **Mechanism evidence:** require favorable support diagnostics and require
   `span_random_support` not to explain the gain equally well. A decreasing
   auxiliary loss alone is insufficient.
3. **Replication:** report seeds 42/43/44, each seed separately and their mean;
   paired group-cluster bootstrap intervals do not replace seed variation.
4. **Broad system comparison:** published SEDS/C²RL values use materially
   stronger or different representation resources. Beating only the corrected
   UPRet scaffold is not field SOTA. A broad claim needs a faithful resource-aware
   comparison or transfer to the stronger representation regime.
5. **Test boundary:** choose configuration on dev, create the immutable selection
   lock, then evaluate test once. Never compare a running or selected dev score
   with the published test values above.

Current repository result status: no reportable Method 1 comparison or test
metric exists yet. The PH seed-42 corrected baseline is training; How2Sign and
CSL-Daily remain blocked on the resource gaps recorded in `resource_ledger.md`.
