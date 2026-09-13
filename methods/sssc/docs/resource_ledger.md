# Method 1 resource ledger

Audit date: 2026-09-13.

| Resource | Identity | Local status | Method 1 use |
|---|---|---|---|
| UPRet | commit `046366227417e1d8ec14145965403462df345984` | Audited checkout at ignored `third_party/UPRet` | Baseline source; no trained checkpoint supplied by repository |
| CiCo / SLRT | commit `38a4f7b00da7a858d59b7fabe5093876a84db8e0` | Present | Feature provenance and compatibility reference |
| PHOENIX-2014T raw data | 7,096/519/642 official records | Present at `/home/dongvk/datasets/phoenix14T` | Membership authority only; no new annotation |
| UPRet PH English captions | 7,096 train, contaminated 7,615 dev container, 642 test | Present in pinned checkout | Must filter dev using official 519 keys |
| PH domain-agnostic I3D | Oxford BSL5K-derived cache | Present under ignored artifacts; provenance already audited by proposal 1/OCEM | Candidate stream; verify hashes again for Method 1 manifests |
| PH domain-adapted I3D | Locally trained P14T stream | Present under ignored artifacts | Candidate aware stream; not yet proven equivalent to a published CiCo release |
| CLIP ViT-B/32 init | SHA-256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` | Present at `artifacts/pretrained/ViT-B-32.pt` | Permitted initialization; passed Method 1 input audit |
| UPRet trained checkpoint | per-seed corrected baseline | Missing | Must be trained and selected on dev before reference caches |
| How2Sign annotations | UPRet train/test plus audited realigned sources | Partial | Full CiCo I3D agnostic/aware train/dev/test features are absent; raw-video audit also reports 118/2/6 missing train/dev/test clips |
| CSL-Daily English annotations | UPRet translated-English train/test; local translated train/dev assets | Partial | Local agnostic/aware features cover train/dev only; established UPRet test features are absent and aware-stream publication equivalence is unresolved |

Reference-cache and miner commands are implemented but remain deliberately unexecuted on real
PH data until that dev-selected checkpoint exists. Their immutable identities include the
teacher, tokenizer, caption/video manifests, mixture/sampler semantics, and implementation
version; the mining table and negative-span vectors are published together as one atomic
auxiliary bundle.

The UPRet repository snapshot contains no visible license file. Source adaptation remains
attributed and patch-based; redistribution/license compatibility is unresolved and must be
reviewed before public release.

The H2/CSL confirmation configs intentionally point at their expected isolated Method 1
resource locations. `audit --stage input` fails closed while those split assets are missing;
no random dev split, duplicated video, copied feature, or fabricated metric is used to make
the confirmation datasets appear runnable.
