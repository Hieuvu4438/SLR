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
| CLIP ViT-B/32 init | local file `artifacts/pretrained/ViT-B-32.pt` | Present | Permitted initialization; checksum gate pending Method 1 audit command |
| UPRet trained checkpoint | per-seed corrected baseline | Missing | Must be trained and selected on dev before reference caches |

The UPRet repository snapshot contains no visible license file. Source adaptation remains
attributed and patch-based; redistribution/license compatibility is unresolved and must be
reviewed before public release.
