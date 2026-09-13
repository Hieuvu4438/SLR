# Method 1 resource ledger

Audit date: 2026-09-13.

| Resource | Identity | Local status | Method 1 use |
|---|---|---|---|
| UPRet | commit `046366227417e1d8ec14145965403462df345984` | Audited checkout at ignored `third_party/UPRet` | Baseline source; no trained checkpoint supplied by repository |
| CiCo / SLRT | commit `38a4f7b00da7a858d59b7fabe5093876a84db8e0` | Present | Feature provenance and compatibility reference |
| PHOENIX-2014T raw data | 7,096/519/642 official records | Present at `/home/dongvk/datasets/phoenix14T` | Membership authority only; no new annotation |
| UPRet PH English captions | 7,096 train, contaminated 7,615 dev container, 642 test | Present in pinned checkout | Must filter dev using official 519 keys |
| PH domain-agnostic I3D | Oxford BSL5K-derived cache | Present under ignored artifacts; provenance already audited by proposal 1/OCEM | Locally re-extracted train/dev/test stream; test rows closely match the CiCo release, but the complete release contains no train/dev files |
| PH domain-adapted I3D | Locally trained P14T stream | Present under ignored artifacts | Locally re-extracted train/dev/test aware stream; it is not equivalent to the CiCo test release and therefore defines a qualified local-resource regime |
| CiCo sign-feature archive | Google Drive file `1Vb-HFZd-rhjN49sB5WwLRpIbyhiC6xTy`, SHA-256 `9ba1956cf416df9a31ae3d1a71a3fa9a2d1e2b3724670288b608c8d4eb895c51`, 2,797,376,661 bytes | Downloaded and ZIP-tested at ignored `artifacts/method1/input/releases/sign_features.zip` | The official archive contains only test features: PH has 642 agnostic and 642 aware files, with no train/dev files; it cannot replace the S1 training regime |
| CLIP ViT-B/32 init | SHA-256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` | Present at `artifacts/pretrained/ViT-B-32.pt` | Permitted initialization; passed Method 1 input audit |
| UPRet trained checkpoint | per-seed corrected baseline | Missing | Must be trained and selected on dev before reference caches |
| How2Sign annotations | UPRet train/test plus audited realigned sources | Partial | Full CiCo I3D agnostic/aware train/dev/test features are absent; raw-video audit also reports 118/2/6 missing train/dev/test clips |
| CSL-Daily English annotations | UPRet translated-English train/test; local translated train/dev assets | Partial | Local agnostic/aware features cover train/dev only; established UPRet test features are absent and aware-stream publication equivalence is unresolved |

Reference-cache and miner commands are implemented but remain deliberately unexecuted on real
PH data until that dev-selected checkpoint exists. Their immutable identities include the
teacher, tokenizer, caption/video manifests, mixture/sampler semantics, and implementation
version; the mining table and negative-span vectors are published together as one atomic
auxiliary bundle.

The active PH manifest uses locally re-extracted agnostic and P14T-adapted aware features for
all three official splits. A complete row-aligned audit over all 642 matching test videos
(54,997 feature rows; every stream has exactly matching per-video lengths) gives mean/median
row cosine `0.99319/0.99940` for local versus released agnostic features and
`0.45116/0.44324` for local P14T-aware versus released aware features. After the configured
`.9/.1` mixture, mean/median row cosine is `0.97124/0.97590`, while mean per-video relative
L2 error is `0.23802`. Consequently, results from this manifest are matched local-resource
measurements, not an exact published-CiCo/UPRet reproduction and not eligible for the broad
SOTA gate. The release archive is test-only, so using its test features would also create a
train/test feature-regime mismatch and is not used for model selection.
The reproducible audit command is `tools/audit_ph_release_features.py`; its ignored report is
`artifacts/method1/audit/ph_release_feature_alignment.json` with content SHA-256
`463d41bb2be5d541b0789ffe26eb53ae5ef756eac2ead6a184335667b3f804a9`.

The UPRet repository snapshot contains no visible license file. Source adaptation remains
attributed and patch-based; redistribution/license compatibility is unresolved and must be
reviewed before public release.

The H2/CSL confirmation configs intentionally point at their expected isolated Method 1
resource locations. `audit --stage input` fails closed while those split assets are missing;
no random dev split, duplicated video, copied feature, or fabricated metric is used to make
the confirmation datasets appear runnable.
