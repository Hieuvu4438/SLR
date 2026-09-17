# UPRet reduction: real-TRAIN gradient screen

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, inline execution and validation
- Origin Date: 2026-09-15
- Verification Status: VERIFIED computational replay; ANALYZED interpretation
- Scope: bounded diagnostic, not benchmark reproduction or method efficacy

## Outcome

The registered sensitivity lead fails. With the frozen amplitude control, median
relative encoder-gradient changes are **0.174891% visual / 0.220684% text** at the
partial checkpoint, below the preregistered1% threshold in both. This deprioritizes
max→sum as a major mechanism in this setting; it does not establish universal
ineffectiveness, harmful gradients, an information ceiling or a retrieval gain.

128 disjoint TRAIN examples were selected without inspecting loss:32 calibration,
three32-example measurement batches. The same batch encodings and stochastic
transport tensors served all four arms. Both CLIP initialization and the exact
historical partial step767 state were tested. No parameters were updated, and
complete state hashes are unchanged before/after each state. No DEV evaluation,
TEST file, all-split manifest, SEDS asset or new pretrained resource was opened.
Existing pretrained CLIP and partial UPRet weights were loaded, so this is NOT an
asset-free audit. Feature inputs retain the historical P14T-aware local regime,
not the stronger CiCo run's H2S-aware regime.

| Model state | Full sum vs max, visual/text | Amplitude-controlled sum vs max, visual/text | No OT vs max, visual/text |
|---|---|---|---|
| CLIP initialization | 1.914920% / 2.047661% | 0.156004% / 0.167960% | 0.609307% / 0.746615% |
| Partial step767 | 2.883354% / 3.055430% | 0.174891% / 0.220684% | 1.139472% / 1.230174% |

Entries are median ||g_arm−g_max||/||g_max|| over the three measurement batches,
with gradients over all trainable parameters in each encoder. They are NOT
percentage-point recall changes. Calibration coefficients are .2279624045 and
.2581481636 respectively. These are one-batch frozen controls, not exact amplitude
matching on every measurement batch. Both states have CLIP logit scale100;
the OT term remains outside that multiplication, preserving source semantics.

Native wrapper scalar losses agree exactly in all six measurement cases.
Selected used visual/text parameter gradient maximum absolute errors are at most
4.768372e−7 at initialization and1.455192e−10 at the partial state. All encoder
base-gradient norms are finite and nonzero. `clip.visual.conv2_trans.weight` is
unused in this input path; the reference and probe agree on this. Source uses
that projection only when the pre-transformer token count exceeds feature_len.
No claim that this conditional unused parameter is an upstream defect.

## Execution and reproducibility

First engineering attempt session64008 failedexit1 on a parity-check assumption
about unused parameters; preserved in UPRET-REAL-GRADIENT_attempt1.md. Before
retry, unused patterns were required to match and visual.proj was added as a used
visual parity parameter. Batches, arms and threshold did not change.

Corrected run session8533 exited0 in9.978587s. Separate replay session26019
exited0 in9.937662s. The program requires exact equality of the entire result
payload excluding wall time; the replay passed. This is same-code deterministic
reproducibility, not an independent implementation of every encoder derivative.
The native untouched objective and isolated log-softmax formula provide separate
algebra controls. Combined regression suite:90passed1.89s, exit0.

- CodeSHA: d33ec28ee2e93e6a75c0cc08a67e6fb9688fd2a2776626f4d73c7bd9fb2cb64e
- ProtocolSHA: c2a3d8da3a3015610b8253d64f187ebbfa36c9a3811b8c76f4be294d6839c371
- RunSHA: 9a9d34bafd5fcbd7702b098dc77e83e2c40bc5517b92a73346ce1973c54be25f
- ReplaySHA: b2bc15f9850d55523e1372ceea9e0141674fb864d41ec2fe3fa42acd8388a55d
- Partial checkpointSHA: c72ecbc4ce35bec1f5fe53692a7d3f88b27e591c4692a93e82df26fdde00ff82

## Interpretation safeguards — 11/11 checked

Simpson: states and encoders separately reported; ecological: no per-query
retrieval inference; Berkson: TRAIN batches random but partial checkpoint is
historically DEV-selected; collider: no outcome conditioning of selected samples;
base rate: all128 selections disclosed; regression-to-mean: no pre/post improvement
claim; survivorship: failed attempt and every completed batch recorded;
look-elsewhere: all four arms reported, no p-values; forking paths: locked protocol
and engineering repair disclosed; causation: controlled effect on gradients only,
not on generalization; reverse causality: no directional real-world causal claim.

Limitations: B32 is not the native B512 negative pool; only one initialization
seed and one incomplete checkpoint; train gradients at the latter are nearly
saturated (losses~4e−5 to3e−4). No full-gallery effect measured. The probe's1%
threshold is a prioritization convention, not a statistical equivalence margin
or a bound on long-horizon training effects. No rescaling sweep, new optimizer,
temperature placement change or generic uncertainty method follows from this.

## Decision consequence

Keep the paper/code discrepancy documented, but do not launch a long max/sum
benchmark campaign on this evidence alone. Return to repo-led search in a different
causal layer, requiring an explicit open decision consequence before another
probe. Existing SAN masking/truncation and hard-negative observations are
reproduction caveats, not permission to revive closed padding/lexical families.
The overall GO search remains active; no proposal8 or method GO is justified.
