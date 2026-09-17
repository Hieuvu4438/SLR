# CiCo short-input padding: zero current TRAIN exposure

## Material Passport

Academic-research-suite / deep-research, inline source verification; 2026-09-16.
Verified deterministic metadata census; no efficacy or linguistic judgment.

## Decision

Reject short-input repeat-last padding as an explanation for the current PH
TRAIN extraction inputs. All 7,096 videos have at least 16 decoded frames and
all 720,914 recorded windows match unshifted stride-one, 16-frame extraction.
This is a bounded negative exposure result, not a new method, information ceiling,
GO, or global search barrier. No padding/sampling repair experiment follows.

## Source behavior and measurement

Pinned SLRT revision `38a4f7b00da7a858d59b7fabe5093876a84db8e0`:
`CiCo/I3D_feature_extractor/datasets/videodataset.py:45-73` computes the number of
windows with a ceiling, moves an incomplete final window back when possible,
and otherwise starts at zero. Lines 139-140 repeat the preceding frame on decode
failure. `phoenix2014.py:80-87` loads the sentence video before extracting windows.
The local `shared/slr_common/features/i3d.py:60-78` mirrors window placement;
lines 203-205 explicitly repeat the final frame only if decoded length is below
the configured 16 frames.

For stride one and integer length N >= 16, the window starts are exactly
0,...,N-16. The last window is already complete, so the shift-back branch is
inactive. N=16 produces one unpadded window. This is input indexing, not a claim
about the network's internal convolutions or effective receptive field.

[Locked protocol](CICO_input_padding_protocol.md),
[machine-readable census](CICO-INPUT-PADDING.json),
[read-only code](../../../../methods/information_probe/input_padding_exposure.py).

| TRAIN endpoint | Result |
|---|---:|
| Exact unique manifest IDs and temporal files | 7,096 |
| Decoded frame-count minimum / maximum | 16 / 475 |
| Below / equal to / above 16 frames | 0 / 13 / 7,083 |
| 16-frame, stride-one recipe | 7,096 / 7,096 |
| Window-start / interval-end mismatch | 0 / 0 |
| Recorded windows | 720,914 |
| Required repeat-last draws from short inputs | 0 |

Executed `python -m methods.information_probe.input_padding_exposure`, exit 0,
0.166641165 seconds reported by execution tool. All manifest/path/split/schema/
recipe assertions pass. The JSON records code/source hashes, manifest hash and
a digest over all sorted filename/content-hash pairs. Counts are a census, not
an estimate requiring a confidence interval or a retrieval performance test.

## Limits and closure discipline

- Only TRAIN manifest and temporal JSON data were opened; no source-video paths
  followed, feature arrays, checkpoints, DEV/TEST, GPU, extraction or training.
- Metadata validates recorded extraction geometry, not video decoder correctness
  or every historical feature tensor. The upstream decode-failure fallback can
  repeat frames for reasons other than short input; this census does not measure
  that behavior. Local decoder behavior was not tested here.
- Internal convolution padding can affect boundary representations even for
  complete windows. This result neither diagnoses its harm nor supplies a new
  boundary-aware method. Do not substitute it as an unregistered positive result.
- Adaptation pseudo-clips and pretraining videos are outside this census; the
  previous pseudo-clip result remains unchanged. No claim about author-released
  feature provenance, other datasets, DEV prevalence or historical pretraining.
- Browser opening of the pinned GitHub source failed with cache miss. Source
  verification uses the available local pinned checkout and hashes, not a claim
  of successful live retrieval. Reading historical shell code exposed TEST path
  strings only; none of its commands was executed and no TEST data was opened.

ARS evidence discipline kept code behavior, actual exposure, and causal efficacy
separate. No candidate is promoted. Return to a distinct mechanism with an open
decision consequence, not more short-clip/padding fixtures or recipe sweeps.
