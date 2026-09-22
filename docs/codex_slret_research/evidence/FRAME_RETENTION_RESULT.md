# Frame-retention result — omission is not established by the window cap

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-22
- Verification Status: ANALYZED, deterministic metadata census
- Version Label: frame_retention_v1

## Result

[M] The predeclared [protocol](FRAME_RETENTION_PROTOCOL.md) finds **zero retained
frames omitted by window selection across all7615 recovered examples**.
In DEV, all55,775 decoded frames are represented in at least one selected
window: no temporal subsampling, hand-support frame removal or window-coverage
gap occurs in these recorded inputs. This does not mean that all their visual
information survives spatial preprocessing, I3D encoding or later pooling.

| Stage | TRAIN7096 | DEV519 |
|---|---:|---:|
| Decoded frames in supplied MP4s | 827,354 | 55,775 |
| Frames on the native subsampling grid | 823,489 | 55,775 |
| Frames retained after support filtering | 823,485 | 55,775 |
| Examples with temporal subsampling | 23 | 0 |
| Examples with support-filter removal | 4 | 0 |
| Retained frames not covered by selected windows | 0 | 0 |

[M] Each of the four affected TRAIN examples loses exactly one frame at support
filtering. None is among the13 CTC failures. Those13 consist of eleven16-frame
inputs and two17-frame inputs, all with every decoded frame retained and covered.
The native16-frame/stride1 window constructor yields respectively one or two
valid windows, fewer than their required2–5 CTC timesteps. Detailed IDs and
decompositions are in the [machine result](frame_retention_20260922.json).

[V/I] A coverage argument agrees with the census. The native pre-filter grid
contains at most300 frames. For79–300 retained frames, uniformly selecting64
window starts from `0..F-16` gives a maximum adjacent start gap of5, smaller than
window width16, while retaining both endpoints. Thus consecutive windows
overlap and cover every retained index. For16–78 frames all valid starts are
used; shorter inputs, if present, are padded within a single window. The cap
can reduce the number of overlapping views without removing a retained frame.

## Decision and limitations

Reject the precise hypothesis that the64-window cap omits retained frames in
these recovered inputs. Reject support filtering and input subsampling as the
cause of these13 CTC length failures. Increasing the cap alone cannot give the
16/17-frame inputs more native windows. No policy or architecture is changed.

This does **not** rule out loss inside a window encoder, insufficient temporal
resolution of its outputs, spatial crop loss, or information already absent
from the supplied MP4s. It does not establish equivalence to the deleted cache
used by historical GCN runs, so no new attribution of those saved retrieval
errors is claimed. Cross-stream timestamp equality and semantic sufficiency
are also different claims from union coverage.

The observation changes the next action: do not pursue denser-window or
support-filter repair on an omitted-frame rationale. Closed sampling/alignment
methods remain closed. A different representation-loss hypothesis still needs
its own distinguishing observation and mechanism; these counts do not justify
another generic readout, backbone swap or raw-acquisition policy.

## Verification

Command, `/home/haipd/SLR`:

```bash
/home/haipd/miniconda3/bin/python docs/codex_slret_research/tools/diagnose_frame_retention.py --output docs/codex_slret_research/evidence/frame_retention_20260922.json
```

Exit0. Every metadata inventory matches the prior recovery audit; source and
protocol hashes are recorded. Four new fixtures distinguish intentional
subsampling, hand filtering, window omission, padding, invalid grid membership
and inconsistent timestamps. No GPU, active process check, raw-video decoding,
TEST, changed data or labels. No full extraction reproducibility claim.

Statistical fallacy applicability scan:11/11 checked. Simpson, ecological,
Berkson, collider, base-rate, regression-to-mean and reverse-causality analyses
are N/A because no sampled association is estimated. Survivorship: all7615
official examples are included. Look-elsewhere/forking paths: exact zero-omission
criterion and stage decomposition were specified before the aggregate census;
no threshold search. Causality: the technical window-count consequence is
distinguished from unmeasured retrieval effects. No significance claim.
