# Duplicate-wrist consistency — Stage0 protocol

2026-09-22, Cycle15, fixed before inspecting aggregate coordinates.
Academic-research-suite run/validate; autonomous experiment authorization comes
from the user's research guide. No human annotation or semantic pseudo labels.

Question: do recovered PH TRAIN poses frequently place hand-root landmarks
near the opposite body wrists, with sustained strong evidence of disagreement?
This is a detector-internal consistency screen, not confirmed anatomical errors.

Source: the local RTMPose COCO-wholebody schema labels indices9/10 as left/right
body wrists and91/112 as left/right hand roots;5/6 are shoulders. Extraction
stores all predictions in original210x260 pixel coordinates, before native
per-hand crop normalization/mirroring. Compare only those original coordinates.
These estimates share a detector; they are not independent ground truth.

For each frame, set shoulder distance d. Eligibility requires all six involved
landmarks (5,6,9,10,91,112) confidence >=0.5, d>1 pixel, and body-wrist separation
>0.25d. Native cost N is the sum of same-side root-to-wrist Euclidean distances;
swapped cost S is the sum of opposite-side distances. Flag a frame only if
S < 0.25N and S <=0.5d (mean swapped distance <=0.25d). The geometry criterion
is explicitly a heuristic; these flags must not be called verified swaps.

Population: all7096 TRAIN clips, no DEV/TEST. Primary exposure uses stored
retained_frame_indices, with each decoded frame counted once, not repeatedly
through overlapping windows. Raw decoded-frame counts are secondary. A primary
affected clip contains >=3 flagged frames with consecutive original indices;
subsample gaps or an ineligible frame interrupt a run. Report all denominators,
clips with >=16 eligible retained frames, flagged frame counts and longest runs.

Locked decision: if fewer than80% of clips have >=16 eligible retained frames,
the screen is INCONCLUSIVE_LOW_COVERAGE. Otherwise, if fewer than5% of all7096
clips have a qualifying retained run, NO_GO for frequent sustained duplicate-
wrist disagreement as the next research lead; no threshold relaxation. If both
coverage and burden pass, GO_FOR_ATTRIBUTION_ONLY: next distinguish body-wrist
error from hand-root error and relate unchanged observations to actual retrieval.
No result directly admits an identity-repair/marginalization method or supplies
novelty. Generic confidence gates, coordinate rescue and new-stream stacks remain
closed. There is no score adjustment, input repair, training or model inference.

Before census: synthetic controls must detect three-frame hand-root-only swaps,
reject aligned/crossed but consistently labeled hands, low-confidence points,
coincident body wrists, transient disagreement and gaps. Check translation/scale
invariance away from the fixed1-pixel eligibility floor, and invalid input failure.
These are numerical fixtures, not synthetic linguistic negatives.

Integrity: exact TRAIN ID count/order and recovered pose hashes; unchanged
metadata inventory must match Cycle7. Hash code, protocol, schema, extraction
and native loader. Read local trusted pose pickles only. Refuse report overwrite.
CPU only, hard60s per census, bounded memory one video at a time. If completed,
repeat once to a separate report and require byte identity. No significance
test or independent-frame sampling claim; report all11 statistical cautions.
