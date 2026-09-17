# AS-C43: gallery-only neighborhood support diagnostic

## Material Passport

academic-research-suite / experiment-agent / run; 2026-09-15; ANALYZED protocol,
not a verified method. Autonomous experiment authority comes from the user's
research loop; roles inline, no subagents. Registered before graph/score results.

## Question and consequence

AS-C07's full-cohort assignment consumes other queries. Does structure available
from the gallery alone support or suppress fine ranking distinctions? This is a
fixed competitor-feature diagnostic under user-loop §§13–15, not a proposed
publication reranker. No use of AS-C42 residual supervision.

If a fixed operator improves its three source baselines in at least two seeds,
beats graph shuffling, and improves the existing uniform-three strongest control,
the next action is mechanism attribution and three genuinely distinct candidate
derivations, not naming graph filtering as new. If neither fixed operator survives,
do not tune graph size, strength, descriptor layer, temperature or graph learning.
The negative would cover this probe only, not all gallery structure.

## Locked intervention

- PH DEV 519, unchanged full galleries and official positive mappings; no TEST.
- Re-encode the three historical best checkpoints (42,1337,2026), preserving
  current native inference settings. Original full519 scores must replay EXACT.
- Per checkpoint and modality: mask-aware mean of raw contextual token vectors,
  float32 accumulation; normalize the resulting 512-vector in float64 for cosine.
  Video mask0/text mask1, exactly bridge conventions. No other pooling variants.
- Directed five-nearest-neighbor gallery graph, self excluded, descending cosine,
  stable index tie ordering. Uniform weight1/5. No caption IDs, paired relevance,
  signers, source IDs or query scores used in graph construction.
- For each single query, let s be its original gallery score vector, and let m_i
  be the mean score of the five gallery neighbors of candidate i. Fixed operators:
  smoothing s'=s+0.5(m-s); sharpening s'=s-0.5(m-s). These symmetric finite probes
  are not claimed optimal or a reproduction of a named diffusion implementation.
- T2V uses ONLY video-gallery graph; V2T ONLY text-gallery graph. Each operation
  accepts a single query independently. Do not combine directions into a fictitious
  shared score function; save distinct video×text matrices for each direction.
- Control: graph node relabeling by one fixed permutation seed20260915, same
  permutation in both modalities/checkpoints. Preserve topology, zero diagonal,
  outdegree5 and in-degree distribution, NOT each item's degree or nuisance stratum.
- Execute both signs for real and shuffled graphs, all three checkpoints. Uniform
  average their corresponding outputs as ONE ensemble condition per sign/control.
  Compare against AS-C32 uniform3, never compare three-model cost to one as novelty.
- Float64 neighborhood arithmetic then float32 scores; preserve original scorer,
  official T2V tie expansion and V2T ordering. No selection/tuning after results.

## Evaluation and validation

Report all16 direction-paired conditions (3seeds+1ensemble ×2operators×2graphs),
official R1/R5/R10, persistent ranks (original92T/87V), fixed-gallery source-prefix
bootstrap10,000draws seed20260915. Compare real vs own baseline and shuffled, and
ensemble vs AS-C32/shuffled ensemble. Report nominal95%CIs and a conservative
16-comparison Bonferroni interval as sensitivity, not fresh confirmatory inference
after repeated DEV reuse. Inferred prefixes are not verified recording identities.

Diagnostic gate: mean gain≥0.5pp, nominal CI lower>0, each direction R1 loss≤0.25pp,
R5/R10 loss≤0.5pp, persistent mean ranks improve both; positive real-vs-shuffle
CI; same operator passes≥2/3seeds AND ensemble passes vs strongest control.
Even a pass is NOT method GO or proof of independent-scalar-score inadequacy.
Node shuffle does not isolate linguistic semantics from all nuisance confounding.

Check full query-batch versus separate single-query operations exactly, and
independent scalar neighbor enumeration versus vectorized outputs. Save descriptors,
graphs, score arrays, hashes, configuration/checkpoint parent hashes and all metrics.
Test fixtures include graph relabeling, self exclusion, identity beta0 and query
independence. No query-bank, matching, extra streams, changed positives or fitting.

Command: `PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.gallery_structure_probe`.
Working directory `/home/haipd/SLR`. External hard timeout600s, progress per seed,
monitor actual PID and `AS-C43-GALLERY_run.json`; expected artifacts under
`artifacts/proposal7/phase2/AS-C43/`. No automatic efficacy rescue/restart.

## Prior and closure boundary (search-bounded)

Searches on2026-09-15: `image text retrieval gallery graph diffusion re ranking
intra modal similarity`; `Efficient Diffusion for Image Retrieval CVPR 2017 Iscen`;
`cross modal retrieval transductive graph re-ranking query independent`.
Primary evidence of established mechanisms:
[Iscen et al., Efficient Diffusion on Region Manifolds](https://arxiv.org/abs/1611.05113)
(author abstract / CVPR2017 indexed primary record), and
[Zhang et al., Understanding Image Retrieval Re-Ranking](https://arxiv.org/abs/2012.07620v2)
(original arXiv abstract), which explicitly relates neighbor graphs to message
propagation. No claim to reproduce their results or code. CVF direct page403;
PMC cross-modal graph paper browser challenge, so no full-text claim from it.
This is a targeted pre-experiment prior check, not a systematic review or novelty
clearance. No private corpus uploaded.

Internal registry: candidate-only offsets/query-bank correction CLOSED; AS-C07
fixed-bank assignment algebraically reduces to such offsets. Here an off-diagonal
gallery operator mixes this query's scores, so it is not merely a query-independent
candidate bias. It still can be represented as an ordinary fixed-gallery scoring
function, and graph filtering is known prior. No scalar-expressivity ceiling claim.
RPCA, PMGR, local paired-evidence methods and relation/GW alignment remain closed.
