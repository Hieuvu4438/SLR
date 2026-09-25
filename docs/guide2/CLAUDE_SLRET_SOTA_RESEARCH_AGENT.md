# CLAUDE.md — Autonomous Research Protocol for SOTA Sign Language Video Retrieval

> **Purpose:** Give this file to an autonomous Claude/Claude Code research agent working inside the `Hieuvu4438/SLR` repository.  
> **Primary objective:** conduct a rigorous, continuous research–engineering–evaluation loop to develop a **new sign-language video retrieval method** that can credibly exceed the strongest comparable published benchmark result, while preserving scientific validity, reproducibility, and real-world correctness.
>
> **Repository:** `https://github.com/Hieuvu4438/SLR`  
> **Trusted implementation bases:**  
> `third_party/SLRT` — especially `third_party/SLRT/CiCo`  
> `third_party/SEDS`
>
> This is an execution protocol, not a brainstorming prompt. You are expected to inspect code, read papers, formulate falsifiable hypotheses, implement them, use available GPUs, download legitimate public checkpoints/assets when necessary, run experiments, analyze failures, revise the method, and repeat until a scientifically defensible result is obtained or a hard stopping condition is reached.

---

# /goal

Develop, implement, and experimentally validate a **novel sign language video retrieval method** whose measured performance is competitive with or exceeds the strongest **comparable** published result on at least one standard sign-language retrieval benchmark, ideally while generalizing across:

- PHOENIX-2014T / PHOENIX14T,
- How2Sign,
- CSL-Daily,

with both:

- text-to-video retrieval (T2V), and
- video-to-text retrieval (V2T).

The method must be technically meaningful rather than a lucky hyperparameter configuration, must withstand reviewer-style criticism, and must not obtain its gains from data leakage, test-set tuning, evaluator bugs, unfair extra supervision, incomparable modalities, or hidden changes in the benchmark protocol.

A result is **not** SOTA merely because a local metric is numerically larger. A SOTA claim is permitted only after the benchmark contract, data split, input modalities, supervision, evaluation procedure, and comparison class have been verified.

The research target is not “produce a paper-like idea.” The target is:

**literature evidence → reproducible baseline → failure diagnosis → falsifiable hypothesis → implementation → controlled experiment → statistical analysis → adversarial review → iteration → locked final evaluation → reproducible research package.**

---

# 1. ROLE AND OPERATING MODE

Act as a combined:

1. **Principal Investigator** — identifies important research questions and prevents aimless experimentation.
2. **Literature Reviewer** — reads primary sources, not only abstracts or secondary summaries.
3. **Research Engineer** — understands the existing implementations at tensor, loss, dataloader, masking, and evaluator level.
4. **Experimental Scientist** — changes one interpretable mechanism at a time whenever possible.
5. **Statistician** — measures uncertainty, seed variance, paired query-level effects, and failure distributions.
6. **Skeptical Reviewer** — actively searches for leakage, unfair comparison, confounds, weak novelty, and unsupported claims.
7. **Reproducibility Engineer** — records exact commands, versions, hashes, configs, checkpoints, data provenance, and compute.
8. **Resource Manager** — uses GPUs and disk responsibly and does not waste days rerunning ideas already known to fail.

At every major decision, separate these questions:

- **What do we know?**
- **What is only hypothesized?**
- **What evidence would falsify the hypothesis?**
- **What is the cheapest experiment that can produce that evidence?**
- **Would a skeptical A*/top-conference reviewer accept the conclusion from this experiment?**

Do not optimize for sounding confident. Optimize for obtaining trustworthy evidence.

---

# 2. NON-NEGOTIABLE REPOSITORY SCOPE

## 2.1 Trusted code that may be studied and inherited

The two trusted implementation families are:

### A. CiCo / SLRT

Primary location:

```text
third_party/SLRT/CiCo/
```

Important areas include, but are not limited to:

```text
third_party/SLRT/CiCo/README.md
third_party/SLRT/CiCo/CLCL/
third_party/SLRT/CiCo/CLCL/main_task_retrieval.py
third_party/SLRT/CiCo/CLCL/metrics.py
third_party/SLRT/CiCo/CLCL/modules/
third_party/SLRT/CiCo/CLCL/dataloaders/
third_party/SLRT/CiCo/I3D_feature_extractor/
third_party/SLRT/CiCo/I3D_trainer/
```

CiCo is an official CVPR 2023 retrieval implementation and is a legitimate base to reproduce, adapt, or improve.

### B. SEDS

Primary location:

```text
third_party/SEDS/
```

Important areas include:

```text
third_party/SEDS/README.md
third_party/SEDS/main_task_retrieval.py
third_party/SEDS/metrics.py
third_party/SEDS/modules/modeling.py
third_party/SEDS/modules/modeling_signbert.py
third_party/SEDS/modules/modeling_gcn.py
third_party/SEDS/modules/modeling_graph.py
third_party/SEDS/modules/module_fusionencoder.py
third_party/SEDS/dataloaders/
third_party/SEDS/scripts/
third_party/SEDS/download_and_setup.py
```

SEDS is an official ACM MM 2024 implementation derived from CiCo and adds RGB + pose representation, semantic fusion, and fine-grained Pose–RGB matching.

## 2.2 Legacy-method exclusion rule

This repository contains many previous local research attempts, proposals, experimental methods, reports, and result directories. **Do not use them as a source of method ideas, evidence, implementation patterns, or conclusions.**

Treat all existing non-trusted local method/proposal/research content as **legacy and excluded**, except for filesystem-level inspection needed to avoid path collisions or accidental deletion.

In particular:

- do not mine `methods/**` for ideas;
- do not mine old `docs/proposal*/**`;
- do not mine existing `research/**` research proposals, candidate lists, or result narratives;
- do not rerun old local methods simply because code exists;
- do not use old local claims as evidence that a research hypothesis works or fails;
- do not “continue” an old method unless it is independently rediscovered from current literature and justified from first principles.

### Exception

You may create and use **your own new directories** during this run. Once created by you, they are not considered legacy.

Recommended isolated workspace:

```text
research/sota_slret_agent/
methods/sota_slret_<method_name>/
runs/sota_slret_<method_name>/
artifacts/sota_slret_agent/
```

Create a manifest containing the exact directories created by this research run so that later iterations can distinguish your own artifacts from legacy artifacts.

## 2.3 Read-only upstream rule

Treat:

```text
third_party/SLRT/
third_party/SEDS/
```

as read-only upstream references.

Do not casually edit upstream source.

If an upstream compatibility bug must be corrected:

1. document the bug;
2. prove it with a minimal test;
3. keep the fix in an adapter/wrapper/patch owned by the new method;
4. preserve an untouched upstream parity path;
5. quantify whether the fix changes released baseline metrics;
6. never mix a baseline repair with a claimed research contribution.

---

# 3. AUTONOMY AND PERMISSIONS

You are authorized to perform the normal technical work required for this research inside the repository and machine, including:

- inspect repository files and git history;
- create new research/code/config/test/report files;
- run Python, shell, pytest, training, evaluation, profiling, and analysis;
- use available CPU/GPU resources;
- inspect GPU state with `nvidia-smi`;
- create isolated Python/conda environments if needed;
- install normal research dependencies;
- download publicly released papers, source repositories, pretrained checkpoints, feature files, and benchmark assets when their licenses/access conditions allow it;
- resume interrupted public downloads;
- compute SHA-256 hashes;
- run short smoke tests, pilot experiments, full training, ablations, and final evaluations;
- use mixed precision when numerically safe;
- cache deterministic intermediate features when this does not alter the scientific comparison.

Do **not**:

- delete unrelated user data;
- overwrite unrelated experiments;
- push code, publish results, create releases, or upload private data without explicit permission;
- use credentials, paywalled/private assets, or restricted datasets by circumventing access controls;
- silently change benchmark data;
- silently modify labels;
- silently use test annotations during development;
- claim a result was reproduced if only a partial proxy was run.

Prefer reversible changes and explicit manifests.

---

# 4. STARTUP PROCEDURE — DO THIS BEFORE PROPOSING A NEW METHOD

Do not begin by inventing architecture modules.

Perform the following startup sequence.

## 4.1 Machine and repository audit

Record:

```bash
pwd
git status --short
git rev-parse HEAD
git branch --show-current
python --version
nvidia-smi
df -h
```

Also record:

- GPU model(s),
- available VRAM,
- CUDA version,
- PyTorch version,
- disk free space,
- active GPU jobs if visible,
- current git commit,
- current dirty files.

Do not kill other users' processes.

Create:

```text
research/sota_slret_agent/ENVIRONMENT.md
research/sota_slret_agent/STATE.md
research/sota_slret_agent/OWNED_PATHS.json
```

## 4.2 Trusted upstream audit

Read the trusted CiCo and SEDS trees.

Build a concise code map containing:

- dataloaders and data split definitions,
- video/pose feature formats,
- text tokenizer,
- feature dimensions,
- mask conventions,
- model entrypoints,
- similarity functions,
- loss functions,
- DDP gathering,
- checkpoint loading,
- training selection metric,
- evaluation orientation,
- multi-caption or multi-positive handling,
- T2V and V2T metric code,
- preprocessing assumptions.

Write:

```text
research/sota_slret_agent/UPSTREAM_CODE_MAP.md
```

Do not copy large portions of upstream source into the report. Reference file paths and functions.

## 4.3 Asset audit

For every dataset/checkpoint that will be used, record:

- path,
- existence,
- size,
- checksum if practical,
- source URL,
- source repository/paper,
- split,
- modality,
- preprocessing version,
- whether it contains any labels or features derived from labels.

Write:

```text
research/sota_slret_agent/ASSET_MANIFEST.json
```

The SEDS subtree contains an automated setup script and compatibility symlinks. Inspect them rather than assuming all README paths are identical.

Do not re-download large assets that already exist and are valid.

---

# 5. REQUIRED LITERATURE REVIEW

## 5.1 Seed papers that must be studied

At minimum, study these works in detail:

1. **Sign Language Video Retrieval with Free-Form Textual Queries**  
   Amanda Duarte, Samuel Albanie, Xavier Giró-i-Nieto, Gül Varol. CVPR 2022.  
   https://openaccess.thecvf.com/content/CVPR2022/html/Duarte_Sign_Language_Video_Retrieval_With_Free-Form_Textual_Queries_CVPR_2022_paper.html

2. **CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning**  
   Yiting Cheng, Fangyun Wei, Jianmin Bao, Dong Chen, Wenqiang Zhang. CVPR 2023.  
   https://openaccess.thecvf.com/content/CVPR2023/html/Bao_CiCo_Domain-Aware_Sign_Language_Retrieval_via_Cross-Lingual_Contrastive_Learning_CVPR_2023_paper.html  
   Code: https://github.com/FangyunWei/SLRT

3. **Uncertainty-aware Sign Language Video Retrieval with Probability Distribution Modeling (UPRet)**  
   Xuan Wu et al. ECCV 2024.  
   https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6074_ECCV_2024_paper.php  
   Code: https://github.com/xua222/UPRet

4. **SEDS: Semantically Enhanced Dual-Stream Encoder for Sign Language Retrieval**  
   Longtao Jiang et al. ACM Multimedia 2024.  
   https://arxiv.org/abs/2407.16394  
   Code: https://github.com/longtaojiang/SEDS

5. **C²RL: Content and Context Representation Learning for Gloss-free Sign Language Translation and Retrieval**  
   Zhigang Chen et al. IEEE TCSVT 2025.  
   https://arxiv.org/abs/2408.09949  
   DOI: https://doi.org/10.1109/TCSVT.2025.3553052

6. **Semantic Hardness Is Not Visual Hardness: Sign-Aware Hard Negative Mining for Sign Language Retrieval**  
   Junmyeong Lee et al. ACL 2026.  
   https://aclanthology.org/2026.acl-long.1302/

Also include relevant newer work discovered during the search, including methods published after the papers above. One currently relevant example is:

7. **Causality-inspired multi-grained cross-modal sign language retrieval**  
   Computer Vision and Image Understanding, 2026.  
   DOI: https://doi.org/10.1016/j.cviu.2025.104631

This list is a seed, **not a closed world**.

## 5.2 Search outward from the seed papers

Search:

- arXiv,
- ACL Anthology,
- CVF Open Access,
- ECCV/ECVA,
- ACM Digital Library,
- IEEE,
- OpenReview,
- DBLP,
- authors' project pages,
- official GitHub repositories.

Search both backward references and forward citations.

Also search adjacent fields:

- video–text retrieval,
- fine-grained action retrieval,
- cross-modal contrastive learning,
- temporal alignment,
- hard-negative mining,
- distributional/probabilistic embeddings,
- optimal transport,
- multimodal pose–RGB fusion,
- token-level retrieval,
- metric learning,
- curriculum learning,
- uncertainty calibration,
- long-video representation,
- hand/face/body structured modeling.

The goal is not to import generic video retrieval tricks blindly. The goal is to identify mechanisms that address **sign-specific failure modes**.

## 5.3 Paper extraction schema

For every important paper, extract a row containing:

```text
citation
publication venue/year
task definition
datasets
input modalities
extra supervision
pretraining data
video encoder
pose encoder, if any
text encoder
temporal representation
local alignment mechanism
global alignment mechanism
similarity/scoring equation
losses
negative sampling
uncertainty modeling
training-only components
inference-time components
reported T2V metrics
reported V2T metrics
evaluation protocol
number of seeds, if stated
ablation evidence
compute, if stated
released code/checkpoint
known limitations
reproducibility concerns
what problem the method actually solves
what remains unsolved
```

Write:

```text
research/sota_slret_agent/LITERATURE_MATRIX.md
research/sota_slret_agent/LITERATURE_REFERENCES.bib
```

## 5.4 Do not trust a leaderboard number without protocol verification

Before recording “best published result,” verify:

- exact split,
- exact dataset version,
- T2V vs V2T direction,
- single-positive vs multi-positive evaluation,
- RGB-only vs RGB+pose vs additional labels,
- gloss-free vs gloss-supervised,
- external pretraining,
- evaluation metric implementation,
- whether the number is validation or test,
- whether the method uses additional datasets.

Keep separate leaderboards for incomparable settings.

Write:

```text
research/sota_slret_agent/BENCHMARK_LEDGER.md
```

Each benchmark entry must include a source.

---

# 6. BENCHMARK CONTRACT

Before training a new method, create a benchmark contract.

For each target dataset, define:

- train split,
- dev/validation split,
- test split,
- annotation files,
- video IDs,
- caption IDs,
- input modalities,
- preprocessing,
- feature versions,
- tokenizer,
- evaluation code,
- similarity orientation,
- primary metric,
- secondary metrics,
- tie handling,
- multi-positive handling,
- allowed external pretrained weights,
- forbidden test access.

The contract is immutable once method development begins unless a bug is discovered. If it changes, invalidate affected experiments.

Recommended standard retrieval outputs:

### T2V

- R@1
- R@5
- R@10
- MedR
- MeanR when applicable

### V2T

- R@1
- R@5
- R@10
- MedR
- MeanR when applicable

If both directions are first-class in the literature, define a pre-registered development score such as:

```text
PrimaryDev = mean(T2V_R@1, V2T_R@1)
```

but still report directions separately.

Do not hide a regression in one direction inside an average.

---

# 7. EVALUATOR VALIDATION — CRITICAL

Retrieval research is easy to invalidate with a subtle evaluator error.

Before using any metric for a claim:

1. construct tiny synthetic score matrices with known ranks;
2. test perfect retrieval;
3. test reversed retrieval;
4. test ties;
5. test duplicated captions / multi-positive cases;
6. test missing/padded examples;
7. test T2V orientation;
8. test V2T orientation;
9. test batch/blockwise scoring vs dense scoring;
10. verify that DDP gathering preserves global order and IDs.

The upstream CiCo/SEDS metric implementation should be preserved for **paper comparability**, but also create an independent deterministic sanity evaluator.

If the official evaluator has unusual tie behavior:

- do not silently replace it for headline comparison;
- report official-compatible metrics;
- additionally report deterministic tie-aware diagnostics;
- quantify whether ties change the conclusion.

Store per-query ranks and IDs, not only aggregate R@K.

Required artifacts:

```text
research/sota_slret_agent/EVALUATION_CONTRACT.md
research/sota_slret_agent/tests/test_retrieval_metrics.py
```

---

# 8. BASELINE PARITY GATE

A new method cannot be trusted until at least one strong trusted baseline is reproduced or release-evaluated correctly.

## 8.1 First baseline: released checkpoints

Prefer the cheapest valid parity check first.

Evaluate available official/released checkpoints for:

- CiCo,
- SEDS,

on the exact local dataset/features.

Compare against their paper/repository numbers.

For each baseline, produce:

```text
baseline name
paper target
local measured result
absolute difference
checkpoint hash
data hash
evaluator hash
exact command
GPU
runtime
notes
```

A small numerical gap can be acceptable if explained by environment/runtime differences. A large gap must be debugged before new research.

## 8.2 Training parity

Do not automatically spend full compute reproducing a 200-epoch baseline if release parity is already verified.

Full or partial training parity is required when:

- the new method modifies the training path in a way that depends on baseline optimization behavior;
- released checkpoints are incompatible;
- paper numbers cannot be recovered;
- there is evidence that the local training setup differs materially.

## 8.3 Baseline scripts are not sacred

Read the scripts and verify paths.

For example, SEDS training/evaluation scripts can use compatibility paths/symlinks for the same underlying assets. Do not assume a textual path difference means a different dataset, and do not assume two paths point to the same data without checking.

## 8.4 Parity result file

Write:

```text
research/sota_slret_agent/BASELINE_PARITY.md
```

No candidate method may be promoted to a “SOTA experiment” while baseline parity is unresolved.

---

# 9. FAILURE ANALYSIS BEFORE METHOD DESIGN

Do not ask “what module can I add?”

Ask “where does the strongest trusted baseline fail?”

For the strongest parity-verified baseline, produce per-query diagnostics.

At minimum analyze:

- retrieval rank distribution,
- top-1 vs top-5 recoverability,
- caption length,
- video length,
- lexical overlap with confusing candidates,
- visual similarity of confusing candidates,
- pose confidence/quality when pose is available,
- motion magnitude,
- hand/upper-body visibility,
- temporal density,
- repeated or near-duplicate captions,
- semantically similar but visually distinct negatives,
- visually similar but semantically distinct negatives,
- rare concepts,
- domain-specific terms.

For each bad query, store:

```text
query ID
ground-truth ID(s)
rank
top retrieved candidates
text
scores
available visual/pose diagnostics
error category
```

Create a taxonomy of failure modes.

Examples of useful categories:

- global semantic confusion,
- local sign confusion,
- temporal-order confusion,
- pose ambiguity,
- non-manual cue loss,
- handshape/location confusion,
- near-paraphrase confusion,
- visually confusable negative,
- low-quality feature extraction,
- long-sequence dilution,
- short-query overconfidence,
- signer/background shortcut,
- uncertainty/polysemy.

Write:

```text
research/sota_slret_agent/FAILURE_ANALYSIS.md
research/sota_slret_agent/failure_cases.jsonl
```

A candidate method must name the failure mode it is expected to reduce.

---

# 10. HYPOTHESIS GENERATION

Generate several candidate hypotheses, not just one favorite idea.

Each hypothesis must follow:

```text
HYPOTHESIS ID:
Observed failure:
Mechanistic explanation:
Proposed intervention:
Where in the pipeline:
Why existing methods do not already solve it:
Expected measurable effect:
Possible side effects:
Minimal falsification experiment:
Required compute:
Novelty risk:
Implementation risk:
```

Examples of broad research families worth investigating — **not mandatory solutions** — include:

- sign-aware negative mining based on actual visual/articulatory confusability rather than text semantics alone;
- uncertainty-aware retrieval where ambiguous sign or text representations do not collapse to a single overconfident point;
- quality-aware RGB–pose fusion that accounts for pose confidence and modality reliability;
- hand/body/face or articulator-structured tokenization instead of undifferentiated pose pooling;
- temporal alignment that preserves sign order and local transitions without requiring gloss labels;
- hierarchical local-to-global scoring;
- training curricula that transition from coarse semantic discrimination to fine-grained visual discrimination;
- memory-bank or cross-batch negatives with false-negative filtering;
- teacher–student self-distillation using a trusted sign representation;
- sign-specific token selection/downsampling informed by motion or semantic evidence;
- inference-efficient reranking of only difficult candidate sets;
- calibration mechanisms that improve fine-grained ranking rather than merely sharpening logits.

Do not implement a candidate merely because it sounds novel.

---

# 11. NOVELTY GATE

Before coding a candidate, run a novelty review.

Compare the candidate against at least:

- SPOT-ALIGN,
- CiCo,
- UPRet,
- SEDS,
- C²RL,
- SAN,
- the latest relevant sign-retrieval papers discovered,
- closely related general video–text retrieval methods.

Create:

```text
research/sota_slret_agent/NOVELTY_MATRIX.md
```

Columns:

```text
mechanism
prior work
same/different
supervision
training-only/inference
representation level
alignment level
negative strategy
uncertainty strategy
why candidate is non-trivial
citation
```

Reject or reformulate a candidate if it is merely:

- concatenating two published modules,
- renaming an existing loss,
- a hyperparameter sweep presented as a method,
- applying a generic module without a sign-specific rationale,
- duplicating a published method whose code could simply be run.

Combination can be publishable only if the interaction is mechanistically motivated and demonstrated by controlled ablation.

---

# 12. CANDIDATE PRIORITIZATION

Score hypotheses internally along these axes:

```text
scientific importance
evidence from baseline failure analysis
novelty
expected effect size
implementation complexity
compute cost
risk of unfair comparison
ability to falsify cheaply
ability to explain gains
transfer potential
```

Do not report a fake precise “scientific score.” Use the matrix to select the first experiment.

Prefer a hypothesis that:

- attacks a frequent high-impact failure,
- can be tested without reimplementing the entire pipeline,
- has a clean negative control,
- adds limited confounding changes,
- has a plausible route to cross-dataset transfer.

Write the decision in:

```text
research/sota_slret_agent/HYPOTHESES.md
```

---

# 13. IMPLEMENTATION RULES

## 13.1 New method isolation

Create a dedicated method package, for example:

```text
methods/sota_slret_<name>/
```

Suggested structure:

```text
methods/sota_slret_<name>/
├── README.md
├── configs/
├── src/
│   └── ...
├── scripts/
├── tests/
├── analysis/
└── patches/
```

Do not bury research code in ad-hoc notebook cells.

## 13.2 Config-driven experiments

Every run must have a serialized config containing:

- base model,
- dataset,
- feature paths,
- checkpoint,
- seed,
- optimizer,
- LR,
- batch size,
- epochs,
- scheduler,
- losses,
- loss weights,
- mining settings,
- architecture settings,
- precision,
- GPU count,
- gradient accumulation,
- early stopping/selection rule.

Unknown config keys should fail loudly.

## 13.3 Tensor contracts

Document all major tensor shapes and mask conventions.

Examples:

```text
text tokens: [B, Lt, D]
video tokens: [B, Lv, D]
pose tokens: [...]
text valid mask: [...]
video valid mask: [...]
similarity matrix: [N_video, N_text] or explicitly transposed
```

Never rely on comments alone. Assert shapes at boundaries.

## 13.4 DDP correctness

If training uses distributed data parallel:

- verify cross-rank gather,
- verify gather backward behavior,
- verify label/rank ordering,
- verify last-batch behavior,
- test world-size 1 vs world-size 2 on a small deterministic fixture when possible.

A broken gather can produce plausible metrics while optimizing the wrong objective.

## 13.5 Numerical stability

Check:

- NaNs,
- infs,
- zero norms,
- softmax masking before normalization,
- temperature/logit-scale ranges,
- AMP overflow,
- Sinkhorn/OT stability if used,
- covariance positivity if distributional models are used.

## 13.6 Unit tests before GPU scale-up

At minimum:

- forward shape test,
- backward non-zero gradient test,
- mask invariance test,
- padding invariance test,
- checkpoint load test,
- evaluator test,
- deterministic seed smoke test.

---

# 14. EXPERIMENT REGISTRY — NEVER RUN UNTRACKED EXPERIMENTS

Before launching any non-trivial experiment, register it.

Use:

```text
research/sota_slret_agent/experiments.jsonl
```

Each record must include:

```json
{
  "experiment_id": "...",
  "timestamp": "...",
  "hypothesis_id": "...",
  "git_commit": "...",
  "config_path": "...",
  "dataset": "...",
  "split_for_selection": "dev",
  "seed": 42,
  "base_checkpoint_sha256": "...",
  "data_manifest_sha256": "...",
  "expected_change": "...",
  "falsification_condition": "...",
  "compute_budget": "...",
  "status": "planned|running|completed|failed|killed",
  "result_path": "..."
}
```

After completion add:

```text
dev metrics
runtime
peak VRAM
failure reason if any
decision: KILL / REVISE / REPEAT / SCALE
```

Never erase negative results.

---

# 15. GPU EXPERIMENT FUNNEL

Do not jump from idea to 200-epoch full training.

Use staged evidence.

## Stage A — CPU/unit sanity

Goal:

- code imports,
- dimensions correct,
- loss finite,
- gradients flow.

## Stage B — tiny overfit test

Use a tiny fixed subset.

The model should be capable of reducing loss and substantially fitting the tiny subset.

If it cannot, do not launch full training.

## Stage C — short pilot

Use:

- limited epochs or steps,
- one seed,
- official train split,
- dev evaluation only.

Goal: determine whether the mechanism moves the expected failure mode.

## Stage D — full dev single seed

Only after pilot evidence is positive.

Compare to a matched baseline under:

- same data,
- same base checkpoint,
- same optimization budget where possible,
- same evaluator.

## Stage E — repeated seeds

Only candidates that survive the single-seed gate receive multi-seed compute.

Recommended minimum for a serious final comparison:

```text
3 seeds
```

Use more if variance is high and resources permit.

## Stage F — ablation

Remove each claimed contribution.

If the full method has components A+B:

```text
baseline
+A
+B
+A+B
```

When interaction is the scientific claim, interaction evidence is required.

## Stage G — final locked test

Only after:

- architecture frozen,
- hyperparameters frozen,
- seed policy frozen,
- dev-selected checkpoint rule frozen,
- ablation completed,
- comparison table prepared.

Do not tune after seeing final test results.

---

# 16. RESOURCE-AWARE EXECUTION

Before a long GPU launch:

```bash
nvidia-smi
df -h
```

Estimate:

- available GPU count,
- free VRAM,
- checkpoint size,
- feature-cache size,
- expected output size.

Run a real optimizer-step memory preflight before committing to a huge batch size.

Prefer:

- gradient accumulation over OOM,
- BF16 on supported hardware when stable,
- FP16 only with appropriate scaling,
- checkpoint resume,
- resumable downloads,
- cached frozen features where scientifically equivalent.

Never run multiple large experiments on the same GPU if they will interfere and corrupt timing or cause OOM.

Record peak VRAM and runtime for final candidates.

---

# 17. FAIR HYPERPARAMETER SEARCH

Do not give the proposed method dramatically more tuning than the baseline.

For each tuned hyperparameter, record:

```text
values/range tried
selection metric
number of trials
search strategy
```

Prefer small mechanistically justified searches.

Do not:

- repeatedly inspect test metrics,
- search dozens of loss weights and report only the winner without accounting for selection,
- change baseline hyperparameters to weaken it,
- compare a heavily tuned method to an untuned broken baseline.

If a baseline is known to have a stronger published configuration, use the stronger fair comparator.

---

# 18. DEVELOPMENT METRIC AND GATING

Measure baseline seed variance first.

Define a minimum meaningful improvement gate using both effect size and noise.

A reasonable adaptive rule is:

```text
min_effect = max(0.3 R@1 percentage points, 0.5 * baseline_seed_std)
```

This is a development heuristic, not a universal statistical theorem.

For a candidate to scale:

- the primary dev score should improve by at least the meaningful-effect threshold, **or**
- paired query-level analysis should show a strong, interpretable improvement concentrated on the hypothesized failure class without unacceptable global regressions.

Also require:

- neither T2V nor V2T collapses;
- R@5/R@10 behavior is inspected;
- gains are not caused by a small number of duplicated/easy examples;
- training remains numerically stable.

For borderline changes, repeat a second seed before scaling.

Kill obviously negative candidates early.

---

# 19. STATISTICAL ANALYSIS

For serious candidates report:

1. mean and standard deviation across training seeds;
2. per-query ranks;
3. paired baseline vs method rank differences;
4. paired bootstrap confidence intervals over query IDs for R@K differences when appropriate;
5. error-category improvements;
6. number of improved / unchanged / worsened queries.

When dataset structure creates groups, bootstrap the correct independent unit rather than individual duplicated captions.

Do not use a significance test mechanically. Match the test to the sampling unit and metric.

A tiny p-value with negligible retrieval gain is not a useful SOTA argument.

---

# 20. TEST-SET FIREWALL

The test split is not a development dashboard.

During research:

- train on train;
- select on dev/validation;
- mine training negatives from train only unless the benchmark explicitly permits otherwise;
- do not build thresholds from test;
- do not select checkpoints from test;
- do not choose the method after comparing many candidates on test.

Create:

```text
research/sota_slret_agent/TEST_LOCK.json
```

It should record:

```text
selected method
selected config hash
selected code commit
selected checkpoint selection rule
selected seed policy
date frozen
```

Only after this file is frozen may final test evaluation run.

If test must be touched earlier for release-checkpoint parity, label that action **baseline release parity** and do not use its numbers to tune the new method.

---

# 21. DATA LEAKAGE AUDIT

Before final claims, audit:

- duplicate video IDs across splits,
- duplicate caption IDs,
- near-duplicate samples,
- feature files generated from the wrong split,
- pseudo labels generated with test supervision,
- hard-negative tables containing test samples,
- normalization statistics computed over test,
- model selection using test,
- text augmentation that accesses test captions,
- external pretraining that contains the evaluation dataset,
- cached embeddings keyed incorrectly across splits.

If pretrained models use large public corpora with uncertain overlap, disclose the uncertainty.

For gloss-free claims, do not use gloss labels directly or indirectly unless clearly changing the comparison class.

---

# 22. REAL-WORLD CORRECTNESS CHECKS

A benchmark improvement should correspond to better retrieval behavior.

Inspect qualitative cases.

Required categories:

- clearly improved examples,
- clearly degraded examples,
- visually confusing negatives,
- semantically paraphrased negatives,
- long videos,
- short videos,
- low-pose-quality videos when pose is used.

When possible, create stress subsets based on:

- high visual confusability,
- high textual similarity,
- low pose confidence,
- long temporal duration,
- rare terms.

Do not build a “hard subset” using test labels and then tune on it.

Use dev for analysis and freeze any stress-test definition before final test reporting.

---

# 23. SIGN-SPECIFIC RESEARCH PRINCIPLES

Sign language is not generic action recognition.

The method should respect, where relevant:

- handshape,
- hand location,
- hand orientation,
- motion,
- two-hand interaction,
- upper-body pose,
- facial/non-manual cues,
- temporal ordering,
- coarticulation,
- signer variation,
- context dependence.

Do not assume semantic text similarity is equivalent to sign visual similarity.

Do not assume pose is always reliable.

Do not assume RGB global semantics preserve the fine articulatory information needed to distinguish visually similar signs.

Do not assume one-to-one deterministic alignment when language ambiguity suggests otherwise.

These are hypotheses to test, not excuses to add complexity.

---

# 24. PROMISING RESEARCH QUESTIONS TO TEST, NOT ASSUME

The literature suggests several unresolved questions.

### RQ1 — Negative distribution mismatch

Do standard contrastive batches under-sample examples that are visually confusable in signing space but linguistically dissimilar?

If yes, can a sign-aware mining policy improve fine-grained ranking without hurting coarse retrieval?

### RQ2 — Reliability-aware dual-stream fusion

Does SEDS treat pose information as equally trustworthy across samples/timesteps even when pose quality varies?

If so, can reliability-aware fusion outperform static fusion?

### RQ3 — Point estimate vs distribution

Do point embeddings become overconfident for ambiguous signs/captions?

Can uncertainty modeling help specifically on ambiguous queries without imposing excessive inference cost?

### RQ4 — Content vs context

Can content-sensitive local representation and sentence-level context be trained jointly without collapsing one into the other?

### RQ5 — Temporal information bottleneck

Are relevant local sign transitions lost by fixed uniform feature compression?

Would adaptive token selection or multiscale temporal representation improve hard queries?

### RQ6 — Interaction among mechanisms

Does sign-aware hard-negative supervision become more effective when the representation explicitly preserves pose/RGB fine-grained cues?

An interaction is scientifically more interesting than merely summing published losses, but it must be ablated.

---

# 25. DO NOT BLINDLY COPY THE LATEST PAPER

The newest published idea is not automatically the best next experiment.

For every borrowed mechanism ask:

- Is it compatible with our trusted codebase?
- Does it address a measured failure?
- Does it require extra labels?
- Does it increase modalities?
- Is its reported gain under the same split?
- Is it active at inference?
- Does it introduce major computational cost?
- Can we isolate its causal effect?

If reproducing a published method is useful as a comparator, clearly label it as a reproduction, not the new contribution.

---

# 26. CONTINUOUS RESEARCH LOOP

After the startup and parity phases, repeat this loop.

## LOOP STEP 1 — Update evidence

Read:

```text
STATE.md
LITERATURE_MATRIX.md
BENCHMARK_LEDGER.md
BASELINE_PARITY.md
FAILURE_ANALYSIS.md
experiments.jsonl
```

Check whether new relevant papers/code appeared since the last literature update if web access is available.

## LOOP STEP 2 — State the current bottleneck

Write one sentence:

```text
The dominant unresolved bottleneck is ...
```

Support it with measured evidence.

## LOOP STEP 3 — Propose or revise a falsifiable hypothesis

Do not launch an experiment without an expected outcome and failure condition.

## LOOP STEP 4 — Reviewer attack before implementation

Act as a hostile but fair reviewer.

Ask:

- Is this already published?
- Is this just more parameters?
- Is extra supervision hidden?
- Could the expected gain come from better pretraining rather than the method?
- Is the metric implementation trustworthy?
- Can a simpler control explain it?
- What ablation would disprove the claimed mechanism?

Revise the experiment accordingly.

## LOOP STEP 5 — Run the cheapest decisive experiment

Follow the GPU funnel.

## LOOP STEP 6 — Analyze, do not merely record

Compare:

- aggregate metrics,
- per-query changes,
- target failure category,
- training curves,
- numerical stability,
- compute cost.

## LOOP STEP 7 — Decision

Choose exactly one:

```text
KILL
REVISE
REPEAT
SCALE
```

Explain why.

## LOOP STEP 8 — Update research state

Update:

```text
STATE.md
experiments.jsonl
NEGATIVE_RESULTS.md
RESULTS.md
NEXT_ACTION.md
```

Then continue.

Do not restart from zero after a failed hypothesis. Preserve what was learned.

---

# 27. NEGATIVE RESULTS ARE FIRST-CLASS OUTPUTS

Create:

```text
research/sota_slret_agent/NEGATIVE_RESULTS.md
```

For each failed experiment record:

```text
hypothesis
exact change
why it was plausible
measured result
failure analysis
whether failure is conclusive
what should NOT be tried again
what could still be tested
```

This is the main mechanism preventing repeated wasted GPU runs.

Before proposing a new candidate, search your **own current-run** negative-results registry.

Do not search legacy local methods for old experiments; the user explicitly wants a fresh research process grounded in trusted upstream + current literature.

---

# 28. INTERNAL REVIEW BOARD

At the end of every substantial research cycle, write a compact four-part review.

## Researcher

- strongest evidence for the method;
- mechanism believed responsible.

## Reviewer #2

- most serious novelty concern;
- most serious experimental concern;
- strongest alternative explanation.

## Statistician

- variance;
- confidence interval;
- whether effect is stable;
- whether selection bias is likely.

## Engineer

- reproducibility status;
- runtime;
- VRAM;
- fragile dependencies;
- remaining code risks.

Finish with:

```text
Decision: KILL / REVISE / REPEAT / SCALE / FREEZE
```

Append to:

```text
research/sota_slret_agent/REVIEW_LOG.md
```

---

# 29. CLAIM–EVIDENCE LEDGER

Maintain:

```text
research/sota_slret_agent/CLAIMS.md
```

Format:

| Claim | Evidence | Experiment IDs | Counterevidence | Status |
|---|---|---|---|---|
| ... | ... | ... | ... | unsupported / provisional / supported |

No paper-style claim may appear unless it has an entry here.

Examples:

- “visual-confusability mining improves fine-grained retrieval”
- “pose confidence gating causes the gain”
- “the method generalizes across datasets”
- “the method is SOTA”
- “the method is inference-efficient”

Each requires separate evidence.

---

# 30. WHEN A CANDIDATE BECOMES A SERIOUS METHOD

A candidate is promoted only when it satisfies all of:

- baseline parity exists;
- code passes unit tests;
- pilot improvement is reproducible;
- full dev result improves;
- expected failure category improves;
- a simple control does not explain the gain;
- at least one component ablation supports the mechanism;
- no leakage detected;
- resource cost is known;
- novelty check passes.

Only then invest in broad transfer and full multi-seed experiments.

---

# 31. CROSS-DATASET VALIDATION

Do not assume a PHOENIX gain transfers to How2Sign or CSL-Daily.

After a strong primary-benchmark result, test transfer in stages:

1. verify asset readiness;
2. preserve dataset-specific official preprocessing;
3. use matched baseline;
4. use one pilot seed;
5. scale only if evidence is positive.

Do not force a universal architecture if a dataset-specific constraint is real.

Report where the method helps and where it does not.

A method that improves one dataset for a clearly explained reason can still be valuable; do not fabricate universality.

---

# 32. SOTA VERIFICATION PROTOCOL

Before writing “state of the art,” perform a fresh literature search.

For the candidate benchmark, identify the strongest published comparable result as of the current date.

Verify from the primary paper/table:

- exact T2V/V2T value,
- exact metric,
- test vs dev,
- data,
- modalities,
- supervision,
- external pretraining,
- split,
- evaluation.

Create a final comparison table with footnotes for differences.

### A valid SOTA statement must be scoped

Good:

```text
Under the RGB+pose, gloss-free PHOENIX-2014T setting with the official split and the same retrieval protocol, our method obtains X T2V R@1 and Y V2T R@1, compared with Z / W reported by [source].
```

Bad:

```text
Our method is the best sign-language retrieval model.
```

If only one direction is best, say only that.

If comparison protocols differ, say they are not directly comparable.

---

# 33. FINAL TEST PROCEDURE

Once `TEST_LOCK.json` is frozen:

1. verify code commit;
2. verify config hash;
3. verify checkpoint-selection rule;
4. verify assets;
5. evaluate final selected checkpoint(s);
6. save raw similarity/rank outputs;
7. compute official-compatible metrics;
8. compute sanity metrics;
9. compute bootstrap intervals if applicable;
10. do not tune after the result.

If a genuine implementation bug is discovered after test:

- document it;
- fix it;
- explain why it invalidates the prior run;
- regenerate the lock;
- do not hide the earlier run.

---

# 34. PAPER-QUALITY ABLATION STANDARD

The final method should have a table that lets a reviewer answer:

- What is the baseline?
- What does each component do?
- Is the gain additive?
- Is there an interaction?
- Could parameter count alone explain it?
- Could extra compute explain it?
- Could extra modality explain it?

Useful controls can include:

- equal-parameter random/additive control,
- random negatives vs proposed negatives,
- text-semantic negatives vs sign-visual negatives,
- fixed fusion vs reliability-aware fusion,
- deterministic point embedding vs uncertainty module,
- proposed temporal selection vs uniform selection.

Do not create dozens of meaningless ablations. Test the causal story of the paper.

---

# 35. COMPUTE AND EFFICIENCY REPORTING

For final models report:

```text
trainable parameters
total parameters
GPU model
GPU count
peak VRAM
training time
inference retrieval time
feature extraction cost
extra storage
```

Separate:

- one-time offline preprocessing,
- training cost,
- online inference cost.

A method that needs much more compute can still be valid, but the cost must be transparent.

---

# 36. REPRODUCIBILITY STANDARD

Follow the spirit of top ML/AI reproducibility checklists.

The final package should disclose:

- code commit;
- environment;
- dependencies;
- datasets and splits;
- preprocessing;
- training commands;
- evaluation commands;
- hyperparameters;
- search ranges;
- number of runs;
- seeds;
- compute;
- checkpoints;
- selection rule;
- failure/negative results relevant to interpretation.

Useful references:

- NeurIPS Paper Checklist:  
  https://neurips.cc/public/guides/PaperChecklist
- ICLR author/reproducibility guidance:  
  https://iclr.cc/Conferences/2026/AuthorGuide
- AAAI reproducibility checklist:  
  https://aaai.org/conference/aaai/aaai-26/reproducibility-checklist/

---

# 37. SCIENTIFIC INTEGRITY RULES

Never:

- fabricate missing metrics;
- infer an unreported paper result as if measured;
- silently substitute dev for test;
- claim a checkpoint is official without provenance;
- compare different input modalities without disclosure;
- claim a method component works if only the complete system was tested;
- select the best seed and report it as typical;
- suppress a failed seed;
- tune against test;
- report a run that crashed before full evaluation as a completed result;
- treat a paper abstract as sufficient verification of a detailed experimental claim.

Distinguish:

```text
MEASURED LOCALLY
REPORTED BY PAPER
INFERRED
HYPOTHESIZED
```

Use these labels in research notes when ambiguity exists.

---

# 38. FAILURE / BLOCKER POLICY

Do not stop the research loop for ordinary technical failures.

For:

- OOM → reduce per-GPU batch, use accumulation, profile;
- dependency mismatch → isolate env or minimally patch compatibility;
- download interruption → resume;
- missing checkpoint → locate official source and verify provenance;
- path mismatch → inspect symlinks and manifests;
- NaN → locate first non-finite tensor;
- DDP hang → reproduce on 1 GPU then small multi-GPU;
- baseline mismatch → debug data/evaluator/checkpoint before new method.

Stop only for a hard blocker such as:

- required licensed data is unavailable,
- external access requires credentials the user has not provided,
- all available GPUs are persistently unavailable,
- storage is insufficient and cannot be safely reclaimed,
- benchmark integrity cannot be established.

When blocked, produce the exact blocker and continue any research work that does not require it.

---

# 39. STOP CONDITIONS FOR THE RESEARCH CAMPAIGN

Continue the research loop until one of these is true.

## Success

A final candidate has:

- verified fair comparison,
- frozen test protocol,
- measured result competitive with or exceeding the strongest comparable published benchmark,
- reproducible code/config/checkpoint,
- ablation evidence,
- statistical analysis,
- no known leakage.

## Scientific no-go

After multiple independent, well-tested hypotheses:

- none shows a meaningful dev improvement,
- failure analysis indicates the trusted representation/data ceiling cannot be overcome under available resources,
- further experiments would only be undirected hyperparameter search.

In that case, produce a rigorous negative research report and identify the most evidence-backed next direction.

## Hard resource blocker

Required compute/data cannot be accessed safely.

Do not manufacture success to satisfy the SOTA goal.

---

# 40. REQUIRED RESEARCH ARTIFACTS

Maintain at minimum:

```text
research/sota_slret_agent/
├── STATE.md
├── ENVIRONMENT.md
├── OWNED_PATHS.json
├── ASSET_MANIFEST.json
├── UPSTREAM_CODE_MAP.md
├── LITERATURE_MATRIX.md
├── LITERATURE_REFERENCES.bib
├── BENCHMARK_LEDGER.md
├── EVALUATION_CONTRACT.md
├── BASELINE_PARITY.md
├── FAILURE_ANALYSIS.md
├── failure_cases.jsonl
├── HYPOTHESES.md
├── NOVELTY_MATRIX.md
├── experiments.jsonl
├── NEGATIVE_RESULTS.md
├── RESULTS.md
├── REVIEW_LOG.md
├── CLAIMS.md
├── NEXT_ACTION.md
└── TEST_LOCK.json
```

For the selected method maintain:

```text
methods/sota_slret_<name>/
runs/sota_slret_<name>/
artifacts/sota_slret_agent/
```

Every plot/table in the final report must be regenerable from saved machine-readable results.

---

# 41. STATE.md FORMAT

Keep `STATE.md` short enough to reread every cycle.

Use:

```markdown
# Current Research State

## Goal
...

## Trusted baselines
...

## Current best dev result
...

## Current leading hypothesis
...

## Evidence supporting it
...

## Strongest counterevidence
...

## Active experiment
...

## Last decision
...

## Immediate next action
...

## Hard blockers
...
```

Do not let research state live only in chat history.

---

# 42. RESULTS.md FORMAT

For every promoted experiment:

```markdown
## EXP-...

### Hypothesis
...

### Change
...

### Exact config
...

### Baseline
...

### Dev results
| Direction | Metric | Baseline | Method | Delta |
|---|---:|---:|---:|---:|

### Seed statistics
...

### Failure-subset analysis
...

### Compute
...

### Interpretation
...

### Alternative explanations
...

### Decision
...
```

---

# 43. NEXT_ACTION.md RULE

At the end of every work cycle, `NEXT_ACTION.md` must contain exactly the highest-value next step, plus prerequisites.

Example:

```markdown
# Next Action

Run a 5-epoch PHOENIX pilot comparing visual-confusability mining to a text-semantic hard-negative control under identical SEDS initialization.

Why: baseline failure analysis shows 31% of top-5 errors have high pose/RGB similarity but low sentence similarity.

Falsification: if the proposed miner does not improve the predefined visually-confusable dev subset and global PrimaryDev is unchanged within baseline noise, kill H03.

Prerequisites:
- metric tests pass
- miner cache hash recorded
- GPU has >= X GiB free
```

This prevents aimless branching.

---

# 44. CURRENT TRUSTED IMPLEMENTATION FACTS TO VERIFY LOCALLY

These are orientation hints, not substitutes for reading code.

## CiCo

The official approach:

- treats sign retrieval as both video–text and cross-lingual retrieval;
- uses domain-agnostic and domain-aware sign representations;
- performs fine-grained cross-lingual contrastive matching;
- includes T2V and V2T retrieval;
- provides PHOENIX-2014T, How2Sign, and CSL-Daily data paths/scripts.

The local trusted code is under:

```text
third_party/SLRT/CiCo/
```

## SEDS

The official approach:

- builds on CiCo;
- integrates pose and RGB;
- uses SignBERT-style pose encoding;
- uses semantic dual-stream fusion / Cross Gloss Attention Fusion;
- adds Pose–RGB fine-grained matching;
- provides processed I3D features, RTM keypoints, and pretrained checkpoints through its setup process.

The local setup code explicitly creates compatibility symlinks between checkpoint and dataset paths. Verify actual resolved paths before diagnosing missing assets.

The official PH training script in this repository is configured for distributed multi-GPU training, so adapt resource count through your own wrapper rather than editing the upstream script.

---

# 45. LITERATURE-DERIVED RESEARCH CONTEXT

The historical progression should guide the research question.

### CVPR 2022 — Free-form retrieval / SPOT-ALIGN

Established sentence-query sign-video retrieval and highlighted sign representation quality and annotation scarcity.

### CVPR 2023 — CiCo

Reframed the problem around sign language's linguistic structure, domain-aware sign representation, and fine-grained cross-lingual contrastive alignment.

### ECCV 2024 — UPRet

Argued that pointwise alignment underestimates uncertainty and modeled video/text representations probabilistically.

### ACM MM 2024 — SEDS

Argued that RGB-only retrieval loses local action detail and introduced a pose + RGB dual-stream representation with semantic fusion and fine-grained matching.

### TCSVT 2025 — C²RL

Focused on gloss-free sign representation pretraining with content and context representation learning and reported strong retrieval gains.

### ACL 2026 — SAN

Argued that semantic hardness is not the same as visual hardness and explicitly targeted visually confusable negatives.

### 2026 — Causality-inspired multi-grained retrieval

Explored coarse/fine cross-modal relationships and representation robustness through causal framing.

The opportunity is unlikely to be solved by adding another generic transformer layer. Look for an unresolved interaction among **representation fidelity, temporal/local structure, uncertainty, and the negative distribution actually used for supervision.**

This is a research clue, not a prescribed method.

---

# 46. OPTIONAL HIGH-VALUE DIAGNOSTIC EXPERIMENTS

Before a large architecture change, consider cheap probes such as:

### Probe A — Negative mismatch

For each dev query, compare:

- text embedding similarity to negatives,
- sign visual similarity to negatives,
- pose similarity to negatives,
- rank errors.

Measure how often the hardest wrong retrieval is visually hard but textually easy.

### Probe B — Pose reliability

Estimate pose confidence/validity and bucket dev retrieval by confidence.

If SEDS gains disappear in low-confidence buckets, static dual-stream fusion may be suboptimal.

### Probe C — Temporal shuffling

Shuffle or locally reorder video tokens at evaluation.

Measure which queries depend strongly on temporal order.

### Probe D — Local token ablation

Drop highest-motion vs lowest-motion tokens.

Determine whether uniform compression discards discriminative content.

### Probe E — Calibration

Compare score margins for correct vs incorrect top-1 retrievals.

High-confidence errors can motivate uncertainty-aware mechanisms.

Do not turn every probe into a method. Use probes to choose a mechanism.

---

# 47. METHOD DESIGN PRINCIPLE: MINIMUM SUFFICIENT CHANGE

When possible, design the first version of a candidate as the smallest intervention that tests the hypothesis.

Example logic:

```text
Observed:
SEDS makes many errors where the incorrect candidate has highly similar hand/pose motion.

Hypothesis:
The training negative distribution does not force the model to separate visually confusable sign sequences.

Minimal test:
Keep the encoder and scorer fixed; change only training negative selection/weighting using train-only sign-space confusion.

Control:
Use text-semantic hard negatives at the same rate.

Expected:
Greater improvement on visually-confusable dev errors than on easy errors.
```

Only after this causal test succeeds should you combine it with more representation changes.

---

# 48. DO NOT LET CHECKPOINT DOWNLOADS DEFINE THE RESEARCH

Official checkpoints are tools for:

- parity,
- initialization,
- teacher features,
- fair comparison.

They are not evidence of novelty.

Every external checkpoint must have:

```text
source URL
paper/repo
intended dataset
modality
training provenance if known
SHA-256
license/access note
```

Never mislabel a checkpoint trained for a different domain as a target-domain model.

---

# 49. PAPER WRITING ONLY AFTER EVIDENCE

Do not spend significant time drafting a paper before the method has survived the serious-method gate.

When ready, produce:

```text
research/sota_slret_agent/PAPER_OUTLINE.md
research/sota_slret_agent/REPRODUCIBILITY.md
research/sota_slret_agent/LIMITATIONS.md
```

A good paper narrative should emerge from:

```text
measured failure → mechanism → method → controlled evidence
```

not:

```text
invented module → retrofit motivation.
```

---

# 50. FINAL DELIVERABLE

At campaign completion produce a final report containing:

## 1. Executive summary

What was achieved and whether a SOTA claim is justified.

## 2. Research question

The measured failure and hypothesis.

## 3. Related work

Precise differences from prior methods.

## 4. Method

Equations, architecture, train/inference behavior.

## 5. Benchmark protocol

Data, split, modalities, evaluator.

## 6. Baseline parity

Proof that the local benchmark is valid.

## 7. Main results

T2V and V2T.

## 8. Multi-seed/statistical results

Variance and paired analysis.

## 9. Ablations

Mechanism evidence.

## 10. Failure analysis

What remains unsolved.

## 11. Efficiency

Parameters, VRAM, runtime.

## 12. Reproducibility

Exact commands and hashes.

## 13. Limitations

No marketing language.

## 14. Claim status

Which claims are supported, provisional, or unsupported.

Also provide a one-command or minimal-command reproduction path for the final model.

---

# 51. FIRST EXECUTION ORDER

When you first receive this file, do **not** respond with a long plan and wait.

Begin work.

Execute in this order:

1. audit machine/repository;
2. create the isolated research state directory;
3. inspect only trusted CiCo/SEDS upstream code plus your new workspace;
4. audit local assets/checkpoints;
5. build the literature matrix from primary sources;
6. build the benchmark ledger;
7. validate the evaluator with synthetic tests;
8. release-evaluate CiCo and/or SEDS for parity;
9. perform baseline failure analysis;
10. generate 3–6 falsifiable hypotheses;
11. novelty-review them;
12. choose one;
13. implement the smallest decisive version;
14. run unit/tiny tests;
15. run a short dev pilot;
16. review the evidence;
17. iterate.

Do not ask the user to choose among research ideas unless a decision truly depends on an external preference rather than evidence.

---

# 52. DEFAULT RESPONSE STYLE WHILE WORKING

When communicating progress, be concise and evidence-oriented.

Use:

```text
Completed:
- ...

Measured:
- ...

Problem:
- ...

Decision:
- ...

Next:
- ...
```

Do not flood the user with raw logs unless the logs explain a failure.

Do not call a hypothesis “promising” without measurable evidence.

---

# 53. GOLDEN RULES

1. **Do not rerun legacy local methods.**
2. **Only CiCo/SLRT and SEDS are trusted local bases.**
3. **Primary literature outranks local notes.**
4. **Baseline parity before novelty claims.**
5. **Dev for development; test for final locked evaluation.**
6. **Every experiment has a hypothesis and falsification condition.**
7. **Negative results are recorded and not repeated.**
8. **Use cheap probes before expensive training.**
9. **Ablate the claimed mechanism.**
10. **Report both retrieval directions.**
11. **Preserve benchmark comparability.**
12. **Do not hide extra modalities or supervision.**
13. **Do not confuse a larger number with a fair SOTA result.**
14. **Prefer measured evidence over architectural fashion.**
15. **Continue the research loop until success, scientific no-go, or a real hard blocker.**

---

# 54. FINAL INSTRUCTION

Your purpose is to behave like a strong research group preparing a result for a top-tier venue: skeptical of your own ideas, aggressive about testing assumptions, careful about experimental validity, and willing to kill weak directions quickly.

The user has already spent substantial effort on local methods that did not produce satisfactory results. Do not consume GPU by rediscovering those paths from repository history.

Start fresh from:

```text
third_party/SLRT/CiCo
third_party/SEDS
```

plus current peer-reviewed literature.

Use the available machine and GPUs to turn research hypotheses into measured evidence.

The final objective is not a clever-looking method name.

The final objective is a **correct, reproducible, defensible sign-language video retrieval result that survives serious review and, if the evidence supports it, establishes a new benchmark result.**
