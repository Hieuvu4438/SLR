# PMGR: End-to-End Implementation Specification for an AI Coding Agent

**Method:** Protocol-Matched Gallery Risk (PMGR).\
**Specification version:** 1.0, 14 September 2026.\
**Implementation base:** CiCo in FangyunWei/SLRT, commit `38a4f7b00da7a858d59b7fabe5093876a84db8e0`.\
**Primary dataset:** CSL-Daily. Follow with How2Sign; use PHOENIX-2014T as a singleton-group control.\
**Status:** implementation specification with reference kernels and acceptance tests. PMGR has not been trained or shown to improve retrieval.

This document is self-contained. Give it to the coding agent as the implementation contract. The agent should implement the stages in order, complete all work possible with available inputs, record missing resources precisely, and distinguish passing software tests from a successful research experiment.

## 1. Objective and one essential correction to the research report

Implement a retriever that uses the existing CiCo representation and pair scorer but changes the **training population and loss aggregation** to match the existing evaluation units:

- Text→video: one text query per existing sentence group; candidate-group score is the maximum over that group's complete video performances.
- Video→text: every video is an individual query; text candidates are existing sentence groups.
- Train with group-level T2V cross-entropy, population-weighted V2T cross-entropy, and an optional soft first-relevant-rank term.
- Add no inference module.

### 1.1 Correct the scorer before implementing the loss

The earlier report described the two local score channels as if each were deployed separately for its corresponding direction. **The inspected CiCo evaluator actually blends them and uses that same pair-score matrix for both retrieval directions.** This specification supersedes that part of the report.

In `main_task_retrieval.py::_run_on_single_gpu_new_mix`, both returned evaluation matrices are built from:

~~~python
i2t * dual_mix + t2i * (1 - dual_mix)
~~~

Both tensors have **video rows and text columns**. The second tensor is not already transposed.

Call the two unscaled local channels \(a_{iq}\) and \(b_{iq}\), and call their mixture

\[
Q_{iq}=\omega a_{iq}+(1-\omega)b_{iq},\qquad
\omega=\texttt{dual\_mix}.
\]

The inherited default is \(\omega=0.5\). Keep the reproduced value fixed for every experimental arm. Both retrieval directions must be constructed from \(Q\). A group maximum applies **after** the mixture:

\[
S_{qg}=\max_{i:g(i)=g}Q_{iq}.
\]

In general, max-of-mixture is different from mixture-of-maxima. Never substitute the latter.

Training in the pinned code also differs from a single CE on the mixed score: its `balance` option combines branch-specific row and column losses. Therefore include a **single-representative mixed-score CE control** before claiming that any gain comes from PMGR's grouping. Section 12 defines all controls.

Source: [pinned evaluation and training implementation](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/main_task_retrieval.py), [model training loss](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/modeling.py).

### 1.2 What must remain fixed

Preserve the initial visual/text weights, frozen I3D features, feature blend, temporal sampling, tokenizer, text augmentation policy, sequence encoders, embedding dimensions, optimizer recipe and validation protocol across matched arms.

No SEDS checkpoint or SEDS-produced feature may be used. No teacher-selected local support, lexical rival branch, overlap/capacity penalty, partial transport, new dataset, new split or new annotations belong in PMGR. The project's rejected R1–R5 families remain closed.

Baseline bug fixes are shared engineering changes. Record them separately; never count their effect as PMGR's modeling contribution.

## 2. Deliverables and completion levels

The coding agent must produce:

1. An isolated implementation branch/worktree with the exact upstream revision recorded.
2. A clean, reproducible CiCo baseline path.
3. A validated canonical group index and feature-provenance record.
4. Group batching, the mixed pair scorer, group risk and rank loss.
5. An exact small-batch implementation followed by memory-bounded replay.
6. A full-gallery evaluator with explicit IDs, relevance and tie policy.
7. Configurations and commands for baseline/control/PMGR runs.
8. Numerical, integration and distributed acceptance tests.
9. Checkpoint/resume support and structured experiment logs.
10. A handoff report saying exactly what was implemented, tested, run, blocked and observed.

Use three separate completion labels:

| Level | Meaning |
|---|---|
| Software ready | Core correctness tests and a real-data smoke test pass |
| Baseline reproduced | Validation, initialization and full-gallery behavior are established; comparison discrepancies documented |
| Research supported | PMGR improves matched held-out retrieval and passes its mechanism controls |

A software-ready implementation is not evidence of improved R@1. If features or official validation metadata are unavailable, implement and test the input-independent components and report the remaining blocker. Do not fabricate training data or report toy results as dataset results.

## 3. Verified repository integration points

All paths in this table are relative to the SLRT repository root and were inspected. Names elsewhere marked **proposed** are new files to create.

| Existing file | Relevant code | Required handling |
|---|---|---|
| `CiCo/CLCL/modules/modeling.py` | `CLIP4Clip`, `forward`, `get_sequence_output`, `get_visual_output`, `flip_similarity_softmax` | Expose representations and unscaled score channels; keep encoder parameters and checkpoint names |
| `CiCo/CLCL/modules/module_clip.py` | `FeatureTransformer`, `VResidualAttentionBlock`, `CLIP.encode_image`, `CLIP.encode_text` | Preserve actual feature encoder, token layout, masks and projections |
| `CiCo/CLCL/modules/until_module.py` | `CrossEn`, `AllGather` | Keep diagonal CE for the legacy control; do not reuse fixed-size gather for variable video batches |
| `CiCo/CLCL/main_task_retrieval.py` | `train_epoch`, `eval_epoch`, `prep_optimizer`, `_run_on_single_gpu_new_mix` | Reference training/evaluation behavior; remove import-time distributed initialization before reuse |
| `CiCo/CLCL/modules/optimization.py` | `BertAdam` | Preserve and log its schedule, state, decay and clipping behavior |
| `CiCo/CLCL/dataloaders/dataloader_csl_retrieval_train.py` | `csl_DataLoader_train`, `_get_text`, `_get_rawvideo` | Replace internal random-performance selection with an explicit group dataset interface |
| `CiCo/CLCL/dataloaders/dataloader_H2_retrieval_train.py` | `H2_DataLoader_train` | Same group interface for How2Sign |
| `CiCo/CLCL/dataloaders/dataloader_ph_retrieval_train.py` | `ph_DataLoader_train` | Adapt dict-valued singleton records; repair the two feature paths |
| `CiCo/CLCL/dataloaders/dataloader_csl_retrieval.py` | `csl_DataLoader` | Reference group ordering, canonical caption and feature packing |
| `CiCo/CLCL/dataloaders/data_dataloaders.py` | `DATALOADER_DICT` and loader factories | Explicit validation routing; never default validation to test |
| `CiCo/CLCL/metrics.py` | `compute_metrics`, `tensor_text_to_video_metrics`, `tensor_video_to_text_sim` | Keep legacy readout and implement an ID-based reference evaluator |
| `CiCo/README.md` | Initialization, extraction and launch instructions | Verify actual resource accessibility and hashes; README links are not evidence that files were obtained |

Suggested **new** files:

~~~text
CiCo/CLCL/pmgr_runtime.py
CiCo/CLCL/train_pmgr.py
CiCo/CLCL/evaluate_pmgr.py
CiCo/CLCL/dataloaders/group_index.py
CiCo/CLCL/dataloaders/group_dataset.py
CiCo/CLCL/dataloaders/group_batch_sampler.py
CiCo/CLCL/modules/pmgr_scoring.py
CiCo/CLCL/modules/protocol_risk.py
CiCo/CLCL/modules/pmgr_replay.py
CiCo/CLCL/modules/pmgr_distributed.py
CiCo/CLCL/pmgr_metrics.py
CiCo/CLCL/diagnose_pmgr.py
CiCo/CLCL/configs/pmgr_csl.json
CiCo/CLCL/tests/test_pmgr_scoring.py
CiCo/CLCL/tests/test_protocol_risk.py
CiCo/CLCL/tests/test_group_data.py
CiCo/CLCL/tests/test_pmgr_replay.py
CiCo/CLCL/tests/test_pmgr_metrics.py
CiCo/CLCL/tests/test_pmgr_distributed.py
~~~

Avoid duplicating CiCo's encoder into a second implementation. Add narrow interfaces or adapters that call the existing model. Stateless scoring/loss helpers should not introduce checkpoint parameters.

## 4. Stage 0A: workspace, dependencies and initialization

### 4.1 Inspect before editing

Read any applicable AGENTS.md, inspect git status, and record the current revision. Preserve user changes. Use an isolated branch/worktree when needed; do not reset an existing checkout.

Use the pinned revision for the first reproduction. If the user's working repository differs, produce a short mapping of changed functions before editing. Do not assume line numbers or launch scripts exist: the inspected CiCo tree supplies a README launch recipe, not the imagined `train_csl.sh` used in some other repositories.

### 4.2 Environment

The historical README specifies Python 3.7/PyTorch 1.7.1. That is provenance, not an instruction to force obsolete dependencies into an existing environment. For implementation, choose a compatible installed PyTorch environment, pin the exact versions, and run parity tests against the audited code behavior. Do not install an arbitrary stronger model.

Record Python, PyTorch, CUDA runtime/driver, GPU model, NumPy, tokenizer/textaugment/NLTK versions, package lockfile, deterministic settings and precision. Replace removed NumPy aliases with explicit integer dtypes as a compatibility fix. Move NLTK downloads out of dataloader import/worker construction into an explicit setup command.

Start core numerical tests in FP64 and model smoke tests in FP32. Add encoder mixed precision only after reference-gradient equivalence passes. Keep pair scoring, softmax, logsumexp and risk reductions in FP32 or FP64.

### 4.3 Required resources

Require explicit paths and hashes for:

- Official train/validation/test annotation sources and their canonical indexes.
- Domain-agnostic non-SEDS I3D checkpoint or provenance of its features.
- Target-adapted non-SEDS I3D checkpoint or provenance of its features.
- CLIP initialization and its tokenizer assets.
- A validation-selected CiCo checkpoint for continuation experiments, if used.

The README names `bsl5k.pth.tar` while the paper describes BSL-1K pretraining. A filename does not resolve this ancestry. Record the actual source and stop a same-pretraining claim if it cannot be verified.

PMGR's main experiment uses frozen offline I3D features. The **retrieval sequence encoder remains trainable**. Do not freeze it merely because the input features are frozen.

Use CiCo's existing initialization mapping. Record missing/unexpected/shape-mismatched keys during initialization and explain expected adaptations. For a trained CiCo checkpoint, require strict compatible loading; do not hide extensive mismatches with `strict=False`. Load without an optimizer step and verify that scores match before changing training.

## 5. Stage 0B: shared baseline corrections

Implement these corrections in the baseline and PMGR paths alike. Preserve an explicitly labeled legacy readout when useful.

| Risk from the inspected code | Concrete action and required check |
|---|---|
| Training evaluates the test loader to track the best epoch | Select only on official validation; make test evaluation a separate explicit command |
| PH dev file contains train+dev; loader aliases do not establish genuine validation | Verify official IDs and disjointness; Section 6 defines the gate |
| PH training loader builds both source paths from the original feature root | Build the adapted path from its own root; assert that distinct required streams are not accidentally the same file |
| Grouped evaluator uses `segment_ids[input_mask, ...]` when selecting a text mask | Select every text tensor using the same `filter_inds`; check dtype/shape and EOT-derived mask |
| Video masks use the opposite convention from text masks | Convert exactly once at the adapter boundary; Section 7 specifies the conventions |
| Inner score softmax includes padded positions | Provide `legacy_unmasked` and `valid_tokens_only` scoring modes; use one declared mode for every matched comparison |
| Legacy metric includes all score ties as multiple entries | Retain a legacy readout; use one explicit deterministic reference tie policy across matched runs |
| Distributed initialization occurs at import time | Move initialization under the executable entrypoint; CPU unit tests must import modules without starting NCCL |
| Fixed-size AllGather assumes equal leading dimensions | Do not pass variable-count group video tensors into it |
| Original feature branch has a `video_frame=-1` route to a commented-out MLP | Call the verified feature-transformer route with `video_frame=1`, not the unused MLP route |

Default PMGR experiments use `valid_tokens_only`, with the corrected baseline using the same setting. First quantify its effect relative to legacy scoring; never merge that delta into the PMGR result.

Keep the existing learned contrast scale and its post-update upper clamp. The pinned training loop clips the global gradient, and `BertAdam` also clips each parameter. Preserve this behavior for initial parity, then document any simplified clipping policy and rerun all controls under it.

## 6. Canonical annotations and official validation

### 6.1 Index schema

Create a derived **index of the existing annotations**, not a new dataset or split.

~~~json
{
  "schema_version": 1,
  "dataset": "csl",
  "split": "train",
  "source_sha256": "ACTUAL_HASH",
  "group_order": "source_insertion_order",
  "groups": [
    {
      "group_id": "EXISTING_SOURCE_GROUP_ID",
      "canonical_text": "EXISTING_TEXT",
      "videos": [
        {
          "video_id": "EXISTING_VIDEO_NAME",
          "original_feature": "ABSOLUTE_PATH",
          "adapted_feature": "ABSOLUTE_PATH"
        }
      ]
    }
  ]
}
~~~

The JSON above is a schema example, not real annotations. Preserve source IDs as strings and persist an ID→dense-index mapping. Never use Python's randomized `hash()` as a durable ID. Keep source group order for legacy parity; keep deterministic within-group video ordering for tie resolution.

CSL/How2 metadata values are lists of performance records. PH values are singleton dictionaries. Normalize only their container structure. Do not merge different existing groups because their captions happen to be identical. Do not invent paraphrase positives.

CiCo uses the first record's caption as the group's text. Verify all member captions against it. If a real discrepancy occurs, record it and preserve the source evaluator's canonical-text rule; never silently select a random member's text.

### 6.2 Required assertions

- Every group has at least one video.
- Video IDs are unique within each split, and duplicates across groups are investigated.
- Train and validation video IDs do not overlap.
- Test records are not used for training, adaptation, early stopping or hyperparameter selection.
- Both feature streams exist and have the same temporal length and extraction alignment.
- No group is silently truncated because loading or memory failed.
- Original-language text is retained as metadata; training uses the same text field as the matched baseline.

Previously computed audit anchors for the pinned release:

| Dataset/split | Videos \(N\) | Existing groups \(G\) |
|---|---:|---:|
| CSL train | 18,401 | 6,598 |
| CSL test | 1,176 | 798 |
| How2 train | 31,085 | 30,852 |
| How2 test | 2,348 | 1,969 |
| PH train | 7,096 | 7,096 |
| PH test | 642 | 642 |

These are input-integrity anchors, not instructions to fabricate missing rows. If a different legitimate release is used, explain and hash the difference.

### 6.3 Validation gate

The inspected PH `dev.pkl` has 7,615 entries including all 7,096 training IDs. The 519 remaining IDs must match the official validation split before use. CSL and How2 validation files are not automatically supplied by the inspected retrieval loaders.

Resolve official validation IDs, captions, feature paths and the existing grouping rule from authoritative metadata. Do not infer that a function named `val` loads validation. If the official protocol cannot be resolved, mark real-data training/model selection **BLOCKED**, continue implementation/unit tests, and list the exact missing resource.

A new random split is not an acceptable workaround.

## 7. Feature and text packing: exact conventions

### 7.1 Frozen feature blend

For each video, source features have shape \([T,1024]\). Verify the actual loaded shape; do not transpose based on a guess.

For `combine_type="sum"`:

\[
H(v)=\alpha H_{\mathrm{agnostic}}(v)+(1-\alpha)H_{\mathrm{adapted}}(v).
\]

CiCo's CSL README uses \(\alpha=0.8\); PH uses \(0.9\). Keep the reproduced dataset value fixed. Restrict PMGR's initial implementation to `sum`; do not silently activate concatenation or a new fusion layer.

Apply the identical temporal indices to both streams. With `feature_len=64`:

- If \(T\ge64\), use the inherited NumPy integer-linspace selection from 0 to \(T-1\).
- If \(T<64\), retain all \(T\) features and right-pad with zeros.
- Do not switch to first-64 truncation or independently sample the streams.
- Fail on empty, nonfinite, inconsistent or inaccessible valid features.

The inherited encoder input is **\([B_v,1024,64,1]\)**, not \([B_v,64,1024]\). Its legacy video mask is \([B_v,65]\):

- Position 0 is the prepended visual CLS position and is **1**.
- Real feature positions are **0**.
- Padded feature positions are **1**.

Thus the legacy video mask means **1 = excluded/padding**. The sequence encoder output includes the CLS position and has shape **\([B_v,65,512]\)** under the standard configuration.

For the new scorer, define `video_valid = legacy_video_mask.eq(0)`. The CLS token is excluded from local score aggregation and from the corrected inner correspondence softmax. Preserve the existing encoder's own mask convention; do not pass `video_valid` into code expecting a padding mask.

### 7.2 Text

Preserve the existing CLIP tokenizer and max length 32. The loader inserts SOT/EOT and, for long text, uses its existing uniform token selection before EOT rather than an arbitrary modern tokenizer truncation rule.

`CLIP.encode_text(return_hidden=True)` returns an EOT-derived mask and hidden token sequence. Its mask means **1 = valid**, including the existing SOT/EOT positions. Preserve those special-token inclusion rules. Check that there is a valid EOT and at least one valid token.

During training, tokenize the original caption and generate **one augmented caption per group per effective step**, using the inherited policy. The default random-swap branch activates with probability 0.5 in the inspected loader. Reuse that augmentation for all performances of the group and for every replay. Do not redraw it for each video or GPU chunk.

For branch compatibility:

- \(a\) uses original-text representations.
- \(b\) uses augmented-text representations during training.
- Evaluation uses original text for both.

An augmentation-off experiment is allowed only as a matched ablation. The core PMGR experiment must not secretly improve because augmentation was removed.

Preserve two training encoder forwards even when an augmentation happens to leave the string unchanged. Do not deduplicate the original/augmented hidden tensors in a stochastic training implementation unless equivalence to the original two-forward policy has been established. The two forwards share parameters, and both gradient contributions must be retained.

### 7.3 Adapter contract

Add a parameter-preserving encoder interface with named outputs:

~~~python
@dataclass
class EncodedPMGRBatch:
    video_hidden: Tensor       # [Bv, Lv, d], raw projected output
    video_valid: Tensor        # [Bv, Lv], bool True = valid
    text_hidden: Tensor        # [B, Mt, d], original text
    text_valid: Tensor          # [B, Mt], bool
    aug_hidden: Tensor         # [B, Ma, d], augmented training text
    aug_valid: Tensor          # [B, Ma], bool
~~~

Call `get_sequence_output(..., shaped=True, get_hidden=True)` and unpack **mask, hidden, pooled** in that order. Call `get_visual_output(..., shaped=True, video_frame=1, get_hidden=True)`. Do not use pooled outputs for PMGR.

Do not squeeze the batch dimension when \(B=1\). Do not detach these outputs in the direct autograd path. Detachment is permitted only inside the explicitly replayed cache algorithm.

## 8. Group dataset, collator and sampler

### 8.1 Dataset item

`GroupDataset.__getitem__(group_index)` returns the whole existing group, its canonical original/augmented text tensors, all member video IDs and all member feature tensors. The dataset length is \(G\), not \(N\).

A group item must not call the original `_get_rawvideo` unchanged: that function randomly chooses a single performance. Refactor the feature-reading/packing part into a helper accepting an **explicit video ID/path pair**, then call it for each group member.

Dataset construction may validate paths without loading all feature arrays into RAM. Cache immutable feature data if useful, but never cache trainable encoder outputs across optimizer steps.

The whole-group contract applies to PMGR and the all-performance controls. One-performance controls use an explicit representative policy on the **same full index**. Keep both full group sizes and loaded group sizes in their records; do not pass their partial batches into PMGR's complete-group loss. No experimental arm may silently truncate a group.

### 8.2 Collated batch

~~~python
{
    "group_ids": list[str],              # length B
    "video_ids": list[str],              # length Bv
    "video_to_group": LongTensor[Bv],    # batch-local text/group index
    "selected_group_sizes": LongTensor[B],
    "video_features": FloatTensor[Bv, 1024, 64, 1],
    "video_padding_mask": LongTensor[Bv, 65],  # legacy convention
    "input_ids": LongTensor[B, 32],
    "input_mask": LongTensor[B, 32],
    "segment_ids": LongTensor[B, 32],
    "aug_input_ids": LongTensor[B, 32],
    "aug_input_mask": LongTensor[B, 32],
    "dataset_group_count": int,          # global TRAIN G
    "dataset_video_count": int,          # global TRAIN N
}
~~~

This is a schema, not a literal executable Python dictionary declaration.

Flatten groups in a recorded order. For sizes [2,1,3], `video_to_group` must be [0,0,1,2,2,2]. A rectangular \(B_v\times B\) matrix is expected. Never apply diagonal CE to its video rows.

Assertions:

- `bincount(video_to_group, minlength=B) == selected_group_sizes`.
- Every group has a text candidate and all its performances.
- Video counts and IDs survive chunking and gathering unchanged.
- Dataset totals \(G,N\) are training constants, not batch counts or validation totals.

### 8.3 Sampling policy

Use a uniform shuffled permutation of existing groups per epoch and take consecutive fixed-size batches. All arms use the same seed and group permutation. Drop the final incomplete training group batch when necessary for a fixed distributed schedule; log the omitted IDs/count and reshuffle next epoch. Do not pad with duplicate groups.

A random permutation's dropped tail preserves equal inclusion probability; it is not a new dataset split. Production batches require at least two distinct groups.

Begin correctness work with \(B=2\)–8. Scale through 32/128 to the chosen effective group count, normally the reproduced 512-group reference if resources permit. Encoder microbatch size and pair-score block size must be separate from **effective group batch size**.

Never:

- Truncate large groups to fit memory.
- Balance groups by size without redefining sampling weights.
- Independently shuffle video rows and forget their group mapping.
- Count small independent contrastive minibatches as one large candidate set.
- Perform an optimizer step between chunks of one effective batch.

Use explicit per-epoch/step/group seeds for augmentation and save the sampler cursor/RNG for exact resumption. Avoid worker-scheduling-dependent augmentation if comparing paired runs.

## 9. Scoring contract

Let video outputs be \(V\in\mathbb R^{B_v\times L_v\times d}\), original text outputs \(T\in\mathbb R^{B\times M_t\times d}\), and augmented text \(T^+\). Under the standard model \(L_v=65\), \(d=512\), with only valid feature positions counted.

Normalize each projected vector along its feature dimension, with an explicit small epsilon. Excluded positions must not contribute to the correspondence softmax or outer average.

For original-text channel \(a\):

\[
C_{iq\ell m}=\widehat V_{i\ell}^{\top}\widehat T_{qm},
\quad
a_{iq}=\frac1{L_i}\sum_{\ell\in\mathcal L_i}
  \sum_{m\in\mathcal M_q}
  \operatorname{softmax}_{m}(C_{iq\ell m}/\sigma)\,C_{iq\ell m}.
\]

For augmented-text channel \(b\), use \(T^+\), softmax over valid video positions, then average valid text positions:

\[
b_{iq}=\frac1{M_q^+}\sum_{m\in\mathcal M_q^+}
 \sum_{\ell\in\mathcal L_i}
 \operatorname{softmax}_{\ell}(C^+_{iq\ell m}/\sigma)\,C^+_{iq\ell m}.
\]

Set \(\sigma=0.07\), the literal value in the audited active scorer. Do not confuse it with the outer contrastive scale or the new rank smoothing parameter.

Then compute \(Q=\omega a+(1-\omega)b\). This is **unscaled**: do not multiply by `exp(logit_scale)` inside the score adapter. CE applies that factor exactly once later. The rank term uses unscaled \(Q\).

Evaluation sets \(T^+=T\) and uses the same mixture. A positive scalar contrast scale does not alter ranks, but avoid scaling the evaluation scores unnecessarily.

### 9.1 Reference PyTorch scorer

The following is implementation reference code. It has been syntax-reviewed; the coding agent must run the specified PyTorch numerical and gradient tests in its actual environment. The present authoring environment does not contain PyTorch.

~~~python
import torch
import torch.nn.functional as F

def mixed_pair_scores(
    video, text, augmented_text,
    video_valid, text_valid, augmented_valid,
    omega=0.5, sigma=0.07,
):
    # All hidden tensors have [items, positions, dimensions].
    # Both score channels and Q have [num_videos, num_text_groups].
    if not 0.0 <= omega <= 1.0 or sigma <= 0:
        raise ValueError("Invalid fixed score mixing/smoothing value")
    if any(x.ndim != 3 for x in (video, text, augmented_text)):
        raise ValueError("Expected three-dimensional hidden tensors")
    if text.shape[0] != augmented_text.shape[0]:
        raise ValueError("Original and augmented group count differs")
    if video.shape[-1] != text.shape[-1] or text.shape[-1] != augmented_text.shape[-1]:
        raise ValueError("Embedding dimension mismatch")
    for x, valid in (
        (video, video_valid),
        (text, text_valid),
        (augmented_text, augmented_valid),
    ):
        if valid.dtype != torch.bool or valid.shape != x.shape[:2]:
            raise ValueError("Mask must be boolean, True = valid")
        if not bool(valid.any(dim=1).all()):
            raise ValueError("Every sequence needs valid positions")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Nonfinite representation")

    work_dtype = (
        torch.float64
        if any(x.dtype == torch.float64 for x in (video, text, augmented_text))
        else torch.float32
    )
    # Call this function outside encoder autocast, with autocast disabled.
    v = F.normalize(video.to(work_dtype), dim=-1, eps=1e-6)
    t = F.normalize(text.to(work_dtype), dim=-1, eps=1e-6)
    ta = F.normalize(augmented_text.to(work_dtype), dim=-1, eps=1e-6)

    c = torch.einsum("ild,qmd->iqlm", v, t)
    logits_a = (c / sigma).masked_fill(~text_valid[None, :, None, :], -torch.inf)
    per_video_position = (torch.softmax(logits_a, dim=-1) * c).sum(dim=-1)
    per_video_position = per_video_position.masked_fill(
        ~video_valid[:, None, :], 0.0
    )
    a = per_video_position.sum(dim=-1) / video_valid.sum(dim=-1)[:, None]

    ca = torch.einsum("ild,qmd->iqlm", v, ta)
    logits_b = (ca / sigma).masked_fill(~video_valid[:, None, :, None], -torch.inf)
    per_text_position = (torch.softmax(logits_b, dim=-2) * ca).sum(dim=-2)
    per_text_position = per_text_position.masked_fill(
        ~augmented_valid[None, :, :], 0.0
    )
    b = per_text_position.sum(dim=-1) / augmented_valid.sum(dim=-1)[None, :]

    q = omega * a + (1.0 - omega) * b
    return q, a, b
~~~

Performance versions may move repeated assertions outside the hot block loop, but must preserve the mathematical result. Do not replace errors with `nansum` or zero loss. Reject an invalid batch and fix its cause.

Run this scorer on video/text blocks. Merely concatenating all graph-bearing blocks does **not** bound backward memory; Section 14 gives the complete replay solution.

## 10. PMGR objective and normalization

Notation:

| Symbol | Meaning |
|---|---|
| \(G\) | Total existing training sentence groups |
| \(N\) | Total existing training videos |
| \(B\) | Groups in the effective optimization batch |
| \(B_v\) | Sum of those groups' performance counts |
| \(g(i)\) | Batch-local group index of video \(i\) |
| \(Q_{iq}\) | Unscaled mixed score, video \(i\) versus text group \(q\) |
| \(S_{qg}\) | Max mixed score over the complete videos of candidate group \(g\) |
| \(\ell\) | Existing learnable logit-scale parameter |
| \(z=e^\ell\) | Outer CE scale |
| \(\eta\) | Fixed rank-smoothing value |
| \(\lambda\) | CE/rank mixture weight |

Group reduction:

\[
S_{qg}=\max_{i:g(i)=g}Q_{iq}.
\]

T2V CE:

\[
L_{\mathrm{CE}}^T
=-\frac1B\sum_q
\log\frac{\exp(zS_{qq})}{\sum_g\exp(zS_{qg})}.
\]

V2T CE:

\[
L_{\mathrm{CE}}^V
=-\frac{G}{NB}\sum_i
\log\frac{\exp(zQ_{i,g(i)})}{\sum_q\exp(zQ_{iq})}.
\]

The V2T factor is **\(G/(NB)\)**. Do not replace it with \(1/B_v\), \(1/B\), or \(1/(Bn_{g(i)})\). Those optimize different finite-batch weightings. For uniform group inclusion, \(G/(NB)\) corrects the query population measure. It does not make the sampled softmax/rank loss an unbiased estimate of full-gallery nonlinear risk.

Define \(h_\eta(x)=\operatorname{sigmoid}(x/\eta)\):

\[
u_q^T=\sum_{g\ne q}h_\eta(S_{qg}-S_{qq}),\qquad
u_i^V=\sum_{q\ne g(i)}h_\eta(Q_{iq}-Q_{i,g(i)}).
\]

Use soft rank \(1+u\), with:

\[
L_{\mathrm{rank}}
=\frac{1}{2B}\sum_q\log(1+u_q^T)
+\frac{G}{2NB}\sum_i\log(1+u_i^V).
\]

Total loss:

\[
L_{\mathrm{PMGR}}
=(1-\lambda)\frac{L_{\mathrm{CE}}^T+L_{\mathrm{CE}}^V}{2}
+\lambda L_{\mathrm{rank}}.
\]

The rank derivative becomes small for already well-separated or strongly saturated pairs; CE supplies continued optimization. Do not learn \(\eta\), use scaled logits in this term, or add a margin/miner to revive a failed mechanism.

Start with \(\lambda=0\). This is the **population-only PMGR experiment**. Only enable the rank term after the population hypothesis passes its pilot.

## 11. Reference group loss implementation

All batch text candidates and candidate-group columns must share the same dense group order. `video_to_group` supplies video targets explicitly. The \(S\) diagonal is valid only because this alignment is asserted.

~~~python
import torch
import torch.nn.functional as F

def whole_group_max(q, video_to_group):
    # q: [Bv, B]; output: [B text queries, B candidate groups]
    if q.ndim != 2:
        raise ValueError("q must be a rectangular score matrix")
    bv, b = q.shape
    if video_to_group.dtype != torch.long or video_to_group.shape != (bv,):
        raise ValueError("Invalid video-to-group mapping")
    if bv == 0 or b == 0:
        raise ValueError("Empty batch")
    if int(video_to_group.min()) < 0 or int(video_to_group.max()) >= b:
        raise ValueError("Group target out of range")
    qt = q.transpose(0, 1)
    columns = []
    for group in range(b):
        positions = torch.nonzero(video_to_group == group, as_tuple=False).flatten()
        if positions.numel() == 0:
            raise ValueError("Candidate group has no performance")
        # Indexed max chooses the first tied maximum in the supplied member order.
        columns.append(qt.index_select(1, positions).max(dim=1).values)
    return torch.stack(columns, dim=1)

def rank_log_terms(scores, targets, eta):
    # scores: [queries, candidates], targets: [queries]
    if eta <= 0:
        raise ValueError("eta must be positive")
    positive = scores.gather(1, targets[:, None])
    is_negative = (
        torch.arange(scores.shape[1], device=scores.device)[None, :]
        != targets[:, None]
    )
    soft_exceedances = torch.sigmoid((scores - positive) / eta)
    soft_exceedances = soft_exceedances.masked_fill(~is_negative, 0.0)
    return torch.log1p(soft_exceedances.sum(dim=1))

def pmgr_loss(
    q, video_to_group, logit_scale,
    dataset_group_count, dataset_video_count,
    rank_mix=0.0, rank_eta=0.03,
):
    # Caller asserts all performances of every sampled group were loaded.
    # q is unscaled and finite. logit_scale is CiCo's scalar parameter ell.
    if not 0.0 <= rank_mix <= 1.0:
        raise ValueError("rank_mix must lie in [0, 1]")
    g = int(dataset_group_count)
    n = int(dataset_video_count)
    if not (0 < g <= n):
        raise ValueError("Invalid training population")
    if logit_scale.numel() != 1:
        raise ValueError("Expected the existing scalar logit-scale parameter")
    if video_to_group.device != q.device or logit_scale.device != q.device:
        raise ValueError("Scores, targets and logit-scale parameter must share a device")
    if not bool(torch.isfinite(q).all()):
        raise ValueError("Nonfinite scores")
    if q.dtype not in (torch.float32, torch.float64):
        q = q.float()
    s = whole_group_max(q, video_to_group)
    b = q.shape[1]
    if b > g:
        raise ValueError("Batch exceeds training group population")
    text_targets = torch.arange(b, device=q.device)
    z = logit_scale.to(q.dtype).exp()
    video_weight = g / (n * b)

    ce_t = F.cross_entropy(z * s, text_targets, reduction="mean")
    ce_v = video_weight * F.cross_entropy(
        z * q, video_to_group, reduction="sum"
    )
    rank_t = rank_log_terms(s, text_targets, rank_eta).mean()
    rank_v = video_weight * rank_log_terms(q, video_to_group, rank_eta).sum()

    ce = 0.5 * (ce_t + ce_v)
    rank = 0.5 * (rank_t + rank_v)
    loss = (1.0 - rank_mix) * ce + rank_mix * rank
    return {
        "loss": loss,
        "ce_t": ce_t,
        "ce_v": ce_v,
        "rank_t": rank_t,
        "rank_v": rank_v,
        "group_scores": s,
    }
~~~

Implementation requirements:

- This function is stateless and adds no trainable parameters.
- Keep `logit_scale` connected to the model; passing `.item()` breaks its gradient.
- Do not apply the scale in the scorer and then apply it again here.
- Use the actual \(B\) of the effective batch, never the encoder microbatch size.
- Keep the full score matrix across all current candidate groups; do not evaluate separate CE denominators inside score blocks.
- A grouped max is a max over video performances. It is not a max over frames.
- At ties, keep the within-group order deterministic. `torch.max(dim=...)` and `torch.amax` have different tie-gradient behavior; do not switch unnoticed. [PyTorch max documentation](https://docs.pytorch.org/docs/2.14/generated/torch.max.html).
- The reference uses a transparent group loop. Optimize only after it passes numerical/gradient tests.

**Singleton reduction:** if the entire dataset and batch have one video per group, \(G=N\), and the loss reduces to mixed-score bidirectional CE/rank risk. This is not automatically equal to the legacy branch-balanced training loss. Test those claims separately.

## 12. Mandatory control objectives

Every control uses the same mixed scorer \(Q\), masks, features, group IDs, augmentation and precision unless explicitly labeled legacy. Give controls the same hyperparameter search opportunity.

### 12.1 Experiment names and definitions

| ID | Definition | Purpose |
|---|---|---|
| C0 | Cleaned legacy CiCo: one performance/group, original branch-balanced loss, deployed mixed score | Strong historical implementation control |
| C1 | One performance/group; bidirectional CE on the mixed score \(Q\) | Isolates loss-of-mixture versus mixture-of-losses |
| C2 | All performances; uniform-positive T2V CE and ordinary video-query CE | Conventional all-positive pressure/exposure control |
| C3 | All performances; any-positive/logsumexp T2V CE and ordinary video-query CE | Standard set-positive control |
| C4 | All performances; group-max CE, population weighting; \(\lambda=0\) | PMGR population-only hypothesis |
| C5 | All performances; group-max CE with batch-mean V2T instead | Isolates the exact query-population factor |
| C6 | Rank training on single representatives | Generic rank-objective control |
| C7 | Full PMGR, group-max/population CE plus rank | Selected method only if C4 and rank interaction pass |
| C8 | Same all-performance scores/groups with a published ranking-loss control | Novelty defense against generic rank training |

For C0 with one representative per group and the inherited \(\texttt{mix\_design}=\texttt{balance}\), let \(\mathcal C(X)\) be row-wise diagonal CE of \(zX\). The exact branch control is

\[
L_0=\tfrac12[
\omega\mathcal C(A)+(1-\omega)\mathcal C(A^\top)
+\omega\mathcal C(B^\top)+(1-\omega)\mathcal C(B)].
\]

Here \(A_{iq}=a_{iq}\), \(B_{iq}=b_{iq}\), and both are square only because this control selects one performance per group. At \(\omega=0.5\), it is the mean of four CEs. It is generally different from \(\tfrac12[\mathcal C(Q)+\mathcal C(Q^\top)]\), which defines C1. If a reproduced baseline uses a different mixing design, implement that exact verified branch instead and record the departure from this default.

Let \(P_q=\{i:g(i)=q\}\) and \(Z_{qi}=zQ_{iq}\). Define C2 T2V:

\[
L_{\mathrm{uniform}}^T
=-\frac1B\sum_q\frac1{|P_q|}
  \sum_{i\in P_q}\log\frac{e^{Z_{qi}}}{\sum_j e^{Z_{qj}}}.
\]

Define C3 T2V:

\[
L_{\mathrm{set}}^T
=-\frac1B\sum_q
\left[
\operatorname{logsumexp}_{i\in P_q} Z_{qi}
-\operatorname{logsumexp}_{j} Z_{qj}
\right].
\]

C2/C3 ordinary V2T averages per-video CE by \(1/B_v\). Also run the strongest of them with PMGR's \(G/(NB)\) weighting to isolate T2V aggregation from V2T normalization. Call that variant explicitly `*_population_weighted`.

C5 replaces PMGR's V2T coefficient with \(1/B_v\). A separate `group_uniform_v2t` variant can use \(1/(B n_{g(i)})\), but it is not the population-correct objective.

Uniform-positive and set-positive CE are different. Do not call both “multi-positive” without saying which was used.

C8 must document how the chosen prior loss treats rectangular queries, relevance and multiple performances. The audited Smooth-AP/Recall@k implementations are not drop-in modules with matching assumptions. Do not claim faithful reproduction of either without an explicit adapter and tests.

A fully specified first C8 control is a **single-positive smooth-AP adaptation** on the same group matrix and video-query matrix. Each query has one relevant candidate under this existing grouped protocol, so exact AP is \(1/r\). With the same soft exceedances \(u^T,u^V\) from Section 10, use

\[
L_{\mathrm{AP}}
=\frac1{2B}\sum_q\left(1-\frac1{1+u_q^T}\right)
+\frac{G}{2NB}\sum_i\left(1-\frac1{1+u_i^V}\right).
\]

Replace PMGR's log-rank term by this term, keep the same CE mixture and tuning allowance, and label the run `smooth_ap_single_positive_control`. This is an explicitly described adaptation of an established ranking principle, **not a claim to reproduce the full authors' training recipe**. Its implementation can reuse the soft-rank computations, since \(1-1/r=1-\exp(-\log r)\). Keep the one-representative generic-rank C6 control as well.

When comparing C2–C8, match the **exact loaded performances and candidate groups**. Against C0/C1, report the extra video work; a larger per-step dataset exposure is a real confound.

## 13. Training engine: direct reference first

Implement a simple, single-process FP32 path before adding caching or distribution.

~~~text
load and validate the resolved configuration and provenance
construct the original CiCo model and load the declared initialization
construct the chosen baseline-compatible optimizer
for each effective group batch:
    zero parameter gradients once
    load all selected performances and one original/augmented text per group
    encode videos and both text branches with gradients enabled
    compute the complete unscaled mixed score matrix Q
    compute the selected control or PMGR loss with global training G and N
    backpropagate once
    check finite loss and gradients
    apply the recorded clipping policy
    take one optimizer step
    apply the inherited upper clamp to logit_scale
    increment effective_step once
    log detached scalars and actual exposure counts
validate only on the official validation gallery
save the checkpoint selected by the predefined validation rule
~~~

The final optimizer update must use gradients from **all** score/candidate blocks. Do not clear gradients between them.

For C0, reproduce the branch-balanced loss separately. For PMGR, do not call the original `CLIP4Clip.forward` and then add PMGR on top: that would train an undocumented mixture of objectives. Expose the encoders/scorer and explicitly choose one configured objective.

### 13.1 Optimizer contract

Verified `coef_lr=1` code path:

- `BertAdam`, learning rate \(10^{-5}\).
- Betas 0.9 and 0.98; epsilon \(10^{-6}\).
- Weight decay 0.001 for its declared decay parameter groups.
- Internal `warmup_cosine` schedule, warmup fraction 0.1.
- Global gradient clipping at 1.0 in the trainer; inherited per-parameter clipping in the optimizer.
- Existing logit scale clamped to at most \(\log 100\) after an update.
- No external scheduler is returned for this path.

Use the reproduced parameter grouping, including its literal name-based exclusions. Replacing it with modern AdamW or changing which LayerNorm weights decay is an optimizer experiment, not a neutral refactor.

The original schedule can assign zero learning rate to the initial warmup step. A one-step smoke test must check valid gradients and then several updates, rather than incorrectly declaring the first unchanged weight a failure.

Calculate schedule length from **effective optimizer steps**, not encoder microbatches, score blocks, or worker count. No second scheduler may advance the same schedule.

## 14. Exact memory-bounded training: two levels of replay

A naive group batch can exceed memory because late interaction materializes \(B_v B L_v M\) pair-position scores. Chunking alone does not solve this if all autograd graphs remain alive.

Use the following exact first-order chain-rule construction. It combines encoder replay with **score-block replay**. It is infrastructure; it is not PMGR's claimed novelty.

**Legacy-control exception:** C0 is a function of the separate channel matrices \(A,B\), not only their mixture \(Q\). To cache C0 exactly, retain \(A,B\) as the scalar payload, differentiate its branch-balanced loss with respect to both, and replay \(\langle A_{\mathrm{block}},D_A\rangle+\langle B_{\mathrm{block}},D_B\rangle\). Obtain the logit-scale derivative once as usual. Do not feed C0 through a \(Q\)-only cache. All C1–C8 mixed-score objectives use the \(Q\)-based algorithm below. A dedicated C0 direct/cache parity test is required if its cached path is implemented.

### 14.1 Level A: encoder caches

1. Keep the model in its training mode and leave weights unchanged for the effective step.
2. Load/tokenize/sample augmentation once.
3. Encode video and original/augmented text microbatches under `no_grad`.
4. Record the RNG state immediately before each forward and retain the exact inputs.
5. Cache raw projected hidden outputs and masks. Do not cache only pooled embeddings.
6. Do not carry these trainable representations into the next optimizer step.

The frozen input I3D feature cache is different: those immutable inputs may persist across steps.

### 14.2 Level B: obtain gradients with respect to scalar scores

1. Compute \(Q\) in pair blocks under `no_grad` from the cached hidden outputs.
2. Assemble only the scalar \(B_v\times B\) matrix.
3. Create `q_leaf = Q.detach().requires_grad_(True)`.
4. Evaluate the full PMGR/control loss using `q_leaf` and the live logit-scale parameter.
5. Obtain \(D_Q=\partial L/\partial Q\) and \(d_\ell=\partial L/\partial\ell\) with `autograd.grad`.
6. Retain \(D_Q\), the detached loss diagnostics and \(d_\ell\); release the scalar-loss graph.

The group max, all candidate denominators and rank comparisons are evaluated globally in this stage. Never approximate them per block.

### 14.3 Replay pair blocks to obtain representation gradients

Initialize gradient buffers for cached video/original-text/augmented-text hidden outputs to zero.

For each block of video rows \(I\) and text columns \(J\):

1. Create separate leaf tensors from the raw cached hidden slices.
2. Recompute the exact unscaled pair-score block \(Q_{IJ}\).
3. Form the scalar surrogate \(\sum_{i\in I,q\in J}Q_{iq}(D_Q)_{iq}\), treating \(D_Q\) as detached.
4. Differentiate this surrogate with respect to the three hidden slices.
5. Add the resulting derivatives into their full representation-gradient buffers.
6. Release the block graph before the next block.

A text group participates in many video blocks; its gradient must **accumulate**, not be overwritten. Video representations likewise receive contributions from every text block and both score channels.

The only learned score parameter in this contract is the outer logit scale, whose derivative was computed once in the scalar stage. Do not replay it through \(Q\), which is unscaled. If an implementation introduces any other learned scorer parameter, it violates the parameter-free scorer contract and needs an explicitly revised gradient treatment.

### 14.4 Replay encoders and update

For each original encoder microbatch:

1. Restore its recorded RNG state inside a context that also restores the surrounding RNG afterward.
2. Recompute its hidden outputs with autograd enabled and the same precision.
3. Backpropagate their dot product with the detached cached representation derivatives.
4. Accumulate into the original model parameters.
5. Replay both text branches even though they share the same text encoder parameters; their contributions add.
6. Add \(d_\ell\) to the model's logit-scale gradient **exactly once**.
7. Clip and update only after all encoder replays are complete.

No extra multiplication/division by chunk count is allowed. The loss normalization is already in \(D_Q\).

### 14.5 Replay pseudocode

~~~text
optimizer.zero_grad()
cached_hidden, masks, replay_states = encode_all_no_grad(batch)

Q = score_all_blocks_no_grad(cached_hidden, masks)
q_leaf = detached_leaf(Q)
loss = objective(q_leaf, model.clip.logit_scale, batch)
dQ, dlogit = grad(loss, [q_leaf, model.clip.logit_scale])

hidden_grads = zero_buffers_like(cached_hidden)
for video_slice, text_slice in complete_pair_block_schedule:
    v, t, ta = detached_leaf_slices(cached_hidden)
    q_block = mixed_pair_scores(v, t, ta, corresponding_masks)
    block_surrogate = sum(q_block * dQ[video_slice, text_slice].detach())
    dv, dt, dta = grad(block_surrogate, [v, t, ta])
    accumulate_slices(hidden_grads, dv, dt, dta)

for encoder_input_chunk, rng_state, hidden_grad in replay_schedule:
    with restore_recorded_rng(rng_state):
        hidden = encode_with_grad(encoder_input_chunk)
    sum(hidden * hidden_grad.detach()).backward()

add_logit_gradient_once(dlogit)
clip_and_step_once()
~~~

This pseudocode omits error handling and shape checks, not mathematical terms. The coding agent must implement and test the full lifecycle.

### 14.6 Why this is exact, and when it is not

With fixed parameters, replayed inputs and RNG, the chain rule gives:

\[
\nabla_\theta L
=\sum_k
\left(\frac{\partial R_k}{\partial\theta}\right)^\top
\frac{\partial L}{\partial R_k}.
\]

Detaching the cached upstream derivative is correct for a first-order training update. It does not provide higher-order gradients.

Exactness fails if replay changes dropout, augmentation, sampling, batch-dependent statistics, precision or model weights; if a shared-parameter contribution is omitted; or if the score graph contains additional state not replayed.

The existing feature Transformer uses LayerNorm and default zero attention dropout in the inspected code, but do not assume future ports remain stateless. Test the actual model. For a stochastic equivalence test, use the same encoder microbatch schedule and saved RNG in the reference path; a single giant stochastic forward need not consume RNG identically.

Keep the first replay implementation FP32. If using mixed precision later, apply one consistent scaling convention. Do not scale cached gradients with GradScaler and then scale the replay surrogate a second time. BF16 encoder autocast, where supported, avoids FP16 gradient scaling but still requires an equivalence/tolerance test.

Audited reference for encoder replay: [GradCache implementation](https://github.com/luyug/GradCache/blob/906f03835fbc183132a9db32612a9e8f180ca3b4/src/grad_cache/grad_cache.py) and [RNG context](https://github.com/luyug/GradCache/blob/906f03835fbc183132a9db32612a9e8f180ca3b4/src/grad_cache/context_managers.py). The scalar-score replay above is the explicit SLRet integration design; do not claim it is provided unchanged by GradCache.

### 14.7 Memory/performance accounting

The persistent per-step data are approximately:

- Hidden caches/gradients: \(O(B_vL_vd+B(M_t+M_a)d)\).
- Scalar score/gradient matrices: \(O(B_vB)\).
- One pair-block graph: \(O(c_v c_t L_v M)\).
- One encoder microbatch's activations during replay.

This avoids keeping the full pair-position tensor or every encoder graph. It increases forward work. Report actual memory and wall time; do not promise a speedup.

Encoder and score chunk sizes are memory controls, not mathematical hyperparameters. Changing them must preserve the reference loss/gradient within documented numerical tolerance.

## 15. Multi-GPU implementation without hidden gradient scaling

First pass all single-process tests. The simplest correct distributed design for this replay engine uses **replicated models and explicit gradient summation**. It does not combine DDP automatic averaging with custom reductions.

### 15.1 Global batch and variable sizes

1. Construct one global shuffled group batch of \(B\) groups.
2. Assign complete groups to workers without duplication. For the first version, require \(B\) divisible by worker count.
3. Each worker gets the same number of groups, but may get different \(B_v\).
4. Assign global batch-local text column IDs before partitioning; record each worker's group/video ranges.
5. Encode local inputs without gradients and keep local replay states.
6. Gather counts, pad hidden caches along the **item** dimension for transport, gather them to rank 0, then discard transport padding using the counts.
7. Gather masks and group maps with exactly the same ordering. Padded transport rows must never become retrieval candidates.

Fixed temporal/text lengths make item-axis padding sufficient. If variable lengths are introduced later, explicitly pad and propagate masks; do not silently alter positional encoding.

### 15.2 Central score-gradient stage

Rank 0 constructs the full current-model score matrix and performs Sections 14.2–14.3, using global \(B,G,N\). It returns each worker's representation-gradient slices. Variable-sized outputs may again be padded for transport and unpadded locally.

Only rank 0 owns the scalar score loss and \(d_\ell\). Set local logit-scale contribution to zero on the other ranks before reduction. This avoids counting that derivative \(W\) times.

This central score stage can become a throughput bottleneck. Accept it as the first correct distributed version; optimize query partitioning only after single/global equivalence is proven.

### 15.3 Encoder gradients and synchronization

Each worker replays its local encoders with the received **global-loss derivatives**, obtaining a local contribution to the common model gradient.

Then:

\[
g_\theta=\sum_{r=0}^{W-1}g_{\theta,r}.
\]

Use `all_reduce(..., SUM)` once for each globally active trainable parameter, in identical parameter order on all workers. **Do not divide by world size.** The full objective was already normalized on rank 0.

Handle parameters with no gradient carefully:

- Exchange/merge an active-gradient flag in a deterministic order.
- If a parameter is active on another worker, contribute a zero tensor locally before summation.
- If it is globally inactive, keep `grad=None`. Replacing every unused gradient with zero would trigger optimizer decay/state updates that did not occur in the baseline.
- Apply global clipping only after gradient summation.
- All workers take the identical optimizer update, with identical optimizer state and scale clamp.

Use an explicit non-DDP model for this engine. Do not also wrap it in DistributedDataParallel or attach automatic reduction hooks. Verify model and optimizer-state checksums remain synchronized.

### 15.4 Required distributed checks

On a tiny deterministic model with uneven group sizes:

- 1-worker and 2-worker global loss agree.
- Every active parameter gradient agrees.
- The logit-scale gradient is neither doubled nor halved.
- The actual next optimizer update agrees.
- Permuting worker ownership does not change the result.
- A globally unused parameter remains unused.
- An invalid batch on any worker leads to a coordinated failure, not another worker hanging in a collective.

The audited CiCo `AllGather.backward` only slices the local gradient. Do not assume it supplies this reduction/scaling contract. Any future DDP/local-query optimization must derive its scaling anew and pass the same tests.

Use the supported collectives for the installed version and preserve collective order on every rank. [PyTorch distributed documentation](https://docs.pytorch.org/docs/2.14/distributed.html).

## 16. Full-gallery evaluation and inference

Evaluation is a separate code path with `model.eval()`, no augmentation, no gradients, no training sampler, no dropped final batch and no random performance selection.

### 16.1 Cache the entire split

Encode each distinct text group once and every video once. Retain explicit IDs. The order may follow the canonical index, but correctness must not depend on an accidental diagonal in a video/text matrix.

Compute \(Q_{\mathrm{eval}}\in\mathbb R^{N_e\times G_e}\) from the original caption for both channels.

V2T:

- Query rows: all \(N_e\) videos.
- Candidate columns: all \(G_e\) text groups.
- Correct column: the video's existing group ID.

T2V:

- Text query \(q\), candidate group \(g\):
  \(S^{\mathrm{eval}}_{qg}=\max_{i:g(i)=g}Q^{\mathrm{eval}}_{iq}\).
- Rank candidate groups, not a newly deduplicated raw-video gallery.
- Correct group is the query's existing group ID.

A saved encoder architecture is unchanged. Recompute its encoded gallery after training; do not reuse stale trainable hidden states from an earlier checkpoint. The frozen I3D input files remain reusable.

### 16.2 Ranking and ties

For each query, obtain the first correct candidate rank \(r\), starting at 1. Report:

\[
R@k=100\,\operatorname{mean}(r\le k),\quad
\mathrm{MedR}=\operatorname{median}(r),\quad
\mathrm{MnR}=\operatorname{mean}(r).
\]

Use a stable lexicographic rule: descending score, then the persisted candidate-order index. Do not use target-aware tie breaking. Record tie incidence and the tie-policy identifier.

For the new reference evaluator, define MedR as the usual middle-value median with the mean of the two middle values for even counts. Keep the legacy metric output separately where its convention differs.

Legacy versus reference metrics can differ on exact-caption ties. That is an evaluation-accounting difference, not a learned improvement. Publish matched-arm results under the same policy and retain the legacy readout to explain comparability.

### 16.3 Memory and caching

Evaluation may stream score blocks into a scalar matrix or a memory map. Group maxima must include **all** performances, even when a group spans blocks. Initialize streaming maxima to negative infinity, then update by max. Do not average block maxima or forget the last block.

The existing CSL test count is 1,176 video queries and 798 text queries/candidate groups. The How2 test count is 2,348 video queries and 1,969 text groups. Verify counts from the actual pinned index every run.

Do not call validation on a random 512-candidate subset and label it full-gallery recall.

## 17. Checkpointing, resumption and run provenance

Save atomically after a completed effective optimizer step. A checkpoint/run ledger must contain:

- Model state, optimizer state and schedule state.
- Effective step, epoch, sampler permutation/cursor and all relevant RNG states.
- Selected checkpoint score, validation criterion and validation protocol ID.
- Full resolved config, git commit, patch/diff identity and dependency versions.
- Dataset/group index and feature/checkpoint hashes.
- Score policy, \(\omega,\alpha,\sigma,\eta,\lambda\), augmentation policy and precision.
- Effective group count, replay/chunk settings and distributed engine mode.
- Training groups/videos processed so far, wall time and validation history.
- Whether this is `resume_exact` or `weights_only_continuation`.

For `resume_exact`, reject changed split IDs, feature hashes, score policy or optimizer schedule unless the operation is explicitly reclassified as a new experiment. Test that an interrupted small run matches an uninterrupted run.

For the first research pilot, use **weights-only continuation from one validation-selected baseline checkpoint**, resetting optimizer/schedule identically in all arms. This avoids unequal inherited moments when comparing new objectives. Record that choice and include the baseline's pretraining/initial training cost in the total experiment account.

Never initialize one candidate from its best-performing rival's checkpoint without applying the same training history to the control.

## 18. Configuration and command-line interface to implement

All filenames/commands in this section are **proposed new interfaces**. The coding agent must implement them; they are not existing upstream commands.

### 18.1 Example first-pilot configuration

The null paths are intentional required inputs. Audit mode must explain missing inputs; train mode must reject unresolved paths. Populate the baseline-argument file from the pinned reproduction, not from guessed settings.

~~~json
{
  "spec_version": "pmgr-1.0",
  "upstream_commit": "38a4f7b00da7a858d59b7fabe5093876a84db8e0",
  "dataset": "csl",
  "seed": 0,
  "paths": {
    "train_index": null,
    "validation_index": null,
    "test_index": null,
    "baseline_resolved_args": null,
    "initialization_checkpoint": null,
    "output_dir": "runs/pmgr_csl_seed0"
  },
  "initialization_mode": "weights_only_continuation",
  "data": {
    "feature_mix": "sum",
    "feature_mix_alpha": 0.8,
    "max_features": 64,
    "max_text_tokens": 32,
    "text_augmentation": {
      "enabled": true,
      "kind": "random_swap",
      "probability": 0.5
    },
    "expected_train_groups": 6598,
    "expected_train_videos": 18401,
    "drop_incomplete_group_batch": true
  },
  "scoring": {
    "mask_policy": "valid_tokens_only",
    "dual_mix": 0.5,
    "inner_temperature": 0.07,
    "normalize_eps": 0.000001,
    "maximum_contrast_scale": 100.0
  },
  "loss": {
    "mode": "group_ce",
    "rank_mix": 0.0,
    "rank_eta": 0.03
  },
  "training": {
    "effective_groups": 512,
    "epochs": 20,
    "optimizer": "inherited_BertAdam",
    "learning_rate": 0.00001,
    "coef_lr": 1.0,
    "warmup_fraction": 0.1,
    "beta1": 0.9,
    "beta2": 0.98,
    "epsilon": 0.000001,
    "weight_decay": 0.001,
    "clipping_policy": "legacy_global_and_parameter",
    "gradient_accumulation_steps": 1
  },
  "engine": {
    "mode": "two_level_replay",
    "distributed": "none",
    "video_encoder_microbatch": 8,
    "text_encoder_microbatch": 32,
    "score_video_block": 8,
    "score_text_block": 16,
    "precision": "fp32"
  },
  "validation": {
    "every_epochs": 1,
    "primary": "mean_t2v_v2t_r1",
    "secondary": "mean_bidirectional_r5_r10",
    "tie_break": "earliest_epoch",
    "metric_policy": "stable_candidate_order_v1"
  }
}
~~~

This is a **20-epoch continuation pilot**, not a replacement claim about CiCo's full training schedule. For baseline reproduction/full training, use the documented fixed schedule, commonly 200 retrieval epochs in the audited configuration, with the actual effective-step count recorded.

Chunk sizes are starting memory settings, not validated optimal values. If memory fails, decrease chunks first. If the effective group count must change, label the run as a new matched-compute/batch setting and change all controls.

Reject unknown config keys and contradictory options. Examples: `group_ce` with nonzero rank mix; a one-performance policy with PMGR's complete-group objective; a frozen sequence encoder; a scorer whose mix differs from evaluation; a test-selected initialization.

### 18.2 Required CLI commands

From `CiCo/CLCL`, support the following after filling the config:

~~~bash
python pmgr_runtime.py audit --config configs/pmgr_csl.json

python -m pytest tests/test_pmgr_scoring.py tests/test_protocol_risk.py tests/test_group_data.py tests/test_pmgr_metrics.py -q

python train_pmgr.py --config configs/pmgr_csl.json --engine direct --effective-groups 4 --max-updates 3 --output-dir runs/pmgr_smoke

python -m pytest tests/test_pmgr_replay.py -q

python diagnose_pmgr.py --config configs/pmgr_csl.json --mode population --output-dir runs/pmgr_diagnostics

python train_pmgr.py --config configs/pmgr_csl.json --loss-mode all_uniform_ce --output-dir runs/control_uniform_seed0

python train_pmgr.py --config configs/pmgr_csl.json --loss-mode all_set_ce --output-dir runs/control_set_seed0

python train_pmgr.py --config configs/pmgr_csl.json --loss-mode group_ce --output-dir runs/group_ce_seed0
~~~

Only after the population-only gate:

~~~bash
python train_pmgr.py --config configs/pmgr_csl.json --loss-mode pmgr --rank-mix 0.25 --rank-eta 0.03 --output-dir runs/pmgr_rank_seed0
~~~

After distributed equivalence tests:

~~~bash
torchrun --standalone --nproc_per_node=2 train_pmgr.py --config configs/pmgr_csl.json --distributed manual_sum_replay --output-dir runs/pmgr_two_gpu
~~~

For validation and final selected testing:

~~~bash
python evaluate_pmgr.py --config configs/pmgr_csl.json --checkpoint PATH_TO_SELECTED_CHECKPOINT --split validation

python evaluate_pmgr.py --config configs/pmgr_csl.json --checkpoint PATH_TO_SELECTED_CHECKPOINT --split test --final-test
~~~

`PATH_TO_SELECTED_CHECKPOINT` is a placeholder to replace. Define every CLI override explicitly and persist the fully resolved config. The training command must never silently run final test evaluation.

For one-performance controls, support `legacy_cico`, `single_mixed_ce` and `single_rank` modes using the same full index but an explicitly declared uniform representative-selection policy. Store full group cardinalities separately from loaded cardinalities. Only these controls may load a single representative; the PMGR collator assertion still requires all group members.

## 19. Required tests with concrete expected outcomes

Do not rely on a successful forward pass. The following tests target ways an apparently working implementation can train the wrong objective.

### 19.1 Numeric loss fixture

Use FP64 and the following **synthetic** unscaled score matrix:

~~~python
q = torch.tensor([
    [0.8, 0.2, 0.0],
    [0.1, 0.7, 0.3],
    [0.4, 0.9, 0.2],
    [0.2, 0.0, 0.95],
    [0.1, 0.3, 0.85],
    [0.3, 0.2, 0.7],
], dtype=torch.float64, requires_grad=True)
owner = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.long)
ell = torch.tensor(0.6931471805599453, dtype=torch.float64, requires_grad=True)
out = pmgr_loss(q, owner, ell, 3, 6, rank_mix=0.25, rank_eta=0.1)
~~~

Expected group score matrix:

~~~text
[[0.80, 0.40, 0.30],
 [0.70, 0.90, 0.30],
 [0.30, 0.20, 0.95]]
~~~

An independent NumPy computation produced:

| Quantity | Expected value |
|---|---:|
| T2V CE | 0.559557424510826 |
| Population-weighted V2T CE | 0.667108048676086 |
| T2V mean log soft rank | 0.047084965497191 |
| Population-weighted V2T log soft rank | 0.182867776418063 |
| Total loss | 0.488743645184499 |
| Derivative with respect to log scale \(\ell\), finite difference | -0.235680363391833 |

Require FP64 scalar agreement within \(10^{-8}\); allow approximately \(2\times10^{-6}\) when comparing autograd to the supplied finite-difference derivative.

The unscaled-score derivative for the first two rows is approximately:

~~~text
[[-0.1647319492,  0.0255603241,  0.0168597596],
 [-0.1112705270,  0.1955881752,  0.0858617589]]
~~~

These are oracle values, not training results. The complete synthetic matrix yields T2V R@1 = 100% and V2T R@1 = \(5/6\times100\%\). Video row 1 has correct text group 0 at rank 3. V2T MnR = \(4/3\); MedR = 1.

### 19.2 Scoring tests

1. **Orientation:** output shape \(B_v\times B\) with unequal counts. Compare every element with an explicit valid-position loop.
2. **Mask polarity:** input video mask conversion excludes CLS/padding; text mask retains valid tokens. A deliberate polarity reversal must fail the fixture.
3. **Padding invariance:** replacing finite excluded hidden vectors with arbitrary large finite values leaves scores and valid-position gradients unchanged.
4. **No valid positions:** fail explicitly; do not return NaN/zero score.
5. **Chunk invariance:** all pair block sizes yield the same scores within precision tolerance.
6. **Mixture order:** for one group with channel scores \(a=[0.9,0.1]\), \(b=[0.1,0.9]\), \(\omega=0.5\), the group score is **0.5**, not 0.9.
7. **Scale isolation:** changing \(\ell\) changes CE but not \(Q\), group ordering or rank-only loss.
8. **Legacy parity:** in eval mode with all positions valid and nonzero, compare the unscaled mixed adapter to the existing implementation divided by its positive outer scale. For padded inputs, compare each declared mask policy to its corresponding reference, not to an incompatible policy.

### 19.3 Group/risk tests

1. Sizes [2,1,3] create the correct group matrix and explicit V2T targets.
2. Entire singleton population reduces to the corresponding mixed-score CE/rank objective.
3. With \(G=10,N=30,B=2,B_v=5\), the V2T coefficient is **1/6**, not 1/5.
4. Reordering videos preserves loss and metrics when their IDs/mapping move with them.
5. Reordering groups/text columns preserves loss after dense target remapping.
6. Known groups with identical text remain distinct unless the official protocol itself merges them.
7. \(\lambda=0\) equals group CE exactly; no hidden legacy loss is still added.
8. The group max's T2V derivative reaches its winning performance only away from ties.
9. A weak nonwinning positive performance still receives V2T supervision. In the numeric fixture, `q[1,0]` has zero T2V-CE derivative but a negative total derivative.
10. Deliberately omitting a group member fails the PMGR collator check.

Use `torch.autograd.gradcheck` on small FP64 scores with unique group maxima; the max is nondifferentiable at ties, so test tie behavior separately.

### 19.4 Replay/optimizer tests

- Direct graph versus two-level replay: scalar loss, every representation gradient, every active parameter gradient, logit-scale gradient and next optimizer update.
- Use an encoder with **shared text parameters across original/augmented branches** to catch omitted contributions.
- Compare identical encoder chunk/RNG schedules when stochastic layers are present.
- Verify pair-block derivative buffers accumulate across all columns and rows.
- No optimizer step occurs until the effective batch is complete.
- No stale representation is reused after an update.
- Exact resume matches an uninterrupted short run.
- Run FP32 first; then set separate, justified mixed-precision tolerances.
- Test declared clipping/schedule semantics, including initial warmup behavior.

The authoring check independently compared finite-difference parameter derivatives with the two-level chain-rule reconstruction in a small NumPy model with shared text weights. Maximum absolute differences were below \(3\times10^{-10}\). **This does not replace PyTorch/autograd, CUDA or distributed tests.**

### 19.5 Evaluation/distributed tests

- Full split retains the last partial evaluation batch and every group member.
- Streaming group maxima equal dense reduction when groups cross block boundaries.
- Stable tie handling is independent of query correctness.
- Legacy/reference metric differences are recorded on deliberate ties.
- Distributed tests in Section 15 pass for unequal per-worker video counts.
- A complete real-data smoke run logs valid counts, finite gradients, checkpoint reload and identical reloaded evaluation scores.

## 20. Diagnostics that must accompany the pilot

### 20.1 Population and representative instability

At a validation-selected checkpoint, do not update weights. Use fixed existing **training** groups for gradient diagnostics, and full validation galleries for held-out error analysis.

For each chosen group set:

1. Load all performances and compute its score tensor once with fixed text views.
2. Draw several uniform single-performance representatives using recorded seeds.
3. Compare the representative competitor/group ordering to the complete-group maximum.
4. Compare score-space gradients for representative CE, conventional positive-set CE and group CE on a common full tensor, embedding absent representative rows as zero derivatives.
5. Report direction cosine, norm ratio and which groups/performances account for differences.
6. Relate the measured discrepancy to changes in full-gallery validation errors after the short continuation.

Do not treat a large gradient difference as proof of improvement. It must predict the errors that the intervention actually corrects.

### 20.2 Weak-performance coverage

For every existing validation group, report individual-video V2T ranks and its within-group positive score spread. Compare distributions before/after training and by group size.

Required question: did T2V improve by concentrating on an easy performance while other valid videos became worse queries? A group-average chart alone can hide this failure.

### 20.3 Ranking geometry and compute

Record positive score minus strongest legitimate negative score in each direction, query-level rank changes, group-max winning frequency and exact ties. Use existing IDs and labels.

Log exposures and compute per effective step:

~~~json
{
  "effective_step": 0,
  "loss_mode": "group_ce",
  "effective_groups": 512,
  "loaded_videos": 0,
  "candidate_pairs": 0,
  "ce_t": 0.0,
  "ce_v": 0.0,
  "rank_t": 0.0,
  "rank_v": 0.0,
  "contrast_scale": 0.0,
  "gradient_norm_before_clip": 0.0,
  "encoder_forward_calls": 0,
  "score_block_replays": 0,
  "seconds_per_update": 0.0,
  "peak_allocated_gpu_bytes": 0
}
~~~

This is a log schema example; zeros are placeholders. Real logs must contain actual measurements. Detach tensors before logging to avoid retaining autograd graphs.

Track cumulative original video performances encoded, text views, optimizer steps and accelerator time. PMGR's all-performance batch has more video work than one-representative training. Use same-performance controls plus exposure-matched and wall-time comparisons; one number cannot equalize all resources simultaneously.

## 21. Experiment sequence and research stop/go criteria

### Phase A — Engineering and baseline

Complete all reference tests and a real-data smoke run. Establish validation-only CiCo behavior under the declared feature and scoring contract. Reproduce a separate cleaned UPRet comparator with the same feature ancestry if resources permit.

Do not add UPRet's transport/distribution branches to PMGR to make the implementation “stronger.” UPRet is initially a separate comparator. Porting PMGR into its rectangular training tensors would require a separate audit.

### Phase B — Minimum causal experiment

Start all arms from the **same** validation-selected baseline weights, reset optimizer/schedule identically, and run one seed for at most 20 continuation epochs.

Run C1, C2, C3 and C4; retain the C0 baseline/continuation reference. Use exactly the same all-performance tensors in C2–C4 and shared group schedules/augmentations across arms. Include the population-weighted version of the stronger positive-set control.

Do not add the rank term yet. The question is whether whole-group/population CE improves standard full-gallery validation beyond the matched positive controls.

**Stop the population hypothesis** if the mismatch is negligible, if its correction does not transfer to held-out ranking, or if weak-performance V2T degradation is the source of apparent T2V gain.

### Phase C — Rank extension only after Phase B passes

Run the fixed seven-setting grid:

- \(\lambda=0\), once.
- \(\lambda\in\{0.25,0.5\}\) crossed with \(\eta\in\{0.01,0.03,0.07\}\).

Keep \(\omega,\alpha,\sigma\), architecture, augmentation and optimizer fixed. Compare C7 to C4, C6 and the justified C8 ranking control. If a known generic rank objective explains all improvement, withdraw the PMGR-specific novelty claim.

### Phase D — Replication and transfer

Use seeds 0, 1 and 2 for final paired runs; report mean and standard deviation. Validate on How2Sign with locked or equally budgeted settings. PH is useful as a singleton control; a gain there may reflect generic score/rank changes rather than PMGR's distinctive group mechanism.

Use a fixed full schedule for final training after the pilot. Select by mean T2V/V2T validation R@1; use mean bidirectional R@5/R@10 only for the declared tie break, then earliest epoch. Final test evaluation occurs after those choices.

A planning threshold is \(\delta=\max(0.5\text{ R@1 points},\text{baseline seed SD})\) in mean bidirectional validation R@1. This is a resource-allocation threshold, not a predicted gain or proof of significance. Report both directions and uncertainty; do not hide an adverse direction behind the mean.

### Phase E — Decision

Proceed toward a paper only if:

- The population intervention helps beyond equal-input positive controls.
- The rank extension has an independent, defensible contribution.
- Full-gallery gains replicate beyond one conveniently grouped dataset.
- Improvements are not explained by masking fixes, the mixed-score training control, extra exposure, larger candidate count or stronger pretraining.

If these conditions fail, preserve the tested code, baseline and negative results. Do not add a local-evidence branch, new teacher, lexical miner or OT module to rescue PMGR. No numerical SOTA gain is promised by this specification.

## 22. Ordered implementation work packages

Each work package should end with code, its focused tests and a short status note. The agent should not expand scope while a previous acceptance gate is failing.

| Package | Concrete changes | Acceptance gate |
|---|---|---|
| WP0: repository/runtime | Isolate checkout; pure importable runtime; dependency/provenance log; baseline factory | Import on CPU without distributed side effects; checkpoint mapping explained |
| WP1: annotations/features | Canonical index, official validation checks, explicit feature reader and masks | Correct existing counts, disjoint IDs, source feature alignment and pack parity |
| WP2: scoring/evaluation | Named encoder outputs, unscaled mixed scorer, stable full-gallery evaluator | Orientation, mixture-order, masks, padding and dense/streamed metrics pass |
| WP3: loss reference | Group max, population CE, rank loss, C1–C5 controls | Numeric fixture and autograd tests pass |
| WP4: direct trainer | Single-process training, inherited optimizer, validation selection, checkpoints | Real-data smoke and exact resume pass |
| WP5: replay | Encoder caches, scalar-score derivatives, block replay, encoder replay | Direct/cached parameter gradients and next update agree |
| WP6: distributed | Count-aware cache transport, central score derivatives, manual gradient sums | 1-worker/2-worker update equivalence and synchronization pass |
| WP7: diagnostics/pilot | Logs, fixed group diagnostics, equal-input C2/C3/C4 continuation | Mechanism decision is supported by recorded results, or direction is closed |
| WP8: optional full study | Rank grid, strongest baseline, seeds, cross-dataset evaluation | Locked test selection and defensible full-gallery comparison |

Use verified existing files only where their behavior is understood. Mark newly introduced modules in the implementation README. Avoid broad cleanup unrelated to these correctness risks.

## 23. Troubleshooting guide

| Symptom | Likely cause to inspect first |
|---|---|
| Loss works only when \(B_v=B\) | Diagonal CE or accidental equal-row assumptions |
| Scores change when only excluded padding changes | Wrong mask polarity, missing inner mask, or zero/NaN normalization |
| Adapter differs from evaluator on valid inputs | Wrong channel transpose, omitted score blend, double scale or special-token policy |
| T2V improves while weak videos collapse | Group-max effect without sufficient V2T supervision; verify exact weighting and diagnose the hypothesis |
| All-group loss too small/large | Used \(1/B_v\), per-worker totals or chunk size instead of \(G/(NB)\) |
| GPU memory still grows across score blocks | Autograd graphs retained until the end; implement scalar-score/block replay |
| Text encoder gradient too small | Augmented branch omitted, shared gradients overwritten or text blocks not accumulated |
| Cached training differs from direct training | Changed RNG/precision/augmentation, early optimizer step, stale cache or doubled loss scaling |
| Multi-GPU gradients differ by worker count | DDP averaging combined with manual sums, global loss repeated, or logit derivative duplicated |
| Some unused parameters start changing | Converted globally absent gradients into zeros, activating optimizer decay |
| Checkpoint reload cannot reproduce evaluation | Missing config/feature/tokenizer identity, wrong group order or stale encoded gallery |
| Apparent gain disappears with C1 | Changing branch-loss averaging to CE on the blended score explained it |
| Apparent gain disappears with C2/C3 | More performances/positive handling explained it |
| Rank term helps but group CE does not | Generic ranking effect; reject the population claim rather than renaming it |

## 24. Copy-ready instruction for the coding agent

Use the following as the launch instruction together with this entire document:

> Implement PMGR according to PMGR_Implementation_Specification.md, using the pinned CiCo code as the base. Read the full specification before editing. Preserve the existing encoder and non-SEDS feature lineage. First establish a clean baseline and official validation routing, then implement canonical groups, the evaluator's mixed pair score, the population-only loss and its required controls. Run the numeric fixture and gradient tests before real training. Add exact score/encoder replay only after the direct path is correct; add distribution only after single-process equivalence. Keep dataset totals, masks, candidate IDs, score scaling and gradient reductions explicit. Do not use the earlier report's separate-direction scorer shorthand; this specification's blended score is authoritative. Do not add a local-evidence, teacher, negative-mining or transport mechanism. Complete all unblocked coding and tests autonomously. If required data/checkpoints are absent, record exact blockers and finish all input-independent work; never fabricate a successful reproduction. End with a changed-file list, exact commands run, test results, unresolved limitations, actual experiment results if any, and the next justified action.

The agent's final handoff should include:

- Which existing functions changed and why.
- All new modules and their public interfaces.
- Resolved configuration and provenance ledger.
- Passing/failing tests, with numerical discrepancies where relevant.
- Baseline and method metrics only when actually measured.
- Actual compute/exposure differences.
- Whether WP7 supported or killed the mechanism.
- No claim of “SOTA achieved” without a verified fair comparison.

## 25. Verification boundary and primary references

**Completed while writing this specification:** inspection of the pinned source paths; correction of the blended-score interpretation; independent NumPy checks of the loss fixture, score masking/blocking, permutation behavior and two-level chain rule with shared text parameters.

**Not completed here:** PyTorch kernel execution, CUDA/distributed tests, baseline training, feature extraction, a full model implementation or a PMGR retrieval experiment. Those are explicit coding-agent deliverables above.

Primary sources:

1. [CiCo paper and supplement](https://arxiv.org/abs/2303.12793).
2. [Pinned CiCo model and loss entrypoints](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/modeling.py).
3. [Pinned feature/text encoder and mask behavior](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/module_clip.py).
4. [Pinned trainer, blended evaluator and optimizer construction](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/main_task_retrieval.py).
5. [CSL training loader](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/dataloaders/dataloader_csl_retrieval_train.py).
6. [CSL evaluation loader](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/dataloaders/dataloader_csl_retrieval.py).
7. [How2 training loader](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/dataloaders/dataloader_H2_retrieval_train.py).
8. [PH training loader](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/dataloaders/dataloader_ph_retrieval_train.py).
9. [Legacy metrics](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/metrics.py).
10. [BertAdam implementation](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/optimization.py).
11. [Inherited gather and diagonal CE](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/until_module.py).
12. [GradCache paper](https://aclanthology.org/2021.repl4nlp-1.31/), [audited code](https://github.com/luyug/GradCache/tree/906f03835fbc183132a9db32612a9e8f180ca3b4).
13. [Smooth-AP](https://arxiv.org/abs/2007.12163), [audited loss](https://github.com/Andrew-Brown1/Smooth_AP/blob/59927ec4c59565bff209301d2c2b449ed1937806/src/Smooth_AP_loss.py).
14. [Recall@k surrogate](https://openaccess.thecvf.com/content/CVPR2022/papers/Patel_Recallk_Surrogate_Loss_With_Large_Batches_and_Similarity_Mixup_CVPR_2022_paper.pdf), [audited loss](https://github.com/yashvarpatel/RecallatK_surrogate/blob/ed052029d258555df2f94dd82d6f7df60ef7cc6f/src/losses.py).
15. [UPRet paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06074.pdf), [separate baseline source](https://github.com/xua222/UPRet/tree/046366227417e1d8ec14145965403462df345984).
16. [PyTorch max/tie behavior](https://docs.pytorch.org/docs/2.14/generated/torch.max.html), [distributed collectives](https://docs.pytorch.org/docs/2.14/distributed.html).

This specification is a concrete implementation contract for testing PMGR. Its acceptance tests establish software correctness; its controlled experiments determine whether the research hypothesis deserves to survive.
