# C²RL-derived release: retrieval task and resource gate

2026-09-16. ANALYZED. AI-assisted academic-research-suite source verification;
bounded fact-check, not an original-C²RL reproduction or full literature review.

## Decision

The independent sltbaselines release supplies an attributable contrastive kernel
inside a translation pipeline, but the checked locations do **not establish a
compatible trained C²RL retrieval system**. Keep the
[padding certificate](C2RL_reimplementation_source_result.md) as kernel evidence.
Do not interpret it as measured retrieval harm, run translation as a retrieval
proxy, or launch another masking experiment. This completes the second C²RL
source/provenance turn; change mechanism rather than repeat availability checks.

## Task correspondence

The original [C²RL preprint, §III-C](https://arxiv.org/html/2408.09949v1#S3.SS3)
describes downstream retrieval with separate visual-input and text-input mBART
encoders, CLCL similarities and contrastive training. Its pretraining combines
contrastive and language-model objectives. A shared contrastive kernel alone
therefore does not establish the downstream retrieval training or evaluation
path. Final published-version parity remains unverified.

Pinned independent repository:
`ozgemercanoglu/sltbaselines@f741330d0b1e2e8e9a602d7ce4ff6e2e06a4ac13`.

| Inspected evidence | What it establishes |
|---|---|
| [README lines 81–133](https://github.com/ozgemercanoglu/sltbaselines/blob/f741330d0b1e2e8e9a602d7ce4ff6e2e06a4ac13/README.md#L81) | C²RL pretraining and subsequent **SLT** fine-tuning use `train_slt.py`; stage two freezes the feature extractor. |
| [SLT evaluator lines 587–706](https://github.com/ozgemercanoglu/sltbaselines/blob/f741330d0b1e2e8e9a602d7ce4ff6e2e06a4ac13/train_slt.py#L587) | Computes contrastive/translation losses, generates sentences, decodes and evaluates BLEU. This is not full-gallery retrieval ranking. |
| [VLP evaluator lines 420–458](https://github.com/ozgemercanoglu/sltbaselines/blob/f741330d0b1e2e8e9a602d7ce4ff6e2e06a4ac13/train_vlp.py#L420) | Evaluates batch contrastive and masked-LM losses; does not accumulate a retrieval gallery or report recall. |
| [Model lines 395–464](https://github.com/ozgemercanoglu/sltbaselines/blob/f741330d0b1e2e8e9a602d7ce4ff6e2e06a4ac13/models/models.py#L395) | Separate text encoder and visual encoder states feed the v2 kernel; the forward path also invokes the translation decoder. Kernel relevance is real, but task parity is not established. |

Full-text keyword scans of the two trainers and model file found no `retriev`,
`recall`, `r@1`, `r@5`, or `median_rank` matches. This corroborates the inspected
evaluators; keyword absence alone is not proof of repository-wide absence.

## Resource and execution boundary

Read-only GitHub API checks returned one branch (`main`, same pin), no tags and
no releases; all list responses had no pagination link. The recursive tree was
untruncated. No trained checkpoint file or retrieval-specific entry point was
identified in that tree. The full README contains checkpoint placeholders, not
a trained retrieval download. Its linked
[GFSLT-VLP preparation README](https://github.com/zhoubenjia/GFSLT-VLP/blob/main/pretrain_models/README.md)
describes trimmed mBART initialization and configuration resources, not trained
C²RL retrieval weights. That linked document was read at mutable `main` and
content-hashed; its linked weights were not downloaded or inspected.

**Do not run the documented SLT `--eval` command here:**
[lines 411–419](https://github.com/ozgemercanoglu/sltbaselines/blob/f741330d0b1e2e8e9a602d7ce4ff6e2e06a4ac13/train_slt.py#L411)
select `test_dataloader`, with DEV calls commented out. This is normal released
evaluation behavior, not an allegation of leakage, but violates this project's
DEV-only research boundary if executed unchanged. No upstream code was run.

No dataset/annotation contents, predictions/references, checkpoints, GPU,
training, SEDS assets or benchmark TEST examples were accessed. Tree/config TEST
path names were read as metadata only. No assets or replacement pipeline created.

## Read scope, provenance and uncertainty

This pass read the entire README, full VLP evaluate function, selected SLT
evaluation/entry-point lines, model constructor/forward sections and PH stage-two
config; it scanned the three Python texts for the stated keywords. Not all
repository source was manually audited. API endpoints were `/branches`, `/tags`,
`/releases` (each `per_page=100`) and `/git/trees/{pin}?recursive=1`.

New raw-file SHA256s:

- `train_vlp.py`: `46f2f3ad452038667d3e5d12fb985b835648b2cbcec1f2f873b25506d155bd5e`
- `configs/phoenix/flallm_config1_stage2.yaml`: `ac5643672151e65c27818000c3953e497a6436fb2f01596459879413665262b2`
- linked GFSLT preparation README: `d2b2f988541e951a40967513371871a95d4a456fd211f6ad60d7c346c42162a2`

README, model and SLT trainer hashes matched the previous audit. Retrieval
checkpoints may exist elsewhere or privately; no global absence claim follows.
Repository attribution is not independent author-byte provenance. Source evidence
is strong for this implementation's task contract, absent for real retrieval
damage. Paper/repository evidence does not independently replicate efficacy.
Bibliometric/retraction/COI investigations were not performed; no certification
of those properties is claimed. No new candidate, Proposal8, GO or global barrier.
