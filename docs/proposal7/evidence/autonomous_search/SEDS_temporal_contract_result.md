# SEDS enabled matching loss: temporal correspondence is an input contract

2026-09-16. academic-research-suite / source verification, inline. AI-assisted;
ANALYZED, no independent human review. No data, pretrained assets or model use.

## Result and decision

The enabled Pose–RGB clip-index matching loss assumes corresponding temporal
windows across streams. The inspected loader **does not verify that assumption**:
it checks clip counts, not RGB-to-pose source-frame membership. This does **not**
establish that the authors' features are misaligned. Their offline preparation
may satisfy the contract exactly.

Do not treat this as a measured bottleneck, launch an alignment repair, or fetch
SEDS assets to settle it. The current prohibition remains binding. A new sampler,
OT alignment or pose-fusion module would neither validate the input contract nor
automatically create a materially new method. No Proposal8, Q38 or GO.

## Source trace

Revision434e3f714fcb6a7d1f4001fb9a246bbd93ec0246. PH TRAIN path traced through
dataset factory, item loading, pose processing, collator, training call and model:

| Stage | Source and observed behavior |
|---|---|
| RGB input | `dataloader_ph_retrieval_train_pose.py:120`: load `item['feature']`, preserve every feature's stored order, pad to feature_len; no frame IDs or clip starts consumed |
| Pose frame selection | `GetTotalFrameList:441`: optional temporal subsampling, union of usable left/right-hand frame lists, numerical frame-name sorting |
| Pose windows | `_get_pose_clips:248`: window/stride on the filtered sequence, last-frame padding for short sequences, uniform window-index selection if above feature_len |
| Cross-stream check | `__getitem__:116`: compare sums of RGB and right-hand masks; no timestamp comparison |
| Batch transport | Collator:650 onward carries RGB features and pose start arrays; training call:392 forwards body start array/mask and RGB features |
| Model use | `modeling.py:get_sign_output:271` slices pose features using body starts; RGB features pass through without a temporal remapping |
| Matching objective | `modeling.py:523–539` uses equal token indices as corresponding clips |

The equality check is meaningful for tensor compatibility. Both masks are valid
prefixes, so equal counts also imply equal slot-validity masks in this path.
It is not meaningless; it simply does not establish equal source-frame windows.

For the supplied window16/stride1/feature_len64, pose sequences with at least79
retained frames all produce64 selected windows. Different retained-frame lists
can therefore pass the same count check. This follows directly from the source
formula and cap, not from a census of the authors' features. No error frequency
or performance bound follows.

The right/left/body streams are derived from a shared retained-frame list and
the same deterministic clipping function. Equal-length assertions plus that
shared construction are stronger than a generic sum-only check; no actual
within-pose desynchronization is alleged. Missing-hand handling was read, not
executed. An unused randomly drawn `start` does not by itself shift the sampled
frame lattice: the actual slice starts at0.

## Counterevidence and scope

The [paper §3.1](https://arxiv.org/html/2407.16394v1#S3.SS1) describes frame filtering,
16-frame windows and uniform clip selection before both encoders. This supplies
an intended shared preparation procedure. It is a concrete alternative to the
hypothesis that offline RGB and online pose use unrelated windows. Read scope:
§3.1 and previously inspected §3.3, not a new full-paper review.

Targeted searches of the pinned Python tree did not identify an offline RGB
feature producer that could certify this mapping. The loader/README expect
processed assets. No claim that preprocessing is unavailable everywhere, no
public resource sweep, and no asset link was followed. Merely finding a consumer
without provenance checks is not evidence that its input producer was wrong.

AST comparison (exit0) found `_get_rawvideo` and `_get_pose_clips` structurally
identical across PH, H2 and CSL TRAIN loaders (ignoring source locations/comments).
That establishes these two routines' equality only, not equality of complete
datasets/preprocessing. No synthetic feature permutation or loss demonstration
was run: such a fixture cannot reveal the actual offline clip provenance.

## Fingerprints and access boundaries

SHA256 of inspected files:

```text
PH TRAIN loader f090565bd99158b3fc74a5c7e2a053800f18c87882e23faae834aa5068dedaff
H2 TRAIN loader 5b156b209aaba8779a8c4991bfa6da0012d639b3343d81b6c5bae1cd618a482e
CSL TRAIN loader 5069c75fbb05fd987d2590341bca4656af8ec2fbbbac5405f2a085c8d2fee563
modeling.py 62bd9aac7076e54a7ed15de947ca7437c5b7c78ec3b50216f9d733d1d09880c9
main_task_retrieval.py 3ebc8c8b0699c615e8c13b5370477720e273eaa49f2a3005051727e60b08bbdd
data_dataloaders.py 8ff42f3ad46bac1d99f651adfde91c9d8acb17d7d04f860be2e7c6f8a5e45707
README.md bbc687986e38e1ed23f358abfaf43ca85e6f2cf492e3b412785473bc0f0439fb
```

Source only. No dataset constructor, feature pickle, annotation, model,
checkpoint, GPU, optimizer, DEV/TEST data/evaluation or upstream edits. Factory
source contains evaluation-loader definitions; reading them is not executing
them or accessing their data. Primary code supports the contract description;
empirical synchronization, prevalence and causal retrieval harm remain unknown.

This follows the [loss activation gate](SEDS_loss_activation_result.md) with a
new check of its actual supervision prerequisite. It closes neither all SEDS
questions nor the overall research search. After these two source-only SEDS
turns, move to a different mechanism with admissible empirical evidence rather
than repeat the missing-provenance analysis or repackage closed alignment methods.
