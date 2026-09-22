# Bounded implementation and comparison audit — Cycle 3

2026-09-22. AI-assisted source verification using academic-research-suite inline.
This completes a targeted mechanism pass over the remaining named baselines,
not an independent reproduction, exhaustive literature review or new experiment.
No training, model initialization, dataset contents, TEST scoring, downloads,
active-job checks or upstream edits occurred in this pass.

## Provenance boundary

`git -C <directory> rev-parse --show-toplevel HEAD` for SLRT, SEDS, UPRet, SAN
and CMCM resolves to the parent `/home/haipd/SLR` at
`53b5f986d74cfeb5f4cc83649eeb44241cab6ea7`, not independent upstream repositories.
Thus these commands cannot certify the historical upstream pins as current
pristine checkouts. Historical pins remain ancestry records; current-file SHA256
snapshots below identify exactly the inspected code. No reset/fetch was performed.

## UPRet: actual local path [V]

Entry point: `third_party/UPRet/main_task_retrieval.py` imports
`modules.modeling.CLIP4Clip`, not the similarly named alternative modules.
`get_text_video_feat` supplies contextual video/text tokens. The inspected
`flip_similarity_softmax` receives [Bv,F,D] / [Bt,W,D], video masks with zero
valid, text masks with one valid, and augmented text during training.

Its learned token MLPs produce masked softmax outer weights. Normalized token
dot products form [Bv,Bt,F,W]. Inner softmax temperature is .07. The current
local source excludes invalid inner positions, unlike the inspected native
CiCo path; this includes historical local repairs and is not pristine-paper code.
Outer weights reduce each channel to [Bv,Bt]. At evaluation the two channels
are mixed by `dual_mix` in `_run_on_single_gpu_new_mix`.

During training only, `dist_text_trans` and `dist_video_trans` produce means
and log-scales; `sample_num=2` includes the mean and one stochastic sample.
Masked temporal pooling yields [K,B,D], followed by normalization and all
sample-pair similarities. A detached Sinkhorn plan uses epsilon .1 and at most
100 iterations. The current reduction averages directional maxima of
transport-weighted similarities, divided by K; it is not the ordinary total
transport-weighted sum. Both logit channels receive the same auxiliary score.

For row cross-entropy C, channels I and T (both video×text), and balanced d:

    L = .5 * [d C(I) + (1-d) C(I.T) + d C(T.T) + (1-d) C(T)]

In training I/T contain scaled deterministic scores plus `ot_weight * S_ot`;
evaluation omits sampling/transport. Positives are the batch diagonal in this
loss, not a semantic-equivalence relation. Training adds O(Bv*Bt*K²) sample-pair
storage/transport on top of token interaction; deployed token matching remains
O(Bv*Bt*F*W*D). Encoder/pretraining costs are separate.

[A/V] The primary paper explicitly makes OT training-only (§3.6, Eq24 and
efficiency paragraph); its omission at inference is intentional, not a defect.
The paper's transport sum and current source's max reduction differ. Historical
real-gradient screening already failed the investment gate for repairing that
reduction; do not reopen it from this rereading.
[Primary paper](https://arxiv.org/html/2405.19689v1).

The current native entry point constructs the test loader and uses its R1 for
selection (lines804–806,868–873). Do not launch it unchanged under DEV-only
rules. Local `train_ph.sh` is a modified one-device launcher, not evidence of
the published training recipe. The historical step767 checkpoint is partial;
its old low recall cannot characterize fully trained UPRet. No checkpoint was
loaded or revalidated here.

## SAN: distinguish mining, trainer, and evaluation [V]

`datasets.py::load_features` reads cached [T,1024] features and produces
[B,64,1024] plus validity masks by linspace subsampling or zero padding.
`models.py::ImageCLIP` projects1024→768, runs a configured mBART encoder,
then projects768→512. `TextCLIP` uses German BERT and768→512. These are not
the CLIP encoders implied by the class names. The paths inspected contain no
gloss-target or pose input. Their pretrained encoders and language differ from
the local English-CiCo setup; initialization availability was not tested.

`generate_hard_negatives` loads an external word-replacement table, substitutes
up to two eligible words, and falls back to other batch captions. The checked
tree exposes consumption, not construction/provenance of that table. Original
captions may then receive a random-swap augmentation. Historical table-exposure
audits remain authoritative; do not create an unverified substitute miner.

`SLRCLIP.forward` normalizes contextual tokens, gathers positives and generated
texts, and forms soft token interactions with temperature .07. Inner pooling
is unmasked; outer averages are masked. Positive global channels are [B,B]
in their respective orientations, while each video's hard row is [1+K], with
its positive in column zero. `utils.CrossEn` selects the diagonal for coarse
loss and column zero for hard loss:

    L = .5 * [CE(I2T, diagonal) + CE(T2I, diagonal)]
        + lambda * CE(hard_I2T, column_zero)

The supplied launcher uses K=5, lambda=.4, four ranks × batch64, SGD .01 and
100 epochs. The parser default lambda is zero; recipe identity matters. This
loss adds a video→generated-text task, not symmetric new video negatives.
All-pair training construction costs O(B²(1+K)FW D); only each owner's hard
captions enter its selected hard row. No FLOPs or memory were measured.

`evaluate` computes full-gallery channel scores separately and assumes square
diagonal positives. It does not establish a CSL grouped/multi-positive contract.
It uses raw `logit_scale` instead of its exponential, unlike training; under
a positive scale this is a common rescaling and not by itself a rank effect.
No learned scale sign or runtime effect was checked. More materially, `main`
uses `test_label_path`, then calls that loader for epoch selection despite
`dev_*` variable names. The YAML's DEV path is not consumed and the collator
has train/test tokenizer branches only: a valid DEV adapter must be explicit.
Do not infer the authors' actual experimental behavior solely from this release.

[A] SAN's PH fine-grained task ranks an original caption against40 generated
negatives. Table1 reports CiCo stress R1 17.9→39.4, but standard T2V/V2T
69.2/70.1→68.1/67.8. This supports the specified stress-task gain, not a
standard-gallery frontier improvement or proof that model capacity never
matters. Its sign–word mining uses learned visual similarity and word identity,
not independent linguistic contrast labels.
[Paper methods/Table1](https://arxiv.org/html/2607.09263v1),
[verified ACL record](https://aclanthology.org/2026.acl-long.1302/).

## C²RL: separate paper mechanism from derivative implementation

[A/V] Pretraining combines CLCL/InfoNCE content alignment with autoregressive
translation likelihood. Downstream retrieval feeds extracted visual features
and text to two independently parameterized mBART encoders and trains CLCL
retrieval scores. Thus translation supervision plus contrastive pretraining
is an established baseline mechanism, not new merely when attached to SLRet.
TableVI reports PH78.7/77.6, CSL90.3/88.4 and H2 62.4/57.5 T2V/V2T R1;
these are author results, not matched local DEV scores. Reading scope this
cycle: §§III-B/C and TablesVI/VII, not the entire final published version.
[Primary preprint](https://arxiv.org/html/2408.09949v1).

The fully read historical derivative-source and task/resource audits identify
`sltbaselines` as an independent translation reimplementation. Its attributed
contrastive kernel is not an official trained retrieval release. Do not repeat
availability lookups or train SLT as a retrieval proxy. See
[source audit](../../proposal7/evidence/autonomous_search/C2RL_reimplementation_source_result.md)
and [task gate](../../proposal7/evidence/autonomous_search/C2RL_retrieval_resource_gate.md).

## CMCM: source components are not an integrated comparator

[A] Publisher search content describes augmentation adjustment, Gaussian
cross-modal alignment, and temporal-motion covariance pooling. The direct
publisher open failed; no fresh full-paper/table access or numeric rank claim
is made. [Publisher source](https://www.sciencedirect.com/science/article/pii/S1077314225003546).

[V] Entire local `Encoder.py`, `CSA_Module.py`, `CCG_Module.py` and
`TMCP_Module.py` were read. The encoder constructs an R(2+1)D classifier despite
its `i3d_encoder` name; its supplied checkpoint argument is unused. CSA references
undefined `DEVICE`. CCG's registered [1,1,1024,1024] mask is sliced on its first
two axes, not its time axes. Its diagonal-Gaussian loss has asymmetric epsilon
placement. TMCP consumes [B*T,C,H,W], applies learned temporal/spatial/channel
operations, covariance pooling, square root and triangular vectorization, but
references `MPNCOV` without importing it in this file. These are source facts;
historical CPU certificates—not a new execution—establish the narrower runtime
failures and numerical discrepancies.

No complete training loss, negative definition, deployed retrieval scorer,
selection policy or trained integration is established by these components.
Do not invent missing implementation details, build a replacement pipeline,
or attribute failures of this release to every reported author experiment.

## Consequence for admission

The six named baselines now have a bounded local/paper mechanism comparison.
Remaining unknowns matter for final fair frontier claims but do not require
exact-paper reproduction before further research. The strongest usable local
control remains adapted SEDS GCN-R1; extra resources must be disclosed.

Historical Q01/Q22 source notes were read fully this cycle. Generic probe
positive controls and scalar-score/rank-dimension impossibility arguments have
already been audited. The Cycle2 next-step text was too broad: do not repeat
those analyses as a new experiment. A next diagnostic needs a new measured
phenomenon, specific admissible intervention and an open decision consequence.
The current source findings do not supply a SUPPORTED-FOR-PILOT mechanism.

## Current source snapshots (SHA256)

```text
771452c11c06228b8b54302eead86645fc9875b64f6f6fce858e0ce2e2f926d0  third_party/UPRet/modules/modeling.py
4763707032ec612f0e029473fe029ea7a907d701e76eb98b1e10345615d8deab  third_party/UPRet/main_task_retrieval.py
baef3ea99dffb8b3115331cf2de0ebc7027145a562ec11a31758094e252afa39  third_party/SAN/models.py
c0ca118fe2bf56512a5b042363fdffc1493408a69d8ad766ac2950d52bca6e7c  third_party/SAN/datasets.py
891021c36e14becd1be32778a2dd12ee47bf2102f27011a516517c644c6bc8a3  third_party/SAN/utils.py
6e15cd95d41deeee12c7848c37787cdb4a674857412b5db4a5735c44c158917a  third_party/SAN/train_vlp_v2.py
970c28246ff86bf82a0c69c4732a7fded0218b4b85672c17dc5b63a5b66e6874  third_party/SAN/configs/config_gloss_free.yaml
c98bd62482166a970340109fc0d6df89ecd6601dfeb8531140fb0b300202ac93  third_party/SAN/train.bash
862dbd02d1e82cc4947d11e06bde0fc686ca9524cfe690720d4c1b3c2a8b8ff1  third_party/CMCM/modules/Encoder.py
277a06c5bf4a5f256b550fe3c0b88a9e09135cca39052afa8a4976db7b4129f7  third_party/CMCM/modules/CCG_Module.py
87383ffd07b8db702896235fa6b8e94101c6866757f23f5c683533a1507e0eb1  third_party/CMCM/modules/CSA_Module.py
d1188f79866d6d1ff1c3021acdd1b1ee274c4901529998baf7fac3717bc28e00  third_party/CMCM/modules/TMCP_Module.py
```

External searches actually issued: exact SAN title plus `arxiv`; exact CMCM
title plus `retrieval`. Primary arXiv pages for UPRet/C²RL/SAN and ACL metadata
were opened. Search snippets from secondary sites were discovery-only.
No systematic bibliometric, retraction or conflict-of-interest audit is claimed.
