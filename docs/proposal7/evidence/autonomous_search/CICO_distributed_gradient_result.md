# CiCo: distributed gradient scaling is not a local error explanation

2026-09-16. ANALYZED, AI-assisted academic-research-suite source verification.
No new method or efficacy claim. Previous goal turn classified as progress:
C²RL task/resource evidence changed the next permissible action. This turn
changes mechanism to distributed optimization fidelity, not another mask audit.

## Verified contract

Pinned SLRT commit `38a4f7b00da7a858d59b7fabe5093876a84db8e0`:
[AllGather backward](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/until_module.py#L265)
returns only the local feature slice, without reducing gradient contributions.
[The scorer](https://github.com/FangyunWei/SLRT/blob/38a4f7b00da7a858d59b7fabe5093876a84db8e0/CiCo/CLCL/modules/modeling.py#L468)
gathers all features and applies a replicated learned logit scale. Each rank then
computes the full global four-CE objective under balance/dual_mix=.5. The trainer
uses ordinary DDP and does not multiply the loss by world size. The README's
training command requests four processes; this is not a historical author-run log.

For identical model replicas, fixed global features and W equal shards, write
the single-process encoder gradient as `g = sum_r g_r`. The source backward gives
rank r only `g_r`, and standard DDP averaging gives `g/W`. The replicated scalar
temperature receives its entire gradient on every rank and remains unchanged
after averaging. Thus multiplying the whole loss by W cannot restore *both*
parameter classes. DDP's averaging behavior is documented by
[PyTorch](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html).
This conditional derivation assumes the same global forward values; it does not
equate differently sampled, stochastic or numerically different real runs.

## Executed fixed fixture

[Protocol](CICO_distributed_gradient_protocol.md),
[script](../../../../methods/information_probe/distributed_gradient_contract.py),
[machine result](CICO-DISTRIBUTED-GRADIENT.json). Source SHA guards passed.
AST-extracted AllGather/CrossEn and scorer were executed without changing their
expressions. Only forward communication was emulated; replica gradients were
arithmetically averaged. This was **not** actual multiprocess DDP/NCCL execution.

| World size | Visual gradient / reference | Text gradient / reference | Temperature gradient / reference |
|---|---:|---:|---:|
| 1 | 1 | 1 | 1 |
| 2 | .5 | .5 | 1 |
| 4 | .25 | .25 | 1 |

All nine fixed world-size/control cases passed coordinate checks at1e-10.
Maximum discrepancy was1.39e-17; unscaled forward loss was identical across ranks.
Scaling gathered-feature gradients restores all three reference gradients;
scaling the whole loss restores encoders but scales temperature by W. These are
algebra controls, not trained improvements. Execution exit0, .87 seconds, CPU
FP64, torch2.11.0+cu128; no optimizer step or dataset/checkpoint access.

## Exposure check and decision

`runs/ph_base_b512_s42/implementation_provenance.json` records ordinary
`python -m elsc.train ... --device cuda:0` at implementation commit
`39449e18def6b154944ceeaed39dfd5c570882a3`. Historical `elsc/upstream/factory.py`
constructs CLIP4Clip directly rather than the classmethod that enables distributed
gather. Its parent constructor initializes `distributed=None`. Historical
trainer/factory/bridge scans found no DDP wrapper or distributed override; current
shared factory has the same construction. Initial historical lookups at today's
`shared/` paths failed because that layout did not exist then; the actual old
`elsc/` paths were subsequently located and inspected. No model was instantiated.

Therefore this particular gather/DDP attenuation is not an active mechanism in
the checked local baseline continuation. Its release initialization may inherit
unknown upstream training effects; the source check cannot determine those.
The check does not retroactively certify every historical runtime parameter.

Moreover, gradient scaling is **not equivalent to a W-fold learning-rate change**:
the upstream BertAdam uses first/second moments, epsilon and clipping, and the
trainer also clips global gradient norm. Adam can cancel constant rescaling in
idealized conditions; actual update/trajectory effects were not measured.

No recipe, optimizer, world-size, temperature or retraining sweep is justified.
This refines reproduction fidelity, not Q01's information ceiling or Q14's
directional-conflict finding, and does not reopen the closed optimizer families.
Move to another causal mechanism; do not spend a second turn proving the same
ratio with a different world size or actual GPU transport. No Q38/Proposal8/GO or
global research barrier. Goal active.

## Limits and provenance

Source files `until_module.py` and `modeling.py` hashes are in the JSON; trainer
SHA256 `5f6e3606a170dd5c5a130ddadab67ee1d276f36e619b445f51bf1b6087420418`.
Existing upstream dataloader edits were untouched. No upstream code modifications,
TEST examples/evaluation, SEDS assets, checkpoints or training. The only outputs
are the protocol, isolated fixture, certificate and this bounded audit.
Current DDP documentation corroborates the reduction semantics, not author
runtime/version parity. No independent real-data replication, novelty or venue/
retraction/COI certification. ARS source checks prevent a kernel discrepancy from
being promoted to retrieval harm or a new method.
