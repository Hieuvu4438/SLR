# CMCM Gaussian alignment: negative and unbounded matched-distribution loss

## Material Passport

2026-09-16; academic-research-suite / deep-research, inline source verification.
ANALYZED; source-contract counterexample verified, trained exposure unmeasured.
AI-assisted derivation and execution; no independent reviewer or linguistic labels.

## Finding and decision

[V/M] The released GaussianAlignmentModule is not a nonnegative Gaussian KL.
For identical means and identical positive variances, its loss is negative and
has no finite lower bound in real arithmetic as the common variance tends to zero.
Six unchanged-source CPU fixtures reproduce this and its derivative. Thus a lower
value of this auxiliary loss need not mean better cross-modal agreement.

This is a new loss-contract finding, separate from the previously measured
covariance square-root backward discrepancy. It is **not** evidence of actual
trained collapse, invalid published retrieval results, or a novel SOTA method.
The mathematical correction is ordinary implementation fidelity; no new training
campaign, variance regularizer, temperature sweep or uncertainty method follows.

## Source and exact derivation

Local official CMCM checkout: `5d458719d1da2f082e188cc44705003d919e7e97`.
[CCG_Module.py](../../../../third_party/CMCM/modules/CCG_Module.py:59) returns only
the scalar alignment loss. Full released training/inference integration remains
unavailable in the previously checked tree. It cannot establish whether these
heads are auxiliary-only in the actual published system.

Its scalar coordinate expression, with e=1e-8, is

`L = .5 * [log(v_t/(v_v+e)) + (v_v+(mu_v-mu_t)^2)/(v_t+e) - 1]`.

`GaussianParameterization` generates strictly positive variances by Softplus in
exact real arithmetic. Set means equal and v_v=v_t=v>0. Then

`L(v) = .5 * [log(v/(v+e)) + v/(v+e) - 1] < 0`.

Both terms inside the brackets are negative after combining the fraction with
minus one. As v→0+, the logarithm tends to minus infinity while the fraction
tends to zero; therefore L→minus infinity. This is a proof about the mathematical
expression, not a claim of infinite finite-precision values in our fixtures.

Zero Gaussian projection weights, zero mean biases, and common variance bias b
realize v=softplus(b) independently of either input. This admissible parameter
path keeps both distributions exactly matched while reducing the loss. It does
not require changing either encoder or the attention weights. To run a simple
full-module fixture, attention parameters were also zeroed, not patched.

The derivative along that common-bias path is

`dL/db = .5 * [e/(v*(v+e)) + e/(v+e)^2] * sigmoid(b) > 0`.

It approaches .5 as b→minus infinity. At equality, the separate video-variance
partial derivative is zero; the cross-modal-variance derivative is positive.
Thus this is also a nonstationary matched pair, not merely a constant offset.
Ordinary gradient descent does not necessarily follow the common-bias path:
existence of a decreasing parameter path is not a measured optimizer trajectory.

For comparison, exact diagonal Gaussian KL is zero for equal distributions.
Replacing both variances consistently by v_v+e and v_t+e also gives zero at
equality. These references diagnose the source mismatch, not retrieval efficacy.
The [PyTorch 2.11 KL API](https://docs.pytorch.org/docs/2.11/distributions.html#torch.distributions.kl.kl_divergence)
provides the Normal–Normal reference used in the fixture; the lower-bound proof
above is our direct algebra, not an attribution to that documentation.

## Locked experiment and complete results

[Protocol](CMCM_gaussian_protocol.md),
[code](../../../../methods/information_probe/cmcm_gaussian_contract.py),
[machine-readable execution](CMCM-GAUSSIAN-CONTRACT.json).

The three source classes were extracted unchanged through AST and executed with
their original forward/backward graph. CPU FP64, B1,T1024,D8,heads2,K3, zero inputs.
T1024 deliberately respects the previously known mask-slicing defect; it is not
a realistic PH clip-length claim or a patch of the defect. No source import side
effects or pretrained weights. All learnable parameters zeroed before setting
the fixed shared variance biases. Mean reduction retained exactly as released.

| Shared Softplus bias b | Variance | Source loss | Common-bias derivative |
|---|---:|---:|---:|
| 0 | .693147 | −1.442695e−8 | 1.040684e−8 |
| −10 | 4.539890e−5 | −.000220233 | .000220192 |
| −18 | 1.522998e−8 | −.450560497 | .317805671 |
| −25 | 1.388794e−11 | −3.789660109 | .499999038 |
| −40 | 4.248354e−18 | −11.289659628 | .5 |
| −80 | 1.804851e−35 | −31.289659628 | .5 |

Exact and consistently shifted KL references are zero in all six cases. Largest
absolute source/formula error1.4210855e−14; largest autograd/analytic directional
derivative error1.6653346e−16, both below the locked1e−10 tolerance. All losses
and derivatives finite. Original module remains unmodified (`git diff` empty).
Execution exit0 in1.047084272s; PyTorch2.11.0+cu128, CPU execution only.

Source SHA256 `277a06c5bf4a5f256b550fe3c0b88a9e09135cca39052afa8a4976db7b4129f7`.
Probe SHA256 `58e090775c2a8d70843bb11b35f79ee7eca808729d8097709244c6027dc49367`.
The evidence JSON retains every fixture, not only the strongest counterexample.

## Interpretation and next action

1. At variance~.69 the discrepancy is only1.44e−8; the spectacular low-variance
   cases are not population prevalence. No trained variance distribution observed.
2. Auxiliary-head parameter degeneracy can lower this module loss without changing
   its inputs. Whether those parameters also affect other losses/inference, whether
   weight decay constrains them, and whether they ever reach this regime are unknown.
   Do not promote the module's unboundedness to the full training objective.
3. A negative value disproves the exact KL interpretation of this source formula,
   not CMCM's entire causal-identification argument or paper's final implementation.
4. Mathematically consistent Gaussian KL, a variance floor or deleting this loss
   are ordinary correction/ablation candidates, not three materially novel methods.
   There is no matched trained CMCM resource for attributing a downstream gain.

The turn first rechecked CiCo's already completed channel-loss argument and
CMCM's missing deployment integration; neither recheck counted as progress.
SAN source also showed fixed train/evaluation text widths70/44, but no tokenizer
or corpus exposure experiment was performed and no truncation-harm claim made.
The new evidence is exclusively this Gaussian-loss counterexample.

Primary-source browsing: GitHub HTML/raw openings failed; provenance comes from
the local pinned checkout. PyTorch2.11 API page retrieved. SAN paper search was
discovery only; no published results used to select the fixture or claim efficacy.
No data/examples/TEST/DEV, feature arrays, trained checkpoints, SEDS assets, GPU,
optimizer steps or external uploads. No new annotation, positive set or benchmark.

ARS discipline keeps mathematical validity, trained exposure and retrieval effect
separate. Record this as a baseline prerequisite and move away from numerical
repair: no resource-query loop, replacement implementation or epsilon rescue.
No Q38, Proposal8, method GO or global research barrier. Goal remains active.
