# CMCM source audit: causal scope and covariance derivative

## Material Passport

- Origin Skill: academic-research-suite, source fact-check and inline numerical validation
- Date: 2026-09-15
- Verification Status: ANALYZED; numerical control verified, efficacy untested
- Disclosure: AI-assisted source analysis; no independent human review

## Outcome

A reproducible numerical discrepancy exists in the pinned covariance module:
the forward normalizes by trace+epsilon, while the backward normalization omits
epsilon. Correcting ONLY those two backward denominators in an isolated in-memory
control restores agreement with independent autodiff to1.12e−16. No upstream
file was modified. This is a baseline-fidelity finding, not a novel method or
evidence that CMCM's published retrieval results are wrong.

## Source and causal scope

Pinned [official CMCM source](https://github.com/vddong-zjut/CMCM) revision
5d458719d1da2f082e188cc44705003d919e7e97. Complete CSA_Module.py, CCG_Module.py,
Encoder.py and MPNCOV.py read; TMCP module inspected previously and covariance
call path refreshed here. Working tree has only existing untracked module caches.

The [publisher record](https://www.sciencedirect.com/science/article/pii/S1077314225003546)
confirms the February2026 paper, DOI10.1016/j.cviu.2025.104631, and claims
augmentation backdoor adjustment, Gaussian cross-modal front-door alignment,
and covariance pooling. Access remains a preview; full identification assumptions
and numeric tables were NOT verified. Authors declare no competing interests
and list public funding. Bibliographic identity is verified; causal efficacy
is unverified, not rejected merely because full text is inaccessible. No new
venue-quality or author-misconduct allegation is made.

CSA computes s=a_aug−g(a_original,a_aug)*(a_aug−a_original). Consequently
s−a_original=(1−g)*(a_aug−a_original). Its final linear output is unconstrained:
the architecture admits identity recovery (g=1), unchanged augmentation (g=0),
and amplified displacement (e.g.g=−1). These are representable parameter settings,
not observations of trained weights. Code alone supplies no invariance guarantee
or causal-identification certificate. A generic bounded gate or nuisance-removal
method remains closed by the project registry.

CCG's inspected module uses a lower-triangular attention mask and diagonal
Gaussian KL. Temporal causal masking does not itself verify a front-door
identification claim. Full training integration is absent from the inspected
tracked tree, so do not equate this module with the complete published method.
Previously documented DEVICE, mask-slicing, missing-import and encoder-shape
issues are not new findings from this turn.

## Covariance experiment

The [locked protocol](CMCM_covariance_protocol.md) uses source-extracted
Covpool→Sqrtm(iterations3)→Triuvec, with symmetric covariance formed from real
feature-shaped FP64 inputs[1,3,2,2]. It does not perturb arbitrary nonsymmetric
covariance entries. One fixed random base was scaled to six preregistered traces.

| Covariance trace | Relative gradient error ||g_source−g_reference||/||g_reference|| |
|---|---|
| 1e−7 | 99.989416 |
| 1e−6 | 9.998611 |
| 1e−5 | .999841 |
| 1e−4 | .100001 |
| 1e−2 | .001000051 |
| 1 | .000010001 |

These are unitless gradient-error ratios, NOT retrieval changes. Independent
forward outputs agree within1.12e−16. Independent exact-forward autodiff agrees
with composed central finite differences within3.31e−11 relative; changing the
FD step gives at most4.45e−11 relative difference. Five of six scales exceed the
registered1e−3 discrepancy criterion. Zero-feature forward is finite but backward
is non-finite. The zero input is a degenerate stress case, not a population estimate.

[Original run](CMCM-COVARIANCE_run.json): first executionexit0,.123270s.
[Posthoc denominator control](CMCM-COVARIANCE_validation.json): exit0,.070721s.
The latter changes only backward trace denominators to trace+1e−5 (and its square);
it does not alter the forward, epsilon value, iterations or sample bank. Every
saved independent reference gradient exactly recomputes. This localizes the
measured mismatch, but does NOT certify a production patch at zero variance.
Covariance code carries original MPN-COV/iSQRT-COV attribution; the finding is
about the vendored CMCM version, not every version from the original authors.

- Original runSHA:2379ca41d5bb82dca95db36a3e741f4f7120ff8d6824887495b2c04803ba4b80
- ValidationSHA:d4fdcea01340f0bd2a78cd204c5a5712dc37550d311f45fe46e49dc71f697226
- SourceSHA:4ee791c579f4925bc07362f2cf2fd88c00c104616259ba76febe2e0c9bdbd3ec
- Regression suite:92passed1.79s, exit0. No dataset/checkpoint/TEST/training/assets.

## Limits and next decision

No trained CMCM activation trace distribution is available. TMCP applies learned
convolutions, batch normalization and attention before covariance formation;
raw CiCo feature traces cannot stand in for that missing distribution. At trace1
the measured error is only~1e−5. Thus large low-trace discrepancies cannot support
a claim of appreciable deployed harm without exposure evidence and a runnable
matched control. No long training campaign or new covariance/causal method follows.

Statistical fallacy scan11/11: results separated by trace (Simpson); no individual
retrieval inference (ecological); fixed stress bank and missing trained exposure
disclosed (Berkson/base-rate); no conditioned outcome variable (collider); no
pre/post benefit or reverse-causal claim; every scale reported (survivorship,
look-elsewhere); locked bank and posthoc control labeled (forking paths); causal
attribution restricted to computational discrepancy, not model performance.

Decision: numerical repair prerequisite established; novel-method GO unproven.
Do not build a CMCM replacement merely to create a runnable baseline, nor promote
ordinary autodiff/stabilization as a research contribution. Continue the broader
source-led search in an open mechanism with a feasible deployment test.
