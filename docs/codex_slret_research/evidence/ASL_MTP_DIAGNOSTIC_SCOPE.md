# Existing-annotation lead: ASL-MTP

Checked 2026-09-22. Decision: **external diagnostic lead only; no pilot admitted**.
The user excludes new human annotation. No reviewer, generated semantic labels,
download, training, or change to PHOENIX relevance was introduced.

## Source and claim scope

- Reference slug: `karabuklu2026aslmtp`.
- Primary source: Karabüklü et al., *Targeted Linguistic Analysis of Sign Language
  Models with Minimal Translation Pairs*,
  [arXiv v1](https://arxiv.org/html/2604.27232v1), 29 April 2026.
- Reading scope: abstract/introduction, §2.1, §3–3.2, §4.2, selected §4.3
  discussion, §4.5 and Table4, conclusion. Not a full-paper/code audit;
  no human-read attestation.
- [Author publication record](https://kanishka.website/publication/sign-language-mtp/)
  identifies GenSign@CVPR2026. Do not describe this as a main-track CVPR paper.
- Claim manifest: dataset/task → §3; construction → §3.1; scoring → §3.2;
  cue-ablation precedent → §4.2/4.5. No PHOENIX performance claim is supported.

## WHY / HOW / WHAT — verified source facts

WHY: evaluate specific linguistic distinctions rather than only aggregate
translation quality. HOW: ASL-MTP supplies 1,275 ASL video/English minimal-pair
examples across nine phenomena, derived from ASLLRP. Its matched/mismatched
translations are compared using video-conditioned per-token surprisal.
WHAT: the SHuBERT+ByT5 case study examines cue masking at inference and during
training. These are translation diagnostics, not PHOENIX retrieval judgments.
[Source, §3–4.5](https://arxiv.org/html/2604.27232v1).
<!-- ref:karabuklu2026aslmtp --><!-- anchor:section:3 --><!-- anchor:section:4.5 -->

## Availability gate

No direct downloadable benchmark release was located in the inspected paper,
author publication pages, or bounded exact-name searches combining ASL-MTP with
GitHub, Hugging Face and dataset/download terms. This means **not located**,
not proven unavailable. A stale GitHub profile with loading errors is not
negative release evidence. Stop this availability search here; do not repeatedly
poll or contact authors without a new reason and appropriate authority.

## Local decision — inference, not an author claim

This cannot adjudicate the current DGS confusers: the language, examples,
candidate construction and task differ. Published contrast labels must not be
transferred to PHOENIX or recreated there by text perturbation. Nor does the
case study establish a missing cue in CiCo/SEDS or justify adding a face branch.
Generic cue masking and matched training/inference ablations now have an
additional explicit prior-art pointer; they are not a new proposed method.

A future use would require a verifiable release with original pair/video IDs,
data-use terms, model/language compatibility and contamination checks. Retain
the author's labels unchanged; report a separate frozen diagnostic, not a new
retrieval benchmark or relabeled PHOENIX score. These conditions do not authorize
execution or make this resource a global research dependency.

No six-route admission status, question priority, baseline result, or SOTA claim
changes. New evidence is the bounded source verification; local efficacy remains
unknown. See [Q17 contract](../07_HYPOTHESES_AND_FALSIFICATION.md) and
[candidate screen](../09_CANDIDATE_SCREEN.md).
