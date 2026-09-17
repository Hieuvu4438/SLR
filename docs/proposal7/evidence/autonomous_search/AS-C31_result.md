## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (exact baseline replay; diagnostic not independently rerun)
- Version Label: AS-C31-v1

## Result

The registered broad shared-prefix burden is NOT supported:1/92 persistent
T2V queries and0/87 persistent V2T queries satisfy it, versus the required20%
in each direction. Causal text attention is real, but this screen does not
establish a widespread retrieval failure from unavailable right context.
No attention-mask change, new text encoder, score correction or method proposed.

## Measured representation inventory

R0 seed42, original32-token inputs, original7096TRAIN/519DEV manifests.
All7615 reencoded text token tensors and masks match their frozen caches
bitwise. Actual text mask exactly equals the32x32 causal triangular mask.
Shared prefixes require at least two DIFFERENT complete deployed inputs;
content excludes SOT/EOT/padding, and each distinct prefix length is counted.

| Split | Rows | Content slots | Shared-prefix rows | Shared-prefix slots | Shared-prefix groups |
|---|---:|---:|---:|---:|---:|
|TRAIN|7096|111054|6991|28277|3739|
|DEV|519|7742|460|1345|251|

Many captions share an early prefix, but that is not equivalent to their actual
retrieval confusers sharing a long prefix. The following uses the original
hardest non-diagonal full-gallery confuser in each direction; no error-driven
model or caption changes.

| Persistent-query property | T2V | V2T |
|---|---:|---:|
|Fixed3-seed persistent population|92|87|
|Confuser has identical deployed text|3|4|
|Different text and common content prefix>=3BPE|6|3|
|Registered prefix-burden signature|1|0|
|Mean entire distinct-text gallery fraction with prefix>=3|.003253|.002419|

The whole-gallery fractions are descriptive references, not lexical/length
matched estimates or significance tests. In T2V the comparison uses the paired
caption of the wrong video solely as post-hoc metadata; the actual T2V query
remains unchanged. Identical-input confusers are retained in the report but
excluded from the burden predicate.

## Score conservation and the one qualifying case

Unchanged mixed score decomposed into per-text-token credits for BOTH channels,
including inner padding and video CLS. Full-gallery channel replay error
1.525879e-5<=2e-5; all ranks exact. Maximum paired channel credit-sum error
1.144409e-5<=5e-5. All grouped margin sums within1e-4 of original margins.

Only T2V query207/confuser142 qualifies, with8 common content BPE tokens:

| Margin component | Positive−confuser |
|---|---:|
|SOT|−.026369|
|Common content prefix|−1.131636|
|Remaining content|+.072758|
|EOT|+1.108448|
|Padding|−.463053|
|Total|−.439850|

This is additive accounting, not causal importance or a deployed correction.
The A softmax denominator depends on all text tokens; B has length-dependent
normalization. Removing/reencoding one component can change other components.
The favorable EOT contribution also shows why early-token context restrictions
cannot be equated with absence of sentence context from the whole scorer.
No semantic equivalence or signed-video minimal-contrast annotation inferred.

## Numerical anomaly and disclosed follow-up

Parent run completed successfully but found nonidentical shared-prefix vectors
in152TRAIN/19DEV groups, maximum absolute difference.00390625. The independent
all-caption cache replay remained exact. Rather than assume mathematical
invariance implied bitwise identity, registered a POST-HOC numerical follow-up
in `AS-C31_numeric_followup.md`; parent outputs and thresholds unchanged.

Representative-to-member comparisons (not all pair combinations):

| Comparison | TRAIN equal / different | DEV equal / different |
|---|---:|---:|
|Original full128 batches|24325 /0|1075 /0|
|Original final tail within itself|1 /0|0 /0|
|Across full128 and tail|0 /212|0 /19|

Tail batch sizes56TRAIN and7DEV. Every measured non-equality crosses batch
sizes. First16 sorted nonexact/different-input prefix cases per split were
tested with unchanged128-row shape, changing only row0's sentence suffix while
preserving its prefix IDs. All32 prefix differences were EXACTLY ZERO.
These cases cover11TRAIN/6DEV unique sentence pairs, not32 independent examples.
Their EOT vectors changed (max-coordinate differences1.570801–4.048828TRAIN,
2.328613–2.921875DEV). The same input at full versus tail batch shape produces
prefix differences.000976563–.00390625 in both splits, consistent with the
cache discrepancy. This documents shape-sensitive numerical computation, not
information from masked future tokens leaking into prefixes.

## Execution and provenance

- `AS-C31-PREFIX_run.json`: exit0,5.469667s, GPU1,329,149,440bytes.
- `AS-C31-PREFIX-NUMERIC_run.json`: exit0,4.437809s, GPU456,776,192bytes.
- Both protocols/follow-up specifications preceded their respective execution.
- No crash, retry, timeout, raw-data change or external upload. Test split never loaded.
-54focused tests pass; compileall and git diff --check pass.
- Code: text_prefix_audit.py, text_prefix_numerics.py, test_text_prefix_audit.py.
  Run records retain code/model/cache/manifest hashes and per-query details.
- No trained model or new score matrix promoted; no independent training seed.

## Fallacy scan:11/11 checked

1. Simpson: directions and whole populations separated; no universal subgroup claim.
2. Ecological: prefix sharing is token equality, not signer/semantic equivalence.
3. Berkson: persistent-error selection explicit; all-query counts and gallery
   references retained, no population-general causal estimate.
4. Collider: no fitted adjustment or routing based on selected error status.
5. Base rate: denominators92/87 and exact-input exclusions shown; threshold unchanged.
6. Regression to mean: no before/after learned improvement claimed; baseline exact.
7. Survivorship: all directions, nonexact vector groups and numerical follow-up reported.
8. Look-elsewhere: one registered signature; post-hoc numeric follow-up labeled,
   no p-value or favorable-case efficacy claim.
9. Forking paths: no mask/precision/score/threshold tuning after negative result.
10. Correlation/causation: additive margins are not causal importance; fixed-shape
    suffix intervention supports prefix invariance only for those tested inputs.
11. Reverse causality: causal mask not inferred from errors; directly verified.
    Numerical differences not relabeled as semantic future-context effects.

## Consequence and next research action

Reject the specified BROAD shared-prefix burden; not every possible right-context
limitation. Do not launch a bidirectional-text model from this negative signature
or reopen closed nuisance/candidate-prior corrections. No GO and no exhaustion.

Next high-information check: deployable cross-checkpoint complementarity on
the existing three strong baseline seeds. Register uniform score averaging and
same-seed repeat controls before measuring; fixed full519 gallery, no learned
weights or checkpoint selection. Track the persistent intersection explicitly:
all three individual rankings can fail while different confusers cancel under
averaging. This is a stronger-control/headroom diagnostic, not a novel ensemble
method or cohort-dependent inference. Any gain must be labeled with3x model
compute and cannot by itself satisfy novelty or method GO. No AS-C32 launched.

ARS influenced preregistration, conservative failure interpretation and explicit
numerical follow-up; no fabricated independent reviewers or causal conclusions.
