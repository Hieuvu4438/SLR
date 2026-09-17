## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (three diagonal replays exact; crossed diagnostic not independently rerun)
- Version Label: AS-C34-v1

## Finding

Original encoder pairing matters in this fixed comparison. Mismatched6 mean
R1=76.107900 versus matched3=77.263969, difference−1.156069pp, conditional
95% CI[−2.190476,−.188324]. The registered pairing-dependence signal passes.
This is not a GO or a new method. Coordinate incompatibility, stored scale
ownership and checkpoint-specific changes remain alternative explanations to
harmful or beneficial semantic co-adaptation.

## All nine fixed cells

PH DEV519, mean bidirectional R1 in percent. Rows: video encoder source;
columns: text encoder source. Scorer/scale belongs to the video source.

| Video / Text |42|1337|2026|
|---|---:|---:|---:|
|42|75.240848|74.759152|70.038536|
|1337|75.144509|74.759152|69.845857|
|2026|74.759152|74.855491|74.951830|

The2026text tower loses strongly when paired with42/1337video towers; the
reverse crosses lose much less. This asymmetry does not prove the text tower
is harmful: source pairing can misalign otherwise useful coordinate systems.
No cell was selected as a model or used for dev-learned routing.

## Locked aggregates

| Aggregate |T2V R1|V2T R1|Mean R1|Mean delta versus matched3|
|---|---:|---:|---:|---:|
|Matched3|76.878613|77.649326|77.263969|0|
|Mismatched6|75.915222|76.300578|76.107900|−1.156069|
|All9|76.107900|77.071291|76.589595|−.674374|
|Fixed video42, three texts|73.988439|74.759152|74.373796|−2.890173|
|Fixed video1337, three texts|74.373796|74.181118|74.277457|−2.986513|
|Fixed video2026, three texts|75.144509|76.300578|75.722543|−1.541426|
|Fixed text42, three videos|75.144509|76.493256|75.818882|−1.445087|
|Fixed text1337, three videos|75.144509|75.915222|75.529865|−1.734104|
|Fixed text2026, three videos|72.061657|75.722543|73.892100|−3.371869|

All9−matched3 conditional95% CI[−1.612980,+.207469]; no equivalence inference.
Remaining exploratory row/column intervals and allR5/R10/ranks are retained
in the run JSON. They are not multiplicity-adjusted discoveries. Primary
comparison specified beforehand: mismatched6 versus matched3, preserving the
marginal frequency of every encoder and video-owned scale.

Persistent mean rank changes versusR0:
matched3−2.489130T/−2.356322V; mismatched6−2.391304T/−2.183908V;
all9−2.456522T/−2.436782V. Thus reduced aggregate R1 does not mean every
persistent rank worsens; all9 has a slightly better persistent V2T mean rank.

## Integrity / execution

All3diagonal scores bitwise equal their original baseline files. Each diagonal
actually rescored3times, each exact. Six-pass/nine-pass repeated diagonal
averages equal matched3 exactly; matched3 equals AS-C32 saved ensemble exactly.
This checks identity and scoring-call-count effects, not an independent
training replication. Total diagnostic15score calls:9unique+6repeats.
Each full grid uses3video and3text encoders; offdiag6/all9 need6/9score calls,
not free matched3inference. Row/column averages have1+3or3+1encoder counts.

Checkpoint/config/manifest hashes, ordered IDs and common masks checked.
All18saved score matrices independently checked after completion against
shared official R1/R5/R10 kernels, exact; read-only check exit0. Full gallery,
official positives and asymmetric tie rules unchanged; test data never loaded.
No new checkpoint, train fitting, overwrite or data deletion.

Run AS-C34-ENCODER-GRID_run.json completed exit0,15.193513s,
peak allocated GPU2,115,348,992bytes. No failure/retry or timeout.
The previous test-session handle expired before its output was collected;
the experiment was NOT restarted. The focused suite was safely rerun and
59tests passed. Unit test covers marginal aggregate identities and explicit
bootstrap comparison scope. Statistical checks use315inferred filename-prefix
clusters,10000draws,seed20260915; conditional on this fixed gallery, not training,
historical selection or dataset-general uncertainty.

## Fallacy scan:11/11 checked

1. Simpson: all cells/directions and persistent ranks retained; average loss
   not presented as a uniform cell/subgroup penalty.
2. Ecological: filename groups not verified signer/recording populations.
3. Berkson: historical dev-selected checkpoints and persistent slices explicit.
4. Collider: no learned adjustment or routing conditional on error outcomes.
5. Base rate:519gallery, fixed3models and different scoring/encoder costs stated.
6. Regression to mean: matched replay/repeat controls; no best cross selected.
7. Survivorship: all9cells/9aggregates and numerical outcomes retained.
8. Look-elsewhere: only the registered primary contrast supplies the signal;
   other intervals are exploratory and not corrected multiple discoveries.
9. Forking paths: video-owned scale and all aggregates fixed before execution;
   no alignment or scale tuning used to improve these results.
10. Correlation/causation: controlled pairing change measured, but its semantic
    cause not identified; coordinate/scale/source effects remain possible.
11. Reverse causality: retrieval losses do not imply one source encoder learned
    inferior language/sign features; coordinate compatibility needs a control.

## Consequence

AS-C35 preregistered next: TRAIN-only shared orthogonal maps to seed42,
same map applied to both towers, identity and shifted-target controls. This
preserves within-checkpoint geometry mathematically and tests whether the
cross-pair penalty can be reduced without new information or new learning.
Generic alignment is prior art, not a new proposal; identity/rank preservation
must pass before interpreting its crossed scores. No GO or global exhaustion.
ARS shaped the preregistration, full control grid and conservative attribution.
