# AS-C17 — low-diversity nearby-view control fails R1 screen

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (canonical replay and view constraints verified)
- Version Label: AS-C17-result-v1

Registered protocol `AS-C17_protocol.md`; completed exit0 in6.99s,
peak allocated GPU690465280bytes. Run `AS-C17-VIEWS_run.json` contains all
view/metric outputs and provenance. `AS-C17-SUMMARY.json` records the subsequent
effective-diversity audit and11/11 fallacy checks. No crash/retry or threshold
change. No new I3D/backbone, raw feature cache, training or test access.

Three canonical encoder passes match the original cache exactly; all score
channel deltas1.5258789e-5≤2e-5 and ranks match. Three canonical score matrices
are mutually identical and their float64 mean is exactly that matrix. Each
registered nearby view preserves count/mask, strictly increasing unique dense
indices and displacement≤1 per canonical slot.

|View|T2V R1|V2T R1|Mean R1|Delta pp|
|---|---:|---:|---:|---:|
|Canonical (each repeat and mean3)|74.181118|76.300578|75.240848|0|
|Jitter42|73.988439|75.722543|74.855491|−.385356|
|Jitter1337|73.603083|75.915222|74.759152|−.481696|
|Jitter2026|73.603083|76.107900|74.855491|−.385356|
|Jitter mean3|73.795761|75.722543|74.759152|−.481696|

Mean3 misses the fixed exploratory+.5pp screen and loses.385356pp T2V/
.578035pp V2T. Persistent mean ranks nevertheless improve.271739T/.149425V;
T2V R5/R10 improve.385356/.192678pp. Do not claim every metric worsens.
All complete R5/R10/rank details remain in metric JSONs; no seed is selected.

## Probe adequacy qualification

379 videos change from canonical in every view;140 short videos remain exactly
unchanged. Mean changed slots per video is3.371869/3.371869/3.391137; maximum
48/47/52. Mean contextual valid-token cosine distance is.005081/.004909/.005133.

The post-hoc exact-index audit finds only31 videos with three distinct views,
13 with two and475 with one. Of the379 changed videos,335 get the SAME alternate
view for all three seeds. The shared sampler attempts random legal indices then
has a deterministic fallback. Its `view_independent` flag means different from
canonical, not independent across seeds. The three score matrices have exactly
475 identical rows, matching the index inventory. This is a **low-diversity**
three-pass average, not evidence from three broadly distinct temporal samples.
No modified sampler was introduced to rescue the result.

All140 unchanged-video V2T score rows remain exactly identical, and none has a
V2T rank change. Eight corresponding T2V query ranks change because other gallery
videos changed. Full-gallery interference prevents treating unchanged-own-video
T2V queries as an isolated untreated control. Mean3 changes34T/24V ranks; it
corrects2T/2V top1 errors and introduces4T/5V errors (per-query rank convention,
not a substitute for official primary metrics).

Decision: the fixed shared-sampler mean fails, but limited diversity prevents
a broad conclusion that temporal sampling/aliasing cannot matter. No novel
method or three-trained-seed pilot. No sampling-consistency/partial-alignment
family reopened. Broader acquisition remains unresolved, not globally exhausted.

Focused suite30passed in1.11s; compileall and git diff --check pass. No full
independent rerun claimed. Research remains active; no Proposal8.
