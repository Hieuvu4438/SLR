# Clean GCN + fusion: locked-recipe training-seed replication

## Completed results (2026-09-20)

All3 seeds42/1337/2026 above release:78.034682/78.516378/78.131021 selected
meanR1; fixed222 means identical, directions differ. Across-seed mean78.227360,
sampleSD.254889, mean gain+.674374pp. No early stop or missing seed.
Seed1337step111 new provisional best. This shows a repeated DEV lead, not
independent confirmation, significance or novelty. Keep all artifacts.
Charge new child times1158.816458s; totalremaining1576.017841s. Next matched
fusion-only1epoch ablation needed because historical controls used3epoch schedules.
Full direction/step metrics in RESULTS.md; no TEST consulted.

## Material Passport / registration

- ARS experiment-agent/run, inline,2026-09-19; existing positive engineering adaptation, not a new novelty claim.
- Seed42 seds-masked-control-002: selected/final222mean78.034682, +.481696pp release. C11 composition did not improve it. Test whether clean adaptation gain repeats.
- New training seeds1337 and2026, plus existing42 =3 seeds. Change seed only; release base, fullTRAIN7096, B32,1epoch222, native augmentation/loss, FP32 optimizer.
- GCN weights1e-6/fusion1e-5, GCN BN/dropout eval; other encoders/text/sign-conv frozen. No reconstruction, masking, LoRA, extra features or scorer.
- Keep inactive decoder and same wrapper as original control for exact recipe; forward unchanged. Per-run step2 expected-update/frozen-buffer checks retained, no redundant baseline audit.
- DEV519 at111/222, inherited step0, unchanged selector/guardrail and mean-drop>2pp earlystop. No TEST loading or analysis.
- Report all3 seeds selected and fixed222 mean/spread, both-direction R1/5/10. If stopped early disclose missing final222; never replace failed/missing seed with a favorable seed.
- Seeds chosen from existing C04 set before observing new runs. Historical C04/fusion3epoch runs differ in schedule/horizon: descriptive comparison only, not fully matched attribution.
- Seed42 chose the recipe; repeated DEV selection and inclusion of discovery seed prevent independent-confirmation claims. No hypothesis test/significance claim planned at this stage.
- Driver `tools/run_masked_pair.py --replicate-control`; chain gcn-clean-replication-001, runs seds-gcn-clean-seed1337-001 and seds-gcn-clean-seed2026-001.
- Expected~1200s/~15GiB VRAM, hard900s per child,parent1860s. Internal allocation+1200s creates pool1860.217904; totalremaining2734.834299 unchanged. Charge children once on return.
- Disk71GiB free,V2used22.357GiB,planned<=8GiB fits36GiB cap/15GiB reserve. Preserve all relevant checkpoints, cache and data; no deletion.
- If gain repeats, retain stronger baseline and pursue targeted improvement/control; if not, report fragility and reprioritize. No threshold/recipe change after new seeds.
- Launch background, confirm alive/initial updates once, yield; no automatic further experiments or quota-polling loop.

Launched timeoutPID2447052; first child2447061 seed1337 confirmed live at3steps,
step2 update/frozen-buffer checks and exact LR groups passed. Await user return.
