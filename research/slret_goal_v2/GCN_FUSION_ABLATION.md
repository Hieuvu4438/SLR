# GCN adaptation contribution: matched fusion-only control

## Completed decision

All3 controls completed222 and passed data/order/LR/frozen-state checks.
Selected GCN-minus-fusion gains42/1337/2026 = +.481696/+.867052/+.578035pp;
paired mean+.642261, sampleSD.200546. Fixed222 gains+.481696/+.867052/+.770713.
Supports GCN adaptation contribution under matched recipe, not novelty/SOTA or
independent dataset confirmation. Retain stronger clean GCN models and controls.
Charge678.949186s; remaining897.068655s. Next C12 native clip-conv adaptation.
Full directional metrics and selector details in RESULTS.md.

## Material Passport / registered card — 2026-09-20

- ARS experiment-agent/run, inline, component attribution for the measured clean GCN lead; no new method/novelty claim.
- Evidence: clean GCN+fusion three seeds selected/finalmean78.227360, +.674374pp release. Prior fusion-only runs differ in schedule (3epochs vs1); they cannot isolate GCN updates here.
- Intervention: freeze GCN parameters, retain original eval BN/dropout; train only fusion1e-5. No change to inference, data, loss, augmentation, batch size or native schedule.
- Keep --masked-pose control plumbing and inactive decoder; --clean-fusion-control excludes GCN from optimizer. Decoder remains frozen/bypassed. Same native pretrained release.
- Seeds42/1337/2026, B32/fullTRAIN7096,1epoch222updates, native auxiliaryweight1, FP32 moments. Passed sign_lr1e-6 stays same but no GCN group is trainable.
- DEV519 initial inherited +111/222; same selector includinginit, reference-.5pp directional guardrail and mean-drop>2pp earlystop. No TEST used.
- Pair each with its exact clean GCN seed, same batch hash and data/base digests; report paired selected/fixed111/fixed222 deltas and all3 seeds. No best-seed cherry-picking.
- Six focused CPU tests pass, including onlyfusionupdated/frozenGCN; normal step2 real-data gates remain. No redundant corpus or baseline score replay.
- Driver tools/run_masked_pair.py --fusion-ablation, chain gcn-fusion-ablation-001, runs seds-gcn-fusion-ablation-seed{42,1337,2026}-001.
- Hard450s per child,parent1410s; expected650–800s, GPU<15GiB. Totalremaining1576.017841s; consolidate existing subpools, no total increase. Charge children once on return.
- Disk67GiBfree,V2used26.161GiB,expected<=6GiB new best/last files fits36GiB cap. Keep new best1337/allGCNseeds/C04/3Dcache, no deletion needed.
- Decision: if paired gains persist, attribute incremental benefit to GCN weight adaptation under this recipe only. If not, retain strongest measured retriever and revise explanation. No significance/SOTA/independent confirmation claim from selected DEV.
- Launch/log/initial-update-check/yield; no automatic further experiments or training polling loop.

Launched timeoutPID2476853, first child2476867 confirmed running6steps;
onlyfusionupdated, frozen-buffer and LR checks passed. Results pending.
