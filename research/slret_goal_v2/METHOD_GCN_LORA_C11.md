# C11 — clean GCN + visual LoRA composition

## Result / decision

Completed222updates: selected222mean77.938343, -.096339pp vs matched clean GCN
78.034682. At111mean77.842004 (+.096339 vs control111); finalV2T deficit outweighs
T2V R5/10 improvement under fixed primary metric. All plumbing checks passed.
Defer this recipe; preserve best/logs. No synergy claim. Charge629.386665s incl
smoke. Next: clean GCN training-seed replication, not further C11 tuning.

## Material Passport / registered card (2026-09-19)

- ARS experiment-agent/run, inline; exploratory engineering adaptation, novelty unresolved.
- Evidence: clean GCN control78.034682 beats release77.552987; C04 LoRA77.938343 independently has a small positive lead. C08 masking hurts its matched control.
- Hypothesis: low-level joint encoder and upper visual attention adaptation can complement one another; synergy is NOT established.
- Release initialization, NOT weights merged from independently trained models; GCN1e-6 + fusion1e-5 + LoRA1e-4, rank8/alpha8, last visual attention block in pose and RGB streams.
- Native pose/RGB paths retained, GCN recomputed online. Text/sign-conv/other native weights frozen; GCN BN/dropout fixed eval, no masking or reconstruction decoder.
- Full TRAIN7096, B32, seed42, one epoch222updates, same batch ordering/native loss/FP32 optimizer as clean GCN control002.
- Native DEV519 initial parity and111/222; primary mean bidirectionalR1, report both R1/5/10; no TEST loaded/used. Selector includes initialization; reference-.5pp per-direction guardrail, earlystopmean-drop>2pp.
- Matched reference for incremental LoRA: seds-masked-control-002 (same222 horizon). Current incumbent78.034682. C04best77.938343 is historical comparison, NOT a horizon-matched synergy control.
- Positive result requires subsequent LoRA-only horizon-matched control and training-seed replication before synergy/generalization claims. Repeated DEV selection disclosed.
- Scope differs from C04 depth/batch/text refinements: enables native spatial GCN weights using the newly positive clean-GCN result; no reopening user-closed C06, protected gradient, teacher, confidence gate or gallery reranker.
- Risks: simultaneous adaptation can drift; gains might be seed noise or ordinary extra trainable capacity. No promise of novelty/SOTA from this combination.
- Two focused CPU checks pass: original PEFT partition and composed groups/updates/frozen BN/dropout; no pytest installation needed (called test functions directly). GPU2step smoke checks actual GCN/LoRA updates and frozen buffers before pilot.
- Entry `tools/run_gcn_lora.py`; chain c11-gcn-lora-pilot-001; smoke seds-gcn-lora-smoke-001, pilot seds-gcn-lora-001. Source snapshots preserved per run.
- Hard limits240/960s,parent1260s; expected650–850s, <=4GiB new files, <=48GiB VRAM; disk73GiBfree. Remainingtotal3364.220964s/currentpool1289.604569s before run; charge children once.
- Preserve release, C04, clean GCN best and corrected H4W cache/compact weights. Background/log/yield; no training polling loop. On return promote/refine/defer from selected AND fixed-step control contrasts.

Execution: timeoutPID2429107; GPU smoke completed2steps27.374265s, fullDEV
zero-init parity and actual GCN/LoRA updates/frozen buffers passed. Main pilot
started and process confirmed live. Charge smoke with pilot once on collection.
