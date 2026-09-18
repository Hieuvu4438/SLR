# Existing I3D tail fine-tuning — bounded B_tuned pilot

2026-09-18 preregistration. ARS experiment-agent inline. Not a novel method.
Motivation/admissibility: RGB_GRADIENT_FEASIBILITY.md and RGB_TRAIN_SMOKE_PROTOCOL.md.
Two-step fullB32 activation passed; feature/loss/headgradient parity exact.

One fixed configuration only: start original SEDS/I3D releases, seed42, all7096
TRAINexamples/222updates in the exact moment-control001 order (lastbatch24).
Same native losses, textaugmentation, masks, negatives, headoptimizer with FP32
moments. I3D trainable Mixed5b/5c only, unchanged fullframes/windows/preprocessing,
fixedmicrobatch8, frozen BN/dropout; same tail optimizer as smoke, LR1e-5. No
new objective/data/scorer/stream/backbone. No freeze/temperature/LR rescue.

Versioned two-runtime worker encodes everybatch afresh beforehead forward,
checks replay equality and delays updates untilbackward completes. Initial
firsttwo batches must matchcontrol losses/headnorm and frozenfeatures. Runtime
and CPU/FPU arithmetic difference remains declared, with exact measuredparity
atinitialization. I3D clipping separate fromhead. Training costs higher than
frozenbaseline; exposure/updates matched, compute not equal.

DEV official519 is historically exposed selection, NOTconfirmation. Evaluate
step0/111/222; attrainedsteps regenerate ALL519 RGBfeatures with currentencoder,
keep same adaptedpose/query/gallery/metric and record new hashes. Nativeevaluation
must preservetrainingRNG. Initialization-eligible selector: maximize fusedmeanR1
amongcheckpoints whereeachdirectionR1>=initialR1-0.5pp. Reportall3streams and
R1/5/10; no test. Strongestmatchedcontrol selected111 mean77.6493256262; pilot
lead requires selectedmean>=78.1493256262 and eachR1 no morethan0.5pp below
strongestcontrol's corresponding76.4932562620/78.8053949904. R5/10declines
remain visible. Also compare111/222 fixedendpoints against matchedcontrol.
Success would admit confirmation/ablation planning, not SOTA/novelty. Failure
ends thisconfiguration; no horizon orLR sweep based onDEV.

Local wall bound9000s, charged toremaining12922s discovery, no6hconfirmation
reserve spent. Smoke worststep*222*1.3=6635s; remaining2365s coversDEVrefresh,
evaluation,checkpointing andruntimeoverhead. No automatic restart aftertimeout.

Storage revision BEFORElaunch: campaigncap24GiB (was15), based on153GiB actual
free afterusercleanup. Preserve15GiB free reserve andallpriorartifacts.
Current14.677GiB + fullheadretention~4.03GiB + allbatchfeature/VJP~3.47GiB +
twoDEVfeature/score sets<=0.5GiB + twoI3Dtail/optimizer<=0.12GiB + logs/overhead
<=0.5GiB => conservative23.3GiB<24GiB. No deletes/overwrites. Retain headmodel111,
fullheadoptimizer222, RNG/config/order atboth; tail+optimizer111/222 andmanifest
linkmatchinghead/encoder versions. Inference can reconstruct completeI3D from
hash-locked originalbackbone plustail. Only222 is jointresumable; 111model-only
head remains inference/selectionartifact. Crossprocesscommit interrupted runs
are invalid except a completedpairedcheckpoint. Never resume partialcommit.

Beforelaunch: tests, GPUidle, source/asset hashes, explicitbudget check. Duringrun:
PID/timeout plusrun.json, perstep log andworkerprogress.json. DEVrefreshmaytake
severalminutes and emitsprogress every25videos. Failure andallmetrics retained.
