# Matched SEDS optimizer-moment correction — preregistration

2026-09-18. Engineering/numerical baseline control,NOT method candidate. Native
control001 selected initialization after222updates. Its step111 checkpoint has
nonzero first moments with exactly-zero second moments at97.5132% of FP16
entries,versus0% FP32. Source BertAdam uses parameter dtype for both moments.
Hypothesis: second-moment rounding materially changes updates. Recall damage
is UNKNOWN;even improved stability need not improve retrieval.

Intervention only: initialize native BertAdam next_m/next_v inFP32 for FP16
parameters at the first gradient-bearing update. Retain parameter/gradient
dtypes,nativeforward,all losses,clipping,weightdecay,learning rates and schedule,
seed42,B32,batchorder,augmentation and222updates fromcontrol001. No master
weights,noAMP/no loss scaling,no augmentation-off/no precision sweep. FP32
parameter states remain native. No conversion of old optimizer history.

Gate: unit comparison against FP32 arithmetic with native parameter rounding,
then two-update real smoke (<=300s),step0 score/metric parity with original
DEV002. Smoke stores metrics,not resumable checkpoint. Fullrun(<=1200s) starts
independently at release,not smoke. Evaluate0/111/222;same initialization-
eligible selector and0.5pp directional guardrail asnativecontrol. Report all
checkpoints/streams including losses,not onlybest. No hyperparameter rescue.
If corrected continuation still selectsstep0,claim no training gain;do not
convert numerical evidence into an efficacy or universal impossibility claim.
If it improves,attribute to numerical correction,not novel retrieval mechanism.

Retention: step111 model/RNG/config only (not resumable);step222 full optimizer/
RNG/model. Both evaluable;only final state supports potential later resume.
No actual resume implementation or new training authorized by checkpoint alone.
Smoke no checkpoint;all logs/scores retained. Estimate storage beforeGPUlaunch;
keep15GiB free and15GiB campaigncap. Prior native/full checkpoints not deleted.
Charge4h matched-control reserve,0 method configurations. Collision scope:
not Q21 augmentation rescue,Q28 collapse precision,Q29 inference-tie tuning;
not CMCM Gaussian-derivative operator repair. No closed mechanism reopened.
