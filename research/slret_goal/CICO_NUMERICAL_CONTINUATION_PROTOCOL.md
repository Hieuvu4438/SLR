# Matched CiCo continuation: native versus FP32 AdamW moments

2026-09-18 preregistration, ARS experiment-agent inline. Numerical baseline
correction/empirical-mechanism pilot, NOT a noveloptimizer method. Registry
collision: no augmentationchange, residualadapter, lexicalteacher, UPRetOT,
gradientprotection or newloss. Those remainclosed. Samegradientarithmetic
exposure passed on originalB512 PH/CSL; this alone is not a recallresult.

First PHseed42 only. Initialize botharms fromsamehistoricallyselected strongest
localCiCo checkpoint ee45bd2d... (fullhash fromselection.json), freshoptimizers.
Retain nativeparameterdtypes,amp_bf16,balancedCLCL,dualmix.5,alpha.9,maxwords32,
featurelen64,wordaugmentation ON,shuffle/drop_last, batch512/511negatives,
lr1e-5,betas.9/.98,epsilon1e-6,weightdecay.001 exceptbias/1D0,clipnorm1.
Botharms usePyTorchAdamW single-tensorkernel(foreach=False,fused=False), whose
FP16weightoutput matcheddefaultforeach in realB512check. Controlstatemoments
inheritparameterdtype; treatmentmoments/gradientarithmeticFP32,parameterstorage
unchanged. No masterweights; residualparameter-rounding remains a limitation.

Keep original200epoch cosine/warmup.1 horizon:13steps/epoch,2600total,260warmup.
Boundedpilot endsafter20epochs/260updates,133120exampleoccurrences, no horizon
compression. Sampling order and clean/aug/visualbatchhashes paired. Capture
firstbatchgradienthash/loss/norm beforedifferentupdates; requireexactmatch.
LatertrainingRNGstates must match atcheckpoints; outcomesneednotmatch.

Sharednative-compatible full519 DEV evaluation atinitialization andeveryepoch,
noTEST. Saveallscores/ranks and R1/5/10/MedR/MeanR; preserve trainingRNG around
evaluation. Step0scores bitexact toalreadyreplayed historicalselectedcheckpoint.
Selector maxbidirectionalmeanR1 subjecttoeachdirection>=initialR1-.5pp, earlier
checkpointonmean ties. Initializationeligible. Retain selectedmodel, finalfull
optimizer/scheduler/RNG/sampler, andepochmetrics; no bestTESTselection.

Accuracylead gate: correctedselectedmean>=strongesteligiblecontrol+.5pp and
eachdirection>=control-.5pp, with R5/10declinesreported. Separately, a possible
empiricalmechanism-study gate is correctedvsnativeFIXEDendpointmean delta>=1pp,
bothdirectionsnonnegative, plus statephenotype removed and pairedintegrity.
This latter gate admits replication planning only, NOT newaccuracy/SOTA claim;
report whether eitherarm exceedsinitialization. Multiseed/second-dataset and
independentconfirmation requirements remainunmet by thispilot. No LR/epsilon/
schedule/masterweight sweep ifbothgatesfail. No augmentation rescue.

Before fullpilot: two-updateB512smoke forbotharms withsame200epochschedule,
<=300s/arm, noDEV orcheckpoints, firstgradientparity andmechanismactivation.
Measure time/VRAM and pairdata/RNG. Reportfinite moments and changedweights.
Fullpilot admission requires <=1800s/arm projectionwithmargin andcapacity.

Storage: prior23.742GiB/24GiB. Smoke<=32MiBtotal fits existingcap. Beforefull
pair explicitcampaigncaprevision to30GiB (filesystem144GiB free,15GiBreserve):
two finalfullcheckpoints~2.7GiB + twoselectedmodels~.7GiB + allscores/ranks/logs
<=.25GiB + overhead.5GiB => projected<28GiB. Onlythisrun's rollingselected.pt
may be atomically replaced whenselectionimproves; preserveALLpriorcampaign
artifacts. Finalfullcheckpoints retainresume state; selectedmodels inferenceonly.

Discoveryremaining8597s; smoke<=600s thenfullpair<=3600s hardbounds leaves
>=4397s,6hconfirmation reserveuntouched. No secondseed/dataset longrun admitted
beforepilotdecision. Recordstartup failures; never restartalivejobs.
