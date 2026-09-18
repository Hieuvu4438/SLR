# Fixed-recipe replication of numerical training degradation

2026-09-18 preregistration, ARS experiment-agent inline. PHseed42 discovery
pair: endpoint+8.959538pp, selecteddelta0 (bothinitialization). Not SOTA,
newoptimizer, or validatedpapercontribution. No hyperparameter rescue.

Run additionalPHseeds1337,2026 andCSLseeds42,1337,2026;eachpairednative/FP32
moments. Keep samefixedselectedinitialization per dataset, same parameterdtype,
loss/augmentation/optimizer/schedule as numericalcontinuation protocol. PH
source ee45bd...; CSL22c15cf... selectedhistorically. Seed changes training
shuffle/augmentation/RNG only; allinitialweights copiedexactly fromcheckpoint.
Withinpair inputs,LR,firstgradient andRNG trajectories mustmatch. Useexisting
sampler forCSL:onevideo/captiongroup,drop_lastB512,12steps/epoch,20epochs=240,
original200epoch/2400stepsschedule/warmup240. PH13steps/epoch,260updates.
CSLalpha.8 andnative1077video×797text groupedDEV; PHalpha.9/519singletonDEV.
Do not forcePHsingleton relevance onCSL. EveryepochDEV andinitialization,
sameguardedselector andallR1/5/10; fixed20epochendpoint is primarymechanism
contrast, not bestcheckpointcontrast. Everyinitialscorematrix mustmatchhistorical
replay. Sourcev1 preserved; v2 onlyseed/dataset/sampler/recording generalization.

Report all3seeds per dataset (PH42 is alreadyobserveddiscovery, labelit),
individualdirectiondeltas, samplemean±samplestd, allselectedvsinit results.
Replication gate per dataset: meanendpointdelta>=1pp, >=2/3seedmeanpositive,
both directionmean>=0, finitepairedstates andphenotype removed. Separately
reportaccuracygate>=.5ppselecteddelta withdirectionguard; no substitution of
degradedcontrolgain for gainoverinitialization. No aggregateacrossdatasets.
DEVhaspriorselectionexposure: seedreplication is NOT freshqueryconfirmation.
No significance/independentCI claim from thisphase. Ifa datasetgatefails,
reportheterogeneity; noLR/epsilon/horizon/masterweight/batch rescue.

BeforeCSLfullruns: two-updatepairedB512smoke,<=300s/arm, samegroupedsampler
and2400schedule; PHusesalreadyvalidatedsmoke andunchangedmodelpath. Ifpaired
integrityfails, preservefailedattempt anddiagnose beforeanysamepurpose retry.
No runningjobrestart. No TEST/extra labels/teacher/auxiliarybranches.

Compute:10newfullruns individually<=900s; allnewreplication GPUwork cumulatively
<=7000s from8055s remainingdiscovery. Trackactualtime andreserve eachnewjob's
fullboundbeforelaunch; stopadmission ifremainingreplicationallowance<bound.
Expected~3000–4000s fromPH264s/arm andCSLgalleryoverhead, notaguarantee.
6hconfirmation reserveuntouched. FullCSLadmission aftersmoke/memorycheck.

Storagecap explicitlyrevised30→44GiB BEFOREnewfullruns: existing26.229GiB +
fivepairs(finalstates~2.43GiB/pair + selectedmodels<=.70GiB/pair + allscores/
metadata<=.20GiB/pair) =~42.88GiB; allow1GiBoverhead<44. Filesystem142GiBfree,
always15GiBfreereserve. Preserveallprioroutputs. Onlyeachnewrun's ownrolling
selected.pt maybereplaced; finalfullstate/RNG/sampler/scheduler retained.
No permission todeleteoldcheckpoints oruploadcorpus. Actualsizesmustbemonitored.

Afterreplication: CPUpairedartifactaudits; then decide scopedclaim andlocked
independentfinalevaluation/extraablations. Passinghere is insufficienttofinish
theusergoal ordeclarepublishability; standardmixedprecision issue ispriorart.
