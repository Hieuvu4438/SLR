# Donor integration status — Goal V4

Latest2026-09-21: UniFormerV2-L/14@336 public pretrained extraction is active
under the user's prior authorization. Frozen1024D stride1/32-frame clips are
complete for TEST642 and DEV519; one snapshot found TRAIN3037/7096. Existing
outputs are finite and paired feature/meta/temporal files, but full TRAIN has not
yet validated and no SEDS bridge/pilot has run. Prospective C24 must align its
receptive fields to SEDS'64-token contract and retain native RGB+pose2D. This is
`weights ready / extraction partial`, not `integrated` or `pilot measured`.
Official repo commit is722a43440fc5b9662cc2a8f23b86caa205e45ebc, code license
MIT. The Kinetics710-pretrained/Kinetics400 checkpoint is1,417,825,259bytes;
separate weight-redistribution terms remain unverified. Full interface/control
contract is in METHOD_UNIFORMERV2_COMPLEMENT_C24.md.

Latest C19: attention-bottleneck principle from Nagrani et al., NeurIPS2021;
independent implementation, no copied donor source/weights/dependencies. Paper
method and fusion ablation read; official implementation license not assessed
because no code imported. No new external data/pretraining. Adapter integrated,
CPU checks passed; actual training efficacy pending. C19 is an adaptation, not
claimed first invention. NativeSEDS kept; see METHOD_GLOBAL_EXCHANGE_C19.md.
C18-002 measured78.227360 belowcontrol78.323699; physicalB64 recipe deferred.

2026-09-20. Known components are credited, not claimed as inventions.
Latest: C17 USER-CLOSED after blend final-eval timeout; no DCL recovery/refinement.
Pure fusedDCL measured77.649326 selected below noaux78.323699. C18 uses native
batch64 without donor code; not GradCache. No transfer/DCL novelty claim.
Original readiness entries below retain historical setup scope.

| Donor | Provenance / rights / resources | Integration and evidence |
|---|---|---|
| DCL, Yeh et al., ECCV2022 | Independent paper-equation implementation; no donor source copied, weights or dependencies. [Inspected third-party implementation](https://github.com/raminnakhli/Decoupled-Contrastive-Learning) is NOT established as author-official and not imported. | C17 code ready, CPU equation/gradient/inference tests passed. TRAIN retrieval pilot not yet measured. No new inference parameters. |
| GradCache | [Paper-linked source](https://github.com/luyug/GradCache); no clone/commit/license verification yet, no code adopted. | Alternate batch-scaling lead; not integrated/tested. |
| SignRep | Existing external checkout06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2; CC-BY-NC-SA4.0 research context, checkpoint f8be8ca4...766b; METHOD_SIGNREP_C16.md carries full IDs. | Cache/strict-load/pilots measured; direct pointwise/RKD-D transfer not above matched noaux, deferred after R2. Retain assets; not used in C17 training. |
| RKD-D, Park et al., CVPR2019 | Independent equation port; author repo inspected, no code copied, root license not established. Full provenance in METHOD_SIGNREP_RELATIONAL_C16_R1.md. | C16-R1/R2 measured; below/tie noaux. No novelty or teacher-superiority claim. |

No donor training corpus, private notes or model files uploaded. New sources are
not blanket-authorized if gated/paid. C17 native RGB+pose2D and labels remain intact.
