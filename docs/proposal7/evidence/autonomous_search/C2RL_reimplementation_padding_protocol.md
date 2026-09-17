# C²RL-derived reproduction kernel: padding dependence

2026-09-16. Locked before fixture execution. ARS source verification, AI-assisted.
Repository: ozgemercanoglu/sltbaselines, commit
f741330d0b1e2e8e9a602d7ce4ff6e2e06a4ac13. This is an independent SLT
reimplementation, not the original C²RL retrieval release. Source comment at
models/models.py:272 attributes its v2 similarity function to shared author code;
that comment does not establish byte-level lineage or published-recipe parity.

## Enabled path inspected

`--model_type c2rl` → gloss_free_model.forward → cross_lingual_similarity_v2
→ loss_itc → train_slt.train_one_epoch's total_loss=translation_loss+loss_itc.
The kernel multiplies padding cosine values by1e-5 before softmax, then masks
outer averages. Dataloader uses batch-dependent pad_sequence and tokenizer
padding=True. Full model/dependency execution is not proposed.

## Fixed synthetic check

Fetch ONLY pinned models/models.py and verify SHA256
81edd2f15e09ac5d388303a142798ac425c04896edffc6ed036d03d1b79e54a2.
Extract only cross_lingual_similarity_v2 via AST. Remove exactly4 zero-argument
`.cuda()` calls to run CPU; no other expression changes. Report this adaptation
and do not claim GPU parity. No repository imports, models/assets/outputs loaded.

B=2, visual length1, embedding dimension3. Video vectors e1,e2. Caption A has
one valid vector with cosine.21 to e1; caption B has two identical valid vectors
with cosine.19 to e1. Each valid vector's remaining component is along e2.
All padding vectors are unit e3, exactly orthogonal to both videos; never zero
vectors. Compare fixed total text lengths2 and32. Valid vectors/masks unchanged,
new slots marked invalid. logit_scale1, source tau.07 and label smoothing.2.

For n valid copies of similarity c and m zero-score padded slots, verify the
independent formula E=n*c*exp(c/tau)/(n*exp(c/tau)+m) against the source I2T
row0 within1e-12. Record score order/margin and the source contrastive loss;
synthetic identity targets are execution inputs, not semantic relevance labels.
Record T2I invariance when only text padding changes. A true inner-exclusion
reference, used ONLY as a mathematical control, must remain invariant.
No claim is made that this is a new masking correction or that it improves SLRet.

Outcome regardless of sign: distinguish dynamic-width semantic scoring dependence
from C38's fixed-shape numeric tie audit. This is related to already-audited
padding semantics(C14), not permission to reopen a masking training campaign.
No real batch-frequency, trained-gradient or retrieval-harm inference from toys.
Runtime120s CPU1thread. No DEV/TEST content, repository outputs, new annotation,
checkpoints, SEDS assets or upstream changes. Save source hashes/result and limits.
