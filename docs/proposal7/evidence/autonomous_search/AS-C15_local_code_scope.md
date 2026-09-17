# AS-C15 — bounded local scorer inspection

## Material Passport

- Origin Skill: academic-research-suite
- Origin Mode: source inspection supporting experiment interpretation
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (local code only, not a literature novelty search)
- Version Label: scorer-scope-v1

No external source lookup, model download, code edit or experimental baseline
change. This records inspected scope, not absence of competing methods.

UPRet checkout046366227417e1d8ec14145965403462df345984 has existing local changes.
`modules/modeling.py:570–695` was read, and committed original code at630–690
was separately inspected with `git show HEAD:modules/modeling.py`. The working
copy's explicit inner masks are LOCAL modifications; original code uses
unmasked inner softmax. Do not attribute the local correction to the authors.
Both inspected versions use learned within-sequence text/video outer weights.
The working text weights depend on augmented text features, video weights on
video features; this is not evaluation-candidate-conditioned inference. The
inspected inference branch returns the weighted pair scores, whereas sampled
distribution transport is in the training branch. None of this establishes a
new method; learned scalar pooling is already present in the inspected baseline.
Working file SHA256:
`771452c11c06228b8b54302eead86645fc9875b64f6f6fce858e0ce2e2f926d0`.

SAN checkout82aba9cbc1beb403abef6e9a3875ca52479805c8 had clean status. Read
`models.py:240–292` and `train_vlp_v2.py:396–418`. The displayed scoring blocks
reduce token similarities within each pair. Training explicitly adds hard-text
comparisons; this is not evidence of a candidate-coupled inference rule.
Hashes: models.py
`baef3ea99dffb8b3115331cf2de0ebc7027145a562ec11a31758094e252afa39`;
train_vlp_v2.py
`6e15cd95d41deeee12c7848c37787cdb4a674857412b5db4a5735c44c158917a`.

CMCM checkout5d458719d1da2f082e188cc44705003d919e7e97 had only untracked compiled
cache files. Read `modules/CCG_Module.py` completely, SHA256
`277a06c5bf4a5f256b550fe3c0b88a9e09135cca39052afa8a4976db7b4129f7`.
Its displayed causal attention uses a sequence axis, not candidate competition.
This single module does NOT audit the full CMCM inference pipeline. No absence
of candidate coupling elsewhere is claimed. Possible implementation concerns
outside this question were not patched or promoted as scientific findings.

Decision: this bounded inspection supplies no new supported candidate-relative
mechanism. It also prevents incorrectly presenting learned outer-token weighting
or the locally corrected UPRet masks as novel. Additional code/paper coverage
would be required for any candidate-specific novelty conclusion.
