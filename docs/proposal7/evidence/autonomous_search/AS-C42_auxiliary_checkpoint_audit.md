## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run monitoring / read-only provenance audit
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (local file/schema comparison; training lineage unverified)
- Version Label: AS-C42-auxiliary-v1

## Scope

**Later byte-provenance update:** the public archive was retrieved and both
registered member hashes verified; see the final section below. Training
lineage remains unverified. Earlier observations are retained as history.

While the registered AS-C42 calibration runs unchanged, inspect whether a
separate, already-present sign-retrieval initialization could eventually be
audited as a resource control. No checkpoint was substituted into AS-C42, no
second training run or inference experiment was launched, and no new method is
proposed. No test data or checkpoint from SEDS was read.

## Local observations

`artifacts/pretrained/H2S_sota.pth` exists, 350,469,151 bytes, SHA256
67f557976d523dc3cd63cae73c29c88145e409b77221c90260ae51b9d0c95b35.
Safe CPU loading with `weights_only=True` returns an OrderedDict of 305 tensor
entries, with no non-tensor training metadata. Compared with the already-pinned
PH release checkpoint, all keys, shapes and dtypes match; 6 tensors are exactly
equal and 299 differ. This is a schema/content comparison, not provenance proof
or a numerical retrieval result. The CPU comparison completed with exit0.

The inspected local CiCo README's Testing section explicitly names
`chpt/H2S_sota.pth` for How2Sign and `chpt/ph_sota.pth` for PH, under the same
linked checkpoint archive. However, the existing
`artifacts/pretrained/cico_checkpoint_provenance.json` records the archive hash
and PH member hash only. It does not bind this H2S member's hash to that archive
or document its complete training/selection history. Filename, common mtime,
tensor compatibility, and different weights cannot establish PH-unfittedness.
No claim of clean cross-dataset pretraining or historical PH recipe is made.

The existing provenance archive SHA256 is
f02ea0b2a64123b2c8386ccb467c00143454ee6405ed4c07e074dcc712224c6c,
971,005,695 bytes. A bounded archive-name search did not recover that archive;
temporary-directory permission errors mean the search is incomplete. No
irrelevant archive was opened, no permissions were bypassed, and no source
download was performed. This is not a research blocker for the live AS-C42 run.

## Correction retained

An initial filename filter omitted the `.pth` suffix and incorrectly suggested
the H2S checkpoint was absent. The corrected ignored-file-inclusive listing
found it immediately. The user-facing absence statement was explicitly corrected.
Do not propagate the earlier false absence or count it as a scientific finding.

## Consequence

This is an unverified resource-control lead, not an AS-C43 experiment, a clean
teacher, a candidate, or GO. Any later use first needs provenance verification
and a separate preregistered resource/initialization comparison. Changing
initialization would change the calibration's resource condition, not isolate
training duration, and cannot be bundled into or used to rescue AS-C42. The
current generic-CLIP run continues to its original fixed8800 endpoint regardless
of its intermediate results.

## Public archive verification completed

The official [CiCo repository](https://github.com/FangyunWei/SLRT/tree/main/CiCo)
still links the checkpoint archive and names H2S_sota.pth for How2Sign. Browser
opening the Drive landing page failed; the public download endpoint returned
HTTP200 with filename final_models.zip and expected971,005,695bytes.

`verify_h2_release_member.py` downloaded the public archive into a new scoped
directory, checked its entire SHA256 against the existing provenance, and
streamed only the H2S and PH member bytes through SHA256. No archive paths were
extracted, no downloaded pickle executed, and no dataset loaded. Both member
hashes exactly match the pre-existing local checkpoints. ArchiveSHA256 remains
f02ea0b2a64123b2c8386ccb467c00143454ee6405ed4c07e074dcc712224c6c.

Result `H2S-RELEASE-MEMBER-VERIFY_run.json`: completed exit0,60.894018seconds,
byte_provenance_verified=true, training_lineage_verified=false. Retained archive
`artifacts/proposal7/phase2/H2S-release-audit/final_models.zip` uses971MB additional
disk. It was not used for training or inference. Original assets/provenance
remain untouched. This closes the local-member-to-public-archive integrity gap,
not the PH-unfittedness, selection-history or clean-comparison questions.

The concurrent AS-C42 interruption and recovery are documented separately.
Archive verification does not validate or modify that training computation.
