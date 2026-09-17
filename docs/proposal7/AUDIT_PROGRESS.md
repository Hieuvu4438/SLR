# Proposal 7 — investigation checkpoint

Status: **FINAL RESEARCH REPORT WRITTEN; NO-GO for selecting a new SOTA method after two cycles.**

Current deliverable: [Vietnamese report](SLRet_SOTA_Method_Proposal.md), all 30 guide sections. [RQ/candidate record](Research_Questions_and_Candidate_Screen.md) contains four mechanism-distinct candidates, four inline review perspectives, two search/diagnostic cycles and redesign decisions. [Final checks](evidence/report_verification.json): 21 focused tests pass, Ruff passes, citation/section/local-link checks pass. No new method training/test retrieval was run. Unknown resource/literature details remain explicitly unknown; no claim of achieved SOTA or exhaustive literature certification.

The chronological checkpoint notes below are retained as a reading/evidence ledger. Their earlier “remaining”/“not begun” statements are superseded by the final report and this status, not erased retroactively. The last goal turn made **progress**: created the final report and new measured information-channel evidence, completed candidate screening and executed verification. No running job requires a wait.

User authority: `docs/guide/Astra_SLRet_SOTA_Research_Prompt.md`. All proposal1–6 directions, including conceptual variants, are CLOSED irrespective of missing logs. Source datasets and proposals1–6 remain unchanged. No commits/pushes; no SEDS pretrained assets. Literature ideation must follow the mandatory prior-project audit.

## Reproducible evidence already generated

- `tools/audit_workspace.py`: source inventories, manifest forensics, compact saved-dev run ledger and historical gates. Outputs in `evidence/` and `artifacts/proposal7/forensics/`.
- `tools/diagnose_saved_ph_dev.py`: recomputes saved PH dev ranks, verifies checkpoint hashes and 519 ordered IDs, measures length/rarity/duplicate strata and paired source-cluster bootstrap. Output `evidence/ph_dev_residuals.json`.
- Both scripts completed successfully. No training or new test retrieval evaluation was run.
- Existing git HEAD: `a5fe287536db55b44ce519050a23d389f27d9c2e`; source inventory records upstream pins.

## Evidence cautions

- PH dev: short clips/captions have lower recall than long ones; correlation is not a causal diagnosis. BPE overflow affects only 10/519 dev captions, so it does not explain most PH failures.
- ELSC gates reject its improvement hypothesis. Two Min seeds selected epoch −1. Verified in `methods/elsc/elsc/train.py:729–775`: initialization is evaluated and saved as the first selection candidate before training.
- OCEM Stage-A gate is NO_GO_SCIENTIFIC, not the older proposal4 GO checkpoint. PMGR improvement over the strongest equal-input control is below its gate.
- H2 dev TSV and local JSON have different populations. Do not silently evaluate the reduced JSON as the full gallery. TSV: 1,741 recording IDs/1,529 sentence IDs; JSON: 1,527 recordings. Two TSV videos missing, 212 alternate performances omitted by JSON; exact generation cause still requires tracing.
- CSL dev/test near-complete exact sentence-text overlap reflects repeated performances; do not label it contamination without establishing use. Do not turn dev captions into new training supervision.
- Historical checkpoint/resource provenance is distinct from reproducing a paper's dev-selected training protocol.
- Never print full saved metric JSON: some contain full rankings and are tens of megabytes. Read selected structured fields only.

## Semantic reading coverage inherited at checkpoint

This is a reading ledger, not an assertion that file inventory equals semantic review.

Complete: proposal1/datasets.md, ELSC_End_to_End_Implementation.md, IMPLEMENTATION_STATUS.md; proposal2 DIVE_SLR_Reviewed_Proposal.md, DIVE_SLR_End_to_End_Implementation_Spec.md, IMPLEMENTATION_STATUS.md, BASELINE_ADAPTATIONS.md, DESIGN_DEVIATIONS.md, REPRODUCIBILITY.md; proposal3/sign_language_retrieval_research_proposal.md; proposal6/SLRet_Research_Report.md. Proposal4 RESEARCH_STATE_CHECKPOINT.md was read completely (1–100 separately, 101–210 in preceding output, 211–280 separately).

Proposal1 PDF: all 23 extracted text units read. Initial ARS preflight UNAVAILABLE (missing pypdf); installed pypdf 6.18.1 into isolated `artifacts/proposal7/pdf_dependencies`, reran and retained both sidecars. New verdict remains UNAVAILABLE: all page counts 23, but xref-coverage warning on object 120. Page anchors are **not trusted**; some extracted glyphs corrupt, e.g. training pseudocode. Corresponding method corroborated by complete Markdown spec. No source PDF edits.

Partial/pending:

- proposal4: main report/spec/README/reference Python/mathematical checks/requirements/evidence JSON/artifact manifest completely read. Checkpoints 01/02 read completely; 03–06 read through complete exact unified diffs against the preceding fully read checkpoint (unchanged lines already covered); checkpoint 07 through its complete exact diff against the fully read current checkpoint. These are exact-source readings, not generated summaries. Historical GO/resource claims are superseded by actual later gates.
- proposal5 report and all 1,316 lines of Method1 implementation spec now complete. A truncated milestone table was explicitly reread at 1230–1266. Shared/independent support, optional reliability, SAN/FSC controls, cached teacher and gradient contracts are all closed prior directions.
- proposal6 report and all 1,432 lines of PMGR implementation spec now complete. Also fully read actual IMPLEMENTATION_STATUS: rank extension intentionally stopped after Phase-B failure. Mask-corrected checkpoint recall is lower than legacy by 2.384/2.878 R@1 points; that is a common scoring change, not a method effect.
- Guide remainder 541–1286 completely reread; all 30 final sections and two-cycle no-go rule confirmed. methods/README and shared/README isolation/dependency boundaries read.
- Implementations/configs/tests/patches: relevant call-path audit recorded in `Implementation_Audit.md`; historical document coverage complete. This is targeted source review, not certification of every repository file. Combined evaluation/reconciliation/package-boundary suite: **17 passed**.
- Current literature search **started 2026-09-14** after that audit and registry. See `Literature_Search_Log.md` for actual queries, exact primary-source links, reading coverage and remaining work. Do not claim completed search or candidate selection.
- H2 test restricted annotation decoding and upstream reconciliation now complete for locally present CiCo/Uni-Sign variants. Full raw official test TSV, pose/features provenance, temporal distributions, near-duplicate/template diagnostics remain pending.

## Latest How2Sign reconciliation (new measured evidence)

`tools/reconcile_how2sign.py` completed; `evidence/how2sign_reconciliation.json` and immutable-format ordered manifests in `artifacts/proposal7/forensics/`.

- Legacy annotation loading denies all pickle globals. Uni-Sign test annotation is gzip-compressed; CiCo annotations are primitive pickles. No pose/model pickle or SEDS asset loaded.
- **Do not map CiCo `clipN` to sentence N.** Clip ordinals differ after merging/filtering. That rejected mapping creates 8,301 train and 680 test caption disagreements among matching guessed filenames. Correct ID = CiCo sentence-group dictionary key + performance suffix extracted from `video_name`.
- Correct mapping: all 31,085 CiCo train entries and 2,348 CiCo test entries match local annotation IDs and captions exactly. Local train has 80 additional rows; Uni-Sign test has one additional row.
- Actual missing named videos under the shared root, with correct mapping: **44 CiCo train / 6 CiCo test**, versus 118 raw-train annotations missing. Full CiCo gallery reproduction still needs those assets, not an implicit intersection.
- H2 JSON dev is exactly consistent with retaining the **last** TSV row for each retained sentence ID (1,527/1,527); first-row matching only 1,315. Two absent groups match the known absent videos. Historical generator not located, so last-wins is a measured equivalence, not an asserted causal code history.
- Previous goal turn = **progress**: completed doc reading, produced PDF integrity evidence, and resolved annotation identity with exact-content checks. No process remains running from these actions.
- `Negative_Results_Registry.md` created with named and conceptual closures, specific negative evidence and missing-evidence qualifications. No candidate generated yet.
- Reconciliation helper tests: first collection failed because of local-module import under repository pytest import mode; corrected package/script dual imports. `python -m pytest -q docs/proposal7/tools/test_reconciliation.py`: **5 passed**. These test safe decoding, sentence-versus-ordinal identity, mismatch refusal and comparisons, not retrieval efficacy.
- `python -m ruff check docs/proposal7/tools`: all checks passed. PH diagnostic script rerun after function/constant cleanup produced exactly the same parsed JSON evidence.

## Earlier checkpoint: then-remaining deliverables (now addressed by final report)

1. Preserve completed historical audit and Negative Results Registry as the exclusion contract; investigate any new implementation-specific question as it arises.
2. Verify literature/current repos and protocol-aware A/B/C comparison, actual date cutoff.
3. Complete dev-only residual diagnostics and identify falsifiable causal gaps.
4. At least four distinct candidates; semantic collision/novelty checks; inline four-perspective review. No spawned agents without new explicit authorization.
5. Deliver Vietnamese `SLRet_SOTA_Method_Proposal.md` with all guide sections, explicit [V]/[A]/[M]/[I]/[H]/[U], implementation/pilot/kill plans and source links. If no candidate survives two honest search/diagnostic cycles, report NO-GO rather than invent novelty.

## New current-literature/code checkpoint

- Official SAN and CMCM repositories cloned into `third_party/` and pinned in the search log; no source edits, training or pretrained asset downloads.
- SAN's constructed stress gains do not imply standard-gallery improvement. Its public default trainer selects checkpoints using the configured **test** loader despite `dev_*` names; the actual paper-run configuration remains unverified.
- CMCM public repository is incomplete for reproduction. Asset-free CPU probes confirm undefined `DEVICE` and a causal-mask dimension failure for 64 tokens, with a successful 1,024-token control. See `evidence/new_upstream_audit.json` and `tools/audit_new_upstreams.py`.
- SEDS paper has H2 31,019/1,738/2,348 retained pairs, unlike CiCo/C²RL 31,085/1,739/2,348. Equal test cardinality is not proof of equal gallery IDs. SEDS assets remain prohibited and unused.
- C²RL official code not found in executed queries; final TCSVT DOI discovered but publisher verification still pending. ArXiv main text/settings/results read; no invented final-version parity.
- Current goal turn = **progress**, not blocked. New source records and independently reproducible public-code diagnostics completed; candidate generation has not begun.
