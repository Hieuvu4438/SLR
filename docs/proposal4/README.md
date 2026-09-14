# Sign Language Retrieval research package

Literature cutoff: **10 September 2026**.

Decision: **GO WITH CONDITIONS** for a falsifiable OCEM pilot. No neural baseline has been reproduced and no SLRet retrieval gain has been measured.

- `slret_research_report.md`: the complete A–Z research report, with primary sources, separate numerical comparison groups, resource/protocol audits, five candidates, novelty challenges, simulated reviews, one selected method, equations, pseudocode and experiment gates.
- `RESEARCH_STATE_CHECKPOINT.md`: authoritative current scientific state and exact next actions. Use this to resume without repeating completed searches or rejected ideas.
- `ocem_reference.py`: CPU implementation of the proposed convex local scorer. This is a mathematical reference, not a retrieval trainer or a pretrained model.
- `ocem_mathematical_checks.json`: results of constructed mathematical checks. These are not sign-language experiments.
- `slret_evidence_audit.json`: endpoint checks, annotation calculations, candidate scores and verification limits.
- `requirements_reference.txt`: versions used for the CPU checks.

To run the reference in an isolated Python environment:

```bash
python -m pip install -r requirements_reference.txt
python ocem_reference.py
```

The recorded environment was Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0. The script writes `ocem_mathematical_checks.json` beside itself. Successful synthetic checks do not establish GPU solver accuracy, baseline reproduction or retrieval performance.

The scorer takes a text-token × local-video-window similarity matrix and the actual video-window intervals. Its support constraint is valid only for features whose receptive fields remain local; do not substitute globally contextual visual tokens without revising the scientific claim. Full equations, masks, null treatment, centering, dual certificates and required controls are in report sections M–P and S.

The proposed pipeline uses public raw data and independently generated representations. **SEDS checkpoints and SEDS/Baidu feature archives are not dependencies.** The complete Oxford checkpoint and all H2S annotation files were downloaded/hashed. Raw-video integrity, full CLIP transfer/parity/loading, exact sample manifests and baseline reproduction remain pending. Earlier H2S test quota failure is resolved; both full CLIP transfer attempts timed out. Unverified CMCM/GTRN details remain explicitly unresolved.

The package excludes third-party paper PDFs, raw datasets and third-party model weights. Their official sources and access status are documented in the report and audit.
