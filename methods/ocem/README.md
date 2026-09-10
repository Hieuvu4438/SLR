# OCEM

Isolated implementation of Overlap-Constrained Evidence Matching for
sentence-level sign-language retrieval. The scientific and software contracts
are defined in `../../docs/proposal4/OCEM_END_TO_END_IMPLEMENTATION_SPEC.md`.

Current status is recorded in `implementation_state.json`. A passing software
test is not a baseline reproduction, research validation, or SOTA result.

Run the package without installing it:

```bash
PYTHONPATH=src python -m ocem --help
PYTHONPATH=src python -m ocem doctor
PYTHONPATH=src python -m ocem config validate --kind experiment path/to/config.yaml
```

The package never downloads models or datasets during import. Resource
verification and data preparation are explicit CLI operations.

