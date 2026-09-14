# PMGR

Isolated implementation of Protocol-Matched Gallery Risk (proposal 6) on the pinned CiCo
retriever. The implementation contract is
[`../../docs/proposal6/PMGR_Implementation_Specification.md`](../../docs/proposal6/PMGR_Implementation_Specification.md).

PMGR keeps CiCo's encoders and parameter names, adds no inference module, and changes only the
training population/objective. It uses complete existing sentence groups, the deployed mixed
late-interaction score, group-max T2V risk, and population-weighted V2T risk. A passing software
suite is not evidence that PMGR improves retrieval.

Run without installing:

```bash
export PYTHONPATH=methods/pmgr/src:shared
python -m pmgr.runtime audit --config methods/pmgr/configs/pmgr_csl.json
python -m pytest methods/pmgr/tests -q
python -m pmgr.train --config methods/pmgr/configs/pmgr_csl.json \
  --engine direct --effective-groups 4 --max-updates 3 --output-dir runs/pmgr_smoke
```

The implementation/test/baseline boundary is recorded in
[`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md). To queue the matched population-only
pilot after GPU 0 becomes idle, commit the relevant code first and launch:

```bash
chmod +x methods/pmgr/scripts/run_csl_phase_b_queue.sh
PMGR_EXPECTED_GIT_HEAD=$(git rev-parse HEAD) \
  methods/pmgr/scripts/run_csl_phase_b_queue.sh
```

The runner requires at least 42,000 MiB free GPU memory, at most 10% utilization for three
consecutive polls, and 20 GiB free disk. It locks itself against duplicate PMGR queues, rechecks
the pinned Git commit before every arm, resumes a compatible `last.pt`, and never evaluates test.
The thresholds and timeouts can be changed with the documented `PMGR_*` environment variables
at the top of the script.

Generated indexes, run logs, score caches, and checkpoints belong under ignored `artifacts/` or
`runs/`; they are never committed. Test evaluation is a separate locked command and training only
selects checkpoints on the official dev split.
