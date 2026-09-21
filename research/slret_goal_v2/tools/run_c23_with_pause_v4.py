"""Run registered C23 while temporarily pausing the exact UniFormerV2 worker."""
import sys
from pathlib import Path

import run_c21_with_pause_v4 as wrapper

wrapper.JOB='v4-c23-graph-bottleneck-adapter-001'
wrapper.OUT=wrapper.ROOT/'artifacts/slret_goal/jobs'/wrapper.JOB
wrapper.REPORT=wrapper.OUT/'pause_contract.json'
wrapper.CANDIDATE_ENTRY=wrapper.ROOT/'research/slret_goal_v2/tools/run_c23_graph_adapter_v4.py'
wrapper.WRAPPER_ENTRY=Path(__file__).resolve()

if __name__=='__main__':
    if sys.argv[1:]==['--launch']:wrapper.launch()
    elif sys.argv[1:]:raise SystemExit('Only --launch is accepted')
    else:wrapper.main()
