# C16 matched control — V4 finite launch

User granted additional18000s local compute and at most20GB VRAM.
Job: `artifacts/slret_goal/jobs/v4-c16-control-001`.
Training: `artifacts/slret_goal_v2/seds-signrep-control-offload-001`.
Launcher start Unix1789903170.5041273; outerPID3559678,
supervisor3559679, torchrun3559681. One startup inspection: RUNNING,
initial fullDEV519 entered; no completion claim. WAITING_FOR_USER, no polling.

Scientific question: does SignRep auxiliary training improve this matched
TRAIN512 /160-update seed42 recipe over the same GCN+fusion adaptation without
auxiliary loss? Keep batch32, native negatives, checkpoint, sample order, LR,
offload/mathSDPA and evaluation schedule. No new architecture or baseline replay
campaign. ARS experiment workflow informed the matched contrast and attribution
boundary; current transfer78.034682 remains exploratory, below incumbent78.709056.
At collection, compare selected AND fixed160 scores; a tie or loss provides no
reason to scale this auxiliary recipe on current evidence. A positive contrast
needs magnitude/curve review before deciding whether to scale/refine; no autoqueue.

Resource-only differences from transfer: allocator16GiB cap and process-group
watchdog trip19GB below conservative decimal20GB user ceiling; total memory is
sampled, not a hardware partition. No other user GPU process managed. Timeout1800s,
outer1850s; reserve1850s, charge actual supervisor once. Available18365.97095631149s,
unreserved16515.97095631149s. No extra1800s grant counted. No cloud/unlimited budget.
Storage reservation1.25GiB from measured compact save sizes, preserves36GiB cap
and15GiB free floor; no checkpoints removed. Native model/objective unchanged.

CPU checks: py_compile both modified entrypoints; comparison accepts matching
fixture and rejects config mismatch/failed report; memory parser isolates own
process group. Successful prior C16 already covers unchanged training path.
Source hashes/command in launch.json; log/status/summary contract provided.
Future collection must check terminal exit, expected last/selected checkpoint,
paired config/order/initial recalls and measured metrics; raw cross-process score
parity remains an explicit historical caveat, not claimed resolved.
