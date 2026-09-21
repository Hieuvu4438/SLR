# C12 — adapt native clip-local temporal aggregation with GCN

## Material Passport / registered card — 2026-09-20

- ARS experiment-agent/run, inline; engineering extension of repeated-seed GCN lead, novelty unresolved.
- Evidence: GCN+fusion beats matched fusion-only across3seeds (pairedmean+.642261pp). Native `signbert.GCN_Conv` was frozen after the adapted GCN.
- Hypothesis: downstream clip-local temporal filters should adapt to the updated joint representation before16-frame mean pooling. Not a proven bottleneck.
- Intervention: enable existing `signbert.GCN_Conv` weights1e-6; GCN1e-6/fusion1e-5 unchanged. Its5-tap/stacked3-tap convolutions and1x1 mixer retain pretrained initialization; BN/dropout stay eval.
- Native path: frame GCN -> gather16frames -> sign_conv -> clip mean -> visual transformer/fusion. No extra branch, loss, scorer or data; pose2D/RGB retained.
- Distinct from user-closed C06: no post-encoder temporal-difference residual, no temporal warp/order objective. Ordinary native encoder adaptation, not renamed C06.
- Release base, seed42 (not selected best1337), TRAIN7096/B32/1epoch222, native losses/FP32 moments; same order/schedule/DEV111+222 and selector as clean GCN seed42.
- Matched control78.034682 seed42; global incumbent78.516378 seed1337 also reported, but cannot attribute cross-seed differences to method.
- Keep initialization as selection candidate, directional reference-.5pp guardrail, earlystopmean-drop>2pp. No TEST loaded or used. Single-seed discovery, further replication needed if promising.
- Seven focused tests pass including new temporal-parameter updates and fixed BN buffers; real-data2step smoke gates pilot. Inference unchanged at initialization.
- Driver `tools/run_masked_pair.py --clip-temporal-pilot`, chain c12-gcn-clip-temporal-001, children seds-gcn-clip-temporal-smoke-001 / seds-gcn-clip-temporal-001.
- Limits60s smoke/800s pilot,parent880s <= remaining897.068655s. Expected650–750s, <=48GiB VRAM, <=4GiB files; disk63GiBfree,V2used29.936GiB within36GiB cap.
- Preserve all best/checkpoints/datasets/cache. No automatic compute top-up; charge children once. If no lead, retain clean GCN and choose only next work feasible within residual resources.
- Background/log/check initial smoke/yield; no training monitoring loop. No SOTA/novelty promise.

Execution: timeoutPID2500740; GPU smoke completed2steps26.831027s, actual native
clip-temporal updates and frozen-buffer gates passed. Main child2501531 confirmed
live at11steps. Retrieval results pending; no further monitoring at handoff.

## Collected result / decision — 2026-09-20

Both children completed successfully: smoke26.831027s, pilot607.510156s,
222updates, no early stop. Actual temporal-weight changes and unchanged frozen
buffers verified. Selected222 T2V77.456647/V2T78.612717, mean78.0346820809:
0pp versus matched GCN+fusion seed42. At111 mean77.745665 also ties matched
control but V2T78.034682 fails reference-.5pp guardrail. R5 tradeoffs do not
change primary-metric conclusion. Global best1337mean78.516378 remains retained;
cross-seed gap is not a causal estimate. Defer this recipe, no broad family claim.
Charge children634.341184s once. Remaining local allocation262.727471s cannot
cover another comparable pilot; no automatic top-up or further launch.
