"""TRAIN-only census of an already documented CiCo pseudo-clip grouping issue."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
INDEX = "methods/ocem/runs/wp04_phoenix/pseudolabel_index.jsonl"
TRAIN = "artifacts/manifests/ph_train.jsonl"
SAMPLER = "methods/ocem/src/ocem/data/pseudoclips.py"
PROTOCOL = "docs/proposal7/evidence/autonomous_search/CICO_pseudoclip_protocol.md"
EXPECTED = {
    INDEX: "f247de98d1917dee969d71dfef971debcd57e7002425dd8313fa3f4a2c07c432",
    TRAIN: "f032260cf21578876fcff997bef1df0d46094635d30032edb6b01c80693ace53",
    SAMPLER: "0369f56ec9e718df94f1f285422ef7e0e9849d7723e3a8c5da045b11c1127e19",
    "methods/ocem/src/ocem/data/pseudolabels.py": "62d745e3965d5516c86fb4a7efa8451161a30df623eb1ad4c5233465b092bf82",
    "third_party/SLRT/CiCo/I3D_feature_extractor/epoch_pseudo.py": "6bb49c02d1e51f3f6f45a2c436f78d0cf5caa7880a407706c41dfe5b2a5f93b8",
    "methods/ocem/locks/adaptation_run.phoenix2014t.json": "0f35bbede709537934336a9c58699e39e0dbf9c84352c66c8e86e6485c61b78d",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def interval_counts(starts: list[int], lo: int, end: int) -> dict[str, int]:
    """Independent start-interval / per-frame multiplicity calculation."""
    coverage = union([(max(lo, s), min(end, s + 16)) for s in starts])
    gaps = []
    cursor = lo
    for a, b in coverage:
        if a > cursor:
            gaps.append((cursor, a))
        cursor = b
    if cursor < end:
        gaps.append((cursor, end))
    last = max(lo, end - 16)
    if end - lo < 16:
        # All supported fixture/index windows are >=16 before materialization;
        # a short materialization is therefore fully covered, padding included.
        assert not gaps
        return dict(starts=1, any_gap=0, all_gap=0, gap_frame_draws=0, gap_frames=0)
    hit = union([(max(lo, a - 15), min(last + 1, b)) for a, b in gaps])
    full = union([(max(lo, a), min(last + 1, b - 15)) for a, b in gaps])
    multiplicity = sum(
        max(0, min(last, frame) - max(lo, frame - 15) + 1)
        for a, b in gaps for frame in range(a, b)
    )
    return dict(starts=last - lo + 1, any_gap=sum(b-a for a,b in hit),
                all_gap=sum(b-a for a,b in full), gap_frame_draws=multiplicity,
                gap_frames=sum(b-a for a,b in gaps))


def enumerate_counts(starts: list[int], lo: int, end: int) -> dict[str, int]:
    covered = {frame for s in starts for frame in range(s, s + 16)}
    draws = [sum(min(start + offset, end - 1) not in covered for offset in range(16))
             for start in range(lo, max(lo, end - 16) + 1)]
    return dict(starts=len(draws), any_gap=sum(n > 0 for n in draws),
                all_gap=sum(n == 16 for n in draws), gap_frame_draws=sum(draws),
                gap_frames=sum(f not in covered for f in range(lo, end)))


class OffsetRng:
    def __init__(self, offset: int, maximum: int):
        self.offset, self.maximum = offset, maximum

    def randint(self, low: int, high: int) -> int:
        assert (low, high) == (0, self.maximum)
        return self.offset


def isolated_sampler():
    tree = ast.parse((ROOT / SAMPLER).read_text())
    fn, = [n for n in tree.body if isinstance(n, ast.FunctionDef)
           and n.name == "temporal_sample_start"]
    ns = dict(Mapping=Mapping, Any=Any, random=random, PseudoClipError=ValueError)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), SAMPLER, "exec"), ns)
    return ns[fn.name]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    assert n
    return {
        "segments": n,
        "source_videos": len({r["source_sample_id"] for r in rows}),
        "source_videos_with_nonlocal_merge": len({r["source_sample_id"] for r in rows if r["nonlocal_merge"]}),
        "nonlocal_segments": sum(r["nonlocal_merge"] for r in rows),
        "segments_with_gap": sum(r["gap_frames"] > 0 for r in rows),
        "total_enumerated_crop_starts": sum(r["starts"] for r in rows),
        "mean_probability_any_gap": math.fsum(r["any_gap"] / r["starts"] for r in rows) / n,
        "mean_probability_all_gap": math.fsum(r["all_gap"] / r["starts"] for r in rows) / n,
        "mean_uncovered_frame_fraction": math.fsum(r["gap_frame_draws"] / (16*r["starts"]) for r in rows) / n,
        "max_materialized_frames": max(r["duration"] for r in rows),
        "max_gap_frames": max(r["gap_frames"] for r in rows),
        "single_window_segments": sum(r["merged_window_count"] == 1 for r in rows),
    }


def run() -> dict:
    started = time.monotonic()
    paths = [*EXPECTED, PROTOCOL, str(Path(__file__).relative_to(ROOT))]
    hashes = {p: sha(ROOT / p) for p in paths}
    assert all(hashes[p] == h for p, h in EXPECTED.items())
    train = [json.loads(line) for line in (ROOT / TRAIN).read_text().splitlines()]
    assert all(r["split"] == "train" for r in train)
    ids = {r["video_id"] for r in train}
    assert len(train) == len(ids) == 7096
    records = [json.loads(line) for line in (ROOT / INDEX).read_text().splitlines()]
    assert len(records) == len({r["pseudo_id"] for r in records}) == 10992
    assert {r["adaptation_split"] for r in records} == {"train", "holdout"}
    sampler = isolated_sampler()
    rows = []
    for r in records:
        assert r["source_sample_id"] in ids
        assert r["schema_version"] == "ocem.p14t_pseudolabel.v1"
        assert r["materialization_semantics"] == "cico_epoch_pseudo_python_slice_v1"
        starts = r["merged_window_starts"]
        assert starts and all(type(s) is int and s >= 0 for s in starts)
        assert r["merged_window_count"] == len(starts)
        assert r["anchor_start_frame"] in starts
        lo, end = r["upstream_clip_start_frame"], r["upstream_clip_end_frame_exclusive"]
        assert lo == r["support_start_frame"] == min(starts)
        assert end + 1 == r["support_end_frame_exclusive"] == max(starts) + 16
        counts = enumerate_counts(starts, lo, end)
        assert counts == interval_counts(starts, lo, end), r["pseudo_id"]
        maximum = max(0, end - lo - 16)
        for offset in range(maximum + 1):
            assert sampler(r, training=True, rng=OffsetRng(offset, maximum)) == lo + offset
        assert sampler(r, training=False) == lo + maximum // 2
        rows.append(dict(pseudo_id=r["pseudo_id"], source_sample_id=r["source_sample_id"],
                         adaptation_split=r["adaptation_split"], duration=end-lo,
                         merged_window_count=len(starts),
                         nonlocal_merge=any(abs(s-r["anchor_start_frame"]) > 3 for s in starts),
                         **counts))
    groups = {s: summarize([r for r in rows if r["adaptation_split"] == s])
              for s in ("train", "holdout")}
    groups["all"] = summarize(rows)
    assert groups["train"]["segments"] == 9854 and groups["holdout"]["segments"] == 1138
    assert hashes == {p: sha(ROOT / p) for p in paths}
    return dict(schema="cico-pseudoclip-census-v1", pid=os.getpid(),
                input_hashes=hashes, official_train_videos=len(ids), groups=groups,
                validation=dict(all_integer_counts_exact=True, all_sampler_offsets_exact=True,
                                inputs_unchanged=True, test_access=False, model_updates=0),
                exposure_threshold=0.10,
                exposure_lead=groups["train"]["mean_probability_any_gap"] >= 0.10,
                method_go=False, rows=rows, wall_seconds=time.monotonic()-started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Refusing to overwrite a census"
    result = run()
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
