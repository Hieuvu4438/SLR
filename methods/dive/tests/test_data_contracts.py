from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from dive.data.collate import pad_feature_sequences
from dive.data.manifest import ManifestError, SampleRecord, load_manifest, validate_split_disjoint
from dive.data.pose import normalize_pose_per_step
from dive.data.relations import RelationError, build_pair_relations
from dive.data.temporal import (
    FrameMap,
    TemporalError,
    build_canonical_grid,
    pose_receptive_field_intervals,
    require_distinct_views,
)
from dive.data.text_units import (
    TextUnitError,
    map_units_to_subwords,
    normalize_text,
    require_target,
    unitize,
)


def _record(identifier: str, split: str) -> SampleRecord:
    return SampleRecord(
        schema_version="sample.v1",
        sample_id=f"sample_{identifier}",
        video_id=f"video_{identifier}",
        text_id=f"text_{identifier}",
        split=split,
        sign_language="ASL",
        text_language_original="en",
        text_language_model="en",
        text_original="the value is 12 meters",
        text_model="the value is 12 meters",
        source_video_id=None,
        signer_id=None,
        source_start_sec=None,
        source_end_sec=None,
        duration_sec=4.0,
        video_path=f"{identifier}.mp4",
        pose_path=f"{identifier}.pkl",
        rgb_feature_key=f"rgb_{identifier}",
        translation_artifact_hash=None,
        frame_map_key=f"frame_{identifier}",
        annotation_provenance="synthetic_fixture",
    )


def test_manifest_split_route_and_disjoint_guards(tmp_path):
    train = _record("train", "train")
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(train.to_dict()) + "\n", encoding="utf-8")
    assert load_manifest(path, expected_split="train") == (train,)
    with pytest.raises(ManifestError, match="expected split dev"):
        load_manifest(path, expected_split="dev")
    dev = replace(train, split="dev")
    with pytest.raises(ManifestError, match="split overlap"):
        validate_split_disjoint({"train": [train], "dev": [dev]})


def test_unit_offsets_round_trip_and_partial_target_rejection():
    text = normalize_text("  the   value is 12 meters  ")
    units = unitize(text)
    assert [item.text for item in units] == ["the", "value", "is", "12 meters"]
    token_ids = [49406, 10, 11, 12, 13, 14, 49407]
    offsets = [None, (0, 3), (4, 9), (10, 12), (13, 15), (16, 22), None]
    mapping = map_units_to_subwords(units, token_ids, offsets)
    target = require_target(mapping, 3)
    assert target.subword_indices == (4, 5)
    assert "".join(text[left:right] for left, right in (offsets[i] for i in target.subword_indices)) == "12meters"
    partial = map_units_to_subwords(units, token_ids[:-2], offsets[:-2])
    assert partial[3].complete_after_truncation is False
    with pytest.raises(TextUnitError, match="incomplete after truncation"):
        require_target(partial, 3)
    with pytest.raises(TextUnitError, match="out of range"):
        require_target(mapping, 99)


def test_pair_relations_keep_positives_and_exclude_ambiguous_negatives():
    relations = build_pair_relations(
        ["v0", "v1"],
        ["t0", "t1"],
        {"v0": {"t0"}, "v1": {"t1"}},
        excluded_pairs={("v0", "t1")},
    )
    assert relations.positives.tolist() == [[True, False], [False, True]]
    assert relations.candidates.tolist() == [[True, False], [True, True]]
    with pytest.raises(RelationError, match="positive cannot be excluded"):
        build_pair_relations(["v0"], ["t0"], {"v0": {"t0"}}, {("v0", "t0")})


def _frame_map(count: int, step_sec: float = 0.08) -> FrameMap:
    intervals = tuple((index * step_sec, index * step_sec + 0.04) for index in range(count))
    return FrameMap(
        sample_id="sample",
        duration_sec=max(0.04, intervals[-1][1]),
        raw_frame_indices=tuple(index * 2 for index in range(count)),
        input_step_intervals_sec=intervals,
        sampling_policy="every_second_raw_frame_v1",
    )


def test_canonical_grid_is_bounded_deterministic_and_maps_rf_to_raw_time():
    frame_map = _frame_map(80)
    first = build_canonical_grid(frame_map, clip_steps=16, stride_steps=1, max_clips=64)
    second = build_canonical_grid(frame_map, clip_steps=16, stride_steps=1, max_clips=64)
    assert first == second
    assert len(first.starts) == len(set(first.starts)) == 64
    assert first.starts[0] == 0 and first.starts[-1] == 64
    receptive_fields = pose_receptive_field_intervals(frame_map, first)
    assert receptive_fields[0] == pytest.approx((0.0, 1.56))
    assert all(right > left for left, right in receptive_fields)


def test_short_video_does_not_fake_distinct_support_views():
    frame_map = _frame_map(10)
    left = build_canonical_grid(frame_map, view_offset_steps=-1)
    right = build_canonical_grid(frame_map, view_offset_steps=1)
    assert left.valid_steps == (10,)
    with pytest.raises(TemporalError, match="NO_DISTINCT_SUPPORT_VIEWS"):
        require_distinct_views(left, right)


def test_pose_normalization_adds_no_cross_step_dependency():
    pose = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
            [[0.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
            [[0.0, 2.0, 1.0], [2.0, 2.0, 1.0]],
        ]
    )
    valid = torch.ones(3, 2, dtype=torch.bool)
    original, original_valid = normalize_pose_per_step(pose, valid)
    changed_pose = pose.clone()
    changed_pose[1, :, :2] += torch.tensor([100.0, -50.0])
    changed, changed_valid = normalize_pose_per_step(changed_pose, valid)
    torch.testing.assert_close(changed[[0, 2]], original[[0, 2]])
    assert torch.equal(original_valid, changed_valid)


def test_collate_uses_zero_padding_and_true_valid_masks():
    first = torch.ones(2, 3)
    second = torch.full((1, 3), 2.0)
    values, valid = pad_feature_sequences([first, second])
    assert valid.dtype == torch.bool
    assert valid.tolist() == [[True, True], [True, False]]
    assert torch.equal(values[1, 1], torch.zeros(3))
