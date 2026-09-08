from __future__ import annotations

import copy

import pytest
import torch

from dive.cache import (
    CacheError,
    TensorShard,
    load_tensor_cache,
    make_cache_fingerprint,
    write_tensor_cache,
)


def _components() -> dict[str, object]:
    return {
        "student_checkpoint": "student_sha",
        "reference": "reference_sha",
        "baseline": "baseline_sha",
        "text_contracts": {"tokenizer": "tokenizer_sha", "unitizer": "units_v1"},
        "grid": "canonical_grid_sha",
        "dtype": "float32",
    }


def _fingerprint(**changes):
    components = copy.deepcopy(_components())
    components.update(changes)
    return make_cache_fingerprint("student_gallery", components)


def _shards() -> tuple[TensorShard, ...]:
    return (
        TensorShard(
            shard_id="00000",
            ordered_ids=("v0", "v1"),
            tensors={"student": torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)},
            masks={"video": torch.tensor([[True, True], [True, False]])},
            timestamps={
                "raw_seconds": torch.tensor(
                    [[[0.0, 0.5], [0.5, 1.0]], [[0.0, 0.5], [0.5, 1.0]]]
                )
            },
            metadata={"grid_id": "canonical"},
        ),
        TensorShard(
            shard_id="00001",
            ordered_ids=("v2",),
            tensors={"student": torch.ones(1, 2, 3)},
            masks={"video": torch.tensor([[True, True]])},
            timestamps={"raw_seconds": torch.tensor([[[0.0, 0.5], [0.5, 1.0]]])},
            metadata={"grid_id": "canonical"},
        ),
    )


def test_tensor_cache_round_trip_preserves_ids_masks_timestamps_and_namespace(tmp_path):
    fingerprint = _fingerprint()
    write_tensor_cache(
        tmp_path,
        namespace="test",
        fingerprint=fingerprint,
        shards=_shards(),
    )
    loaded = load_tensor_cache(
        tmp_path,
        expected_namespace="test",
        expected_fingerprint=fingerprint,
        expected_ordered_ids=("v0", "v1", "v2"),
    )
    assert tuple(item for shard in loaded for item in shard.ordered_ids) == ("v0", "v1", "v2")
    assert loaded[0].masks["video"].dtype == torch.bool
    torch.testing.assert_close(loaded[0].timestamps["raw_seconds"], _shards()[0].timestamps["raw_seconds"])
    with pytest.raises(CacheError, match="CACHE_NAMESPACE_MISMATCH"):
        load_tensor_cache(
            tmp_path,
            expected_namespace="train",
            expected_fingerprint=fingerprint,
        )
    with pytest.raises(CacheError, match="CACHE_ORDER_MISMATCH"):
        load_tensor_cache(
            tmp_path,
            expected_namespace="test",
            expected_fingerprint=fingerprint,
            expected_ordered_ids=("v1", "v0", "v2"),
        )


@pytest.mark.parametrize(
    "changed",
    [
        {"grid": "changed_grid"},
        {"reference": "changed_reference"},
        {"dtype": "float16"},
        {"text_contracts": {"tokenizer": "changed_tokenizer", "unitizer": "units_v1"}},
    ],
)
def test_grid_tokenizer_reference_or_precision_change_rejects_stale_cache(tmp_path, changed):
    fingerprint = _fingerprint()
    write_tensor_cache(tmp_path, namespace="dev", fingerprint=fingerprint, shards=_shards())
    with pytest.raises(CacheError, match="CACHE_HASH_MISMATCH"):
        load_tensor_cache(
            tmp_path,
            expected_namespace="dev",
            expected_fingerprint=_fingerprint(**changed),
        )


def test_cache_verifies_checksum_before_deserialization(tmp_path):
    fingerprint = _fingerprint()
    write_tensor_cache(tmp_path, namespace="dev", fingerprint=fingerprint, shards=_shards())
    with (tmp_path / "shard-00000.pt").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(CacheError, match="checksum mismatch"):
        load_tensor_cache(
            tmp_path,
            expected_namespace="dev",
            expected_fingerprint=fingerprint,
        )


def test_reference_correctness_cache_requires_float32_and_masks_are_bool(tmp_path):
    components = {
        "reference_state": "reference_sha",
        "rgb_preprocess": "rgb_v1",
        "pose_preprocess": "pose_v1",
        "grid_view": "canonical",
        "bn_policy": "frozen_stats_affine",
        "dtype": "float16",
    }
    with pytest.raises(CacheError, match="must use float32"):
        make_cache_fingerprint("reference_local", components)
    bad = list(_shards())
    bad[0] = TensorShard(
        **{**bad[0].__dict__, "masks": {"video": torch.ones(2, 2)}}
    )
    with pytest.raises(CacheError, match="must be bool"):
        write_tensor_cache(
            tmp_path,
            namespace="dev",
            fingerprint=_fingerprint(),
            shards=bad,
        )
