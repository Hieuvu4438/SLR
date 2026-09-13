from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from method1.reference import (
    ReferenceCacheError,
    ReferenceCacheIdentity,
    encode_reference_text,
    encode_reference_video,
    freeze_reference,
    pool_reference_spans,
    validate_reference_cache,
    validate_negative_span_cache,
    write_negative_span_cache,
    write_reference_cache,
)


class FakeReference(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))

    def get_video_feat(self, video, mask, **kwargs):
        batch = video.shape[0]
        tokens = torch.zeros(batch, 65, 2)
        tokens[:, 1, 0] = 2.0
        tokens[:, 2, 1] = 3.0
        return mask, tokens, torch.zeros(batch, 2)

    def get_text_feat(self, input_ids, token_type_ids, text_valid, **kwargs):
        tokens = torch.stack((input_ids.float(), input_ids.float() + 1.0), dim=-1)
        return text_valid, tokens, torch.zeros(input_ids.shape[0], 2)


def _identity(**changes) -> ReferenceCacheIdentity:
    values = {
        "teacher_checkpoint_sha256": "teacher",
        "tokenizer_sha256": "tokenizer",
        "caption_manifest_sha256": "captions",
        "feature_manifest_sha256": "features",
        "mixture_sampler_sha256": "sampler",
        "implementation_version": "reference_cache_v1",
    }
    values.update(changes)
    return ReferenceCacheIdentity(**values)


def test_reference_video_is_frozen_normalized_and_zero_padded() -> None:
    reference = freeze_reference(FakeReference())
    mask = torch.ones(1, 65, dtype=torch.bool)
    mask[:, 1:3] = False
    tokens, valid = encode_reference_video(
        reference,
        {"video_features": torch.zeros(1, 1024, 64, 1), "video_ignore_raw": mask},
    )
    assert not tokens.requires_grad
    torch.testing.assert_close(tokens[0, :2], torch.eye(2))
    torch.testing.assert_close(tokens[0, 2:], torch.zeros(62, 2))
    assert valid.sum().item() == 2


def test_reference_text_pools_raw_tokens_before_normalization() -> None:
    reference = freeze_reference(FakeReference())
    ids = torch.tensor([[0, 1, 3, 9, 0]])
    valid = torch.tensor([[True, True, True, True, False]])
    tokens, returned_valid = encode_reference_text(
        reference, ids, torch.zeros_like(ids), valid
    )
    pooled = pool_reference_spans(reference, tokens, [(1, 2)])
    expected_raw = torch.tensor([[2.0, 3.0]])
    torch.testing.assert_close(pooled, torch.nn.functional.normalize(expected_raw, dim=-1))
    torch.testing.assert_close(returned_valid, valid)


def test_cache_round_trip_mmap_and_strict_identity(tmp_path: Path) -> None:
    arrays = {
        "reference_video_tokens": np.array(
            [[[1.0, 0.0], [0.0, 1.0], *([[0.0, 0.0]] * 62)]], dtype=np.float32
        ),
        "reference_video_valid": np.array([[True, True, *([False] * 62)]], dtype=np.bool_),
        "reference_positive_spans": np.array([[0.6, 0.8]], dtype=np.float32),
    }
    destination = tmp_path / "reference"
    metadata = write_reference_cache(
        destination,
        identity=_identity(),
        arrays=arrays,
        video_index=[
            {
                "row": 0,
                "video_uid": "video-1",
                "selected_feature_indices": [0, 1, *([-1] * 62)],
                "valid_length": 2,
                "agnostic_sha256": "a",
                "aware_sha256": "b",
            }
        ],
        positive_span_index=[
            {"row": 0, "text_uid": "text-1", "occurrence_uid": "text-1:0:4", "positions": [1]}
        ],
    )
    assert metadata["status"] == "complete"
    loaded = validate_reference_cache(destination, expected_identity=_identity())
    assert isinstance(loaded["reference_video_tokens"], np.memmap)
    np.testing.assert_array_equal(loaded["reference_video_tokens"], arrays["reference_video_tokens"])
    for identity_change in (
        {"teacher_checkpoint_sha256": "other"},
        {"tokenizer_sha256": "other"},
        {"mixture_sampler_sha256": "other"},
    ):
        with pytest.raises(ReferenceCacheError, match="identity"):
            validate_reference_cache(
                destination, expected_identity=_identity(**identity_change)
            )
    with pytest.raises(ReferenceCacheError, match="overwrite"):
        write_reference_cache(
            destination,
            identity=_identity(),
            arrays=arrays,
            video_index=[{}],
            positive_span_index=[{}],
        )


def test_online_reference_encodings_equal_float32_cached_values(tmp_path: Path) -> None:
    reference = freeze_reference(FakeReference())
    video_mask = torch.ones(1, 65, dtype=torch.bool)
    video_mask[:, 1:3] = False
    online_video, online_valid = encode_reference_video(
        reference,
        {
            "video_features": torch.zeros(1, 1024, 64, 1),
            "video_ignore_raw": video_mask,
        },
    )
    input_ids = torch.tensor([[0, 1, 3, 9, 0]])
    text_valid = torch.tensor([[True, True, True, True, False]])
    text_tokens, _ = encode_reference_text(
        reference, input_ids, torch.zeros_like(input_ids), text_valid
    )
    online_span = pool_reference_spans(reference, text_tokens, [(1, 2)])
    cache_root = tmp_path / "online_equivalence"
    write_reference_cache(
        cache_root,
        identity=_identity(),
        arrays={
            "reference_video_tokens": online_video.numpy(),
            "reference_video_valid": online_valid.numpy(),
            "reference_positive_spans": online_span.numpy(),
        },
        video_index=[{"row": 0, "video_uid": "video-1"}],
        positive_span_index=[{"row": 0, "occurrence_uid": "text-1:0:4"}],
    )
    cached = validate_reference_cache(cache_root, expected_identity=_identity())
    np.testing.assert_array_equal(cached["reference_video_tokens"], online_video.numpy())
    np.testing.assert_array_equal(cached["reference_video_valid"], online_valid.numpy())
    np.testing.assert_array_equal(cached["reference_positive_spans"], online_span.numpy())


def test_negative_span_cache_is_bound_to_reference_and_miner(tmp_path: Path) -> None:
    vectors = np.array([[0.6, 0.8], [0.0, 1.0]], dtype=np.float32)
    metadata = write_negative_span_cache(
        tmp_path,
        reference_identity_sha256="reference",
        mining_content_sha256="miner",
        vectors=vectors,
        edit_uids=["e1", "e2"],
    )
    loaded, index = validate_negative_span_cache(
        tmp_path,
        reference_identity_sha256="reference",
        mining_content_sha256="miner",
    )
    assert metadata["count"] == 2
    assert index == {"e1": 0, "e2": 1}
    np.testing.assert_array_equal(loaded, vectors)
    with pytest.raises(ReferenceCacheError, match="identity"):
        validate_negative_span_cache(
            tmp_path,
            reference_identity_sha256="reference",
            mining_content_sha256="other",
        )
