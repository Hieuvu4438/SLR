from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .utils import stable_seed


def legacy_feature_indices(num_rows: int, feature_len: int = 64) -> np.ndarray:
    if num_rows < 1 or feature_len < 1:
        raise ValueError("num_rows and feature_len must be positive")
    if num_rows >= feature_len:
        return np.linspace(0, num_rows - 1, feature_len, dtype=np.int64)
    output = np.full(feature_len, -1, dtype=np.int64)
    output[:num_rows] = np.arange(num_rows, dtype=np.int64)
    return output


def mix_and_sample_features(
    agnostic: np.ndarray,
    aware: np.ndarray,
    *,
    agnostic_weight: float,
    feature_len: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    agnostic = np.asarray(agnostic)
    aware = np.asarray(aware)
    if agnostic.shape != aware.shape or agnostic.ndim != 2:
        raise ValueError(
            f"feature streams must have the same [N,D] shape, got {agnostic.shape} and {aware.shape}"
        )
    if not 0.0 <= agnostic_weight <= 1.0:
        raise ValueError("agnostic_weight must be in [0,1]")
    if not np.isfinite(agnostic).all() or not np.isfinite(aware).all():
        raise ValueError("feature streams contain NaN or infinity")
    indices = legacy_feature_indices(len(agnostic), feature_len)
    output = np.zeros((feature_len, agnostic.shape[1]), dtype=np.float32)
    valid = indices >= 0
    selected = indices[valid]
    output[valid] = (
        agnostic_weight * agnostic[selected].astype(np.float32, copy=False)
        + (1.0 - agnostic_weight) * aware[selected].astype(np.float32, copy=False)
    )
    return output, valid, indices


def choose_group_member(
    video_uids: Sequence[str], *, seed: int, epoch: int, group_uid: str
) -> str:
    if not video_uids:
        raise ValueError("group has no videos")
    generator = np.random.default_rng(stable_seed(seed, epoch, group_uid, "member"))
    return video_uids[int(generator.integers(0, len(video_uids)))]


def choose_epoch_edits(
    edit_uids: Sequence[str], *, k: int, seed: int, epoch: int, text_uid: str
) -> tuple[tuple[str | None, ...], np.ndarray]:
    if k < 1:
        raise ValueError("k must be positive")
    unique = tuple(dict.fromkeys(edit_uids))
    if len(unique) != len(edit_uids):
        raise ValueError("cached edit IDs must be distinct")
    generator = np.random.default_rng(stable_seed(seed, epoch, text_uid, "edit_sampling_v1"))
    if unique:
        count = min(k, len(unique))
        selected_indexes = generator.choice(len(unique), size=count, replace=False)
        selected = [unique[int(index)] for index in selected_indexes]
    else:
        selected = []
    valid = np.zeros(k, dtype=np.bool_)
    valid[: len(selected)] = True
    return tuple([*selected, *([None] * (k - len(selected)))]), valid


def cyclically_shift_support(
    support: torch.Tensor,
    video_valid: torch.Tensor,
    *,
    seed: int,
    epoch: int,
    video_uids: Sequence[str],
    edit_uids: Sequence[Sequence[Sequence[str | None]]],
) -> torch.Tensor:
    """Shift every edit's support using its persistent video/edit identity.

    A one-valid-clip row is deliberately unchanged: it is an uninformative random-support
    diagnostic but preserving it keeps the support mass contract exact.
    """
    if support.ndim != 4 or video_valid.shape != (support.shape[0], support.shape[-1]):
        raise ValueError("support must be [B,K,E,L] and video_valid [B,L]")
    batch, negatives, edits, _ = support.shape
    if len(video_uids) != batch or len(edit_uids) != batch:
        raise ValueError("video_uids/edit_uids must align with the support batch")
    output = support.clone()
    output.masked_fill_(~video_valid[:, None, None, :], 0.0)
    for batch_index in range(support.shape[0]):
        if len(edit_uids[batch_index]) != negatives or any(
            len(row) != edits for row in edit_uids[batch_index]
        ):
            raise ValueError("edit_uids must have shape [B,K,E]")
        indexes = torch.nonzero(video_valid[batch_index], as_tuple=False).flatten()
        n_valid = int(indexes.numel())
        if n_valid < 2:
            continue
        for negative_index in range(negatives):
            for edit_index in range(edits):
                edit_uid = edit_uids[batch_index][negative_index][edit_index]
                if edit_uid is None:
                    continue
                shift = 1 + stable_seed(
                    seed,
                    epoch,
                    video_uids[batch_index],
                    edit_uid,
                    "random_support",
                ) % (n_valid - 1)
                output[batch_index, negative_index, edit_index, indexes] = support[
                    batch_index, negative_index, edit_index, indexes
                ].roll(shifts=int(shift), dims=-1)
    return output
