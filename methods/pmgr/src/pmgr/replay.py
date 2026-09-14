from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import Tensor

from pmgr.losses import objective_loss
from pmgr.scoring import EncodedPMGRBatch, mixed_pair_scores, mixed_pair_scores_blocked


@dataclass(frozen=True)
class ScoreReplayResult:
    loss: Tensor
    diagnostics: dict[str, Tensor]
    q: Tensor
    a: Tensor
    b: Tensor
    video_gradient: Tensor
    text_gradient: Tensor
    aug_gradient: Tensor
    logit_scale_gradient: Tensor | None
    block_replays: int


@dataclass(frozen=True)
class EncoderReplayRecord:
    branch: str
    item_slice: slice
    inputs: tuple[Tensor, ...]
    cpu_rng_state: Tensor
    cuda_rng_state: list[Tensor] | None


@dataclass(frozen=True)
class EncoderCache:
    encoded: EncodedPMGRBatch
    records: tuple[EncoderReplayRecord, ...]


def _slices(length: int, size: int) -> Iterator[slice]:
    if size < 1:
        raise ValueError("replay chunk size must be positive")
    for start in range(0, length, size):
        yield slice(start, min(start + size, length))


def _objective(
    mode: str,
    q: Tensor,
    a: Tensor,
    b: Tensor,
    video_to_group: Tensor,
    logit_scale: Tensor,
    dataset_group_count: int,
    dataset_video_count: int,
    *,
    omega: float,
    rank_mix: float,
    rank_eta: float,
) -> dict[str, Tensor]:
    return objective_loss(
        mode,
        q,
        video_to_group,
        logit_scale,
        dataset_group_count,
        dataset_video_count,
        a=a,
        b=b,
        omega=omega,
        rank_mix=rank_mix,
        rank_eta=rank_eta,
    )


def score_space_replay(
    encoded: EncodedPMGRBatch,
    video_to_group: Tensor,
    logit_scale: Tensor,
    dataset_group_count: int,
    dataset_video_count: int,
    *,
    loss_mode: str,
    omega: float = 0.5,
    sigma: float = 0.07,
    normalize_eps: float = 1e-6,
    mask_policy: str = "valid_tokens_only",
    rank_mix: float = 0.0,
    rank_eta: float = 0.03,
    video_block: int,
    text_block: int,
) -> ScoreReplayResult:
    """Differentiate the global objective in scalar-score space, then replay pair blocks."""
    encoded.validate()
    with torch.no_grad():
        cached_q, cached_a, cached_b = mixed_pair_scores_blocked(
            encoded.video_hidden,
            encoded.text_hidden,
            encoded.aug_hidden,
            encoded.video_valid,
            encoded.text_valid,
            encoded.aug_valid,
            video_block=video_block,
            text_block=text_block,
            omega=omega,
            sigma=sigma,
            normalize_eps=normalize_eps,
            mask_policy=mask_policy,
        )
    q_leaf = cached_q.detach().requires_grad_(True)
    a_leaf = cached_a.detach().requires_grad_(True)
    b_leaf = cached_b.detach().requires_grad_(True)
    diagnostics = _objective(
        loss_mode,
        q_leaf,
        a_leaf,
        b_leaf,
        video_to_group,
        logit_scale,
        dataset_group_count,
        dataset_video_count,
        omega=omega,
        rank_mix=rank_mix,
        rank_eta=rank_eta,
    )
    if loss_mode == "legacy_cico":
        derivatives = torch.autograd.grad(
            diagnostics["loss"], (a_leaf, b_leaf, logit_scale), allow_unused=True
        )
        dq, da, db, dlogit = None, derivatives[0], derivatives[1], derivatives[2]
    else:
        derivatives = torch.autograd.grad(
            diagnostics["loss"], (q_leaf, logit_scale), allow_unused=True
        )
        dq, dlogit = derivatives
        da = db = None
    video_gradient = torch.zeros_like(encoded.video_hidden)
    text_gradient = torch.zeros_like(encoded.text_hidden)
    aug_gradient = torch.zeros_like(encoded.aug_hidden)
    blocks = 0
    for row in _slices(encoded.video_hidden.shape[0], video_block):
        for column in _slices(encoded.text_hidden.shape[0], text_block):
            video = encoded.video_hidden[row].detach().requires_grad_(True)
            text = encoded.text_hidden[column].detach().requires_grad_(True)
            augmented = encoded.aug_hidden[column].detach().requires_grad_(True)
            q_block, a_block, b_block = mixed_pair_scores(
                video,
                text,
                augmented,
                encoded.video_valid[row],
                encoded.text_valid[column],
                encoded.aug_valid[column],
                omega=omega,
                sigma=sigma,
                normalize_eps=normalize_eps,
                mask_policy=mask_policy,
            )
            if loss_mode == "legacy_cico":
                surrogate = (a_block * da[row, column].detach()).sum()
                surrogate = surrogate + (b_block * db[row, column].detach()).sum()
            else:
                surrogate = (q_block * dq[row, column].detach()).sum()
            gradients = torch.autograd.grad(surrogate, (video, text, augmented), allow_unused=True)
            for destination, gradient, target_slice in (
                (video_gradient, gradients[0], row),
                (text_gradient, gradients[1], column),
                (aug_gradient, gradients[2], column),
            ):
                if gradient is not None:
                    destination[target_slice].add_(gradient)
            blocks += 1
    detached_diagnostics = {name: value.detach() for name, value in diagnostics.items()}
    return ScoreReplayResult(
        loss=diagnostics["loss"].detach(),
        diagnostics=detached_diagnostics,
        q=cached_q,
        a=cached_a,
        b=cached_b,
        video_gradient=video_gradient,
        text_gradient=text_gradient,
        aug_gradient=aug_gradient,
        logit_scale_gradient=dlogit.detach() if dlogit is not None else None,
        block_replays=blocks,
    )


def _rng_record(branch: str, item_slice: slice, inputs: tuple[Tensor, ...]) -> EncoderReplayRecord:
    return EncoderReplayRecord(
        branch=branch,
        item_slice=item_slice,
        inputs=tuple(value.detach() for value in inputs),
        cpu_rng_state=torch.get_rng_state(),
        cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


@contextmanager
def _replay_rng(record: EncoderReplayRecord):
    surrounding_cpu = torch.get_rng_state()
    surrounding_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.set_rng_state(record.cpu_rng_state)
    if record.cuda_rng_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(record.cuda_rng_state)
    try:
        yield
    finally:
        torch.set_rng_state(surrounding_cpu)
        if surrounding_cuda is not None:
            torch.cuda.set_rng_state_all(surrounding_cuda)


def cache_encoder_outputs(
    model: Any,
    batch: dict[str, Any],
    *,
    video_microbatch: int,
    text_microbatch: int,
) -> EncoderCache:
    """Encode one effective batch without graph while recording exact replay RNG/input state."""
    videos: list[Tensor] = []
    video_valid: list[Tensor] = []
    texts: list[Tensor] = []
    text_valid: list[Tensor] = []
    augmented: list[Tensor] = []
    augmented_valid: list[Tensor] = []
    records: list[EncoderReplayRecord] = []
    with torch.no_grad():
        for item_slice in _slices(len(batch["video_features"]), video_microbatch):
            inputs = (batch["video_features"][item_slice], batch["video_padding_mask"][item_slice])
            records.append(_rng_record("video", item_slice, inputs))
            hidden, valid = model.encode_video(*inputs)
            videos.append(hidden)
            video_valid.append(valid)
        for branch, names, hidden_values, valid_values in (
            ("text", ("input_ids", "segment_ids", "input_mask"), texts, text_valid),
            (
                "aug",
                ("aug_input_ids", "aug_segment_ids", "aug_input_mask"),
                augmented,
                augmented_valid,
            ),
        ):
            for item_slice in _slices(len(batch[names[0]]), text_microbatch):
                inputs = tuple(batch[name][item_slice] for name in names)
                records.append(_rng_record(branch, item_slice, inputs))
                hidden, valid = model.encode_text(*inputs)
                hidden_values.append(hidden)
                valid_values.append(valid)
    encoded = EncodedPMGRBatch(
        video_hidden=torch.cat(videos),
        video_valid=torch.cat(video_valid),
        text_hidden=torch.cat(texts),
        text_valid=torch.cat(text_valid),
        aug_hidden=torch.cat(augmented),
        aug_valid=torch.cat(augmented_valid),
    )
    encoded.validate()
    return EncoderCache(encoded=encoded, records=tuple(records))


def replay_encoder_gradients(
    model: Any,
    cache: EncoderCache,
    score_result: ScoreReplayResult,
) -> int:
    """Replay both shared text branches and all video chunks into model parameters."""
    branch_gradients = {
        "video": score_result.video_gradient,
        "text": score_result.text_gradient,
        "aug": score_result.aug_gradient,
    }
    calls = 0
    for record in cache.records:
        with _replay_rng(record):
            if record.branch == "video":
                hidden, valid = model.encode_video(*record.inputs)
                expected_valid = cache.encoded.video_valid[record.item_slice]
            else:
                hidden, valid = model.encode_text(*record.inputs)
                expected_valid = (
                    cache.encoded.text_valid[record.item_slice]
                    if record.branch == "text"
                    else cache.encoded.aug_valid[record.item_slice]
                )
        if not torch.equal(valid, expected_valid):
            raise RuntimeError(f"{record.branch} replay changed its validity mask")
        derivative = branch_gradients[record.branch][record.item_slice].detach()
        (hidden * derivative).sum().backward()
        calls += 1
    if score_result.logit_scale_gradient is not None:
        gradient = score_result.logit_scale_gradient.to(model.logit_scale)
        if model.logit_scale.grad is None:
            model.logit_scale.grad = gradient.clone()
        else:
            model.logit_scale.grad.add_(gradient)
    return calls


def two_level_backward(
    model: Any,
    batch: dict[str, Any],
    config: dict[str, Any],
) -> ScoreReplayResult:
    """Populate exact first-order model gradients for one unchanged effective batch."""
    engine = config["engine"]
    scoring = config["scoring"]
    loss = config["loss"]
    cache = cache_encoder_outputs(
        model,
        batch,
        video_microbatch=int(engine["video_encoder_microbatch"]),
        text_microbatch=int(engine["text_encoder_microbatch"]),
    )
    result = score_space_replay(
        cache.encoded,
        batch["video_to_group"],
        model.logit_scale,
        batch["dataset_group_count"],
        batch["dataset_video_count"],
        loss_mode=loss["mode"],
        omega=float(scoring["dual_mix"]),
        sigma=float(scoring["inner_temperature"]),
        normalize_eps=float(scoring["normalize_eps"]),
        mask_policy=scoring["mask_policy"],
        rank_mix=float(loss["rank_mix"]),
        rank_eta=float(loss["rank_eta"]),
        video_block=int(engine["score_video_block"]),
        text_block=int(engine["score_text_block"]),
    )
    replay_encoder_gradients(model, cache, result)
    return result
