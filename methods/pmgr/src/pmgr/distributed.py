from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist
from torch import Tensor, nn

from pmgr.replay import (
    EncoderCache,
    ScoreReplayResult,
    cache_encoder_outputs,
    replay_encoder_gradients,
    score_space_replay,
)
from pmgr.scoring import EncodedPMGRBatch


class DistributedContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class Partition:
    group_start: int
    group_end: int
    video_start: int
    video_end: int


def complete_group_partitions(group_sizes: Iterable[int], world_size: int) -> tuple[Partition, ...]:
    sizes = [int(value) for value in group_sizes]
    if world_size < 1 or len(sizes) < world_size or len(sizes) % world_size:
        raise DistributedContractError("global group batch must be divisible by worker count")
    if any(value < 1 for value in sizes):
        raise DistributedContractError("every distributed group must remain complete")
    groups_per_rank = len(sizes) // world_size
    video_prefix = [0]
    for value in sizes:
        video_prefix.append(video_prefix[-1] + value)
    return tuple(
        Partition(
            group_start=rank * groups_per_rank,
            group_end=(rank + 1) * groups_per_rank,
            video_start=video_prefix[rank * groups_per_rank],
            video_end=video_prefix[(rank + 1) * groups_per_rank],
        )
        for rank in range(world_size)
    )


def pad_items(value: Tensor, item_count: int) -> Tensor:
    if item_count < value.shape[0]:
        raise DistributedContractError("transport padding cannot truncate items")
    if item_count == value.shape[0]:
        return value
    output = value.new_zeros((item_count, *value.shape[1:]))
    output[: value.shape[0]] = value
    return output


def all_gather_variable_items(value: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    """Gather a variable leading dimension without promoting transport padding to data."""
    if not dist.is_initialized():
        return value, (value.shape[0],)
    world_size = dist.get_world_size()
    count = torch.tensor([value.shape[0]], dtype=torch.long, device=value.device)
    gathered_counts = [torch.zeros_like(count) for _ in range(world_size)]
    dist.all_gather(gathered_counts, count)
    counts = tuple(int(item.item()) for item in gathered_counts)
    maximum = max(counts)
    padded = pad_items(value.contiguous(), maximum)
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return torch.cat([item[:count] for item, count in zip(gathered, counts, strict=True)]), counts


def manual_sum_gradients(model: nn.Module) -> tuple[str, ...]:
    """SUM global-loss contributions exactly once while preserving globally unused grads."""
    if isinstance(model, nn.parallel.DistributedDataParallel):
        raise DistributedContractError("manual PMGR reduction cannot be combined with DDP hooks")
    if not dist.is_initialized():
        return tuple(name for name, parameter in model.named_parameters() if parameter.grad is not None)
    active_names: list[str] = []
    for name, parameter in model.named_parameters():
        local_active = torch.tensor(
            [parameter.grad is not None], dtype=torch.long, device=parameter.device
        )
        dist.all_reduce(local_active, op=dist.ReduceOp.SUM)
        if int(local_active.item()) == 0:
            parameter.grad = None
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        active_names.append(name)
    return tuple(active_names)


def coordinated_error(local_error: BaseException | None, device: torch.device) -> None:
    """Make every worker fail before later collectives if any worker rejected its batch."""
    if not dist.is_initialized():
        if local_error is not None:
            raise local_error
        return
    flag = torch.tensor([local_error is not None], dtype=torch.long, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    if int(flag.item()):
        detail = f": {local_error}" if local_error is not None else " on another worker"
        raise DistributedContractError("coordinated PMGR batch failure" + detail)


@dataclass(frozen=True)
class DistributedReplayResult:
    loss: Tensor
    diagnostics: dict[str, Tensor]
    block_replays: int
    encoder_forward_calls: int
    global_video_count: int
    global_group_count: int
    group_ids: tuple[str, ...]
    video_ids: tuple[str, ...]
    full_group_sizes: tuple[int, ...]
    selected_group_sizes: tuple[int, ...]


def distributed_two_level_backward(
    model: nn.Module,
    local_batch: dict,
    config: dict,
) -> DistributedReplayResult:
    """Central score replay plus explicit SUM of local encoder-gradient contributions."""
    if not dist.is_initialized() or dist.get_world_size() < 2:
        raise DistributedContractError("manual_sum_replay requires an initialized multi-worker group")
    if isinstance(model, nn.parallel.DistributedDataParallel):
        raise DistributedContractError("manual_sum_replay uses an unwrapped replicated model")
    rank = dist.get_rank()
    device = model.logit_scale.device
    local_error: BaseException | None = None
    cache: EncoderCache | None = None
    try:
        cache = cache_encoder_outputs(
            model,
            local_batch,
            video_microbatch=int(config["engine"]["video_encoder_microbatch"]),
            text_microbatch=int(config["engine"]["text_encoder_microbatch"]),
        )
    except BaseException as error:  # coordinated failure must precede later collectives
        local_error = error
    coordinated_error(local_error, device)
    assert cache is not None

    global_video, video_counts = all_gather_variable_items(cache.encoded.video_hidden)
    global_video_valid, valid_video_counts = all_gather_variable_items(cache.encoded.video_valid)
    global_text, text_counts = all_gather_variable_items(cache.encoded.text_hidden)
    global_text_valid, valid_text_counts = all_gather_variable_items(cache.encoded.text_valid)
    global_aug, aug_counts = all_gather_variable_items(cache.encoded.aug_hidden)
    global_aug_valid, valid_aug_counts = all_gather_variable_items(cache.encoded.aug_valid)
    if video_counts != valid_video_counts or text_counts != valid_text_counts:
        raise DistributedContractError("hidden and mask transport counts differ")
    if text_counts != aug_counts or text_counts != valid_aug_counts:
        raise DistributedContractError("original and augmented text transport counts differ")
    group_offset = sum(text_counts[:rank])
    global_owner, owner_counts = all_gather_variable_items(
        local_batch["video_to_group"] + group_offset
    )
    if owner_counts != video_counts:
        raise DistributedContractError("video ownership transport counts differ")
    global_group_count = sum(text_counts)
    if global_group_count != int(config["training"]["effective_groups"]):
        raise DistributedContractError("transported group count differs from configured global batch")

    gathered_group_ids: list[list[str] | None] = [None] * dist.get_world_size()
    gathered_video_ids: list[list[str] | None] = [None] * dist.get_world_size()
    gathered_full_sizes: list[list[int] | None] = [None] * dist.get_world_size()
    gathered_selected_sizes: list[list[int] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_group_ids, list(local_batch["group_ids"]))
    dist.all_gather_object(gathered_video_ids, list(local_batch["video_ids"]))
    dist.all_gather_object(gathered_full_sizes, local_batch["full_group_sizes"].cpu().tolist())
    dist.all_gather_object(
        gathered_selected_sizes, local_batch["selected_group_sizes"].cpu().tolist()
    )
    group_ids = tuple(value for values in gathered_group_ids for value in (values or []))
    video_ids = tuple(value for values in gathered_video_ids for value in (values or []))
    full_group_sizes = tuple(value for values in gathered_full_sizes for value in (values or []))
    selected_group_sizes = tuple(
        value for values in gathered_selected_sizes for value in (values or [])
    )
    identity_error = None
    if len(group_ids) != len(set(group_ids)) or len(video_ids) != len(set(video_ids)):
        identity_error = DistributedContractError("distributed batch contains duplicate IDs")
    coordinated_error(identity_error, device)

    global_encoded = EncodedPMGRBatch(
        global_video,
        global_video_valid,
        global_text,
        global_text_valid,
        global_aug,
        global_aug_valid,
    )
    score_result: ScoreReplayResult | None = None
    score_error: BaseException | None = None
    if rank == 0:
        try:
            scoring = config["scoring"]
            loss = config["loss"]
            score_result = score_space_replay(
                global_encoded,
                global_owner,
                model.logit_scale,
                local_batch["dataset_group_count"],
                local_batch["dataset_video_count"],
                loss_mode=loss["mode"],
                omega=float(scoring["dual_mix"]),
                sigma=float(scoring["inner_temperature"]),
                normalize_eps=float(scoring["normalize_eps"]),
                mask_policy=scoring["mask_policy"],
                rank_mix=float(loss["rank_mix"]),
                rank_eta=float(loss["rank_eta"]),
                video_block=int(config["engine"]["score_video_block"]),
                text_block=int(config["engine"]["score_text_block"]),
            )
        except BaseException as error:
            score_error = error
    coordinated_error(score_error, device)

    scalar_names = ("loss", "ce_t", "ce_v", "rank_t", "rank_v")
    scalar_values = torch.zeros(len(scalar_names) + 1, dtype=torch.float64, device=device)
    if rank == 0:
        assert score_result is not None
        scalar_values[:-1] = torch.tensor(
            [float(score_result.diagnostics[name]) for name in scalar_names],
            dtype=torch.float64,
            device=device,
        )
        scalar_values[-1] = score_result.block_replays
    dist.broadcast(scalar_values, src=0)

    global_derivatives = [
        torch.zeros_like(global_video),
        torch.zeros_like(global_text),
        torch.zeros_like(global_aug),
    ]
    if rank == 0:
        global_derivatives = [
            score_result.video_gradient,
            score_result.text_gradient,
            score_result.aug_gradient,
        ]
    for value in global_derivatives:
        dist.broadcast(value, src=0)
    video_start = sum(video_counts[:rank])
    video_end = video_start + video_counts[rank]
    text_start = sum(text_counts[:rank])
    text_end = text_start + text_counts[rank]
    local_result = ScoreReplayResult(
        loss=scalar_values[0].to(model.logit_scale.dtype),
        diagnostics={},
        q=torch.empty(0, device=device),
        a=torch.empty(0, device=device),
        b=torch.empty(0, device=device),
        video_gradient=global_derivatives[0][video_start:video_end],
        text_gradient=global_derivatives[1][text_start:text_end],
        aug_gradient=global_derivatives[2][text_start:text_end],
        logit_scale_gradient=(
            score_result.logit_scale_gradient if rank == 0 and score_result is not None else None
        ),
        block_replays=int(scalar_values[-1].item()),
    )
    replay_calls = replay_encoder_gradients(model, cache, local_result)
    manual_sum_gradients(model)
    diagnostics = {
        name: scalar_values[index].to(model.logit_scale.dtype)
        for index, name in enumerate(scalar_names)
    }
    return DistributedReplayResult(
        loss=diagnostics["loss"],
        diagnostics=diagnostics,
        block_replays=int(scalar_values[-1].item()),
        encoder_forward_calls=2 * replay_calls,
        global_video_count=len(video_ids),
        global_group_count=len(group_ids),
        group_ids=group_ids,
        video_ids=video_ids,
        full_group_sizes=full_group_sizes,
        selected_group_sizes=selected_group_sizes,
    )
