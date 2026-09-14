from __future__ import annotations

import os
import socket
from multiprocessing import get_context

import pytest
import torch
import torch.distributed as dist
from torch import nn

from pmgr.distributed import (
    all_gather_variable_items,
    complete_group_partitions,
    coordinated_error,
    distributed_two_level_backward,
    manual_sum_gradients,
)
from pmgr.losses import objective_loss
from pmgr.scoring import mixed_pair_scores


def test_complete_groups_partition_with_uneven_video_counts():
    partitions = complete_group_partitions([2, 1, 3, 4], 2)
    assert [(item.group_start, item.group_end) for item in partitions] == [(0, 2), (2, 4)]
    assert [(item.video_start, item.video_end) for item in partitions] == [(0, 3), (3, 10)]


def _free_port():
    with socket.socket() as value:
        value.bind(("127.0.0.1", 0))
        return value.getsockname()[1]


def _worker(rank, world_size, port, queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        local = torch.full((rank + 1, 2), float(rank + 1))
        gathered, counts = all_gather_variable_items(local)
        model = nn.Linear(2, 1, bias=True)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(0.0)
        model.weight.grad = torch.full_like(model.weight, float(rank + 1))
        model.bias.grad = None
        active = manual_sum_gradients(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()
        queue.put(
            (
                rank,
                gathered.tolist(),
                counts,
                active,
                model.weight.detach().tolist(),
                model.bias.detach().tolist(),
                model.bias.grad is None,
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_two_worker_variable_gather_sum_update_and_unused_parameter():
    context = get_context("spawn")
    queue = context.Queue()
    port = _free_port()
    processes = [context.Process(target=_worker, args=(rank, 2, port, queue)) for rank in range(2)]
    for process in processes:
        process.start()
    results = [queue.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    for _, gathered, counts, active, weight, bias, bias_unused in results:
        assert counts == (1, 2)
        assert gathered == [[1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]
        assert active == ("weight",)
        assert weight[0] == pytest.approx([0.7, 0.7])
        assert bias == [0.0]
        assert bias_unused


def _error_worker(rank, world_size, port, queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        local_error = RuntimeError("broken local feature") if rank == 1 else None
        try:
            coordinated_error(local_error, torch.device("cpu"))
        except RuntimeError as error:
            queue.put((rank, str(error)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_one_rank_input_failure_is_reported_by_every_worker_without_deadlock():
    context = get_context("spawn")
    queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(target=_error_worker, args=(rank, 2, port, queue)) for rank in range(2)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    assert {rank for rank, _ in results} == {0, 1}
    assert all("coordinated PMGR batch failure" in message for _, message in results)


class _DistributedTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2, bias=False, dtype=torch.float64)
        self.logit_scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        with torch.no_grad():
            self.projection.weight.copy_(torch.tensor([[0.4, -0.2, 0.3], [0.1, 0.5, -0.4]]))

    def encode_video(self, value, mask):
        return self.projection(value), mask.bool()

    def encode_text(self, value, segments, mask):
        del segments
        return self.projection(value), mask.bool()


def _global_tiny_batch():
    generator = torch.Generator().manual_seed(19)
    return {
        "video_features": torch.randn(8, 3, 3, generator=generator, dtype=torch.float64),
        "video_padding_mask": torch.ones(8, 3, dtype=torch.long),
        "input_ids": torch.randn(4, 4, 3, generator=generator, dtype=torch.float64),
        "segment_ids": torch.zeros(4, 4),
        "input_mask": torch.ones(4, 4, dtype=torch.long),
        "aug_input_ids": torch.randn(4, 5, 3, generator=generator, dtype=torch.float64),
        "aug_segment_ids": torch.zeros(4, 5),
        "aug_input_mask": torch.ones(4, 5, dtype=torch.long),
        "video_to_group": torch.tensor([0, 0, 1, 2, 2, 2, 3, 3]),
        "dataset_group_count": 10,
        "dataset_video_count": 30,
        "group_ids": ["g0", "g1", "g2", "g3"],
        "video_ids": [f"v{index}" for index in range(8)],
        "full_group_sizes": torch.tensor([2, 1, 3, 2]),
        "selected_group_sizes": torch.tensor([2, 1, 3, 2]),
    }


def _distributed_config():
    return {
        "training": {"effective_groups": 4},
        "engine": {"video_encoder_microbatch": 2, "text_encoder_microbatch": 1,
                   "score_video_block": 3, "score_text_block": 2},
        "scoring": {"dual_mix": 0.5, "inner_temperature": 0.07,
                    "normalize_eps": 1e-6, "mask_policy": "valid_tokens_only"},
        "loss": {"mode": "pmgr", "rank_mix": 0.25, "rank_eta": 0.1},
    }


def _local_tiny_batch(rank):
    global_batch = _global_tiny_batch()
    group_slice = slice(rank * 2, (rank + 1) * 2)
    video_slice = slice(0, 3) if rank == 0 else slice(3, 8)
    group_offset = rank * 2
    output = {
        key: value[video_slice]
        for key, value in global_batch.items()
        if key in {"video_features", "video_padding_mask"}
    }
    for key in ("input_ids", "segment_ids", "input_mask", "aug_input_ids", "aug_segment_ids", "aug_input_mask"):
        output[key] = global_batch[key][group_slice]
    output["video_to_group"] = global_batch["video_to_group"][video_slice] - group_offset
    output["dataset_group_count"] = 10
    output["dataset_video_count"] = 30
    output["group_ids"] = global_batch["group_ids"][group_slice]
    output["video_ids"] = global_batch["video_ids"][video_slice]
    output["full_group_sizes"] = global_batch["full_group_sizes"][group_slice]
    output["selected_group_sizes"] = global_batch["selected_group_sizes"][group_slice]
    return output


def _replay_worker(rank, world_size, port, queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        model = _DistributedTiny()
        result = distributed_two_level_backward(
            model, _local_tiny_batch(rank), _distributed_config()
        )
        gradients = {name: parameter.grad.tolist() for name, parameter in model.named_parameters()}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        optimizer.step()
        parameters = {name: parameter.detach().tolist() for name, parameter in model.named_parameters()}
        queue.put((rank, float(result.loss), gradients, parameters,
                   result.global_video_count, result.global_group_count))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_distributed_replay_matches_one_worker_global_loss_gradients_and_update():
    direct = _DistributedTiny()
    batch = _global_tiny_batch()
    video, video_valid = direct.encode_video(batch["video_features"], batch["video_padding_mask"])
    text, text_valid = direct.encode_text(batch["input_ids"], batch["segment_ids"], batch["input_mask"])
    aug, aug_valid = direct.encode_text(
        batch["aug_input_ids"], batch["aug_segment_ids"], batch["aug_input_mask"]
    )
    q, a, b = mixed_pair_scores(video, text, aug, video_valid, text_valid, aug_valid)
    expected_loss = objective_loss(
        "pmgr", q, batch["video_to_group"], direct.logit_scale, 10, 30,
        a=a, b=b, rank_mix=0.25, rank_eta=0.1,
    )["loss"]
    expected_loss.backward()
    expected_gradients = {name: parameter.grad.detach() for name, parameter in direct.named_parameters()}
    optimizer = torch.optim.SGD(direct.parameters(), lr=0.02)
    optimizer.step()
    expected_parameters = {name: parameter.detach() for name, parameter in direct.named_parameters()}

    context = get_context("spawn")
    queue = context.Queue()
    port = _free_port()
    processes = [context.Process(target=_replay_worker, args=(rank, 2, port, queue)) for rank in range(2)]
    for process in processes:
        process.start()
    results = [queue.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    for _, loss, gradients, parameters, videos, groups in results:
        assert loss == pytest.approx(float(expected_loss.detach()), abs=1e-12)
        assert (videos, groups) == (8, 4)
        for name in expected_gradients:
            torch.testing.assert_close(
                torch.tensor(gradients[name], dtype=expected_gradients[name].dtype),
                expected_gradients[name], atol=1e-11, rtol=1e-11,
            )
            torch.testing.assert_close(
                torch.tensor(parameters[name], dtype=expected_parameters[name].dtype),
                expected_parameters[name], atol=1e-12, rtol=1e-12,
            )
