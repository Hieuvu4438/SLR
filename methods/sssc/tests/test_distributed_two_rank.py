from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from method1.distributed import (
    DistributedRuntime,
    ddp_weighted_auxiliary,
    gather_with_grad,
)
from method1.schemas import AuxiliaryTerms


class GatherProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return gather_with_grad(self.weight * value)


class AuxiliaryProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, coefficient: torch.Tensor) -> torch.Tensor:
        return coefficient * self.weight.square()


def _two_rank_worker(rank: int, init_file: str, output_dir: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        gather_model = DistributedDataParallel(GatherProbe())
        gathered = gather_model(torch.tensor([float(rank + 1)]))
        gathered.square().sum().backward()
        gather_gradient = float(gather_model.module.weight.grad)

        auxiliary_model = DistributedDataParallel(AuxiliaryProbe())
        coefficient = torch.tensor(float(1 if rank == 0 else 3))
        numerator = auxiliary_model(coefficient)
        terms = AuxiliaryTerms(
            numerator=numerator,
            denominator=coefficient.detach(),
            diagnostics={},
        )
        backward, logs = ddp_weighted_auxiliary(
            terms, DistributedRuntime(rank=rank, world_size=2)
        )
        backward.backward()
        payload = {
            "gather_gradient": gather_gradient,
            "auxiliary_gradient": float(auxiliary_model.module.weight.grad),
            "auxiliary_log": float(logs["auxiliary"]),
            "auxiliary_count": float(logs["auxiliary_weight_count"]),
        }
        torch.save(payload, Path(output_dir) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_rank_gather_and_weighted_auxiliary_match_global_reference(
    tmp_path: Path,
) -> None:
    init_file = tmp_path / "distributed-init"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    mp.spawn(
        _two_rank_worker,
        args=(str(init_file), str(output_dir)),
        nprocs=2,
        join=True,
    )
    results = [
        torch.load(output_dir / f"rank{rank}.pt", weights_only=True) for rank in range(2)
    ]
    for result in results:
        # d/dw sum_i (w*x_i)^2 at w=2, x=[1,2].
        assert result["gather_gradient"] == pytest.approx(20.0)
        # (1*w^2 + 3*w^2)/(1+3) has derivative 2*w = 4.
        assert result["auxiliary_gradient"] == pytest.approx(4.0)
        assert result["auxiliary_log"] == pytest.approx(4.0)
        assert result["auxiliary_count"] == pytest.approx(4.0)
