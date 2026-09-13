from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .schemas import AuxiliaryTerms


class GatherWithSummedBackward(torch.autograd.Function):
    """Equal-shape all-gather whose backward sums global contributions before slicing."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized():
            ctx.world_size = 1
            ctx.rank = 0
            ctx.local_batch = tensor.shape[0]
            return tensor
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        outputs = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(outputs, tensor)
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.local_batch = tensor.shape[0]
        return torch.cat(outputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        if ctx.world_size == 1:
            return (grad_output,)
        full_gradient = grad_output.contiguous()
        dist.all_reduce(full_gradient, op=dist.ReduceOp.SUM)
        start = ctx.rank * ctx.local_batch
        return (full_gradient.narrow(0, start, ctx.local_batch),)


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    return GatherWithSummedBackward.apply(tensor)


@dataclass(frozen=True)
class DistributedRuntime:
    rank: int = 0
    world_size: int = 1

    @classmethod
    def current(cls) -> "DistributedRuntime":
        if dist.is_available() and dist.is_initialized():
            return cls(rank=dist.get_rank(), world_size=dist.get_world_size())
        return cls()


def ddp_weighted_auxiliary(
    terms: AuxiliaryTerms,
    runtime: DistributedRuntime | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    runtime = runtime or DistributedRuntime.current()
    global_count = terms.denominator.detach().clone()
    global_numerator = terms.numerator.detach().clone()
    if runtime.world_size > 1:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("world_size > 1 but torch.distributed is not initialized")
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_numerator, op=dist.ReduceOp.SUM)
    divisor = global_count.clamp_min(1.0)
    backward_loss = runtime.world_size * terms.numerator / divisor
    logged_loss = global_numerator / divisor
    return backward_loss, {
        "auxiliary": logged_loss,
        "auxiliary_numerator": global_numerator,
        "auxiliary_weight_count": global_count,
    }
