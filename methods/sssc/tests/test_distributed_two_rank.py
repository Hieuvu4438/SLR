from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from method1.baseline import baseline_loss_from_local
from method1.distributed import (
    DistributedRuntime,
    ddp_weighted_auxiliary,
    gather_with_grad,
)
from method1.schemas import AuxiliaryTerms
from method1.schemas import LocalEncoding


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


class DistributionProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mu_scale = nn.Parameter(torch.tensor(0.8))
        self.log_sigma = nn.Parameter(torch.tensor(-0.7))

    def forward(self, tokens, mask=None, weight=None):
        return (
            self.mu_scale * tokens,
            self.log_sigma.expand_as(tokens),
            tokens,
        )


class FullBaseProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_scale = nn.Parameter(torch.tensor(1.1))
        self.video_weight_fc = nn.Linear(2, 1)
        self.text_weight_fc = nn.Linear(2, 1)
        self.dist_text_trans = DistributionProbe()
        self.dist_video_trans = DistributionProbe()
        self.clip = nn.Module()
        self.clip.logit_scale = nn.Parameter(torch.tensor(0.2))
        self.eps = 0.1
        self.max_iter = 20
        self.ot_weight = 1.0
        self.dual_mix = 0.5
        self.mix_design = "balance"

    def forward(self, local_video: torch.Tensor, local_text: torch.Tensor) -> torch.Tensor:
        batch = local_video.shape[0]
        encoding = LocalEncoding(
            video_raw=self.encoder_scale * local_video,
            video_ignore_raw=torch.tensor(
                [[True, False, False]], device=local_video.device
            ).expand(batch, -1),
            text_raw=self.encoder_scale * local_text,
            text_valid=torch.ones(batch, 2, dtype=torch.bool, device=local_text.device),
            text_aug_raw=self.encoder_scale * (local_text + 0.05),
            text_aug_valid=torch.ones(batch, 2, dtype=torch.bool, device=local_text.device),
        )
        return baseline_loss_from_local(
            self,
            encoding,
            runtime=DistributedRuntime.current(),
            seed=42,
            optimizer_step=7,
        )


def _full_probe_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [
                [[3.0, 3.0], [1.0, 0.2], [0.1, 1.0]],
                [[2.0, 2.0], [0.7, 0.4], [0.2, 0.9]],
            ]
        ),
        torch.tensor(
            [
                [[1.0, 0.1], [0.3, 0.8]],
                [[0.2, 1.0], [0.9, 0.2]],
            ]
        ),
    )


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
        weighted_cases = {}
        for label, local_count in (
            ("unequal_with_empty_rank", 3.0 if rank == 0 else 0.0),
            ("noninteger", 0.5 if rank == 0 else 1.25),
            ("all_empty", 0.0),
        ):
            auxiliary_model.zero_grad(set_to_none=True)
            count = torch.tensor(local_count)
            numerator = auxiliary_model(count)
            current_backward, current_logs = ddp_weighted_auxiliary(
                AuxiliaryTerms(numerator, count.detach(), {}),
                DistributedRuntime(rank=rank, world_size=2),
            )
            current_backward.backward()
            weighted_cases[label] = {
                "gradient": float(auxiliary_model.module.weight.grad),
                "log": float(current_logs["auxiliary"]),
                "count": float(current_logs["auxiliary_weight_count"]),
            }
        payload["weighted_cases"] = weighted_cases
        torch.manual_seed(123)
        full_model = DistributedDataParallel(FullBaseProbe())
        videos, texts = _full_probe_inputs()
        full_loss = full_model(videos[rank : rank + 1], texts[rank : rank + 1])
        full_loss.backward()
        payload["full_loss"] = float(full_loss.detach())
        payload["full_gradients"] = {
            name: parameter.grad.detach().cpu()
            for name, parameter in full_model.module.named_parameters()
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
    torch.manual_seed(123)
    reference = FullBaseProbe()
    videos, texts = _full_probe_inputs()
    reference_loss = reference(videos, texts)
    reference_loss.backward()
    reference_gradients = {
        name: parameter.grad.detach() for name, parameter in reference.named_parameters()
    }
    for result in results:
        # d/dw sum_i (w*x_i)^2 at w=2, x=[1,2].
        assert result["gather_gradient"] == pytest.approx(20.0)
        # (1*w^2 + 3*w^2)/(1+3) has derivative 2*w = 4.
        assert result["auxiliary_gradient"] == pytest.approx(4.0)
        assert result["auxiliary_log"] == pytest.approx(4.0)
        assert result["auxiliary_count"] == pytest.approx(4.0)
        assert result["weighted_cases"]["unequal_with_empty_rank"] == pytest.approx(
            {"gradient": 4.0, "log": 4.0, "count": 3.0}
        )
        assert result["weighted_cases"]["noninteger"] == pytest.approx(
            {"gradient": 4.0, "log": 4.0, "count": 1.75}
        )
        assert result["weighted_cases"]["all_empty"] == pytest.approx(
            {"gradient": 0.0, "log": 0.0, "count": 0.0}
        )
        assert result["full_loss"] == pytest.approx(float(reference_loss.detach()), rel=1e-6)
        assert set(result["full_gradients"]) == set(reference_gradients)
        for name, gradient in reference_gradients.items():
            torch.testing.assert_close(
                result["full_gradients"][name], gradient, atol=2e-5, rtol=2e-5
            )
