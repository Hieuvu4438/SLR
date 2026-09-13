from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .distributed import DistributedRuntime, gather_with_grad
from .schemas import LocalEncoding, SchemaError
from .utils import stable_seed


def encode_local(student: Any, batch: dict[str, Any]) -> LocalEncoding:
    required = {
        "input_ids",
        "token_type_ids",
        "text_valid",
        "input_ids_aug",
        "text_aug_valid",
        "video_features",
        "video_ignore_raw",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise SchemaError(f"batch is missing local-encoding fields: {', '.join(missing)}")
    result = student.get_text_video_feat(
        batch["input_ids"],
        batch["token_type_ids"],
        batch["text_valid"],
        batch["video_features"],
        batch["video_ignore_raw"],
        shaped=True,
        video_frame=1,
        input_ids_aug=batch["input_ids_aug"],
        attention_mask_aug=batch["text_aug_valid"],
    )
    if not isinstance(result, tuple) or len(result) != 9:
        raise RuntimeError("UPRet get_text_video_feat must return exactly nine values")
    (
        text_raw,
        text_valid,
        _text_cls,
        video_raw,
        video_ignore_raw,
        _video_cls,
        text_aug_raw,
        text_aug_valid,
        _text_aug_cls,
    ) = result
    batch_size = batch["input_ids"].shape[0]
    if video_raw.ndim != 3 or video_raw.shape[:2] != (batch_size, 65):
        raise SchemaError(f"video encoder must return [B,65,D], got {tuple(video_raw.shape)}")
    if video_ignore_raw.shape != (batch_size, 65):
        raise SchemaError("video ignore mask must include the class position")
    if not bool(video_ignore_raw[:, 0].bool().all()):
        raise SchemaError("video class position must be ignored by the baseline scorer")
    if text_raw.ndim != 3 or text_aug_raw.shape != text_raw.shape:
        raise SchemaError("original/augmented text tokens must be aligned [B,M,D]")
    if text_valid.shape != text_raw.shape[:2] or text_aug_valid.shape != text_raw.shape[:2]:
        raise SchemaError("text masks must align with returned token tensors")
    expected_text_valid = batch["text_valid"].bool()
    expected_aug_valid = batch["text_aug_valid"].bool()
    if not torch.equal(text_valid.bool(), expected_text_valid):
        raise SchemaError("encoder original-text mask disagrees with tokenizer mask")
    if not torch.equal(text_aug_valid.bool(), expected_aug_valid):
        raise SchemaError("encoder augmented-text mask disagrees with tokenizer mask")
    return LocalEncoding(
        video_raw=video_raw,
        video_ignore_raw=video_ignore_raw.bool(),
        text_raw=text_raw,
        text_valid=text_valid.bool(),
        text_aug_raw=text_aug_raw,
        text_aug_valid=text_aug_valid.bool(),
    )


def _masked_softmax(logits: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
    if valid.dtype != torch.bool:
        raise SchemaError("masked softmax validity must be boolean")
    expanded = torch.broadcast_to(valid, logits.shape)
    if not bool(expanded.any(dim=dim).all()):
        raise SchemaError("masked softmax has an all-invalid reduction slice")
    probabilities = logits.masked_fill(~expanded, float("-inf")).softmax(dim=dim)
    probabilities = probabilities.masked_fill(~expanded, 0.0)
    if not bool(torch.isfinite(probabilities).all()):
        raise FloatingPointError("masked softmax produced NaN or infinity")
    return probabilities


def _token_weights(head: Any, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    logits = head(tokens).squeeze(-1).float()
    return _masked_softmax(logits, valid, dim=-1)


def directional_scores_dense(
    core: Any,
    video_raw: torch.Tensor,
    video_ignore_raw: torch.Tensor,
    text_raw: torch.Tensor,
    text_valid: torch.Tensor,
    *,
    text_aug_raw: torch.Tensor | None = None,
    text_aug_valid: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("inner similarity temperature must be positive")
    text_aug_raw = text_raw if text_aug_raw is None else text_aug_raw
    text_aug_valid = text_valid if text_aug_valid is None else text_aug_valid
    if video_raw.ndim != 3 or text_raw.ndim != 3 or text_aug_raw.ndim != 3:
        raise SchemaError("directional scorer expects [B,N,D] token tensors")
    if video_raw.shape[-1] != text_raw.shape[-1] or text_aug_raw.shape != text_raw.shape:
        raise SchemaError("video/original/augmented token dimensions are incompatible")
    video_valid = ~video_ignore_raw.bool()
    text_valid = text_valid.bool()
    text_aug_valid = text_aug_valid.bool()
    if video_valid.shape != video_raw.shape[:2]:
        raise SchemaError("video mask/token shape mismatch")
    if text_valid.shape != text_raw.shape[:2] or text_aug_valid.shape != text_aug_raw.shape[:2]:
        raise SchemaError("text mask/token shape mismatch")
    if not bool(video_valid.any(-1).all()):
        raise SchemaError("each video needs at least one scorer-visible token")
    if not bool(text_valid.any(-1).all()) or not bool(text_aug_valid.any(-1).all()):
        raise SchemaError("each caption needs at least one scorer-visible token")
    for name, tensor in (
        ("video_raw", video_raw),
        ("text_raw", text_raw),
        ("text_aug_raw", text_aug_raw),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise SchemaError(f"{name} contains NaN or infinity")

    video = F.normalize(video_raw.float(), dim=-1, eps=1e-6)
    text = F.normalize(text_raw.float(), dim=-1, eps=1e-6)
    text_aug = F.normalize(text_aug_raw.float(), dim=-1, eps=1e-6)
    video_weight = _token_weights(core.video_weight_fc, video_raw, video_valid)
    text_weight = _token_weights(core.text_weight_fc, text_aug_raw, text_aug_valid)

    affinity = torch.einsum("vld,tmd->vtlm", video, text)
    text_inner_valid = text_valid[None, :, None, :].expand_as(affinity)
    video_directed_prob = _masked_softmax(
        affinity / float(temperature), text_inner_valid, dim=3
    )
    video_position_score = (affinity * video_directed_prob).sum(dim=3)
    video_directed = (video_position_score * video_weight[:, None, :]).sum(dim=2)

    affinity_aug = torch.einsum("vld,tmd->vtlm", video, text_aug)
    video_inner_valid = video_valid[:, None, :, None].expand_as(affinity_aug)
    text_directed_prob = _masked_softmax(
        affinity_aug / float(temperature), video_inner_valid, dim=2
    )
    text_position_score = (affinity_aug * text_directed_prob).sum(dim=2)
    text_directed = (text_position_score * text_weight[None, :, :]).sum(dim=2)
    if not bool(torch.isfinite(video_directed).all()) or not bool(
        torch.isfinite(text_directed).all()
    ):
        raise FloatingPointError("directional scorer produced NaN or infinity")
    return video_directed, text_directed


def directional_scores_blocked(
    core: Any,
    video_raw: torch.Tensor,
    video_ignore_raw: torch.Tensor,
    text_raw: torch.Tensor,
    text_valid: torch.Tensor,
    *,
    text_aug_raw: torch.Tensor | None = None,
    text_aug_valid: torch.Tensor | None = None,
    temperature: float = 0.07,
    video_block: int = 32,
    text_block: int = 64,
    checkpoint_blocks: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if video_block < 1 or text_block < 1:
        raise ValueError("score blocks must be positive")
    text_aug_raw = text_raw if text_aug_raw is None else text_aug_raw
    text_aug_valid = text_valid if text_aug_valid is None else text_aug_valid
    video_rows: list[torch.Tensor] = []
    text_rows: list[torch.Tensor] = []
    for video_start in range(0, video_raw.shape[0], video_block):
        video_end = min(video_raw.shape[0], video_start + video_block)
        row_video: list[torch.Tensor] = []
        row_text: list[torch.Tensor] = []
        for text_start in range(0, text_raw.shape[0], text_block):
            text_end = min(text_raw.shape[0], text_start + text_block)
            inputs = (
                video_raw[video_start:video_end],
                video_ignore_raw[video_start:video_end],
                text_raw[text_start:text_end],
                text_valid[text_start:text_end],
                text_aug_raw[text_start:text_end],
                text_aug_valid[text_start:text_end],
            )

            def score_block(*values):
                return directional_scores_dense(
                    core,
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    text_aug_raw=values[4],
                    text_aug_valid=values[5],
                    temperature=temperature,
                )

            if checkpoint_blocks and torch.is_grad_enabled():
                current = checkpoint(score_block, *inputs, use_reentrant=False)
            else:
                current = score_block(*inputs)
            row_video.append(current[0])
            row_text.append(current[1])
        video_rows.append(torch.cat(row_video, dim=1))
        text_rows.append(torch.cat(row_text, dim=1))
    return torch.cat(video_rows, dim=0), torch.cat(text_rows, dim=0)


def _gather_boolean(tensor: torch.Tensor, runtime: DistributedRuntime) -> torch.Tensor:
    if runtime.world_size == 1:
        return tensor
    outputs = [torch.empty_like(tensor) for _ in range(runtime.world_size)]
    torch.distributed.all_gather(outputs, tensor)
    return torch.cat(outputs, dim=0)


def gather_local_encoding(
    encoding: LocalEncoding, runtime: DistributedRuntime
) -> LocalEncoding:
    if runtime.world_size == 1:
        return encoding
    return LocalEncoding(
        video_raw=gather_with_grad(encoding.video_raw),
        video_ignore_raw=_gather_boolean(encoding.video_ignore_raw, runtime),
        text_raw=gather_with_grad(encoding.text_raw),
        text_valid=_gather_boolean(encoding.text_valid, runtime),
        text_aug_raw=gather_with_grad(encoding.text_aug_raw),
        text_aug_valid=_gather_boolean(encoding.text_aug_valid, runtime),
    )


@contextmanager
def scoped_distribution_rng(seed: int, step: int, microstep: int, device: torch.device) -> Iterator[None]:
    devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(stable_seed(seed, step, microstep, "base_distribution") % (2**63 - 1))
        yield


def _masked_pool_samples(
    samples: torch.Tensor, valid: torch.Tensor, *, exclude_first: bool
) -> torch.Tensor:
    # samples [S,B,N,D] -> [S,B,D]
    current_valid = valid.bool().clone()
    if exclude_first:
        current_valid[:, 0] = False
    if not bool(current_valid.any(dim=1).all()):
        raise SchemaError("distribution pooling has an empty valid sequence")
    weights = current_valid[None, :, :, None].to(samples.dtype)
    return (samples * weights).sum(dim=2) / weights.sum(dim=2)


def _sinkhorn_plan(core: Any, similarity: torch.Tensor) -> torch.Tensor:
    batch = similarity.shape[0]
    count = similarity.shape[1]
    if similarity.shape != (batch, count, count):
        raise SchemaError("sample transport matrix must be square per pair")
    cost = 1.0 - similarity
    kernel = torch.exp(-cost / float(core.eps))
    target = torch.full(
        (batch, count), 1.0 / count, dtype=similarity.dtype, device=similarity.device
    )
    row_scale = torch.ones_like(target)
    column_scale = torch.ones_like(target)
    with torch.no_grad():
        for _ in range(int(core.max_iter)):
            previous = row_scale
            row_scale = target / torch.matmul(kernel, column_scale.unsqueeze(-1)).squeeze(-1)
            column_scale = target / torch.matmul(
                kernel.transpose(1, 2), row_scale.unsqueeze(-1)
            ).squeeze(-1)
            if (row_scale - previous).abs().mean().item() < 1e-2:
                break
        plan = row_scale.unsqueeze(-1) * column_scale.unsqueeze(-2) * kernel
    if not bool(torch.isfinite(plan).all()):
        raise FloatingPointError("sample transport plan contains NaN or infinity")
    return plan


def distribution_transport_score(
    core: Any,
    encoding: LocalEncoding,
    *,
    seed: int,
    optimizer_step: int,
    microstep: int,
    text_noise: torch.Tensor | None = None,
    video_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    video_valid = ~encoding.video_ignore_raw.bool()
    text_valid = encoding.text_aug_valid.bool()
    video_mask = (~video_valid).float()[:, None, None, :] * -1_000_000.0
    text_mask = (~text_valid).float()[:, None, None, :] * -1_000_000.0
    video = encoding.video_raw.float()
    text = encoding.text_aug_raw.float()
    with scoped_distribution_rng(seed, optimizer_step, microstep, video.device):
        text_mu, text_logsigma, _ = core.dist_text_trans(text, mask=text_mask, weight=None)
        video_mu, video_logsigma, _ = core.dist_video_trans(video, mask=video_mask, weight=None)
        if text_noise is None:
            text_noise = torch.randn_like(text_mu)
        if video_noise is None:
            video_noise = torch.randn_like(video_mu)
    text_samples = torch.stack((text_mu, text_mu + torch.exp(text_logsigma) * text_noise))
    video_samples = torch.stack((video_mu, video_mu + torch.exp(video_logsigma) * video_noise))
    text_pooled = F.normalize(
        _masked_pool_samples(text_samples, text_valid, exclude_first=True), dim=-1, eps=1e-6
    )
    video_pooled = F.normalize(
        _masked_pool_samples(video_samples, video_valid, exclude_first=False), dim=-1, eps=1e-6
    )
    similarity = torch.einsum("svd,twd->stvw", video_pooled, text_pooled)
    similarity = similarity.permute(2, 3, 0, 1).contiguous()
    pairs = similarity.shape[0] * similarity.shape[1]
    flat = similarity.view(pairs, 2, 2)
    plan = _sinkhorn_plan(core, flat)
    transported = (plan * flat).view(*similarity.shape)
    text_to_video = transported.max(dim=2).values.sum(dim=2) / 2.0
    video_to_text = transported.max(dim=3).values.sum(dim=2) / 2.0
    output = (text_to_video + video_to_text) / 2.0
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError("distribution transport score is nonfinite")
    return output


def baseline_loss_from_local(
    core: Any,
    encoding: LocalEncoding,
    *,
    runtime: DistributedRuntime,
    seed: int,
    optimizer_step: int,
    microstep: int = 0,
    temperature: float = 0.07,
    checkpoint_score_blocks: bool = False,
    video_block: int = 32,
    text_block: int = 64,
) -> torch.Tensor:
    global_encoding = gather_local_encoding(encoding, runtime)
    scorer = directional_scores_blocked if checkpoint_score_blocks else directional_scores_dense
    score_kwargs = {
        "text_aug_raw": global_encoding.text_aug_raw,
        "text_aug_valid": global_encoding.text_aug_valid,
        "temperature": temperature,
    }
    if checkpoint_score_blocks:
        score_kwargs.update(
            {
                "video_block": video_block,
                "text_block": text_block,
                "checkpoint_blocks": True,
            }
        )
    video_score, text_score = scorer(
        core,
        global_encoding.video_raw,
        global_encoding.video_ignore_raw,
        global_encoding.text_raw,
        global_encoding.text_valid,
        **score_kwargs,
    )
    if video_score.shape[0] != video_score.shape[1]:
        raise SchemaError("base contrastive matrices must be square")
    transport = distribution_transport_score(
        core,
        global_encoding,
        seed=seed,
        optimizer_step=optimizer_step,
        microstep=microstep,
    )
    scale = core.clip.logit_scale.exp().float()
    video_logits = scale * video_score + float(core.ot_weight) * transport
    text_logits = scale * text_score + float(core.ot_weight) * transport
    labels = torch.arange(video_logits.shape[0], device=video_logits.device)
    beta = float(core.dual_mix)
    if getattr(core, "mix_design", None) != "balance":
        raise ValueError("corrected_v1 currently supports the pinned balance loss only")
    loss = 0.5 * (
        beta * F.cross_entropy(video_logits, labels)
        + (1.0 - beta) * F.cross_entropy(video_logits.T, labels)
        + beta * F.cross_entropy(text_logits.T, labels)
        + (1.0 - beta) * F.cross_entropy(text_logits, labels)
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("base retrieval loss is nonfinite")
    return loss
