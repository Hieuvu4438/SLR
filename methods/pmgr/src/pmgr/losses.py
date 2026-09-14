from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class RiskContractError(ValueError):
    """Raised when a score matrix cannot represent the declared protocol."""


def _validate_mapping(q: Tensor, video_to_group: Tensor) -> tuple[int, int]:
    if q.ndim != 2:
        raise RiskContractError("q must be a rectangular [videos, groups] score matrix")
    videos, groups = q.shape
    if videos == 0 or groups == 0:
        raise RiskContractError("score matrix cannot be empty")
    if video_to_group.dtype != torch.long or video_to_group.shape != (videos,):
        raise RiskContractError("video_to_group must be a LongTensor with one target per row")
    if video_to_group.device != q.device:
        raise RiskContractError("scores and video targets must share a device")
    if int(video_to_group.min()) < 0 or int(video_to_group.max()) >= groups:
        raise RiskContractError("video target is outside the score columns")
    counts = torch.bincount(video_to_group, minlength=groups)
    if not bool(counts.gt(0).all()):
        raise RiskContractError("every candidate group must contain a loaded performance")
    if not bool(torch.isfinite(q).all()):
        raise RiskContractError("score matrix contains NaN or infinity")
    return videos, groups


def whole_group_max(q: Tensor, video_to_group: Tensor) -> Tensor:
    """Reduce complete performance groups after channel mixing.

    Output rows are text queries and columns are candidate groups. Indexed ``max`` deliberately
    keeps the first tied member's gradient in persisted within-group order.
    """
    _, groups = _validate_mapping(q, video_to_group)
    transposed = q.transpose(0, 1)
    columns = []
    for group in range(groups):
        positions = torch.nonzero(video_to_group == group, as_tuple=False).flatten()
        columns.append(transposed.index_select(1, positions).max(dim=1).values)
    return torch.stack(columns, dim=1)


def rank_log_terms(scores: Tensor, targets: Tensor, eta: float) -> Tensor:
    if scores.ndim != 2 or targets.dtype != torch.long or targets.shape != (scores.shape[0],):
        raise RiskContractError("rank targets do not align with the score matrix")
    if eta <= 0:
        raise RiskContractError("rank_eta must be positive")
    if scores.shape[1] < 2:
        raise RiskContractError("rank risk requires at least two candidates")
    positive = scores.gather(1, targets[:, None])
    candidates = torch.arange(scores.shape[1], device=scores.device)[None, :]
    negative = candidates != targets[:, None]
    exceedances = torch.sigmoid((scores - positive) / eta).masked_fill(~negative, 0.0)
    return torch.log1p(exceedances.sum(dim=1))


def _population_values(
    q: Tensor,
    video_to_group: Tensor,
    logit_scale: Tensor,
    dataset_group_count: int,
    dataset_video_count: int,
) -> tuple[int, int, Tensor, Tensor, Tensor, float]:
    _, groups = _validate_mapping(q, video_to_group)
    total_groups = int(dataset_group_count)
    total_videos = int(dataset_video_count)
    if not (0 < total_groups <= total_videos) or groups > total_groups:
        raise RiskContractError("invalid global training population")
    if logit_scale.numel() != 1 or logit_scale.device != q.device:
        raise RiskContractError("live scalar logit_scale must share the score device")
    work_q = q if q.dtype in (torch.float32, torch.float64) else q.float()
    targets_t = torch.arange(groups, device=q.device)
    scale = logit_scale.to(work_q.dtype).exp()
    video_weight = total_groups / (total_videos * groups)
    return total_groups, total_videos, work_q, targets_t, scale, video_weight


def pmgr_loss(
    q: Tensor,
    video_to_group: Tensor,
    logit_scale: Tensor,
    dataset_group_count: int,
    dataset_video_count: int,
    *,
    rank_mix: float = 0.0,
    rank_eta: float = 0.03,
) -> dict[str, Tensor]:
    if not 0.0 <= rank_mix <= 1.0:
        raise RiskContractError("rank_mix must lie in [0,1]")
    _, _, q, text_targets, scale, video_weight = _population_values(
        q, video_to_group, logit_scale, dataset_group_count, dataset_video_count
    )
    group_scores = whole_group_max(q, video_to_group)
    ce_t = F.cross_entropy(scale * group_scores, text_targets, reduction="mean")
    ce_v = video_weight * F.cross_entropy(scale * q, video_to_group, reduction="sum")
    rank_t = rank_log_terms(group_scores, text_targets, rank_eta).mean()
    rank_v = video_weight * rank_log_terms(q, video_to_group, rank_eta).sum()
    ce = 0.5 * (ce_t + ce_v)
    rank = 0.5 * (rank_t + rank_v)
    loss = (1.0 - rank_mix) * ce + rank_mix * rank
    return {
        "loss": loss,
        "ce_t": ce_t,
        "ce_v": ce_v,
        "rank_t": rank_t,
        "rank_v": rank_v,
        "group_scores": group_scores,
    }


def _ordinary_video_ce(q: Tensor, targets: Tensor, scale: Tensor) -> Tensor:
    return F.cross_entropy(scale * q, targets, reduction="mean")


def _all_uniform_t2v(q: Tensor, owner: Tensor, scale: Tensor) -> Tensor:
    groups = q.shape[1]
    video_logits = scale * q.transpose(0, 1)
    log_prob = F.log_softmax(video_logits, dim=1)
    terms = []
    for group in range(groups):
        members = torch.nonzero(owner == group, as_tuple=False).flatten()
        terms.append(-log_prob[group].index_select(0, members).mean())
    return torch.stack(terms).mean()


def _all_set_t2v(q: Tensor, owner: Tensor, scale: Tensor) -> Tensor:
    groups = q.shape[1]
    video_logits = scale * q.transpose(0, 1)
    terms = []
    for group in range(groups):
        members = torch.nonzero(owner == group, as_tuple=False).flatten()
        row = video_logits[group]
        terms.append(-(torch.logsumexp(row.index_select(0, members), 0) - torch.logsumexp(row, 0)))
    return torch.stack(terms).mean()


def legacy_branch_balanced_loss(a: Tensor, b: Tensor, logit_scale: Tensor, omega: float) -> Tensor:
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != a.shape:
        raise RiskContractError("legacy control requires square one-representative channel matrices")
    if not 0.0 <= omega <= 1.0:
        raise RiskContractError("omega must lie in [0,1]")
    target = torch.arange(a.shape[0], device=a.device)
    scale = logit_scale.to(a.dtype).exp()
    def ce(value: Tensor) -> Tensor:
        return F.cross_entropy(scale * value, target)
    return 0.5 * (
        omega * ce(a)
        + (1.0 - omega) * ce(a.transpose(0, 1))
        + omega * ce(b.transpose(0, 1))
        + (1.0 - omega) * ce(b)
    )


def objective_loss(
    mode: str,
    q: Tensor,
    video_to_group: Tensor,
    logit_scale: Tensor,
    dataset_group_count: int,
    dataset_video_count: int,
    *,
    a: Tensor | None = None,
    b: Tensor | None = None,
    omega: float = 0.5,
    rank_mix: float = 0.0,
    rank_eta: float = 0.03,
) -> dict[str, Tensor]:
    """Evaluate a named C0--C8 objective under one explicit score contract."""
    aliases = {
        "legacy_cico": "legacy_cico",
        "single_mixed_ce": "single_mixed_ce",
        "all_uniform_ce": "all_uniform_ce",
        "all_set_ce": "all_set_ce",
        "all_uniform_ce_population_weighted": "all_uniform_ce_population_weighted",
        "all_set_ce_population_weighted": "all_set_ce_population_weighted",
        "group_ce": "group_ce",
        "group_ce_batch_mean": "group_ce_batch_mean",
        "single_rank": "single_rank",
        "pmgr": "pmgr",
        "smooth_ap_single_positive_control": "smooth_ap_single_positive_control",
    }
    if mode not in aliases:
        raise RiskContractError(f"unsupported objective mode: {mode}")
    total_groups, total_videos, q, targets_t, scale, population_weight = _population_values(
        q, video_to_group, logit_scale, dataset_group_count, dataset_video_count
    )
    videos, groups = q.shape
    singleton = videos == groups and torch.equal(video_to_group, targets_t)
    zero = q.sum() * 0.0

    if mode == "legacy_cico":
        if not singleton or a is None or b is None:
            raise RiskContractError("legacy_cico requires one aligned representative and both channels")
        return {
            "loss": legacy_branch_balanced_loss(a, b, logit_scale, omega),
            "ce_t": zero,
            "ce_v": zero,
            "rank_t": zero,
            "rank_v": zero,
        }

    if mode in {"single_mixed_ce", "single_rank"} and not singleton:
        raise RiskContractError(f"{mode} requires one aligned representative per group")
    if mode == "single_mixed_ce":
        ce_t = F.cross_entropy(scale * q.transpose(0, 1), targets_t)
        ce_v = F.cross_entropy(scale * q, video_to_group)
        return {"loss": 0.5 * (ce_t + ce_v), "ce_t": ce_t, "ce_v": ce_v,
                "rank_t": zero, "rank_v": zero}
    if mode == "single_rank":
        rank_t = rank_log_terms(q.transpose(0, 1), targets_t, rank_eta).mean()
        rank_v = rank_log_terms(q, video_to_group, rank_eta).mean()
        return {"loss": 0.5 * (rank_t + rank_v), "ce_t": zero, "ce_v": zero,
                "rank_t": rank_t, "rank_v": rank_v}

    if mode in {"group_ce", "pmgr"}:
        return pmgr_loss(
            q,
            video_to_group,
            logit_scale,
            total_groups,
            total_videos,
            rank_mix=rank_mix if mode == "pmgr" else 0.0,
            rank_eta=rank_eta,
        )

    if mode == "group_ce_batch_mean":
        group_scores = whole_group_max(q, video_to_group)
        ce_t = F.cross_entropy(scale * group_scores, targets_t)
        ce_v = _ordinary_video_ce(q, video_to_group, scale)
        return {"loss": 0.5 * (ce_t + ce_v), "ce_t": ce_t, "ce_v": ce_v,
                "rank_t": zero, "rank_v": zero, "group_scores": group_scores}

    if mode in {
        "all_uniform_ce", "all_set_ce", "all_uniform_ce_population_weighted",
        "all_set_ce_population_weighted",
    }:
        ce_t = (
            _all_uniform_t2v(q, video_to_group, scale)
            if mode.startswith("all_uniform_ce")
            else _all_set_t2v(q, video_to_group, scale)
        )
        ce_v = (
            population_weight * F.cross_entropy(scale * q, video_to_group, reduction="sum")
            if mode.endswith("population_weighted")
            else _ordinary_video_ce(q, video_to_group, scale)
        )
        return {"loss": 0.5 * (ce_t + ce_v), "ce_t": ce_t, "ce_v": ce_v,
                "rank_t": zero, "rank_v": zero}

    group_scores = whole_group_max(q, video_to_group)
    log_rank_t = rank_log_terms(group_scores, targets_t, rank_eta)
    log_rank_v = rank_log_terms(q, video_to_group, rank_eta)
    ap_t = (1.0 - torch.exp(-log_rank_t)).mean()
    ap_v = population_weight * (1.0 - torch.exp(-log_rank_v)).sum()
    ce_t = F.cross_entropy(scale * group_scores, targets_t)
    ce_v = population_weight * F.cross_entropy(scale * q, video_to_group, reduction="sum")
    ap = 0.5 * (ap_t + ap_v)
    ce = 0.5 * (ce_t + ce_v)
    loss = (1.0 - rank_mix) * ce + rank_mix * ap
    return {"loss": loss, "ce_t": ce_t, "ce_v": ce_v, "rank_t": ap_t,
            "rank_v": ap_v, "group_scores": group_scores}
