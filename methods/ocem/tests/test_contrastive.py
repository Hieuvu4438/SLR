from __future__ import annotations

import pytest
import torch

from ocem.training.contrastive import (
    ContrastiveContractError,
    exact_duplicate_negative_mask,
    score_pair_union_in_blocks,
    select_mixed_candidates,
    symmetric_candidate_loss,
    symmetric_loss_from_pair_scores,
)


def test_duplicate_caption_mask_keeps_only_original_diagonal_positive() -> None:
    mask = exact_duplicate_negative_mask(["a", "b", "c"], ["same", "same", "other"])
    assert torch.equal(
        mask,
        torch.tensor([[True, False, True], [False, True, True], [True, True, True]]),
    )
    with pytest.raises(ContrastiveContractError, match="sampler"):
        exact_duplicate_negative_mask(["a", "a"], ["x", "y"])


def test_row_and_column_mining_use_their_declared_directions() -> None:
    allowed = torch.ones(4, 4, dtype=torch.bool)
    v2t = torch.tensor(
        [[0.0, 9.0, 2.0, 1.0], [3.0, 0.0, 8.0, 1.0], [7.0, 2.0, 0.0, 1.0], [1.0, 6.0, 5.0, 0.0]]
    )
    t2v = torch.tensor(
        [[0.0, 1.0, 8.0, 2.0], [9.0, 0.0, 2.0, 3.0], [4.0, 7.0, 0.0, 5.0], [3.0, 2.0, 6.0, 0.0]]
    )
    selected = select_mixed_candidates(
        v2t,
        t2v,
        allowed_mask=allowed,
        sample_ids=["a", "b", "c", "d"],
        seed=7,
        max_hard=1,
        max_random=0,
    )
    assert selected.row_candidates == ((0, 1), (1, 2), (2, 0), (3, 1))
    assert selected.column_candidates == ((0, 1), (1, 2), (2, 0), (3, 2))
    assert set(selected.pair_union) == {
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (0, 1),
        (1, 2),
        (2, 0),
        (3, 1),
        (1, 0),
        (2, 1),
        (0, 2),
        (2, 3),
    }


def test_compute_union_does_not_change_anchor_denominators() -> None:
    logits = torch.tensor(
        [[3.0, 2.0, -4.0], [1.0, 3.0, 2.5], [2.0, -3.0, 3.0]], requires_grad=True
    )
    allowed = torch.ones(3, 3, dtype=torch.bool)
    rows = ((0, 1), (1, 2), (2, 0))
    columns = ((0, 2), (1, 0), (2, 1))
    selected = {(row, column) for row, values in enumerate(rows) for column in values}
    selected.update((row, column) for column, values in enumerate(columns) for row in values)
    pair_scores = {(row, column): logits[row, column] for row, column in selected}
    exact = symmetric_candidate_loss(
        logits, allowed_mask=allowed, row_candidates=rows, column_candidates=columns
    )
    assembled = symmetric_loss_from_pair_scores(
        pair_scores, row_candidates=rows, column_candidates=columns
    )
    union_as_every_denominator = symmetric_candidate_loss(
        logits,
        allowed_mask=allowed,
        row_candidates=tuple(tuple(range(3)) for _ in range(3)),
        column_candidates=tuple(tuple(range(3)) for _ in range(3)),
    )
    assert torch.equal(exact, assembled)
    assert not torch.equal(exact, union_as_every_denominator)


def test_dense_and_block_pair_scores_have_matching_loss_gradient_and_update() -> None:
    torch.manual_seed(4)
    videos_dense = torch.nn.Parameter(torch.randn(4, 5))
    texts_dense = torch.nn.Parameter(torch.randn(4, 5))
    videos_block = torch.nn.Parameter(videos_dense.detach().clone())
    texts_block = torch.nn.Parameter(texts_dense.detach().clone())
    allowed = torch.ones(4, 4, dtype=torch.bool)
    rows = ((0, 1, 2), (1, 0, 3), (2, 0, 3), (3, 1, 2))
    columns = ((0, 2, 3), (1, 0, 3), (2, 0, 1), (3, 1, 2))
    union = {(row, column) for row, values in enumerate(rows) for column in values}
    union.update((row, column) for column, values in enumerate(columns) for row in values)

    dense_logits = videos_dense @ texts_dense.T
    dense_loss = symmetric_candidate_loss(
        dense_logits, allowed_mask=allowed, row_candidates=rows, column_candidates=columns
    )
    dense_loss.backward()

    def scorer(video_indices: torch.Tensor, text_indices: torch.Tensor) -> torch.Tensor:
        return (videos_block[video_indices] * texts_block[text_indices]).sum(dim=-1)

    pair_scores = score_pair_union_in_blocks(scorer, sorted(union), block_size=3, device="cpu")
    block_loss = symmetric_loss_from_pair_scores(
        pair_scores, row_candidates=rows, column_candidates=columns
    )
    block_loss.backward()
    assert torch.allclose(block_loss, dense_loss, atol=1e-7)
    assert torch.allclose(videos_block.grad, videos_dense.grad, atol=1e-7)
    assert torch.allclose(texts_block.grad, texts_dense.grad, atol=1e-7)

    dense_optimizer = torch.optim.SGD([videos_dense, texts_dense], lr=0.01, momentum=0.9)
    block_optimizer = torch.optim.SGD([videos_block, texts_block], lr=0.01, momentum=0.9)
    dense_optimizer.step()
    block_optimizer.step()
    assert torch.allclose(videos_block, videos_dense, atol=1e-7)
    assert torch.allclose(texts_block, texts_dense, atol=1e-7)


def test_batch_without_real_negative_fails_closed() -> None:
    allowed = exact_duplicate_negative_mask(["a", "b"], ["same", "same"])
    with pytest.raises(ContrastiveContractError, match="no eligible real negative"):
        select_mixed_candidates(
            torch.eye(2),
            torch.eye(2),
            allowed_mask=allowed,
            sample_ids=["a", "b"],
            seed=0,
        )
