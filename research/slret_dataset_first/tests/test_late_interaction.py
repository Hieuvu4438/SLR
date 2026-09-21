import torch

from methods.sl_mvr.late_interaction import (
    DualLevelLateInteraction,
    mean_maxsim,
    mean_pool_similarity,
)


def test_mean_maxsim_uses_best_valid_window_and_ignores_padding():
    video = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [99.0, 99.0]]])
    text = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [99.0, 99.0]]])
    video_mask = torch.tensor([[True, True, False]])
    text_mask = torch.tensor([[True, True, False]])
    score = mean_maxsim(video, text, video_mask, text_mask)
    assert score.shape == (1, 1)
    assert torch.allclose(score, torch.ones_like(score))


def test_late_interaction_distinguishes_composition_lost_by_mean_pooling():
    video = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [0.5, 0.5]],
        ]
    )
    text = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    video_mask = torch.ones(2, 2, dtype=torch.bool)
    text_mask = torch.ones(1, 2, dtype=torch.bool)
    late = mean_maxsim(video, text, video_mask, text_mask)
    pooled = mean_pool_similarity(video, text, video_mask, text_mask)
    assert late[0, 0] > late[1, 0]
    assert torch.allclose(pooled[0, 0], pooled[1, 0])


def test_dual_level_loss_is_finite_and_backpropagates():
    torch.manual_seed(7)
    model = DualLevelLateInteraction(visual_dim=5, text_dim=7, projection_dim=3)
    visual = torch.randn(3, 4, 5)
    latent = torch.randn(3, 4, 5)
    text = torch.randn(3, 6, 7)
    video_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    text_mask = torch.tensor(
        [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]],
        dtype=torch.bool,
    )
    outputs = model(visual, latent, text, video_mask, text_mask)
    loss = model.loss(outputs)
    loss.backward()
    assert outputs["scores"].shape == (3, 3)
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_matched_mean_control_uses_same_parameters_and_shapes():
    torch.manual_seed(9)
    model = DualLevelLateInteraction(visual_dim=4, text_dim=6, projection_dim=3)
    visual = torch.randn(2, 3, 4)
    text = torch.randn(2, 5, 6)
    video_mask = torch.ones(2, 3, dtype=torch.bool)
    text_mask = torch.ones(2, 5, dtype=torch.bool)
    late = model(visual, visual, text, video_mask, text_mask, interaction="late")
    mean = model(visual, visual, text, video_mask, text_mask, interaction="mean")
    assert late["scores"].shape == mean["scores"].shape == (2, 2)
    assert set(late) == set(mean) == {"feature_scores", "latent_scores", "scores"}


def test_cached_projection_scoring_matches_forward_in_both_modes():
    torch.manual_seed(11)
    model = DualLevelLateInteraction(visual_dim=4, text_dim=6, projection_dim=3)
    visual = torch.randn(3, 4, 4)
    latent = torch.randn(3, 4, 4)
    text = torch.randn(2, 5, 6)
    video_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    text_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    feature_video, latent_video = model.encode_video(visual, latent)
    feature_text, latent_text = model.encode_text(text)
    for interaction in ("late", "mean"):
        direct = model(visual, latent, text, video_mask, text_mask, interaction=interaction)
        cached = model.score_encoded(
            feature_video, latent_video, feature_text, latent_text,
            video_mask, text_mask, interaction=interaction,
        )
        for key in ("feature_scores", "latent_scores", "scores"):
            assert torch.allclose(direct[key], cached[key])
