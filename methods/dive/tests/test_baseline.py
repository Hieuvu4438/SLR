from __future__ import annotations

import numpy as np
import torch

from dive.adapters import NativeTextFeatures, NativeVideoFeatures, PrelogitScores
from dive.baseline import score_feature_gallery


class DotAdapter:
    def score_prelogit(self, video_features, text_features):
        scores = video_features.pooled @ text_features.pooled.T
        return PrelogitScores(
            video_ids=video_features.sample_ids,
            text_ids=text_features.text_ids,
            scores=scores,
            logit_scale=torch.tensor(1.0),
        )


def test_score_feature_gallery_covers_every_chunk_in_id_order():
    video_values = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
    text_values = torch.nn.functional.normalize(torch.randn(7, 4), dim=-1)
    video = NativeVideoFeatures(
        sample_ids=tuple(f"v{index}" for index in range(5)),
        pooled=video_values,
        validity=torch.ones(5, 1, dtype=torch.bool),
    )
    text = NativeTextFeatures(
        text_ids=tuple(f"t{index}" for index in range(7)),
        pooled=text_values,
        token_features=text_values[:, None],
        token_validity=torch.ones(7, 1, dtype=torch.bool),
    )
    scores = score_feature_gallery(
        DotAdapter(),
        video,
        text,
        device=torch.device("cpu"),
        video_chunk=2,
        text_chunk=3,
    )
    np.testing.assert_allclose(scores, (video_values @ text_values.T).numpy(), atol=1e-7)
    assert scores.shape == (5, 7)
