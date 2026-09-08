from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from dive.cache_build import CacheBuildError, _expand_rows, _rf_timestamps, load_frozen_reference
from dive.models.evidence import EvidenceEncoder, state_hash


class PointwisePose(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 4)

    def forward(self, pose, grid):
        return self.projection(pose[:, : grid.shape[1]])


class FakeAdapter:
    def clone_local_pose_encoder(self):
        return PointwisePose()


class FakeRfAdapter:
    def describe_receptive_field(self, native, grid_id):
        assert native.grid_id == grid_id == "canonical"
        return (
            {"sample_id": "s0", "clip_index": 0, "raw_seconds_interval": [0.0, 0.5]},
            {"sample_id": "s1", "clip_index": 0, "raw_seconds_interval": [0.0, 0.4]},
            {"sample_id": "s1", "clip_index": 1, "raw_seconds_interval": [0.4, 0.8]},
        )


def _reference_payload(model: EvidenceEncoder) -> dict[str, object]:
    return {
        "schema_version": "dive_reference.v1",
        "model": copy.deepcopy(model.state_dict()),
        "state_hash": state_hash(model),
        "config_hash": "config",
        "fingerprints": {
            "baseline": "baseline",
            "data": "data",
            "grid": "grid",
            "units": "units",
            "dev_gallery": "dev",
        },
        "dtype": "float32",
        "selection_split": "dev",
        "selection_metric": "mean_t2v_v2t_r1",
        "torch_version": str(torch.__version__),
    }


def test_reference_loader_is_weights_only_and_checks_parent_fingerprints(tmp_path):
    torch.manual_seed(41)
    model = EvidenceEncoder(PointwisePose(), rgb_dim=5, pose_dim=4, hidden_dim=12, output_dim=8)
    path = tmp_path / "reference.pt"
    payload = _reference_payload(model)
    torch.save(payload, path)
    expected = {
        name: str(payload["fingerprints"][name]) for name in ("baseline", "data", "grid", "units")
    }
    loaded, metadata = load_frozen_reference(
        path,
        FakeAdapter(),
        rgb_dim=5,
        pose_dim=4,
        hidden_dim=12,
        output_dim=8,
        normalize_epsilon=1e-6,
        expected_config_hash="config",
        expected_fingerprints=expected,
        device=torch.device("cpu"),
    )
    assert state_hash(loaded) == metadata["state_hash"] == state_hash(model)
    assert all(not parameter.requires_grad for parameter in loaded.parameters())

    with pytest.raises(CacheBuildError, match="parent fingerprints differ"):
        load_frozen_reference(
            path,
            FakeAdapter(),
            rgb_dim=5,
            pose_dim=4,
            hidden_dim=12,
            output_dim=8,
            normalize_epsilon=1e-6,
            expected_config_hash="config",
            expected_fingerprints=expected | {"baseline": "changed"},
            device=torch.device("cpu"),
        )


def test_cache_helpers_expand_shared_text_rows_and_preserve_exact_rf_coverage():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert _expand_rows(values, [0, 1, 0]).tolist() == [
        [1.0, 2.0],
        [3.0, 4.0],
        [1.0, 2.0],
    ]
    native = type("Native", (), {"sample_ids": ("s0", "s1"), "grid_id": "canonical"})()
    valid = torch.tensor([[True, False], [True, True]])
    timestamps = _rf_timestamps(FakeRfAdapter(), native, valid)
    torch.testing.assert_close(
        timestamps,
        torch.tensor(
            [
                [[0.0, 0.5], [0.0, 0.0]],
                [[0.0, 0.4], [0.4, 0.8]],
            ]
        ),
    )
