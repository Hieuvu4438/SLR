from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

import method1.inference as inference_module
from method1.config import load_config
from method1.baseline import directional_scores_dense
from method1.inference import evaluate_checkpoint, export_student
from method1.model_factory import load_exact_student_state


class ZeroHead(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*tokens.shape[:-1], 1, device=tokens.device)


class FakeInferenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.video_weight_fc = ZeroHead()
        self.text_weight_fc = ZeroHead()


class WeightedInferenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_weight_fc = nn.Linear(2, 1)
        self.text_weight_fc = nn.Linear(2, 1)


def test_complete_pool_inference_uses_grouped_max_and_persists_identities(
    tmp_path: Path, monkeypatch
) -> None:
    base = load_config("methods/sssc/configs/method1/ph_local.yaml")
    manifest = tmp_path / "manifests"
    manifest.mkdir()
    for name in ("texts.jsonl", "videos.jsonl", "groups.jsonl"):
        (manifest / name).write_text("fixture\n", encoding="utf-8")
    config = replace(
        base,
        data=replace(base.data, manifest_dir=str(manifest)),
        output=replace(base.output, root=str(tmp_path / "run")),
    )
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"checkpoint")
    model = FakeInferenceModel()
    monkeypatch.setattr(inference_module, "audit_resources", lambda *args: {})
    monkeypatch.setattr(
        inference_module,
        "build_upret_model",
        lambda *args, **kwargs: (model, {"architecture": {"embedding_dim": 2}}),
    )
    monkeypatch.setattr(
        inference_module, "load_exact_student_state", lambda *args: {"arm": "span_shared"}
    )
    monkeypatch.setattr(inference_module, "create_upret_tokenizer", lambda *args: object())
    video = torch.tensor(
        [
            [[9.0, 9.0], [1.0, 0.0]],
            [[9.0, 9.0], [0.9, 0.1]],
            [[9.0, 9.0], [0.0, 1.0]],
        ]
    )
    video_ignore = torch.tensor([[True, False], [True, False], [True, False]])
    text = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    text_valid = torch.ones(2, 1, dtype=torch.bool)
    videos = [
        SimpleNamespace(video_uid="v0", group_uid="g0"),
        SimpleNamespace(video_uid="v1", group_uid="g0"),
        SimpleNamespace(video_uid="v2", group_uid="g1"),
    ]
    groups = [SimpleNamespace(group_uid="g0"), SimpleNamespace(group_uid="g1")]
    monkeypatch.setattr(
        inference_module,
        "_encode_inference_pool",
        lambda *args, **kwargs: (
            video,
            video_ignore,
            text,
            text_valid,
            videos,
            groups,
        ),
    )
    report = evaluate_checkpoint(config, checkpoint, split="dev", persist=False)
    assert report["candidate_counts"] == {"videos": 3, "groups": 2}
    assert report["V2T"]["R1"] == 100.0
    assert report["T2V"]["R1"] == 100.0
    assert report["query_ids"] == {"V2T": ["v0", "v1", "v2"], "T2V": ["g0", "g1"]}
    assert report["query_group_ids"] == {
        "V2T": ["g0", "g0", "g1"],
        "T2V": ["g0", "g1"],
    }
    assert report["top_candidate_ids"]["T2V"][0][0] == "g0"


def test_export_contains_only_student_inference_dependencies(tmp_path: Path, monkeypatch) -> None:
    config = load_config("methods/sssc/configs/method1/ph_local.yaml")
    checkpoint = tmp_path / "training.pt"
    checkpoint.write_bytes(b"training-checkpoint")
    output = tmp_path / "export.pt"
    model = FakeInferenceModel()
    monkeypatch.setattr(inference_module, "audit_resources", lambda *args: {})
    monkeypatch.setattr(
        inference_module,
        "build_upret_model",
        lambda *args, **kwargs: (model, {"architecture": {"embedding_dim": 2}}),
    )
    monkeypatch.setattr(
        inference_module, "load_exact_student_state", lambda *args: {"arm": "span_shared"}
    )
    report = export_student(config, checkpoint, output)
    artifact = torch.load(output, map_location="cpu", weights_only=True)
    assert report["training_only_dependencies"] == []
    assert artifact["artifact_type"] == "method1_student_inference"
    assert artifact["source_arm"] == "span_shared"
    assert "optimizer_state_dict" not in artifact
    assert all("teacher" not in key and "reference" not in key for key in artifact)


def test_exported_student_has_identical_fixed_fixture_scores(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_config("methods/sssc/configs/method1/ph_local.yaml")
    source_model = WeightedInferenceModel()
    with torch.no_grad():
        source_model.video_weight_fc.weight.copy_(torch.tensor([[0.3, -0.2]]))
        source_model.video_weight_fc.bias.fill_(0.1)
        source_model.text_weight_fc.weight.copy_(torch.tensor([[-0.4, 0.5]]))
        source_model.text_weight_fc.bias.fill_(-0.2)
    checkpoint = tmp_path / "training.pt"
    torch.save(
        {"student_state_dict": source_model.state_dict(), "arm": "span_shared"},
        checkpoint,
    )
    output = tmp_path / "export.pt"
    monkeypatch.setattr(inference_module, "audit_resources", lambda *args: {})
    monkeypatch.setattr(
        inference_module,
        "build_upret_model",
        lambda *args, **kwargs: (
            WeightedInferenceModel(),
            {"architecture": {"embedding_dim": 2}},
        ),
    )
    export_student(config, checkpoint, output)
    exported_model = WeightedInferenceModel()
    load_exact_student_state(exported_model, output)
    video = torch.tensor(
        [[[1.0, 0.0], [0.2, 0.8]], [[0.0, 1.0], [0.7, 0.3]]]
    )
    video_ignore = torch.zeros(2, 2, dtype=torch.bool)
    text = torch.tensor(
        [[[0.8, 0.2], [0.1, 0.9]], [[0.3, 0.7], [0.9, 0.1]]]
    )
    text_valid = torch.ones(2, 2, dtype=torch.bool)

    expected = directional_scores_dense(
        source_model, video, video_ignore, text, text_valid
    )
    actual = directional_scores_dense(
        exported_model, video, video_ignore, text, text_valid
    )
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
