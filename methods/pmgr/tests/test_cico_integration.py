from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pmgr.config import load_config
from pmgr.data.group_dataset import GroupCollator, GroupDataset
from pmgr.model import build_retriever, load_tokenizer
from pmgr.scoring import mixed_pair_scores


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "methods/pmgr/configs/pmgr_csl.json"


@pytest.mark.skipif(
    not (ROOT / "artifacts/pmgr/indexes/csl_train.json").is_file(),
    reason="real CSL PMGR index is unavailable",
)
def test_real_cico_adapter_legacy_score_parity_and_batch_shapes(monkeypatch):
    monkeypatch.chdir(ROOT)
    config = load_config(CONFIG, mode="train")
    model, _ = build_retriever(config, device="cpu")
    tokenizer = load_tokenizer(config)
    dataset = GroupDataset(
        config["paths"]["train_index"],
        feature_len=int(config["data"]["max_features"]),
        alpha=float(config["data"]["feature_mix_alpha"]),
    )
    collator = GroupCollator(
        tokenizer,
        int(config["data"]["max_text_tokens"]),
        dataset_group_count=dataset.index.group_count,
        dataset_video_count=dataset.index.video_count,
        seed=0,
        augment=True,
    )
    batch = collator([dataset[0], dataset[1]])
    model.eval()
    with torch.no_grad():
        encoded = model.encode_pmgr_batch(batch)
        q, a, b = mixed_pair_scores(
            encoded.video_hidden,
            encoded.text_hidden,
            encoded.aug_hidden,
            encoded.video_valid,
            encoded.text_valid,
            encoded.aug_valid,
            omega=0.5,
            sigma=0.07,
            mask_policy="legacy_unmasked",
        )
        upstream_a, upstream_b = model.core.get_similarity_logits(
            encoded.text_hidden,
            encoded.video_hidden,
            encoded.text_valid.long(),
            batch["video_padding_mask"],
            shaped=True,
            loose_type=True,
            # CiCo's flag selects the Filip scorer; module.eval() independently disables gather.
            is_train=True,
            sequence_hidden_aug=encoded.aug_hidden,
            text_mask_aug=encoded.aug_valid.long(),
        )[:2]
    scale = model.logit_scale.exp()
    assert q.shape == (len(batch["video_ids"]), len(batch["group_ids"]))
    assert encoded.video_hidden.shape[1:] == (65, 512)
    assert encoded.text_hidden.shape[1:] == (32, 512)
    torch.testing.assert_close(a, upstream_a / scale, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(b, upstream_b / scale, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(q, 0.5 * (upstream_a + upstream_b) / scale, atol=2e-6, rtol=2e-6)
