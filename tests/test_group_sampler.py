from __future__ import annotations

from elsc.data.group_sampler import CaptionGroupSampler
from elsc.data.manifest import ManifestRecord


def _record(index: int, caption_id: str) -> ManifestRecord:
    return ManifestRecord(
        schema_version=1,
        dataset="csl_daily",
        split="train",
        pair_id=f"pair-{index}",
        video_id=f"video-{index}",
        caption_id=caption_id,
        caption_original="source",
        caption_model="target",
        caption_language="en",
        feature_agnostic="agnostic.pkl",
        feature_aware="aware.pkl",
        dense_length=4,
        feature_dim=1024,
    )


def test_caption_group_sampler_is_epoch_deterministic_and_group_balanced():
    records = [_record(0, "a"), _record(1, "a"), _record(2, "b")]
    sampler = CaptionGroupSampler(records, seed=42)
    assert len(sampler) == 2
    selections = []
    for epoch in range(10):
        sampler.set_epoch(epoch)
        first = list(sampler)
        second = list(sampler)
        assert first == second
        assert len(first) == 2
        assert sum(index in {0, 1} for index in first) == 1
        assert 2 in first
        selections.extend(index for index in first if index in {0, 1})
    assert set(selections) == {0, 1}
