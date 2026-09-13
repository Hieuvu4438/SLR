from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from method1.miner import (
    OccurrenceDescriptor,
    build_candidate_graph,
    construct_caption_edits,
    fit_visual_prototypes,
    occurrence_descriptor,
    write_mining_artifacts,
)
from method1.token_spans import tokenize_with_spans


def _descriptor(uid: str, word: str, group: str, vector, concentration=0.8):
    return OccurrenceDescriptor(
        occurrence_uid=uid,
        word=word,
        text_uid=f"text:{uid}",
        video_uid=f"video:{uid}",
        group_uid=group,
        vector=np.asarray(vector, dtype=np.float32),
        concentration=concentration,
        peak_affinity=0.9,
        support_entropy=1.0 - concentration,
        valid_clip_count=2,
    )


def test_occurrence_descriptor_reports_concentrated_normalized_pool() -> None:
    result = occurrence_descriptor(
        occurrence_uid="o1",
        word="rain",
        text_uid="t1",
        video_uid="v1",
        group_uid="g1",
        x_ref=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        q=np.array([1.0, 0.0], dtype=np.float32),
        video_valid=np.array([True, True]),
        tau=0.07,
    )
    assert np.linalg.norm(result.vector) == pytest.approx(1.0)
    assert result.concentration > 0.99
    assert result.peak_affinity == pytest.approx(1.0)


def test_prototype_balances_groups_and_candidate_graph_is_stable() -> None:
    descriptors = [
        _descriptor("a1", "rain", "g1", [1.0, 0.0]),
        _descriptor("a2", "rain", "g1", [1.0, 0.0]),
        _descriptor("a3", "rain", "g2", [0.0, 1.0]),
        _descriptor("b1", "snow", "g3", [0.8, 0.6]),
        _descriptor("b2", "snow", "g4", [0.8, 0.6]),
    ]
    prototypes, report = fit_visual_prototypes(
        descriptors,
        retained_fraction=1.0,
        minimum_concentration=1e-6,
        minimum_retained_occurrences=2,
        minimum_distinct_groups=2,
    )
    np.testing.assert_allclose(
        prototypes["rain"].vector,
        np.array([1.0, 1.0]) / np.sqrt(2.0),
        atol=1e-6,
    )
    assert report["rain"]["distinct_groups"] == 2
    graph = build_candidate_graph(
        prototypes, minimum_cosine=0.7, candidates_per_word=20, block_size=1
    )
    assert graph["rain"][0][0] == "snow"
    assert graph["snow"][0][0] == "rain"


def test_caption_edits_round_robin_and_train_hash_filter(cico_tokenizer) -> None:
    caption = tokenize_with_spans(
        "rain wind",
        text_uid="ph:train:text:1",
        tokenizer=cico_tokenizer,
    )
    blocked = tokenize_with_spans(
        "snow wind",
        text_uid="ph:train:text:block",
        tokenizer=cico_tokenizer,
    )
    result = construct_caption_edits(
        {caption.text_uid: caption},
        {
            "rain": (("snow", 0.95), ("cloud", 0.90)),
            "wind": (("storm", 0.92),),
        },
        tokenizer=cico_tokenizer,
        original_train_caption_hashes={caption.caption_hash, blocked.caption_hash},
        max_edits_per_caption=2,
        miner_version="visual_prototype_v1",
    )
    edits = result.edits_by_text_uid[caption.text_uid]
    # rain->snow is blocked because it recreates a real train caption. Round-robin then
    # retains wind->storm before returning to rain->cloud.
    assert [(edit.source_word, edit.replacement_word) for edit in edits] == [
        ("wind", "storm"),
        ("rain", "cloud"),
    ]
    assert result.rejection_counts["matches_original_train_caption"] == 1
    assert len({edit.edit_uid for edit in edits}) == 2


def test_mining_artifacts_are_hashed_and_immutable(
    tmp_path: Path, cico_tokenizer
) -> None:
    caption = tokenize_with_spans(
        "rain", text_uid="ph:train:text:1", tokenizer=cico_tokenizer
    )
    mined = construct_caption_edits(
        {caption.text_uid: caption},
        {"rain": (("snow", 0.8),)},
        tokenizer=cico_tokenizer,
        original_train_caption_hashes={caption.caption_hash},
        max_edits_per_caption=2,
        miner_version="visual_prototype_v1",
    )
    descriptor = _descriptor("a1", "rain", "g1", [1.0, 0.0])
    prototypes, prototype_report = fit_visual_prototypes(
        [descriptor],
        retained_fraction=1.0,
        minimum_concentration=1e-6,
        minimum_retained_occurrences=1,
        minimum_distinct_groups=1,
    )
    report = write_mining_artifacts(
        tmp_path,
        descriptors=[descriptor],
        prototypes=prototypes,
        prototype_report=prototype_report,
        candidate_graph={"rain": (("snow", 0.8),)},
        mined=mined,
        resource_hashes={"reference": "abc"},
        diagnostics={"normalized_text_difference": {"below_margin_fraction": 0.25}},
    )
    assert report["edit_count"] == 1
    assert report["diagnostics"]["normalized_text_difference"][
        "below_margin_fraction"
    ] == 0.25
    assert (tmp_path / "mining" / "mining_report.json").is_file()
    with pytest.raises(RuntimeError, match="overwrite"):
        write_mining_artifacts(
            tmp_path,
            descriptors=[descriptor],
            prototypes=prototypes,
            prototype_report=prototype_report,
            candidate_graph={"rain": (("snow", 0.8),)},
            mined=mined,
            resource_hashes={"reference": "abc"},
        )
