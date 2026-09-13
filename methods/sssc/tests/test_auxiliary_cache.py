from __future__ import annotations

from pathlib import Path

import numpy as np

from method1.auxiliary_cache import Method1AuxiliaryCache
from method1.miner import (
    MinedCaptionEdits,
    OccurrenceDescriptor,
    PrototypeRecord,
    construct_caption_edits,
    write_mining_artifacts,
)
from method1.reference import (
    ReferenceCacheIdentity,
    write_negative_span_cache,
    write_reference_cache,
)
from method1.token_spans import tokenize_with_spans


def test_auxiliary_cache_materializes_deterministic_padded_training_fields(
    tmp_path: Path, cico_tokenizer
) -> None:
    identity = ReferenceCacheIdentity("t", "b", "c", "f", "s", "v1")
    cache_root = tmp_path / "cache"
    caption = tokenize_with_spans(
        "rain", text_uid="ph:train:text:1", tokenizer=cico_tokenizer
    )
    occurrence = caption.lexical_spans[0]
    arrays = {
        "reference_video_tokens": np.array(
            [[[1.0, 0.0], [0.0, 1.0], *([[0.0, 0.0]] * 62)]], dtype=np.float32
        ),
        "reference_video_valid": np.array([[True, True, *([False] * 62)]], dtype=np.bool_),
        "reference_positive_spans": np.array([[1.0, 0.0]], dtype=np.float32),
    }
    write_reference_cache(
        cache_root,
        identity=identity,
        arrays=arrays,
        video_index=[{"row": 0, "video_uid": "ph:train:video:1"}],
        positive_span_index=[
            {
                "row": 0,
                "occurrence_uid": occurrence.occurrence_uid,
                "text_uid": caption.text_uid,
            }
        ],
    )
    mined = construct_caption_edits(
        {caption.text_uid: caption},
        {"rain": (("snow", 0.9),)},
        tokenizer=cico_tokenizer,
        original_train_caption_hashes={caption.caption_hash},
        max_edits_per_caption=20,
        miner_version="visual_prototype_v1",
    )
    descriptor = OccurrenceDescriptor(
        occurrence.occurrence_uid,
        "rain",
        caption.text_uid,
        "ph:train:video:1",
        "ph:train:group:1",
        np.array([1.0, 0.0], dtype=np.float32),
        0.8,
        1.0,
        0.2,
        2,
    )
    prototype = PrototypeRecord("rain", descriptor.vector, 1, 1, 1, 0.8, 0.0)
    bundle_root = cache_root / "auxiliary"
    bundle_root.mkdir()
    mining_report = write_mining_artifacts(
        bundle_root,
        descriptors=[descriptor],
        prototypes={"rain": prototype},
        prototype_report={"rain": {"eligible": True}},
        candidate_graph={"rain": (("snow", 0.9),)},
        mined=MinedCaptionEdits(mined.edits_by_text_uid, mined.rejection_counts),
        resource_hashes={"reference_identity_sha256": identity.digest},
    )
    edit = mined.edits_by_text_uid[caption.text_uid][0]
    write_negative_span_cache(
        bundle_root,
        reference_identity_sha256=identity.digest,
        mining_content_sha256=mining_report["content_sha256"],
        vectors=np.array([[0.0, 1.0]], dtype=np.float32),
        edit_uids=[edit.edit_uid],
    )
    cache = Method1AuxiliaryCache(cache_root, expected_identity=identity)
    fields = cache.training_fields(
        video_uid="ph:train:video:1",
        text_uid=caption.text_uid,
        caption_hash=caption.caption_hash,
        seed=42,
        epoch=3,
        negatives_per_caption=2,
    )
    assert fields["edit_uids"] == [[edit.edit_uid], [None]]
    assert fields["edit_valid"].tolist() == [[True], [False]]
    assert fields["confidence"].tolist() == [[1.0], [0.0]]
    assert fields["q_pos"].shape == (2, 1, 2)
    assert fields["q_neg"].shape == (2, 1, 2)
