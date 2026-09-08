from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

from dive.adapters.seds import PINNED_SEDS_COMMIT
from dive.artifacts import ArtifactResolver
from dive.config import config_hash

from .manifest import SampleRecord
from .temporal import CompactFrameMap
from .text_units import normalize_text


class PreparationError(ValueError):
    """Raw How2Sign/SEDS sources cannot produce the controlled DIVE data contract."""


@dataclass(frozen=True)
class _SourceItem:
    split: str
    sentence_id: str
    video_stem: str
    text: str
    source_video_id: str
    source_start_sec: float | None
    source_end_sec: float | None


@dataclass(frozen=True)
class _AssetProbe:
    video_frames: int
    pose_frames: int
    fps: float

    @property
    def duration_sec(self) -> float:
        return self.video_frames / self.fps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
            )
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise PreparationError(f"cannot inspect pinned SEDS checkout: {result.stderr.strip()}")
    return result.stdout.strip()


def _validate_seds_checkout(root: Path) -> str:
    if not root.is_dir() or _run_git(root, "rev-parse", "HEAD") != PINNED_SEDS_COMMIT:
        raise PreparationError("SEDS checkout is missing or not at the pinned commit")
    if _run_git(root, "status", "--porcelain"):
        raise PreparationError("SEDS checkout must be clean before preparing protocol annotations")
    return PINNED_SEDS_COMMIT


def _load_pickle_mapping(path: Path) -> Mapping[str, Any]:
    # These pickle files belong to the validated, pinned upstream checkout.
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
    except Exception as exc:
        raise PreparationError(f"cannot load pinned SEDS annotation: {path}") from exc
    if not isinstance(value, Mapping) or not value:
        raise PreparationError(f"SEDS annotation is not a nonempty mapping: {path}")
    return value


def _load_train_timing(path: Path) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PreparationError(f"invalid train timing JSON at {path}:{line_number}") from exc
        if not isinstance(row, Mapping) or not isinstance(row.get("sentence_name"), str):
            raise PreparationError(f"invalid train timing record at {path}:{line_number}")
        key = str(row["sentence_name"])
        if key in result:
            raise PreparationError(f"duplicate train timing sentence_name: {key}")
        result[key] = row
    if not result:
        raise PreparationError("train timing annotation is empty")
    return result


def _load_seds_split(
    path: Path,
    *,
    split: str,
    timings: Mapping[str, Mapping[str, Any]] | None,
) -> list[_SourceItem]:
    captions = _load_pickle_mapping(path)
    result: list[_SourceItem] = []
    for sentence_id, raw_videos in captions.items():
        if not isinstance(sentence_id, str) or not isinstance(raw_videos, list) or not raw_videos:
            raise PreparationError(f"invalid SEDS {split} sentence record")
        expected_text: str | None = None
        for raw in raw_videos:
            if not isinstance(raw, Mapping):
                raise PreparationError(f"invalid SEDS {split} video record")
            text = raw.get("text")
            stem = raw.get("new_video_name")
            if not isinstance(text, str) or not text.strip() or not isinstance(stem, str) or not stem:
                raise PreparationError(f"incomplete SEDS {split} video record")
            normalized = normalize_text(text)
            if expected_text is not None and normalized != expected_text:
                raise PreparationError(f"SEDS sentence {sentence_id} maps to inconsistent captions")
            expected_text = normalized
            timing = None if timings is None else timings.get(stem)
            if timings is not None and timing is None:
                raise PreparationError(f"missing train timing record for SEDS item {stem}")
            source_id = (
                str(timing["video_id"])
                if timing is not None
                else sentence_id.rsplit("_", 1)[0]
            )
            start = None if timing is None else float(timing["start_time"])
            end = None if timing is None else float(timing["end_time"])
            result.append(
                _SourceItem(split, sentence_id, stem, normalized, source_id, start, end)
            )
    return result


def _load_dev(path: Path) -> list[_SourceItem]:
    try:
        captions = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"cannot load controlled dev annotation: {path}") from exc
    if not isinstance(captions, Mapping) or not captions:
        raise PreparationError("controlled dev annotation must be a nonempty mapping")
    result: list[_SourceItem] = []
    for sentence_id, raw in captions.items():
        if not isinstance(sentence_id, str) or not isinstance(raw, Mapping):
            raise PreparationError("invalid controlled dev record")
        stem = raw.get("sentence_name")
        text = raw.get("text")
        source_id = raw.get("video_id")
        if not all(isinstance(item, str) and item for item in (stem, text, source_id)):
            raise PreparationError(f"incomplete controlled dev record: {sentence_id}")
        result.append(
            _SourceItem(
                "dev",
                sentence_id,
                stem,
                normalize_text(text),
                source_id,
                float(raw["start_time"]),
                float(raw["end_time"]),
            )
        )
    return result


def _relative_paths(item: _SourceItem) -> tuple[str, str, str]:
    split_layout = {
        "train": ("train/raw_videos", "train/train_pose"),
        "dev": ("eval/raw_videos", "eval/eval_pose"),
        "test": ("test/raw_videos", "test/test_pose"),
    }
    video_dir, pose_dir = split_layout[item.split]
    return (
        f"{video_dir}/{item.video_stem}.mp4",
        f"{pose_dir}/{item.video_stem}.pkl",
        f"{item.split}/{item.video_stem}.pkl",
    )


def _probe_asset(item: _SourceItem, root: Path) -> _AssetProbe:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise PreparationError("prepare-data requires the optional OpenCV dependency") from exc
    video_relative, pose_relative, _ = _relative_paths(item)
    video_path = root / video_relative
    pose_path = root / pose_relative
    if not video_path.is_file() or not pose_path.is_file():
        raise PreparationError(f"missing video/pose asset for {item.video_stem}")
    capture = cv2.VideoCapture(str(video_path))
    try:
        frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frames <= 0 or not math.isfinite(fps) or fps <= 0:
        raise PreparationError(f"invalid video timing metadata for {item.video_stem}")
    try:
        with pose_path.open("rb") as handle:
            pose = pickle.load(handle)
        pose_frames = len(pose["keypoints"])
        if len(pose["scores"]) != pose_frames:
            raise ValueError("keypoint/score lengths differ")
    except Exception as exc:
        raise PreparationError(f"invalid RTM pose artifact for {item.video_stem}") from exc
    if frames != pose_frames:
        raise PreparationError(
            f"video/pose frame count mismatch for {item.video_stem}: {frames} != {pose_frames}"
        )
    return _AssetProbe(frames, pose_frames, fps)


def _frame_map(item: _SourceItem, probe: _AssetProbe) -> CompactFrameMap:
    payload = {
        "sample_id": item.video_stem,
        "video_frame_count": probe.video_frames,
        "pose_input_step_count": probe.pose_frames,
        "fps": probe.fps,
        "mapping_policy": "identity_pose_video_frames_v1",
    }
    key = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CompactFrameMap(
        schema_version="compact_frame_map.v1",
        frame_map_key=key,
        sample_id=item.video_stem,
        video_frame_count=probe.video_frames,
        pose_input_step_count=probe.pose_frames,
        fps=probe.fps,
        duration_sec=probe.duration_sec,
        raw_frame_start=0,
        raw_frame_stride=1,
        mapping_policy="identity_pose_video_frames_v1",
    )


def _manifest_record(
    item: _SourceItem,
    frame_map: CompactFrameMap,
    *,
    annotation_provenance: str,
) -> SampleRecord:
    video_path, pose_path, rgb_key = _relative_paths(item)
    return SampleRecord(
        schema_version="sample.v1",
        sample_id=item.video_stem,
        video_id=item.video_stem,
        text_id=item.sentence_id,
        split=item.split,
        sign_language="ase",
        text_language_original="en",
        text_language_model="en",
        text_original=item.text,
        text_model=item.text,
        source_video_id=item.source_video_id,
        signer_id=None,
        source_start_sec=item.source_start_sec,
        source_end_sec=item.source_end_sec,
        duration_sec=frame_map.duration_sec,
        video_path=video_path,
        pose_path=pose_path,
        rgb_feature_key=rgb_key,
        translation_artifact_hash=None,
        frame_map_key=frame_map.frame_map_key,
        annotation_provenance=annotation_provenance,
    )


def _overlap(first: _SourceItem, second: _SourceItem) -> bool:
    if first.source_video_id != second.source_video_id:
        return False
    if first.source_start_sec is None or second.source_start_sec is None:
        return False
    assert first.source_end_sec is not None and second.source_end_sec is not None
    return max(first.source_start_sec, second.source_start_sec) < min(
        first.source_end_sec, second.source_end_sec
    )


def _excluded_negative_rows(items: list[_SourceItem]) -> list[dict[str, Any]]:
    text_groups: dict[str, set[str]] = defaultdict(set)
    source_groups: dict[str, list[_SourceItem]] = defaultdict(list)
    videos_by_text: dict[str, list[str]] = defaultdict(list)
    for item in items:
        text_groups[item.text].add(item.sentence_id)
        source_groups[item.source_video_id].append(item)
        videos_by_text[item.sentence_id].append(item.video_stem)
    reasons: dict[tuple[str, str], set[str]] = defaultdict(set)
    for text_ids in text_groups.values():
        for first, second in combinations(sorted(text_ids), 2):
            for video in videos_by_text[first]:
                reasons[(video, second)].add("exact_normalized_caption_duplicate")
            for video in videos_by_text[second]:
                reasons[(video, first)].add("exact_normalized_caption_duplicate")
    for group in source_groups.values():
        for first, second in combinations(group, 2):
            if first.sentence_id == second.sentence_id or not _overlap(first, second):
                continue
            reasons[(first.video_stem, second.sentence_id)].add("overlapping_source_interval")
            reasons[(second.video_stem, first.sentence_id)].add("overlapping_source_interval")
    return [
        {
            "schema_version": "excluded_negative.v1",
            "video_id": video_id,
            "text_id": text_id,
            "reason": "+".join(sorted(pair_reasons)),
            "provenance": "controlled_annotation_identity_v1",
        }
        for (video_id, text_id), pair_reasons in sorted(reasons.items())
    ]


def prepare_how2sign_data(
    config: Mapping[str, Any], *, workers: int | None = None
) -> dict[str, Any]:
    """Create controlled train/dev/test manifests from pinned SEDS and independent dev labels."""
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise PreparationError("data config must be a mapping")
    if (
        data.get("dataset") != "how2sign"
        or data.get("preparation_protocol") != "seds_how2sign_controlled_v1"
    ):
        raise PreparationError("prepare-data supports only seds_how2sign_controlled_v1")

    def required(field: str, *, directory: bool = False) -> Path:
        value = data.get(field)
        if not isinstance(value, str) or not value:
            raise PreparationError(f"data.{field} is required")
        path = Path(os.path.expanduser(os.path.expandvars(value))).resolve()
        present = path.is_dir() if directory else path.is_file()
        if not present:
            raise PreparationError(f"data.{field} does not exist: {path}")
        return path

    upstream = required("upstream_root", directory=True)
    commit = _validate_seds_checkout(upstream)
    video_root = required("video_root", directory=True)
    pose_root = required("pose_root", directory=True)
    if video_root != pose_root:
        raise PreparationError("controlled How2Sign preparation requires one common asset root")
    annotations = {
        "train": required("train_annotation"),
        "train_timing": required("train_timing_annotation"),
        "dev": required("dev_annotation"),
        "test": required("test_annotation"),
    }
    if annotations["train"] != (upstream / "data_h2" / "train.pkl").resolve():
        raise PreparationError("train annotation is not owned by the pinned SEDS checkout")
    if annotations["test"] != (upstream / "data_h2" / "test.pkl").resolve():
        raise PreparationError("test annotation is not owned by the pinned SEDS checkout")

    timings = _load_train_timing(annotations["train_timing"])
    source_items = {
        "train": _load_seds_split(
            annotations["train"], split="train", timings=timings
        ),
        "dev": _load_dev(annotations["dev"]),
        "test": _load_seds_split(annotations["test"], split="test", timings=None),
    }
    all_items = [item for split in ("train", "dev", "test") for item in source_items[split]]
    sample_ids = [item.video_stem for item in all_items]
    if len(sample_ids) != len(set(sample_ids)):
        raise PreparationError("controlled splits contain duplicate sample/video IDs")
    source_sets = {
        split: {item.source_video_id for item in items} for split, items in source_items.items()
    }
    for first, second in combinations(("train", "dev", "test"), 2):
        if source_sets[first] & source_sets[second]:
            raise PreparationError(f"controlled splits overlap by source video: {first}/{second}")

    worker_count = workers or min(32, os.cpu_count() or 1)
    if worker_count <= 0:
        raise PreparationError("workers must be positive")
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        probes = list(pool.map(lambda item: _probe_asset(item, video_root), all_items))
    probe_by_sample = dict(zip(sample_ids, probes, strict=True))

    resolver = ArtifactResolver(config)
    data_dir = resolver.output_path("shared", "data")
    manifest_dir = data_dir / "manifests"
    relevance_dir = data_dir / "relevance"
    relations_dir = data_dir / "relations"
    frame_maps_dir = data_dir / "frame_maps"
    provenance = {
        split: (
            f"seds_how2sign_controlled_v1:{commit}:"
            f"{_sha256(annotations['train' if split == 'train' else split])}"
        )
        for split in ("train", "dev", "test")
    }
    outputs: dict[str, Path] = {}
    split_reports: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        items = source_items[split]
        frame_maps = [_frame_map(item, probe_by_sample[item.video_stem]) for item in items]
        manifests = [
            _manifest_record(item, frame_map, annotation_provenance=provenance[split])
            for item, frame_map in zip(items, frame_maps, strict=True)
        ]
        manifest_path = manifest_dir / f"{split}.jsonl"
        frame_path = frame_maps_dir / f"{split}.jsonl"
        relevance_path = relevance_dir / f"{split}.jsonl"
        _atomic_jsonl(manifest_path, (record.to_dict() for record in manifests))
        _atomic_jsonl(frame_path, (asdict(record) for record in frame_maps))
        _atomic_jsonl(
            relevance_path,
            (
                {
                    "schema_version": "relevance.v1",
                    "video_id": item.video_stem,
                    "positive_text_ids": [item.sentence_id],
                }
                for item in items
            ),
        )
        outputs[f"{split}_manifest"] = manifest_path
        split_reports[split] = {
            "sample_count": len(items),
            "video_count": len({item.video_stem for item in items}),
            "text_count": len({item.sentence_id for item in items}),
            "source_video_count": len(source_sets[split]),
            "duplicate_text_string_count": len(items) - len({item.text for item in items}),
            "manifest_sha256": _sha256(manifest_path),
            "frame_maps_sha256": _sha256(frame_path),
            "relevance_sha256": _sha256(relevance_path),
        }
    exclusion_rows = _excluded_negative_rows(source_items["train"])
    relations_path = relations_dir / "train_excluded_negatives.jsonl"
    _atomic_jsonl(relations_path, exclusion_rows)
    audit = {
        "schema_version": "how2sign_prepare.v1",
        "ready": True,
        "config_sha256": config_hash(config),
        "protocol": data["preparation_protocol"],
        "upstream": {
            "root": str(upstream),
            "commit": commit,
            "clean": True,
        },
        "annotations": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in annotations.items()
        },
        "splits": split_reports,
        "excluded_negative_count": len(exclusion_rows),
        "asset_probe": {
            "sample_count": len(probes),
            "video_pose_frame_count_equal": True,
            "timing_policy": "container_fps_and_frame_count_v1",
            "stored_pose_video_mapping": "identity_pose_video_frames_v1",
            "native_seds_pose_selection": (
                "captured_by_manifest_input_builder_after_subsampling_and_hand_filter"
            ),
        },
        "released_seds_protocol_note": (
            "upstream has no dev.pkl and selects during training on test.pkl; controlled DIVE uses "
            "the disjoint labels.dev.json as dev and reserves pinned test.pkl for final test"
        ),
        "test_content_used_for_tuning": False,
    }
    audit_path = data_dir / "data_audit.json"
    _atomic_json(audit_path, audit)
    outputs.update(
        {
            "relevance_dir": relevance_dir,
            "train_relations": relations_path,
            "frame_maps_dir": frame_maps_dir,
            "data_audit": audit_path,
        }
    )
    records = resolver.record_stage(
        "prepare_data",
        outputs,
        scope="shared",
        metadata={
            "protocol": data["preparation_protocol"],
            "upstream_commit": commit,
            "sample_count": len(all_items),
        },
    )
    return {
        **audit,
        "outputs": {
            name: {"path": str(outputs[name].resolve()), "artifact_id": record.artifact_id}
            for name, record in records.items()
        },
    }
