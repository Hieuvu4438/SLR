from __future__ import annotations

import copy
import hashlib
import math
import random
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dive.data.manifest import SampleRecord
from dive.data.text_units import (
    SEDS_CLIP_NORMALIZATION_VERSION,
    MappedTextUnit,
    map_units_to_subwords,
    normalize_text,
    unit_mapping_hash,
    unitize,
)

from .seds import SedsTextBatch, SedsVideoBatch
from .seds_reproduction import load_seds_reproduction, verify_seds_checkout


class SedsDataError(ValueError):
    """A manifest row cannot be represented by the pinned native SEDS preprocessing path."""


_RANDOM_STATE_LOCK = threading.Lock()


def hash_seds_input(path: str | Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise SedsDataError(f"native SEDS input is not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SedsTrainingBatch:
    video: SedsVideoBatch
    text: SedsTextBatch
    augmented_text: SedsTextBatch
    augmented_strings: tuple[str, ...]
    augmented: tuple[bool, ...]


@dataclass(frozen=True)
class SedsTextUnitLineage:
    text_id: str
    text_model: str
    token_ids: tuple[int, ...]
    token_offsets: tuple[tuple[int, int] | None, ...]
    units: tuple[MappedTextUnit, ...]
    unit_mapping_sha256: str


def _random_swap(sentence: str, generator: random.Random) -> str:
    # Exact n=1 textaugment.EDA.random_swap/swap_word algorithm called by SEDS,
    # expressed against a caller-owned RNG so epochs and resumes are reproducible.
    words = sentence.split()
    if not words:
        raise SedsDataError("SEDS text augmentation requires nonempty text")
    first = generator.randint(0, len(words) - 1)
    second = first
    counter = 0
    while second == first:
        second = generator.randint(0, len(words) - 1)
        counter += 1
        if counter > 3:
            return " ".join(words)
    words[first], words[second] = words[second], words[first]
    return " ".join(words)


def _bpe_offsets(
    lexical: str,
    pieces: Sequence[str],
    *,
    char_start: int,
    byte_decoder: Mapping[str, int],
) -> tuple[tuple[int, int], ...]:
    """Map byte-level BPE pieces back to covering Unicode character spans."""
    boundaries = [0]
    for character in lexical:
        boundaries.append(boundaries[-1] + len(character.encode("utf-8")))
    raw_pieces: list[bytes] = []
    for piece in pieces:
        core = piece.removesuffix("</w>")
        try:
            raw_pieces.append(bytes(byte_decoder[character] for character in core))
        except KeyError as exc:
            raise SedsDataError("native SEDS BPE piece contains an unknown byte symbol") from exc
    if b"".join(raw_pieces) != lexical.encode("utf-8"):
        raise SedsDataError("native SEDS BPE pieces do not round-trip their lexical token")
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for piece in raw_pieces:
        end = cursor + len(piece)
        left = next(index for index in range(len(lexical)) if boundaries[index + 1] > cursor)
        right = next(
            index + 1
            for index in range(len(lexical))
            if boundaries[index] < end <= boundaries[index + 1]
        )
        offsets.append((char_start + left, char_start + right))
        cursor = end
    return tuple(offsets)


def _load_source_module(path: Path, name: str) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(path)
    try:
        source = path.read_text(encoding="utf-8")
        exec(compile(source, str(path), "exec"), module.__dict__)
    except Exception as exc:
        raise SedsDataError(f"pinned SEDS source module failed to import: {path}") from exc
    return module


class SedsManifestInputBuilder:
    """Route controlled manifest rows through the unmodified pinned SEDS eval transforms."""

    def __init__(
        self,
        *,
        upstream_root: str | Path,
        reproduction_config: str | Path,
        pose_root: str | Path,
        rgb_root: str | Path,
        cache_video_samples: bool = False,
    ) -> None:
        self.upstream_root = verify_seds_checkout(upstream_root)
        self.reproduction = load_seds_reproduction(
            reproduction_config, upstream_root=self.upstream_root
        )
        self.pose_root = Path(pose_root).resolve()
        self.rgb_root = Path(rgb_root).resolve()
        self.cache_video_samples = bool(cache_video_samples)
        self._video_cache: dict[str, tuple[Mapping[str, Any], tuple[int, ...]]] = {}
        if not self.pose_root.is_dir():
            raise SedsDataError(f"SEDS pose root does not exist: {self.pose_root}")
        if not self.rgb_root.is_dir():
            raise SedsDataError(f"SEDS RGB root does not exist: {self.rgb_root}")

        loader_module = _load_source_module(
            self.upstream_root / "dataloaders" / "dataloader_H2_retrieval_pose.py",
            "dive_pinned_seds_h2_eval_loader",
        )
        tokenizer_module = _load_source_module(
            self.upstream_root / "modules" / "tokenization_clip.py",
            "dive_pinned_seds_clip_tokenizer",
        )
        self._loader_class = loader_module.H2_DataLoader_pose
        self._collate = loader_module.H2_pose_collate_fn
        tokenizer_path = self.upstream_root / self.reproduction.external_assets["tokenizer"]
        self.tokenizer_path = tokenizer_path.resolve()
        try:
            self._tokenizer = tokenizer_module.SimpleTokenizer(str(tokenizer_path))
        except Exception as exc:
            raise SedsDataError(
                f"cannot initialize pinned SEDS tokenizer: {tokenizer_path}"
            ) from exc

    def _helper(self, record: SampleRecord, *, require_features: bool) -> Any:
        arguments = self.reproduction.published_eval_arguments
        helper = self._loader_class.__new__(self._loader_class)
        helper.features_path = str(self.pose_root)
        helper.features_RGB_path = str(self.rgb_root)
        helper.max_words = arguments["max_words"]
        helper.tokenizer = self._tokenizer
        helper.feature_len = arguments["feature_len"]
        helper.slide_windows = arguments["slide_windows"]
        helper.windows_stride = arguments["windows_stride"]
        helper.max_length_frames = arguments["max_length_frames"]
        helper.original_size_w = arguments["original_size_w"]
        helper.original_size_h = arguments["original_size_h"]
        helper.threshold = arguments["threshold"]
        helper.frames_threshold = arguments["frames_threshold"]
        helper.crop_img_size = np.array(
            [[arguments["crop_size"], arguments["crop_size"]]], dtype=np.float32
        )
        helper.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }
        expected_model_text = normalize_text(record.text_original, SEDS_CLIP_NORMALIZATION_VERSION)
        if record.text_model != expected_model_text:
            raise SedsDataError(
                f"manifest text_model differs from native SEDS normalization: {record.text_id}"
            )
        helper.sentences_dict = {record.text_id: record.text_model}
        if not require_features:
            return helper
        if record.pose_path is None or record.rgb_feature_key is None:
            raise SedsDataError(f"manifest row lacks native feature paths: {record.sample_id}")
        pose_path = (self.pose_root / record.pose_path).resolve()
        rgb_path = (self.rgb_root / record.rgb_feature_key).resolve()
        try:
            pose_path.relative_to(self.pose_root)
            rgb_path.relative_to(self.rgb_root)
        except ValueError as exc:
            raise SedsDataError(
                f"manifest feature path escapes its root: {record.sample_id}"
            ) from exc
        if not pose_path.is_file():
            raise SedsDataError(f"missing SEDS pose feature: {pose_path}")
        if not rgb_path.is_file():
            raise SedsDataError(f"missing SEDS RGB feature: {rgb_path}")
        helper.video_dict = {0: (record.text_id, str(pose_path))}
        helper.video_RGB_dict = {0: (record.text_id, str(rgb_path))}
        return helper

    def _native_sample(self, record: SampleRecord) -> tuple[Mapping[str, Any], tuple[int, ...]]:
        cached = self._video_cache.get(record.sample_id)
        if cached is not None:
            return cached
        helper = self._helper(record, require_features=True)
        # Pinned GetTotalFrameList consumes Python RNG even though its sampled offset is unused.
        # Restore the process state so manifest-order preprocessing has no hidden RNG side effect.
        with _RANDOM_STATE_LOCK:
            state = random.getstate()
            raw_indices: tuple[int, ...] | None = None
            native_total_frames = helper.GetTotalFrameList

            def capture_total_frames(video_data: Mapping[str, Any], original_size: Any) -> Any:
                nonlocal raw_indices
                selected = native_total_frames(video_data, original_size)
                frame_names = video_data.get("img_list")
                if not isinstance(frame_names, list) or len(frame_names) != len(set(frame_names)):
                    raise SedsDataError("native pose img_list must contain unique frame names")
                index_by_name = {name: index for index, name in enumerate(frame_names)}
                if selected is None or any(name not in index_by_name for name in selected):
                    raise SedsDataError("native selected frames do not map to pose img_list")
                raw_indices = tuple(index_by_name[name] for name in selected)
                if not raw_indices or any(
                    left >= right for left, right in zip(raw_indices, raw_indices[1:])
                ):
                    raise SedsDataError("native selected raw frame indices are not increasing")
                return selected

            helper.GetTotalFrameList = capture_total_frames
            try:
                video_feature, video_mask = helper._get_rawvideo(0)
                sample, sentence_id = helper._get_pose(0)
                if sentence_id != record.text_id:
                    raise SedsDataError("native SEDS loader returned a different text ID")
                sample["RGB"] = video_feature
                # The pinned collate function requires text fields even when the caller only
                # consumes video tensors; preserve that native contract here.
                sample["text"] = helper._get_text(record.text_id)
                if torch.sum(video_mask) != torch.sum(sample["right"]["pose_mask"]):
                    raise SedsDataError("native SEDS RGB and pose masks have different lengths")
            except Exception as exc:
                raise SedsDataError(
                    f"pinned SEDS preprocessing failed for {record.sample_id}"
                ) from exc
            finally:
                random.setstate(state)
        if not isinstance(sample, Mapping):
            raise SedsDataError("pinned SEDS loader returned a non-mapping sample")
        if raw_indices is None:
            raise SedsDataError("pinned SEDS preprocessing did not expose selected frames")
        result = (sample, raw_indices)
        if self.cache_video_samples:
            self._video_cache[record.sample_id] = result
        return result

    def build_video_batch(
        self,
        records: Sequence[SampleRecord],
        *,
        raw_frame_counts: Sequence[int],
        frames_per_second: Sequence[float] | None = None,
        grid_id: str = "canonical",
    ) -> SedsVideoBatch:
        if not records or len(records) != len(raw_frame_counts):
            raise SedsDataError("video records and raw frame counts must be nonempty and aligned")
        if frames_per_second is not None and len(frames_per_second) != len(records):
            raise SedsDataError("video FPS values must align with records")
        if len({record.sample_id for record in records}) != len(records):
            raise SedsDataError("video batch sample IDs must be unique")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in raw_frame_counts
        ):
            raise SedsDataError("raw frame counts must be positive integers")
        if frames_per_second is not None and any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in frames_per_second
        ):
            raise SedsDataError("FPS values must be positive")

        prepared = [self._native_sample(record) for record in records]
        # The pinned collate mutates pose tensors to pad a batch; isolate cached samples.
        samples = [copy.deepcopy(item[0]) for item in prepared]
        pose_raw_frame_indices = tuple(item[1] for item in prepared)
        if any(
            indices[-1] >= raw_count
            for indices, raw_count in zip(pose_raw_frame_indices, raw_frame_counts, strict=True)
        ):
            raise SedsDataError("native selected pose frame exceeds recorded raw frame count")
        try:
            batch = self._collate(samples)
        except Exception as exc:
            raise SedsDataError("pinned SEDS batch collation failed") from exc
        required = {
            "right_pose",
            "right_clips_start",
            "left_pose",
            "left_clips_start",
            "body_pose",
            "body_mask",
            "body_clips_start",
            "RGB_feature",
        }
        if not isinstance(batch, Mapping) or not required <= set(batch):
            raise SedsDataError("pinned SEDS collate output is incomplete")
        clip_starts = batch["body_clips_start"]
        if not torch.equal(batch["right_clips_start"], clip_starts) or not torch.equal(
            batch["left_clips_start"], clip_starts
        ):
            raise SedsDataError("pinned SEDS pose streams produced different temporal grids")
        feature_len = self.reproduction.model_arguments["feature_len"]
        if batch["body_mask"].shape != (len(records), feature_len + 1):
            raise SedsDataError("pinned SEDS body mask has an unexpected shape")
        if batch["RGB_feature"].shape != (len(records), 1024, feature_len, 1):
            raise SedsDataError("pinned SEDS RGB features have an unexpected shape")
        if not torch.isfinite(batch["RGB_feature"]).all():
            raise SedsDataError("pinned SEDS RGB features contain NaN/Inf")
        valid_clip_counts = (batch["body_mask"] == 0).sum(dim=1) - 1
        expected_clip_counts = torch.tensor(
            [
                min(
                    feature_len,
                    max(
                        1,
                        math.ceil(
                            (len(indices) - self.reproduction.model_arguments["slide_windows"])
                            / self.reproduction.model_arguments["windows_stride"]
                        )
                        + 1,
                    ),
                )
                for indices in pose_raw_frame_indices
            ],
            dtype=valid_clip_counts.dtype,
        )
        if not torch.equal(valid_clip_counts.cpu(), expected_clip_counts):
            raise SedsDataError("native SEDS mask count differs from selected pose geometry")
        return SedsVideoBatch(
            sample_ids=tuple(record.sample_id for record in records),
            right_pose=batch["right_pose"],
            left_pose=batch["left_pose"],
            body_pose=batch["body_pose"],
            clip_starts=clip_starts,
            legacy_video_mask=batch["body_mask"],
            rgb_features=batch["RGB_feature"],
            grid_id=grid_id,
            raw_frame_counts=tuple(raw_frame_counts),
            frames_per_second=(
                None
                if frames_per_second is None
                else tuple(float(value) for value in frames_per_second)
            ),
            pose_raw_frame_indices=pose_raw_frame_indices,
        )

    def build_text_batch(self, records: Sequence[SampleRecord]) -> SedsTextBatch:
        if not records or len({record.text_id for record in records}) != len(records):
            raise SedsDataError("text batch must contain unique text IDs")
        encoded: list[Mapping[str, torch.Tensor]] = []
        for record in records:
            helper = self._helper(record, require_features=False)
            value = helper._get_text(record.text_id)
            if not isinstance(value, Mapping):
                raise SedsDataError("pinned SEDS tokenizer returned a non-mapping sample")
            encoded.append(value)
        return SedsTextBatch(
            text_ids=tuple(record.text_id for record in records),
            input_ids=torch.cat([value["pairs_text"] for value in encoded], dim=0).long(),
            token_type_ids=torch.cat([value["pairs_segment"] for value in encoded], dim=0).long(),
            attention_mask=torch.cat([value["pairs_mask"] for value in encoded], dim=0).long(),
        )

    def build_text_unit_lineage(
        self, records: Sequence[SampleRecord]
    ) -> tuple[SedsTextUnitLineage, ...]:
        """Instrument the pinned regex/BPE/token selection path and bind exact unit indices."""
        native = self.build_text_batch(records)
        maximum = int(self.reproduction.published_eval_arguments["max_words"])
        result: list[SedsTextUnitLineage] = []
        for row, record in enumerate(records):
            full_pieces: list[tuple[str, tuple[int, int] | None, int]] = [
                ("<|startoftext|>", None, 0)
            ]
            for match in self._tokenizer.pat.finditer(record.text_model):
                lexical = match.group(0)
                encoded = "".join(
                    self._tokenizer.byte_encoder[value] for value in lexical.encode("utf-8")
                )
                bpe_pieces = self._tokenizer.bpe(encoded).split(" ")
                bpe_offsets = _bpe_offsets(
                    lexical,
                    bpe_pieces,
                    char_start=match.start(),
                    byte_decoder=self._tokenizer.byte_decoder,
                )
                first_index = len(full_pieces)
                full_pieces.extend(
                    (piece, offset, first_index + index)
                    for index, (piece, offset) in enumerate(
                        zip(bpe_pieces, bpe_offsets, strict=True)
                    )
                )
            if [piece for piece, _, _ in full_pieces[1:]] != self._tokenizer.tokenize(
                record.text_model
            ):
                raise SedsDataError("instrumented SEDS BPE pieces differ from native tokenizer")
            total_with_cls = maximum - 1
            selected_indices = list(range(len(full_pieces)))
            if len(full_pieces) > total_with_cls:
                selected_indices = [0]
                selected_indices.extend(
                    int(value)
                    for value in np.linspace(
                        1,
                        len(full_pieces) - 1,
                        total_with_cls - 1,
                        dtype=int,
                    )
                )
            pieces = [full_pieces[index] for index in selected_indices]
            pieces.append(("<|endoftext|>", None, -1))
            token_ids = self._tokenizer.convert_tokens_to_ids([piece for piece, _, _ in pieces])
            offsets = [offset for _, offset, _ in pieces]
            mask = [1] * len(token_ids)
            while len(token_ids) < maximum:
                token_ids.append(0)
                offsets.append(None)
                mask.append(0)
            if len(token_ids) != maximum or len(offsets) != maximum:
                raise SedsDataError("instrumented SEDS token sequence exceeds native maximum")
            if (
                token_ids != native.input_ids[row].tolist()
                or mask != native.attention_mask[row].tolist()
            ):
                raise SedsDataError("instrumented token IDs/mask differ from native SEDS loader")
            preliminary = map_units_to_subwords(unitize(record.text_model), token_ids, offsets)
            selected_set = set(selected_indices)
            mapped: tuple[MappedTextUnit, ...] = tuple(
                replace(
                    item,
                    complete_after_truncation=(
                        item.complete_after_truncation
                        and {
                            original_index
                            for _, offset, original_index in full_pieces
                            if offset is not None
                            and offset[0] < item.unit.char_end
                            and offset[1] > item.unit.char_start
                        }
                        <= selected_set
                    ),
                )
                for item in preliminary
            )
            result.append(
                SedsTextUnitLineage(
                    text_id=record.text_id,
                    text_model=record.text_model,
                    token_ids=tuple(token_ids),
                    token_offsets=tuple(offsets),
                    units=mapped,
                    unit_mapping_sha256=unit_mapping_hash(mapped),
                )
            )
        return tuple(result)

    def build_training_batch(
        self,
        records: Sequence[SampleRecord],
        *,
        raw_frame_counts: Sequence[int],
        frames_per_second: Sequence[float] | None,
        generator: random.Random,
    ) -> SedsTrainingBatch:
        """Build the published random-swap objective with an explicit caller-owned RNG."""
        if not isinstance(generator, random.Random):
            raise SedsDataError("training augmentation requires a dedicated random.Random")
        video = self.build_video_batch(
            records,
            raw_frame_counts=raw_frame_counts,
            frames_per_second=frames_per_second,
        )
        text = self.build_text_batch(records)
        augmented_strings: list[str] = []
        flags: list[bool] = []
        augmented_records: list[SampleRecord] = []
        for record in records:
            should_augment = generator.random() > 0.5
            value = (
                _random_swap(record.text_model, generator) if should_augment else record.text_model
            )
            augmented_strings.append(value)
            flags.append(should_augment and value != record.text_model)
            augmented_records.append(replace(record, text_original=value, text_model=value))
        augmented_text = self.build_text_batch(augmented_records)
        return SedsTrainingBatch(
            video=video,
            text=text,
            augmented_text=augmented_text,
            augmented_strings=tuple(augmented_strings),
            augmented=tuple(flags),
        )
