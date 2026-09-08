from __future__ import annotations

import math
import random
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import torch
import numpy as np

from dive.data.manifest import SampleRecord

from .seds import SedsTextBatch, SedsVideoBatch
from .seds_reproduction import load_seds_reproduction, verify_seds_checkout


class SedsDataError(ValueError):
    """A manifest row cannot be represented by the pinned native SEDS preprocessing path."""


_RANDOM_STATE_LOCK = threading.Lock()


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
    ) -> None:
        self.upstream_root = verify_seds_checkout(upstream_root)
        self.reproduction = load_seds_reproduction(
            reproduction_config, upstream_root=self.upstream_root
        )
        self.pose_root = Path(pose_root).resolve()
        self.rgb_root = Path(rgb_root).resolve()
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
        try:
            self._tokenizer = tokenizer_module.SimpleTokenizer(str(tokenizer_path))
        except Exception as exc:
            raise SedsDataError(f"cannot initialize pinned SEDS tokenizer: {tokenizer_path}") from exc

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
            raise SedsDataError(f"manifest feature path escapes its root: {record.sample_id}") from exc
        if not pose_path.is_file():
            raise SedsDataError(f"missing SEDS pose feature: {pose_path}")
        if not rgb_path.is_file():
            raise SedsDataError(f"missing SEDS RGB feature: {rgb_path}")
        helper.video_dict = {0: (record.text_id, str(pose_path))}
        helper.video_RGB_dict = {0: (record.text_id, str(rgb_path))}
        return helper

    def _native_sample(self, record: SampleRecord) -> Mapping[str, Any]:
        helper = self._helper(record, require_features=True)
        # Pinned GetTotalFrameList consumes Python RNG even though its sampled offset is unused.
        # Restore the process state so manifest-order preprocessing has no hidden RNG side effect.
        with _RANDOM_STATE_LOCK:
            state = random.getstate()
            try:
                sample = helper[0]
            except Exception as exc:
                raise SedsDataError(
                    f"pinned SEDS preprocessing failed for {record.sample_id}"
                ) from exc
            finally:
                random.setstate(state)
        if not isinstance(sample, Mapping):
            raise SedsDataError("pinned SEDS loader returned a non-mapping sample")
        return sample

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

        samples = [self._native_sample(record) for record in records]
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
            token_type_ids=torch.cat(
                [value["pairs_segment"] for value in encoded], dim=0
            ).long(),
            attention_mask=torch.cat([value["pairs_mask"] for value in encoded], dim=0).long(),
        )
