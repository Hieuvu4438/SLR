from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from slr_common.upstream.cico_bridge import CiCoBridge, TextEncoding, VideoEncoding


@dataclass
class GalleryEncodings:
    videos: VideoEncoding
    texts: TextEncoding
    video_ids: list[str]
    text_ids: list[str]
    video_to_text: dict[str, list[str]] | None = None
    text_to_video: dict[str, list[str]] | None = None


def _cat_video(values: list[VideoEncoding]) -> VideoEncoding:
    return VideoEncoding(
        *(
            torch.cat([getattr(value, field) for value in values])
            for field in ("mask", "tokens", "cls")
        )
    )


def _cat_text(values: list[TextEncoding]) -> TextEncoding:
    return TextEncoding(
        *(
            torch.cat([getattr(value, field) for value in values])
            for field in ("mask", "tokens", "cls")
        )
    )


@torch.no_grad()
def encode_gallery(model, dataloader, device: torch.device) -> GalleryEncodings:
    model.eval()
    videos: list[VideoEncoding] = []
    texts: list[TextEncoding] = []
    video_ids: list[str] = []
    text_ids: list[str] = []
    video_seen: set[str] = set()
    text_seen: set[str] = set()
    video_to_text: dict[str, list[str]] = {}
    text_to_video: dict[str, list[str]] = {}
    for batch in dataloader:
        h = batch["h"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        video, _ = model.encode_video(h, valid)
        ids, segments, mask = (value.to(device, non_blocking=True) for value in batch["clean_text"])
        text = model.encode_text(ids, segments, mask)
        for index, (video_id, text_id) in enumerate(
            zip(batch["video_id"], batch["caption_id"], strict=True)
        ):
            video_to_text.setdefault(video_id, [])
            text_to_video.setdefault(text_id, [])
            if text_id not in video_to_text[video_id]:
                video_to_text[video_id].append(text_id)
            if video_id not in text_to_video[text_id]:
                text_to_video[text_id].append(video_id)
            if video_id not in video_seen:
                videos.append(
                    VideoEncoding(
                        video.mask[index : index + 1].cpu(),
                        video.tokens[index : index + 1].cpu(),
                        video.cls[index : index + 1].cpu(),
                    )
                )
                video_ids.append(video_id)
                video_seen.add(video_id)
            if text_id not in text_seen:
                texts.append(
                    TextEncoding(
                        text.mask[index : index + 1].cpu(),
                        text.tokens[index : index + 1].cpu(),
                        text.cls[index : index + 1].cpu(),
                    )
                )
                text_ids.append(text_id)
                text_seen.add(text_id)
    return GalleryEncodings(
        _cat_video(videos),
        _cat_text(texts),
        video_ids,
        text_ids,
        video_to_text,
        text_to_video,
    )


@torch.no_grad()
def score_gallery_blockwise(
    bridge: CiCoBridge,
    gallery: GalleryEncodings,
    *,
    device: torch.device,
    dual_mix: float,
    video_block: int,
    text_block: int,
) -> np.ndarray:
    n_video = len(gallery.video_ids)
    n_text = len(gallery.text_ids)
    output = np.empty((n_video, n_text), dtype=np.float32)
    for video_start in range(0, n_video, video_block):
        video_end = min(n_video, video_start + video_block)
        video = VideoEncoding(
            gallery.videos.mask[video_start:video_end].to(device),
            gallery.videos.tokens[video_start:video_end].to(device),
            gallery.videos.cls[video_start:video_end].to(device),
        )
        for text_start in range(0, n_text, text_block):
            text_end = min(n_text, text_start + text_block)
            text = TextEncoding(
                gallery.texts.mask[text_start:text_end].to(device),
                gallery.texts.tokens[text_start:text_end].to(device),
                gallery.texts.cls[text_start:text_end].to(device),
            )
            i2t, t2i = bridge.score(video, text, objective=True)
            output[video_start:video_end, text_start:text_end] = (
                bridge.mixed_score(i2t, t2i, dual_mix).float().cpu().numpy()
            )
    return output
