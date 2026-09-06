from __future__ import annotations

import random
from typing import Any, Sequence

import numpy as np
import torch

from elsc.utils import stable_seed


TEXT_AUGMENTATION_RECIPE = "cico_random_swap_v1"


def encode_cico_text(text: str, tokenizer: Any, max_words: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    words = ["<|startoftext|>", *tokenizer.tokenize(text)]
    if len(words) > max_words - 1:
        indexes = [0, *np.linspace(1, len(words) - 1, max_words - 2, dtype=int).tolist()]
        words = [words[index] for index in indexes]
    words.append("<|endoftext|>")
    ids = tokenizer.convert_tokens_to_ids(words)
    mask = [1] * len(ids)
    ids.extend([0] * (max_words - len(ids)))
    mask.extend([0] * (max_words - len(mask)))
    segments = [0] * max_words
    return torch.tensor(ids), torch.tensor(segments), torch.tensor(mask)


def random_swap_words(text: str, rng: Any = random, swaps: int = 1) -> str:
    words = text.split()
    if len(words) < 2:
        return text
    for _ in range(swaps):
        left = rng.randint(0, len(words) - 1)
        right = left
        counter = 0
        while right == left:
            right = rng.randint(0, len(words) - 1)
            counter += 1
            if counter > 3:
                return " ".join(words)
        words[left], words[right] = words[right], words[left]
    return " ".join(words)


def augment_caption(text: str, pair_id: str, *, seed: int, epoch: int) -> str:
    """Apply the CiCo-compatible one-swap recipe deterministically per sample and epoch."""
    rng = random.Random(stable_seed(seed, epoch, pair_id, TEXT_AUGMENTATION_RECIPE))
    return random_swap_words(text, rng) if rng.random() > 0.5 else text


class CiCoCollator:
    def __init__(self, tokenizer: Any, max_words: int, *, augment: bool = False, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __call__(self, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key in ("pair_id", "video_id", "caption_id", "split", "view_hash"):
            output[key] = [item[key] for item in items]
        for key in ("h", "valid", "dense_index"):
            output[key] = torch.stack([item[key] for item in items])
        if "h_b" in items[0]:
            for key in ("h_b", "valid_b", "dense_index_b"):
                output[key] = torch.stack([item[key] for item in items])
            output["view_b_hash"] = [item["view_b_hash"] for item in items]
            output["view_independent"] = torch.tensor(
                [item["view_independent"] for item in items], dtype=torch.bool
            )
        clean = [encode_cico_text(item["caption"], self.tokenizer, self.max_words) for item in items]
        augmented_text = [
            augment_caption(
                item["caption"], item["pair_id"], seed=self.seed, epoch=self.epoch
            )
            if self.augment
            else item["caption"]
            for item in items
        ]
        augmented = [encode_cico_text(text, self.tokenizer, self.max_words) for text in augmented_text]
        output["clean_text"] = tuple(torch.stack([value[index] for value in clean]) for index in range(3))
        output["aug_text"] = tuple(torch.stack([value[index] for value in augmented]) for index in range(3))
        output["caption"] = [item["caption"] for item in items]
        return output
