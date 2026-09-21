import gzip
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from methods.translation_retrieval.csl_data import (
    UniSignCSLPoseDataset,
    collate_pose_batch,
    select_frames,
    split_pose_parts,
)


class CSLDataTests(unittest.TestCase):
    def test_frame_selection_is_reproducible_and_dev_is_uniform(self):
        dev = select_frames(300, 100, name="clip", seed=7, epoch=0, train=False)
        self.assertEqual((dev[0], dev[-1]), (0, 299))
        self.assertEqual(len(np.unique(dev)), 100)
        train_a = select_frames(300, 100, name="clip", seed=7, epoch=0, train=True)
        train_b = select_frames(300, 100, name="clip", seed=7, epoch=0, train=True)
        train_next = select_frames(300, 100, name="clip", seed=7, epoch=1, train=True)
        np.testing.assert_array_equal(train_a, train_b)
        self.assertFalse(np.array_equal(train_a, train_next))

    def test_part_normalization_and_confidence_mask(self):
        xy = np.zeros((2, 133, 2), dtype=np.float32)
        score = np.ones((2, 133), dtype=np.float32)
        xy[:, 0] = [0, 0]
        xy[:, 3] = [2, 2]
        xy[:, 4] = [1, 1]
        xy[:, 91] = [3, 3]
        xy[:, 92] = [4, 3]
        score[:, 92] = 0.2
        parts = split_pose_parts(xy, score)
        self.assertEqual(parts["body"].shape, (2, 9, 3))
        self.assertEqual(parts["left"].shape, (2, 21, 3))
        self.assertEqual(parts["face_all"].shape, (2, 18, 3))
        self.assertTrue(all(part.dtype == torch.float32 for part in parts.values()))
        self.assertTrue(torch.equal(parts["left"][:, 1], torch.zeros(2, 3)))
        self.assertTrue(torch.isfinite(parts["body"]).all())

    def test_dataset_and_collate_preserve_order_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = {}
            for name, length in (("a", 4), ("b", 2)):
                points = [np.zeros((1, 133, 2), dtype=np.float32) for _ in range(length)]
                scores = [np.ones((1, 133), dtype=np.float32) for _ in range(length)]
                for frame in range(length):
                    points[frame][0, 3] = [2, 2]
                with (root / f"{name}.pkl").open("wb") as stream:
                    pickle.dump({"keypoints": points, "scores": scores}, stream)
                records[name] = {"name": name, "video_path": f"{name}.mp4", "text": name}
            labels = root / "labels.dev"
            with gzip.open(labels, "wb") as stream:
                pickle.dump(records, stream)
            dataset = UniSignCSLPoseDataset(labels, root, split="dev", max_length=3)
            self.assertEqual(len(dataset), 2)
            first = dataset[0]
            second = dataset[1]
            self.assertEqual(first[1]["body"].shape[0], 3)
            names, parts, mask, sentences = collate_pose_batch([first, second])
            self.assertEqual(names, ["a", "b"])
            self.assertEqual(sentences, ["a", "b"])
            self.assertTrue(torch.equal(mask, torch.tensor([[1, 1, 1], [1, 1, 0]])))
            self.assertTrue(torch.equal(parts["body"][1, 2], parts["body"][1, 1]))


if __name__ == "__main__":
    unittest.main()
