"""Import the untouched upstream SEDS code as a library (single-process, world size 1).

Upstream `main_task_retrieval.py` calls `init_process_group` and parses sys.argv at import /
get_args() time, and resolves several assets by relative path. This module isolates those
side effects so our own entrypoints can reuse upstream model/dataloader/eval code verbatim.
"""
import os
import sys

SEDS_DIR = "/home/haipd/SLR/third_party/SEDS"


def load_upstream(argv):
    """argv: list of upstream CLI args. Returns (upstream_module, args)."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 400))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.chdir(SEDS_DIR)
    if SEDS_DIR not in sys.path:
        sys.path.insert(0, SEDS_DIR)
    saved = sys.argv
    sys.argv = ["main_task_retrieval.py"] + list(argv)
    try:
        import main_task_retrieval as up  # noqa: E402  (performs init_process_group)
        args = up.get_args()
    finally:
        sys.argv = saved
    return up, args


def seds_argv(dataset, split_dir=None, extra=()):
    """Upstream hyperparameters from scripts/train_*.sh; data paths point at our split."""
    common = ["--signbert", "--init_sign_model", "ckpt/pretrain_signbert.pth",
              "--fusion_type", "gloss_atten", "--rgb_pose_match", "--rgb_pose_match_loss", "0.4",
              "--lr", "1e-5", "--sign_lr", "1e-4", "--max_words", "32", "--feature_len", "64",
              "--max_length_frames", "300", "--slide_windows", "16", "--windows_stride", "1",
              "--crop_size", "256", "--frames_threshold", "0.1", "--threshold", "0.4",
              "--batch_size_val", "64", "--coef_lr", "1.", "--freeze_layer_num", "0",
              "--linear_patch", "2d", "--sim_header", "Filip", "--pretrained_clip_name", "ViT-B/32"]
    per = {
        "ph": ["--features_path", "./PHOENIX-2014-T/RTM_Keypoints/", "--datatype", "ph_pose",
               "--data_path", "data_ph", "--features_RGB_path", "./PHOENIX-2014-T/I3D_features/"],
        "csl": ["--features_path", "./CSL/RTM_Keypoints/", "--datatype", "csl_pose", "--original_size", "512",
                "--data_path", "data_csl", "--features_RGB_path", "./CSL/I3D_features/"],
        "h2s": ["--features_path", "./How2Sign/RTMpose/Pose_all_24rates/", "--datatype", "h2s_pose",
                "--original_size_w", "256", "--original_size_h", "256",
                "--data_path", "data_h2", "--features_RGB_path", "./How2Sign/I3D_features/"],
    }[dataset]
    if split_dir is not None:
        per = per[:-4] + ["--data_path", os.path.join(split_dir, "data"),
                          "--features_RGB_path", os.path.join(split_dir, "rgb")]
    return common + per + list(extra)
