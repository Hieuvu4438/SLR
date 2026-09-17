#!/usr/bin/env python3
"""
Automated downloader & dataset setup for SEDS.
Downloads models and feature archives from Google Drive using aria2c multi-connection for maximum speed and stability,
verifies file sizes, extracts dataset archives, and sets up required directory symlinks.
"""

import os
import sys
import time
import shutil
import tarfile
import subprocess
from pathlib import Path
import gdown
from gdown.download import _get_session, get_url_from_gdrive_confirmation, _get_filename_from_response

BASE_DIR = Path("/home/haipd/SLR/third_party/SEDS").resolve()
ARIA2C_BIN = "/home/haipd/.local/bin/aria2c"

DOWNLOADS_DIR = BASE_DIR / "downloads"
CKPT_DIR = BASE_DIR / "ckpt"
MODULES_DIR = BASE_DIR / "modules"
DATASETS_DIR = BASE_DIR / "datasets"

# File definitions
FILES_TO_DOWNLOAD = [
    # Models
    {
        "name": "csl_best_model.bin",
        "id": "1SYu3uHFpVg3f9Rlw8aHBUDWRgodo5xHT",
        "dest_dir": CKPT_DIR,
        "expected_size": 804176411,
        "is_archive": False,
    },
    {
        "name": "h2s_best_model.bin",
        "id": "1PEj8gJVmP9ExQC0D_Pjtd2pKdJVJHEq_",
        "dest_dir": CKPT_DIR,
        "expected_size": 804176411,
        "is_archive": False,
    },
    {
        "name": "ph_best_model.bin",
        "id": "1ngYQxQhCESN20qc5aGAlLnROHFB3p8Du",
        "dest_dir": CKPT_DIR,
        "expected_size": 804176411,
        "is_archive": False,
    },
    {
        "name": "pretrain_signbert.pth",
        "id": "1cX1Nllqxwid6KDPdH8--hypQC6G0sOo-",
        "dest_dir": CKPT_DIR,
        "expected_size": 1673799021,
        "is_archive": False,
    },
    # Datasets
    {
        "name": "PHOENIX-2014-T.tar.gz",
        "id": "1stmSF1llAQFtv9TPbp4KV_RSOoNrrWyV",
        "dest_dir": DOWNLOADS_DIR,
        "expected_size": 2018674802,
        "is_archive": True,
        "extract_target": DATASETS_DIR,
    },
    {
        "name": "CSL.tar.gz",
        "id": "1clYxX1DL-Qxw_T9CT5wV10cJ577auCXg",
        "dest_dir": DOWNLOADS_DIR,
        "expected_size": 4984952190,
        "is_archive": True,
        "extract_target": DATASETS_DIR,
    },
    {
        "name": "How2Sign.tar.gz",
        "id": "1Gfsi4TGRld44lCkj0Rf3Yw3-Wd1PbPvj",
        "dest_dir": DOWNLOADS_DIR,
        "expected_size": 9527117163,
        "is_archive": True,
        "extract_target": DATASETS_DIR,
    },
]


def resolve_gdrive_url(session, file_id, max_retries=5):
    url = f"https://drive.google.com/uc?id={file_id}"
    for attempt in range(1, max_retries + 1):
        try:
            res = session.get(url, stream=True, timeout=30)
            if "Content-Disposition" in res.headers:
                return url, res.headers.get("Content-Length")
            direct_url = get_url_from_gdrive_confirmation(res.text)
            res2 = session.get(direct_url, stream=True, timeout=30)
            return direct_url, res2.headers.get("Content-Length")
        except Exception as e:
            print(f"[Warning] Resolving URL attempt {attempt}/{max_retries} failed: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not resolve direct Google Drive URL for file id {file_id}")


def download_file(item, session, cookies_file):
    dest_dir = item["dest_dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = item["name"]
    target_path = dest_dir / filename
    expected_size = item["expected_size"]

    if target_path.exists() and target_path.stat().st_size == expected_size:
        print(f"[*] {filename} already exists and size matches ({expected_size / 1024 / 1024:.1f} MB). Skipping download.")
        return target_path

    print(f"\n=======================================================")
    print(f"[*] Starting download: {filename} ({expected_size / 1024 / 1024:.1f} MB)")
    print(f"=======================================================")

    direct_url, size_header = resolve_gdrive_url(session, item["id"])
    print(f"[*] Resolved URL for {filename}, starting aria2c...")

    aria2_cmd = [
        ARIA2C_BIN,
        f"--load-cookies={cookies_file}",
        "--header=User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36",
        "-s", "8",
        "-x", "8",
        "-k", "1M",
        "-c",  # resume
        "-d", str(dest_dir),
        "-o", filename,
        "--summary-interval=5",
        direct_url,
    ]

    ret = subprocess.run(aria2_cmd)
    if ret.returncode != 0:
        print(f"[!] aria2c failed with return code {ret.returncode} for {filename}, falling back to gdown...")
        gdown.download(id=item["id"], output=str(target_path), quiet=False, resume=True)

    if not target_path.exists():
        raise FileNotFoundError(f"Download failed: {target_path} not found!")

    cur_size = target_path.stat().st_size
    print(f"[+] Download finished: {filename} (Size: {cur_size / 1024 / 1024:.1f} MB)")
    return target_path


def extract_archive(tar_path, target_dir):
    print(f"\n[*] Extracting {tar_path.name} -> {target_dir} ...")
    target_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["tar", "-xzf", str(tar_path), "-C", str(target_dir)]
    print(f"[*] Executing: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"Failed to extract {tar_path} (return code {ret.returncode})")
    print(f"[+] Successfully extracted {tar_path.name}")


def setup_symlinks():
    print("\n[*] Setting up directory symlinks for full compatibility...")

    # 1. ckpts -> ckpt
    ckpts_link = BASE_DIR / "ckpts"
    if not ckpts_link.exists():
        ckpts_link.symlink_to("ckpt", target_is_directory=True)
        print(f"[+] Created symlink: {ckpts_link} -> ckpt")

    # 2. Datasets links:
    # README expects datasets/{CSL, How2Sign, PHOENIX-2014-T}
    # scripts expect ./CSL, ./How2Sign, ./PHOENIX-2014-T
    for name in ["CSL", "How2Sign", "PHOENIX-2014-T"]:
        dataset_target = DATASETS_DIR / name
        root_link = BASE_DIR / name

        # If extracted in datasets/<name>, link root/<name> -> datasets/<name>
        if dataset_target.exists():
            if not root_link.exists():
                root_link.symlink_to(f"datasets/{name}", target_is_directory=True)
                print(f"[+] Created symlink: {root_link} -> datasets/{name}")
        # If extracted in root/<name>, link datasets/<name> -> root/<name>
        elif root_link.exists() and not dataset_target.exists():
            DATASETS_DIR.mkdir(parents=True, exist_ok=True)
            dataset_target.symlink_to(f"../{name}", target_is_directory=True)
            print(f"[+] Created symlink: {dataset_target} -> ../{name}")

    # Subdirectory alignment for PHOENIX-2014-T
    # eval_ph.sh uses ./PHOENIX-2014-T/RTM_Keypoints/ and ./PHOENIX-2014-T/I3D_features/
    # train_ph.sh uses ./PHOENIX-2014-T/features/RTM_Keypoints/ and ./PHOENIX-2014-T/features/I3D_features/
    ph_dir = BASE_DIR / "PHOENIX-2014-T"
    if ph_dir.exists():
        # Check if features/ exists inside
        ph_features = ph_dir / "features"
        if (ph_dir / "RTM_Keypoints").exists() and not ph_features.exists():
            ph_features.mkdir(exist_ok=True)
            if not (ph_features / "RTM_Keypoints").exists():
                (ph_features / "RTM_Keypoints").symlink_to("../RTM_Keypoints", target_is_directory=True)
            if not (ph_features / "I3D_features").exists():
                (ph_features / "I3D_features").symlink_to("../I3D_features", target_is_directory=True)
            print("[+] Linked PHOENIX-2014-T/features to parent RTM_Keypoints and I3D_features")
        elif ph_features.exists():
            if not (ph_dir / "RTM_Keypoints").exists() and (ph_features / "RTM_Keypoints").exists():
                (ph_dir / "RTM_Keypoints").symlink_to("features/RTM_Keypoints", target_is_directory=True)
            if not (ph_dir / "I3D_features").exists() and (ph_features / "I3D_features").exists():
                (ph_dir / "I3D_features").symlink_to("features/I3D_features", target_is_directory=True)
            print("[+] Linked PHOENIX-2014-T root to internal features/ subdirectories")

    # Subdirectory alignment for CSL
    # CSL/RTMpose vs CSL/RTM_Keypoints
    csl_dir = BASE_DIR / "CSL"
    if csl_dir.exists():
        if (csl_dir / "RTMpose").exists() and not (csl_dir / "RTM_Keypoints").exists():
            (csl_dir / "RTM_Keypoints").symlink_to("RTMpose", target_is_directory=True)
            print("[+] Created symlink: CSL/RTM_Keypoints -> RTMpose")
        elif (csl_dir / "RTM_Keypoints").exists() and not (csl_dir / "RTMpose").exists():
            (csl_dir / "RTMpose").symlink_to("RTM_Keypoints", target_is_directory=True)
            print("[+] Created symlink: CSL/RTMpose -> RTM_Keypoints")

    # Subdirectory alignment for How2Sign
    # scripts use ./How2Sign/RTMpose/Pose_all_24rates/
    h2s_dir = BASE_DIR / "How2Sign"
    if h2s_dir.exists():
        rtm_dir = h2s_dir / "RTMpose"
        if rtm_dir.exists() and not (rtm_dir / "Pose_all_24rates").exists():
            # Check if files are directly under RTMpose
            pkl_files = list(rtm_dir.glob("*.pkl"))
            if pkl_files:
                # Link Pose_all_24rates -> .
                (rtm_dir / "Pose_all_24rates").symlink_to(".", target_is_directory=True)
                print("[+] Created symlink: How2Sign/RTMpose/Pose_all_24rates -> .")


def main():
    print("=== SEDS Automated Download & Setup Pipeline ===")
    start_time = time.time()

    sess, cookies_file = _get_session(
        proxy=None, use_cookies=True, user_agent=None, return_cookies_file=True
    )

    # 1. Download all files
    downloaded_files = []
    for item in FILES_TO_DOWNLOAD:
        path = download_file(item, sess, cookies_file)
        downloaded_files.append((item, path))

    # 2. Extract archives
    for item, path in downloaded_files:
        if item.get("is_archive", False):
            extract_target = item.get("extract_target", DATASETS_DIR)
            extract_archive(path, extract_target)

    # 3. Setup symlinks & compatibility paths
    setup_symlinks()

    elapsed = time.time() - start_time
    print(f"\n=======================================================")
    print(f"[✓] Pipeline complete in {elapsed / 60:.2f} minutes!")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
