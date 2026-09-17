#!/usr/bin/env python3
"""
Downloads and sets up How2Sign from the user's Google Drive copy.
Uses aria2c for maximum throughput, verifies, extracts, and links directories.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import gdown
from gdown.download import _get_session, get_url_from_gdrive_confirmation

BASE_DIR = Path("/home/haipd/SLR/third_party/SEDS").resolve()
ARIA2C_BIN = "/home/haipd/.local/bin/aria2c"
DOWNLOADS_DIR = BASE_DIR / "downloads"
DATASETS_DIR = BASE_DIR / "datasets"
TARGET_FILE = DOWNLOADS_DIR / "How2Sign.tar.gz"

FILE_ID = "1JLkwC2w1otcSkhWjYSpwM0MtMZ9IHKQu"


def download():
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Starting Download of How2Sign.tar.gz ===")
    print(f"File ID: {FILE_ID}")

    sess, cookies_file = _get_session(
        proxy=None, use_cookies=True, user_agent=None, return_cookies_file=True
    )
    url = f"https://drive.google.com/uc?id={FILE_ID}&export=download"
    res = sess.get(url)
    direct_url = get_url_from_gdrive_confirmation(res.text)
    print(f"[*] Resolved direct URL. Launching aria2c...")

    aria2_cmd = [
        ARIA2C_BIN,
        f"--load-cookies={cookies_file}",
        "--header=User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36",
        "-s", "8",
        "-x", "8",
        "-k", "1M",
        "-c",
        "-d", str(DOWNLOADS_DIR),
        "-o", "How2Sign.tar.gz",
        "--summary-interval=5",
        direct_url,
    ]

    ret = subprocess.run(aria2_cmd)
    if ret.returncode != 0:
        print(f"[!] aria2c returned {ret.returncode}, falling back to gdown...")
        gdown.download(id=FILE_ID, output=str(TARGET_FILE), quiet=False, resume=True)

    if not TARGET_FILE.exists() or TARGET_FILE.stat().st_size < 1000000000:
        raise RuntimeError(f"Download failed! File size: {TARGET_FILE.stat().st_size if TARGET_FILE.exists() else 0}")

    print(f"[+] Download complete: {TARGET_FILE.stat().st_size / 1024 / 1024:.1f} MB")


def extract():
    print(f"\n=== Extracting How2Sign.tar.gz ===")
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["tar", "-xzf", str(TARGET_FILE), "-C", str(DATASETS_DIR)]
    print(f"[*] Executing: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"Extraction failed with exit code {ret.returncode}")
    print("[+] Successfully extracted How2Sign.tar.gz!")


def link():
    print(f"\n=== Configuring Symlinks for How2Sign ===")
    h2s_dir = DATASETS_DIR / "How2Sign"
    root_link = BASE_DIR / "How2Sign"

    if h2s_dir.exists():
        if root_link.is_symlink():
            root_link.unlink()
        elif root_link.is_dir() and not any(root_link.iterdir()):
            root_link.rmdir()
        root_link.symlink_to("datasets/How2Sign", target_is_directory=True)
        print(f"[+] Linked: {root_link} -> datasets/How2Sign")

    # In How2Sign/RTMpose, check if Pose_all_24rates exists or needs link
    rtm_dir = h2s_dir / "RTMpose"
    if rtm_dir.exists() and not (rtm_dir / "Pose_all_24rates").exists():
        (rtm_dir / "Pose_all_24rates").symlink_to(".", target_is_directory=True)
        print(f"[+] Linked: {rtm_dir / 'Pose_all_24rates'} -> .")

    print("[✓] How2Sign symlinks setup complete!")


def main():
    download()
    extract()
    link()
    print("\n=======================================================")
    print("[✓] How2Sign dataset setup fully finished!")
    print("=======================================================")


if __name__ == "__main__":
    main()
