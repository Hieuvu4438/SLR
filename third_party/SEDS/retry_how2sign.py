#!/usr/bin/env python3
"""
Continuous daemon for downloading and setting up How2Sign dataset from Google Drive.
Handles Google Drive anonymous rate-limit/quota-exceeded by periodic retries,
and allows user override by passing a new Drive File ID or custom URL.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import gdown

BASE_DIR = Path("/home/haipd/SLR/third_party/SEDS").resolve()
DOWNLOADS_DIR = BASE_DIR / "downloads"
DATASETS_DIR = BASE_DIR / "datasets"
TARGET_FILE = DOWNLOADS_DIR / "How2Sign.tar.gz"
EXPECTED_SIZE = 9527117163  # ~8.87 GB

DEFAULT_FILE_ID = "1Gfsi4TGRld44lCkj0Rf3Yw3-Wd1PbPvj"


def try_download(file_id):
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET_FILE.exists() and TARGET_FILE.stat().st_size == EXPECTED_SIZE:
        print(f"[*] How2Sign.tar.gz already downloaded and verified! Size: {TARGET_FILE.stat().st_size / 1024 / 1024:.1f} MB")
        return True

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Attempting download for How2Sign (File ID: {file_id})...")

    try:
        res = gdown.download(id=file_id, output=str(TARGET_FILE), quiet=False, resume=True)
        if res is None or not TARGET_FILE.exists():
            print("[!] gdown returned None or file missing.")
            return False
    except Exception as e:
        print(f"[!] Download attempt error: {e}")
        if TARGET_FILE.exists() and TARGET_FILE.stat().st_size < 100000:
            TARGET_FILE.unlink(missing_ok=True)
        return False

    if TARGET_FILE.exists() and TARGET_FILE.stat().st_size > 1000000:
        print(f"[+] Download succeeded! Size: {TARGET_FILE.stat().st_size / 1024 / 1024:.1f} MB")
        return True
    else:
        if TARGET_FILE.exists():
            TARGET_FILE.unlink(missing_ok=True)
        return False


def extract_and_link():
    print(f"\n[*] Extracting How2Sign.tar.gz to {DATASETS_DIR}...")
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["tar", "-xzf", str(TARGET_FILE), "-C", str(DATASETS_DIR)]
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"[!] Error extracting How2Sign.tar.gz")
        return False

    print("[+] Extraction successful! Setting up symlinks...")
    h2s_dataset = DATASETS_DIR / "How2Sign"
    root_link = BASE_DIR / "How2Sign"
    if h2s_dataset.exists():
        if root_link.is_symlink():
            root_link.unlink()
        elif root_link.is_dir() and not any(root_link.iterdir()):
            root_link.rmdir()
        root_link.symlink_to("datasets/How2Sign", target_is_directory=True)
        print(f"[+] Linked: {root_link} -> datasets/How2Sign")

    rtm_dir = h2s_dataset / "RTMpose"
    if rtm_dir.exists() and not (rtm_dir / "Pose_all_24rates").exists():
        (rtm_dir / "Pose_all_24rates").symlink_to(".", target_is_directory=True)
        print(f"[+] Linked: {rtm_dir / 'Pose_all_24rates'} -> .")

    print("[✓] How2Sign setup completely finished!")
    return True


def main():
    file_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE_ID
    print(f"=== How2Sign Auto-Downloader Daemon (ID: {file_id}) ===")

    while True:
        success = try_download(file_id)
        if success:
            extract_and_link()
            print("[✓] All done!")
            break
        print(f"[*] Shared file quota exceeded on Google Drive. Will retry in 10 minutes...")
        print(f"[*] NOTE: If you make a copy of this file to your personal Google Drive, you can provide its File ID via:")
        print(f"[*]       python retry_how2sign.py <NEW_FILE_ID>")
        time.sleep(600)


if __name__ == "__main__":
    main()
