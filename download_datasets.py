#!/usr/bin/env python3
"""
Download datasets for the text-in-image generation project.

Downloads TextOCR + TextCaps annotations and images from TextVQA/Flickr.
Both datasets share the same image pool (~28k images).

Datasets:
  - TextOCR: Ground-truth text region annotations (bbox + transcription)
  - TextCaps: Image captions referencing visible text

Usage:
    # Download annotations only (fast, ~500 MB)
    python download_datasets.py --data-dir ~/data

    # Download annotations + images from Flickr
    python download_datasets.py --data-dir ~/data --download-images
"""

import json
import argparse
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlretrieve


# ── Annotation URLs ───────────────────────────────────────────────────────────
ANNOTATION_URLS = {
    "TextOCR_0.1_train.json":
        "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json",
    "TextOCR_0.1_val.json":
        "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json",
    "TextCaps_0.1_train.json":
        "https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_train.json",
    "TextCaps_0.1_val.json":
        "https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_val.json",
}


def download_file(url, dest, desc=""):
    """Download a single file with skip-if-exists logic."""
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  [SKIP] {desc or dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    print(f"  [GET]  {desc or dest.name} ...", end="", flush=True)
    try:
        urlretrieve(url, str(dest))
        print(f" OK ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f" FAIL: {e}")
        return False


def download_one_image(args):
    """Download a single image. Returns (image_id, success, message)."""
    image_id, url, dest = args
    if dest.exists():
        return image_id, True, "cached"
    try:
        urlretrieve(url, str(dest))
        return image_id, True, "ok"
    except Exception as e:
        return image_id, False, str(e)[:60]


def download_images_from_textcaps(json_path, output_dir, workers=20, limit=0):
    """Download images referenced in a TextCaps JSON from Flickr."""
    print(f"\n  Loading URLs from {Path(json_path).name} ...")
    with open(json_path) as f:
        data = json.load(f)

    images = {}
    for entry in data["data"]:
        iid = entry["image_id"]
        if iid not in images:
            url = entry.get("flickr_300k_url") or entry.get("flickr_original_url", "")
            if url:
                images[iid] = url

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for iid, url in images.items():
        dest = output_dir / f"{iid}.jpg"
        if not dest.exists():
            tasks.append((iid, url, dest))

    already = len(images) - len(tasks)
    print(f"  Unique images   : {len(images)}")
    print(f"  Already on disk : {already}")
    print(f"  To download     : {len(tasks)}")

    if limit > 0:
        tasks = tasks[:limit]

    if not tasks:
        print("  Nothing to download.")
        return

    ok = fail = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one_image, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            _, success, _ = future.result()
            ok += success
            fail += (not success)
            if i % 500 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - i) / rate / 60 if rate > 0 else 0
                print(f"    [{i}/{len(tasks)}] ok={ok} fail={fail} "
                      f"{rate:.1f} img/s  ETA {eta:.1f}min")

    total = len(list(output_dir.glob("*.jpg")))
    print(f"  Done: {ok} downloaded, {fail} failed. Total: {total}")


def main():
    parser = argparse.ArgumentParser(
        description="Download TextOCR & TextCaps datasets"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.expanduser("~/data"),
        help="Root data directory (default: ~/data)",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Also download images from Flickr",
    )
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0,
                        help="Max images to download (0 = all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  TextOCR + TextCaps Dataset Downloader")
    print("=" * 55)

    # ── Step 1: Annotation JSONs ──────────────────────────────────────────────
    print("\n[1/2] Downloading annotation files...")
    for filename, url in ANNOTATION_URLS.items():
        dest = data_dir / filename
        download_file(url, dest, desc=filename)

    # ── Step 2: Images ────────────────────────────────────────────────────────
    if args.download_images:
        print("\n[2/2] Downloading images from Flickr...")
        images_dir = data_dir / "train_images"

        for json_name in ["TextCaps_0.1_train.json", "TextCaps_0.1_val.json"]:
            json_path = data_dir / json_name
            if json_path.exists():
                download_images_from_textcaps(
                    json_path, images_dir,
                    workers=args.workers, limit=args.limit,
                )
    else:
        print("\n[2/2] Skipping image download (use --download-images to enable)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Download Summary")
    print("=" * 55)
    for json_file in sorted(data_dir.glob("*.json")):
        size = json_file.stat().st_size
        print(f"  {json_file.name:40s} {size / 1e6:8.1f} MB")

    images_dir = data_dir / "train_images"
    if images_dir.exists():
        count = len(list(images_dir.glob("*.jpg")))
        print(f"  {'train_images/':40s} {count:>5d} images")

    test_dir = data_dir / "test_images"
    if test_dir.exists():
        count = len(list(test_dir.glob("*.jpg")))
        print(f"  {'test_images/':40s} {count:>5d} images")


if __name__ == "__main__":
    main()
