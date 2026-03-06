#!/usr/bin/env python3
"""
Download / pre-cache the AnyWord-3M dataset from HuggingFace.

The main pipeline (create_dataset_cluster.py) uses HuggingFace streaming by
default, so downloading is optional. Use this script to pre-download the
dataset for faster repeated runs or offline usage.

Usage:
    # Download default subsets (laion + OCR subsets)
    python download_datasets.py --data-dir ~/data/anyword3m

    # Download specific subsets
    python download_datasets.py --data-dir ~/data/anyword3m --subsets laion OCR_COCO_Text

    # List all available subsets
    python download_datasets.py --list-subsets
"""

import argparse
import os
import sys


DATASET_NAME = "stzhao/AnyWord-3M"

ALL_SUBSETS = [
    "laion",
    "OCR_Art",
    "OCR_COCO_Text",
    "OCR_LSVT",
    "OCR_MTWI2018",
    "OCR_ReCTS",
    "OCR_icdar2017rctw",
    "OCR_mlt2019",
    "wukong_1of5",
    "wukong_2of5",
    "wukong_3of5",
    "wukong_4of5",
    "wukong_5of5",
]

DEFAULT_SUBSETS = [
    "laion",
    "OCR_COCO_Text",
    "OCR_mlt2019",
    "OCR_Art",
]


def list_subsets():
    """Print all available subsets."""
    print(f"\nAvailable subsets for {DATASET_NAME}:")
    print("=" * 50)
    for s in ALL_SUBSETS:
        marker = " (default)" if s in DEFAULT_SUBSETS else ""
        print(f"  {s}{marker}")
    print(f"\nTotal: {len(ALL_SUBSETS)} subsets")
    print(f"Default: {len(DEFAULT_SUBSETS)} subsets")


def download_subsets(subsets, data_dir):
    """Download specified subsets using HuggingFace datasets library."""
    from datasets import load_dataset

    os.makedirs(data_dir, exist_ok=True)
    print(f"\nDownloading {len(subsets)} subset(s) to: {data_dir}")
    print(f"Dataset: {DATASET_NAME}")
    print("=" * 50)

    for subset in subsets:
        print(f"\n[{subset}] Downloading ...", flush=True)
        try:
            ds = load_dataset(
                DATASET_NAME,
                subset,
                cache_dir=data_dir,
                trust_remote_code=True,
            )
            for split_name, split_ds in ds.items():
                print(f"  {split_name}: {len(split_ds)} samples", flush=True)
            print(f"  OK", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)

    print("\n" + "=" * 50)
    print("Download complete.")
    print(f"Cache directory: {data_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=f"Download AnyWord-3M dataset ({DATASET_NAME})"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.expanduser("~/data/anyword3m"),
        help="Directory to cache the dataset (default: ~/data/anyword3m)",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        help="Subsets to download (default: laion + OCR subsets)",
    )
    parser.add_argument(
        "--list-subsets",
        action="store_true",
        help="List all available subsets and exit",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download ALL subsets (warning: ~214 GB total)",
    )
    args = parser.parse_args()

    if args.list_subsets:
        list_subsets()
        sys.exit(0)

    subsets = args.subsets
    if subsets is None:
        subsets = ALL_SUBSETS if args.all else DEFAULT_SUBSETS

    # Validate subset names
    for s in subsets:
        if s not in ALL_SUBSETS:
            print(f"ERROR: Unknown subset '{s}'")
            print(f"Valid subsets: {', '.join(ALL_SUBSETS)}")
            sys.exit(1)

    download_subsets(subsets, args.data_dir)


if __name__ == "__main__":
    main()
