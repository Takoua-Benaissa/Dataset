#!/usr/bin/env python3
"""
Download TextCaps train images from Flickr URLs.
Usage: python download_train_images.py --json ~/data/TextCaps_0.1_train.json \
                                        --output ~/data/train_images/
"""
import json
import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlretrieve
from urllib.error import URLError
import time

def download_one(args):
    image_id, url, dest = args
    if dest.exists():
        return image_id, True, "cached"
    try:
        urlretrieve(url, str(dest))
        return image_id, True, "ok"
    except Exception as e:
        return image_id, False, str(e)[:60]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",    required=True, help="TextCaps JSON file")
    parser.add_argument("--output",  required=True, help="Output directory for images")
    parser.add_argument("--workers", type=int, default=20, help="Parallel workers")
    parser.add_argument("--limit",   type=int, default=0,  help="Max images (0=all)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading annotations from {args.json} ...", flush=True)
    with open(args.json) as f:
        data = json.load(f)

    # Collect unique image_id -> best URL
    images = {}
    for entry in data["data"]:
        iid = entry["image_id"]
        if iid not in images:
            url = entry.get("flickr_300k_url") or entry.get("flickr_original_url", "")
            if url:
                images[iid] = url

    tasks = []
    for iid, url in images.items():
        dest = out_dir / f"{iid}.jpg"
        if not dest.exists():
            tasks.append((iid, url, dest))

    already = len(images) - len(tasks)
    print(f"Total unique images : {len(images)}", flush=True)
    print(f"Already downloaded  : {already}", flush=True)
    print(f"To download         : {len(tasks)}", flush=True)

    if args.limit > 0:
        tasks = tasks[:args.limit]
        print(f"Limited to          : {args.limit}", flush=True)

    if not tasks:
        print("Nothing to do.", flush=True)
        return

    ok = 0
    fail = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            iid, success, msg = future.result()
            if success:
                ok += 1
            else:
                fail += 1
            if i % 500 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(tasks) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(tasks)}] ok={ok} fail={fail}  "
                      f"{rate:.1f} img/s  ETA {eta/60:.1f}min", flush=True)

    elapsed = time.time() - t0
    total_downloaded = len(list(out_dir.glob("*.jpg")))
    print(f"\nDone in {elapsed/60:.1f}min: {ok} downloaded, {fail} failed", flush=True)
    print(f"Total images in {out_dir}: {total_downloaded}", flush=True)

if __name__ == "__main__":
    main()
