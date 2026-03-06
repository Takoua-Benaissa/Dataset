"""
Text-in-Image Dataset Creation Pipeline  v8
============================================
Loads the AnyWord-3M dataset (from HuggingFace: stzhao/AnyWord-3M) and filters
for short, readable English text with high OCR confidence. Uses dataset captions
when available. Verifies readability with Tesseract OCR, then exports
train / val / test splits.

Key filtering criteria
──────────────────────
  * Language        : English only (annotation language field + langdetect)
  * Phrase length   : 1–5 words (< 6 words)
  * OCR gate        : Tesseract confidence >= 85 %
  * Density filter  : reject dense text images (books, newspapers)
  * Bbox height     : >= 30 px
  * Image quality   : min 256 px, sharpness >= 80, contrast >= 20
  * Target          : 1000 images
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

# Set Tesseract data path
for _td in [
    os.path.expanduser("~/miniconda3/share/tessdata"),
    "/usr/share/tesseract-ocr/5/tessdata",
    "/usr/share/tesseract-ocr/4.00/tessdata",
    "/usr/share/tessdata",
]:
    if os.path.isfile(os.path.join(_td, "eng.traineddata")):
        os.environ.setdefault("TESSDATA_PREFIX", _td)
        break

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    _LANGDETECT_OK = True
except ImportError:
    _LANGDETECT_OK = False

print("[IMPORTS_DONE]", flush=True)


# ─── Constants ────────────────────────────────────────────────────────────────

VERSION = "8.0"

# Image quality
MIN_RESOLUTION  = 256
MIN_SHARPNESS   = 80
BRIGHTNESS_MIN  = 30
BRIGHTNESS_MAX  = 230
MIN_CONTRAST    = 20

# Text filters
MAX_WORDS        = 5
MIN_TEXT_LEN     = 1
MAX_TEXT_LEN     = 40

# Density filter (per image) — reject dense-text scenes
MAX_TEXT_REGIONS = 4         # max distinct valid text annotations per image
MAX_TOTAL_CHARS  = 40        # max total chars across all valid annotations

# Bbox minimum
MIN_BBOX_HEIGHT  = 30   # pixels

# OCR gate
MIN_OCR_CONF     = 85
MIN_OCR_COVERAGE = 0.3

# Run target
TARGET_IMAGES    = 1000

# AnyWord-3M HuggingFace dataset
DATASET_NAME = "stzhao/AnyWord-3M"

# Subsets to load — these have the most English content
# laion is the largest (~1.72M) and mostly English/multilingual
# OCR subsets also contain English text
DATASET_SUBSETS = [
    "laion",
    "OCR_COCO_Text",
    "OCR_mlt2019",
    "OCR_Art",
]


# ─── Text utilities ───────────────────────────────────────────────────────────

def is_valid_text(text: str) -> bool:
    """Accept short, letter-containing English phrases (1-5 words)."""
    text = text.strip()
    if not (MIN_TEXT_LEN <= len(text) <= MAX_TEXT_LEN):
        return False
    if len(text.split()) > MAX_WORDS:
        return False
    if not re.search(r"[A-Za-z]", text):
        return False
    # Reject if text contains non-ASCII letters (Chinese, Arabic, etc.)
    if re.search(r"[^\x00-\x7F]", text):
        return False
    return True


def is_english(text: str) -> bool:
    """Return True if text is detected as English."""
    # Quick check: all ASCII characters
    if all(ord(c) < 128 for c in text):
        if not re.search(r"[A-Za-z]", text):
            return False
        return True
    # Non-ASCII present → not English for our purposes
    return False


def is_english_langdetect(text: str) -> bool:
    """Use langdetect as a secondary check for English."""
    if not _LANGDETECT_OK:
        return True
    try:
        return detect(text) == "en"
    except Exception:
        return True


# ─── Annotation filtering for AnyWord-3M ─────────────────────────────────────

def extract_english_texts(annotations: List[Dict]) -> List[Dict]:
    """
    From AnyWord-3M annotations, extract valid English text entries.

    Each annotation has: text, polygon, language, valid, (illegibility, pos, rec_score)
    Returns filtered list of annotations that pass English + validity checks.
    """
    results = []
    for ann in annotations:
        # Skip invalid annotations
        if not ann.get("valid", True):
            continue
        # Skip illegible text
        if ann.get("illegibility", False):
            continue

        text = ann.get("text", "").strip()
        if not text:
            continue

        # Check language field from dataset (quick filter)
        lang = ann.get("language", "").lower()
        if lang and lang not in ("latin", "english", ""):
            continue

        # Check text validity (length, word count, ASCII)
        if not is_valid_text(text):
            continue

        # Check English with langdetect for multi-word texts
        if len(text.split()) > 1 and not is_english_langdetect(text):
            continue

        # Get bounding box from polygon
        polygon = ann.get("polygon", [])
        if not polygon or len(polygon) < 3:
            continue

        # Convert polygon to bounding box [x, y, w, h]
        poly = np.array(polygon)
        x_min, y_min = poly.min(axis=0)
        x_max, y_max = poly.max(axis=0)
        bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

        # Check bbox height
        if bbox[3] < MIN_BBOX_HEIGHT:
            continue

        results.append({
            "text": text,
            "bbox": bbox,
            "polygon": polygon,
            "language": lang,
        })

    return results


def is_dense_text_image(annotations: List[Dict]) -> bool:
    """
    Return True if this is a dense-text scene (book page, newspaper, etc.)
    that should be rejected.
    """
    valid = extract_english_texts(annotations)
    # Also count non-English valid annotations to detect total text density
    total_valid = sum(
        1 for ann in annotations
        if ann.get("valid", True)
        and not ann.get("illegibility", False)
        and len(ann.get("text", "").strip()) >= MIN_TEXT_LEN
    )
    if total_valid > MAX_TEXT_REGIONS * 2:
        return True
    if len(valid) > MAX_TEXT_REGIONS:
        return True
    total_chars = sum(len(ann.get("text", "")) for ann in annotations
                      if ann.get("valid", True))
    if total_chars > MAX_TOTAL_CHARS * 2:
        return True
    return False


def select_best_text(english_texts: List[Dict]) -> Optional[Dict]:
    """
    From filtered English text entries, pick the best one.
    Prefers multi-word phrases, then longer text.
    """
    if not english_texts:
        return None
    scored = []
    for entry in english_texts:
        text = entry["text"]
        word_count = len(text.split())
        # Prefer multi-word phrases, but also value length
        score = len(text) + word_count * 5
        scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


# ─── OCR gate ─────────────────────────────────────────────────────────────────

def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    w, h = img.size
    scale = max(1, 200 // min(w, h))
    if scale > 1:
        img = img.resize((w * scale, h * scale), Image.LANCZOS)
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    return img


def verify_text_with_ocr(
    image: Image.Image,
    expected_text: Optional[str] = None,
    bbox: Optional[List[int]] = None,
) -> Tuple[Optional[str], float]:
    """
    Run Tesseract on a PIL Image and verify expected_text is readable.
    Returns (ocr_text, confidence) or (None, 0) on failure.
    """
    try:
        img = image.convert("RGB")

        if bbox is not None:
            iw, ih = img.size
            x, y, w, h = [int(v) for v in bbox]
            pad = max(5, int(h * 0.15))
            crop = img.crop((max(0, x - pad), max(0, y - pad),
                             min(iw, x + w + pad), min(ih, y + h + pad)))
        else:
            crop = img

        crop = _preprocess_for_ocr(crop)

        def run_ocr(src, psm):
            cfg = f"--psm {psm} --oem 1 -l eng"
            d = pytesseract.image_to_data(src, config=cfg,
                      output_type=pytesseract.Output.DICT)
            out = []
            for i, t in enumerate(d["text"]):
                t = t.strip()
                if not t:
                    continue
                try:
                    c = int(d["conf"][i])
                except (ValueError, TypeError):
                    c = 0
                if c >= MIN_OCR_CONF and len(t) >= 2:
                    out.append((t, c))
            return out

        tokens: List[Tuple[str, int]] = []
        for src, psms in [(crop, [7, 6]), (img, [11, 3])]:
            for psm in psms:
                tokens.extend(run_ocr(src, psm))
            if tokens:
                break

        if not tokens:
            return None, 0

        if expected_text:
            exp = expected_text.lower().strip()
            matching = []
            for tok, conf in tokens:
                cov = len(tok) / max(len(exp), 1)
                if tok.lower() in exp or exp in tok.lower() or cov >= MIN_OCR_COVERAGE:
                    matching.append((tok, conf))
            if not matching:
                return None, 0
            matching.sort(key=lambda x: x[1], reverse=True)
            return matching[0]

        tokens.sort(key=lambda x: x[1], reverse=True)
        return tokens[0]

    except Exception:
        return None, 0


# ─── Image quality ────────────────────────────────────────────────────────────

def check_image_quality(img_array: np.ndarray) -> Tuple[bool, Dict]:
    """Check image quality from a numpy array (BGR or RGB)."""
    try:
        if img_array is None:
            return False, {}
        h, w = img_array.shape[:2]
        if h < MIN_RESOLUTION or w < MIN_RESOLUTION:
            return False, {}
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) \
               if img_array.shape[2] == 3 else img_array
        sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        contrast   = float(np.std(gray))
        if sharpness < MIN_SHARPNESS:
            return False, {}
        if not (BRIGHTNESS_MIN <= brightness <= BRIGHTNESS_MAX):
            return False, {}
        if contrast < MIN_CONTRAST:
            return False, {}
        return True, {
            "sharpness":  round(sharpness, 2),
            "brightness": round(brightness, 2),
            "contrast":   round(contrast, 2),
            "resolution": [int(w), int(h)],
        }
    except Exception:
        return False, {}


# ─── Dataset creator ──────────────────────────────────────────────────────────

class DatasetCreator:

    def __init__(self, output_dir: str, max_images: int = TARGET_IMAGES,
                 subsets: Optional[List[str]] = None, streaming: bool = True):
        self.base_dir = Path(output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(
            [d for d in self.base_dir.glob("v[0-9]*") if d.is_dir()],
            key=lambda d: int(d.name[1:]))
        version      = (int(existing[-1].name[1:]) + 1) if existing else 1
        self.out_dir = self.base_dir / f"v{version}"
        print(f"[DIR] Output -> {self.out_dir}", flush=True)

        for split in ("train", "val", "test"):
            (self.out_dir / split / "images").mkdir(parents=True, exist_ok=True)

        self.max_images = max_images
        self.subsets    = subsets or DATASET_SUBSETS
        self.streaming  = streaming
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DEVICE] {self.device}", flush=True)
        if self.device == "cuda":
            print(f"[GPU] {torch.cuda.get_device_name(0)}", flush=True)

    def load_dataset(self):
        """Load AnyWord-3M from HuggingFace."""
        from datasets import load_dataset

        print(f"[DATASET] Loading {DATASET_NAME} from HuggingFace ...", flush=True)
        print(f"[DATASET] Subsets: {self.subsets}", flush=True)
        print(f"[DATASET] Streaming: {self.streaming}", flush=True)

        all_datasets = []
        for subset in self.subsets:
            print(f"  Loading subset: {subset} ...", flush=True)
            try:
                ds = load_dataset(DATASET_NAME, subset, streaming=self.streaming,
                                  trust_remote_code=True)
                if "train" in ds:
                    all_datasets.append((subset, ds["train"]))
                    print(f"  {subset}: loaded", flush=True)
                else:
                    # Some subsets may have different split names
                    for split_name in ds:
                        all_datasets.append((subset, ds[split_name]))
                        print(f"  {subset}/{split_name}: loaded", flush=True)
                        break
            except Exception as e:
                print(f"  WARNING: Failed to load {subset}: {e}", flush=True)

        if not all_datasets:
            raise RuntimeError("No dataset subsets could be loaded!")

        print(f"[DATASET] {len(all_datasets)} subset(s) ready", flush=True)
        return all_datasets

    def process_sample(self, sample: Dict, subset_name: str) -> Optional[Dict]:
        """
        Process a single AnyWord-3M sample.
        Returns a record dict if accepted, None if rejected.
        """
        annotations = sample.get("annotations", [])
        if not annotations:
            return None

        # Density filter — reject dense text scenes
        if is_dense_text_image(annotations):
            return None

        # Extract valid English texts
        english_texts = extract_english_texts(annotations)
        if not english_texts:
            return None

        # Select best text
        best = select_best_text(english_texts)
        if not best:
            return None

        # Get the image
        pil_image = sample.get("image")
        if pil_image is None:
            return None
        pil_image = pil_image.convert("RGB")

        # Image quality check
        img_array = np.array(pil_image)
        ok, metrics = check_image_quality(img_array)
        if not ok:
            return None

        # OCR verification
        anno_text = best["text"]
        anno_bbox = best["bbox"]
        ocr_text, ocr_conf = verify_text_with_ocr(
            pil_image, expected_text=anno_text, bbox=anno_bbox)
        if ocr_text is None:
            return None

        # Get caption from dataset
        caption = sample.get("caption", "").strip()
        if not caption:
            caption = "a photograph"

        # Build prompt
        final_text = anno_text
        prompt = (f"{caption}, "
                  f"with the text \u201c{final_text}\u201d clearly visible, "
                  f"sharp focus, high resolution photography")

        return {
            "image":           pil_image,
            "image_id":        sample.get("img_name", "unknown"),
            "text":            final_text,
            "annotation_text": final_text,
            "ocr_text":        ocr_text,
            "caption":         caption,
            "prompt":          prompt,
            "subset":          subset_name,
            "metadata": {
                "ocr_confidence": float(ocr_conf),
                "text_length":    len(final_text),
                "word_count":     len(final_text.split()),
                "source":         f"AnyWord-3M/{subset_name}",
                **metrics,
            },
        }

    def run(self) -> List[Dict]:
        """Main processing loop — stream through the dataset and filter."""
        all_datasets = self.load_dataset()

        accepted: List[Dict] = []
        stats = {
            "total_seen": 0,
            "rej_no_annotation": 0,
            "rej_density": 0,
            "rej_no_english": 0,
            "rej_quality": 0,
            "rej_ocr": 0,
        }

        print(f"\n[PROC] Starting filtering (target: {self.max_images}) ...",
              flush=True)

        for subset_name, ds in all_datasets:
            if len(accepted) >= self.max_images:
                break

            print(f"\n[SUBSET] Processing: {subset_name}", flush=True)

            for sample in tqdm(ds, desc=f"{subset_name}", disable=False):
                if len(accepted) >= self.max_images:
                    break

                stats["total_seen"] += 1

                result = self.process_sample(sample, subset_name)

                if result is None:
                    # Count rejection reason (approximate — process_sample
                    # combines multiple checks)
                    annotations = sample.get("annotations", [])
                    if not annotations:
                        stats["rej_no_annotation"] += 1
                    elif is_dense_text_image(annotations):
                        stats["rej_density"] += 1
                    elif not extract_english_texts(annotations):
                        stats["rej_no_english"] += 1
                    else:
                        # Either quality or OCR failure
                        stats["rej_quality"] += 1
                    continue

                accepted.append(result)

                if len(accepted) % 50 == 0:
                    print(f"  ACCEPTED {len(accepted):4d}/{self.max_images}  "
                          f"(seen={stats['total_seen']}  "
                          f"text='{result['text']}')", flush=True)

        print(f"\n[SUMMARY]", flush=True)
        print(f"  Total seen     : {stats['total_seen']}", flush=True)
        print(f"  Accepted       : {len(accepted)}", flush=True)
        print(f"  Rej no-annot   : {stats['rej_no_annotation']}", flush=True)
        print(f"  Rej density    : {stats['rej_density']}", flush=True)
        print(f"  Rej no-english : {stats['rej_no_english']}", flush=True)
        print(f"  Rej quality/OCR: {stats['rej_quality']}", flush=True)

        return accepted

    def save(self, records: List[Dict]):
        if not records:
            print("[WARN] No records to save.", flush=True)
            return

        n       = len(records)
        n_train = int(n * 0.8)
        n_val   = int(n * 0.1)
        splits  = [
            ("train", records[:n_train]),
            ("val",   records[n_train:n_train + n_val]),
            ("test",  records[n_train + n_val:]),
        ]

        print("[SAVE] Saving images and writing annotations ...", flush=True)
        all_records: List[Dict] = []
        global_id = 0

        for split_name, split_records in splits:
            split_data: List[Dict] = []
            for rec in split_records:
                filename = f"img_{global_id:04d}.jpg"
                dst = self.out_dir / split_name / "images" / filename

                # Save PIL image to disk
                pil_img = rec["image"]
                pil_img.save(str(dst), "JPEG", quality=95)

                entry = {
                    "id":               global_id,
                    "image_id":         rec["image_id"],
                    "filename":         filename,
                    "filepath":         f"{split_name}/images/{filename}",
                    "split":            split_name,
                    "text":             rec["text"],
                    "annotation_text":  rec["annotation_text"],
                    "ocr_text":         rec["ocr_text"],
                    "caption":          rec["caption"],
                    "prompt":           rec["prompt"],
                    "metadata":         rec["metadata"],
                }
                split_data.append(entry)
                all_records.append(entry)
                global_id += 1
            with open(self.out_dir / f"{split_name}.json", "w") as fh:
                json.dump({"split": split_name,
                           "count": len(split_data),
                           "data": split_data}, fh, indent=2)

        with open(self.out_dir / "dataset_complete.json", "w") as fh:
            json.dump({
                "metadata": {
                    "version":     VERSION,
                    "description": (
                        "Text-in-image dataset built from AnyWord-3M. "
                        "Filtered for short English text, OCR-verified."),
                    "source_dataset": DATASET_NAME,
                    "subsets_used": self.subsets,
                    "thresholds": {
                        "min_resolution":   MIN_RESOLUTION,
                        "min_sharpness":    MIN_SHARPNESS,
                        "min_ocr_conf":     MIN_OCR_CONF,
                        "min_ocr_coverage": MIN_OCR_COVERAGE,
                        "max_text_len":     MAX_TEXT_LEN,
                        "max_words":        MAX_WORDS,
                        "max_text_regions": MAX_TEXT_REGIONS,
                        "max_total_chars":  MAX_TOTAL_CHARS,
                        "min_bbox_height":  MIN_BBOX_HEIGHT,
                    },
                    "statistics": {
                        "total": len(all_records),
                        "train": len(splits[0][1]),
                        "val":   len(splits[1][1]),
                        "test":  len(splits[2][1]),
                    },
                },
                "data": all_records,
            }, fh, indent=2)

        print(f"[SAVED] {len(all_records)} total  "
              f"(train={len(splits[0][1])}  val={len(splits[1][1])}  "
              f"test={len(splits[2][1])})", flush=True)
        print(f"[PATH]  {self.out_dir}", flush=True)

        print("\n[SAMPLES]", flush=True)
        for e in all_records[:8]:
            print(f"  id={e['id']}  text='{e['text']}'  "
                  f"ocr='{e['ocr_text']}'  conf={e['metadata']['ocr_confidence']:.0f}",
                  flush=True)
            print(f"    caption: {e['caption']}", flush=True)
            print(f"    prompt : {e['prompt']}", flush=True)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    print(f"[START] Text-in-Image Dataset Pipeline  v{VERSION}", flush=True)
    print(f"[START] Source: AnyWord-3M (HuggingFace)", flush=True)

    parser = argparse.ArgumentParser(
        description="Create text-in-image training dataset from AnyWord-3M")
    parser.add_argument("--output", default="dataset_output",
                        help="Output directory (default: dataset_output)")
    parser.add_argument("--max-images", type=int, default=TARGET_IMAGES,
                        help=f"Target number of images (default: {TARGET_IMAGES})")
    parser.add_argument("--subsets", nargs="+", default=None,
                        help="AnyWord-3M subsets to use (default: laion + OCR subsets)")
    parser.add_argument("--no-streaming", action="store_true",
                        help="Download full dataset instead of streaming")
    args = parser.parse_args()

    print(f"[ARGS] output={args.output}  max={args.max_images}  "
          f"subsets={args.subsets or 'default'}  "
          f"streaming={not args.no_streaming}", flush=True)

    creator = DatasetCreator(
        output_dir = args.output,
        max_images = args.max_images,
        subsets    = args.subsets,
        streaming  = not args.no_streaming,
    )
    records = creator.run()
    creator.save(records)
    print("[END]", flush=True)


if __name__ == "__main__":
    main()
