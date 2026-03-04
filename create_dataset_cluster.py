"""
Dataset Creation Script - Text-to-Image (v3)
Pipeline:
  1. Load image
  2. Quality check (resolution, sharpness, brightness)
  3. OCR → extract ONLY text physically visible in the image
  4. Reject if no short, confident text found
  5. BLIP → generate visual description
  6. Build prompt: "[description], with the text '[OCR_TEXT]' clearly visible"

No TextCaps captions used — the test set has none. Text comes 100% from OCR.
"""

import os
import sys

# Set tessdata path before importing pytesseract
os.environ.setdefault("TESSDATA_PREFIX", "/usr/share/tesseract-ocr/4.00/tessdata")

import json
import argparse

# Set TESSDATA_PREFIX if not already set (required for Tesseract)
if 'TESSDATA_PREFIX' not in os.environ:
    for _candidate in [
        os.path.expanduser('~/miniconda3/share/tessdata'),
        '/usr/share/tessdata',
        '/usr/share/tesseract-ocr/4.00/tessdata',
    ]:
        if os.path.isdir(_candidate):
            os.environ['TESSDATA_PREFIX'] = _candidate
            break
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import pytesseract
import cv2
import numpy as np
import re
import shutil
from itertools import groupby
from operator import itemgetter

print("[IMPORTS_DONE]", flush=True)  # Debug: confirm imports completed


# ─── Quality thresholds ───────────────────────────────────────────────────────
MIN_RESOLUTION  = 300       # min width AND height (pixels)
MIN_SHARPNESS   = 150       # Laplacian variance – ensures sharp, readable text
BRIGHTNESS_MIN  = 40        # avoid near-black images
BRIGHTNESS_MAX  = 215       # avoid washed-out images
MIN_CONTRAST    = 30        # std-dev of gray – reject flat/uniform images
MIN_OCR_CONF    = 70        # per-word OCR confidence – HIGH quality only
MAX_TEXT_LEN    = 40        # max characters (allow short sentences)
MIN_TEXT_LEN    = 3         # min characters


# ─── OCR helpers ─────────────────────────────────────────────────────────────

def preprocess_for_ocr(pil_image):
    """Enhance image for better OCR accuracy."""
    w, h = pil_image.size
    if w < 512 or h < 512:
        scale = max(512 / w, 512 / h)
        pil_image = pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    pil_image = pil_image.filter(ImageFilter.SHARPEN)
    pil_image = ImageEnhance.Contrast(pil_image).enhance(1.5)
    return pil_image


def extract_short_text_ocr(image_path):
    """
    Extract the best readable text visible in the image.
    Tries individual high-confidence words AND multi-word phrases.
    Returns (text, confidence) or (None, 0).
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img_proc = preprocess_for_ocr(img)

        best_words  = []   # (word, conf, block_num, line_num, word_num)
        for psm in (6, 11, 3):
            config = f"--psm {psm} --oem 3"
            data = pytesseract.image_to_data(
                img_proc,
                output_type=pytesseract.Output.DICT,
                lang="eng",
                config=config,
            )
            words_in_run = []
            for i, raw_text in enumerate(data["text"]):
                word = raw_text.strip()
                if not word:
                    continue
                conf = int(data["conf"][i])
                if conf < MIN_OCR_CONF:
                    continue
                cleaned = re.sub(r"[^A-Za-z0-9 '\-\.!?,]", "", word).strip()
                if not cleaned or not re.search(r"[A-Za-z]", cleaned):
                    continue
                words_in_run.append((
                    cleaned, conf,
                    data["block_num"][i], data["line_num"][i]
                ))
            if words_in_run:
                best_words = words_in_run   # use last psm that gave results

        if not best_words:
            return None, 0

        candidates = []

        # Single high-confidence words
        for w, conf, bn, ln in best_words:
            if MIN_TEXT_LEN <= len(w) <= MAX_TEXT_LEN:
                # Short words need even higher confidence
                required = MIN_OCR_CONF if len(w) >= 5 else 85
                if conf >= required:
                    candidates.append((w, conf))

        # Multi-word phrases on the same block+line → merge adjacent words
        for (bn, ln), grp in groupby(best_words, key=itemgetter(2, 3)):
            group = list(grp)
            if len(group) < 2:
                continue
            phrase = " ".join(w for w, _, _, _ in group)
            avg_conf = int(sum(c for _, c, _, _ in group) / len(group))
            if MIN_TEXT_LEN <= len(phrase) <= MAX_TEXT_LEN:
                candidates.append((phrase, avg_conf))

        if not candidates:
            return None, 0

        # Score: confidence weighted heavily, length bonus for moderate length
        def score(item):
            text, conf = item
            length_bonus = min(len(text), 20) * 1.5
            return conf * 0.7 + length_bonus * 0.3

        candidates.sort(key=score, reverse=True)
        return candidates[0]

    except Exception:
        return None, 0


# ─── Image quality check ──────────────────────────────────────────────────────

def check_quality(image_path):
    """Return (ok, metrics_dict) after quality checks."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False, {}
        h, w = img.shape[:2]
        if h < MIN_RESOLUTION or w < MIN_RESOLUTION:
            return False, {}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < MIN_SHARPNESS:
            return False, {}

        brightness = float(np.mean(gray))
        if not (BRIGHTNESS_MIN <= brightness <= BRIGHTNESS_MAX):
            return False, {}

        contrast = float(np.std(gray))
        if contrast < MIN_CONTRAST:
            return False, {}

        return True, {
            "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "resolution": [int(w), int(h)],
        }
    except Exception:
        return False, {}


# ─── BLIP caption ─────────────────────────────────────────────────────────────

class BLIPCaptioner:
    def __init__(self, device):
        print(f"[BLIP] Loading model on {device}...", flush=True)
        self.device = device
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        self.model.eval()
        print("[BLIP] Model ready.", flush=True)

    @torch.no_grad()
    def caption(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            # Unconditional caption
            inputs = self.processor(image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model.generate(
                **inputs,
                max_length=75,
                num_beams=5,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
                early_stopping=True,
                length_penalty=1.2,
            )
            raw = self.processor.decode(out[0], skip_special_tokens=True).strip()
            # Capitalise first letter
            return raw[0].upper() + raw[1:] if raw else raw
        except Exception:
            return None


# ─── Dataset creator ──────────────────────────────────────────────────────────

class DatasetCreator:
    def __init__(self, output_dir, max_images=400, batch_size=16):
        self.base_dir = Path(output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(
            [d for d in self.base_dir.glob("v[0-9]*") if d.is_dir()],
            key=lambda d: int(d.name[1:])
        )
        version = (int(existing[-1].name[1:]) + 1) if existing else 1
        self.out_dir = self.base_dir / f"v{version}"
        print(f"\n[DIR] Output: {self.out_dir}", flush=True)

        for split in ("train", "val", "test"):
            (self.out_dir / split / "images").mkdir(parents=True, exist_ok=True)

        self.max_images = max_images
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DEVICE] {self.device}", flush=True)
        if torch.cuda.is_available():
            print(f"[GPU] {torch.cuda.get_device_name(0)}", flush=True)

    def get_image_list(self, images_dir):
        images_dir = Path(images_dir)
        paths = sorted(
            list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        )
        print(f"[DATA] Found {len(paths)} images in {images_dir}", flush=True)
        return paths

    def run(self, images_dir):
        image_paths = self.get_image_list(images_dir)
        captioner = BLIPCaptioner(self.device)

        accepted = []
        rejected_quality = 0
        rejected_ocr = 0

        print(f"\n[PROC] Processing (target: {self.max_images} images)...", flush=True)

        for idx, path in enumerate(image_paths):
            ok, metrics = check_quality(path)
            if not ok:
                rejected_quality += 1
                if idx % 200 == 0:
                    print(f"  [{idx}/{len(image_paths)}] acc={len(accepted)} rej_q={rejected_quality} rej_ocr={rejected_ocr}", flush=True)
                continue

            ocr_text, ocr_conf = extract_short_text_ocr(path)
            if not ocr_text:
                rejected_ocr += 1
                continue

            blip_caption = captioner.caption(path)
            if not blip_caption:
                blip_caption = "a photograph"

            # Build a rich, training-friendly prompt
            prompt = (
                f"{blip_caption}, "
                f"with the text \u201c{ocr_text}\u201d clearly visible, "
                f"sharp focus, high resolution photography"
            )

            accepted.append({
                "image_id": path.stem,
                "image_path": str(path),
                "text": ocr_text,
                "caption": blip_caption,
                "prompt": prompt,
                "metadata": {
                    "ocr_confidence": float(ocr_conf),
                    "text_length": len(ocr_text),
                    **metrics,
                },
            })

            if len(accepted) % 20 == 0:
                print(f"  [{idx}/{len(image_paths)}] acc={len(accepted)} rej_q={rejected_quality} rej_ocr={rejected_ocr}", flush=True)

            if len(accepted) >= self.max_images:
                print(f"[DONE] Reached target {self.max_images}.", flush=True)
                break

        print(f"\n[SUMMARY] Accepted={len(accepted)}  Rej-quality={rejected_quality}  Rej-OCR={rejected_ocr}", flush=True)
        return accepted

    def save(self, records):
        if not records:
            print("[WARN] No records to save.", flush=True)
            return

        n = len(records)
        n_train = int(n * 0.8)
        n_val   = int(n * 0.1)
        splits = (
            ("train", records[:n_train]),
            ("val",   records[n_train:n_train + n_val]),
            ("test",  records[n_train + n_val:]),
        )

        print("[SAVE] Copying images and writing annotations...", flush=True)
        all_records = []
        global_id = 0

        for split_name, split_records in splits:
            split_data = []
            for rec in split_records:
                src = Path(rec["image_path"])
                filename = f"img_{global_id:04d}.jpg"
                dst = self.out_dir / split_name / "images" / filename
                shutil.copy2(src, dst)

                entry = {
                    "id": global_id,
                    "image_id": rec["image_id"],
                    "filename": filename,
                    "filepath": f"{split_name}/images/{filename}",
                    "split": split_name,
                    "text": rec["text"],
                    "caption": rec["caption"],
                    "prompt": rec["prompt"],
                    "metadata": rec["metadata"],
                }
                split_data.append(entry)
                all_records.append(entry)
                global_id += 1

            with open(self.out_dir / f"{split_name}.json", "w") as fh:
                json.dump({"split": split_name, "data": split_data}, fh, indent=2)

        with open(self.out_dir / "dataset_complete.json", "w") as fh:
            json.dump({
                "metadata": {
                    "version": "3.0",
                    "description": "OCR-based text-to-image dataset from TextCaps images",
                    "text_source": "OCR (Tesseract) — text physically visible in image",
                    "thresholds": {
                        "min_resolution": MIN_RESOLUTION,
                        "min_sharpness": MIN_SHARPNESS,
                        "min_ocr_confidence": MIN_OCR_CONF,
                        "max_text_length": MAX_TEXT_LEN,
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

        print(f"[SAVED] {len(all_records)} total  (train={len(splits[0][1])} val={len(splits[1][1])} test={len(splits[2][1])})", flush=True)
        print(f"[PATH]  {self.out_dir}", flush=True)

        # Show sample entries
        print("\n[SAMPLES]", flush=True)
        for entry in all_records[:5]:
            print(f"  id={entry['id']}  text='{entry['text']}'  conf={entry['metadata']['ocr_confidence']:.0f}%", flush=True)
            print(f"    caption: {entry['caption']}", flush=True)
            print(f"    prompt : {entry['prompt']}", flush=True)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    print("[START]", flush=True)

    parser = argparse.ArgumentParser(description="Create OCR-based text-to-image dataset")
    parser.add_argument("--images",     required=True,         help="Directory of input images")
    parser.add_argument("--output",     default="dataset_output")
    parser.add_argument("--max-images", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    print(f"[ARGS] images={args.images}  output={args.output}  max={args.max_images}", flush=True)

    creator = DatasetCreator(
        output_dir=args.output,
        max_images=args.max_images,
        batch_size=args.batch_size,
    )

    records = creator.run(args.images)
    creator.save(records)
    print("[END]", flush=True)


if __name__ == "__main__":
    main()
