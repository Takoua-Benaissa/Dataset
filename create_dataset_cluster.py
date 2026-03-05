"""
Text-in-Image Dataset Creation Pipeline  v7
============================================
Loads TextOCR + TextCaps ground-truth word bounding boxes, groups co-linear
words into full phrases, filters for short readable English text, verifies
readability with Tesseract OCR (used as a gate only — annotation text is
canonical), adds a BLIP visual caption, then exports train / val / test splits.

Key filtering criteria
──────────────────────
  • Image quality   : min 300 px, sharpness ≥ 100, contrast ≥ 25
  • Phrase length   : 3–30 characters, ≤ 5 words
  • Density filter  : ≤ 3 valid text regions AND ≤ 25 total chars per image
  • Bbox height     : ≥ 30 px  (reject tiny unreadable text)
  • OCR gate        : Tesseract confidence ≥ 85 %, coverage ≥ 30 %
  • Language        : English only (langdetect + ASCII fallback)
  • Target          : 100 images per run (raise TARGET_IMAGES for production)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from transformers import BlipProcessor, BlipForConditionalGeneration

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

VERSION = "7.0"

# Image quality
MIN_RESOLUTION  = 300
MIN_SHARPNESS   = 100
BRIGHTNESS_MIN  = 40
BRIGHTNESS_MAX  = 220
MIN_CONTRAST    = 25

# Text filters
MAX_WORDS        = 5
MIN_TEXT_LEN     = 3
MAX_TEXT_LEN     = 30

# Density filter (per image)
MAX_TEXT_REGIONS = 3
MAX_TOTAL_CHARS  = 25

# Bbox minimum
MIN_BBOX_HEIGHT  = 30   # pixels

# Line-grouping tolerances (relative to average word height)
LINE_Y_TOL_RATIO = 0.6
WORD_GAP_RATIO   = 2.5

# OCR gate
MIN_OCR_CONF     = 85
MIN_OCR_COVERAGE = 0.3

# Run target
TARGET_IMAGES    = 100


# ─── Text utilities ───────────────────────────────────────────────────────────

def is_valid_text(text: str) -> bool:
    """Accept short, letter-containing English phrases."""
    text = text.strip()
    if not (MIN_TEXT_LEN <= len(text) <= MAX_TEXT_LEN):
        return False
    if len(text.split()) > MAX_WORDS:
        return False
    if not re.search(r"[A-Za-z]", text):
        return False
    return True


def is_english(text: str) -> bool:
    """Return True if text is detected as English."""
    if all(ord(c) < 128 for c in text):
        return True
    if not _LANGDETECT_OK:
        return True
    try:
        return detect(text) == "en"
    except Exception:
        return True


# ─── Annotation loaders ───────────────────────────────────────────────────────

def load_textocr_annotations(json_path: str) -> Dict[str, List[Dict]]:
    """Load TextOCR JSON → {image_id: [{"text": str, "bbox": [x,y,w,h]}, ...]}"""
    print(f"[ANN] Loading TextOCR from {json_path} …", flush=True)
    with open(json_path) as fh:
        data = json.load(fh)
    per_image: Dict[str, List[Dict]] = {}
    anns = data.get("anns", data)
    if isinstance(anns, dict):
        anns = list(anns.values())
    for ann in anns:
        img_id = str(ann.get("image_id", ""))
        text   = ann.get("utf8_string", "").strip()
        bbox   = ann.get("bbox", None)
        if not text or bbox is None:
            continue
        per_image.setdefault(img_id, []).append({"text": text, "bbox": bbox})
    print(f"[ANN] TextOCR: {len(per_image)} images", flush=True)
    return per_image


def load_textcaps_annotations(json_path: str) -> Dict[str, List[Dict]]:
    """Load TextCaps JSON → {image_id: [{"text": str, "bbox": [x,y,w,h]}, ...]}"""
    print(f"[ANN] Loading TextCaps from {json_path} …", flush=True)
    with open(json_path) as fh:
        data = json.load(fh)
    per_image: Dict[str, List[Dict]] = {}
    for item in data.get("data", []):
        img_id = str(item.get("image_id", ""))
        tokens = item.get("reference_strs", [])
        bboxes = item.get("normalized_bbox", [])
        for tok, bb in zip(tokens, bboxes):
            tok = tok.strip()
            if not tok:
                continue
            per_image.setdefault(img_id, []).append({"text": tok, "bbox": bb})
    print(f"[ANN] TextCaps: {len(per_image)} images", flush=True)
    return per_image


# ─── Phrase grouping ──────────────────────────────────────────────────────────

def group_annotations_into_lines(entries: List[Dict]) -> List[Dict]:
    """
    Merge per-word bboxes from a single image into phrase-level entries.

    1. Compute average word height.
    2. Sort by y-centre; cluster into lines (y deviation < avg_h × LINE_Y_TOL_RATIO).
    3. Within each line, sort by x and merge adjacent words whose gap ≤ avg_h × WORD_GAP_RATIO.
    4. Discard phrases whose bbox height < MIN_BBOX_HEIGHT.

    Returns [{"text": phrase, "bbox": [x, y, w, h]}].
    """
    if not entries:
        return []

    heights = [e["bbox"][3] for e in entries if e["bbox"][3] > 0]
    avg_h   = float(np.mean(heights)) if heights else 20.0
    y_tol     = avg_h * LINE_Y_TOL_RATIO
    gap_limit = avg_h * WORD_GAP_RATIO

    def yc(e):
        b = e["bbox"]; return b[1] + b[3] / 2.0

    sorted_entries = sorted(entries, key=yc)
    lines: List[List[Dict]] = []
    for entry in sorted_entries:
        placed = False
        for line in lines:
            line_yc = float(np.mean([yc(e) for e in line]))
            if abs(yc(entry) - line_yc) < y_tol:
                line.append(entry); placed = True; break
        if not placed:
            lines.append([entry])

    merged: List[Dict] = []
    for line in lines:
        line.sort(key=lambda e: e["bbox"][0])
        groups: List[List[Dict]] = [[line[0]]]
        for entry in line[1:]:
            prev_right = groups[-1][-1]["bbox"][0] + groups[-1][-1]["bbox"][2]
            if (entry["bbox"][0] - prev_right) <= gap_limit:
                groups[-1].append(entry)
            else:
                groups.append([entry])
        for grp in groups:
            bboxes = [e["bbox"] for e in grp]
            xs  = [b[0]        for b in bboxes]
            ys  = [b[1]        for b in bboxes]
            x2s = [b[0] + b[2] for b in bboxes]
            y2s = [b[1] + b[3] for b in bboxes]
            mh  = max(y2s) - min(ys)
            if mh < MIN_BBOX_HEIGHT:
                continue
            merged.append({
                "text": " ".join(e["text"] for e in grp),
                "bbox": [min(xs), min(ys), max(x2s) - min(xs), mh],
            })
    return merged


# ─── Density / selection ──────────────────────────────────────────────────────

def image_text_density_ok(entries: List[Dict]) -> bool:
    """Return True if the image is NOT a dense-text scene (book page, sign wall…)."""
    lines = group_annotations_into_lines(entries)
    valid = [l for l in lines if is_valid_text(l["text"])]
    if len(valid) > MAX_TEXT_REGIONS:
        return False
    if sum(len(l["text"]) for l in valid) > MAX_TOTAL_CHARS:
        return False
    return True


def select_best_text(entries: List[Dict]) -> Optional[Dict]:
    """
    Pick the best phrase from grouped lines.
    Scores by length + word_bonus×5 (strongly prefers multi-word phrases).
    Returns {"text": str, "bbox": [x,y,w,h]} or None.
    """
    lines = group_annotations_into_lines(entries)
    candidates = []
    for line in lines:
        if not is_valid_text(line["text"]):
            continue
        if not is_english(line["text"]):
            continue
        score = len(line["text"]) + len(line["text"].split()) * 5
        candidates.append((score, line))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


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
    image_path: str,
    expected_text: Optional[str] = None,
    bbox: Optional[List[int]] = None,
) -> Tuple[Optional[str], float]:
    """
    Run Tesseract and verify `expected_text` is readable.
    OCR is a readability gate only — caller keeps annotation as canonical text.

    Critical fixes applied:
    • When expected_text is given but no matching token found → return (None, 0).
      Do NOT fall through to an arbitrary high-confidence token.
    • Coverage threshold relaxed to MIN_OCR_COVERAGE=0.3.
    """
    try:
        img = Image.open(image_path).convert("RGB")

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
            d   = pytesseract.image_to_data(src, config=cfg,
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
                return None, 0          # ← critical: no fallthrough
            matching.sort(key=lambda x: x[1], reverse=True)
            return matching[0]

        tokens.sort(key=lambda x: x[1], reverse=True)
        return tokens[0]

    except Exception:
        return None, 0


# ─── Image quality ────────────────────────────────────────────────────────────

def check_image_quality(image_path: str) -> Tuple[bool, Dict]:
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False, {}
        h, w = img.shape[:2]
        if h < MIN_RESOLUTION or w < MIN_RESOLUTION:
            return False, {}
        gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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


# ─── BLIP captioner ───────────────────────────────────────────────────────────

class BLIPCaptioner:
    def __init__(self, device: str):
        print(f"[BLIP] Loading model on {device} …", flush=True)
        self.device    = device
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base").to(device)
        self.model.eval()
        print("[BLIP] Model ready.", flush=True)

    @torch.no_grad()
    def caption(self, image_path: str) -> Optional[str]:
        try:
            image  = Image.open(image_path).convert("RGB")
            inputs = self.processor(image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out    = self.model.generate(
                **inputs, max_length=75, num_beams=5,
                no_repeat_ngram_size=3, repetition_penalty=1.5,
                early_stopping=True, length_penalty=1.2,
            )
            raw = self.processor.decode(out[0], skip_special_tokens=True).strip()
            return (raw[0].upper() + raw[1:]) if raw else raw
        except Exception:
            return None


# ─── Dataset creator ──────────────────────────────────────────────────────────

class DatasetCreator:

    def __init__(self, output_dir: str, annotation_files: List[str],
                 image_dirs: List[str], max_images: int = TARGET_IMAGES,
                 batch_size: int = 16):
        self.base_dir = Path(output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(
            [d for d in self.base_dir.glob("v[0-9]*") if d.is_dir()],
            key=lambda d: int(d.name[1:]))
        version      = (int(existing[-1].name[1:]) + 1) if existing else 1
        self.out_dir = self.base_dir / f"v{version}"
        print(f"[DIR] Output → {self.out_dir}", flush=True)

        for split in ("train", "val", "test"):
            (self.out_dir / split / "images").mkdir(parents=True, exist_ok=True)

        self.annotation_files = annotation_files
        self.image_dirs       = [Path(d) for d in image_dirs]
        self.max_images       = max_images
        self.batch_size       = batch_size
        self.device           = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DEVICE] {self.device}", flush=True)
        if self.device == "cuda":
            print(f"[GPU] {torch.cuda.get_device_name(0)}", flush=True)

    def load_annotations(self) -> Dict[str, List[Dict]]:
        merged: Dict[str, List[Dict]] = {}
        for path in self.annotation_files:
            loader = load_textcaps_annotations if "textcaps" in path.lower() \
                     else load_textocr_annotations
            for img_id, entries in loader(path).items():
                merged.setdefault(img_id, []).extend(entries)
        print(f"[ANN] Total: {len(merged)} image IDs", flush=True)
        return merged

    def find_images(self) -> Dict[str, Path]:
        found: Dict[str, Path] = {}
        for d in self.image_dirs:
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for p in d.glob(ext):
                    found[p.stem] = p
        print(f"[IMG] Found {len(found)} images on disk", flush=True)
        return found

    def prefilter_annotations(self, annotations, images):
        candidates, skip_miss, skip_dens, skip_text = [], 0, 0, 0
        for img_id, entries in annotations.items():
            stem = img_id.replace(".jpg", "").replace(".png", "")
            path = images.get(stem) or images.get(img_id)
            if path is None:
                skip_miss += 1; continue
            if not image_text_density_ok(entries):
                skip_dens += 1; continue
            best = select_best_text(entries)
            if best is None:
                skip_text += 1; continue
            candidates.append((img_id, path, best))
        print(f"[PRE] Candidates={len(candidates)}  miss={skip_miss}  "
              f"density={skip_dens}  no-text={skip_text}", flush=True)
        return candidates

    def run(self) -> List[Dict]:
        annotations = self.load_annotations()
        images      = self.find_images()
        candidates  = self.prefilter_annotations(annotations, images)
        captioner   = BLIPCaptioner(self.device)

        accepted: List[Dict] = []
        rej_q = rej_ocr = 0

        print(f"\n[PROC] {len(candidates)} candidates  "
              f"(target: {self.max_images}) …", flush=True)

        for idx, (img_id, img_path, best_entry) in enumerate(candidates):
            if len(accepted) >= self.max_images:
                print(f"[DONE] Target {self.max_images} reached.", flush=True)
                break
            if idx % 500 == 0:
                print(f"  [{idx}/{len(candidates)}] "
                      f"acc={len(accepted)}  rej_q={rej_q}  rej_ocr={rej_ocr}",
                      flush=True)

            ok, metrics = check_image_quality(str(img_path))
            if not ok:
                rej_q += 1; continue

            anno_text = best_entry["text"]
            anno_bbox = best_entry["bbox"]

            # OCR gate — verify text is readable
            ocr_text, ocr_conf = verify_text_with_ocr(
                str(img_path), expected_text=anno_text, bbox=anno_bbox)
            if ocr_text is None:
                rej_ocr += 1; continue

            blip_caption = captioner.caption(str(img_path)) or "a photograph"

            # Canonical text = annotation phrase (NOT the OCR partial read)
            final_text = anno_text
            prompt = (f"{blip_caption}, "
                      f"with the text \u201c{final_text}\u201d clearly visible, "
                      f"sharp focus, high resolution photography")

            accepted.append({
                "image_id":   img_id,
                "image_path": str(img_path),
                "text":       final_text,
                "ocr_text":   ocr_text,
                "caption":    blip_caption,
                "prompt":     prompt,
                "metadata": {"ocr_confidence": float(ocr_conf),
                              "text_length": len(final_text), **metrics},
            })
            print(f"  #ACCEPTED {len(accepted):4d}  "
                  f"text='{final_text}'  ocr='{ocr_text}'  conf={ocr_conf}",
                  flush=True)

        print(f"\n[SUMMARY] Processed={idx+1}  Accepted={len(accepted)}  "
              f"Rej-quality={rej_q}  Rej-OCR={rej_ocr}", flush=True)
        return accepted

    def save(self, records: List[Dict]):
        if not records:
            print("[WARN] No records to save.", flush=True); return

        n       = len(records)
        n_train = int(n * 0.8)
        n_val   = int(n * 0.1)
        splits  = [
            ("train", records[:n_train]),
            ("val",   records[n_train:n_train + n_val]),
            ("test",  records[n_train + n_val:]),
        ]

        print("[SAVE] Copying images and writing annotations …", flush=True)
        all_records: List[Dict] = []
        global_id = 0

        for split_name, split_records in splits:
            split_data: List[Dict] = []
            for rec in split_records:
                filename = f"img_{global_id:04d}.jpg"
                dst = self.out_dir / split_name / "images" / filename
                shutil.copy2(rec["image_path"], dst)
                entry = {
                    "id":       global_id,
                    "image_id": rec["image_id"],
                    "filename": filename,
                    "filepath": f"{split_name}/images/{filename}",
                    "split":    split_name,
                    "text":     rec["text"],
                    "ocr_text": rec["ocr_text"],
                    "caption":  rec["caption"],
                    "prompt":   rec["prompt"],
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
                    "version":     VERSION,
                    "description": (
                        "Text-in-image dataset built from TextOCR + TextCaps. "
                        "Phrases are ground-truth annotations; OCR is a readability gate."),
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

    parser = argparse.ArgumentParser(
        description="Create text-in-image training dataset from TextOCR + TextCaps")
    parser.add_argument("--annotation-files", nargs="+", required=True,
                        help="Path(s) to TextOCR / TextCaps JSON annotation files")
    parser.add_argument("--image-dirs", nargs="+", required=True,
                        help="Directory / directories containing the source images")
    parser.add_argument("--output",     default="dataset_output")
    parser.add_argument("--max-images", type=int, default=TARGET_IMAGES)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    print(f"[ARGS] annotation_files={args.annotation_files}  "
          f"image_dirs={args.image_dirs}  "
          f"output={args.output}  max={args.max_images}", flush=True)

    creator = DatasetCreator(
        output_dir       = args.output,
        annotation_files = args.annotation_files,
        image_dirs       = args.image_dirs,
        max_images       = args.max_images,
        batch_size       = args.batch_size,
    )
    records = creator.run()
    creator.save(records)
    print("[END]", flush=True)


if __name__ == "__main__":
    main()
