"""
Text-in-Image Dataset Creation Pipeline  v12
=============================================
OCR     : EasyOCR  – accurate on natural scene text, GPU-accelerated
Captions: BLIP-2   – rich visual descriptions (Salesforce/blip2-flan-t5-xl)
Source  : stzhao/AnyWord-3M  (HuggingFace, streaming)

Pipeline per image
------------------
1. Filter annotations  → keep valid, short, English text regions
2. Quality gate        → resolution / sharpness / brightness / contrast
3. EasyOCR verify      → confirm the annotation text is readable in the image
4. EasyOCR reconstruct → expand to the full visible phrase near the bbox
5. BLIP-2 caption      → general visual description of the whole image
6. BLIP-2 prompt       → natural training sentence that embeds the OCR text
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
import easyocr
from PIL import Image
from tqdm import tqdm
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BlipProcessor,
    BlipForConditionalGeneration,
)

try:
    import Levenshtein as _levenshtein
    _LEVENSHTEIN_OK = True
except ImportError:
    _LEVENSHTEIN_OK = False

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    _LANGDETECT_OK = True
except ImportError:
    _LANGDETECT_OK = False

print("[IMPORTS_DONE]", flush=True)


# ─── Constants ────────────────────────────────────────────────────────────────

VERSION = "12.0"

# Image quality thresholds
MIN_RESOLUTION  = 256
MIN_SHARPNESS   = 80
BRIGHTNESS_MIN  = 30
BRIGHTNESS_MAX  = 230
MIN_CONTRAST    = 20

# Annotation text filters
MAX_WORDS    = 5
MIN_TEXT_LEN = 1
MAX_TEXT_LEN = 40

# Dense-text rejection
MAX_TEXT_REGIONS = 4
MAX_TOTAL_CHARS  = 40

# Bounding-box minimum height (pixels)
MIN_BBOX_HEIGHT = 30

# EasyOCR acceptance thresholds
MIN_OCR_CONF     = 0.40   # raw EasyOCR confidence in [0, 1]
MIN_OCR_COVERAGE = 0.30   # Levenshtein coverage score to accept a match

# Full-phrase reconstruction
MAX_PHRASE_WORDS = 10
MAX_PHRASE_CHARS = 80

# Run target
TARGET_IMAGES = 1000

# AnyWord-3M
DATASET_NAME    = "stzhao/AnyWord-3M"
DATASET_SUBSETS = ["laion", "OCR_COCO_Text", "OCR_mlt2019", "OCR_Art"]

# BLIP-2 generation settings
BLIP2_MODEL_NAME = "Salesforce/blip2-flan-t5-xl"
BLIP_MODEL_NAME  = "Salesforce/blip-image-captioning-large"
CAPTION_MAX_LEN  = 200
CAPTION_MIN_LEN  = 15
NUM_BEAMS        = 5


# ─── Text normalisation & scoring ────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, ASCII-only, no punctuation – used for text comparison."""
    text = text.lower()
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def _coverage(expected: str, candidate: str) -> float:
    """
    Normalised character-level similarity in [0, 1].
    Uses Levenshtein when available, otherwise substring containment.
    """
    exp  = _normalize(expected)
    cand = _normalize(candidate)
    if not exp:
        return 1.0 if not cand else 0.0
    if _LEVENSHTEIN_OK:
        edits   = _levenshtein.distance(exp, cand)
        correct = max(0, len(exp) - edits)
        denom   = edits + correct
        return correct / denom if denom else 1.0
    return 1.0 if (exp in cand or cand in exp) else 0.0


# ─── EasyOCR singleton ────────────────────────────────────────────────────────

_ocr_reader: Optional[easyocr.Reader] = None


def get_ocr_reader(gpu: bool = True) -> easyocr.Reader:
    """Initialise EasyOCR once and reuse for the whole run."""
    global _ocr_reader
    if _ocr_reader is None:
        print(f"[OCR] Initialising EasyOCR (gpu={gpu}) …", flush=True)
        _ocr_reader = easyocr.Reader(["en"], gpu=gpu)
        print("[OCR] EasyOCR ready.", flush=True)
    return _ocr_reader


def _ocr_detections(image: Image.Image) -> List[Dict]:
    """
    Run EasyOCR and return a list of detection dicts:
      { text, conf, x1, y1, x2, y2, cy }
    Only detections with conf >= MIN_OCR_CONF are kept.
    """
    reader  = get_ocr_reader()
    arr     = np.array(image.convert("RGB"))
    results = reader.readtext(arr, detail=1, paragraph=False)
    dets = []
    for (pts, text, conf) in results:
        text = text.strip()
        if not text or float(conf) < MIN_OCR_CONF:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        dets.append({
            "text": text,
            "conf": float(conf),
            "x1": min(xs), "y1": min(ys),
            "x2": max(xs), "y2": max(ys),
            "cy": (min(ys) + max(ys)) / 2.0,
        })
    return dets


# ─── OCR verification & reconstruction ───────────────────────────────────────

def verify_text_with_ocr(
    image: Image.Image,
    expected_text: str,
    bbox: Optional[List[int]] = None,
) -> Tuple[Optional[str], float]:
    """
    Check that `expected_text` is actually readable in the image.

    Tries the cropped bbox region first (more focused), then the full image.
    A detection is accepted when its Levenshtein coverage >= MIN_OCR_COVERAGE
    or it is a substring of / contains the expected text.

    Returns (best_token, confidence_0_to_100) or (None, 0.0).
    """
    img    = image.convert("RGB")
    iw, ih = img.size

    # Build sources: tight crop first, full image as fallback
    sources: List[Image.Image] = []
    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        pad  = max(8, int(h * 0.20))
        crop = img.crop((max(0, x - pad), max(0, y - pad),
                         min(iw, x + w + pad), min(ih, y + h + pad)))
        sources.append(crop)
    sources.append(img)

    exp_lower = expected_text.lower()

    for src in sources:
        try:
            dets = _ocr_detections(src)
        except Exception as e:
            print(f"[OCR] error in verify: {e}", flush=True)
            continue

        scored: List[Tuple[str, float]] = []
        for d in dets:
            cov       = _coverage(expected_text, d["text"])
            tok_lower = d["text"].lower()
            if (cov >= MIN_OCR_COVERAGE
                    or tok_lower in exp_lower
                    or exp_lower in tok_lower):
                blended = (d["conf"] + cov) / 2.0
                scored.append((d["text"], blended))

        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            tok, conf = scored[0]
            return tok, round(conf * 100, 2)

    return None, 0.0


def reconstruct_phrase_with_easyocr(
    image: Image.Image,
    anchor_bbox: Optional[List[int]] = None,
) -> Tuple[Optional[str], float]:
    """
    Reconstruct the most likely complete text phrase visible near `anchor_bbox`.

    EasyOCR bounding boxes allow spatial filtering: we keep only detections
    that vertically overlap the anchor region, then join them left-to-right.

    Returns (phrase, avg_confidence_0_to_100) or (None, 0.0).
    """
    img  = image.convert("RGB")
    try:
        dets = _ocr_detections(img)
    except Exception as e:
        print(f"[OCR] error in reconstruct: {e}", flush=True)
        return None, 0.0

    if not dets:
        return None, 0.0

    # ── Filter to detections near the anchor bbox ─────────────────────────────
    if anchor_bbox is not None:
        ax, ay, aw, ah = [int(v) for v in anchor_bbox]
        anchor_y1 = ay
        anchor_y2 = ay + ah
        anchor_cy = ay + ah / 2.0

        avg_h     = max(1.0, float(np.mean([d["y2"] - d["y1"] for d in dets])))
        tolerance = avg_h * 1.2

        nearby = [
            d for d in dets
            if d["y1"] <= anchor_y2 + tolerance and d["y2"] >= anchor_y1 - tolerance
        ]
        if not nearby:
            nearby = [min(dets, key=lambda d: abs(d["cy"] - anchor_cy))]
    else:
        nearby = dets

    # ── Sort top-to-bottom, left-to-right and join ────────────────────────────
    nearby.sort(key=lambda d: (round(d["cy"] / 20) * 20, d["x1"]))
    phrase   = " ".join(d["text"] for d in nearby).strip()
    avg_conf = float(np.mean([d["conf"] for d in nearby]))

    # ── Validate length and content ───────────────────────────────────────────
    words = phrase.split()
    if len(words) > MAX_PHRASE_WORDS:
        phrase = " ".join(words[:MAX_PHRASE_WORDS])
    if len(phrase) > MAX_PHRASE_CHARS:
        return None, 0.0
    if not re.search(r"[A-Za-z]", phrase):
        return None, 0.0
    if re.search(r"[^\x00-\x7F]", phrase):
        return None, 0.0

    return phrase, round(avg_conf * 100, 2)


# ─── BLIP-2 Caption + Prompt Generator ───────────────────────────────────────

class CaptionGenerator:
    """
    Generates a visual caption and a training prompt using BLIP-2.

    caption  – general description of the image (no OCR text forced in)
    prompt   – natural sentence for text-to-image training, with OCR text embedded
    """

    def __init__(self, device: str = "cuda"):
        self.device     = device
        self.model_type = "blip2"
        self._load()

    def _load(self):
        try:
            print(f"[CAPTION] Loading BLIP-2 ({BLIP2_MODEL_NAME}) …", flush=True)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.processor = Blip2Processor.from_pretrained(BLIP2_MODEL_NAME)
            self.model     = Blip2ForConditionalGeneration.from_pretrained(
                BLIP2_MODEL_NAME, torch_dtype=dtype
            ).to(self.device)
            self.model.eval()
            print("[CAPTION] BLIP-2 ready.", flush=True)
        except Exception as e:
            print(f"[CAPTION] BLIP-2 failed ({e}), falling back to BLIP-large.", flush=True)
            self.model_type = "blip"
            self.processor  = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
            self.model      = BlipForConditionalGeneration.from_pretrained(
                BLIP_MODEL_NAME
            ).to(self.device)
            self.model.eval()

    @torch.no_grad()
    def _generate(self, image: Image.Image, text_prompt: str) -> str:
        image = image.convert("RGB")
        if self.model_type == "blip2":
            inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
        else:
            inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=CAPTION_MAX_LEN,
            min_length=CAPTION_MIN_LEN,
            num_beams=NUM_BEAMS,
            do_sample=False,
        )
        return self._clean(self.processor.decode(output_ids[0], skip_special_tokens=True))

    @staticmethod
    def _clean(text: str) -> str:
        """Strip prompt echoes and normalise capitalisation."""
        text = text.strip()
        noise = [
            "describe this image in one natural sentence.",
            "describe this image in one sentence.",
            "describe this image:",
            "write one natural sentence describing this image.",
            "the image shows",
            "this image shows",
            "the photo shows",
            "this photo shows",
            "the image features",
            "this image features",
            "this is a photograph of",
            "this is a",
            "this is an",
        ]
        lower = text.lower()
        for phrase in noise:
            if lower.startswith(phrase):
                for sep in [":", ".", "\n"]:
                    idx = text.find(sep)
                    if idx != -1 and len(text) - idx > 10:
                        text  = text[idx + 1:].strip()
                        lower = text.lower()
                        break
                break
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text

    def generate(self, image: Image.Image, ocr_text: str) -> Tuple[str, str]:
        """
        Return (caption, training_prompt).

        caption         – general visual description, OCR text NOT forced in
        training_prompt – natural sentence that embeds `ocr_text` in quotes.
                          BLIP-2 is prompted first; if the model fails to
                          include the exact text, a reliable template is used.
        """
        try:
            # ── Caption (no OCR context) ──────────────────────────────────────
            caption = self._generate(image, "Describe this image in one natural sentence.")
            if not caption or len(caption) < 8:
                caption = "A photograph"

            # ── Training prompt ───────────────────────────────────────────────
            if not ocr_text:
                return caption, caption

            if self.model_type == "blip2":
                instruction = (
                    f'Write one natural sentence describing this image. '
                    f'The image contains visible text: "{ocr_text}". '
                    f'Your sentence must include the exact text "{ocr_text}" '
                    f'in quotation marks and describe what object or surface '
                    f'displays it. Start directly with the subject.'
                )
                raw_prompt = self._generate(image, instruction)
            else:
                raw_prompt = ""

            # Validate: the OCR text must appear verbatim in the prompt
            if raw_prompt and ocr_text.lower() in raw_prompt.lower() and len(raw_prompt) >= 15:
                training_prompt = raw_prompt
            else:
                # Template fallback – always correct for training data
                caption_clean   = caption.rstrip(".,;!?").strip()
                training_prompt = f'{caption_clean}, with the text "{ocr_text}" clearly visible'

            return caption, training_prompt

        except Exception as e:
            print(f"[CAPTION] Error: {e}", flush=True)
            fb_caption = "A photograph"
            fb_prompt  = f'A photograph with the text "{ocr_text}" clearly visible'
            return fb_caption, fb_prompt


# ─── Annotation / image utility functions ────────────────────────────────────

def is_valid_text(text: str) -> bool:
    text = text.strip()
    if not (MIN_TEXT_LEN <= len(text) <= MAX_TEXT_LEN):
        return False
    if len(text.split()) > MAX_WORDS:
        return False
    if not re.search(r"[A-Za-z]", text):
        return False
    if re.search(r"[^\x00-\x7F]", text):
        return False
    return True


def is_english_langdetect(text: str) -> bool:
    if not _LANGDETECT_OK:
        return True
    try:
        return detect(text) == "en"
    except Exception:
        return True


def extract_english_texts(annotations: List[Dict]) -> List[Dict]:
    results = []
    for ann in annotations:
        if not ann.get("valid", True):
            continue
        if ann.get("illegibility", False):
            continue
        text = ann.get("text", "").strip()
        if not text:
            continue
        lang = ann.get("language", "").lower()
        if lang and lang not in ("latin", "english", ""):
            continue
        if not is_valid_text(text):
            continue
        if len(text.split()) > 1 and not is_english_langdetect(text):
            continue
        polygon = ann.get("polygon", [])
        if not polygon or len(polygon) < 3:
            continue
        poly  = np.array(polygon)
        x_min, y_min = poly.min(axis=0)
        x_max, y_max = poly.max(axis=0)
        bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
        if bbox[3] < MIN_BBOX_HEIGHT:
            continue
        results.append({"text": text, "bbox": bbox, "polygon": polygon, "language": lang})
    return results


def is_dense_text_image(annotations: List[Dict]) -> bool:
    valid = extract_english_texts(annotations)
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
    total_chars = sum(
        len(ann.get("text", "")) for ann in annotations if ann.get("valid", True)
    )
    return total_chars > MAX_TOTAL_CHARS * 2


def select_best_text(english_texts: List[Dict]) -> Optional[Dict]:
    if not english_texts:
        return None
    # Sort by score only (using index as tiebreaker avoids comparing dicts)
    scored = [(len(e["text"]) + len(e["text"].split()) * 5, i, e)
              for i, e in enumerate(english_texts)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][2]


def _is_annotation_consistent_with_phrase(anno: str, phrase: str) -> bool:
    if not anno or not phrase:
        return False
    a_n = _normalize(anno)
    p_n = _normalize(phrase)
    if a_n in p_n or p_n in a_n:
        return True
    return bool(set(a_n.split()) & set(p_n.split()))


def check_image_quality(img_array: np.ndarray) -> Tuple[bool, Dict]:
    try:
        if img_array is None:
            return False, {}
        h, w = img_array.shape[:2]
        if h < MIN_RESOLUTION or w < MIN_RESOLUTION:
            return False, {}
        gray       = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        contrast   = float(np.std(gray))
        if sharpness  < MIN_SHARPNESS:
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


# ─── Dataset Creator ──────────────────────────────────────────────────────────

class DatasetCreator:

    def __init__(
        self,
        output_dir:    str,
        max_images:    int  = TARGET_IMAGES,
        subsets:       Optional[List[str]] = None,
        streaming:     bool = True,
        caption_model: str  = "blip2",
    ):
        self.base_dir = Path(output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(
            [d for d in self.base_dir.glob("v[0-9]*") if d.is_dir()],
            key=lambda d: int(d.name[1:])
        )
        version      = (int(existing[-1].name[1:]) + 1) if existing else 1
        self.out_dir = self.base_dir / f"v{version}"
        print(f"[DIR] Output → {self.out_dir}", flush=True)

        for split in ("train", "val", "test"):
            (self.out_dir / split / "images").mkdir(parents=True, exist_ok=True)

        self.max_images = max_images
        self.subsets    = subsets or DATASET_SUBSETS
        self.streaming  = streaming
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DEVICE] {self.device}", flush=True)
        if self.device == "cuda":
            print(f"[GPU] {torch.cuda.get_device_name(0)}", flush=True)

        # Initialise EasyOCR early (downloads model once)
        get_ocr_reader(gpu=(self.device == "cuda"))

        # Initialise BLIP-2
        self.caption_gen = CaptionGenerator(device=self.device)

    # ── Dataset loading ───────────────────────────────────────────────────────

    def load_dataset(self):
        from datasets import load_dataset

        print(f"[DATASET] Loading {DATASET_NAME} …", flush=True)
        all_datasets = []
        for subset in self.subsets:
            print(f"  subset: {subset} …", flush=True)
            try:
                ds = load_dataset(
                    DATASET_NAME, subset,
                    streaming=self.streaming,
                )
                split_name = "train" if "train" in ds else next(iter(ds))
                all_datasets.append((subset, ds[split_name]))
                print(f"  {subset}: loaded (split={split_name})", flush=True)
            except Exception as e:
                print(f"  WARNING: failed to load {subset}: {e}", flush=True)

        if not all_datasets:
            raise RuntimeError("No dataset subsets could be loaded!")

        print(f"[DATASET] {len(all_datasets)} subset(s) ready.", flush=True)
        return all_datasets

    # ── Per-sample processing ─────────────────────────────────────────────────

    def process_sample(self, sample: Dict, subset_name: str) -> Optional[Dict]:
        # 1. Annotation filtering
        annotations = sample.get("annotations", [])
        if not annotations:
            return None
        if is_dense_text_image(annotations):
            return None
        english_texts = extract_english_texts(annotations)
        if not english_texts:
            return None
        best = select_best_text(english_texts)
        if not best:
            return None

        # 2. Image quality
        pil_image = sample.get("image")
        if pil_image is None:
            return None
        pil_image = pil_image.convert("RGB")
        ok, quality_metrics = check_image_quality(np.array(pil_image))
        if not ok:
            return None

        anno_text = best["text"]
        anno_bbox = best["bbox"]

        # 3. EasyOCR: verify the annotation text is readable
        ocr_token, ocr_conf = verify_text_with_ocr(
            pil_image, expected_text=anno_text, bbox=anno_bbox
        )
        if ocr_token is None:
            return None

        # 4. EasyOCR: reconstruct the full visible phrase
        reconstructed, phrase_conf = reconstruct_phrase_with_easyocr(
            pil_image, anchor_bbox=anno_bbox
        )

        if reconstructed is not None:
            if _is_annotation_consistent_with_phrase(anno_text, reconstructed):
                if len(reconstructed.split()) >= len(anno_text.split()):
                    final_text, final_conf, phrase_reconstructed = reconstructed, phrase_conf, True
                else:
                    final_text, final_conf, phrase_reconstructed = anno_text, ocr_conf, False
            else:
                return None   # OCR and annotation disagree – reject
        else:
            final_text, final_conf, phrase_reconstructed = anno_text, ocr_conf, False

        # 5 & 6. BLIP-2: caption + training prompt
        caption, prompt = self.caption_gen.generate(pil_image, ocr_text=final_text)

        # Hard guarantee: final_text must appear in the prompt
        if not prompt or final_text.lower() not in prompt.lower():
            caption_clean = caption.rstrip(".,;!?").strip()
            prompt = f'{caption_clean}, with the text "{final_text}" clearly visible'

        return {
            "image":    pil_image,
            "image_id": sample.get("img_name", "unknown"),
            "text":     final_text,
            "caption":  caption,
            "prompt":   prompt,
            "subset":   subset_name,
            "metadata": {
                "ocr_confidence":       round(final_conf, 2),
                "phrase_reconstructed": phrase_reconstructed,
                "text_length":          len(final_text),
                "word_count":           len(final_text.split()),
                "caption_model":        self.caption_gen.model_type,
                "ocr_backend":          "easyocr",
                "source":               f"AnyWord-3M/{subset_name}",
                **quality_metrics,
            },
        }

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> List[Dict]:
        all_datasets = self.load_dataset()
        accepted: List[Dict] = []
        stats = {
            "total_seen":        0,
            "rej_no_annotation": 0,
            "rej_density":       0,
            "rej_no_english":    0,
            "rej_quality":       0,
            "rej_ocr":           0,
        }

        print(f"\n[PROC] Starting (target: {self.max_images}) …", flush=True)

        for subset_name, ds in all_datasets:
            if len(accepted) >= self.max_images:
                break
            print(f"\n[SUBSET] {subset_name}", flush=True)

            for sample in tqdm(ds, desc=subset_name):
                if len(accepted) >= self.max_images:
                    break

                stats["total_seen"] += 1
                result = self.process_sample(sample, subset_name)

                if result is None:
                    anns = sample.get("annotations", [])
                    if not anns:
                        stats["rej_no_annotation"] += 1
                    elif is_dense_text_image(anns):
                        stats["rej_density"] += 1
                    elif not extract_english_texts(anns):
                        stats["rej_no_english"] += 1
                    else:
                        stats["rej_quality"] += 1
                    continue

                accepted.append(result)

                if len(accepted) % 50 == 0:
                    print(
                        f"  ACCEPTED {len(accepted):4d}/{self.max_images}"
                        f"  seen={stats['total_seen']}"
                        f"  text='{result['text']}'",
                        flush=True,
                    )
                    print(f"    Caption : {result['caption'][:100]}", flush=True)
                    print(f"    Prompt  : {result['prompt'][:120]}", flush=True)

        print("\n[SUMMARY]", flush=True)
        for k, v in stats.items():
            print(f"  {k:<22}: {v}", flush=True)

        return accepted

    # ── Saving ────────────────────────────────────────────────────────────────

    def save(self, records: List[Dict]):
        if not records:
            print("[WARN] No records to save.", flush=True)
            return

        n       = len(records)
        n_train = int(n * 0.8)
        n_val   = int(n * 0.1)
        splits  = [
            ("train", records[:n_train]),
            ("val",   records[n_train: n_train + n_val]),
            ("test",  records[n_train + n_val:]),
        ]

        print("[SAVE] Writing images and annotations …", flush=True)
        all_records: List[Dict] = []
        global_id = 0

        for split_name, split_records in splits:
            split_data: List[Dict] = []
            for rec in split_records:
                filename = f"img_{global_id:04d}.jpg"
                dst = self.out_dir / split_name / "images" / filename
                rec["image"].save(str(dst), "JPEG", quality=95)

                entry = {
                    "id":       global_id,
                    "image_id": rec["image_id"],
                    "filename": filename,
                    "filepath": f"{split_name}/images/{filename}",
                    "split":    split_name,
                    "text":     rec["text"],
                    "caption":  rec["caption"],
                    "prompt":   rec["prompt"],
                    "metadata": rec["metadata"],
                }
                split_data.append(entry)
                all_records.append(entry)
                global_id += 1

            with open(self.out_dir / f"{split_name}.json", "w") as fh:
                json.dump({"split": split_name, "count": len(split_data),
                           "data": split_data}, fh, indent=2)

        with open(self.out_dir / "dataset_complete.json", "w") as fh:
            json.dump({
                "metadata": {
                    "version":        VERSION,
                    "description":    "Text-in-image dataset. OCR: EasyOCR. Captions: BLIP-2.",
                    "ocr_backend":    "easyocr",
                    "caption_model":  self.caption_gen.model_type,
                    "source_dataset": DATASET_NAME,
                    "subsets_used":   self.subsets,
                    "thresholds": {
                        "min_resolution": MIN_RESOLUTION,
                        "min_sharpness":  MIN_SHARPNESS,
                        "min_ocr_conf":   MIN_OCR_CONF,
                        "min_coverage":   MIN_OCR_COVERAGE,
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

        print(
            f"[SAVED] {len(all_records)} total  "
            f"train={len(splits[0][1])}  val={len(splits[1][1])}  "
            f"test={len(splits[2][1])}",
            flush=True,
        )
        print(f"[PATH]  {self.out_dir}", flush=True)

        print("\n[SAMPLES]", flush=True)
        for e in all_records[:5]:
            print(f"\n  id={e['id']}  text='{e['text']}'", flush=True)
            print(f"    Caption : {e['caption']}", flush=True)
            print(f"    Prompt  : {e['prompt']}", flush=True)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    print(f"[START] Text-in-Image Dataset Pipeline v{VERSION}", flush=True)

    parser = argparse.ArgumentParser(
        description="Text-in-image dataset creator  (OCR: EasyOCR | Captions: BLIP-2)"
    )
    parser.add_argument("--output",        default="dataset_output",
                        help="Output directory")
    parser.add_argument("--max-images",    type=int, default=TARGET_IMAGES,
                        help=f"Target number of images (default: {TARGET_IMAGES})")
    parser.add_argument("--subsets",       nargs="+", default=None,
                        help="AnyWord-3M subsets (default: all four)")
    parser.add_argument("--no-streaming",  action="store_true",
                        help="Download full dataset instead of streaming")
    parser.add_argument("--caption-model", choices=["blip", "blip2"], default="blip2",
                        help="Caption model (default: blip2)")
    args = parser.parse_args()

    print(
        f"[ARGS]  output={args.output}  max={args.max_images}"
        f"  caption_model={args.caption_model}",
        flush=True,
    )

    creator = DatasetCreator(
        output_dir    = args.output,
        max_images    = args.max_images,
        subsets       = args.subsets,
        streaming     = not args.no_streaming,
        caption_model = args.caption_model,
    )
    records = creator.run()
    creator.save(records)
    print("[END]", flush=True)


if __name__ == "__main__":
    main()