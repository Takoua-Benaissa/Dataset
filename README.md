# AnyWord-3M Text-in-Image Pipeline

A fully automated pipeline that builds a **high-quality dataset of 1000 images for training text-to-image generation models** from the **AnyWord-3M** dataset.

It loads the AnyWord-3M dataset (designed for the [AnyText](https://github.com/tyxsspa/AnyText) project) via HuggingFace and applies strict filtering to produce entries where:

- the **text is short** (1-5 words), clearly visible, and **verified by OCR** (confidence > 85%)
- the text is **English only** (filtered by annotation language + langdetect)
- the image is **sharp, well-lit**, and of adequate resolution
- **dataset captions** (BLIP-2 generated) describe the scene
- a **ready-to-use training prompt** follows the form:
  > *"A storefront at night, with the text "OPEN" clearly visible, sharp focus, high resolution photography"*

---

## Why AnyWord-3M?

The [AnyWord-3M](https://huggingface.co/datasets/stzhao/AnyWord-3M) dataset was specifically created for the AnyText visual text generation project (ICLR 2024). It contains **~3M images** with:

- OCR annotations (text, polygons, language labels)
- BLIP-2 generated captions describing each image
- Images from LAION-400M, Noah-Wukong, and OCR benchmark datasets

This is more suitable for text-in-image generation training than combining separate OCR datasets, because the annotations and captions are specifically designed for this task.

**Note:** AnyWord-3M is multilingual (~1.6M Chinese, ~1.39M English, ~10K other). This pipeline filters for **English-only** text.

---

## Pipeline Overview

```
AnyWord-3M (HuggingFace: stzhao/AnyWord-3M)
        |
        v
[1] Stream Dataset     --- Load via HuggingFace datasets (streaming)
        |                  Subsets: laion, OCR_COCO_Text, OCR_mlt2019, OCR_Art
        v
[2] Language Filter    --- Keep only English text (annotation language field
        |                  + langdetect + ASCII check)
        v
[3] Text Filter        --- 1-5 words, valid characters, no dense-text scenes
        |
        v
[4] Image Quality      --- resolution >= 256 px, sharpness >= 80,
        |                  brightness 30-230, contrast >= 20
        v
[5] OCR Verification   --- Tesseract (PSM 6/7/11/3), confidence >= 85%
        |                  Bbox height >= 30 px
        v
[6] Prompt Building    --- "{caption}, with the text "{text}" clearly visible,
        |                   sharp focus, high resolution photography"
        v
[7] Dataset Split      --- 80% train / 10% val / 10% test
        |
        v
    dataset_output/v{N}/
    +-- train/images/   (800 images)
    +-- val/images/     (100 images)
    +-- test/images/    (100 images)
    +-- train.json
    +-- val.json
    +-- test.json
    +-- dataset_complete.json
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd dataset_creation_project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the pipeline (streaming -- no download needed)

```bash
python create_dataset_cluster.py \
    --output dataset_output \
    --max-images 1000
```

The dataset is streamed from HuggingFace, so no pre-download is required.

### 3. (Optional) Pre-download the dataset

```bash
# Download default subsets for offline use
python download_datasets.py --data-dir ~/data/anyword3m

# List available subsets
python download_datasets.py --list-subsets
```

### 4. Run on SLURM cluster

```bash
sbatch job_creation_dataset.sh
```

### 5. Monitor progress
```bash
squeue -u $USER                          # job status
tail -f logs/dataset_*.out               # live output
```

---

## Filtering Criteria

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| **Language** | English only | Annotation `language` field + langdetect |
| **Max words** | 1-5 | Keep only short, clear text |
| **OCR confidence** | >= 85% | Only clearly readable text |
| **Bbox height** | >= 30 px | Reject tiny unreadable text |
| **Dense text** | <= 4 regions, <= 80 total chars | Reject books, newspapers |
| **Resolution** | >= 256 px | Adequate for training |
| **Sharpness** | >= 80 (Laplacian) | No blurry images |
| **Brightness** | 30-230 | Not too dark or washed out |
| **Contrast** | >= 20 (std dev) | Good text/background separation |

---

## CLI Options

```
python create_dataset_cluster.py [OPTIONS]

Options:
  --output DIR          Output directory (default: dataset_output)
  --max-images N        Target number of images (default: 1000)
  --subsets S [S ...]   AnyWord-3M subsets to use (default: laion + OCR subsets)
  --no-streaming        Download full dataset instead of streaming
```

---

## Output Format

Each entry in `dataset_complete.json`:

```json
{
  "id": 42,
  "image_id": "00001234.jpg",
  "filename": "img_0042.jpg",
  "filepath": "train/images/img_0042.jpg",
  "split": "train",
  "text": "OPEN",
  "annotation_text": "OPEN",
  "ocr_text": "OPEN",
  "caption": "a storefront at night with a neon sign",
  "prompt": "a storefront at night with a neon sign, with the text \u201cOPEN\u201d clearly visible, sharp focus, high resolution photography",
  "metadata": {
    "ocr_confidence": 94.0,
    "text_length": 4,
    "word_count": 1,
    "source": "AnyWord-3M/laion",
    "resolution": [512, 512],
    "sharpness": 1523.45,
    "brightness": 128.3,
    "contrast": 62.1
  }
}
```

---

## Project Structure

```
+-- create_dataset_cluster.py   # Main pipeline (AnyWord-3M -> filtering -> export)
+-- download_datasets.py        # Optional: pre-download AnyWord-3M subsets
+-- job_creation_dataset.sh     # SLURM batch script (L40S GPU, 48GB RAM, 8h)
+-- requirements.txt            # Python dependencies
+-- setup.sh                    # One-command environment setup
+-- dataset_output/
    +-- v{N}/                   # Versioned output (auto-increments)
        +-- train/ val/ test/   # Image splits
        +-- *.json              # Annotations
```

---

## Requirements

- **Python** >= 3.10
- **Tesseract OCR** >= 4.0
- **Internet access** (for HuggingFace streaming) or pre-downloaded dataset
- **~2 GB disk** for 1000 output images + annotations

### Python packages
```
torch, torchvision, transformers, Pillow, pytesseract,
opencv-python, tqdm, numpy, datasets, langdetect
```

---

## SLURM Configuration

The provided `job_creation_dataset.sh` is configured for:
- **Partition:** ENSTA-l40s (node 01 excluded)
- **GPU:** 1x NVIDIA L40S (46 GB VRAM)
- **RAM:** 48 GB
- **Time:** 8 hours
- **Target:** 1000 images

---

## Data Source

- **AnyWord-3M:** Tuo et al., "AnyText: Multilingual Visual Text Generation And Editing," ICLR 2024
  https://github.com/tyxsspa/AnyText
  https://huggingface.co/datasets/stzhao/AnyWord-3M
