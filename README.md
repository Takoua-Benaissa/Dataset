# Text-in-Image Dataset Creation Pipeline — v7

A fully automated pipeline that builds a **high-quality image dataset for training text-to-image generation models**.

Each entry pairs a real photograph containing clearly visible text with:
- the **ground-truth annotation phrase** from TextOCR (not a raw OCR read)
- a **BLIP visual caption** of the scene
- a **ready-to-use training prompt** of the form:
  > *"A neon sign on the side of a building at night, with the text \"OPEN 24 HRS\" clearly visible, sharp focus, high resolution photography"*

**Key improvement over earlier versions:** word-level bboxes are merged into full line phrases before any filtering, so the dataset contains natural multi-word text ("COFFEE SHOP", "OPEN DAILY") instead of isolated single words.

---

## Why this dataset?

Text-to-image models (Stable Diffusion, DALL·E, etc.) notoriously struggle to render readable text inside generated images.  
Training or fine-tuning on a curated dataset where:
1. the text is **physically present and OCR-verified** (not just described in a caption),
2. the image is **sharp, well-lit and high-resolution**,
3. the prompt **explicitly names the text**,

…gives the model concrete examples of what "text clearly visible" actually looks like, significantly improving its text-rendering capability.

---

## Pipeline overview

```
TextOCR / TextCaps JSON annotations
        │
        ▼
[1] Phrase Grouping     ─── merge co-linear word bboxes into full line phrases
        │                   (e.g. "COFFEE" + "SHOP" → "COFFEE SHOP")
        ▼
[2] Density Filter      ─── reject images with > 3 text regions or > 25 total chars
        │                   (removes book pages, sign-cluttered scenes)
        ▼
[3] Best-phrase Select  ─── pick highest-scoring English phrase (≤ 5 words, 3–30 chars)
        │
        ▼
[4] Quality Filter      ─── resolution ≥ 300 px, sharpness ≥ 100, contrast ≥ 25
        │
        ▼
[5] OCR Gate            ─── Tesseract (PSM 7/6/11/3), confidence ≥ 85 %
        │                   verifies the phrase is actually readable
        │                   annotation text is canonical (OCR is gate only)
        ▼
[6] BLIP Captioning     ─── Salesforce/blip-image-captioning-base (GPU)
        │                   5-beam search, max 75 tokens
        ▼
[7] Prompt Building     ─── "{description}, with the text \"{phrase}\" clearly visible,
        │                    sharp focus, high resolution photography"
        ▼
[8] Dataset Split       ─── 80 % train / 10 % val / 10 % test
        │
        ▼
   dataset_output/vN/
   ├── train/images/
   ├── val/images/
   ├── test/images/
   ├── train.json
   ├── val.json
   ├── test.json
   └── dataset_complete.json
```

---

## Repository structure

```
Dataset_creation/
├── create_dataset_cluster.py   # Main pipeline (quality filter + OCR + BLIP + export)
├── download_train_images.py    # Download TextCaps train images from Flickr URLs
├── config.py                   # Centralised configuration (thresholds, paths, etc.)
├── job_creation_dataset.sh     # SLURM batch job script (GPU cluster)
├── setup.sh                    # One-time environment setup script
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Data sources

The pipeline uses **[TextOCR](https://textvqa.org/textocr/)** (Singh et al., 2021) as the primary annotation source, with optional **[TextCaps](https://textvqa.org/textcaps/)** (Sidorov et al., 2020) support.

### TextOCR annotations (recommended)
| File | Images | URL |
|------|--------|-----|
| `TextOCR_0.1_train.json` | 21 778 | [download](https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json) |
| `TextOCR_0.1_val.json`   |  3 124 | [download](https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json) |

### Images
Images are shared with **TextVQA** / **TextCaps** (Open Images COCO-style):
```bash
# Train images (~21 k, ~12 GB)
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip -d ~/data/train_images/

# Test images
wget https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip
unzip test_images.zip -d ~/data/test_images/
```

---

## Requirements

### System
- Python ≥ 3.10
- **Tesseract OCR** ≥ 4.1 with English language data:
  ```bash
  # Ubuntu / Debian
  sudo apt install tesseract-ocr tesseract-ocr-eng

  # CentOS / RHEL
  sudo yum install tesseract
  ```
- A **CUDA-capable GPU** is strongly recommended for BLIP (runs on CPU too, but ~50× slower)

### Python packages
```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
Pillow>=9.0.0
pytesseract>=0.3.10
opencv-python>=4.7.0
tqdm>=4.65.0
numpy>=1.24.0
```

---

## Quickstart — run locally or on any Linux machine

### Step 1 — Clone and set up the environment

```bash
git clone https://github.com/Takoua-Benaissa/Dataset_creation.git
cd Dataset_creation

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2 — Download annotations and images

```bash
mkdir -p ~/data

# TextOCR annotations
wget https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json \
     -O ~/data/TextOCR_0.1_train.json
wget https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json \
     -O ~/data/TextOCR_0.1_val.json

# Images
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip -P ~/data/
unzip ~/data/train_val_images.zip -d ~/data/train_images/
```

### Step 3 — Run the pipeline

```bash
python create_dataset_cluster.py \
    --annotation-files ~/data/TextOCR_0.1_train.json ~/data/TextOCR_0.1_val.json \
    --image-dirs       ~/data/train_images/ ~/data/test_images/ \
    --output           ./dataset_output/ \
    --max-images       100
```

Raise `--max-images` to 1000+ for production runs.

After completion you will find:

```
dataset_output/v1/
├── train/images/        # 80 images (80 %)
├── val/images/          # 10 images (10 %)
├── test/images/         # 10 images (10 %)
├── train.json
├── val.json
├── test.json
└── dataset_complete.json
```

Each JSON entry looks like:
```json
{
  "id": 0,
  "image_id": "b880baf0a6b997ef",
  "filename": "img_0000.jpg",
  "filepath": "train/images/img_0000.jpg",
  "split": "train",
  "text": "Brooklyn",
  "annotation_text": "Brooklyn",
  "ocr_text": "Brooklyn",
  "caption_blip": "A street sign with the words brooklyn bridge on it",
  "caption_textcaps": "",
  "prompt": "A street sign with the words brooklyn bridge on it, with the text \"Brooklyn\" clearly visible, sharp focus, high resolution photography",
  "metadata": {
    "ocr_confidence": 96.0,
    "text_length": 8,
    "word_count": 1,
    "source": "TextOCR",
    "resolution": [480, 640],
    "sharpness": 1074.25,
    "brightness": 136.6,
    "contrast": 67.22
  }
}
```

> `text` / `annotation_text` — canonical ground-truth phrase from TextOCR.  
> `ocr_text` — token Tesseract actually read (readability gate only).  
> `caption_blip` — BLIP-generated visual description.

---

## Run on a SLURM GPU cluster

If you are on an HPC cluster with SLURM (e.g. ENSTA Paris cluster):

### One-time setup
```bash
bash setup.sh
```

### Download data (on login node, no GPU needed)
```bash
# TextOCR annotations
wget https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json \
     -O ~/data/TextOCR_0.1_train.json
wget https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json \
     -O ~/data/TextOCR_0.1_val.json

# Images
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip -P ~/data/ &
```

### Submit the job
```bash
sbatch job_creation_dataset.sh
```

The script automatically:
- detects which GPU partition is available
- selects the best image source (train > combined train+test > test)
- sets all Tesseract paths
- saves versioned output (`dataset_output/v1/`, `v2/`, …)

### Monitor
```bash
squeue -u $USER
tail -f logs/dataset_<JOBID>.out
```

---

## Parameters

All thresholds are defined at the top of `create_dataset_cluster.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_RESOLUTION` | 300 px | Minimum image width and height |
| `MIN_SHARPNESS` | 100 | Laplacian variance — rejects blurry images |
| `BRIGHTNESS_MIN/MAX` | 40 / 220 | Rejects too dark or washed-out images |
| `MIN_CONTRAST` | 25 | Std-dev of gray channel — rejects flat images |
| `MAX_WORDS` | 5 | Maximum words in a selected phrase |
| `MIN_TEXT_LEN` | 3 chars | Minimum phrase length |
| `MAX_TEXT_LEN` | 30 chars | Maximum phrase length |
| `MAX_TEXT_REGIONS` | 3 | Max distinct text lines per image (density filter) |
| `MAX_TOTAL_CHARS` | 25 | Max total chars across all lines (density filter) |
| `MIN_BBOX_HEIGHT` | 30 px | Minimum bbox height — rejects tiny unreadable text |
| `MIN_OCR_CONF` | 85 % | Minimum Tesseract token confidence |
| `MIN_OCR_COVERAGE` | 0.30 | Min length overlap between annotation and OCR read |
| `TARGET_IMAGES` | 100 | Images per run — raise to 1000+ for production |

To generate a larger dataset simply pass `--max-images 1000`.

---

## Expected output quality

With the default settings, from ~18 000 train images you can expect:

- ~**25–35 %** acceptance rate (the rest are rejected by quality or OCR filters)
- Average OCR confidence: **> 85 %**
- Text length distribution: mostly **4–20 characters** (single words to short phrases)
- All images: **256 × 256 px**, JPEG quality 95

Sample accepted entries:

| Text | BLIP caption | Scene |
|------|-------------|-------|
| `BANANAS` | A cork board with a bunch of fruit on it | Price tag on market display |
| `Games` | The amazon fire hd tablet | Tablet showing app store |
| `CALCULATOR` | A calculator sitting on top of a table | Office desk |
| `OPEN 24 HRS` | A neon sign on the side of a building | Storefront at night |

---

## Citation

If you use TextOCR annotations, please cite:

```bibtex
@inproceedings{singh2021textocr,
  title     = {TextOCR: Towards large-scale end-to-end reasoning for arbitrary-shaped scene text},
  author    = {Singh, Amanpreet and Pang, Guan and Toh, Mandy and Huang, Jing and Galuba, Wojciech and Hassner, Tal},
  booktitle = {CVPR},
  year      = {2021}
}

@inproceedings{sidorov2020textcaps,
  title     = {TextCaps: a Dataset for Image Captioning with Reading Comprehension},
  author    = {Sidorov, Oleksii and Hu, Ronghang and Rohrbach, Marcus and Singh, Amanpreet},
  booktitle = {ECCV},
  year      = {2020}
}
```

---

## License

Code: MIT  
Images: subject to original Flickr / TextCaps licensing terms.
