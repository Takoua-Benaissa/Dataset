#!/bin/bash
#SBATCH --job-name=create_dataset
#SBATCH --output=logs/dataset_%j.out
#SBATCH --error=logs/dataset_%j.err
#SBATCH --partition=ENSTA-l40s
#SBATCH --exclude=ensta-l40s01.r2.enst.fr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail

echo "=========================================="
echo "  Text-to-Image Dataset Creation Pipeline"
echo "=========================================="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Date   : $(date)"
echo ""

# ── Project directory ────────────────────────────────────────────────────────
PROJECT_DIR="$HOME/dataset_creation_project"
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found"; exit 1; }

# ── Activate venv ────────────────────────────────────────────────────────────
source "$PROJECT_DIR/venv/bin/activate"

echo "Python : $(python --version)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "VRAM   : $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# ── Tesseract data path ───────────────────────────────────────────────────────
for _td in "$HOME/miniconda3/share/tessdata" \
           "/usr/share/tesseract-ocr/5/tessdata" \
           "/usr/share/tesseract-ocr/4.00/tessdata" \
           "/usr/share/tessdata"; do
    if [ -f "$_td/eng.traineddata" ]; then
        export TESSDATA_PREFIX="$_td"
        break
    fi
done
echo "Tesseract: $(tesseract --version 2>&1 | head -1)"
echo "TESSDATA : $TESSDATA_PREFIX"
echo ""

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_IMAGES="$HOME/data/train_images"   # 21 953 images (TextCaps train)
TEST_IMAGES="$HOME/data/test_images"     # 3 353 images  (TextCaps test)
OUTPUT_DIR="$PROJECT_DIR/dataset_output"
MAX_IMAGES=500
BATCH_SIZE=16

# ── Choose best available image source ───────────────────────────────────────
TRAIN_COUNT=$(ls "$TRAIN_IMAGES"/*.jpg 2>/dev/null | wc -l)
TEST_COUNT=$(ls  "$TEST_IMAGES"/*.jpg  2>/dev/null | wc -l)

echo "Available sources:"
echo "  train_images : $TRAIN_COUNT images"
echo "  test_images  : $TEST_COUNT images"
echo ""

if [ "$TRAIN_COUNT" -ge 5000 ]; then
    # Enough train images → use train set (gives most variety)
    IMAGES_DIR="$TRAIN_IMAGES"
    echo "Using: train_images ($TRAIN_COUNT images)"
elif [ "$TRAIN_COUNT" -ge 100 ]; then
    # Partial train download → combine with test via symlink dir
    COMBINED="$HOME/data/combined_images"
    mkdir -p "$COMBINED"
    find "$TRAIN_IMAGES" -maxdepth 1 -name "*.jpg" -exec ln -sf {} "$COMBINED/" \; 2>/dev/null || true
    find "$TEST_IMAGES"  -maxdepth 1 -name "*.jpg" -exec ln -sf {} "$COMBINED/" \; 2>/dev/null || true
    IMAGES_DIR="$COMBINED"
    COMBINED_COUNT=$(ls "$COMBINED"/*.jpg 2>/dev/null | wc -l)
    echo "Using: combined ($COMBINED_COUNT images)"
else
    # Fallback: test images only
    IMAGES_DIR="$TEST_IMAGES"
    echo "Using: test_images ($TEST_COUNT images) – train not ready yet"
fi

if [ ! -d "$IMAGES_DIR" ] || [ "$(ls "$IMAGES_DIR"/*.jpg 2>/dev/null | wc -l)" -eq 0 ]; then
    echo "ERROR: No images found in $IMAGES_DIR"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Images dir : $IMAGES_DIR"
echo "  Output dir : $OUTPUT_DIR"
echo "  Max images : $MAX_IMAGES"
echo "  Batch size : $BATCH_SIZE"
echo ""

# ── Run pipeline ─────────────────────────────────────────────────────────────
echo "Starting pipeline at $(date)..."
python -u create_dataset_cluster.py \
    --images    "$IMAGES_DIR" \
    --output    "$OUTPUT_DIR" \
    --max-images "$MAX_IMAGES" \
    --batch-size "$BATCH_SIZE"

# ── Report ────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Dataset created successfully!"
echo "=========================================="

LAST_VERSION=$(ls -d "$OUTPUT_DIR"/v* 2>/dev/null | sort -V | tail -1)
echo ""
echo "Output directory: $LAST_VERSION"
ls -lh "$LAST_VERSION" 2>/dev/null

echo ""
echo "Split counts:"
echo "  Train : $(find "$LAST_VERSION/train/images" -name '*.jpg' 2>/dev/null | wc -l) images"
echo "  Val   : $(find "$LAST_VERSION/val/images"   -name '*.jpg' 2>/dev/null | wc -l) images"
echo "  Test  : $(find "$LAST_VERSION/test/images"  -name '*.jpg' 2>/dev/null | wc -l) images"
echo ""
echo "Total dataset size: $(du -sh "$LAST_VERSION" 2>/dev/null | cut -f1)"
echo ""
echo "Job finished: $(date)"
