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
echo "  Text-in-Image Dataset Pipeline  v7"
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
TRAIN_IMAGES="$HOME/data/train_images"   # TextOCR/TextCaps train images
TEST_IMAGES="$HOME/data/test_images"     # TextOCR/TextCaps test images
OUTPUT_DIR="$PROJECT_DIR/dataset_output"
MAX_IMAGES=100    # raise to 1000+ for production runs
BATCH_SIZE=16

# TextOCR ground-truth annotation files
TEXTOCR_TRAIN="$HOME/data/TextOCR_0.1_train.json"
TEXTOCR_VAL="$HOME/data/TextOCR_0.1_val.json"

# Verify annotation files exist
for f in "$TEXTOCR_TRAIN" "$TEXTOCR_VAL"; do
    if [ ! -f "$f" ]; then
        echo "WARNING: annotation file not found: $f"
    fi
done

TRAIN_COUNT=$(ls "$TRAIN_IMAGES"/*.jpg 2>/dev/null | wc -l)
TEST_COUNT=$(ls  "$TEST_IMAGES"/*.jpg  2>/dev/null | wc -l)

echo "Available images:"
echo "  train_images : $TRAIN_COUNT"
echo "  test_images  : $TEST_COUNT"
echo ""
echo "Configuration:"
echo "  Annotations : $TEXTOCR_TRAIN  $TEXTOCR_VAL"
echo "  Image dirs  : $TRAIN_IMAGES  $TEST_IMAGES"
echo "  Output dir  : $OUTPUT_DIR"
echo "  Max images  : $MAX_IMAGES"
echo ""

# ── Run pipeline ─────────────────────────────────────────────────────────────
echo "Starting pipeline at $(date)..."
python -u create_dataset_cluster.py \
    --annotation-files "$TEXTOCR_TRAIN" "$TEXTOCR_VAL" \
    --image-dirs       "$TRAIN_IMAGES"  "$TEST_IMAGES" \
    --output           "$OUTPUT_DIR" \
    --max-images       "$MAX_IMAGES" \
    --batch-size       "$BATCH_SIZE"

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
