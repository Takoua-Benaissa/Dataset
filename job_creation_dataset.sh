#!/bin/bash
#SBATCH --job-name=dataset_v12
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
echo "  Text-in-Image Dataset Pipeline  v12"
echo "  OCR: EasyOCR  |  Captions: BLIP-2"
echo "  Source: AnyWord-3M (HuggingFace)"
echo "=========================================="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Date   : $(date)"
echo ""

# ── Project directory ─────────────────────────────────────────────────────────
PROJECT_DIR="$HOME/dataset_creation_project"
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found"; exit 1; }

# ── Activate venv ─────────────────────────────────────────────────────────────
source "$PROJECT_DIR/venv/bin/activate"

echo "Python : $(python --version)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "VRAM   : $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR="$PROJECT_DIR/dataset_output_v12"
MAX_IMAGES=500
CAPTION_MODEL="blip2"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Configuration:"
echo "  Dataset       : stzhao/AnyWord-3M (HuggingFace, streaming)"
echo "  Subsets       : laion, OCR_COCO_Text, OCR_mlt2019, OCR_Art"
echo "  OCR           : EasyOCR"
echo "  Caption model : $CAPTION_MODEL"
echo "  Output        : $OUTPUT_DIR"
echo "  Max images    : $MAX_IMAGES"
echo "  HF cache      : $HF_HOME"
echo ""

# ── Install / verify required packages ───────────────────────────────────────
echo "[DEPS] Checking required packages …"

# transformers >= 4.30.0  (for BLIP-2)
REQUIRED_TRANSFORMERS="4.30.0"
CURRENT_TRANSFORMERS=$(pip show transformers 2>/dev/null | grep Version | cut -d' ' -f2 || echo "")
if [ -z "$CURRENT_TRANSFORMERS" ]; then
    echo "  Installing transformers >= $REQUIRED_TRANSFORMERS …"
    pip install --quiet "transformers>=$REQUIRED_TRANSFORMERS"
elif python -c "from packaging import version; exit(0 if version.parse('$CURRENT_TRANSFORMERS') >= version.parse('$REQUIRED_TRANSFORMERS') else 1)" 2>/dev/null; then
    echo "  transformers $CURRENT_TRANSFORMERS ✓"
else
    echo "  Upgrading transformers to >= $REQUIRED_TRANSFORMERS …"
    pip install --quiet --upgrade "transformers>=$REQUIRED_TRANSFORMERS"
fi

# easyocr
if ! pip show easyocr &>/dev/null; then
    echo "  Installing easyocr …"
    pip install --quiet easyocr
else
    echo "  easyocr ✓"
fi

# python-Levenshtein  (optional but improves coverage scoring)
if ! pip show python-Levenshtein &>/dev/null; then
    echo "  Installing python-Levenshtein …"
    pip install --quiet python-Levenshtein
else
    echo "  python-Levenshtein ✓"
fi

echo ""

# ── Run pipeline ──────────────────────────────────────────────────────────────
echo "Starting pipeline at $(date) …"
echo ""

python -u create_dataset_cluster.py \
    --output        "$OUTPUT_DIR" \
    --max-images    "$MAX_IMAGES" \
    --caption-model "$CAPTION_MODEL"

EXIT_CODE=$?

# ── Report ────────────────────────────────────────────────────────────────────
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "  ✅ Dataset created successfully!"
    echo "=========================================="

    LAST_VERSION=$(ls -d "$OUTPUT_DIR"/v* 2>/dev/null | sort -V | tail -1)

    if [ -n "$LAST_VERSION" ]; then
        echo ""
        echo "Output directory: $LAST_VERSION"
        ls -lh "$LAST_VERSION" 2>/dev/null

        echo ""
        echo "Split counts:"
        TRAIN_COUNT=$(find "$LAST_VERSION/train/images" -name '*.jpg' 2>/dev/null | wc -l)
        VAL_COUNT=$(find   "$LAST_VERSION/val/images"   -name '*.jpg' 2>/dev/null | wc -l)
        TEST_COUNT=$(find  "$LAST_VERSION/test/images"  -name '*.jpg' 2>/dev/null | wc -l)
        TOTAL_COUNT=$((TRAIN_COUNT + VAL_COUNT + TEST_COUNT))
        echo "  Train : $TRAIN_COUNT images"
        echo "  Val   : $VAL_COUNT images"
        echo "  Test  : $TEST_COUNT images"
        echo "  TOTAL : $TOTAL_COUNT images"
        echo ""
        echo "Dataset size: $(du -sh "$LAST_VERSION" 2>/dev/null | cut -f1)"

        # Show sample prompts
        if [ -f "$LAST_VERSION/dataset_complete.json" ]; then
            echo ""
            echo "Sample prompts (first 3 images):"
            python -c "
import json, sys
try:
    with open('$LAST_VERSION/dataset_complete.json') as f:
        data = json.load(f)
    for i, item in enumerate(data['data'][:3]):
        print(f\"\n  [{i+1}] text    : '{item['text']}'\")
        print(f\"       caption : {item['caption'][:100]}\")
        print(f\"       prompt  : {item['prompt'][:120]}\")
except Exception as e:
    print(f'Could not parse JSON: {e}', file=sys.stderr)
"
        fi
    fi
else
    echo "=========================================="
    echo "  ❌ Dataset creation FAILED"
    echo "=========================================="
    echo "Exit code: $EXIT_CODE"
    echo "Check error log: logs/dataset_${SLURM_JOB_ID}.err"
fi

echo ""
echo "Job finished: $(date)"
echo "=========================================="