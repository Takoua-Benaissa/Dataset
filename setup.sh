#!/bin/bash
# ============================================================
# Environment Setup for Dataset Creation Pipeline
# Run once before submitting sbatch jobs.
# ============================================================

set -euo pipefail

PROJECT_DIR="$HOME/dataset_creation_project"
cd "$PROJECT_DIR"

echo "Setting up dataset creation environment..."
echo ""

# ---- 1. Python virtual environment ----
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/4] Virtual environment already exists."
fi

source venv/bin/activate
echo "  Python: $(python --version)"

# ---- 2. Install dependencies ----
echo "[2/4] Installing Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Installed: torch, transformers, pytesseract, opencv-python, etc."

# ---- 3. Verify Tesseract ----
echo "[3/4] Checking Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    echo "  Tesseract: $(tesseract --version 2>&1 | head -1)"
else
    echo "  WARNING: Tesseract not found in PATH."
    echo "  On Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-eng"
    echo "  On cluster: module load tesseract (if available)"
fi

# ---- 4. Verify data ----
echo "[4/4] Checking data files..."

ANNOTATIONS="$HOME/data/TextCaps_0.1_test.json"
IMAGES="$HOME/data/test_images"

if [ -f "$ANNOTATIONS" ]; then
    entries=$(python -c "import json; d=json.load(open('$ANNOTATIONS')); print(len(d.get('data',[])))")
    echo "  Annotations: $ANNOTATIONS ($entries entries)"
else
    echo "  WARNING: Annotations not found: $ANNOTATIONS"
fi

if [ -d "$IMAGES" ]; then
    count=$(find "$IMAGES" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
    echo "  Images: $IMAGES ($count files)"
else
    echo "  WARNING: Images directory not found: $IMAGES"
fi

# ---- Create output dirs ----
mkdir -p dataset_output logs

echo ""
echo "Setup complete!"
echo ""
echo "To submit the job:"
echo "  sbatch run_pipeline.sbatch"
echo ""
echo "To monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/dataset_*.out"
