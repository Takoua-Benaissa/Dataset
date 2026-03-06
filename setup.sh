#!/bin/bash
# ============================================================
# Environment Setup for Dataset Creation Pipeline (v8)
# Source: AnyWord-3M (HuggingFace)
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
echo "  Installed: torch, transformers, pytesseract, opencv-python, datasets, langdetect, etc."

# ---- 3. Verify Tesseract ----
echo "[3/4] Checking Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    echo "  Tesseract: $(tesseract --version 2>&1 | head -1)"
else
    echo "  WARNING: Tesseract not found in PATH."
    echo "  On Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-eng"
    echo "  On cluster: module load tesseract (if available)"
fi

# ---- 4. Verify HuggingFace access ----
echo "[4/4] Checking HuggingFace datasets access..."
python -c "from datasets import load_dataset; print('  HuggingFace datasets library: OK')" 2>/dev/null || \
    echo "  WARNING: Could not import datasets library. Run: pip install datasets"

python -c "from langdetect import detect; print('  langdetect: OK')" 2>/dev/null || \
    echo "  WARNING: Could not import langdetect. Run: pip install langdetect"

# ---- Create output dirs ----
mkdir -p dataset_output logs

echo ""
echo "Setup complete!"
echo ""
echo "To run the pipeline (streaming, no download needed):"
echo "  python create_dataset_cluster.py --output dataset_output --max-images 1000"
echo ""
echo "To submit on SLURM:"
echo "  sbatch job_creation_dataset.sh"
echo ""
echo "To monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/dataset_*.out"
