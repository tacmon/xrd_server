#!/bin/bash

# Configuration
# This script assumes it is run from the 'Novel-Space/' directory
BASE_DIR=$(pwd)
SRC_DIR="src"

echo "=== XRD-AutoAnalyzer: Inference Workflow ==="

# 1. Environment Check
if [ ! -d "$SRC_DIR" ]; then
    echo "Error: 'src/' directory not found. Please run this script from 'Novel-Space/'."
    exit 1
fi

# 2. Dependency Check
echo "Step 1: Checking dependencies..."
MISSING=false

# Check Spectra
if [ ! -d "Spectra" ] || [ ! "$(ls -A Spectra 2>/dev/null)" ]; then
    echo "  [Error] 'Spectra/' directory is missing or empty. Please add experimental data."
    MISSING=true
fi

# Check Models
if [ ! -d "Models" ]; then
    echo "  [Error] 'Models/' directory is missing. Please run ./train.sh first."
    MISSING=true
fi

# Check References
if [ ! -d "References" ]; then
    echo "  [Error] 'References/' directory is missing. Please run ./train.sh first."
    MISSING=true
fi

if [ "$MISSING" = true ]; then
    echo ""
    echo "Inference aborted due to missing dependencies."
    exit 1
fi

echo "  [OK] Dependencies found."

# 3. Running Inference
echo ""
echo "Step 2: Starting Phase Identification (CNN Inference)..."
echo "  (Using combined XRD + PDF analysis, generating plots and result.csv)"

python3 "$SRC_DIR/run_CNN.py" --inc_pdf

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Inference Workflow Complete! ==="
    echo "Predictions saved to: Novel-Space/result.csv"
    echo "Visualization saved to: Novel-Space/figure/"
else
    echo "  [Error] Inference script failed."
    exit 1
fi
