#!/bin/bash

# Configuration
# This script assumes it is run from the 'Novel-Space/' directory
BASE_DIR=$(pwd)
SRC_DIR="src"

# Color constants for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "=== XRD-AutoAnalyzer: Auxiliary Toolbox ==="

# 1. Environment Check
if [ ! -d "$SRC_DIR" ]; then
    echo -e "${RED}[Error]${NC} 'src/' directory not found. Please run this script from 'Novel-Space/'."
    exit 1
fi

while true; do
    echo ""
    echo "------------------------------------------------"
    echo "Please select a task to perform:"
    echo "1. [Post-Processing] Process Analysis Results (result.csv -> processed_result.csv)"
    echo "2. [Visualization] Generate Reference GIFs (Create animations from figure/)"
    echo "3. [Visualization] Preview Experimental Spectra (Plot raw data from Spectra/)"
    echo "4. [Data Prep] Generate Theoretical Patterns (Pre-simulate patterns for references)"
    echo "q. Exit"
    echo "------------------------------------------------"
    read -p "Your choice: " choice

    case "$choice" in
        1)
            # Process results check
            if [ ! -f "result.csv" ]; then
                echo -e "${RED}[Error]${NC} 'result.csv' not found. Please run ./inference.sh first."
            else
                echo -e "${GREEN}[Running]${NC} Post-processing results..."
                python3 "$SRC_DIR/process_results.py"
            fi
            ;;
        2)
            # GIF generation check
            # make_gifs.py targets figure/real_data subfolders
            if [ ! -d "figure/real_data" ]; then
                echo -e "${RED}[Error]${NC} 'figure/real_data' directory not found. Please run ./inference.sh first."
            else
                echo -e "${GREEN}[Running]${NC} Generating GIF animations..."
                python3 "$SRC_DIR/make_gifs.py"
            fi
            ;;
        3)
            # Plot spectra check
            # Supports .txt, .xy, .gk files in Spectra/
            count=$(ls Spectra/*.txt Spectra/*.xy Spectra/*.gk 2>/dev/null | wc -l)
            if [ "$count" -eq 0 ]; then
                echo -e "${RED}[Error]${NC} No experimental spectra (.txt, .xy, .gk) found in 'Spectra/'."
            else
                echo -e "${GREEN}[Running]${NC} Plotting experimental spectra (output to figure/real_data/)..."
                python3 "$SRC_DIR/plot_real_spectra.py"
            fi
            ;;
        4)
            # Generate theoretical patterns check
            if [ ! -d "References" ] || [ ! "$(ls -A References 2>/dev/null)" ]; then
                echo -e "${RED}[Error]${NC} 'References/' directory is empty or missing. Please add CIF files."
            else
                echo -e "${GREEN}[Running]${NC} Pre-generating theoretical patterns..."
                python3 "$SRC_DIR/generate_theoretical_spectra.py"
            fi
            ;;
        q|Q|exit|Exit)
            echo "Exiting toolbox. Goodbye!"
            break
            ;;
        *)
            echo -e "${RED}[Invalid]${NC} Please enter a number between 1 and 4, or 'q' to exit."
            ;;
    esac
done
