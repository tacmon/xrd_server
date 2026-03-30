# XRD-AutoAnalyzer-PyTorch Automated Analysis Toolbox

[English](README_en.md) | [简体中文](README.md)

This project is a handy toolbox designed for X-ray Diffraction (XRD) analysis. It not only helps you identify phases using CNNs but also streamlines your entire experiment workflow from data acquisition to final reporting.

---

## 🚀 1. Quick Start (Getting Started)

To keep things tidy, local datasets are ignored by Git. After cloning or downloading the project, follow these simple steps to set up your environment:

### Step 1: Configure Your API Key
Create a `.env` file in the `Novel-Space/` directory and add your Materials Project API Key:
```bash
# Novel-Space/.env
MP_API_KEY=your_actual_api_key_here
```

### Step 2: Initialize Your Lab Space
Run this script to automatically set up the necessary symbolic links:
```bash
cd Novel-Space
./setup_links.sh --init
```
> [!TIP]
> **New Feature: One-Click Lab Result Saving**
> If you've trained a great model in the `temp` area and want to keep it, running `--init` will kindly ask if you'd like to "promote" your results. Just give your experiment a name (e.g., `Trial_V1`), and the script will automatically archive all your files for you.

### Step 3: Launch Training
Ready to download structures and train your models? Simply run:
```bash
./train.sh
```
*   **Safety Tip**: To prevent accidental deletion of important production data, we've added a "Safety Belt." If you're linked to an established dataset instead of `temp`, the script will ask for manual `y` confirmation before cleaning anything up.

### Step 4: Run Automated Inference
Place your experimental patterns in the `Spectra/` directory, then run:
```bash
./inference.sh
```
*   **Handle Imperfect Data**: Don't worry about minor data issues. The script now automatically manages negative intensity offsets and handles BGMN refinement errors to keep your batch analysis moving.

---

## 🛠️ 2. The Interactive Toolbox (`tools.sh`)

If you need to plot graphs, create animations, or post-process your results, head to our "Central Console":
```bash
./tools.sh
```
In this interactive menu, you can:
*   **Process Results**: Filter material labels and confidence scores.
*   **Create Animations**: Compile sequence images into GIFs.
*   **Preview Patterns**: Quickly scan all experimental samples.

---

## 📖 3. For Developers

*   **Path-Agnostic**: All our Python scripts are optimized. Whether you run them from the root or the `Novel-Space/` directory, the scripts will find the correct paths automatically.
*   **Related Projects**: For original reference code, visit [XRD-1.0](https://github.com/tacmon/XRD-1.0).

This project aims to provide an efficient, robust, and enjoyable XRD analysis experience. If you have any suggestions, feel free to reach out!
