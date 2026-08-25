#!/usr/bin/env bash
# TruthLens - Dataset Download Script
# Run this on your own machine (requires internet access).
# Prerequisites: git, wget or curl, unzip, and the Kaggle CLI
#   (pip install kaggle --break-system-packages) with kaggle.json
#   credentials placed in ~/.kaggle/kaggle.json for the ISOT step.

set -e  # stop on first error

mkdir -p data/raw/liar data/raw/fakenewsnet data/raw/isot

# ---------------------------------------------------------------------
# 1. LIAR (primary dataset)
# ---------------------------------------------------------------------
echo "Downloading LIAR..."
cd data/raw/liar
if command -v wget &> /dev/null; then
    wget -N https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
else
    curl -O https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
fi
unzip -o liar_dataset.zip
cd ../../../

# Fallback mirror if the UCSB link is unreachable:
# git clone https://github.com/thiagorainmaker77/liar_dataset.git data/raw/liar

# ---------------------------------------------------------------------
# 2. FakeNewsNet (secondary dataset - minimal CSVs, no Twitter API needed)
# ---------------------------------------------------------------------
echo "Downloading FakeNewsNet..."
cd data/raw/fakenewsnet
if [ ! -d "FakeNewsNet" ]; then
    git clone --depth 1 https://github.com/KaiDMML/FakeNewsNet.git
else
    echo "FakeNewsNet already cloned, pulling latest..."
    cd FakeNewsNet && git pull && cd ..
fi
cd ../../../

# ---------------------------------------------------------------------
# 3. ISOT (tertiary / baseline volume dataset, via Kaggle mirror)
# ---------------------------------------------------------------------
echo "Downloading ISOT..."
cd data/raw/isot
kaggle datasets download -d emineyetm/fake-news-detection-datasets
unzip -o fake-news-detection-datasets.zip
cd ../../../

echo ""
echo "All downloads complete. Run verify_datasets.py next to check integrity and schema."