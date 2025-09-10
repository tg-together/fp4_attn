#!/bin/bash
set -e  # exit on error

# Initialize conda
eval "$(conda shell.bash hook)"

# Create env
conda create -n fp4 python=3.12 -y

# Activate env
conda activate fp4

# Install packages
pip install lm_eval matplotlib

# Install local package if directory exists
if [ -d "fast_fp4" ]; then
  cd fast_fp4
  pip install -e .
else
  echo "Directory fast_fp4 not found. Skipping local install."
fi