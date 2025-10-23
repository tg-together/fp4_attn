#!/bin/bash
set -e  # exit on error


git submodule update --init --recursive


eval "$(conda shell.bash hook)"

# Create env
conda create -n fp4 python=3.12 -y

# Activate env
conda activate fp4

# Install packages
pip install matplotlib glog


rm -rf ./third_party/lm-eval/lm_eval/tasks/gpqa
rm -rf ./third_party/lm-eval/lm_eval/tasks/gsm8k
rm -rf ./third_party/lm-eval/lm_eval/tasks/gsm8k_platinum

cp -r ./tasks/aime/ ./tasks/gpqa/ ./tasks/gsm8k/ ./tasks/gsm8k_platinum/ ./third_party/lm-eval/lm_eval/tasks/

python -m pip install -e third_party/lm-eval