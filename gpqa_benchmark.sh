#!/bin/bash

# Parse command line arguments
SEED=${1:-0}  # Default seed is 0

# Base path configuration
BASE_PATH="/share/desa/nfs01/tg456/"

# Export environment variables based on base path
export HF_HOME_DATASETS="${BASE_PATH}/huggingface/datasets"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="${BASE_PATH}/huggingface"
export VLLM_ASSETS_CACHE="${BASE_PATH}/vllm_cache"
export XDG_CACHE_HOME="${BASE_PATH}/vllm_cache"
# export HF_HUB_OFFLINE=1

# Create directories if they don't exist
mkdir -p "${HF_HOME_DATASETS}"
mkdir -p "${HF_HOME}"
mkdir -p "${VLLM_ASSETS_CACHE}"

# Common log output path configuration
LOG_OUTPUT_PATH="logs/gpqa_benchmark"
OUTPUT_PATH="outputs/gpqa_benchmark"

# Use unbuffered Python output for better logging
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_OUTPUT_PATH}/"
mkdir -p "${OUTPUT_PATH}/"







echo "Starting GPQA Evaluation Suite with seed: ${SEED}"
echo "==============================="


MODEL="Qwen/Qwen3-4B-Thinking-2507"
#deepseek-ai/DeepSeek-R1-Distill-Llama-8B
#Qwen/Qwen3-4B-Thinking-2507
#meta-llama/Llama-3.1-8B-Instruct

nvidia-smi


echo "${MODEL}"

CUDA_VISIBLE_DEVICES=0 python lighteval_main.py --model "${MODEL}" --task gpqa --seed ${SEED} --batch_size 1 --output_dir "${LOG_OUTPUT_PATH}/${MODEL##*/}" > "${LOG_OUTPUT_PATH}/${MODEL##*/}_seed${SEED}_baseline.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 python lighteval_main.py --model "${MODEL}" --task gpqa --seed ${SEED} --batch_size 1  --quantize --output_dir "${LOG_OUTPUT_PATH}/${MODEL##*/}" > "${LOG_OUTPUT_PATH}/${MODEL##*/}_seed${SEED}_ssa.txt" 2>&1

CUDA_VISIBLE_DEVICES=0 SA3=True python lighteval_main.py --model "${MODEL}" --task gpqa --seed ${SEED} --batch_size 1  --quantize --output_dir "${LOG_OUTPUT_PATH}/${MODEL##*/}" > "${LOG_OUTPUT_PATH}/${MODEL##*/}_seed${SEED}_sa3.txt" 2>&1

echo "==============================="
echo "GPQA Evaluation Suite Complete!"
echo "==============================="




