#!/bin/bash

# Select which mode to run: baseline, fp4, or kvquant
MODE="all"  # Options: baseline, fp4, kvquant, all

# Base path configuration
BASE_PATH="/anvil/scratch/x-yli19/tgupta/"

# Export environment variables based on base path
export HF_HOME_DATASETS="${BASE_PATH}/huggingface/datasets"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="${BASE_PATH}/huggingface"
export VLLM_ASSETS_CACHE="${BASE_PATH}/vllm_cache"
export XDG_CACHE_HOME="${BASE_PATH}/vllm_cache"

# Create directories if they don't exist
mkdir -p "${HF_HOME_DATASETS}"
mkdir -p "${HF_HOME}"
mkdir -p "${VLLM_ASSETS_CACHE}"

# Common log output path configuration
LOG_OUTPUT_PATH="logs/ablation/"

# Use unbuffered Python output for better logging
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_OUTPUT_PATH}/"

# List of models to evaluate - modify this list to add/remove models
MODEL="meta-llama/Llama-3.1-8B"


echo "SSA"
CUDA_VISIBLE_DEVICES=0 python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "${LOG_OUTPUT_PATH}/${MODEL##*/}_ppl_ssa.txt" 2>&1

echo "SSA No search"
CUDA_VISIBLE_DEVICES=0 SEARCH=False python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "${LOG_OUTPUT_PATH}/${MODEL##*/}_ppl_ssa_nosearch.txt" 2>&1

echo "SSA No IP"
CUDA_VISIBLE_DEVICES=0 IP=False python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "${LOG_OUTPUT_PATH}/${MODEL##*/}_ppl_ssa_noip.txt" 2>&1

echo "SSA No FP Mask"
CUDA_VISIBLE_DEVICES=0 FP_MASK=False python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "${LOG_OUTPUT_PATH}/${MODEL##*/}_ppl_ssa_mpkv.txt" 2>&1
