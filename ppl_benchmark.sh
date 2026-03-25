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
LOG_OUTPUT_PATH="logs/ppl_benchmark"

# Use unbuffered Python output for better logging
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_OUTPUT_PATH}/fp4/"

MODEL="meta-llama/Llama-3.1-8B"

#Qwen/Qwen3-8B
#Qwen/Qwen3-4B
#meta-llama/Llama-3.1-70B

CUDA_VISIBLE_DEVICES=0 python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 > "${LOG_OUTPUT_PATH}/temp2.txt" 2>&1

# CUDA_VISIBLE_DEVICES=0,1 SA3=True python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "${LOG_OUTPUT_PATH}/fp4/${MODEL##*/}_ppl_sa3.txt" 2>&1

# CUDA_VISIBLE_DEVICES=0,1 SA3=True SEARCH=True python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "${LOG_OUTPUT_PATH}/fp4/${MODEL##*/}_ppl_sa3_search.txt" 2>&1 

# CUDA_VISIBLE_DEVICES=0,1 NAIVE=True python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "${LOG_OUTPUT_PATH}/fp4/${MODEL##*/}_ppl_naive.txt" 2>&1 

# CUDA_VISIBLE_DEVICES=0,1 NAIVE=True SEARCH=True python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "${LOG_OUTPUT_PATH}/fp4/${MODEL##*/}_ppl_naive_search.txt" 2>&1 

# CUDA_VISIBLE_DEVICES=0,1 python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "${LOG_OUTPUT_PATH}/fp4/${MODEL##*/}_ppl_ssa.txt" 2>&1 
