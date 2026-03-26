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

mkdir -p "logs/record/"

#FP4 Record Hessians

# # # Model and dataset configuration
# MODEL="meta-llama/Llama-3.1-8B"
# CUDA_VISIBLE_DEVICES=0  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --record_hessian 2>&1 | tee "logs/record/${MODEL##*/}_hessian_record.txt" &

# MODEL="meta-llama/Llama-3.1-70B"
# CUDA_VISIBLE_DEVICES=1  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --record_hessian --batch_size 1 2>&1 | tee "logs/record/${MODEL##*/}_hessian_record.txt" &

# MODEL="Qwen/Qwen3-8B"
# CUDA_VISIBLE_DEVICES=2  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --record_hessian 2>&1 | tee "logs/record/${MODEL##*/}_hessian_record.txt" &

# MODEL="Qwen/Qwen3-4B"
# CUDA_VISIBLE_DEVICES=3  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --record_hessian 2>&1 | tee "logs/record/${MODEL##*/}_hessian_record.txt" &

# MODEL="meta-llama/Llama-3.2-3B-Instruct"
# CUDA_VISIBLE_DEVICES=4  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --record_hessian 2>&1 | tee "logs/record/${MODEL##*/}_hessian_record.txt" &

# MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
# CUDA_VISIBLE_DEVICES=5  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --record_hessian 2>&1 | tee "logs/record/${MODEL##*/}_hessian_record.txt" &

# MODEL="meta-llama/Llama-3.3-70B-Instruct"
# CUDA_VISIBLE_DEVICES=6  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --record_hessian --batch_size 1 2>&1 | tee "logs/record/${MODEL##*/}_hessian_record.txt" &

# MODEL="Qwen/Qwen3-4B-Thinking-2507"
# CUDA_VISIBLE_DEVICES=7  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --record_hessian 2>&1 | tee "logs/record/${MODEL##*/}_hessian_record.txt" &

# MODEL="meta-llama/Llama-3.1-8B-Instruct"
# CUDA_VISIBLE_DEVICES=0  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --record_hessian 2>&1 | tee "logs/record/${MODEL##*/}_hessian_record.txt" &

MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
CUDA_VISIBLE_DEVICES=1  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --record_hessian 2>&1 | tee "logs/record/${MODEL##*/}_hessian_record.txt" &
