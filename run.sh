#!/bin/bash


export HF_HOME_DATASETS="/share/desa/nfs01/tg456/huggingface/datasets"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="/share/desa/nfs01/tg456/huggingface"
export VLLM_ASSETS_CACHE="/share/desa/nfs01/tg456/vllm_cache"
export XDG_CACHE_HOME="/share/desa/nfs01/tg456/vllm_cache"


OUTPUT_DIR="./outputs"
mkdir -p $OUTPUT_DIR/logs

# # export PYTHONUNBUFFERED=1



# # Small Models (4B-8B): Single GPU, optimized batch size
# echo "[1/6] Qwen3-4B-Thinking (GSM8K CoT)"
CUDA_VISIBLE_DEVICES=0 python lighteval_main.py --model ""Qwen/Qwen3-4B"" --task "aime24" --seed 0 --output_dir $OUTPUT_DIR

# python slrm.py > temp3.txt 2>&1