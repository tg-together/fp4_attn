#!/bin/bash
# =============================================================================
# HumanEval (Code Generation) Evaluation
# =============================================================================
# HumanEval uses zero-shot code generation for Python programming problems
# Memory-optimized GPU allocation based on model size
# =============================================================================

export HF_HOME=/scratch/huggingface
export HF_TOKEN=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
export HF_HOME_DATASETS=/scratch/huggingface/datasets
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Starting HumanEval Evaluation Suite"
echo "==================================="

# Small Models (4B-8B): Single GPU, optimized batch size
echo "[1/5] Qwen3-4B-Thinking (HumanEval Instruct)"
CUDA_VISIBLE_DEVICES=0 python baseline.py "Qwen/Qwen3-4B-Thinking-2507" --task humaneval_instruct --num_repeats 3 --batch_size 8

echo "[2/5] Qwen3-8B (HumanEval Instruct)"
CUDA_VISIBLE_DEVICES=0 python baseline.py "Qwen/Qwen3-8B" --task humaneval_instruct --num_repeats 3 --batch_size 4

echo "[3/5] Llama-3.1-8B (HumanEval Instruct)"
CUDA_VISIBLE_DEVICES=1 python baseline.py "meta-llama/Llama-3.1-8B-Instruct" --task humaneval_instruct --num_repeats 3 --batch_size 4

# Medium Model (14B): Multi-GPU
echo "[4/5] Qwen3-14B (HumanEval Instruct)"
CUDA_VISIBLE_DEVICES=0,1 python baseline.py "Qwen/Qwen3-14B" --task humaneval_instruct --num_repeats 3 --batch_size 2

# Large Model (70B): 4 GPUs
echo "[5/5] Llama-3.3-70B (HumanEval Instruct)"
CUDA_VISIBLE_DEVICES=0,1,2,3 python baseline.py "meta-llama/Llama-3.3-70B-Instruct" --task humaneval_instruct --num_repeats 3 --batch_size 1

echo "==================================="
echo "HumanEval Evaluation Suite Complete!"
echo "==================================="