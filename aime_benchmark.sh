#!/bin/bash
# =============================================================================
# AIME (American Invitational Mathematics Examination) Evaluation
# =============================================================================
# AIME tasks are zero-shot math competition problems requiring detailed reasoning
# Memory-optimized GPU allocation and batch sizes based on model size
# =============================================================================



export HF_HOME_DATASETS=/scratch/huggingface/datasets
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="/scratch/huggingface"
export HF_TOKEN="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"

export PYTHONUNBUFFERED=1

echo "Starting AIME Evaluation Suite"
echo "==============================="

mkdir -p "logs/aime_benchmark"

# Small Models (4B-8B): Single GPU, optimized batch size
echo "[1/6] Qwen3-4B-Thinking (AIME24)"
CUDA_VISIBLE_DEVICES=0 python lmeval_main.py "Qwen/Qwen3-4B-Thinking-2507" --task aime24 --num_repeats 10 --batch_size 1 > "logs/aime_benchmark/aime24_qwen3_4b_thinking_2507.txt" 2>&1

echo "[2/6] Qwen3-8B (AIME24)"
CUDA_VISIBLE_DEVICES=1 python lmeval_main.py "Qwen/Qwen3-8B" --task aime24 --num_repeats 10 --batch_size 8 > "logs/aime_benchmark/aime24_qwen3_8b.txt" 2>&1

echo "[3/6] Llama-3.1-8B (AIME24)"
CUDA_VISIBLE_DEVICES=1 python lmeval_main.py "meta-llama/Llama-3.1-8B-Instruct" --task aime24 --num_repeats 10 --batch_size 4 > "logs/aime_benchmark/aime24_llama3_1_8b_instruct.txt" 2>&1

Medium Model (14B): Multi-GPU
echo "[4/6] Qwen3-14B (AIME24)"
CUDA_VISIBLE_DEVICES=2 python lmeval_main.py "Qwen/Qwen3-14B" --task aime24 --num_repeats 10 --batch_size  4 > "logs/aime_benchmark/aime24_qwen3_14b.txt" 2>&1


echo "[2/6] Qwen3-8B (AIME25)"
CUDA_VISIBLE_DEVICES=3 python lmeval_main.py "Qwen/Qwen3-8B" --task aime25 --num_repeats 10 --batch_size 8 > "logs/aime_benchmark/aime25_qwen3_8b.txt" 2>&1

# AIME25 Evaluations
echo "[5/6] Qwen3-4B-Thinking (AIME25)"
CUDA_VISIBLE_DEVICES=4 python lmeval_main.py "Qwen/Qwen3-4B-Thinking-2507" --task aime25 --num_repeats 10 --batch_size  8 > "logs/aime_benchmark/aime25_qwen3_4b_thinking_2507.txt" 2>&1

echo "[6/6] Qwen3-14B (AIME25)"
CUDA_VISIBLE_DEVICES=5 python lmeval_main.py "Qwen/Qwen3-14B" --task aime25 --num_repeats 10 --batch_size 4 > "logs/aime_benchmark/aime25_qwen3_14b.txt" 2>&1

echo "==============================="
echo "AIME Evaluation Suite Complete!"
echo "==============================="


