#!/bin/bash
# =============================================================================
# GSM8K (Grade School Math 8K) Evaluation
# =============================================================================
# GSM8K uses 8-shot chain-of-thought prompting for grade school math problems
# Memory-optimized GPU allocation based on model size
# =============================================================================


export HF_HOME_DATASETS=/scratch/huggingface/datasets
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="/scratch/huggingface"
export HF_TOKEN="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"

export PYTHONUNBUFFERED=1
echo "Starting GSM8K Evaluation Suite"
echo "=============================="

mkdir -p "logs/gsm8k_benchmark"
# Small Models (4B-8B): Single GPU, optimized batch size
echo "[1/6] Qwen3-4B-Thinking (GSM8K CoT)"
CUDA_VISIBLE_DEVICES=0 python lmeval_main.py "Qwen/Qwen3-4B-Thinking-2507" --task gsm8k_cot --num_repeats 3 --batch_size 16 > "logs/gsm8k_benchmark/gsm8k_cot_qwen3_4b_thinking_2507.txt" 2>&1

echo "[2/6] Qwen3-8B (GSM8K CoT)"
CUDA_VISIBLE_DEVICES=1 python lmeval_main.py "Qwen/Qwen3-8B" --task gsm8k_cot --num_repeats 3 --batch_size 16 > "logs/gsm8k_benchmark/gsm8k_cot_qwen3_8b.txt" 2>&1

echo "[3/6] Llama-3.1-3B (GSM8K CoT Llama)"
CUDA_VISIBLE_DEVICES=2 python lmeval_main.py "meta-llama/Llama-3.2-3B-Instruct" --task gsm8k_cot_llama --num_repeats 3 --batch_size 32 > "logs/gsm8k_benchmark/gsm8k_cot_llama_llama3_2_3b_instruct.txt" 2>&1

echo "[4/6] Llama-3.1-8B (GSM8K CoT Llama)"
CUDA_VISIBLE_DEVICES=3 python lmeval_main.py "meta-llama/Llama-3.1-8B-Instruct" --task gsm8k_cot_llama --num_repeats 3 --batch_size 32 > "logs/gsm8k_benchmark/gsm8k_cot_llama_llama3_1_8b_instruct.txt" 2>&1

# # Medium Model (14B): Multi-GPU
echo "[4/6] Qwen3-14B (GSM8K CoT)"
CUDA_VISIBLE_DEVICES=4 python lmeval_main.py "Qwen/Qwen3-14B" --task gsm8k_cot --num_repeats 3 --batch_size 16 > "logs/gsm8k_benchmark/gsm8k_cot_qwen3_14b.txt" 2>&1

# Large Model (70B): 4 GPUs
# echo "[5/6] Llama-3.3-70B (GSM8K CoT Llama)"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python lmeval_main.py "meta-llama/Llama-3.3-70B-Instruct" --task gsm8k_cot_llama --num_repeats 3 --batch_size 1&>logs/gsm8k_benchmark/gsm8k_cot_llama_llama3_3_70b.txt&

echo "=============================="
echo "GSM8K Evaluation Suite Complete!"
echo "=============================="

