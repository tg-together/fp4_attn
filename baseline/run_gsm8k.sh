#!/bin/bash
# =============================================================================
# GSM8K (Grade School Math 8K) Evaluation
# =============================================================================
# GSM8K uses 8-shot chain-of-thought prompting for grade school math problems
# Memory-optimized GPU allocation based on model size
# =============================================================================

export HF_HOME=/scratch/huggingface
<<<<<<< HEAD
export HF_TOKEN=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
=======
export HF_TOKEN=XXX

>>>>>>> 7fc8fae454dd74fb65a2aa12555f5ca4fe413997
export HF_HOME_DATASETS=/scratch/huggingface/datasets
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Starting GSM8K Evaluation Suite"
echo "=============================="

# Small Models (4B-8B): Single GPU, optimized batch size
echo "[1/6] Qwen3-4B-Thinking (GSM8K CoT)"
CUDA_VISIBLE_DEVICES=0 python baseline.py "Qwen/Qwen3-4B-Thinking-2507" --task gsm8k_cot --num_repeats 3 --batch_size 16&>logs/gsm8k_cot_qwen3_4b_thinking_2507.txt&

echo "[2/6] Qwen3-8B (GSM8K CoT)"
CUDA_VISIBLE_DEVICES=1 python baseline.py "Qwen/Qwen3-8B" --task gsm8k_cot --num_repeats 3 --batch_size 16&>logs/gsm8k_cot_qwen3_8b.txt&

# echo "[3/6] Llama-3.1-8B (GSM8K CoT Llama)"
# CUDA_VISIBLE_DEVICES=6 python baseline.py "meta-llama/Llama-3.2-3B-Instruct" --task gsm8k_cot_llama --num_repeats 3 --batch_size 32&>logs/gsm8k_cot_llama_llama3_2_3b_instruct.txt&

# echo "[4/6] Llama-3.1-8B (GSM8K CoT Llama)"
# CUDA_VISIBLE_DEVICES=7 python baseline.py "meta-llama/Llama-3.1-8B-Instruct" --task gsm8k_cot_llama --num_repeats 3 --batch_size 32&>logs/gsm8k_cot_llama_llama3_1_8b_instruct.txt&

# # Medium Model (14B): Multi-GPU
# echo "[4/6] Qwen3-14B (GSM8K CoT)"
CUDA_VISIBLE_DEVICES=3 python baseline.py "Qwen/Qwen3-14B" --task gsm8k_cot --num_repeats 3 --batch_size 16&>logs/gsm8k_cot_qwen3_14b.txt&      

# Large Model (70B): 4 GPUs
# echo "[5/6] Llama-3.3-70B (GSM8K CoT Llama)"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python baseline.py "meta-llama/Llama-3.3-70B-Instruct" --task gsm8k_cot_llama --num_repeats 3 --batch_size 1

# # Zero-shot comparison (optional)
# echo "[6/6] Qwen3-4B-Thinking (GSM8K Zero-shot)"
# CUDA_VISIBLE_DEVICES=2 python baseline.py "Qwen/Qwen3-4B-Thinking-2507" --task gsm8k_cot_zeroshot --num_repeats 3 --batch_size 16

echo "=============================="
echo "GSM8K Evaluation Suite Complete!"
echo "=============================="


