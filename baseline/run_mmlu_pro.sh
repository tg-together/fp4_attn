#!/bin/bash
# =============================================================================
# MMLU (Massive Multitask Language Understanding) Evaluation
# =============================================================================
# MMLU uses 4-shot few-shot prompting for multitask language understanding
# Memory-optimized GPU allocation based on model size
# =============================================================================

export HF_HOME=/scratch/huggingface
export HF_TOKEN=XXX

export HF_HOME_DATASETS=/scratch/huggingface/datasets
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Starting MMLU Evaluation Suite"
echo "=============================="

# Small Models (4B-8B): Single GPU, optimized batch size
echo "[1/5] Qwen3-4B-Thinking (MMLU CoT Fewshot)"
CUDA_VISIBLE_DEVICES=2 python baseline.py "Qwen/Qwen3-4B-Thinking-2507" --task mmlu_flan_cot_fewshot --num_repeats 1 --num_fewshot 4 --batch_size 16 &>logs/mmlu_flan_cot_fewshot_qwen3_4b_thinking_2507.txt&

echo "[2/5] Qwen3-8B (MMLU CoT Fewshot)"
CUDA_VISIBLE_DEVICES=4 python baseline.py "Qwen/Qwen3-8B" --task mmlu_flan_cot_fewshot --num_repeats 1 --num_fewshot 4 --batch_size 16 &>logs/mmlu_flan_cot_fewshot_qwen3_8b.txt&

echo "[3/5] Llama-3.1-8B (MMLU CoT Fewshot)"
CUDA_VISIBLE_DEVICES=6 python baseline.py "meta-llama/Llama-3.1-8B-Instruct" --task mmlu_flan_cot_fewshot --num_repeats 1 --num_fewshot 4 --batch_size 16 &>logs/mmlu_flan_cot_fewshot_llama3_1_8b.txt&

echo "[3/5] Llama-3.1-8B (MMLU CoT Fewshot)"
CUDA_VISIBLE_DEVICES=7 python baseline.py "meta-llama/Llama-3.2-3B-Instruct" --task mmlu_flan_cot_fewshot --num_repeats 1 --num_fewshot 4 --batch_size 16 &>logs/mmlu_flan_cot_fewshot_llama3_2_3b.txt&

# Medium Model (14B): Multi-GPU
echo "[4/5] Qwen3-14B (MMLU CoT Fewshot)"
CUDA_VISIBLE_DEVICES=5 python baseline.py "Qwen/Qwen3-14B" --task mmlu_flan_cot_fewshot --num_repeats 1 --num_fewshot 4 --batch_size 16 &>logs/mmlu_flan_cot_fewshot_qwen3_14b.txt&

# Large Model (70B): 4 GPUs
# echo "[5/5] Llama-3.3-70B (MMLU CoT Fewshot)"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python baseline.py "meta-llama/Llama-3.3-70B-Instruct" --task mmlu_flan_cot_fewshot --num_repeats 1 --num_fewshot 4 --batch_size8

echo "=============================="
echo "MMLU Evaluation Suite Complete!"
echo "=============================="
