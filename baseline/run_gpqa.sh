#!/bin/bash
# =============================================================================
# GPQA (Graduate-Level Google-Proof Q&A) Evaluation
# =============================================================================
# GPQA uses 5-shot chain-of-thought prompting for graduate-level science questions
# Memory-optimized GPU allocation based on model size
# =============================================================================

export HF_HOME=/scratch/huggingface

export HF_TOKEN=XXX
export HF_HOME_DATASETS=/scratch/huggingface/datasets
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "Starting GPQA Evaluation Suite"
echo "=============================="

# # Small Models (4B-8B): Single GPU, optimized batch size
# echo "[1/5] Qwen3-4B-Thinking (GPQA Diamond)"
# CUDA_VISIBLE_DEVICES=6 python baseline.py "Qwen/Qwen3-4B-Thinking-2507" --task gpqa_diamond_cot_n_shot --num_repeats 3 --num_fewshot 5 --batch_size 32 &> logs/gpqa_diamond_cot_n_shot_qwen3_4b_thinking_2507.txt&

# echo "[2/5] Qwen3-8B (GPQA Diamond)"
# CUDA_VISIBLE_DEVICES=7 python baseline.py "Qwen/Qwen3-8B" --task gpqa_diamond_cot_n_shot --num_repeats 3 --num_fewshot 5 --batch_size 32 &> logs/gpqa_diamond_cot_n_shot_qwen3_8b.txt&

echo "[3/5] Llama-3.1-8B (GPQA Diamond)"
CUDA_VISIBLE_DEVICES=6 python baseline.py "meta-llama/Llama-3.1-8B-Instruct" --task gpqa_diamond_cot_n_shot --num_repeats 3 --num_fewshot 5 --batch_size 32 &> logs/gpqa_diamond_cot_n_shot_llama3_1_8b.txt&

# Medium Model (14B): Multi-GPU
echo "[4/5] Qwen3-14B (GPQA Diamond)"
CUDA_VISIBLE_DEVICES=7 python baseline.py "Qwen/Qwen3-14B" --task gpqa_diamond_cot_n_shot --num_repeats 3 --num_fewshot 5 --batch_size 32 &> logs/gpqa_diamond_cot_n_shot_qwen3_14b.txt&

# # Large Model (70B): 4 GPUs
# echo "[5/5] Llama-3.3-70B (GPQA Diamond)"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python baseline.py "meta-llama/Llama-3.3-70B-Instruct" --task gpqa_diamond_cot_n_shot --num_repeats 3 --num_fewshot 5 --batch_size 32 &> logs/gpqa_diamond_cot_n_shot_llama3_3_70b.txt&

# echo "=============================="
# echo "GPQA Evaluation Suite Complete!"
# echo "=============================="


