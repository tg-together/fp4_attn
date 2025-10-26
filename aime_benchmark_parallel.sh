#!/bin/bash
# =============================================================================
# AIME (American Invitational Mathematics Examination) Parallel Evaluation
# =============================================================================
# AIME tasks distributed across 4 nodes with 8 GPUs each
# Uncomment the section for the node you want to run on
# =============================================================================

export HF_HOME_DATASETS=/scratch/huggingface/datasets
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="/scratch/huggingface"
export HF_TOKEN="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"

export PYTHONUNBUFFERED=1

# Number of repeats (0-based, so 4 means 0..4 = 5 repeats)
# Total: 6 models × 5 repeats = 30 jobs across 4 nodes (8 GPUs each)
MAX_REPEAT=4

echo "Starting AIME Parallel Evaluation Suite"
echo "==============================="

mkdir -p "logs/aime_benchmark"

# =============================================================================
# NODE 1: 8 GPUs
# Model 1 (Qwen3-4B-Thinking AIME24): repeats 0-4 (5 GPUs)
# Model 2 (Qwen3-8B AIME24): repeats 0-2 (3 GPUs)
# =============================================================================
# echo "[NODE 1] Starting jobs on 8 GPUs..."
# 
# # Qwen3-4B-Thinking AIME24 - repeats 0-4
# for i in 0 1 2 3 4; do
#     gpu_id=$i
#     CUDA_VISIBLE_DEVICES=$gpu_id python lmeval_main.py "Qwen/Qwen3-4B-Thinking-2507" --task aime24 --batch_size 8 --repeat_id $i > "logs/aime_benchmark/aime24_qwen3_4b_thinking_2507_repeat_$i.txt" 2>&1 &
#     echo "  Launched Qwen3-4B-Thinking AIME24 repeat $i on GPU $gpu_id"
# done
# 
# # Qwen3-8B AIME24 - repeats 0-2
# for i in 0 1 2; do
#     gpu_id=$((i + 5))
#     CUDA_VISIBLE_DEVICES=$gpu_id python lmeval_main.py "Qwen/Qwen3-8B" --task aime24 --batch_size 8 --repeat_id $i > "logs/aime_benchmark/aime24_qwen3_8b_repeat_$i.txt" 2>&1 &
#     echo "  Launched Qwen3-8B AIME24 repeat $i on GPU $gpu_id"
# done

# =============================================================================
# NODE 2: 8 GPUs
# Model 2 (Qwen3-8B AIME24): repeats 3-4 (2 GPUs)
# Model 3 (Qwen3-14B AIME24): repeats 0-4 (5 GPUs)
# Model 4 (Qwen3-8B AIME25): repeat 0 (1 GPU)
# =============================================================================
# echo "[NODE 2] Starting jobs on 8 GPUs..."
# 
# # Qwen3-8B AIME24 - repeats 3-4
# for i in 3 4; do
#     gpu_id=$((i - 3))
#     CUDA_VISIBLE_DEVICES=$gpu_id python lmeval_main.py "Qwen/Qwen3-8B" --task aime24 --batch_size 8 --repeat_id $i > "logs/aime_benchmark/aime24_qwen3_8b_repeat_$i.txt" 2>&1 &
#     echo "  Launched Qwen3-8B AIME24 repeat $i on GPU $gpu_id"
# done
# 
# # Qwen3-14B AIME24 - repeats 0-4
# for i in 0 1 2 3 4; do
#     gpu_id=$((i + 2))
#     CUDA_VISIBLE_DEVICES=$gpu_id python lmeval_main.py "Qwen/Qwen3-14B" --task aime24 --batch_size 4 --repeat_id $i > "logs/aime_benchmark/aime24_qwen3_14b_repeat_$i.txt" 2>&1 &
#     echo "  Launched Qwen3-14B AIME24 repeat $i on GPU $gpu_id"
# done
# 
# # Qwen3-8B AIME25 - repeat 0
# CUDA_VISIBLE_DEVICES=7 python lmeval_main.py "Qwen/Qwen3-8B" --task aime25 --batch_size 8 --repeat_id 0 > "logs/aime_benchmark/aime25_qwen3_8b_repeat_0.txt" 2>&1 &
# echo "  Launched Qwen3-8B AIME25 repeat 0 on GPU 7"

# =============================================================================
# NODE 3: 8 GPUs
# Model 4 (Qwen3-8B AIME25): repeats 1-4 (4 GPUs)
# Model 5 (Qwen3-4B-Thinking AIME25): repeats 0-3 (4 GPUs)
# =============================================================================
# echo "[NODE 3] Starting jobs on 8 GPUs..."
# 
# # Qwen3-8B AIME25 - repeats 1-4
# for i in 1 2 3 4; do
#     gpu_id=$((i - 1))
#     CUDA_VISIBLE_DEVICES=$gpu_id python lmeval_main.py "Qwen/Qwen3-8B" --task aime25 --batch_size 8 --repeat_id $i > "logs/aime_benchmark/aime25_qwen3_8b_repeat_$i.txt" 2>&1 &
#     echo "  Launched Qwen3-8B AIME25 repeat $i on GPU $gpu_id"
# done
# 
# # Qwen3-4B-Thinking AIME25 - repeats 0-3
# for i in 0 1 2 3; do
#     gpu_id=$((i + 4))
#     CUDA_VISIBLE_DEVICES=$gpu_id python lmeval_main.py "Qwen/Qwen3-4B-Thinking-2507" --task aime25 --batch_size 8 --repeat_id $i > "logs/aime_benchmark/aime25_qwen3_4b_thinking_2507_repeat_$i.txt" 2>&1 &
#     echo "  Launched Qwen3-4B-Thinking AIME25 repeat $i on GPU $gpu_id"
# done

# =============================================================================
# NODE 4: 8 GPUs (6 used, 2 spare)
# Model 5 (Qwen3-4B-Thinking AIME25): repeat 4 (1 GPU)
# Model 6 (Qwen3-14B AIME25): repeats 0-4 (5 GPUs)
# =============================================================================
# echo "[NODE 4] Starting jobs on 8 GPUs (6 GPUs used, 2 spare)..."
# 
# # Qwen3-4B-Thinking AIME25 - repeat 4
# CUDA_VISIBLE_DEVICES=0 python lmeval_main.py "Qwen/Qwen3-4B-Thinking-2507" --task aime25 --batch_size 8 --repeat_id 4 > "logs/aime_benchmark/aime25_qwen3_4b_thinking_2507_repeat_4.txt" 2>&1 &
# echo "  Launched Qwen3-4B-Thinking AIME25 repeat 4 on GPU 0"
# 
# # Qwen3-14B AIME25 - repeats 0-4
# for i in 0 1 2 3 4; do
#     gpu_id=$((i + 1))
#     CUDA_VISIBLE_DEVICES=$gpu_id python lmeval_main.py "Qwen/Qwen3-14B" --task aime25 --batch_size 4 --repeat_id $i > "logs/aime_benchmark/aime25_qwen3_14b_repeat_$i.txt" 2>&1 &
#     echo "  Launched Qwen3-14B AIME25 repeat $i on GPU $gpu_id"
# done

# Wait for all background processes to complete
wait

echo "==============================="
echo "AIME Parallel Evaluation Suite Complete!"
echo "==============================="
