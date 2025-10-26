#!/bin/bash
# =============================================================================
# AIME (American Invitational Mathematics Examination) Parallel Evaluation
# =============================================================================
# AIME tasks with parallel repeats using --repeat_id flag
# =============================================================================

export HF_HOME_DATASETS=/scratch/huggingface/datasets
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="/scratch/huggingface"
export HF_TOKEN="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"

export PYTHONUNBUFFERED=1

# Number of repeats (0-based, so 9 means 0..9 = 10 repeats)
MAX_REPEAT=5

echo "Starting AIME Parallel Evaluation Suite"
echo "==============================="

mkdir -p "logs/aime_benchmark"

# Small Models (4B-8B): Single GPU, optimized batch size
echo "[1/7] Qwen3-4B-Thinking (AIME24)"
for i in $(seq 0 $MAX_REPEAT); do
    CUDA_VISIBLE_DEVICES=$i python lmeval_main.py "Qwen/Qwen3-4B-Thinking-2507" --task aime24 --batch_size 8 --repeat_id $i > "logs/aime_benchmark/aime24_qwen3_4b_thinking_2507_repeat_$i.txt" 2>&1 &
done

echo "[2/7] Qwen3-8B (AIME24)"
for i in $(seq 0 $MAX_REPEAT); do
    CUDA_VISIBLE_DEVICES=$i python lmeval_main.py "Qwen/Qwen3-8B" --task aime24 --batch_size 8 --repeat_id $i > "logs/aime_benchmark/aime24_qwen3_8b_repeat_$i.txt" 2>&1 &
done

# Medium Model (14B): Multi-GPU
echo "[4/7] Qwen3-14B (AIME24)"
for i in $(seq 0 $MAX_REPEAT); do
    CUDA_VISIBLE_DEVICES=$i python lmeval_main.py "Qwen/Qwen3-14B" --task aime24 --batch_size 4 --repeat_id $i > "logs/aime_benchmark/aime24_qwen3_14b_repeat_$i.txt" 2>&1 &
done

echo "[5/7] Qwen3-8B (AIME25)"
for i in $(seq 0 $MAX_REPEAT); do
    CUDA_VISIBLE_DEVICES=$i python lmeval_main.py "Qwen/Qwen3-8B" --task aime25 --batch_size 8 --repeat_id $i > "logs/aime_benchmark/aime25_qwen3_8b_repeat_$i.txt" 2>&1 &
done

# AIME25 Evaluations
echo "[6/7] Qwen3-4B-Thinking (AIME25)"
for i in $(seq 0 $MAX_REPEAT); do
    CUDA_VISIBLE_DEVICES=$i python lmeval_main.py "Qwen/Qwen3-4B-Thinking-2507" --task aime25 --batch_size 8 --repeat_id $i > "logs/aime_benchmark/aime25_qwen3_4b_thinking_2507_repeat_$i.txt" 2>&1 &
done

echo "[7/7] Qwen3-14B (AIME25)"
for i in $(seq 0 $MAX_REPEAT); do
    CUDA_VISIBLE_DEVICES=$i python lmeval_main.py "Qwen/Qwen3-14B" --task aime25 --batch_size 4 --repeat_id $i > "logs/aime_benchmark/aime25_qwen3_14b_repeat_$i.txt" 2>&1 &
done

# Wait for all background processes to complete
wait

echo "==============================="
echo "AIME Parallel Evaluation Suite Complete!"
echo "==============================="
