#!/bin/bash

# Master script to submit baseline evaluation jobs for multiple models
# Simple version: 3 models on 3 nodes (8B uses 1 GPU, 14B uses 1 GPU, 32B uses 2 GPUs)

echo "=========================================="
echo "Submitting Baseline Evaluation Jobs"
echo "=========================================="
echo "Total: 3 model sizes (8B, 14B, 32B)"
echo "Nodes required: 1-3 depending on configuration"
echo ""

# Check available idle nodes
IDLE_COUNT=$(sinfo -N -h -t idle -o "%N" | grep research-secure | wc -l)
echo "Available idle nodes: $IDLE_COUNT"
echo ""

if [ $IDLE_COUNT -lt 1 ]; then
    echo "⚠️  WARNING: Need at least 1 idle node"
    echo "Jobs may queue. Continue? (Ctrl-C to cancel)"
    sleep 3
fi

# Configuration
TASK="${TASK:-aime24}"
NUM_REPEATS="${NUM_REPEATS:-10}"

# Job 1: Qwen3-8B - 1 node, 8 models in parallel (1 GPU each)
echo "[1/3] Submitting: Qwen/Qwen3-8B (1 node, 8 parallel tasks)"
sbatch --job-name=baseline_qwen8b \
    --nodes=1 \
    --export=ALL,MODEL="Qwen/Qwen3-8B",TASK_NAME="$TASK",NUM_GPUS=1,BATCH_SIZE=2,NUM_REPEATS="$NUM_REPEATS",MODEL_START=0,MODEL_END=7 \
    /home/shirley/RoCK-KV/baseline/sbatch/run_baseline.sh

sleep 1

# Job 2: Qwen3-14B - 1 node, 8 models in parallel (1 GPU each)
echo "[2/3] Submitting: Qwen/Qwen3-14B (1 node, 8 parallel tasks)"
sbatch --job-name=baseline_qwen14b \
    --nodes=1 \
    --export=ALL,MODEL="Qwen/Qwen3-14B",TASK_NAME="$TASK",NUM_GPUS=1,BATCH_SIZE=2,NUM_REPEATS="$NUM_REPEATS",MODEL_START=0,MODEL_END=7 \
    /home/shirley/RoCK-KV/baseline/sbatch/run_baseline.sh

sleep 1

# Job 3: Qwen3-32B - 1 node, 4 models in parallel (2 GPUs each)
echo "[3/3] Submitting: Qwen/Qwen3-32B (1 node, 4 parallel tasks, 2 GPUs each)"
sbatch --job-name=baseline_qwen32b \
    --nodes=1 \
    --export=ALL,MODEL="Qwen/Qwen3-32B",TASK_NAME="$TASK",NUM_GPUS=2,BATCH_SIZE=1,NUM_REPEATS="$NUM_REPEATS",MODEL_START=0,MODEL_END=3 \
    /home/shirley/RoCK-KV/baseline/sbatch/run_baseline.sh

echo ""
echo "=========================================="
echo "All 3 jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor with: squeue -u $USER"
echo "Logs: ~/RoCK-KV/baseline/log/"
echo "Results: ~/RoCK-KV/eval_results/"
echo ""
echo "To submit for a different task:"
echo "  TASK=gsm8k_cot_llama ./submit_all_models.sh"
echo ""
