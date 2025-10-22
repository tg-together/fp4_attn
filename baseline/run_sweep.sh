#!/bin/bash

# =============================================================================
# Comprehensive Model Evaluation Sweep
# =============================================================================
# This script runs a comprehensive sweep across different models and tasks
# with optimized GPU allocation and batch sizes based on model memory requirements
# =============================================================================

export HF_HOME=/scratch/huggingface
export HF_TOKEN=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
export HF_HOME_DATASETS=/scratch/huggingface/datasets
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Define models with their GPU and batch size requirements
declare -A MODEL_CONFIG=(
    # Small models (4B-8B): 1 GPU, higher batch size
    ["Qwen/Qwen3-4B-Thinking-2507"]="0|816"
    ["Qwen/Qwen3-8B"]="0|16"
    ["meta-llama/Llama-3.1-8B-Instruct"]="1|16"

    # Medium models (14B): 2 GPUs, medium batch size
    ["Qwen/Qwen3-14B"]="0,1|16"

    # Large models (70B): 4 GPUs, small batch size
    ["meta-llama/Llama-3.3-70B-Instruct"]="0,1,2,3|32"
)

# Task configurations with appropriate parameters
declare -A TASK_CONFIG=(
    ["aime24"]="10|0"      # repeats|fewshot (zero-shot for AIME)
    ["aime25"]="10|0"      # repeats|fewshot
    ["gpqa_diamond_cot_n_shot"]="3|5"  # repeats|fewshot (5-shot standard)
    ["gsm8k_cot_llama"]="3|5"          # repeats|fewshot (8-shot standard)
    ["humaneval_instruct"]="3|0"       # repeats|fewshot (zero-shot for code)
    # ["mmlu_flan_cot_fewshot"]="1|4"    # repeats|fewshot (4-shot standard)
)

# Function to run evaluation with proper GPU allocation
run_evaluation() {
    local model=$1
    local task=$2
    local gpus=$3
    local batch_size=$4
    local repeats=$5
    local fewshot=$6

    echo "=========================================="
    echo "Running: $model on $task"
    echo "GPUs: $gpus | Batch Size: $batch_size"
    echo "Repeats: $repeats | Few-shot: $fewshot"
    echo "=========================================="

    local cmd="CUDA_VISIBLE_DEVICES=$gpus python baseline.py \"$model\" --task $task --num_repeats $repeats --batch_size $batch_size"

    # Add fewshot parameter if not zero
    if [ "$fewshot" -ne "0" ]; then
        cmd="$cmd --num_fewshot $fewshot"
    fi

    echo "Command: $cmd"
    eval $cmd

    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS: $model - $task"
    else
        echo "✗ FAILED: $model - $task"
    fi
    echo ""
}

# Main execution loop
main() {
    echo "Starting Comprehensive Model Evaluation Sweep"
    echo "=============================================="

    local total_jobs=0
    local completed_jobs=0

    # Count total jobs
    for model in "${!MODEL_CONFIG[@]}"; do
        for task in "${!TASK_CONFIG[@]}"; do
            ((total_jobs++))
        done
    done

    echo "Total evaluations to run: $total_jobs"
    echo ""

    # Execute all combinations
    for model in "${!MODEL_CONFIG[@]}"; do
        # Parse model configuration
        IFS='|' read -r gpus batch_size <<< "${MODEL_CONFIG[$model]}"

        echo "Processing model: $model (GPUs: $gpus, Batch: $batch_size)"

        for task in "${!TASK_CONFIG[@]}"; do
            # Parse task configuration
            IFS='|' read -r repeats fewshot <<< "${TASK_CONFIG[$task]}"

            # Run evaluation
            run_evaluation "$model" "$task" "$gpus" "$batch_size" "$repeats" "$fewshot"

            ((completed_jobs++))
            echo "Progress: $completed_jobs/$total_jobs completed"
            echo ""

            # Optional: Add delay between runs to prevent resource conflicts
            sleep 5
        done
    done

    echo "=============================================="
    echo "Sweep completed! $completed_jobs/$total_jobs evaluations finished"
    echo "=============================================="
}

# Run the sweep
main 2>&1 | tee "sweep_log_$(date +%Y%m%d_%H%M%S).txt"