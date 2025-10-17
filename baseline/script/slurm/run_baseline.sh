#!/bin/bash
#SBATCH --job-name=baseline_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=200:00:00
#SBATCH --partition=batch
#SBATCH --output=log/baseline_slurm_%j.out
#SBATCH --error=log/baseline_slurm_%j.err
#SBATCH --exclude=research-secure-20

# ============================================================================
# Baseline FP16 Evaluation - SLURM Job Script
# ============================================================================
# This script runs baseline FP16 evaluations on multiple GPUs in parallel
# Each GPU runs one model independently
# ============================================================================

# Configuration (can be overridden by sbatch --export)
export MODEL="${MODEL:-Qwen/Qwen3-8B}"
export TASK_NAME="${TASK_NAME:-aime24}"
export NUM_REPEATS="${NUM_REPEATS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export NUM_GPUS="${NUM_GPUS:-1}"
export MODEL_START="${MODEL_START:-0}"
export MODEL_END="${MODEL_END:-7}"

# DEBUG mode (set to 1 for quick test: 3 samples, 1 repeat)
export DEBUG="${DEBUG:-0}"

# Environment variables
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HOME=/workspace/.cache/huggingface
export HF_TOKEN="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"

# Apptainer paths
APPTAINER_SIF="$HOME/RoCK-KV/build/kchanboost.sif"
APPTAINER_IMG="$HOME/RoCK-KV/build/kchanboost.img"

# ============================================================================
# Model Configuration - 8 different models on 8 GPUs
# ============================================================================
# You can run different models or same model with different tasks
# Format: "GPU_ID | MODEL_NAME | TASK_NAME"
declare -a MODELS=(
    "0 | Qwen/Qwen3-8B | aime24"
    "1 | Qwen/Qwen3-8B | gsm8k_cot_llama"
    "2 | Qwen/Qwen3-8B | humaneval_instruct"
    "3 | Qwen/Qwen3-8B | gpqa_diamond_cot_n_shot"
    "4 | meta-llama/Llama-3.1-8B-Instruct | aime24"
    "5 | meta-llama/Llama-3.1-8B-Instruct | gsm8k_cot_llama"
    "6 | Qwen/Qwen3-14B | aime24"
    "7 | Qwen/Qwen3-14B | gsm8k_cot_llama"
)

# ============================================================================
# Main Program
# ============================================================================

echo "=========================================="
echo "Baseline FP16 Evaluation - SLURM Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Default Model: $MODEL"
echo "Default Task: $TASK_NAME"
echo "Num Repeats: $NUM_REPEATS"
echo "Batch Size: $BATCH_SIZE"
if [ "$DEBUG" = "1" ]; then
    echo "DEBUG MODE: 3 samples, 1 repeat"
fi
echo "=========================================="
echo ""

mkdir -p "$HOME/RoCK-KV/baseline/log"

# Store background process PIDs
declare -a PIDS=()

echo "Starting parallel baseline evaluations..."
echo ""

# Launch GPU tasks (process models in range [MODEL_START, MODEL_END])
MODEL_IDX=0
GPU_OFFSET=0

for model_config in "${MODELS[@]}"; do
    IFS='|' read -ra PARTS <<< "$model_config"
    # Trim whitespace
    GPU_ID=$(echo "${PARTS[0]}" | xargs)
    MODEL_NAME=$(echo "${PARTS[1]}" | xargs)
    TASK=$(echo "${PARTS[2]}" | xargs)

    # Skip models outside our assigned range
    if [ "$GPU_ID" -lt "$MODEL_START" ] || [ "$GPU_ID" -gt "$MODEL_END" ]; then
        continue
    fi

    # Configure GPU string based on NUM_GPUS for tensor parallelism
    if [ "$NUM_GPUS" -eq 1 ]; then
        # 8B models: 1 GPU each
        GPU_STRING="${GPU_OFFSET}"
        ((GPU_OFFSET++))
    elif [ "$NUM_GPUS" -eq 2 ]; then
        # 32B models: 2 GPUs each
        GPU_START=$((GPU_OFFSET))
        GPU_END=$((GPU_OFFSET + 1))
        GPU_STRING="${GPU_START},${GPU_END}"
        GPU_OFFSET=$((GPU_OFFSET + 2))
    elif [ "$NUM_GPUS" -eq 4 ]; then
        # 70B models: 4 GPUs each
        GPU_START=$((GPU_OFFSET))
        GPU_END=$((GPU_OFFSET + 3))
        GPU_STRING=$(seq -s, ${GPU_START} ${GPU_END})
        GPU_OFFSET=$((GPU_OFFSET + 4))
    fi

    echo "GPU ${GPU_STRING}: ${MODEL_NAME} on ${TASK}"

    # Prepare debug flag
    DEBUG_FLAG=""
    if [ "$DEBUG" = "1" ] || [ "$DEBUG" = "true" ]; then
        DEBUG_FLAG="--debug"
    fi

    # Run in Apptainer container (background)
    apptainer exec --nv \
        --bind $HOME:/workspace \
        --bind /data:/data \
        --overlay "$APPTAINER_IMG":ro \
        "$APPTAINER_SIF" \
        bash -c "
            cd /workspace/RoCK-KV/baseline

            # Export environment variables
            export CUDA_VISIBLE_DEVICES='${GPU_STRING}'
            export TORCH_CUDA_ARCH_LIST='9.0'
            export HF_HOME=/data/huggingface
            export HF_TOKEN='$HF_TOKEN'
            export TOKENIZERS_PARALLELISM=false
            export HF_DATASETS_TRUST_REMOTE_CODE=1

            # Extract model short name for logging
            MODEL_SHORT=\$(basename '${MODEL_NAME}' | tr '[:upper:]' '[:lower:]' | tr '-' '_')
            LOG_FILE=\"log/\${CUDA_VISIBLE_DEVICES//,/}_\${MODEL_SHORT}_${TASK}.log\"

            echo \"Starting baseline evaluation...\" > \"\$LOG_FILE\"
            echo \"Model: ${MODEL_NAME}\" >> \"\$LOG_FILE\"
            echo \"Task: ${TASK}\" >> \"\$LOG_FILE\"
            echo \"GPUs: \${CUDA_VISIBLE_DEVICES}\" >> \"\$LOG_FILE\"
            echo \"Repeats: $NUM_REPEATS\" >> \"\$LOG_FILE\"
            echo \"Batch Size: $BATCH_SIZE\" >> \"\$LOG_FILE\"
            echo \"\" >> \"\$LOG_FILE\"

            # Run baseline evaluation
            python baseline.py '${MODEL_NAME}' \
                --task '${TASK}' \
                --num_repeats $NUM_REPEATS \
                --batch_size $BATCH_SIZE \
                $DEBUG_FLAG >> \"\$LOG_FILE\" 2>&1

            echo \"\" >> \"\$LOG_FILE\"
            echo \"Baseline evaluation completed at \$(date)\" >> \"\$LOG_FILE\"
        " &

    PIDS+=($!)
    sleep 2
done

echo ""
echo "All tasks launched. PIDs: ${PIDS[@]}"
echo ""

# Wait for all tasks to complete
echo "Waiting for all evaluations to complete..."
wait

echo ""
echo "=========================================="
echo "All baseline evaluations completed!"
echo "End: $(date)"
echo "=========================================="
echo ""

# Get node name
NODE_NAME=$(hostname | grep -oP 'external-\K\d+' | head -1)
if [ -n "$NODE_NAME" ]; then
    NODE_NAME="node_${NODE_NAME}"
else
    NODE_NAME=$(hostname | cut -d. -f1)
fi

LOG_DIR="$HOME/RoCK-KV/baseline/log"

# Check results
echo "Results Summary:"
echo "----------------------------------------"
for model_config in "${MODELS[@]}"; do
    IFS='|' read -ra PARTS <<< "$model_config"
    GPU_ID=$(echo "${PARTS[0]}" | xargs)

    # Skip models outside our range
    if [ "$GPU_ID" -lt "$MODEL_START" ] || [ "$GPU_ID" -gt "$MODEL_END" ]; then
        continue
    fi

    # Find log file
    LOG_FILES=$(ls ${LOG_DIR}/${GPU_ID}_*.log 2>/dev/null)

    if [ -n "$LOG_FILES" ]; then
        for LOG_FILE in $LOG_FILES; do
            COMPLETED=$(grep -c "Completed" "$LOG_FILE" 2>/dev/null || echo "0")
            FILENAME=$(basename "$LOG_FILE")
            echo "GPU ${GPU_ID}: ${FILENAME} - Check log for status"
        done
    else
        echo "GPU ${GPU_ID}: No log file found"
    fi
done

echo ""
echo "Results: eval_results/"
echo "Logs: ${LOG_DIR}/"
