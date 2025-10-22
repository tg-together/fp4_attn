#!/bin/bash
# =============================================================================
# Complete Baseline Evaluation Suite
# =============================================================================
# Runs all evaluation tasks in sequence with optimized GPU allocation
# =============================================================================

export HF_HOME=/scratch/huggingface
export HF_TOKEN=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
export HF_HOME_DATASETS=/scratch/huggingface/datasets
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=============================================="
echo "Starting Complete Baseline Evaluation Suite"
echo "=============================================="
echo "Tasks: AIME, GPQA, GSM8K, MMLU, HumanEval"
echo "Models: Qwen3-4B-Thinking, Qwen3-8B, Qwen3-14B,"
echo "        Llama-3.1-8B, Llama-3.3-70B"
echo "=============================================="

START_TIME=$(date +%s)

# Run each task suite
echo ""
echo "🔢 [1/5] Running AIME (Math Competition Problems)..."
bash run_amie.sh
echo "✓ AIME completed"

echo ""
echo "🧪 [2/5] Running GPQA (Graduate-Level Science)..."
bash run_gpqa.sh
echo "✓ GPQA completed"

echo ""
echo "📚 [3/5] Running GSM8K (Grade School Math)..."
bash run_gsm8k.sh
echo "✓ GSM8K completed"

echo ""
echo "🎓 [4/5] Running MMLU (Multitask Language Understanding)..."
bash run_mmlu_pro.sh
echo "✓ MMLU completed"

echo ""
echo "💻 [5/5] Running HumanEval (Code Generation)..."
bash run_humaneval.sh
echo "✓ HumanEval completed"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "=============================================="
echo "🎉 Complete Baseline Evaluation Suite Finished!"
echo "=============================================="
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Results saved in: eval_results/"
echo "Logs saved in: sweep_log_*.txt"
echo "=============================================="