#!/bin/bash

# Weight Quantization Benchmark Script
# Compares baseline vs weight quantized models

MODE="fp8_weight_quant"  # Options: baseline, weight_quant, dual_weight_quant, fp8_weight_quant, all

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="/scratch/huggingface"
export HF_DATASETS_CACHE="/scratch/huggingface/datasets1"
export HF_TOKEN="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"
export PYTHONUNBUFFERED=1

if [ "$MODE" = "baseline" ]; then
    mkdir -p "logs/weight_quant_benchmark/baseline/"
    
    # Baseline Qwen models
    MODEL="Qwen/Qwen3-4B"
    CUDA_VISIBLE_DEVICES=0 python -u eval_weight_quant.py \
        --model "${MODEL}" \
        --dataset wikitext2 \
        --nsamples 128 \
        --output "logs/weight_quant_benchmark/baseline/${MODEL##*/}_results.json" \
        > "logs/weight_quant_benchmark/baseline/${MODEL##*/}_log.txt" 2>&1 &
    
    MODEL="Qwen/Qwen3-8B"
    CUDA_VISIBLE_DEVICES=1 python -u eval_weight_quant.py \
        --model "${MODEL}" \
        --dataset wikitext2 \
        --nsamples 128 \
        --batch_size 1 \
        --output "logs/weight_quant_benchmark/baseline/${MODEL##*/}_results.json" \
        > "logs/weight_quant_benchmark/baseline/${MODEL##*/}_log.txt" 2>&1 &

elif [ "$MODE" = "weight_quant" ]; then
    mkdir -p "logs/weight_quant_benchmark/weight_quant/"
    
    # Weight quantized models
    MODEL="Qwen/Qwen3-4B"
    CUDA_VISIBLE_DEVICES=0 python -u eval_weight_quant.py \
        --model "${MODEL}" \
        --dataset wikitext2 \
        --nsamples 128 \
        --batch_size 1 \
        --quantize_weights \
        --output "logs/weight_quant_benchmark/weight_quant/${MODEL##*/}_results.json" \
        > "logs/weight_quant_benchmark/weight_quant/${MODEL##*/}_log.txt" 2>&1 &
    
    MODEL="Qwen/Qwen3-8B"
    CUDA_VISIBLE_DEVICES=0 python -u eval_weight_quant.py \
        --model "${MODEL}" \
        --dataset wikitext2 \
        --nsamples 128 \
        --batch_size 1 \
        --quantize_weights \
        --output "logs/weight_quant_benchmark/weight_quant/${MODEL##*/}_results.json" \
        > "logs/weight_quant_benchmark/weight_quant/${MODEL##*/}_log.txt" 2>&1 &
elif [ "$MODE" = "dual_weight_quant" ]; then
    mkdir -p "logs/weight_quant_benchmark/weight_quant/"
    
    # Weight quantized models
    MODEL="Qwen/Qwen3-4B"
    CUDA_VISIBLE_DEVICES=0 python -u eval_weight_quant.py \
        --model "${MODEL}" \
        --dataset wikitext2 \
        --nsamples 128 \
        --quantize_weights \
        --use_dual_weight_quant \
        --output "logs/weight_quant_benchmark/weight_quant/${MODEL##*/}_results.json" \
        > "logs/weight_quant_benchmark/weight_quant/${MODEL##*/}_log.txt" 2>&1 &
    
    MODEL="Qwen/Qwen3-8B"
    CUDA_VISIBLE_DEVICES=1 python -u eval_weight_quant.py \
        --model "${MODEL}" \
        --dataset wikitext2 \
        --nsamples 128 \
        --batch_size 1 \
        --quantize_weights \
        --use_dual_weight_quant \
        --output "logs/weight_quant_benchmark/weight_quant/${MODEL##*/}_results.json" \
        > "logs/weight_quant_benchmark/weight_quant/${MODEL##*/}_log.txt" 2>&1 &

elif [ "$MODE" = "all" ]; then
    mkdir -p "logs/weight_quant_benchmark/baseline/"
    mkdir -p "logs/weight_quant_benchmark/weight_quant/"
    
    # Run baseline evaluations
    (
        MODEL="Qwen/Qwen3-4B"
        CUDA_VISIBLE_DEVICES=0 python -u eval_weight_quant.py \
            --model "${MODEL}" \
            --dataset wikitext2 \
            --nsamples 128 \
            --output "logs/weight_quant_benchmark/baseline/${MODEL##*/}_results.json" \
            > "logs/weight_quant_benchmark/baseline/${MODEL##*/}_log.txt" 2>&1
        
        MODEL="Qwen/Qwen3-8B"
        CUDA_VISIBLE_DEVICES=0 python -u eval_weight_quant.py \
            --model "${MODEL}" \
            --dataset wikitext2 \
            --nsamples 128 \
            --batch_size 1 \
            --output "logs/weight_quant_benchmark/baseline/${MODEL##*/}_results.json" \
            > "logs/weight_quant_benchmark/baseline/${MODEL##*/}_log.txt" 2>&1
    ) &
    
    # Run weight quantized evaluations
    (
        MODEL="Qwen/Qwen3-4B"
        CUDA_VISIBLE_DEVICES=1 python -u eval_weight_quant.py \
            --model "${MODEL}" \
            --dataset wikitext2 \
            --nsamples 128 \
            --quantize_weights \
            --use_dual_weight_quant \
            --output "logs/weight_quant_benchmark/weight_quant/${MODEL##*/}_results.json" \
            > "logs/weight_quant_benchmark/weight_quant/${MODEL##*/}_log.txt" 2>&1
        
        MODEL="Qwen/Qwen3-8B"
        CUDA_VISIBLE_DEVICES=1 python -u eval_weight_quant.py \
            --model "${MODEL}" \
            --dataset wikitext2 \
            --nsamples 128 \
            --batch_size 1 \
            --quantize_weights \
            --use_dual_weight_quant \
            --output "logs/weight_quant_benchmark/weight_quant/${MODEL##*/}_results.json" \
            > "logs/weight_quant_benchmark/weight_quant/${MODEL##*/}_log.txt" 2>&1
    ) &

elif [ "$MODE" = "fp8_weight_quant" ]; then
    mkdir -p "logs/weight_quant_benchmark/fp8_weight_quant/"
    
    # FP8 Weight quantized models
    MODEL="Qwen/Qwen3-4B"
    CUDA_VISIBLE_DEVICES=0 python -u eval_weight_quant.py \
        --model "${MODEL}" \
        --dataset wikitext2 \
        --nsamples 128 \
        --quantize_weights \
        --weight_precision fp8 \
        --output "logs/weight_quant_benchmark/fp8_weight_quant/${MODEL##*/}_results.json" \
        > "logs/weight_quant_benchmark/fp8_weight_quant/${MODEL##*/}_log.txt" 2>&1 &
    
    MODEL="Qwen/Qwen3-8B"
    CUDA_VISIBLE_DEVICES=1 python -u eval_weight_quant.py \
        --model "${MODEL}" \
        --dataset wikitext2 \
        --nsamples 128 \
        --batch_size 1 \
        --quantize_weights \
        --weight_precision fp8 \
        --output "logs/weight_quant_benchmark/fp8_weight_quant/${MODEL##*/}_results.json" \
        > "logs/weight_quant_benchmark/fp8_weight_quant/${MODEL##*/}_log.txt" 2>&1 &

fi

# Wait for all jobs to complete
wait
echo "Weight quantization benchmark completed!"

# Summary script
python3 << 'EOF'
import json
import os
import glob

def summarize_results():
    baseline_files = glob.glob("logs/weight_quant_benchmark/baseline/*_results.json")
    weight_quant_files = glob.glob("logs/weight_quant_benchmark/weight_quant/*_results.json")
    
    print("\n" + "="*60)
    print("WEIGHT QUANTIZATION BENCHMARK SUMMARY") 
    print("="*60)
    
    results = {}
    
    # Load baseline results
    for file in baseline_files:
        with open(file, 'r') as f:
            data = json.load(f)
            model = data['model'].split('/')[-1]
            results[model] = {'baseline': data['perplexity']}
    
    # Load weight quantized results
    for file in weight_quant_files:
        with open(file, 'r') as f:
            data = json.load(f)
            model = data['model'].split('/')[-1]
            if model in results:
                results[model]['weight_quant'] = data['perplexity']
    
    # Print comparison
    print(f"{'Model':<20} {'Baseline PPL':<15} {'Weight Quant PPL':<18} {'Degradation':<12}")
    print("-" * 70)
    
    for model, data in results.items():
        if 'baseline' in data and 'weight_quant' in data:
            baseline_ppl = data['baseline']
            weight_quant_ppl = data['weight_quant']
            degradation = ((weight_quant_ppl - baseline_ppl) / baseline_ppl) * 100
            
            print(f"{model:<20} {baseline_ppl:<15.3f} {weight_quant_ppl:<18.3f} {degradation:<12.2f}%")
    
    print("\n" + "="*60)

if os.path.exists("logs/weight_quant_benchmark/"):
    summarize_results()
else:
    print("No results found yet.")
EOF