#!/bin/bash

# Select which mode to run: baseline, fp4, or kvquant
MODE="baseline"  # Options: baseline, fp4, kvquant, all

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="/scratch/huggingface"
export HF_TOKEN="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"

# Use unbuffered Python output for better logging
export PYTHONUNBUFFERED=1

if [ "$MODE" = "baseline" ]; then
    mkdir -p "logs/ppl_benchmark/baseline/"
    
    # Run non-70B models sequentially for baseline on single GPU
    (
        MODEL="meta-llama/Llama-3.1-8B"
        CUDA_VISIBLE_DEVICES=0  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 > "logs/ppl_benchmark/baseline/${MODEL##*/}_ppl.txt" 2>&1
        
        MODEL="Qwen/Qwen3-8B"
        CUDA_VISIBLE_DEVICES=0  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --batch_size 3 > "logs/ppl_benchmark/baseline/${MODEL##*/}_ppl.txt" 2>&1
        
        MODEL="Qwen/Qwen3-4B"
        CUDA_VISIBLE_DEVICES=0  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 > "logs/ppl_benchmark/baseline/${MODEL##*/}_ppl.txt" 2>&1
    ) &
    
    # Run 70B model for baseline on 2 GPUs
    MODEL="meta-llama/Llama-3.1-70B"
    CUDA_VISIBLE_DEVICES=1,2  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --batch_size 1 > "logs/ppl_benchmark/baseline/${MODEL##*/}_ppl.txt" 2>&1 &

elif [ "$MODE" = "all" ]; then
    mkdir -p "logs/ppl_benchmark/baseline/"
    mkdir -p "logs/ppl_benchmark/kvquant/"
    mkdir -p "logs/ppl_benchmark/fp4/"
    
    # Run non-70B models sequentially for baseline and kvquant on single GPU
    (
        # Baseline non-70B models
        MODEL="meta-llama/Llama-3.1-8B"
        CUDA_VISIBLE_DEVICES=0  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 > "logs/ppl_benchmark/baseline/${MODEL##*/}_ppl.txt" 2>&1
        
        MODEL="Qwen/Qwen3-8B"
        CUDA_VISIBLE_DEVICES=0  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --batch_size 3 > "logs/ppl_benchmark/baseline/${MODEL##*/}_ppl.txt" 2>&1
        
        MODEL="Qwen/Qwen3-4B"
        CUDA_VISIBLE_DEVICES=0  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 > "logs/ppl_benchmark/baseline/${MODEL##*/}_ppl.txt" 2>&1
        
        # KVquant non-70B models
        MODEL="meta-llama/Llama-3.1-8B"
        CUDA_VISIBLE_DEVICES=0  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --kvquant --batch_size 1 > "logs/ppl_benchmark/kvquant/${MODEL##*/}_ppl.txt" 2>&1
        
        MODEL="Qwen/Qwen3-8B"
        CUDA_VISIBLE_DEVICES=0  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --kvquant --batch_size 1 > "logs/ppl_benchmark/kvquant/${MODEL##*/}_ppl.txt" 2>&1
        
        MODEL="Qwen/Qwen3-4B"
        CUDA_VISIBLE_DEVICES=0  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --kvquant --batch_size 1 > "logs/ppl_benchmark/kvquant/${MODEL##*/}_ppl.txt" 2>&1
    ) &
    
    # Run 70B models sequentially for baseline and kvquant on 2 GPUs
    (
        # Baseline 70B model
        MODEL="meta-llama/Llama-3.1-70B"
        CUDA_VISIBLE_DEVICES=1,2  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --batch_size 1 > "logs/ppl_benchmark/baseline/${MODEL##*/}_ppl.txt" 2>&1
        
        # KVquant 70B model
        MODEL="meta-llama/Llama-3.1-70B"
        CUDA_VISIBLE_DEVICES=1,2  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --kvquant --batch_size 1 > "logs/ppl_benchmark/kvquant/${MODEL##*/}_ppl.txt" 2>&1
    ) &
    
    # FP4 70B model on 2 GPUs
    MODEL="meta-llama/Llama-3.1-70B"
    CUDA_VISIBLE_DEVICES=3,4  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --batch_size 1 --quantize > "logs/ppl_benchmark/fp4/${MODEL##*/}_ppl.txt" 2>&1 &
    
    # FP4 non-70B models on 3 different GPUs
    MODEL="meta-llama/Llama-3.1-8B"
    CUDA_VISIBLE_DEVICES=5  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "logs/ppl_benchmark/fp4/${MODEL##*/}_ppl.txt" 2>&1 &
    
    MODEL="Qwen/Qwen3-8B"
    CUDA_VISIBLE_DEVICES=6  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize --batch_size 3 > "logs/ppl_benchmark/fp4/${MODEL##*/}_ppl.txt" 2>&1 &
    
    MODEL="Qwen/Qwen3-4B"
    CUDA_VISIBLE_DEVICES=7  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "logs/ppl_benchmark/fp4/${MODEL##*/}_ppl.txt" 2>&1 &

elif [ "$MODE" = "fp4" ]; then
    mkdir -p "logs/ppl_benchmark/fp4/"
    
    # FP4 70B model on 2 GPUs
    MODEL="meta-llama/Llama-3.1-70B"
    CUDA_VISIBLE_DEVICES=3,4  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --batch_size 1 --quantize > "logs/ppl_benchmark/fp4/${MODEL##*/}_ppl.txt" 2>&1 &
    
    # FP4 non-70B models on 3 different GPUs
    MODEL="meta-llama/Llama-3.1-8B"
    CUDA_VISIBLE_DEVICES=5  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "logs/ppl_benchmark/fp4/${MODEL##*/}_ppl.txt" 2>&1 &
    
    MODEL="Qwen/Qwen3-8B"
    CUDA_VISIBLE_DEVICES=6  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize --batch_size 3 > "logs/ppl_benchmark/fp4/${MODEL##*/}_ppl.txt" 2>&1 &
    
    MODEL="Qwen/Qwen3-4B"
    CUDA_VISIBLE_DEVICES=7  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "logs/ppl_benchmark/fp4/${MODEL##*/}_ppl.txt" 2>&1 &

elif [ "$MODE" = "kvquant" ]; then
    mkdir -p "logs/ppl_benchmark/kvquant/"
    
    # Run non-70B models sequentially on single GPU
    (
        MODEL="meta-llama/Llama-3.1-8B"
        CUDA_VISIBLE_DEVICES=0  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --kvquant --batch_size 1 > "logs/ppl_benchmark/kvquant/${MODEL##*/}_ppl.txt" 2>&1
        
        MODEL="Qwen/Qwen3-8B"
        CUDA_VISIBLE_DEVICES=0  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --kvquant --batch_size 1 > "logs/ppl_benchmark/kvquant/${MODEL##*/}_ppl.txt" 2>&1
        
        MODEL="Qwen/Qwen3-4B"
        CUDA_VISIBLE_DEVICES=0  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --kvquant --batch_size 1 > "logs/ppl_benchmark/kvquant/${MODEL##*/}_ppl.txt" 2>&1
    ) &
    
    # Run 70B model on 2 GPUs
    MODEL="meta-llama/Llama-3.1-70B"
    CUDA_VISIBLE_DEVICES=1,2  python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --kvquant --batch_size 1 > "logs/ppl_benchmark/kvquant/${MODEL##*/}_ppl.txt" 2>&1 &
fi

# Wait for all background jobs to complete
wait
echo "All PPL benchmark runs completed!"