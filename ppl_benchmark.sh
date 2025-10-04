#!/bin/bash

# Select which mode to run: baseline, fp4, or kvquant
MODE="baseline"  # Options: baseline, fp4, kvquant

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="/scratch/huggingface"
export HF_TOKEN="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"

if [ "$MODE" = "baseline" ] || [ "$MODE" = "all" ]; then
    mkdir -p "logs/ppl_benchmark/baseline/"
    
    #baseline
    
    # # # Model and dataset configuration
    MODEL="meta-llama/Llama-3.1-8B"
    CUDA_VISIBLE_DEVICES=0  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 2>&1 | tee "logs/ppl_benchmark/baseline/${MODEL##*/}_ppl.txt" &
    
    MODEL="meta-llama/Llama-3.1-70B"
    CUDA_VISIBLE_DEVICES=1  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --batch_size 1 2>&1 | tee "logs/ppl_benchmark/baseline/${MODEL##*/}_ppl.txt" &
    
    MODEL="Qwen/Qwen3-8B"
    CUDA_VISIBLE_DEVICES=2  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 2>&1 | tee "logs/ppl_benchmark/baseline/${MODEL##*/}_ppl.txt" &
    
    MODEL="Qwen/Qwen3-4B"
    CUDA_VISIBLE_DEVICES=3  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 2>&1 | tee "logs/ppl_benchmark/baseline/${MODEL##*/}_ppl.txt" &
    

elif [ "$MODE" = "fp4" ] || [ "$MODE" = "all" ]; then
    mkdir -p "logs/ppl_benchmark/fp4/"
    
    #FP4
    
    # # # Model and dataset configuration
    MODEL="meta-llama/Llama-3.1-8B"
    CUDA_VISIBLE_DEVICES=1  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize 2>&1 | tee "logs/ppl_benchmark/fp4/${MODEL##*/}_ppl.txt" &
    
    MODEL="meta-llama/Llama-3.1-70B"
    CUDA_VISIBLE_DEVICES=1  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --batch_size 1 --quantize 2>&1 | tee "logs/ppl_benchmark/fp4/${MODEL##*/}_ppl.txt" &
    
    MODEL="Qwen/Qwen3-8B"
    CUDA_VISIBLE_DEVICES=2  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize 2>&1 | tee "logs/ppl_benchmark/fp4/${MODEL##*/}_ppl.txt" &
    
    MODEL="Qwen/Qwen3-4B"
    CUDA_VISIBLE_DEVICES=3  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize 2>&1 | tee "logs/ppl_benchmark/fp4/${MODEL##*/}_ppl.txt" &
    

elif [ "$MODE" = "kvquant" ] || [ "$MODE" = "all" ]; then
    mkdir -p "logs/ppl_benchmark/kvquant/"
    
    #KVquant
    
    # # # Model and dataset configuration
    MODEL="meta-llama/Llama-3.1-8B"
    CUDA_VISIBLE_DEVICES=4  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --kvquant --batch_size 1 2>&1 | tee "logs/ppl_benchmark/kvquant/${MODEL##*/}_ppl.txt" &
    
    MODEL="meta-llama/Llama-3.1-70B"
    CUDA_VISIBLE_DEVICES=1  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --batch_size 1 --kvquant --batch_size 1 2>&1 | tee "logs/ppl_benchmark/kvquant/${MODEL##*/}_ppl.txt" &
    
    MODEL="Qwen/Qwen3-8B"
    CUDA_VISIBLE_DEVICES=2  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --kvquant --batch_size 1 2>&1 | tee "logs/ppl_benchmark/kvquant/${MODEL##*/}_ppl.txt" &
    
    MODEL="Qwen/Qwen3-4B"
    CUDA_VISIBLE_DEVICES=3  python eval_ppl.py --model "${MODEL}" --dataset wikitext2  --kvquant --batch_size 1 2>&1 | tee "logs/ppl_benchmark/kvquant/${MODEL##*/}_ppl.txt" &
    
fi