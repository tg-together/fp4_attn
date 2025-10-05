#!/bin/bash

# Select which mode to run: baseline, fp4, or kvquant
MODE="all"  # Options: baseline, fp4, kvquant

# Task-specific configuration
TASK="aime25"

# Common suffix for all commands
SUFFIX_CMD="--task ${TASK} --apply_chat_template --fewshot_as_multiturn --log_samples"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="/scratch/huggingface"
export HF_TOKEN="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"

if [ "$MODE" = "baseline" ]; then
    mkdir -p "logs/aime_benchmark/baseline/"
    
    # Run non-70B models sequentially for baseline on single GPU
    (
        MODEL="meta-llama/Llama-3.2-3B-Instruct"
        CUDA_VISIBLE_DEVICES=0  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --output "./logs/aime_benchmark/baseline/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/baseline/${MODEL##*/}_aime25.txt"
        
        MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
        CUDA_VISIBLE_DEVICES=0  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --output "./logs/aime_benchmark/baseline/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/baseline/${MODEL##*/}_aime25.txt"
        
        MODEL="Qwen/Qwen3-4B-Thinking-2507"
        CUDA_VISIBLE_DEVICES=0 python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --output "./logs/aime_benchmark/baseline/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/baseline/${MODEL##*/}_aime25.txt"
    ) &
    
    # Run 70B model for baseline on 2 GPUs
    MODEL="meta-llama/Llama-3.3-70B-Instruct"
    CUDA_VISIBLE_DEVICES=1,2  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --output "./logs/aime_benchmark/baseline/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/baseline/${MODEL##*/}_aime25.txt" &

elif [ "$MODE" = "all" ]; then
    mkdir -p "logs/aime_benchmark/baseline/"
    mkdir -p "logs/aime_benchmark/kvquant/"
    mkdir -p "logs/aime_benchmark/fp4/"
    
    # Run non-70B models sequentially for baseline and kvquant on single GPU
    (
        # Baseline non-70B models
        MODEL="meta-llama/Llama-3.2-3B-Instruct"
        CUDA_VISIBLE_DEVICES=0  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --output "./logs/aime_benchmark/baseline/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/baseline/${MODEL##*/}_aime25.txt"
        
        MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
        CUDA_VISIBLE_DEVICES=0  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --output "./logs/aime_benchmark/baseline/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/baseline/${MODEL##*/}_aime25.txt"
        
        MODEL="Qwen/Qwen3-4B-Thinking-2507"
        CUDA_VISIBLE_DEVICES=0 python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --output "./logs/aime_benchmark/baseline/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/baseline/${MODEL##*/}_aime25.txt"
        
        # KVquant non-70B models
        MODEL="meta-llama/Llama-3.2-3B-Instruct"
        CUDA_VISIBLE_DEVICES=0  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --kvquant --output "./logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.txt"
        
        MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
        CUDA_VISIBLE_DEVICES=0  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --kvquant --output "./logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.txt"
        
        MODEL="Qwen/Qwen3-4B-Thinking-2507"
        CUDA_VISIBLE_DEVICES=0  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --kvquant --output "./logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.txt"
    ) &
    
    # Run 70B models sequentially for baseline and kvquant on 2 GPUs
    (
        # Baseline 70B model
        MODEL="meta-llama/Llama-3.3-70B-Instruct"
        CUDA_VISIBLE_DEVICES=1,2  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --output "./logs/aime_benchmark/baseline/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/baseline/${MODEL##*/}_aime25.txt"
        
        # KVquant 70B model
        MODEL="meta-llama/Llama-3.3-70B-Instruct"
        CUDA_VISIBLE_DEVICES=1,2  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --kvquant --output "./logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.txt"
    ) &
    
    # FP4 70B model on 2 GPUs
    MODEL="meta-llama/Llama-3.3-70B-Instruct"
    CUDA_VISIBLE_DEVICES=3,4  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --quantize --output "./logs/aime_benchmark/fp4/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/fp4/${MODEL##*/}_aime25.txt" &
    
    # FP4 non-70B models on 3 different GPUs
    MODEL="meta-llama/Llama-3.2-3B-Instruct"
    CUDA_VISIBLE_DEVICES=5  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --quantize --output "./logs/aime_benchmark/fp4/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/fp4/${MODEL##*/}_aime25.txt" &
    
    MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    CUDA_VISIBLE_DEVICES=6  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --quantize --output "./logs/aime_benchmark/fp4/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/fp4/${MODEL##*/}_aime25.txt" &
    
    MODEL="Qwen/Qwen3-4B-Thinking-2507"
    CUDA_VISIBLE_DEVICES=7  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --quantize --output "./logs/aime_benchmark/fp4/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/fp4/${MODEL##*/}_aime25.txt" &

elif [ "$MODE" = "fp4" ]; then
    mkdir -p "logs/aime_benchmark/fp4/"
    
    #FP4
    
    # FP4 70B model on 2 GPUs
    MODEL="meta-llama/Llama-3.3-70B-Instruct"
    CUDA_VISIBLE_DEVICES=3,4  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --quantize --output "./logs/aime_benchmark/fp4/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/fp4/${MODEL##*/}_aime25.txt" &
    
    # FP4 non-70B models on 3 different GPUs
    MODEL="meta-llama/Llama-3.2-3B-Instruct"
    CUDA_VISIBLE_DEVICES=5  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --quantize --output "./logs/aime_benchmark/fp4/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/fp4/${MODEL##*/}_aime25.txt" &
    
    MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    CUDA_VISIBLE_DEVICES=6  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --quantize --output "./logs/aime_benchmark/fp4/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/fp4/${MODEL##*/}_aime25.txt" &
    
    MODEL="Qwen/Qwen3-4B-Thinking-2507"
    CUDA_VISIBLE_DEVICES=7  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --quantize --output "./logs/aime_benchmark/fp4/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/fp4/${MODEL##*/}_aime25.txt" &

elif [ "$MODE" = "kvquant" ]; then
    mkdir -p "logs/aime_benchmark/kvquant/"
    
    #KVquant
    
    # Run non-70B models sequentially on single GPU
    (
        MODEL="meta-llama/Llama-3.2-3B-Instruct"
        CUDA_VISIBLE_DEVICES=0  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --kvquant --output "./logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.txt"
        
        MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
        CUDA_VISIBLE_DEVICES=0  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --kvquant --output "./logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.txt"
        
        MODEL="Qwen/Qwen3-4B-Thinking-2507"
        CUDA_VISIBLE_DEVICES=0  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --kvquant --output "./logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.txt"
    ) &
    
    # Run 70B model on 2 GPUs
    MODEL="meta-llama/Llama-3.3-70B-Instruct"
    CUDA_VISIBLE_DEVICES=1,2  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --kvquant --output "./logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.txt" &
fi
