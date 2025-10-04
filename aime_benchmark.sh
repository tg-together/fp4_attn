#!/bin/bash

# Select which mode to run: baseline, fp4, or kvquant
MODE="baseline"  # Options: baseline, fp4, kvquant

# Task-specific configuration
TASK="aime25"
NUM_SAMPLES="50"  # You can adjust this value as needed

# Common suffix for all commands
SUFFIX_CMD="--task ${TASK} --apply_chat_template --fewshot_as_multiturn --log_samples --num_samples ${NUM_SAMPLES}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="/scratch/huggingface"
export HF_TOKEN="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"

if [ "$MODE" = "baseline" ]; then
    mkdir -p "logs/aime_benchmark/baseline/"
    
    #baseline

    MODEL="meta-llama/Llama-3.2-3B-Instruct"
    CUDA_VISIBLE_DEVICES=4  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --output "./logs/aime_benchmark/baseline/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/baseline/${MODEL##*/}_aime25.txt" &
    
    MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    CUDA_VISIBLE_DEVICES=5  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --output "./logs/aime_benchmark/baseline/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/baseline/${MODEL##*/}_aime25.txt" &
    
    MODEL="meta-llama/Llama-3.3-70B-Instruct"
    CUDA_VISIBLE_DEVICES=6  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --output "./logs/aime_benchmark/baseline/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/baseline/${MODEL##*/}_aime25.txt" &
    
    MODEL="Qwen/Qwen3-4B-Thinking-2507"
    CUDA_VISIBLE_DEVICES=7  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --output "./logs/aime_benchmark/baseline/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/baseline/${MODEL##*/}_aime25.txt" &

elif [ "$MODE" = "fp4" ]; then
    mkdir -p "logs/aime_benchmark/fp4/"
    
    #FP4
    

    MODEL="meta-llama/Llama-3.2-3B-Instruct"
    CUDA_VISIBLE_DEVICES=4  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --quantize --output "./logs/aime_benchmark/fp4/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/fp4/${MODEL##*/}_aime25.txt" &
    
    MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    CUDA_VISIBLE_DEVICES=5  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --quantize --output "./logs/aime_benchmark/fp4/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/fp4/${MODEL##*/}_aime25.txt" &
    
    MODEL="meta-llama/Llama-3.3-70B-Instruct"
    CUDA_VISIBLE_DEVICES=6  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --quantize --output "./logs/aime_benchmark/fp4/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/fp4/${MODEL##*/}_aime25.txt" &
    
    MODEL="Qwen/Qwen3-4B-Thinking-2507"
    CUDA_VISIBLE_DEVICES=7  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --quantize --output "./logs/aime_benchmark/fp4/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/fp4/${MODEL##*/}_aime25.txt" &

elif [ "$MODE" = "kvquant" ]; then
    mkdir -p "logs/aime_benchmark/kvquant/"
    
    #KVquant
    

    MODEL="meta-llama/Llama-3.2-3B-Instruct"
    CUDA_VISIBLE_DEVICES=4  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --kvquant --output "./logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.txt" &
    
    MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    CUDA_VISIBLE_DEVICES=5  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --kvquant --output "./logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.txt" &
    
    MODEL="meta-llama/Llama-3.3-70B-Instruct"
    CUDA_VISIBLE_DEVICES=6  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --kvquant --output "./logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.txt" &
    
    MODEL="Qwen/Qwen3-4B-Thinking-2507"
    CUDA_VISIBLE_DEVICES=7  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --kvquant --output "./logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.jsonl" 2>&1 | tee "logs/aime_benchmark/kvquant/${MODEL##*/}_aime25.txt" &
fi
