#!/bin/bash

# Select which mode to run: baseline, fp4, or kvquant
MODE="all"  # Options: baseline, fp4, kvquant

# Task-specific configuration
TASK="gsm8k_cot"
NUM_FEWSHOT="8"

# Common suffix for all commands
SUFFIX_CMD="--apply_chat_template --num_fewshot ${NUM_FEWSHOT} --fewshot_as_multiturn --log_samples"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="/scratch/huggingface"
export HF_TOKEN="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"

if [ "$MODE" = "baseline" ] || [ "$MODE" = "all" ]; then
    mkdir -p "logs/gsm8k_benchmark/baseline/"
    
    #baseline
    

    MODEL="meta-llama/Llama-3.2-3B-Instruct"
    CUDA_VISIBLE_DEVICES=4  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --output "./logs/gsm8k_benchmark/baseline/${MODEL##*/}_gsm8k.jsonl" --task gsm8k_cot_llama 2>&1 | tee "logs/gsm8k_benchmark/baseline/${MODEL##*/}_gsm8k.txt" &
    
    MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    CUDA_VISIBLE_DEVICES=5  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --output "./logs/gsm8k_benchmark/baseline/${MODEL##*/}_gsm8k.jsonl" --task gsm8k_cot 2>&1 | tee "logs/gsm8k_benchmark/baseline/${MODEL##*/}_gsm8k.txt" &
    
    # MODEL="meta-llama/Llama-3.3-70B-Instruct"
    # CUDA_VISIBLE_DEVICES=6  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --output "./logs/gsm8k_benchmark/baseline/${MODEL##*/}_gsm8k.jsonl" --task gsm8k_cot_llama 2>&1 | tee "logs/gsm8k_benchmark/baseline/${MODEL##*/}_gsm8k.txt" &
    
    MODEL="Qwen/Qwen3-4B-Thinking-2507"
    CUDA_VISIBLE_DEVICES=7  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --output "./logs/gsm8k_benchmark/baseline/${MODEL##*/}_gsm8k.jsonl" --task gsm8k_cot 2>&1 | tee "logs/gsm8k_benchmark/baseline/${MODEL##*/}_gsm8k.txt" &

elif [ "$MODE" = "fp4" ] || [ "$MODE" = "all" ]; then
    mkdir -p "logs/gsm8k_benchmark/fp4/"
    
    #FP4
    
    MODEL="meta-llama/Llama-3.2-3B-Instruct"
    CUDA_VISIBLE_DEVICES=0  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --quantize --output "./logs/gsm8k_benchmark/fp4/${MODEL##*/}_gsm8k.jsonl" --task gsm8k_cot_llama 2>&1 | tee "logs/gsm8k_benchmark/fp4/${MODEL##*/}_gsm8k.txt" &
    
    MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    CUDA_VISIBLE_DEVICES=1  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --quantize --output "./logs/gsm8k_benchmark/fp4/${MODEL##*/}_gsm8k.jsonl" --task gsm8k_cot 2>&1 | tee "logs/gsm8k_benchmark/fp4/${MODEL##*/}_gsm8k.txt" &
    
    # MODEL="meta-llama/Llama-3.3-70B-Instruct"
    # CUDA_VISIBLE_DEVICES=6  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --quantize --output "./logs/gsm8k_benchmark/fp4/${MODEL##*/}_gsm8k.jsonl" --task gsm8k_cot_llama 2>&1 | tee "logs/gsm8k_benchmark/fp4/${MODEL##*/}_gsm8k.txt" &
    
    MODEL="Qwen/Qwen3-4B-Thinking-2507"
    CUDA_VISIBLE_DEVICES=2  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --quantize --output "./logs/gsm8k_benchmark/fp4/${MODEL##*/}_gsm8k.jsonl" --task gsm8k_cot 2>&1 | tee "logs/gsm8k_benchmark/fp4/${MODEL##*/}_gsm8k.txt" &

elif [ "$MODE" = "kvquant" ] || [ "$MODE" = "all" ]; then
    mkdir -p "logs/gsm8k_benchmark/kvquant/"
    
    #KVquant
    

    MODEL="meta-llama/Llama-3.2-3B-Instruct"
    CUDA_VISIBLE_DEVICES=3  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --kvquant --output "./logs/gsm8k_benchmark/kvquant/${MODEL##*/}_gsm8k.jsonl" --task gsm8k_cot_llama 2>&1 | tee "logs/gsm8k_benchmark/kvquant/${MODEL##*/}_gsm8k.txt" 
    
    MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    CUDA_VISIBLE_DEVICES=6  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --kvquant --output "./logs/gsm8k_benchmark/kvquant/${MODEL##*/}_gsm8k.jsonl" --task gsm8k_cot 2>&1 | tee "logs/gsm8k_benchmark/kvquant/${MODEL##*/}_gsm8k.txt" &
    
    # MODEL="meta-llama/Llama-3.3-70B-Instruct"
    # CUDA_VISIBLE_DEVICES=6  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --kvquant --output "./logs/gsm8k_benchmark/kvquant/${MODEL##*/}_gsm8k.jsonl" --task gsm8k_cot_llama 2>&1 | tee "logs/gsm8k_benchmark/kvquant/${MODEL##*/}_gsm8k.txt" &
    
    # MODEL="Qwen/Qwen3-4B-Thinking-2507"
    # CUDA_VISIBLE_DEVICES=7  python lmeval_main.py --model "${MODEL}" ${SUFFIX_CMD} --kvquant --output "./logs/gsm8k_benchmark/kvquant/${MODEL##*/}_gsm8k.jsonl" --task gsm8k_cot 2>&1 | tee "logs/gsm8k_benchmark/kvquant/${MODEL##*/}_gsm8k.txt" &

fi
