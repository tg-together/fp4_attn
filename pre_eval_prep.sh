#!/bin/bash



export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="/scratch/huggingface"
export HF_TOKEN="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"

mkdir -p "logs/record/"

#FP4 Record Hessians

# # # Model and dataset configuration
# MODEL="meta-llama/Llama-3.1-8B"
# CUDA_VISIBLE_DEVICES=0  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --record_hessian 2>&1 | tee "logs/record/${MODEL##*/}_hessian_record.txt" &

MODEL="meta-llama/Llama-3.1-70B"
CUDA_VISIBLE_DEVICES=1  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --record_hessian --batch_size 1 2>&1 | tee "logs/record/${MODEL##*/}_hessian_record.txt" &

MODEL="Qwen/Qwen3-8B"
CUDA_VISIBLE_DEVICES=2  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --record_hessian 2>&1 | tee "logs/record/${MODEL##*/}_hessian_record.txt" &

MODEL="Qwen/Qwen3-4B"
CUDA_VISIBLE_DEVICES=3  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --record_hessian 2>&1 | tee "logs/record/${MODEL##*/}_hessian_record.txt" &

MODEL="meta-llama/Llama-3.2-3B-Instruct"
CUDA_VISIBLE_DEVICES=4  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --record_hessian 2>&1 | tee "logs/record/${MODEL##*/}_hessian_record.txt" &

MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
CUDA_VISIBLE_DEVICES=5  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --record_hessian 2>&1 | tee "logs/record/${MODEL##*/}_hessian_record.txt" &

MODEL="meta-llama/Llama-3.3-70B-Instruct"
CUDA_VISIBLE_DEVICES=6  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --record_hessian --batch_size 1 2>&1 | tee "logs/record/${MODEL##*/}_hessian_record.txt" &

MODEL="Qwen/Qwen3-4B-Thinking-2507"
CUDA_VISIBLE_DEVICES=7  python eval_ppl.py --model "${MODEL}" --dataset wikitext2 --record_hessian 2>&1 | tee "logs/record/${MODEL##*/}_hessian_record.txt" &

