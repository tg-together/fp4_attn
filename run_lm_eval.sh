#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

num_samples=100

CUDA_VISIBLE_DEVICES=0  python lmeval_main.py --model meta-llama/Meta-Llama-3-8B --task pile_10k --quantize QKVP --num_samples $num_samples 2>&1 | tee quantized_lm_eval.txt &

CUDA_VISIBLE_DEVICES=1  python lmeval_main.py --model meta-llama/Meta-Llama-3-8B --task pile_10k --num_samples $num_samples 2>&1 | tee unquantized_lm_eval.txt &
