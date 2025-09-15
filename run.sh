#!/bin/bash


# Available environment variables:
# FP4_USE_DUAL_QUANT_Q=true/false    - Use dual/single quantization for Q (query states)
# FP4_USE_DUAL_QUANT_ATTN=true/false - Use dual/single quantization for attention weights
# FP4_USE_P_SEARCH=true/false        - Search mode for P quant
# FP4_USE_Q_SEARCH=true/false        - Search mode for Q/K/V quant
# ZERO_POINT=min|mean                - Zero-point centering option for Q/K/V
# SHIFTED_SM=true|false              - Use shifted softmax for P quantization
# FP4_QUANTIZE=QKVP                  - Selective quantization spec (subset of Q,K,V,P). If set, you can omit --quantize.

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"


num_samples=20


CUDA_VISIBLE_DEVICES=0 MEAN_BEFORE_ROPE=true python eval.py --device cuda:0 --model meta-llama/Meta-Llama-3-8B --task pile_10k --quantize QKVP --num_samples $num_samples 2>&1 | tee temp.txt &

CUDA_VISIBLE_DEVICES=1  python eval.py --device cuda:1 --model meta-llama/Meta-Llama-3-8B --task pile_10k --num_samples $num_samples 2>&1 | tee temp1.txt &
