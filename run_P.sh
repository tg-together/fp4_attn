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


num_samples=2500


CUDA_VISIBLE_DEVICES=0 FP4_USE_DUAL_QUANT_Q=true FP4_USE_Q_SEARCH=true FP4_USE_DUAL_QUANT_ATTN=false FP4_USE_P_SEARCH=false SHIFTED_SM=false python eval.py --device cuda:0 --model meta-llama/Meta-Llama-3-8B --task pile_10k --num_samples $num_samples --quantize QKV 2>&1 | tee logs/P_not_quantized.txt &

CUDA_VISIBLE_DEVICES=1 FP4_USE_DUAL_QUANT_Q=true FP4_USE_Q_SEARCH=true FP4_USE_DUAL_QUANT_ATTN=false FP4_USE_P_SEARCH=false SHIFTED_SM=false python eval.py --device cuda:1 --model meta-llama/Meta-Llama-3-8B --task pile_10k --num_samples $num_samples --quantize QKVP 2>&1 | tee logs/P_single_nosearch.txt &

CUDA_VISIBLE_DEVICES=2 FP4_USE_DUAL_QUANT_Q=true FP4_USE_Q_SEARCH=true FP4_USE_DUAL_QUANT_ATTN=false FP4_USE_P_SEARCH=true SHIFTED_SM=false python eval.py --device cuda:2 --model meta-llama/Meta-Llama-3-8B --task pile_10k --num_samples $num_samples --quantize QKVP 2>&1 | tee logs/P_single_search.txt &

CUDA_VISIBLE_DEVICES=3 FP4_USE_DUAL_QUANT_Q=true FP4_USE_Q_SEARCH=true FP4_USE_DUAL_QUANT_ATTN=true FP4_USE_P_SEARCH=false SHIFTED_SM=false python eval.py --device cuda:3 --model meta-llama/Meta-Llama-3-8B --task pile_10k --num_samples $num_samples --quantize QKVP 2>&1 | tee logs/P_dual_nosearch.txt &

CUDA_VISIBLE_DEVICES=4 FP4_USE_DUAL_QUANT_Q=true FP4_USE_Q_SEARCH=true FP4_USE_DUAL_QUANT_ATTN=true FP4_USE_P_SEARCH=true SHIFTED_SM=false python eval.py --device cuda:4 --model meta-llama/Meta-Llama-3-8B --task pile_10k --num_samples $num_samples --quantize QKVP 2>&1 | tee logs/P_dual_search.txt &

CUDA_VISIBLE_DEVICES=5 FP4_USE_DUAL_QUANT_Q=true FP4_USE_Q_SEARCH=true FP4_USE_DUAL_QUANT_ATTN=true FP4_USE_P_SEARCH=true SHIFTED_SM=true python eval.py --device cuda:5 --model meta-llama/Meta-Llama-3-8B --task pile_10k --num_samples $num_samples --quantize QKVP 2>&1 | tee logs/P_dual_search_shifted.txt &

CUDA_VISIBLE_DEVICES=6 FP4_USE_DUAL_QUANT_Q=true FP4_USE_Q_SEARCH=true FP4_USE_DUAL_QUANT_ATTN=false FP4_USE_P_SEARCH=true SHIFTED_SM=true python eval.py --device cuda:6 --model meta-llama/Meta-Llama-3-8B --task pile_10k --num_samples $num_samples --quantize QKVP 2>&1 | tee logs/P_single_search_shifted.txt &

CUDA_VISIBLE_DEVICES=7 FP4_USE_DUAL_QUANT_Q=true FP4_USE_Q_SEARCH=true FP4_USE_DUAL_QUANT_ATTN=true FP4_USE_P_SEARCH=false SHIFTED_SM=true python eval.py --device cuda:7 --model meta-llama/Meta-Llama-3-8B --task pile_10k --num_samples $num_samples --quantize QKVP 2>&1 | tee logs/P_dual_nosearch_shifted.txt &


# CUDA_VISIBLE_DEVICES=7 python eval.py --device cuda:7 --model meta-llama/Meta-Llama-3-8B --task pile_10k --num_samples $num_samples 2>&1 | tee logs/baseline.txt &
