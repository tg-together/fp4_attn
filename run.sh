#!/bin/bash

# For individual command runs, you can set env vars per command like this:
# VAR1=value1 VAR2=value2 command &

# Available environment variables:
# FP4_USE_DUAL_QUANT_Q=true/false    - Use dual/single quantization for Q (query states)
# FP4_USE_DUAL_QUANT_ATTN=true/false - Use dual/single quantization for attention weights
# FP4_USE_SEARCH=true/false          - Use kernel optimizations

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Example: Default run (dual Q quantization, dual attention quantization, with kernel)
# FP4_USE_DUAL_QUANT_Q=true FP4_USE_DUAL_QUANT_ATTN=true FP4_USE_SEARCH=true python eval.py --device cuda:0 --model meta-llama/Meta-Llama-3-8B --task pile_10k --tag llama3_8B_after_rope_token_wise --quantize --num_samples 10 2>&1 | tee temp.txt 
num_samples=10
# # Example background runs with different configurations:

# Run 1: Dual Q quantization, dual attention quantization, no kernel
CUDA_VISIBLE_DEVICES=1 FP4_USE_DUAL_QUANT_Q=true FP4_USE_DUAL_QUANT_ATTN=true FP4_USE_SEARCH=false python eval.py --device cuda:1 --model meta-llama/Meta-Llama-3-8B --task pile_10k --tag llama3_8B_dual_q_dual_attn_no_search --quantize --num_samples $num_samples 2>&1 | tee logs/dual_q_dual_attn_no_search.txt &

# # # # # Run 2: Single Q quantization, single attention quantization, no kernel
CUDA_VISIBLE_DEVICES=2 FP4_USE_DUAL_QUANT_Q=false FP4_USE_DUAL_QUANT_ATTN=false FP4_USE_SEARCH=false python eval.py --device cuda:2 --model meta-llama/Meta-Llama-3-8B --task pile_10k --tag llama3_8B_single_q_single_attn_no_search --quantize --num_samples $num_samples 2>&1 | tee logs/single_q_single_attn_no_search.txt &

# # # # # Run 3: Dual Q quantization, single attention quantization, with kernel
CUDA_VISIBLE_DEVICES=3 FP4_USE_DUAL_QUANT_Q=true FP4_USE_DUAL_QUANT_ATTN=true FP4_USE_SEARCH=true python eval.py --device cuda:3 --model meta-llama/Meta-Llama-3-8B --task pile_10k --tag llama3_8B_dual_q_dual_attn_with_search --quantize --num_samples $num_samples 2>&1 | tee logs/dual_q_dual_attn_with_search.txt &

# # # # Run 4: Single Q quantization, dual attention quantization, with kernel
CUDA_VISIBLE_DEVICES=4 FP4_USE_DUAL_QUANT_Q=false FP4_USE_DUAL_QUANT_ATTN=false FP4_USE_SEARCH=true python eval.py --device cuda:4 --model meta-llama/Meta-Llama-3-8B --task pile_10k --tag llama3_8B_single_q_single_attn_with_search --quantize --num_samples $num_samples 2>&1 | tee logs/single_q_single_attn_with_search.txt &

# # cd ~/fp4_attn/LiveCodeBench && CUDA_VISIBLE_DEVICES=4 python -m lcb_runner.runner.main --model meta-llama/Meta-Llama-3-8B --scenario codegeneration --evaluate --release_version release_v6 --custom_output_save_name ~/fp4_attn/eval/llama3_8B_lcb 2>&1 | tee ~/fp4_attn/eval/llama3_8B_lcb.txt


CUDA_VISIBLE_DEVICES=5 python eval.py --device cuda:5 --model meta-llama/Meta-Llama-3-8B --task pile_10k --tag llama3_8B --num_samples $num_samples 2>&1 | tee logs/baseline_llama3_8B_pile_10k.txt &   
