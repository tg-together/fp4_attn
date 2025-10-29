#!/bin/bash

# Select which mode to run: baseline, fp4, or kvquant
MODE="all"  # Options: baseline, fp4, kvquant, all

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HOME="/scratch/huggingface"
export HF_TOKEN="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"

# Use unbuffered Python output for better logging
export PYTHONUNBUFFERED=1


mkdir -p "logs/ppl_benchmark/fp4_naive/"
    
# FP4 70B model on 2 GPUs
MODEL="meta-llama/Llama-3.1-70B"
CUDA_VISIBLE_DEVICES=0,1  NAIVE_FP4=true SEARCH=false python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --batch_size 1 --quantize > "logs/ppl_benchmark/fp4_naive/${MODEL##*/}_ppl.txt" 2>&1 &

# FP4 non-70B models on 3 different GPUs
MODEL="meta-llama/Llama-3.1-8B"
CUDA_VISIBLE_DEVICES=2,3  NAIVE_FP4=true SEARCH=false python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "logs/ppl_benchmark/fp4_naive/${MODEL##*/}_ppl.txt" 2>&1 &

MODEL="Qwen/Qwen3-8B"
CUDA_VISIBLE_DEVICES=4,5  NAIVE_FP4=true SEARCH=false python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize --batch_size 3 > "logs/ppl_benchmark/fp4_naive/${MODEL##*/}_ppl.txt" 2>&1 &

MODEL="Qwen/Qwen3-4B"
CUDA_VISIBLE_DEVICES=6,7  NAIVE_FP4=true SEARCH=false python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "logs/ppl_benchmark/fp4_naive/${MODEL##*/}_ppl.txt" 2>&1 &



# mkdir -p "logs/ppl_benchmark/sage_attention/"
    
# # FP4 70B model on 2 GPUs
# MODEL="meta-llama/Llama-3.1-70B"
# CUDA_VISIBLE_DEVICES=0,1  SAGE_ATTENTION=true SEARCH=false python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --batch_size 1 --quantize > "logs/ppl_benchmark/sage_attention/${MODEL##*/}_ppl.txt" 2>&1 &

# # FP4 non-70B models on 3 different GPUs
# MODEL="meta-llama/Llama-3.1-8B"
# CUDA_VISIBLE_DEVICES=2,3  SAGE_ATTENTION=true SEARCH=false python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "logs/ppl_benchmark/sage_attention/${MODEL##*/}_ppl.txt" 2>&1 &

# MODEL="Qwen/Qwen3-8B"
# CUDA_VISIBLE_DEVICES=4,5  SAGE_ATTENTION=true SEARCH=false python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize --batch_size 3 > "logs/ppl_benchmark/sage_attention/${MODEL##*/}_ppl.txt" 2>&1 &

# MODEL="Qwen/Qwen3-4B"
# CUDA_VISIBLE_DEVICES=6,7  SAGE_ATTENTION=true SEARCH=false python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "logs/ppl_benchmark/sage_attention/${MODEL##*/}_ppl.txt" 2>&1 &


# mkdir -p "logs/ppl_benchmark/naive_search/"
    
# # FP4 70B model on 2 GPUs
# MODEL="meta-llama/Llama-3.1-70B"
# CUDA_VISIBLE_DEVICES=0,1  NAIVE_FP4=true SEARCH=true python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --batch_size 1 --quantize > "logs/ppl_benchmark/naive_search/${MODEL##*/}_ppl.txt" 2>&1 &

# # FP4 non-70B models on 3 different GPUs
# MODEL="meta-llama/Llama-3.1-8B"
# CUDA_VISIBLE_DEVICES=2,3  NAIVE_FP4=true SEARCH=true python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "logs/ppl_benchmark/naive_search/${MODEL##*/}_ppl.txt" 2>&1 &

# MODEL="Qwen/Qwen3-8B"
# CUDA_VISIBLE_DEVICES=4,5  NAIVE_FP4=true SEARCH=true python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize --batch_size 3 > "logs/ppl_benchmark/naive_search/${MODEL##*/}_ppl.txt" 2>&1 &

# MODEL="Qwen/Qwen3-4B"
# CUDA_VISIBLE_DEVICES=6,7  NAIVE_FP4=true SEARCH=true python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "logs/ppl_benchmark/naive_search/${MODEL##*/}_ppl.txt" 2>&1 &

# mkdir -p "logs/ppl_benchmark/sage_attention_search/"
    
# # FP4 70B model on 2 GPUs
# MODEL="meta-llama/Llama-3.1-70B"
# CUDA_VISIBLE_DEVICES=0,1  SAGE_ATTENTION=true SEARCH=true python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --batch_size 1 --quantize > "logs/ppl_benchmark/sage_attention_search/${MODEL##*/}_ppl.txt" 2>&1 &

# # FP4 non-70B models on 3 different GPUs
# MODEL="meta-llama/Llama-3.1-8B"
# CUDA_VISIBLE_DEVICES=2,3  SAGE_ATTENTION=true SEARCH=true python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "logs/ppl_benchmark/sage_attention_search/${MODEL##*/}_ppl.txt" 2>&1 &

# MODEL="Qwen/Qwen3-8B"
# CUDA_VISIBLE_DEVICES=4,5  SAGE_ATTENTION=true SEARCH=true python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize --batch_size 3 > "logs/ppl_benchmark/sage_attention_search/${MODEL##*/}_ppl.txt" 2>&1 &

# MODEL="Qwen/Qwen3-4B"
# CUDA_VISIBLE_DEVICES=6,7  SAGE_ATTENTION=true SEARCH=true python -u eval_ppl.py --model "${MODEL}" --dataset wikitext2 --hessian_dataset wikitext2 --quantize > "logs/ppl_benchmark/sage_attention_search/${MODEL##*/}_ppl.txt" 2>&1 &


# Wait for all background jobs to complete
wait
echo "All PPL benchmark runs completed!"