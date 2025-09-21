#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

num_samples=20

# CUDA_VISIBLE_DEVICES=0  python lmeval_main.py --model meta-llama/Meta-Llama-3-8B --task pile_10k --quantize QKVP --num_samples $num_samples --hessian_dataset pile 2>&1 | tee quantized_lm_eval.txt &

CUDA_VISIBLE_DEVICES=6  python lmeval_main.py --model /scratch/huggingface/Llama-3.2-3B-Instruct  --task gpqa_diamond_cot_n_shot  2>&1 | tee unquantized_gpqa_diamond_cot_zeroshot_llama3Ba.txt

#CUDA_VISIBLE_DEVICES=7 python lmeval_main.py --dataset wikitext2 --record_hessian --num_samples 5 --model /scratch/huggingface/Llama-3.2-3B-Instruct  2>&1 | tee llama3B.txt --num_samples $num_samples

# gpqa_diamond_cot_n_shot