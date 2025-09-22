#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

num_samples=20

CUDA_VISIBLE_DEVICES=0  python lmeval_main.py --model meta-llama/Meta-Llama-3-8B --task pile_10k --quantize QKVP --num_samples $num_samples --hessian_dataset pile 2>&1 | tee quantized_lm_eval.txt &

CUDA_VISIBLE_DEVICES=1  python lmeval_main.py --model meta-llama/Meta-Llama-3-8B --task pile_10k --num_samples $num_samples 2>&1 | tee unquantized_lm_eval.txt &


##GPQA
# CUDA_VISIBLE_DEVICES=6  python lmeval_main.py \
#  --model /scratch/huggingface/Llama-3.2-3B-Instruct \
#  --task gpqa_diamond_cot_n_shot \
#  --apply_chat_template \
#  --num_fewshot 16 \
#  --fewshot_as_multiturn --log_samples --output ./test.jsonl 2>&1 | tee llama3Ba.txt

##gsm8k
# CUDA_VISIBLE_DEVICES=6  python lmeval_main.py \
#  --model /scratch/huggingface/Llama-3.2-3B-Instruct \
#  --task gsm8k_cot \
#  --apply_chat_template \
#  --num_fewshot 8 \
#  --fewshot_as_multiturn --log_samples --output ./test.jsonl 2>&1 | tee llama3Ba.txt


##aime25
CUDA_VISIBLE_DEVICES=6  python lmeval_main.py \
 --model /scratch/huggingface/Llama-3.2-3B-Instruct \
 --task aime25 \
 --apply_chat_template \
 --num_fewshot 8 \
 --fewshot_as_multiturn --log_samples --output ./test.jsonl --num_samples $num_samples 2>&1 | tee llama3Ba.txt
