#!/bin/bash


export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"


# Model and dataset configuration
MODEL="meta-llama/Meta-Llama-3-8B"
DATASET="wikitext2"
# Create logs directory
MODEL_SHORT="${MODEL##*/}"  # Extract model name after last slash
LOG_DIR="logs/${MODEL_SHORT}/${DATASET}"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=0  python eval_ppl.py --model "${MODEL}" --dataset "${DATASET}" --quantize  --hessian_dataset pile 2>&1 | tee "${LOG_DIR}/hadamard_hessians_ppl.txt" &

CUDA_VISIBLE_DEVICES=1  IP=false python eval_ppl.py --model "${MODEL}" --dataset "${DATASET}" --quantize  --hessian_dataset pile 2>&1 | tee "${LOG_DIR}/no_ip_ppl.txt" &

CUDA_VISIBLE_DEVICES=2  RANDOMIZATION=ortho HESSIANS=false python eval_ppl.py --model "${MODEL}" --dataset "${DATASET}" --quantize  --hessian_dataset pile 2>&1 | tee "${LOG_DIR}/ortho_no_hessians_ppl.txt" &

CUDA_VISIBLE_DEVICES=3  RANDOMIZATION=ortho python eval_ppl.py --model "${MODEL}" --dataset "${DATASET}" --quantize  --hessian_dataset pile 2>&1 | tee "${LOG_DIR}/ortho_hessians_ppl.txt" &

MODEL="meta-llama/Meta-Llama-3-8B"
DATASET="c4"
# Create logs directory
MODEL_SHORT="${MODEL##*/}"  # Extract model name after last slash
LOG_DIR="logs/${MODEL_SHORT}/${DATASET}"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=4  python eval_ppl.py --model "${MODEL}" --dataset "${DATASET}" --quantize  --hessian_dataset pile 2>&1 | tee "${LOG_DIR}/hadamard_hessians_ppl.txt" &

CUDA_VISIBLE_DEVICES=5  IP=false python eval_ppl.py --model "${MODEL}" --dataset "${DATASET}" --quantize  --hessian_dataset pile 2>&1 | tee "${LOG_DIR}/no_ip_ppl.txt" &

CUDA_VISIBLE_DEVICES=6  RANDOMIZATION=ortho HESSIANS=false python eval_ppl.py --model "${MODEL}" --dataset "${DATASET}" --quantize  --hessian_dataset pile 2>&1 | tee "${LOG_DIR}/ortho_no_hessians_ppl.txt" &

CUDA_VISIBLE_DEVICES=7  RANDOMIZATION=ortho python eval_ppl.py --model "${MODEL}" --dataset "${DATASET}" --quantize  --hessian_dataset pile 2>&1 | tee "${LOG_DIR}/ortho_hessians_ppl.txt" &
