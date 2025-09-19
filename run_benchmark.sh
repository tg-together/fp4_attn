#!/bin/bash


export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"



CUDA_VISIBLE_DEVICES=0  python eval_ppl.py --quantize  --hessian_dataset pile 2>&1 | tee hadamard_hessians_ppl.txt &

CUDA_VISIBLE_DEVICES=1  IP=false python eval_ppl.py --quantize  --hessian_dataset pile 2>&1 | tee no_ip_ppl.txt &

CUDA_VISIBLE_DEVICES=2  RANDOMIZATION=ortho HESSIANS=false python eval_ppl.py --quantize  --hessian_dataset pile 2>&1 | tee ortho_no_hessians_ppl.txt &

CUDA_VISIBLE_DEVICES=3  RANDOMIZATION=ortho python eval_ppl.py --quantize  --hessian_dataset pile 2>&1 | tee ortho_hessians_ppl.txt &

CUDA_VISIBLE_DEVICES=4  MEAN_BEFORE_ROPE=true python eval_ppl.py --quantize  --hessian_dataset pile 2>&1 | tee mean_before_rope_ppl.txt &

CUDA_VISIBLE_DEVICES=5  python eval_ppl.py 2>&1 | tee unquantized_ppl.txt &
