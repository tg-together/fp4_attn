#!/bin/bash


export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"



CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --dataset wikitext2 --hessian_dataset pile --quantize  2>&1 | tee quantized_ppl.txt &

CUDA_VISIBLE_DEVICES=1 python eval_ppl.py --dataset wikitext2 2>&1 | tee unquantized_ppl.txt &