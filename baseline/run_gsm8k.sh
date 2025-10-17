export HF_HOME=/scratch/huggingface
export HF_TOKEN=XXX

export HF_HOME_DATASETS=/scratch/huggingface/datasets


CUDA_VISIBLE_DEVICES=0 python baseline.py "Qwen/Qwen3-4B-Thinking-2507" --num_repeats 3 --task  gsm8k_cot

CUDA_VISIBLE_DEVICES=0 python baseline.py "Qwen/Qwen3-4B-Thinking-2507" --num_repeats 3 --task gsm8k_cot_zeroshot



CUDA_VISIBLE_DEVICES=0 python baseline.py "Qwen/Qwen3-14B" --num_repeats 3 --task  gsm8k_cot

CUDA_VISIBLE_DEVICES=0 python baseline.py "Qwen/Qwen3-14B" --num_repeats 3 --task gsm8k_cot_zeroshot


###
# Llama
CUDA_VISIBLE_DEVICES=0 python baseline.py "meta-llama/Llama-3.1-8B-Instruct" --num_repeats 3 --task  gsm8k_cot_llama

CUDA_VISIBLE_DEVICES=0 python baseline.py "meta-llama/Llama-3.1-8B-Instruct" --num_repeats 3 --task gsm8k_cot_zeroshot

###
# Llama
CUDA_VISIBLE_DEVICES=0 python baseline.py "meta-llama/Llama-3.3-70B-Instruct" --num_repeats 3 --task  gsm8k_cot_llama

CUDA_VISIBLE_DEVICES=0 python baseline.py "meta-llama/Llama-3.3-70B-Instruct" --num_repeats 3 --task gsm8k_cot_zeroshot


