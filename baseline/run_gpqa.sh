


export HF_HOME=/scratch/huggingface
export HF_TOKEN=XXX

export HF_HOME_DATASETS=/scratch/huggingface/datasets


CUDA_VISIBLE_DEVICES=0 python baseline.py "Qwen/Qwen3-8B" --num_repeats 3 --task gpqa_diamond_cot_n_shot --num_fewshot 5 --debug


CUDA_VISIBLE_DEVICES=0 python baseline.py "Qwen/Qwen3-14B" --num_repeats 3 --task gpqa_diamond_cot_n_shot  --num_fewshot 5



###
# Llama
CUDA_VISIBLE_DEVICES=0 python baseline.py "meta-llama/Llama-3.1-8B-Instruct" --num_repeats 3 --task gpqa_diamond_cot_n_shot  --num_fewshot 5


###
# Llama
CUDA_VISIBLE_DEVICES=0 python baseline.py "meta-llama/Llama-3.3-70B-Instruct" --num_repeats 3 --task gpqa_diamond_cot_n_shot  --num_fewshot 5


