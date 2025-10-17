export HF_HOME=/scratch/huggingface
export HF_TOKEN=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

export HF_HOME_DATASETS=/scratch/huggingface/datasets


CUDA_VISIBLE_DEVICES=0 python baseline.py "Qwen/Qwen3-4B-Thinking-2507" --num_repeats 10 --task aime24


CUDA_VISIBLE_DEVICES=0 python baseline.py "Qwen/Qwen3-8B" --num_repeats 10 --task aime2 --debug


CUDA_VISIBLE_DEVICES=0 python baseline.py "Qwen/Qwen3-14B" --num_repeats 10 --task aime24  

# Llama
CUDA_VISIBLE_DEVICES=0 python baseline.py "Qwen/Qwen3-8B" --num_repeats 10 --task aime25  


CUDA_VISIBLE_DEVICES=1 python baseline.py "Qwen/Qwen3-4B-Thinking-2507"  --num_repeats 10 --task aime25 --debug

###
# Llama
CUDA_VISIBLE_DEVICES=0 python baseline.py "Qwen/Qwen3-14B" --num_repeats 10 --task aime25 --num_fewshot 5



