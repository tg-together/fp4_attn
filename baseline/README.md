

## 🎯 Popular Tasks
 "gsm8k_cot_llama" "minerva_math_algebra" "humaneval_instruct" "gpqa_diamond_cot_n_shot" "mmlu_flan_cot_fewshot" "aime24" "aime25"
| Task | Description | Typical Repeats |
|------|-------------|-----------------|
| `aime24` | Math competition problems | 10 |
| `gsm8k_cot_llama` | Grade school math | 5 |
| `humaneval_instruct` | Code generation | 3 |
| `gpqa_diamond_cot_n_shot` | Graduate science QA | 3 |


```bash
# Quick test (debug mode, 3 samples only)
CUDA_VISIBLE_DEVICES=0 python baseline.py "Qwen/Qwen3-4B-Thinking-2507" --task aime24 --debug

# Production run
CUDA_VISIBLE_DEVICES=0 python baseline.py "Qwen/Qwen3-4B-Thinking-2507" --task aime24 --num_repeats 10 --batch_size 2


CUDA_VISIBLE_DEVICES=0,1 python baseline.py "Qwen/Qwen3-4B-Thinking-2507" --task gsm8k_cot  --debug

# Different tasks
# Multi-GPU for larger models (32B)
CUDA_VISIBLE_DEVICES=0,1 python baseline.py "meta-llama/Llama-3.1-8B-Instruct" --task gsm8k_cot_llama  --debug

CUDA_VISIBLE_DEVICES=0 python baseline.py "Qwen/Qwen3-8B" --task humaneval_instruct --batch_size 4
CUDA_VISIBLE_DEVICES=1 python baseline.py "meta-llama/Llama-3.1-8B-Instruct" --task gpqa_diamond_cot_n_shot
```
