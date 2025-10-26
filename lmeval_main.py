#!/usr/bin/env python3
"""
Baseline FP16 Evaluation for LLMs
==================================

This script provides a faithful baseline evaluation that matches RoCK-KV evaluation
in every aspect EXCEPT the KV cache (uses FP16 DynamicCache instead of quantization).

All subtle task-specific parameters are preserved:
- Sampling parameters (temperature, top_p, top_k) from task YAML
- Model-specific and task-specific stop words
- Few-shot configuration per task
- Max tokens per task type
- Batch size control
- Multi-repeat evaluation with unique seeds
- Chat template application (for AIME)
- Code execution (for HumanEval)

The ONLY difference from RoCK-KV: past_key_values=None → FP16 DynamicCache

Usage Examples:
    python baseline.py "Qwen/Qwen3-8B" --task aime24 --num_repeats 10 --batch_size 2
    python baseline.py "meta-llama/Llama-3.1-8B-Instruct" --task gsm8k_cot_llama --batch_size 8
    python baseline.py "Qwen/Qwen3-8B" --task aime24 --debug
"""

import argparse
import torch
from transformers import AutoModelForCausalLM
import transformers

import llama_patch
import qwen3_patch
from llama_patch import llama_fp4_attention_forward
from qwen3_patch import qwen3_fp4_attention_forward

# Import all utilities from baseline_utils
from lmeval_utils import (
    eval_model_baseline,
    release_model_memory,
    print_gpu_memory,
)


def patch_attention():



    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_fp4_attention_forward
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = qwen3_fp4_attention_forward

    original_init_llama = transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__
    original_init_qwen3 = transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.__init__

    def patched_init_llama(self, config):
        original_init_llama(self, config)             
        self.config._attn_implementation = "eager"

    def patched_init_qwen3(self, config):
        original_init_qwen3(self, config)             
        self.config._attn_implementation = "eager"

    transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__ = patched_init_llama
    transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.__init__ = patched_init_qwen3


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser with all CLI options."""
    parser = argparse.ArgumentParser(
        description="Baseline FP16 evaluation for LLMs (matches RoCK-KV except cache)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick debug test (3 samples)
  python baseline.py "Qwen/Qwen3-8B" --task aime24 --debug

  # Production run (10 repeats)
  python baseline.py "Qwen/Qwen3-8B" --task aime24 --num_repeats 10 --batch_size 2

  # Multi-GPU for large models
  CUDA_VISIBLE_DEVICES=0,1 python baseline.py "Qwen/Qwen3-32B" --task gsm8k_cot_llama

Available tasks:
  - aime24, aime25: Math competition problems
  - gsm8k_cot_llama: Grade school math
  - minerva_math_algebra: Advanced math
  - humaneval_instruct: Code generation
  - gpqa_diamond_cot_n_shot: Graduate-level science QA
  - mmlu_flan_cot_fewshot: Multitask language understanding
        """
    )

    parser.add_argument(
        'model',
        type=str,
        help='HuggingFace model name or path (e.g., "Qwen/Qwen3-8B")'
    )

    parser.add_argument(
        "--task",
        type=str,
        default="gsm8k_cot_llama",
        help="Task name to evaluate (default: gsm8k_cot_llama)"
    )

    parser.add_argument(
        "--num_repeats",
        type=int,
        default=None,
        help="Number of evaluation repeats. If not specified, reads from task YAML (default: 1)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1). Increase for speed if memory allows"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: only evaluate 3 samples"
    )

    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model loading (default: auto)"
    )

    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples (overrides task default). Use 5 for GPQA standard evaluation"
    )

    parser.add_argument(
        "--repeat_id",
        type=int,
        default=None,
        help="Run a specific repeat ID (0-based). When set, num_repeats is ignored and only this repeat is run."
    )

    return parser


def main():
    """Main entry point for baseline evaluation."""
    args = build_parser().parse_args()

    model_short = args.model.split('/')[-1] if '/' in args.model else args.model
    llama_patch.hessian_folder = f"dumps/{model_short}_wikitext2"
    qwen3_patch.hessian_folder = f"dumps/{model_short}_wikitext2"
    patch_attention()
    llama_fp4_attention_forward.quantize_enabled = True
    qwen3_fp4_attention_forward.quantize_enabled = True
    print(f"Quantization enabled!!!")

    # Extract clean model name for directory structure
    model_name = args.model.split("/")[-1]

    # Generate output filename
    file_name = f"{model_name.lower().replace('-', '_')}_fp16_baseline_{args.task}"

    print(f"\n{'='*80}")
    print("BASELINE FP16 EVALUATION")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device Map: {args.device_map}")
    if args.repeat_id is not None:
        print(f"Repeat ID: {args.repeat_id} (running single repeat)")
    elif args.num_repeats:
        print(f"Num Repeats: {args.num_repeats}")
    else:
        print(f"Num Repeats: Will read from task YAML config (default: 1)")
    if args.num_fewshot is not None:
        print(f"Few-shot: {args.num_fewshot} examples (overriding task default)")
    else:
        print(f"Few-shot: Will use task default")
    if args.debug:
        print(f"DEBUG MODE: Limiting to 3 samples")
    print(f"{'='*80}\n")

    # Load model in FP16
    print(f"Loading model: {args.model}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map=args.device_map,
            trust_remote_code=True
        )
        print(f"✓ Model loaded successfully!")
        print(f"  Model dtype: {model.dtype}")
        print(f"  Model device: {next(model.parameters()).device}")
        print()
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("  - Make sure you have access to the model (some require authentication)")
        print("  - Try: huggingface-cli login")
        print("  - Check if the model name is correct")
        return
    ##check if attention has been patched

    # Run evaluation
    try:
        eval_model_baseline(
            model=model,
            task=args.task,
            model_name=model_name,
            file_name=file_name,
            debug=args.debug,
            num_repeats=args.num_repeats,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            repeat_id=args.repeat_id
        )
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Evaluation interrupted by user (Ctrl+C)")
        print("[Info] Progress has been saved. Re-run the same command to resume from checkpoint.")
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        print("\nCleaning up model memory...")
        release_model_memory(model)

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: eval_results/{model_name}/{args.task}/")
    print(f"  - Individual repeats: {file_name}_repeat_*.json")
    print(f"  - Summary statistics: {file_name}_summary.json")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()