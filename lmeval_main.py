#!/usr/bin/env python3
import argparse
import json
import os
import sys
from lm_eval import simple_evaluate
from transformers import AutoModelForCausalLM, AutoConfig
import transformers
import torch
import llama_patch
import qwen3_patch
from llama_patch import llama_fp4_attention_forward
from qwen3_patch import qwen3_fp4_attention_forward
import socket
from datetime import datetime, timedelta
import logging
import numpy as np


def get_kvquant_model(model_name):
    """Get KVQuant quantized model with default settings from run.sh"""
    # Add KVQuant path to sys.path
    kvquant_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'KVQuant', 'quant')
    kvquant_path_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'KVQuant')
    sys.path.insert(0, kvquant_path)
    
    from llama_simquant import run_kvquant, create_parser
    
    # Default KVQuant arguments from run.sh (without seqlen - matching lmeval_main.py)
    kvquant_args = [
        '--abits', '4',
        '--nuq',
        '--first_few_fp16', '1',
        '--quantizer-path', f'{kvquant_path_root}/output/{model_name.split("/")[-1]}/quantizers.pickle'
    ]
    
    # Parse KVQuant arguments
    parser = create_parser()
    args = parser.parse_args([model_name] + kvquant_args)
    
    # Get quantized model
    model = run_kvquant(args, return_model=True)

    print(f"KVQuant model fetched")
    
    # Remove from path
    sys.path.remove(kvquant_path)
    
    return model


def calculate_perplexity(model, tasks, num_samples=None, device="auto", max_length=2048, **eval_kwargs):
    """Calculate perplexity using lm_eval."""
    
    # Base arguments
    base_args = {
        "model": "hf",
        "model_args": {"pretrained":model,"max_length":max_length,"trust_remote_code":True},
        "tasks": tasks,
        "batch_size": 1,
        "device": device,
        "confirm_run_unsafe_code":True
    }
    
    # Add limit if specified
    if num_samples:
        base_args["limit"] = num_samples
    
    # Merge with additional eval arguments
    base_args.update(eval_kwargs)

    
    results = simple_evaluate(**base_args)
    return results


def parse_arguments():
    """Parse command line arguments and return args and eval_kwargs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name to evaluate")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--visualize", action="store_true", help="Enable QKV visualization")
    parser.add_argument("--record_hessian", action="store_true", help="Record Q Hessian")
    parser.add_argument("--record_means", action="store_true", help="Record K means")
    parser.add_argument("--quantize", type=str, default="", help="Selective FP4 quantization; subset of 'QKVP'")
    parser.add_argument("--tag", default="", help="Tag to append to filenames")
    parser.add_argument("--task", nargs='+', default=["pile_10k", "gsm8k"], help="Task(s) to evaluate (can specify multiple)")
    parser.add_argument("--hessian_dataset", type=str, default="wikitext2", help="Dataset name to load hessians from (e.g., 'pile_10k')")
    parser.add_argument("--kvquant", action="store_true", help="Use KVQuant quantization")
    
    # Parse known args to capture additional eval arguments
    args, unknown_args = parser.parse_known_args()
    
    # Convert unknown args to kwargs for simple_evaluate
    eval_kwargs = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        
        # Handle key=value format
        if '=' in arg and not arg.startswith('--'):
            key, value = arg.split('=', 1)
            # Try to convert to appropriate type
            if value.lower() == 'true':
                eval_kwargs[key] = True
            elif value.lower() == 'false':
                eval_kwargs[key] = False
            elif value.isdigit():
                eval_kwargs[key] = int(value)
            else:
                eval_kwargs[key] = value
            i += 1
        # Handle --key value format
        elif arg.startswith('--'):
            key = arg[2:].replace('-', '_')
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                # Has value
                value = unknown_args[i + 1]
                # Try to convert to appropriate type
                if value.lower() == 'true':
                    eval_kwargs[key] = True
                elif value.lower() == 'false':
                    eval_kwargs[key] = False
                elif value.isdigit():
                    eval_kwargs[key] = int(value)
                else:
                    eval_kwargs[key] = value
                i += 2
            else:
                # Boolean flag
                eval_kwargs[key] = True
                i += 1
        else:
            i += 1
    
    return args, eval_kwargs


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

def patch_lm_eval(kv_quant_model):
    """Example patches for lm_eval.models.huggingface functions."""
    # Need to import the specific submodule - lm_eval doesn't expose .models directly
    from lm_eval.models import huggingface

    
    def patched_create_model(self, *args, **kwargs):

        self._model = kv_quant_model
        print(f"Using pre-initialized KVQuant model: {self._model}")
        # Otherwise use original creation logic
        return
    
    huggingface.HFLM._create_model = patched_create_model

def main():
    args, eval_kwargs = parse_arguments()
    model_str = args.model
    
    print("=" * 50)
    print("RUNNING WITH ARGUMENTS:")
    print("=" * 50)
    for arg, value in vars(args).items():
        print(f"{arg:25}: {value}")
    if eval_kwargs:
        print("Additional eval args:")
        for key, value in eval_kwargs.items():
            print(f"{key:25}: {value}")
    print("=" * 50)

    # Set quantization flag for llama_patch
    llama_patch.visualize = args.visualize
    qwen3_patch.visualize = args.visualize

    # Set hessian dataset path for loading

    model_short = args.model.split('/')[-1] if '/' in args.model else args.model
    llama_patch.hessian_folder = f"dumps/{model_short}_{args.hessian_dataset}"
    qwen3_patch.hessian_folder = f"dumps/{model_short}_{args.hessian_dataset}"


    if args.visualize:
        llama_fp4_attention_forward.visualize = args.visualize
        qwen3_fp4_attention_forward.visualize = args.visualize

    if args.kvquant:
        args.quantize = False
        kv_quant_model = get_kvquant_model(model_str)
        patch_lm_eval(kv_quant_model)

          
    else:
        patch_attention()
        llama_fp4_attention_forward.quantize_enabled = args.quantize
        qwen3_fp4_attention_forward.quantize_enabled = args.quantize
    
    
    # Common evaluation path for both FP4 and KVQuant
    for max_length in [32768+1]:
        with torch.no_grad():
            results = calculate_perplexity(
                model=args.model,
                tasks=args.task,
                device="auto",
                num_samples=args.num_samples,
                max_length=max_length,
                **eval_kwargs
            )
        

        print(f"Model: {args.model}")
        print(f"Max length: {max_length}")
        print(f"Quantize: {args.quantize}")
        print(f"Tag: {args.tag}")
        for task in args.task:
            print(f"Task: {task}")
            try:
                print(f"Metrics: {results['results'][task]}")
            except:
                if results is not None:
                    print(f"Metrics: {results['results']}")

            
        if args.output:
            output_filename = f"{args.output}_{args.tag}.json" if args.tag else args.output
            with open(output_filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
    
    # if args.visualize:
    #     # Create a combined tag that includes both the original tag and task names
    #     task_str = "_".join(args.task)
    #     combined_tag = f"{args.tag}_{task_str}" if args.tag else task_str
    #     save_plots(combined_tag)

if __name__ == "__main__":
    main() 