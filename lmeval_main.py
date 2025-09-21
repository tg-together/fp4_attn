#!/usr/bin/env python3
import argparse
import json
from lm_eval import simple_evaluate
import lm_eval
print(f"lm_eval path: {lm_eval.__file__}")
from transformers import AutoModelForCausalLM, AutoConfig
import transformers
import torch
import llama_patch
from llama_patch import llama_fp4_attention_forward
import socket
from datetime import datetime, timedelta
import logging
import numpy as np
import os


def save_qk_hessians(model_name, dataset, tag=""):
    """Save the running averages of Q Hessian per layer.
    Each mean has shape [1, H, 1, D] where H is the number of heads."""
    from llama_patch import hessian_running_averages
    
    if not hessian_running_averages['counts']:
        print("No Q/K Hessian averages to save")
        return
    
    # Collect the averages (already computed incrementally)
    averages = {}
    for layer_idx in sorted(hessian_running_averages['counts'].keys()):
        count = hessian_running_averages['counts'][layer_idx]
        if count > 0:
            averages[f'layer_{layer_idx}'] = {
                'q_hessian': hessian_running_averages['q_means'][layer_idx],  # Shape: [1, H_q, 1, D]
                'k_hessian': hessian_running_averages['k_means'][layer_idx],  # Shape: [1, H_k, 1, D]
            }
    
    # Create folder structure
    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
    folder_name = f"dumps/{model_short}_{dataset}"
    os.makedirs(folder_name, exist_ok=True)
    
    # Save to file using torch.save
    filename = f"dumps/{folder_name}/qk_hessians_{tag}.pt" if tag else f"dumps/{folder_name}/qk_hessians.pt"
    torch.save(averages, filename)
    
    print(f"Saved Q/K Hessian to {filename}")


def save_k_means(model_name, dataset, tag=""):
    """Save the running averages of K means per layer.
    Each mean has shape [1, H_k, 1, D] where H_k is the number of KV heads."""
    from llama_patch import means_running_averages
    
    if not means_running_averages['counts']:
        print("No K means to save")
        return
    
    # Collect the averages (already computed incrementally)
    averages = {}
    for layer_idx in sorted(means_running_averages['counts'].keys()):
        count = means_running_averages['counts'][layer_idx]
        if count > 0:
            averages[f'layer_{layer_idx}'] = {
                'k_mean_avg': means_running_averages['k_means'][layer_idx],  # Shape: [1, H_k, 1, D]
            }
    
    # Create folder structure
    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
    folder_name = f"dumps/{model_short}_{dataset}"
    os.makedirs(folder_name, exist_ok=True)
    
    # Save to file using torch.save
    filename = f"dumps/{folder_name}/k_mean_averages_before_rope_{tag}.pt" if tag else f"dumps/{folder_name}/k_mean_averages_before_rope.pt"
    torch.save(averages, filename)
    
    print(f"Saved K means to {filename}")


def calculate_perplexity(model, tasks, num_samples=None, device="auto", max_length=2048, **eval_kwargs):
    """Calculate perplexity using lm_eval."""
    
    # Base arguments max_length={max_length},
    base_args = {
        "model": "hf",
        "model_args": f"pretrained={model},trust_remote_code=True",
        "tasks": tasks,
        "batch_size": 1,
        "device": device,
        "confirm_run_unsafe_code":True,
        "apply_chat_template":True,
        "fewshot_as_multiturn":True,
        "num_fewshot":16,
        # "show_config":True,
        "log_samples":True
    }
    
    # Add limit if specified
    if num_samples:
        base_args["limit"] = num_samples
    
    # Merge with additional eval arguments
    base_args.update(eval_kwargs)
    print (base_args)
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
    original_init = transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__

    def patched_init(self, config):
        original_init(self, config)             
        self.config._attn_implementation = "eager"
    transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__ = patched_init


def main():
    args, eval_kwargs = parse_arguments()

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
    llama_patch.quantize_enabled = args.quantize.upper() if isinstance(args.quantize, str) else args.quantize
    llama_patch.visualize = args.visualize

    # Set hessian dataset path for loading

    model_short = args.model.split('/')[-1] if '/' in args.model else args.model
    llama_patch.hessian_folder = f"dumps/{model_short}_{args.hessian_dataset}"


    if args.record_hessian:
        args.quantize = False
        llama_fp4_attention_forward.store_hessian = True

    if args.record_means:
        args.quantize = False
        llama_fp4_attention_forward.store_means = True

    if args.visualize:
        llama_fp4_attention_forward.visualize = args.visualize

    if args.quantize:
        llama_fp4_attention_forward.quantize_enabled = args.quantize.upper() if isinstance(args.quantize, str) else args.quantize

    if args.quantize or args.visualize or args.record_hessian or args.record_means:
        patch_attention()   ## Enable FP4 attention

    # start_record_memory_history()

    for max_length in [32000]:
        with torch.no_grad():
            results = calculate_perplexity(
                model=args.model,
                tasks=args.task,
                device="auto",
                num_samples=args.num_samples,
                max_length=max_length,
                **eval_kwargs
                )
        
        if args.record_hessian:
            task_str = "_".join(args.task) if len(args.task) > 1 else args.task[0]
            save_qk_hessians(args.model, task_str, args.tag)
        
        if args.record_means:
            task_str = "_".join(args.task) if len(args.task) > 1 else args.task[0]
            save_k_means(args.model, task_str, args.tag)
        
        # export_memory_snapshot()
        # stop_record_memory_history()
        
        # Print results using a similar format to standard lm-evaluation-harness
        if results is not None:
            # Print model info similar to standard CLI
            batch_sizes = ",".join(map(str, results["config"].get("batch_sizes", [])))
            print(
                f"{args.model} (pretrained={args.model}), gen_kwargs: (None), limit: {args.num_samples}, num_fewshot: 0, "
                f"batch_size: 20{f' ({batch_sizes})' if batch_sizes else ''}"
            )
            
            # Print formatted table manually
            def print_results_table(result_dict):
                """Print results in a table format similar to lm-evaluation-harness"""
                print(f"{'Tasks':<25}|{'Version':<7}|{'Filter':<15}|{'n-shot':<6}|{'Metric':<11}|{'':<3}|{'Value':<6}|{'':<3}|{'Stderr':<6}|")
                print("-" * 90)
                
                for task_name in result_dict["results"]:
                    task_results = result_dict["results"][task_name]
                    version = result_dict.get("versions", {}).get(task_name, "1")
                    n_shot = str(result_dict.get("n-shot", {}).get(task_name, "0"))
                    higher_is_better = result_dict.get("higher_is_better", {}).get(task_name, {})
                    
                    # Handle alias
                    display_name = task_results.get("alias", task_name)
                    
                    # Sort metrics for consistent display
                    metric_items = sorted(task_results.items())
                    
                    first_row = True
                    for metric_key, value in metric_items:
                        if metric_key == "alias":
                            continue
                            
                        # Parse metric name and filter
                        if "," in metric_key:
                            metric, filter_name = metric_key.split(",", 1)
                        else:
                            metric, filter_name = metric_key, ""
                            
                        # Skip stderr entries (they'll be handled with their main metric)
                        if metric.endswith("_stderr"):
                            continue
                            
                        # Get stderr if available
                        stderr_key = f"{metric}_stderr,{filter_name}" if filter_name else f"{metric}_stderr"
                        stderr = task_results.get(stderr_key, "")
                        stderr_str = f"±{stderr:.4f}" if isinstance(stderr, (int, float)) else ""
                        
                        # Format value
                        value_str = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                        
                        # Higher is better symbol
                        hib_symbol = "↑" if higher_is_better.get(metric, None) else ""
                        
                        # Print row
                        name_col = display_name if first_row else ""
                        version_col = str(version) if first_row else ""
                        print(f"{name_col:<25}|{version_col:<7}|{filter_name:<15}|{n_shot:<6}|{metric:<11}|{hib_symbol:<3}|{value_str:<6}|{'':<3}|{stderr_str:<6}|")
                        first_row = False
            
            print_results_table(results)
            
            # Additional debug info
            print(f"Model: {args.model}")
            print(f"Max length: {max_length}")
            print(f"Quantize: {args.quantize}")
            print(f"Tag: {args.tag}")
            for task in args.task:
                print(f"Task: {task}")
        else:
            print("No results returned from evaluation")

        if args.output:
            output_filename = f"{args.output}_{args.tag}.json" if args.tag else args.output
            with open(output_filename, 'w') as f:
                json.dump(results, f, indent=2)
    
    if args.visualize:
        # Create a combined tag that includes both the original tag and task names
        task_str = "_".join(args.task)
        combined_tag = f"{args.tag}_{task_str}" if args.tag else task_str
        save_plots(combined_tag)

if __name__ == "__main__":
    main() 