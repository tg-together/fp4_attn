#!/usr/bin/env python3
import argparse
import json
from lm_eval import simple_evaluate
from transformers import AutoModelForCausalLM, AutoConfig
import transformers
import torch
from visualize import collect_qkv, collect_qkv_diff, save_plots
import llama_patch
from llama_patch import llama_fp4_attention_forward
import socket
from datetime import datetime, timedelta
import logging

llama_patch.collect_qkv = collect_qkv
llama_patch.collect_qkv_diff = collect_qkv_diff


# MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000000

# # Make collect_qkv available to llama_patch module


# logging.basicConfig(
#    format="%(levelname)s:%(asctime)s %(message)s",
#    level=logging.INFO,
#    datefmt="%Y-%m-%d %H:%M:%S",
# )
# logger: logging.Logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.INFO)

# TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


# def start_record_memory_history() -> None:
#    if not torch.cuda.is_available():
#        logger.info("CUDA unavailable. Not recording memory history")
#        return

#    logger.info("Starting snapshot record_memory_history")
#    torch.cuda.memory._record_memory_history(
#        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
#    )

# def stop_record_memory_history() -> None:
#    if not torch.cuda.is_available():
#        logger.info("CUDA unavailable. Not recording memory history")
#        return

#    logger.info("Stopping snapshot record_memory_history")
#    torch.cuda.memory._record_memory_history(enabled=None)

# def export_memory_snapshot() -> None:
#    if not torch.cuda.is_available():
#        logger.info("CUDA unavailable. Not exporting memory snapshot")
#        return

#    # Prefix for file names.
#    host_name = socket.gethostname()
#    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
#    file_prefix = f"{host_name}_{timestamp}"

#    try:
#        logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
#        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
#    except Exception as e:
#        logger.error(f"Failed to capture memory snapshot {e}")
#        return


def print_pile_10k_metrics(metrics, tag="", model="", quantize=False):
    """Print pile_10k metrics as comma-separated values for Excel paste."""
    
    # Extract the metrics we want
    word_perplexity = metrics.get('word_perplexity,none', 'N/A')
    byte_perplexity = metrics.get('byte_perplexity,none', 'N/A')
    bits_per_byte = metrics.get('bits_per_byte,none', 'N/A')
    
    # Create a comma-separated line
    csv_line = f"{tag},{model},{quantize},{word_perplexity},{byte_perplexity},{bits_per_byte}"
    
    print("\n" + "="*80)
    print("PILE_10K METRICS FOR EXCEL:")
    print("="*80)
    print("Header: Tag,Model,Quantize,Word_Perplexity,Byte_Perplexity,Bits_Per_Byte")
    print(f"Data:   {csv_line}")
    print("="*80)
    
    return csv_line


def calculate_perplexity(model, tasks, num_samples=None, device="auto", max_length=2048, **eval_kwargs):
    """Calculate perplexity using lm_eval."""
    
    # Base arguments
    base_args = {
        "model": "hf",
        "model_args": f"pretrained={model},max_length={max_length}",
        "tasks": tasks,
        "num_fewshot": 0,
        "batch_size": 20,
        "device": device,
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
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--visualize", action="store_true", help="Enable QKV visualization")
    parser.add_argument("--quantize", type=str, default="", help="Selective FP4 quantization; subset of 'QKVP'")
    parser.add_argument("--tag", default="", help="Tag to append to filenames")
    parser.add_argument("--task", nargs='+', default=["pile_10k", "gsm8k"], help="Task(s) to evaluate (can specify multiple)")
    
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
    """Monkey patch the LlamaAttention forward method with FP4 version."""
    print("Patching FP4 attention")
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_fp4_attention_forward



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

    if args.visualize:
        llama_fp4_attention_forward.visualize = args.visualize

    if args.quantize:
        llama_fp4_attention_forward.quantize_enabled = args.quantize.upper() if isinstance(args.quantize, str) else args.quantize

    # if args.quantize or args.visualize:
    patch_attention()   ## Enable FP4 attention

    # start_record_memory_history()

    for max_length in [2048]:
        with torch.no_grad():
            results = calculate_perplexity(
                model=args.model,
                tasks=args.task,
                device=args.device,
                num_samples=args.num_samples,
                max_length=max_length,
                **eval_kwargs
                )
        # export_memory_snapshot()
        # stop_record_memory_history()
        print(f"Model: {args.model}")
        print(f"Max length: {max_length}")
        print(f"Quantize: {args.quantize}")
        print(f"Tag: {args.tag}")
        for task in args.task:
            print(f"Task: {task}")
            print(f"Metrics: {results['results'][task]}")
            
            # Special formatting for pile_10k metrics
            if task == "pile_10k":
                print_pile_10k_metrics(
                    metrics=results['results'][task],
                    tag=args.tag,
                    model=args.model,
                    quantize=args.quantize
                )

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