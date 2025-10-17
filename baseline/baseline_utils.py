"""Helper functions for baseline FP16 evaluation."""

import os
import gc
import json
import copy
import torch
import numpy as np
import lm_eval
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Any, Optional


########################################### Memory & Utility Functions ###########################################

def print_gpu_memory():
    """Print GPU memory usage for all visible devices."""
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def release_model_memory(model):
    """Release model memory from GPU."""
    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()


########################################### Checkpoint Management ###########################################

def load_completed_repeats(file_dir: str, file_name: str, num_repeats: int) -> Tuple[List[int], List[Dict]]:
    """
    Check which repeats are already completed and load their results.
    Only loads minimal data (tasks and repeat_idx) to save memory.

    Args:
        file_dir: Directory containing repeat result files
        file_name: Base filename for repeat results
        num_repeats: Total number of repeats expected

    Returns:
        Tuple of (list of completed repeat indices, list of minimal results for summary)
    """
    completed_repeats = []
    all_results = []

    for repeat_idx in range(num_repeats):
        repeat_file = f"{file_dir}/{file_name}_repeat_{repeat_idx}.json"
        if os.path.exists(repeat_file):
            print(f"[Checkpoint] Found existing result for repeat {repeat_idx}: {repeat_file}")
            completed_repeats.append(repeat_idx)
            # Only load minimal data needed for summary (not samples)
            with open(repeat_file, "r") as f:
                full_result = json.load(f)
                minimal_result = {
                    "repeat_idx": full_result["repeat_idx"],
                    "tasks": full_result["tasks"]
                }
                all_results.append(minimal_result)
                del full_result  # Free memory immediately

    return completed_repeats, all_results


def print_checkpoint_status(completed_repeats: List[int], num_repeats: int) -> bool:
    """
    Print checkpoint status and return whether all repeats are completed.

    Args:
        completed_repeats: List of completed repeat indices
        num_repeats: Total number of repeats expected

    Returns:
        True if all repeats are completed, False otherwise
    """
    if len(completed_repeats) == num_repeats:
        print(f"[Checkpoint] All {num_repeats} repeats already completed. Skipping evaluation.")
        return True
    else:
        print(f"[Checkpoint] {len(completed_repeats)}/{num_repeats} repeats completed. "
              f"Continuing from repeat {len(completed_repeats)}.")
        return False


########################################### Output JSON Builder ###########################################

def build_eval_output(
    results: Dict,
    task: str,
    repeat_idx: int,
    model,
    model_configs: Dict
) -> Dict:
    """
    Build output dictionary in lm_eval format.

    Args:
        results: Results dictionary from lm_eval.simple_evaluate
        task: Task name
        repeat_idx: Current repeat index
        model: The model being evaluated
        model_configs: Model configuration dictionary

    Returns:
        Dictionary in lm_eval output format
    """
    cdt_timezone = timezone(timedelta(hours=-5))  # CDT: UTC-5

    output = {}
    output["This file is generated_at"] = datetime.now(cdt_timezone).isoformat()
    output["repeat_idx"] = repeat_idx
    output["tasks"] = results['results']
    output["model_configs"] = model_configs

    # Only deep copy eval_configs to avoid modifying the original (important for multiple repeats)
    output["eval_configs"] = copy.deepcopy(results["config"])

    # Model dtype
    output["eval_configs"]["model_dtype"] = str(model.dtype)

    # No special handling needed for past_key_values since it's None for baseline

    if "samples" in results:
        # For samples, we only copy the fields we need
        output["samples"] = {}

        for task_name, task_samples in results["samples"].items():
            output["samples"][task_name] = []

            # Extract shared gen_kwargs from the first sample (only once)
            if task_samples and "arguments_from_samples" not in output["eval_configs"]:
                first_sample = task_samples[0]
                arguments_from_samples = copy.deepcopy(first_sample["arguments"][0][1])
                output["eval_configs"]["arguments_from_samples"] = arguments_from_samples

            # Process each sample - only copy the fields we need
            for sample in task_samples:
                sample_copy = {
                    "doc_id": sample.get("doc_id"),
                    "doc": sample.get("doc"),
                    "target": sample.get("target"),
                    "resps": sample.get("resps"),
                    "filtered_resps": sample.get("filtered_resps"),
                    "arguments": sample["arguments"][0][0],  # Only keep the first part (without gen_kwargs)
                }
                # Add other fields if they exist
                for key in ["doc_hash", "prompt", "metrics"]:
                    if key in sample:
                        sample_copy[key] = sample[key]

                output["samples"][task_name].append(sample_copy)

    return output


########################################### Evaluation Loop ###########################################

def run_evaluation_repeats(
    lm,
    model,
    task: str,
    num_fewshot: int,
    limit: Optional[int],
    gen_kwargs: Dict,
    model_configs: Dict,
    file_dir: str,
    file_name: str,
    num_repeats: int,
    completed_repeats: List[int],
    all_results: List[Dict],
    batch_size: int = 8,
    base_random_seed: int = 0,
    base_numpy_seed: int = 1234,
    base_torch_seed: int = 1234,
    base_fewshot_seed: int = 1234
) -> List[Dict]:
    """
    Run evaluation for remaining repeats and save results.

    Args:
        lm: Language model instance (lm_eval model)
        model: The PreTrainedModel being evaluated
        task: Task name (e.g., "aime24")
        num_fewshot: Number of few-shot examples
        limit: Number of samples to evaluate (None for all)
        gen_kwargs: Generation kwargs
        model_configs: Model configuration dictionary
        file_dir: Directory to save results
        file_name: Base filename for results
        num_repeats: Total number of repeats to run
        completed_repeats: List of already completed repeat indices
        all_results: List to accumulate results (will be modified in-place)
        batch_size: Batch size for inference (default: 8)
        base_random_seed: Base random seed for Python's random module (default: 0)
        base_numpy_seed: Base random seed for numpy (default: 1234)
        base_torch_seed: Base random seed for torch (default: 1234)
        base_fewshot_seed: Base random seed for fewshot sampler (default: 1234)

    Returns:
        Updated list of all results (same as all_results parameter)
    """
    # Task-specific configuration (same for all repeats)
    task_lower = task.lower()
    if "aime" in task_lower:
        # AIME: apply chat template, but not fewshot_as_multiturn
        apply_chat = True
        fewshot_multiturn = False
        print(f"Task config: {task} (apply_chat_template=True, fewshot_as_multiturn=False)")
    else:
        # All other tasks: disable both
        apply_chat = False
        fewshot_multiturn = False
        print(f"Task config: {task} (apply_chat_template=False, fewshot_as_multiturn=False)")

    # Check if task requires code execution (unsafe tasks like humaneval, mbpp)
    unsafe_task_keywords = ["humaneval"]
    needs_unsafe_code = any(keyword in task_lower for keyword in unsafe_task_keywords)
    if needs_unsafe_code:
        print(f"Task requires code execution, setting confirm_run_unsafe_code=True")

    for repeat_idx in range(num_repeats):
        if repeat_idx in completed_repeats:
            continue

        # Calculate unique seeds for this repeat
        current_random_seed = base_random_seed + repeat_idx
        current_numpy_seed = base_numpy_seed + repeat_idx
        current_torch_seed = base_torch_seed + repeat_idx
        current_fewshot_seed = base_fewshot_seed + repeat_idx

        print(f"\n{'='*80}")
        print(f"Running repeat {repeat_idx + 1}/{num_repeats}")
        print(f"Seeds: random={current_random_seed}, numpy={current_numpy_seed}, "
              f"torch={current_torch_seed}, fewshot={current_fewshot_seed}")
        print(f"{'='*80}\n")

        # Run evaluation with unique seeds for this repeat
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=num_fewshot,
            limit=limit,
            batch_size=batch_size,
            gen_kwargs=gen_kwargs,
            log_samples=True,
            apply_chat_template=apply_chat,
            fewshot_as_multiturn=fewshot_multiturn,
            # Different seeds for each repeat
            random_seed=current_random_seed,
            numpy_random_seed=current_numpy_seed,
            torch_random_seed=current_torch_seed,
            fewshot_random_seed=current_fewshot_seed,
            # Only allow unsafe code execution for specific tasks
            confirm_run_unsafe_code=needs_unsafe_code,
        )

        print(f"\n[Completed] All samples evaluated for repeat {repeat_idx}")

        # Build output in lm_eval format
        output = build_eval_output(
            results=results,
            task=task,
            repeat_idx=repeat_idx,
            model=model,
            model_configs=model_configs
        )

        # Save individual repeat result (with full samples) to file
        repeat_file = f"{file_dir}/{file_name}_repeat_{repeat_idx}.json"
        with open(repeat_file, "w") as f:
            json.dump(output, f, indent=4)
        print(f"[Saved] Repeat {repeat_idx} result: {repeat_file}")

        # Only keep minimal data in memory for summary statistics (drop samples to save memory)
        minimal_result = {
            "repeat_idx": output["repeat_idx"],
            "tasks": output["tasks"]
        }
        all_results.append(minimal_result)

        # Clean up memory after each repeat to prevent OOM
        del results
        del output
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[Memory] Cleaned up after repeat {repeat_idx}")

    return all_results


########################################### Summary Statistics ###########################################

def generate_summary_statistics(
    all_results: List[Dict],
    task: str,
    model_configs: Dict[str, Any],
    num_repeats: int
) -> Dict:
    """
    Generate summary statistics from multiple repeat results.

    Args:
        all_results: List of result dictionaries from each repeat
        task: Task name (e.g., "aime24")
        model_configs: Model configuration dictionary
        num_repeats: Total number of repeats

    Returns:
        Dictionary containing summary statistics (mean, std, variance, min, max, median)
    """
    cdt_timezone = timezone(timedelta(hours=-5))  # CDT: UTC-5

    summary = {
        "generated_at": datetime.now(cdt_timezone).isoformat(),
        "num_repeats": num_repeats,
        "model_configs": model_configs,
        "task": task,
        "statistics": {}
    }

    # Extract metrics from all repeats
    task_name = list(all_results[0]["tasks"].keys())[0]
    metrics = all_results[0]["tasks"][task_name].keys()

    for metric in metrics:
        # Skip stderr metrics (they contain "_stderr" in the name)
        if "_stderr" in metric:
            continue

        values = []
        for result in all_results:
            value = result["tasks"][task_name].get(metric)
            if isinstance(value, (int, float)):
                values.append(value)

        if values:
            values_array = np.array(values)
            summary["statistics"][metric] = {
                "values": values,
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "variance": float(np.var(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "median": float(np.median(values_array)),
            }

    return summary


########################################### Main Evaluation Function ###########################################

@torch.no_grad()
def eval_model_baseline(
    model,
    task: str,
    model_name: str,
    file_name: str,
    debug: bool = False,
    num_repeats: Optional[int] = None,
    batch_size: int = 8,
    num_fewshot: Optional[int] = None
):
    """
    Evaluate model on downstream tasks with baseline FP16 (no quantization).

    Args:
        model: PreTrainedModel to evaluate
        task: Task name (e.g., "aime24", "gsm8k_cot_llama")
        model_name: Clean model name for directory structure
        file_name: Filename for saving results
        debug: If True, limit to 3 samples and 1 repeat
        num_repeats: Number of times to repeat the evaluation (for sampling-based tasks).
                     If None, will try to read from task's YAML config. If not found, defaults to 1.
        batch_size: Batch size for inference (default: 8).
        num_fewshot: Number of few-shot examples (overrides task default). Use 5 for GPQA standard evaluation.
    """
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    lm = HFLM(model, batch_size=batch_size)

    # If num_repeats is not specified, try to read from task config
    if num_repeats is None:
        try:
            task_manager = TaskManager()
            task_dict = task_manager.load_task_or_group([task])
            task_obj = task_dict[task][0] if isinstance(task_dict[task], list) else task_dict[task]
            num_repeats = getattr(task_obj.config, 'repeats', 1)
            print(f"[Info] Using repeats={num_repeats} from task YAML config")
        except Exception as e:
            print(f"[Warning] Could not read repeats from task config: {e}. Using default repeats=1")
            num_repeats = 1

    # Simplified model configs (no RoCK-KV parameters)
    model_configs = {
        "ModelArch": model.__class__.__name__,
        "ModelPath": model.config._name_or_path
    }

    # Set the number of shots for different tasks
    if num_fewshot is None:
        # Use defaults if not specified
        few_shot_dict = {"mmlu": 4, "gsm8k": 8, "gpqa": 5, "math": 4, "bbh": 3}
        for key, value in few_shot_dict.items():
            if key in task:
                num_fewshot = value
                break
        else:
            num_fewshot = 0
        print(f"[Info] Using default num_fewshot={num_fewshot} for task '{task}'")
    else:
        print(f"[Info] Using override num_fewshot={num_fewshot} (overriding task default)")

    # Model-specific stop words
    model_name_lower = model.config._name_or_path.lower()

    if "qwen" in model_name_lower:
        # Qwen models
        stop_words = [
            "<|endoftext|>",       # Qwen-3  (151643)   real EOS
            "<|im_end|>",          # Qwen-3  (151645)   message ended
        ]
    elif "llama" in model_name_lower:
        # LLaMA models
        stop_words = [
            "<|end_of_text|>",     # Llama-3 (128001)   real EOS
            "<|eot_id|>",          # Llama-3 (128009)   turn ended
            "<|end_header_id|>",   # Llama-3 (128008)   head ended
        ]
    else:
        # Fallback: include all common stop tokens
        stop_words = [
            "<|end_of_text|>",     # Llama-3
            "<|eot_id|>",          # Llama-3
            "<|end_header_id|>",   # Llama-3
            "<|endoftext|>",       # Qwen-3
            "<|im_end|>",          # Qwen-3
        ]

    # Set the stop words for different tasks
    stop_words_dict = {
        "gsm8k":     ["Given the following problem"],
        "math":      ["Problem:"],
        "gpqa":      ["Question:"],
        "humaneval": ["\n```"],
        "aime":      ["Given the following problem"],
    }
    for key, value in stop_words_dict.items():
        if key in task:
            stop_words.extend(value)
            break

    # Set max_new_tokens based on task type
    # AIME tasks need longer generation for detailed reasoning
    if "aime24" in task or "aime25" in task:
        max_new_tokens = 32768
        apply_chat = True
        fewshot_multiturn = False
        print(f"Task config: {task} (apply_chat_template=True, fewshot_as_multiturn=False)")
 
    else:
        # All other tasks: disable both
        apply_chat = False
        fewshot_multiturn = False
        max_new_tokens = 4096
        print(f"Task config: {task} (apply_chat_template=False, fewshot_as_multiturn=False)")
    
    gen_kwargs = {
        # "past_key_values": None,  # Use default HF DynamicCache (FP16)
        "max_new_tokens": max_new_tokens,
        "max_length": None,
        "until": stop_words,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_new_tokens": 4096,
    }

    # Enable code evaluation for humaneval task
    if "humaneval" in task:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    # For DEBUG mode, limit the number of samples
    limit = None
    if debug:
        limit = 3

    print("=" * 80)
    print("Evaluating model accuracy on downstream tasks (Baseline FP16)...")
    print("ModelName: ", model_configs["ModelPath"])
    print("Task: ", task)
    print("Eval_Configs: ", file_name)
    print("Num_Fewshot: ", num_fewshot)
    print("Num_Repeats: ", num_repeats)
    print("Batch_Size: ", batch_size)
    print("GPU Memory:")
    print_gpu_memory()
    print("=" * 80)

    # Prepare output directory
    file_dir = "./eval_results/{}/{}".format(model_name, task)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)

    # Check which repeats are already completed (for checkpoint resumption)
    completed_repeats, all_results = load_completed_repeats(file_dir, file_name, num_repeats)
    all_completed = print_checkpoint_status(completed_repeats, num_repeats)

    # Run evaluations for remaining repeats (skip if all completed)
    if not all_completed:
        all_results = run_evaluation_repeats(
            lm=lm,
            model=model,
            task=task,
            num_fewshot=num_fewshot,
            limit=limit,
            gen_kwargs=gen_kwargs,
            model_configs=model_configs,
            file_dir=file_dir,
            file_name=file_name,
            num_repeats=num_repeats,
            completed_repeats=completed_repeats,
            all_results=all_results,
            batch_size=batch_size
        )

    # Generate summary statistics across all repeats
    print(f"\n{'='*80}")
    print("Generating summary statistics...")
    print(f"{'='*80}\n")

    summary = generate_summary_statistics(all_results, task, model_configs, num_repeats)

    # Save summary file
    summary_file = "{}/{}_summary.json".format(file_dir, file_name)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[Saved] Summary statistics: {summary_file}")
