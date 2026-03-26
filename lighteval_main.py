import os
import lighteval
import torch
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.models.model_input import GenerationParameters
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from datetime import datetime
import argparse
import json
from fsspec import url_to_fs

import llama_patch
import qwen3_patch
import transformers
from llama_patch import llama_fp4_attention_forward
from qwen3_patch import qwen3_fp4_attention_forward

# from deepspeed.profiling.flops_profiler import FlopsProfiler

__version__ = f"2.0_lighteval@{lighteval.__version__}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--model", type=str, required=True
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--max_model_length", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--token_forcing", type=int, default=None)
    parser.add_argument("--custom_tasks_directory", type=str, default="lighteval_tasks.py")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument('--quantize', action='store_true')
    return parser.parse_args()

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


def main():


    torch.cuda.reset_peak_memory_stats()



    start = datetime.now()
    args = parse_args()
    fs, output_dir = url_to_fs(args.output_dir)

    # Print all arguments at the start of job
    print("=" * 80)
    print("JOB STARTING - Arguments:")
    print("=" * 80)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 80)


    model_short = args.model.split('/')[-1] if '/' in args.model else args.model
    llama_patch.hessian_folder = f"dumps/{model_short}_wikitext2"
    qwen3_patch.hessian_folder = f"dumps/{model_short}_wikitext2"
    patch_attention()
    llama_fp4_attention_forward.quantize_enabled = args.quantize
    qwen3_fp4_attention_forward.quantize_enabled = args.quantize



    max_model_length = args.max_model_length
    if args.max_model_length is None:
        print("max_model_length not set. Setting it to max_new_tokens + 5000.")
        max_model_length = args.max_new_tokens + 5000
    elif args.max_model_length == -1:
        print("max_model_length is -1. Setting it to None.")
        max_model_length = None

    folder = args.model.replace("/", "_")
    fname = f"seed_{args.seed}"
    if args.quantize:
        fname += "-quant"
    if os.environ.get("SA3", "").lower() == "true":
        fname += "-sa3"
    fpath = os.path.join(output_dir, folder, f"{fname}.json")
    
    # Check if evaluation is already completed
    output_path = os.path.join(args.output_dir, f"{fname}.txt")
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                lines = f.readlines()
                if lines and lines[-1].strip() == "ALL RESULTS SAVED":
                    print(f"Evaluation already completed for {fname}. Skipping.")
                    return
        except Exception as e:
            print(f"Warning: Could not check completion status: {e}")
    
    # if fs.exists(fpath) and not args.overwrite:
    #     print(f"File {fpath} already exists. Skipping.")
    #     return


    if args.token_forcing is not None:
        system_prompt = f"Stricly use less than {args.token_forcing} tokens."
    else:
        system_prompt = None

    evaluation_tracker = EvaluationTracker(
        output_dir=args.output_dir,
        save_details=True,
        push_to_hub=False,
        push_to_tensorboard=False,
        public=False,
        hub_results_org=None,
    )
 
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        job_id=0,
        dataset_loading_processes=1,
        num_fewshot_seeds=1,
        max_samples=args.max_samples,
        load_responses_from_details_date_id=None,
        remove_reasoning_tags=False
    )


    model_config = VLLMModelConfig(
        model_name=args.model,
        dtype=args.dtype,
        seed=args.seed,
        # override_chat_template=True,
        max_model_length=max_model_length,
        max_num_seqs=args.batch_size,
        # system_prompt=system_prompt,
        generation_parameters=GenerationParameters(
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        ),
    )

    pipeline = Pipeline(
        tasks=args.task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
        metric_options={},
    )

    llm=pipeline.model.model

    
    
    # llm.start_profile()
    print("Starting pipeline evaluation...")
    eval_start_time = datetime.now()
    pipeline.evaluate()
    eval_end_time = datetime.now()
    eval_duration = eval_end_time - eval_start_time
    print(f"Pipeline evaluation completed in {eval_duration.total_seconds():.2f} seconds")
    # llm.stop_profile()

    
    pipeline.show_results()
    results = pipeline.get_results()
    pipeline.save_and_push_results()

    
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    print(f"Peak GPU memory used: {peak / 1e9:.2f} GB")

    data = {
        "start_time": start.isoformat(),
        "end_time": datetime.now().isoformat(),
        "total_evaluation_time_seconds": (datetime.now() - start).total_seconds(),
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "task": args.task,
        "max_new_tokens": args.max_new_tokens,
        "max_model_length": max_model_length,
        "dtype": args.dtype,
        "seed": args.seed,
        "system_prompt": system_prompt,
        "use_chat_template": args.use_chat_template,
        "results": results["results"]["all"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "version": __version__,
        "device_name": torch.cuda.get_device_name(),
        "lighteval_config": results["config_general"],
        "max_samples": args.max_samples,
    }

    # print(json.dumps(data, indent=2))
    # fs.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    # with fs.open(fpath, "w") as f:
    #     f.write(json.dumps(data) + "\n")

    details = pipeline.get_details()
    results = pipeline.get_results()
    # save generated text
    task_name = list(details.keys())[0]

    with open(output_path, "w") as f:
        f.write(f"{results}\n")
        f.write("="*50 + "\n")
        for detail in details[task_name]:
            f.write(f"{detail}\n")
            f.write("="*50 + "\n")
        f.write("ALL RESULTS SAVED\n")
    print(f"Generated text saved at {output_path}.")


if __name__ == "__main__":
    main()
