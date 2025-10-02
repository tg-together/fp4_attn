import argparse
import json
import math
import os
import sys
import random
from transformers import AutoModelForCausalLM
import datasets
import glog
import torch
from tqdm import tqdm
from llama_patch import llama_fp4_attention_forward
from qwen3_patch import qwen3_fp4_attention_forward
import data_utils
import transformers
from transformers import AutoModelForCausalLM


torch.set_grad_enabled(False)


def save_qk_hessians(model_name, dataset, tag=""):
    """Save the running averages of Q Hessian per layer.
    Each mean has shape [1, H, 1, D] where H is the number of heads."""
    from llama_patch import hessian_running_averages as llama_hess
    from qwen3_patch import hessian_running_averages as qwen_hess

    hessian_running_averages = llama_hess if llama_hess['counts'] else qwen_hess

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
    filename = f"{folder_name}/qk_hessians_{tag}.pt" if tag else f"{folder_name}/qk_hessians.pt"
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
    filename = f"{folder_name}/k_mean_averages_before_rope_{tag}.pt" if tag else f"{folder_name}/k_mean_averages_before_rope.pt"
    torch.save(averages, filename)
    
    print(f"Saved K means to {filename}")


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--seqlen', default=16384, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--num_samples', default=50, type=int)
parser.add_argument('--quantize', action='store_true')
parser.add_argument('--no_use_flash_attn', action='store_true')
parser.add_argument("--dataset", nargs='+', default=["wikitext2"], help="Task(s) to evaluate (can specify multiple)")
parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B", type=str)
parser.add_argument("--record_hessian", action="store_true", help="Record Q/K Hessian")
parser.add_argument("--record_means", action="store_true", help="Record K means")
parser.add_argument("--tag", default="", help="Tag to append to filenames")
parser.add_argument("--hessian_dataset", type=str, default="wikitext2", help="Dataset name to load hessians from (e.g., 'wikitext2')")
parser.add_argument("--kvquant", action="store_true", help="Use KVQuant quantization")


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
    
    # Remove from path
    sys.path.remove(kvquant_path)
    
    return model


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


def main(args):
    datasets = args.dataset
    model_str= args.model
    
    # Configure llama_patch with hessian settings
    import llama_patch
    import qwen3_patch
    

    model_short = model_str.split('/')[-1] if '/' in model_str else model_str
    llama_patch.hessian_folder = f"dumps/{model_short}_{args.hessian_dataset}"
    qwen3_patch.hessian_folder = f"dumps/{model_short}_{args.hessian_dataset}"

    
    if args.record_hessian:
        args.quantize = False
        args.kvquant = False
        llama_fp4_attention_forward.store_hessian = True
        qwen3_fp4_attention_forward.store_hessian = True

    
    if args.record_means:
        args.quantize = False
        llama_fp4_attention_forward.store_means = True
        qwen3_fp4_attention_forward.store_means = True

    # Handle KVQuant model saving/loading
    if args.kvquant:
        args.quantize = False
        model = get_kvquant_model(model_str)

          
    else:
        patch_attention()
        model = AutoModelForCausalLM.from_pretrained(
            model_str, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        llama_fp4_attention_forward.quantize_enabled = args.quantize
        qwen3_fp4_attention_forward.quantize_enabled = args.quantize
    
    # Common model loading path for both FP4 and KVQuant



    model.eval() 
    first_device = next(model.parameters()).device

    
    for dataset in datasets:
        print("Dataset: ", dataset)
        # Use train data when recording hessian or means
        use_train = args.record_hessian or args.record_means
        dataloader = data_utils.get_test_tokens(dataset,
                                                    nsamples=args.num_samples,
                                                    seed=args.seed,
                                                    seqlen=args.seqlen-1,
                                                    batch_size=args.batch_size,
                                                    model=model_str,
                                                    train=use_train)

        loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
        acc_loss = 0.0
        total_tokens = 0

        progress = tqdm(enumerate(dataloader), total=len(dataloader))
        for ii, (input,) in progress:

            input = input.to(first_device)  
            output = model(
                input,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False
            )[0] 

            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = input[:, 1:]

        
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)).to(torch.float32),
                shift_labels.reshape(-1)
            )

            acc_loss += loss.item()
            total_tokens += shift_labels.numel()         

            progress.set_description(f"avg_loss = {acc_loss / total_tokens:.4f}")
            del output, shift_logits, shift_labels, loss
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
                print(f"  Cached:    {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        


        avg_loss = acc_loss / total_tokens
        ppl = torch.exp(torch.tensor(avg_loss)).item()

        glog.info(f'{dataset} perplexity: {ppl:.4f}')
        
        # Save hessians and means after evaluation
        if args.record_hessian:
            save_qk_hessians(model_str, dataset, args.tag)
        
        if args.record_means:
            save_k_means(model_str, dataset, args.tag)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)