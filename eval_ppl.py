import argparse
import json
import math
import os
import random
from transformers import AutoModelForCausalLM
import datasets
import glog
import torch
from tqdm import tqdm
from llama_patch import llama_fp4_attention_forward
import data_utils
import transformers
from transformers import AutoModelForCausalLM


torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='hfized/quantized_hada_70b', type=str)
parser.add_argument('--seqlen', default=8192, type=int)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--num_samples', default=100, type=int)
parser.add_argument('--quantize', action='store_true')
parser.add_argument('--no_use_flash_attn', action='store_true')

def patch_attention():
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_fp4_attention_forward
    original_init = transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__

    def patched_init(self, config):
        original_init(self, config)             
        self.config._attn_implementation = "eager"
    transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__ = patched_init


def main(args):
    datasets = ['wikitext2']
    model_str= 'meta-llama/Meta-Llama-3-8B'
    patch_attention()
    llama_fp4_attention_forward.quantize_enabled = args.quantize
    model = AutoModelForCausalLM.from_pretrained(
        model_str, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )


    model.eval() 
    first_device = next(model.parameters()).device
    for dataset in datasets:
        dataloader = data_utils.get_test_tokens(dataset,
                                                    seed=args.seed,
                                                    seqlen=args.seqlen-1,
                                                    batch_size=args.batch_size,
                                                    model=model_str)

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
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )

            acc_loss += loss.item()
            total_tokens += shift_labels.numel()         

            progress.set_description(f"avg_loss = {acc_loss / total_tokens:.4f}")
            del input, output, shift_logits, shift_labels, loss
            torch.cuda.empty_cache()


        avg_loss = acc_loss / total_tokens
        ppl = torch.exp(torch.tensor(avg_loss)).item()

        glog.info(f'{dataset} perplexity: {ppl:.4f}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)