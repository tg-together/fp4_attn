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
parser.add_argument('--batch_size', default=5, type=int)
parser.add_argument('--num_samples', default=100, type=int)
parser.add_argument('--quantize', action='store_true')
parser.add_argument('--no_use_flash_attn', action='store_true')


def main(args):
    datasets = ['wikitext2']
    model_str= 'meta-llama/Meta-Llama-3-8B'
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_fp4_attention_forward
    llama_fp4_attention_forward.quantize_enabled = args.quantize
    model = AutoModelForCausalLM.from_pretrained(
        model_str, 
        trust_remote_code=True, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    model.eval() 

    for dataset in datasets:
        dataloader = data_utils.get_test_tokens(dataset,
                                                    seed=args.seed,
                                                    seqlen=args.seqlen-1,
                                                    batch_size=args.batch_size,
                                                    model=model_str)

        loss_fct = torch.nn.CrossEntropyLoss(reduction='sum').cuda()
        acc_loss = 0.0
        total_tokens = 0

        progress = tqdm(enumerate(dataloader), total=len(dataloader))
        for ii, (input,) in progress:
            input = input.cuda()  
            # print(input)

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
            print(total_tokens)         

            progress.set_description(f"avg_loss = {acc_loss / total_tokens:.4f}")


        avg_loss = acc_loss / total_tokens
        ppl = torch.exp(torch.tensor(avg_loss)).item()

        glog.info(f'{dataset} perplexity: {ppl:.4f}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)