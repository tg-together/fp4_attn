#!/usr/bin/env python3
"""
Evaluation script for weight quantization using the existing framework.
Integrates qwen3_patch_weightQuant.py with eval_ppl.py and lmeval_main.py patterns.
"""

import argparse
import json
import math
import os
import sys
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import glog
import torch
from tqdm import tqdm
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

# Import existing utilities
import data_utils
import transformers

# Import your weight quantization patch
import qwen3_patch_weightQuant as weight_patch

torch.set_grad_enabled(False)


def apply_weight_quantization_patch(model, args):
    """Apply weight quantization to Qwen3 model."""
    print("Applying weight quantization patch...")
    
    # Enable quantization
    weight_patch.qwen3_weight_quantized_attention_forward.quantize_enabled = True
    
    # Set configuration from args
    os.environ['FP4_USE_DUAL_WEIGHT_QUANT'] = 'true' if args.use_dual_weight_quant else 'false'
    precision = getattr(args, 'weight_precision', 'fp4')
    os.environ['WEIGHT_PRECISION'] = precision
    
    # Apply to all attention modules
    patched_count = 0
    for name, module in model.named_modules():
        if isinstance(module, Qwen3Attention):
            # Store original forward for comparison if needed
            module._original_forward = module.forward
            
            # Replace with weight quantized forward
            module.forward = weight_patch.qwen3_weight_quantized_attention_forward.__get__(
                module, Qwen3Attention
            )
            patched_count += 1
    
    print(f"Successfully patched {patched_count} attention modules")
    return model


def evaluate_perplexity_with_weight_quant(args):
    """Evaluate perplexity using weight quantization."""
    
    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if args.precision == 'fp32' else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Apply weight quantization if requested
    if args.quantize_weights:
        model = apply_weight_quantization_patch(model, args)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == 'wikitext2':
        testdata = data_utils.get_wikitext2(args.nsamples, args.seed, args.seqlen, args.batch_size, args.model)
    elif args.dataset == 'pile_10k':
        testdata = data_utils.get_pile(args.nsamples, args.seed, args.seqlen, args.batch_size, args.model)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Evaluate perplexity
    print("Evaluating perplexity...")
    model.eval()
    
    nlls = []
    n_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(testdata, desc="Evaluating")):
            # Extract input_ids from batch tuple
            input_ids = batch[0].to(model.device)
            
            # Forward pass
            outputs = model(input_ids, labels=input_ids.clone())
            neg_log_likelihood = outputs.loss
            
            nlls.append(neg_log_likelihood.item())
            n_samples += 1
            
            if (i + 1) % 10 == 0:
                current_ppl = torch.exp(torch.tensor(nlls).mean()).item()
                print(f"Sample {i+1}/{n_samples}, Current PPL: {current_ppl:.3f}")
    
    # Calculate final perplexity
    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    print(f"\nFinal Perplexity: {ppl:.3f}")
    
    # Analyze quantization if applied
    if args.quantize_weights:
        print("\nAnalyzing weight quantization...")
        analyze_weight_quantization_impact(model)
    
    return ppl


def analyze_weight_quantization_impact(model):
    """Analyze the impact of weight quantization on the model."""
    total_original_params = 0
    total_quantized_params = 0
    layer_errors = []
    
    for name, module in model.named_modules():
        if isinstance(module, Qwen3Attention) and hasattr(module, 'weight_quantized'):
            if module.weight_quantized:
                print(f"\nAnalyzing layer: {name}")
                errors = weight_patch.analyze_quantization_error(module)
                layer_errors.append(errors)
                
                # Count parameters
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    proj = getattr(module, proj_name)
                    total_original_params += proj.weight.numel()
                    
                # Quantized weights are roughly 1/4 the size for FP4
                total_quantized_params += total_original_params // 4
    
    if layer_errors:
        # Calculate average errors
        avg_errors = {'q_proj': 0, 'k_proj': 0, 'v_proj': 0, 'o_proj': 0}
        for errors in layer_errors:
            for proj, error in errors.items():
                avg_errors[proj] += error
        
        for proj in avg_errors:
            avg_errors[proj] /= len(layer_errors)
        
        print(f"\n{'='*50}")
        print("WEIGHT QUANTIZATION SUMMARY")
        print(f"{'='*50}")
        print(f"Layers analyzed: {len(layer_errors)}")
        print(f"Original parameters: {total_original_params:,}")
        print(f"Estimated quantized parameters: {total_quantized_params:,}")
        print(f"Compression ratio: {total_original_params/total_quantized_params:.1f}x")
        print(f"\nAverage quantization errors:")
        for proj, error in avg_errors.items():
            print(f"  {proj}: {error:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Weight Quantization Evaluation")
    
    # Model and dataset
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-4B', 
                       help='Model to evaluate')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                       choices=['wikitext2', 'pile_10k'], help='Dataset to use')
    
    # Quantization settings  
    parser.add_argument('--quantize_weights', action='store_true',
                       help='Apply weight quantization')
    parser.add_argument('--use_dual_weight_quant', action='store_true', 
                       help='Use dual quantization for weights')
    parser.add_argument('--weight_precision', type=str, default='fp4',
                       choices=['fp4', 'fp8'], help='Weight quantization precision')
    
    # Evaluation settings
    parser.add_argument('--nsamples', type=int, default=128,
                       help='Number of samples to evaluate')
    parser.add_argument('--seqlen', type=int, default=2048,
                       help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--precision', type=str, default='fp32',
                       choices=['fp16', 'fp32'], help='Model precision')
    
    # Output
    parser.add_argument('--output', type=str, help='Output file for results')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Evaluate
    ppl = evaluate_perplexity_with_weight_quant(args)
    
    # Save results
    results = {
        'model': args.model,
        'dataset': args.dataset,
        'quantize_weights': args.quantize_weights,
        'use_dual_weight_quant': args.use_dual_weight_quant,
        'perplexity': ppl,
        'nsamples': args.nsamples,
        'seqlen': args.seqlen
    }
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    else:
        print(f"\nFinal Results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()