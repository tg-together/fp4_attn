#!/usr/bin/env python3
"""
Example usage of qwen3_patch_weightQuant.py for weight quantization.

This script demonstrates how to:
1. Load a Qwen3 model with weight quantization
2. Run inference with quantized weights
3. Analyze quantization error
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

# Import your weight quantization patch
import qwen3_patch_weightQuant as weight_patch


def apply_weight_quantization_to_model(model, use_dual=True):
    """Apply weight quantization to all Qwen3Attention modules in the model."""
    
    # Enable quantization on the forward function
    weight_patch.qwen3_weight_quantized_attention_forward.quantize_enabled = True
    
    # Set environment variables for configuration
    os.environ['FP4_USE_DUAL_WEIGHT_QUANT'] = 'true' if use_dual else 'false'
    
    # Replace attention forward methods
    for name, module in model.named_modules():
        if isinstance(module, Qwen3Attention):
            print(f"Patching attention module: {name}")
            # Replace the forward method
            module.forward = weight_patch.qwen3_weight_quantized_attention_forward.__get__(
                module, Qwen3Attention
            )
    
    print("Weight quantization applied to all attention modules")
    return model


def analyze_model_quantization(model):
    """Analyze quantization error across all attention modules."""
    total_errors = {}
    
    for name, module in model.named_modules():
        if isinstance(module, Qwen3Attention) and hasattr(module, 'weight_quantized'):
            print(f"\nAnalyzing {name}:")
            errors = weight_patch.analyze_quantization_error(module)
            total_errors[name] = errors
    
    return total_errors


def main():
    """Main example demonstrating weight quantization usage."""
    
    # Model configuration
    model_name = "Qwen/Qwen3-4B"  # Use a smaller model for testing
    
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for better quantization analysis
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Apply weight quantization
    print("\nApplying weight quantization...")
    model = apply_weight_quantization_to_model(model, use_dual=True)
    
    # Run a simple inference to trigger quantization
    print("\nRunning inference to trigger weight quantization...")
    test_text = "The future of artificial intelligence is"
    inputs = tokenizer(test_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
    
    # Analyze quantization errors
    print("\n" + "="*50)
    print("QUANTIZATION ERROR ANALYSIS")
    print("="*50)
    errors = analyze_model_quantization(model)
    
    # Calculate average errors across all layers
    if errors:
        avg_errors = {'q_proj': 0, 'k_proj': 0, 'v_proj': 0, 'o_proj': 0}
        layer_count = len(errors)
        
        for layer_errors in errors.values():
            for proj, error in layer_errors.items():
                avg_errors[proj] += error
        
        for proj in avg_errors:
            avg_errors[proj] /= layer_count
        
        print(f"\nAverage quantization errors across {layer_count} layers:")
        for proj, error in avg_errors.items():
            print(f"  {proj}: {error:.6f}")


if __name__ == "__main__":
    main()