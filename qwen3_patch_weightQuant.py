import argparse
import json
import os
from pickle import NONE
from lm_eval import simple_evaluate
from transformers import AutoModelForCausalLM, AutoConfig
import transformers
import torch
from torch import nn
from typing import Optional, Callable, Unpack
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward, repeat_kv
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput
from fp4_quant_utils import FP4Quantizer
import time
import math
import numpy as np


# Define TransformersKwargs if not available
try:
    from transformers.modeling_utils import TransformersKwargs
except ImportError:
    # Fallback for older versions
    class TransformersKwargs:
        pass


# Global variable to store the hessian folder path for weight analysis
weight_hessian_folder = None


def store_weight_gradients(module, layer_idx):
    """Store running averages of weight gradients for sensitivity analysis."""
    if not hasattr(store_weight_gradients, 'running_averages'):
        store_weight_gradients.running_averages = {
            'q_weight_grads': {},
            'k_weight_grads': {}, 
            'v_weight_grads': {},
            'o_weight_grads': {},
            'counts': {}
        }
    
    running_averages = store_weight_gradients.running_averages
    
    if layer_idx not in running_averages['counts']:
        running_averages['q_weight_grads'][layer_idx] = torch.zeros_like(module.q_proj.weight).cpu()
        running_averages['k_weight_grads'][layer_idx] = torch.zeros_like(module.k_proj.weight).cpu()
        running_averages['v_weight_grads'][layer_idx] = torch.zeros_like(module.v_proj.weight).cpu()
        running_averages['o_weight_grads'][layer_idx] = torch.zeros_like(module.o_proj.weight).cpu()
        running_averages['counts'][layer_idx] = 0
    
    count = running_averages['counts'][layer_idx]
    
    # Compute gradient norms as proxy for importance
    if module.q_proj.weight.grad is not None:
        q_grad_norm = torch.norm(module.q_proj.weight.grad, dim=1, keepdim=True)
        running_averages['q_weight_grads'][layer_idx] = (
            (running_averages['q_weight_grads'][layer_idx] * count + 
             q_grad_norm.detach().cpu()) / (count + 1)
        )
    
    if module.k_proj.weight.grad is not None:
        k_grad_norm = torch.norm(module.k_proj.weight.grad, dim=1, keepdim=True) 
        running_averages['k_weight_grads'][layer_idx] = (
            (running_averages['k_weight_grads'][layer_idx] * count + 
             k_grad_norm.detach().cpu()) / (count + 1)
        )
    
    if module.v_proj.weight.grad is not None:
        v_grad_norm = torch.norm(module.v_proj.weight.grad, dim=1, keepdim=True)
        running_averages['v_weight_grads'][layer_idx] = (
            (running_averages['v_weight_grads'][layer_idx] * count + 
             v_grad_norm.detach().cpu()) / (count + 1)
        )
    
    if module.o_proj.weight.grad is not None:
        o_grad_norm = torch.norm(module.o_proj.weight.grad, dim=1, keepdim=True)
        running_averages['o_weight_grads'][layer_idx] = (
            (running_averages['o_weight_grads'][layer_idx] * count + 
             o_grad_norm.detach().cpu()) / (count + 1)
        )
    
    running_averages['counts'][layer_idx] += 1


def quantize_weight(weight, fp4_quantizer, dual=False, precision="fp4"):
    """Quantize a weight tensor using FP4 or FP8 quantization.
    
    Args:
        weight: Weight tensor of shape [out_features, in_features]
        fp4_quantizer: FP4Quantizer instance (used for both FP4 and FP8)
        dual: Whether to use dual quantization (only for FP4)
        precision: "fp4" or "fp8" quantization precision
    
    Returns:
        Tuple of (quantized_weight, scales)
    """
    original_shape = weight.shape
    
    # Reshape to 2D if needed for quantization
    weight_2d = weight.view(-1, original_shape[-1])
    
    if precision == "fp8":
        # FP8 quantization - more strict baseline than FP4
        # Use PyTorch's built-in FP8 support or simulate with scaled quantization
        quantized_weight, scales = _quantize_fp8(weight_2d, original_shape)
        print("FP8 quantization!")
        
    elif precision == "fp4":
        if dual:
            Wq_hi, Wq_lo, Ws_hi, Ws_lo = fp4_quantizer.dual_nvfp4(weight_2d, search=True)
            quantized_weight = (Wq_hi * Ws_hi + Wq_lo * Ws_lo).view(original_shape)
            scales = (Ws_hi, Ws_lo)
            print("dual FP4!")
        else:
            Wq, Ws = fp4_quantizer.single_nvfp4(weight_2d, search=True)
            quantized_weight = (Wq * Ws).view(original_shape)
            scales = Ws
            print("single FP4!")
    else:
        raise ValueError(f"Unsupported precision: {precision}. Use 'fp4' or 'fp8'")
    
    return quantized_weight, scales


def _quantize_fp8(weight_2d, original_shape):
    """FP8 quantization implementation.
    
    Uses E4M3 format which is commonly used for weights.
    """
    # FP8 E4M3 has range [-448, 448] approximately
    fp8_max = 448.0
    
    # Compute per-channel scales (more accurate than per-tensor)
    abs_max = torch.max(torch.abs(weight_2d), dim=1, keepdim=True)[0]
    scales = abs_max / fp8_max
    scales = torch.clamp(scales, min=1e-8)  # Avoid division by zero
    
    # Quantize to FP8 E4M3 range
    weight_scaled = weight_2d / scales
    
    # Simulate FP8 E4M3 quantization (PyTorch doesn't have native FP8 yet)
    # Clamp to FP8 range and add small noise to simulate quantization
    weight_fp8 = torch.clamp(weight_scaled, min=-fp8_max, max=fp8_max)
    
    # Simulate limited precision by rounding to fewer mantissa bits
    # FP8 E4M3 has 3 mantissa bits vs FP32's 23 bits
    scale_factor = 2**3  # 3 mantissa bits = 8 levels
    weight_fp8 = torch.round(weight_fp8 * scale_factor) / scale_factor
    
    # Dequantize back to full precision
    quantized_weight = (weight_fp8 * scales).view(original_shape)
    
    return quantized_weight, scales


def create_quantized_linear(original_linear, fp4_quantizer, dual=False, precision="fp4"):
    """Create a quantized version of a linear layer."""
    
    class QuantizedLinear(nn.Module):
        def __init__(self, original_linear, fp4_quantizer, dual=False, precision="fp4"):
            super().__init__()
            self.in_features = original_linear.in_features
            self.out_features = original_linear.out_features
            self.dual = dual
            self.precision = precision
            self.fp4_quantizer = fp4_quantizer
            
            # Quantize the weights
            with torch.no_grad():
                self.quantized_weight, self.scales = quantize_weight(
                    original_linear.weight.data, fp4_quantizer, dual, precision
                )
            
            # Keep bias if present
            if original_linear.bias is not None:
                self.bias = nn.Parameter(original_linear.bias.data.clone())
            else:
                self.bias = None
                
            # Store original weight for comparison/fallback
            self.register_buffer('original_weight', original_linear.weight.data.clone())
            
        def forward(self, x):
            # Use quantized weights for forward pass
            return nn.functional.linear(x, self.quantized_weight, self.bias)
            
        def dequantize_weights(self):
            """Return dequantized weights for analysis."""
            return self.quantized_weight
    
    return QuantizedLinear(original_linear, fp4_quantizer, dual, precision)


def apply_weight_quantization(module, use_dual_weight_quant=False, precision="fp4"):
    """Apply weight quantization to attention module projections."""
    if not hasattr(module, 'weight_quantized'):
        print(f"Applying {precision.upper()} weight quantization to layer {module.layer_idx}")
        
        # Initialize quantizer for weights (FP4Quantizer works for both FP4 and FP8)
        weight_quantizer = FP4Quantizer(
            global_sf_max=None,  # Let each weight matrix find its own scale
            device=module.q_proj.weight.device
        )
        
        # Quantize projection weights
        module.q_proj_quantized = create_quantized_linear(
            module.q_proj, weight_quantizer, dual=use_dual_weight_quant, precision=precision
        )
        module.k_proj_quantized = create_quantized_linear(
            module.k_proj, weight_quantizer, dual=use_dual_weight_quant, precision=precision
        )
        module.v_proj_quantized = create_quantized_linear(
            module.v_proj, weight_quantizer, dual=use_dual_weight_quant, precision=precision
        )
        module.o_proj_quantized = create_quantized_linear(
            module.o_proj, weight_quantizer, dual=use_dual_weight_quant, precision=precision
        )
        
        module.weight_quantized = True
        module.weight_quantizer = weight_quantizer


def qwen3_weight_quantized_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Qwen3 attention forward pass with weight quantization.
    Quantizes the projection weights (Q, K, V, O) instead of activations.
    """
    
    # Initialize weight quantization if not already present
    if (hasattr(qwen3_weight_quantized_attention_forward, 'quantize_enabled') and 
        qwen3_weight_quantized_attention_forward.quantize_enabled and 
        not hasattr(self, 'weight_quantized')):
        
        self.quantize = True
        self.use_dual_weight_quant = os.getenv('FP4_USE_DUAL_WEIGHT_QUANT', 'true').lower() == 'true'
        self.weight_precision = os.getenv('WEIGHT_PRECISION', 'fp4').lower()  # fp4 or fp8
        print ("check, self.weight_precision", self.weight_precision)
        self.weight_quant_log = False
        
        # Apply weight quantization
        apply_weight_quantization(self, self.use_dual_weight_quant, self.weight_precision)
        
        print(f"Weight quantization enabled: precision={self.weight_precision}, dual={self.use_dual_weight_quant}")
    
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Use quantized projections if available, otherwise use original
    if hasattr(self, 'weight_quantized') and self.weight_quantized:
        if not self.weight_quant_log:
            print("Using quantized weights for projections")
            self.weight_quant_log = True
            
        # Apply quantized projections
        query_states = self.q_norm(
            self.q_proj_quantized(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj_quantized(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj_quantized(hidden_states).view(hidden_shape).transpose(1, 2)
    else:
        # Use original projections
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Store weight gradients for analysis if needed
    if (hasattr(qwen3_weight_quantized_attention_forward, 'store_weight_grads') and 
        qwen3_weight_quantized_attention_forward.store_weight_grads):
        store_weight_gradients(self, self.layer_idx)

    # Update past key values
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # Use standard attention mechanism
    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,  
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    
    # Use quantized output projection if available
    if hasattr(self, 'weight_quantized') and self.weight_quantized:
        attn_output = self.o_proj_quantized(attn_output)
        # print ("checking weight quantized output projection!!!")
    else:
        attn_output = self.o_proj(attn_output)
        
    return attn_output, attn_weights


def save_weight_analysis(model_name, dataset, tag=""):
    """Save weight gradient analysis for quantization sensitivity."""
    if not hasattr(store_weight_gradients, 'running_averages'):
        print("No weight gradient data to save")
        return
        
    running_averages = store_weight_gradients.running_averages
    
    if not running_averages['counts']:
        print("No weight analysis to save")
        return
    
    # Collect the averages
    analysis = {}
    for layer_idx in sorted(running_averages['counts'].keys()):
        count = running_averages['counts'][layer_idx]
        if count > 0:
            analysis[f'layer_{layer_idx}'] = {
                'q_weight_sensitivity': running_averages['q_weight_grads'][layer_idx],
                'k_weight_sensitivity': running_averages['k_weight_grads'][layer_idx], 
                'v_weight_sensitivity': running_averages['v_weight_grads'][layer_idx],
                'o_weight_sensitivity': running_averages['o_weight_grads'][layer_idx],
            }
    
    # Create folder structure
    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
    folder_name = f"dumps/{model_short}_{dataset}"
    os.makedirs(folder_name, exist_ok=True)
    
    # Save to file
    filename = f"{folder_name}/weight_analysis_{tag}.pt" if tag else f"{folder_name}/weight_analysis.pt"
    torch.save(analysis, filename)
    
    print(f"Saved weight analysis to {filename}")


# Utility function to analyze quantization error
def analyze_quantization_error(module):
    """Analyze quantization error for each projection."""
    if not hasattr(module, 'weight_quantized') or not module.weight_quantized:
        print("Module not weight quantized")
        return
        
    errors = {}
    
    # Q projection error
    q_orig = module.q_proj.weight.data
    q_quant = module.q_proj_quantized.dequantize_weights()
    q_error = torch.norm(q_orig - q_quant) / torch.norm(q_orig)
    errors['q_proj'] = q_error.item()
    
    # K projection error  
    k_orig = module.k_proj.weight.data
    k_quant = module.k_proj_quantized.dequantize_weights()
    k_error = torch.norm(k_orig - k_quant) / torch.norm(k_orig)
    errors['k_proj'] = k_error.item()
    
    # V projection error
    v_orig = module.v_proj.weight.data
    v_quant = module.v_proj_quantized.dequantize_weights() 
    v_error = torch.norm(v_orig - v_quant) / torch.norm(v_orig)
    errors['v_proj'] = v_error.item()
    
    # O projection error
    o_orig = module.o_proj.weight.data
    o_quant = module.o_proj_quantized.dequantize_weights()
    o_error = torch.norm(o_orig - o_quant) / torch.norm(o_orig)
    errors['o_proj'] = o_error.item()
    
    print(f"Layer {module.layer_idx} quantization errors:")
    for proj, error in errors.items():
        print(f"  {proj}: {error:.6f}")
        
    return errors