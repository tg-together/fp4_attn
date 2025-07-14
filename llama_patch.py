import argparse
import json
import os
from lm_eval import simple_evaluate
from transformers import AutoModelForCausalLM, AutoConfig
import transformers
import torch
from torch import nn
from typing import Optional, Callable, Unpack
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward, repeat_kv
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput
from fp4_quant_utils import FP4Quantizer
from visualize import collect_qkv, collect_qkv_diff
import time

# Define TransformersKwargs if not available
try:
    from transformers.modeling_utils import TransformersKwargs
except ImportError:
    # Fallback for older versions
    class TransformersKwargs:
        pass


# def print_diff(q_orig, k_orig, v_orig, q_quant, k_quant, v_quant,tag):
    
#     with torch.no_grad():
       
        
#         # Calculate fractional differences (relative to original values)
#         eps = 1e-8  # Small epsilon to avoid division by zero
#         q_diff = torch.abs(q_orig - q_quant) / (torch.abs(q_orig) + eps)
#         k_diff = torch.abs(k_orig - k_quant) / (torch.abs(k_orig) + eps)
#         v_diff = torch.abs(v_orig - v_quant) / (torch.abs(v_orig) + eps)
        
#         # Average across tokens (0th dimension)
#         q_diff_mag = torch.mean(q_diff, dim=0)
#         k_diff_mag = torch.mean(k_diff, dim=0)
#         v_diff_mag = torch.mean(v_diff, dim=0)
        

#         print(f"Diff_{tag}:{k_diff_mag.min().item()},{k_diff_mag.max().item()},{k_diff_mag.mean().item()}")
 
        
#         # Calculate Mean Squared Error
#         q_mse = torch.mean((q_orig - q_quant)**2, dim=0)
#         k_mse = torch.mean((k_orig - k_quant)**2, dim=0)
#         v_mse = torch.mean((v_orig - v_quant)**2, dim=0)
        

#         print(f"MSE_{tag}:{k_mse.min().item()},{k_mse.max().item()},{k_mse.mean().item()}")

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask


    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Check if quantization is enabled and apply to attention weights
    if hasattr(llama_fp4_attention_forward, 'quantize_enabled') and llama_fp4_attention_forward.quantize_enabled:
        if hasattr(module, 'fp4_quantizer'):
            # Get environment variable for attention weights quantization method
            use_dual_quant_attn = os.getenv('FP4_USE_DUAL_QUANT_ATTN', 'true').lower() == 'true'
            
            # Store original shape
            original_shape = attn_weights.shape
            
            # Reshape to 2D: [batch_size * num_heads * seq_len, seq_len]
            attn_weights_2d = attn_weights.view(-1, attn_weights.shape[-1])
            
            # Apply quantization
            if use_dual_quant_attn:
                attn_weights_2d = module.fp4_quantizer.dual_nvfp4_fake_quant(attn_weights_2d)
            else:
                attn_weights_2d = module.fp4_quantizer.single_nvfp4_fake_quant(attn_weights_2d)
            
            # Reshape back to original shape
            attn_weights = attn_weights_2d.view(original_shape)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def llama_fp4_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:

    
    device= self.q_proj.weight.device
    # Initialize FP4Quantizer if not already present
    if not hasattr(self, 'fp4_quantizer'):
        # Infer dtype from q_proj weight
        inferred_dtype = self.q_proj.weight.dtype
        
        # Get environment variables for configuration
        use_kernel = os.getenv('FP4_USE_KERNEL', 'false').lower() == 'true'
        
        # Initialize FP4Quantizer with appropriate parameters
        self.fp4_quantizer = FP4Quantizer(
            block_size=16,  # Common block size for attention quantization
            float4_e2m1_max=6.0,  # Max value for FP4 E2M1 format
            global_sf=0.5,  # Global scale factor
            dequant_dtype=inferred_dtype,
            use_kernel=use_kernel
        )
        self.fp4_quantizer.float4_e2m1_max = self.fp4_quantizer.float4_e2m1_max.to(device)
        self.fp4_quantizer.global_sf = self.fp4_quantizer.global_sf.to(device)
        self.fp4_quantizer.zero_tensor = self.fp4_quantizer.zero_tensor.to(device)
        self.fp4_quantizer.one_tensor = self.fp4_quantizer.one_tensor.to(device)

    
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    B, H_q, T, D = query_states.shape  
    _, H_kv, _, _ = key_states.shape    

    # Reshape to 2D: [B * T, H * D]



    cos, sin = position_embeddings
    
    
    # Check if we need to reshape (for quantization or visualization)
    needs_reshape = ((hasattr(llama_fp4_attention_forward, 'quantize_enabled') and llama_fp4_attention_forward.quantize_enabled) or 
                     (hasattr(llama_fp4_attention_forward, 'visualize') and llama_fp4_attention_forward.visualize))
    
    if needs_reshape:
        # Reshape to 2D for quantization/visualization
        q2d = query_states.permute(0, 2, 1, 3).reshape(B * T, H_q * D)
        k2d = key_states.permute(0, 2, 1, 3).reshape(B * T, H_kv * D)
        v2d = value_states.permute(0, 2, 1, 3).reshape(B * T, H_kv * D)
        
        if hasattr(llama_fp4_attention_forward, 'visualize') and llama_fp4_attention_forward.visualize:
            collect_qkv(q2d, k2d, v2d, self.layer_idx)
        # Check if quantization is enabled
        if hasattr(llama_fp4_attention_forward, 'quantize_enabled') and llama_fp4_attention_forward.quantize_enabled:
            # Store original values for comparison
            if hasattr(llama_fp4_attention_forward, 'visualize') and llama_fp4_attention_forward.visualize:
                pass
            q2d_orig = q2d.clone()
            k2d_orig = k2d.clone()
            v2d_orig = v2d.clone()
            
            # Get environment variable for query quantization method
            use_dual_quant_q = os.getenv('FP4_USE_DUAL_QUANT_Q', 'true').lower() == 'true'

            # start_time = time.time_ns()
            # Apply quantization
            if use_dual_quant_q:
                q2d = self.fp4_quantizer.dual_nvfp4_fake_quant(q2d)
            else:
                q2d = self.fp4_quantizer.single_nvfp4_fake_quant(q2d)
            
            k2d = self.fp4_quantizer.single_nvfp4_fake_quant(k2d)
            v2d = self.fp4_quantizer.single_nvfp4_fake_quant(v2d.T).T
            # print(f"Time taken: {(time.time_ns() - start_time)/1e6:.4f} ms\n")

            # Collect differences if visualizing
            if hasattr(llama_fp4_attention_forward, 'visualize') and llama_fp4_attention_forward.visualize:
                collect_qkv_diff(q2d_orig, k2d_orig, v2d_orig, q2d, k2d, v2d, self.layer_idx)


        # Reshape back to 4D
        query_states = q2d.reshape(B, T, H_q, D).permute(0, 2, 1, 3)
        key_states = k2d.reshape(B, T, H_kv, D).permute(0, 2, 1, 3)
        value_states = v2d.reshape(B, T, H_kv, D).permute(0, 2, 1, 3)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = eager_attention_forward

    self.config._attn_implementation = "eager"

    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]


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
    torch.cuda.empty_cache()
    
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights