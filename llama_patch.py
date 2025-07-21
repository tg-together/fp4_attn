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
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def eager_attention_forward_quantized(
    module: nn.Module,
    Qq_hi: torch.Tensor,
    Qq_lo: torch.Tensor,
    Qs_hi: torch.Tensor,
    Qs_lo: torch.Tensor,
    Kq: torch.Tensor,
    Ks: torch.Tensor,
    Vq: torch.Tensor,
    Vs: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):


    Kq=repeat_kv(Kq, module.num_key_value_groups)
    Vq=repeat_kv(Vq, module.num_key_value_groups)



    attn_weights = ((Qs_hi * (Qq_hi @ Kq.transpose(2, 3)) *Ks.transpose(2, 3)) +  (Qs_lo * (Qq_lo @ Kq.transpose(2, 3)) *Ks.transpose(2, 3))  )* scaling


    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : (Ks*Kq).shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_weights_scaled=attn_weights * Vs.transpose(2, 3)

    if hasattr(llama_fp4_attention_forward, 'quantize_enabled') and llama_fp4_attention_forward.quantize_enabled:
        if hasattr(module, 'fp4_quantizer'):
            # Get environment variable for attention weights quantization method
            use_dual_quant_attn = module.use_dual_quant_attn
            
            # Store original shape
            original_shape = attn_weights_scaled.shape
            
            # Reshape to 2D: [batch_size * num_heads * seq_len, seq_len]
            attn_weights_2d = attn_weights_scaled.view(-1, attn_weights_scaled.shape[-1])

            
            # Apply quantization
            if use_dual_quant_attn:
                Aq_hi, Aq_lo, As_hi, As_lo = module.fp4_quantizer.dual_nvfp4(attn_weights_2d, search=module.use_search)

                Aq_hi = Aq_hi.reshape(original_shape)
                Aq_lo = Aq_lo.reshape(original_shape)
                As_hi = As_hi.reshape(*original_shape[:-1],1)
                As_lo = As_lo.reshape(*original_shape[:-1],1)


            else:

                Aq_hi,As_hi = module.fp4_quantizer.single_nvfp4(attn_weights_2d, search=module.use_search)
                Aq_hi = Aq_hi.reshape(original_shape)
                As_hi = As_hi.reshape(*original_shape[:-1],1)
                Aq_lo = torch.zeros_like(Aq_hi)
                As_lo = torch.zeros_like(As_hi)


    attn_output = (As_hi* (Aq_hi @ Vq)) + (As_lo* (Aq_lo @ Vq))
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output.to(torch.bfloat16), attn_weights.to(torch.bfloat16)


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
        self.dequant_dtype = self.q_proj.weight.dtype
        
        # Get environment variables for configuration
        self.use_search = os.getenv('FP4_USE_SEARCH', 'false').lower() == 'true'
        self.use_dual_quant_q = os.getenv('FP4_USE_DUAL_QUANT_Q', 'true').lower() == 'true'
        self.use_dual_quant_attn = os.getenv('FP4_USE_DUAL_QUANT_ATTN', 'true').lower() == 'true'
        
        # Initialize FP4Quantizer with appropriate parameters
        self.fp4_quantizer = FP4Quantizer(global_sf_max=1536)
        self.fp4_quantizer.float4_e2m1_max = self.fp4_quantizer.float4_e2m1_max.to(device)
        self.fp4_quantizer.float8_e4m3_max = self.fp4_quantizer.float8_e4m3_max.to(device)
        self.fp4_quantizer.global_sf_max = self.fp4_quantizer.global_sf_max.to(device)
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

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    
    # Check if we need to reshape (for quantization or visualization)
    needs_reshape = ((hasattr(llama_fp4_attention_forward, 'quantize_enabled') and llama_fp4_attention_forward.quantize_enabled) or 
                     (hasattr(llama_fp4_attention_forward, 'visualize') and llama_fp4_attention_forward.visualize))
    
    if needs_reshape:

        # Reshape to 2D for quantization/visualization
        q2d = query_states.permute(0, 2, 1, 3).reshape(B * T, H_q * D)
        k2d = key_states.permute(0, 2, 1, 3).reshape(B * T, H_kv * D)
        v2d = value_states.permute(0, 2, 1, 3).reshape(B * T, H_kv * D)

        # q_mean=torch.mean(q2d,dim=0,keepdim=True)
        # k_mean=torch.mean(k2d,dim=0,keepdim=True)
        # v_mean=torch.mean(v2d,dim=0,keepdim=True)

        # q_mean=torch.zeros_like(q2d)
        # k_mean=torch.zeros_like(k2d)
        # v_mean=torch.zeros_like(v2d)

        # q2d=q2d-q_mean
        # k2d=k2d-k_mean
        # v2d=v2d-v_mean

        
        if hasattr(llama_fp4_attention_forward, 'visualize') and llama_fp4_attention_forward.visualize:
            collect_qkv(q2d, k2d, v2d, self.layer_idx)
        # Check if quantization is enabled
        if hasattr(llama_fp4_attention_forward, 'quantize_enabled') and llama_fp4_attention_forward.quantize_enabled:

            
            # Get environment variable for query quantization method
            

            # start_time = time.time_ns()
            # Apply quantization
            if self.use_dual_quant_q:


                
                Qq_hi, Qq_lo, Qs_hi, Qs_lo = self.fp4_quantizer.dual_nvfp4(q2d, search=self.use_search)

                Qq_hi = Qq_hi.reshape(B, T, H_q, D).permute(0, 2, 1, 3)
                Qq_lo = Qq_lo.reshape(B, T, H_q, D).permute(0, 2, 1, 3)
                Qs_hi = Qs_hi.reshape(B, T, 1, 1).permute(0, 2, 1, 3)
                Qs_lo = Qs_lo.reshape(B, T, 1, 1).permute(0, 2, 1, 3)
                # q_mean=q_mean.reshape(1, 1, H_q, D).permute(0, 2, 1, 3)

                
            else:

                Qq_hi, Qs_hi = self.fp4_quantizer.single_nvfp4(q2d, search=self.use_search)
                Qq_hi = Qq_hi.reshape(B, T, H_q, D).permute(0, 2, 1, 3)
                Qs_hi = Qs_hi.reshape(B, T, 1, 1).permute(0, 2, 1, 3)
                Qq_lo = torch.zeros_like(Qq_hi)
                Qs_lo = torch.zeros_like(Qs_hi)
                # q_mean=q_mean.reshape(1, 1, H_q, D).permute(0, 2, 1, 3)


  

            Kq,Ks = self.fp4_quantizer.single_nvfp4(k2d, search=self.use_search)
            Kq = Kq.reshape(B, T, H_kv, D).permute(0, 2, 1, 3)
            Ks = Ks.reshape(B, T, 1, 1).permute(0, 2, 1, 3)
            # k_mean=k_mean.reshape(1, 1, H_kv, D).permute(0, 2, 1, 3)

            
            Vq,Vs = self.fp4_quantizer.single_nvfp4(v2d.T, global_scale_aligned=False, search=self.use_search)
            Vq=Vq.T
            Vs=Vs.T
            Vq = Vq.reshape(B, T, H_kv, D).permute(0, 2, 1, 3)
            Vs = Vs.reshape(B, T, 1, 1).permute(0, 2, 1, 3)
            # v_mean=v_mean.reshape(1, 1, H_kv, D).permute(0, 2, 1, 3)

            if hasattr(llama_fp4_attention_forward, 'visualize') and llama_fp4_attention_forward.visualize:
                collect_qkv_diff((Qq_hi*Qs_hi + Qq_lo*Qs_lo).permute(0, 2, 1, 3).reshape(B * T, H_q * D), (Kq*Ks).permute(0, 2, 1, 3).reshape(B * T, H_kv * D) , (Vq*Vs).permute(0, 2, 1, 3).reshape(B * T, H_kv * D) , q2d, k2d, v2d, self.layer_idx)
            


            # query_states = (Qq_hi*Qs_hi + Qq_lo*Qs_lo) 
            key_states = (Kq*Ks) 
            value_states = (Vq*Vs) 

            # Collect differences if visualizing



        # Reshape back to 4D


    

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = eager_attention_forward


    ##### UNCOMMENT THIS FOR EAGER ATTENTION AND P QUANTIZATION #####

    self.config._attn_implementation = "eager"

    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]


    if torch.isnan(key_states).any() or torch.isnan(value_states).any() or torch.isnan(query_states).any():
        print("K nan:", torch.isnan(key_states).any())
        print("V nan:", torch.isnan(value_states).any())
        print("Q nan:", torch.isnan(query_states).any())
        raise

    if hasattr(llama_fp4_attention_forward, 'quantize_enabled') and llama_fp4_attention_forward.quantize_enabled:


        attn_output, attn_weights = eager_attention_forward_quantized(
            self,
            Qq_hi,
            Qq_lo,
            Qs_hi,
            Qs_lo,
            Kq,
            Ks,
            Vq,
            Vs,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

    else:


        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

    # print(torch.mean((attn_output_ref-attn_output)**2))
    # print(torch.mean((attn_weights_ref-attn_weights)**2))
    # raise

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights