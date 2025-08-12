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

def quantize_q(module, q_orig, dual=True, search=True, permute=True, zero_point=False):

    B, H_q, T, D = q_orig.shape

    q_ = q_orig
    sorted_mean=torch.zeros(H_q,D).to(q_orig.device)
    perm=None

    mean_q = q_orig.mean(dim=(0, 2))  # torch.amin(q.abs(), dim = (0,2))

    if permute:
        mean_q, perm = torch.sort(mean_q, dim=1)
        for h in range(H_q):
            q_[:, h, :, :] = q_orig[:, h, :, :][:, :, perm[h]]
    

    if zero_point:
        sorted_mean=mean_q
        q_ = q_ - sorted_mean[None, :, None, :]

    q_=q_.permute(0, 2, 1, 3)

    q_=q_.reshape(B * T, H_q, D)

    if dual:
        Qq_hi, Qq_lo, Qs_hi, Qs_lo = module.fp4_quantizer.dual_nvfp4(q_, search=search)
    else:
        Qq_hi, Qs_hi = module.fp4_quantizer.single_nvfp4(q_, search=search)
        Qq_lo = torch.zeros_like(Qq_hi)
        Qs_lo = torch.zeros_like(Qs_hi)

    Qq_hi=Qq_hi.reshape(B, T, H_q, D).permute(0, 2, 1, 3)
    Qq_lo = Qq_lo.reshape(B, T, H_q, D).permute(0, 2, 1, 3)

    if module.fp4_quantizer.global_sf_max is not None:
        Qs_hi = Qs_hi.reshape(B, T, H_q, 1).permute(0, 2, 1, 3)
        Qs_lo = Qs_lo.reshape(B, T, H_q, 1).permute(0, 2, 1, 3)


    sorted_mean=sorted_mean[None, :, None, :]

    

    return Qq_hi, Qq_lo, Qs_hi, Qs_lo, sorted_mean, perm


def quantize_k(module, k_orig, perm=None, dual=False, search=True, permute=True, zero_point=False):

    # k_orig=repeat_kv(k_orig, module.num_key_value_groups)

    B, H_k, T, D = k_orig.shape

    k_ = k_orig
    sorted_mean=torch.zeros(H_k,D).to(k_orig.device)


    mean_k = k_orig.mean(dim=(0, 2))  # torch.amin(q.abs(), dim = (0,2))

    if permute:
        mean_k, _ = torch.sort(mean_k, dim=1)
        for h in range(H_k):
            k_[:, h, :, :] = k_orig[:, h, :, :][:, :, perm[h]]

    if zero_point:
        sorted_mean=mean_k
        k_ = k_ - sorted_mean[None, :, None, :]

    k_=k_.permute(0, 2, 1, 3)

    k_=k_.reshape(B * T, H_k, D)

    if dual:
        Kq_hi, Kq_lo, Ks_hi, Ks_lo = module.fp4_quantizer.dual_nvfp4(k_, search=search)
    else:
        Kq_hi, Ks_hi = module.fp4_quantizer.single_nvfp4(k_, search=search)
        Kq_lo = torch.zeros_like(Kq_hi)
        Ks_lo = torch.zeros_like(Ks_hi)

    Kq_hi=Kq_hi.reshape(B, T, H_k, D).permute(0, 2, 1, 3)
    Kq_lo = Kq_lo.reshape(B, T, H_k, D).permute(0, 2, 1, 3)
    Ks_hi = Ks_hi.reshape(B, T, H_k, 1).permute(0, 2, 1, 3)
    Ks_lo = Ks_lo.reshape(B, T, H_k, 1).permute(0, 2, 1, 3)


    sorted_mean=sorted_mean[None, :, None, :]



    return Kq_hi, Ks_hi, sorted_mean


def quantize_v(module, v_orig, dual=False, search=True, permute=False, zero_point=False):

    B, H_v, T, D = v_orig.shape


    v_ = v_orig
    sorted_mean=torch.zeros(H_v,D).to(v_orig.device)


    mean_v = v_orig.mean(dim=(1, 3))  # torch.amin(q.abs(), dim = (0,2))

    if permute:
   
        mean_v, perm = torch.sort(mean_v, dim=1)
        inv_perm = torch.argsort(perm, dim=1) 
        for h in range(H_v):
            v_[:, h, :, :] = v_orig[:, h, :, :][:, :, perm[h]]

    if zero_point:
  
        sorted_mean=mean_v
        v_ = v_ - sorted_mean[None, :, None, :]

    v_=v_.permute(0, 2, 1, 3)

    v_=v_.reshape(B * T, H_v, D)



    if dual:
        Vq_hi, Vq_lo, Vs_hi, Vs_lo = module.fp4_quantizer.dual_nvfp4(v_, search=search, transpose=True)
    else:

        Vq_hi, Vs_hi = module.fp4_quantizer.single_nvfp4(v_, search=search, transpose=True)
        Vq_lo = torch.zeros_like(Vq_hi)
        Vs_lo = torch.zeros_like(Vs_hi)

    Vq_hi=Vq_hi.reshape(B, T, H_v, D).permute(0, 2, 1, 3)
    Vq_lo = Vq_lo.reshape(B, T, H_v, D).permute(0, 2, 1, 3)
    Vs_hi = Vs_hi.reshape(B, T, H_v, 1).permute(0, 2, 1, 3)
    Vs_lo = Vs_lo.reshape(B, T, H_v, 1).permute(0, 2, 1, 3)


    sorted_mean=sorted_mean[None, :, None, :]

 
    return Vq_hi, Vs_hi, sorted_mean




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

    
    
    # Initialize FP4Quantizer if not already present
    if not hasattr(self, 'fp4_quantizer'):
        # Infer dtype from q_proj weight
        self.dequant_dtype = self.q_proj.weight.dtype
        
        # Get environment variables for configuration
        self.use_search = os.getenv('FP4_USE_SEARCH', 'false').lower() == 'true'
        self.use_dual_quant_q = os.getenv('FP4_USE_DUAL_QUANT_Q', 'true').lower() == 'true'
        self.use_dual_quant_attn = os.getenv('FP4_USE_DUAL_QUANT_ATTN', 'true').lower() == 'true'
        
        # Initialize FP4Quantizer with appropriate parameters
        self.fp4_quantizer = FP4Quantizer(global_sf_max=1536, device=self.q_proj.weight.device)


    
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    B, H_q, T, D = query_states.shape  
    _, H_kv, _, _ = key_states.shape    

    cos, sin = position_embeddings

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


    if hasattr(llama_fp4_attention_forward, 'visualize') and llama_fp4_attention_forward.visualize:
        collect_qkv(query_states, key_states, value_states, self.layer_idx)
    # Check if quantization is enabled
    if hasattr(llama_fp4_attention_forward, 'quantize_enabled') and llama_fp4_attention_forward.quantize_enabled:

        
        Qq_hi, Qq_lo, Qs_hi, Qs_lo, Q_mean, perm = quantize_q(self, query_states, self.use_dual_quant_q, self.use_search)
        Kq, Ks, K_mean = quantize_k(self, key_states, perm, search=self.use_search)
        Vq, Vs, V_mean = quantize_v(self, value_states,search=self.use_search)


    # if past_key_value is not None:
    #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
    #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        query_states=Qq_hi*Qs_hi+Qq_lo*Qs_lo+Q_mean
        key_states=Kq*Ks+K_mean
        value_states=Vq*Vs+V_mean


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

    # if hasattr(llama_fp4_attention_forward, 'quantize_enabled') and llama_fp4_attention_forward.quantize_enabled:


    #     attn_output, attn_weights = eager_attention_forward_quantized(
    #         self,
    #         Qq_hi,
    #         Qq_lo,
    #         Qs_hi,
    #         Qs_lo,
    #         Kq,
    #         Ks,
    #         Vq,
    #         Vs,
    #         attention_mask,
    #         dropout=0.0 if not self.training else self.attention_dropout,
    #         scaling=self.scaling,
    #         **kwargs,
    #     )

    # else:
    
    attn_output, attn_weights = eager_attention_forward(
        self,
        query_states.to(torch.bfloat16),
        key_states.to(torch.bfloat16),
        value_states.to(torch.bfloat16),
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