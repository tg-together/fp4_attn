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
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward, repeat_kv
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput
from fp4_quant_utils import FP4Quantizer
import time
import math
import numpy as np
from scipy.linalg import hadamard
from scipy.stats import ortho_group


# Define TransformersKwargs if not available
try:
    from transformers.modeling_utils import TransformersKwargs
except ImportError:
    # Fallback for older versions
    class TransformersKwargs:
        pass
hessian_running_averages = {
    'q_means': {},  # layer_idx -> running average of Q means (1 x H_q x 1 x D)
    'counts': {}    # layer_idx -> count of samples
}


def store_hessian(query_states, key_states, layer_idx):

    query_states=query_states.to(torch.float32)
    key_states=key_states.to(torch.float32)


    B, H_q, T, D = query_states.shape
    B, H_k, T, D = key_states.shape

    H_all = torch.einsum("bhtd,bhte->hde", query_states, query_states)

    n_rep=H_q//H_k

    H_grouped = H_all.view(H_q // n_rep, n_rep, D, D).mean(dim=1)
    

    if layer_idx not in hessian_running_averages['counts']:
        hessian_running_averages['q_means'][layer_idx] = torch.zeros_like(H_grouped).cpu()
        hessian_running_averages['counts'][layer_idx] = 0
    

    count = hessian_running_averages['counts'][layer_idx]
    hessian_running_averages['q_means'][layer_idx] = (
        (hessian_running_averages['q_means'][layer_idx] * count + 
         H_grouped.detach().cpu()) / (count + 1)
    )
    hessian_running_averages['counts'][layer_idx] += 1
    
    return


def batch_matrix_sqrt_and_inv_sqrt(H, eps=1e-10):

    L = torch.linalg.cholesky(H.cpu())  
    
    H_sqrt = L

    L_inv = torch.inverse(L)          
    H_invsqrt = L_inv 
    
    return H_sqrt, H_invsqrt

def incoherence_processing(Q,K, H, K_mean=None):

    B, H_q, T, D = Q.shape
    B, H_k, T, D = K.shape


    M=torch.tensor(ortho_group.rvs(dim=D), dtype=torch.float64).to(Q.device)
    reg_scale=1e-2

    H.div_(H.diagonal(dim1=-2, dim2=-1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1))

    H.diagonal(dim1=-2, dim2=-1).add_(reg_scale)

    C_sqrt, C_inv_sqrt=batch_matrix_sqrt_and_inv_sqrt(H)

    C_sqrt=C_sqrt.to(Q.device)
    C_inv_sqrt=C_inv_sqrt.to(Q.device)
    C_inv_sqrt=C_inv_sqrt.repeat_interleave(dim=0, repeats=H_q//H_k)

    # Q = torch.einsum("bhtd,hde->bhte", Q, C_inv_sqrt)
    Q = torch.einsum("bhtd,de->bhte", Q, M)

    # K = torch.einsum("bhtd,hed->bhte", K, C_sqrt)
    K = torch.einsum("bhtd,de->bhte", K, M)

    if K_mean is not None:

        # K_mean = torch.einsum("bhtd,hed->bhte", K_mean, C_sqrt)
        K_mean = torch.einsum("bhtd,de->bhte", K_mean, M)

    return Q, K, K_mean



def quantize_q(module, q_orig, dual=True):

    B, H_q, T, D = q_orig.shape

    q_ = q_orig

    q_=q_.permute(0, 2, 1, 3)

    q_=q_.reshape(B * T, H_q, D)

    if dual:
        Qq_hi, Qq_lo, Qs_hi, Qs_lo = module.fp4_quantizer.dual_nvfp4(q_, search=False)
    else:
        Qq_hi, Qs_hi = module.fp4_quantizer.single_nvfp4(q_, search=False)
        Qq_lo = torch.zeros_like(Qq_hi)
        Qs_lo = torch.zeros_like(Qs_hi)

    Qq_hi=Qq_hi.reshape(B, T, H_q, D).permute(0, 2, 1, 3)
    Qq_lo = Qq_lo.reshape(B, T, H_q, D).permute(0, 2, 1, 3)

    if module.fp4_quantizer.global_sf_max is not None:
        Qs_hi = Qs_hi.reshape(B, T, H_q, 1).permute(0, 2, 1, 3)
        Qs_lo = Qs_lo.reshape(B, T, H_q, 1).permute(0, 2, 1, 3)


    return Qq_hi, Qq_lo, Qs_hi, Qs_lo


def quantize_k(module, k_orig):


    B, H_k, T, D = k_orig.shape

    k_ = k_orig


    k_=k_.permute(0, 2, 1, 3)

    k_=k_.reshape(B * T, H_k, D)


    Kq_hi, Ks_hi = module.fp4_quantizer.single_nvfp4(k_, search=True)


    Kq_hi=Kq_hi.reshape(B, T, H_k, D).permute(0, 2, 1, 3)

    if module.fp4_quantizer.global_sf_max is not None:
        Ks_hi = Ks_hi.reshape(B, T, H_k, 1).permute(0, 2, 1, 3)

    return Kq_hi, Ks_hi


def quantize_v(module, v_orig):

  
    B, H_v, T, D = v_orig.shape


    v_ = v_orig
    

    v_=v_.permute(0, 2, 1, 3)

    v_=v_.reshape(B * T, H_v, D)




    Vq_hi, Vs_hi = module.fp4_quantizer.single_nvfp4(v_, search=True, transpose=True)
    Vq_lo = torch.zeros_like(Vq_hi)

    Vq_hi=Vq_hi.reshape(B, T, H_v, D).permute(0, 2, 1, 3)
    if module.fp4_quantizer.global_sf_max is not None:
        Vs_hi = Vs_hi.reshape(B, T, H_v, 1).permute(0, 2, 1, 3)

    return Vq_hi, Vs_hi

def quantize_p(module, attn_weights, dual=True):

  
    original_shape = attn_weights.shape
    attn_weights_2d = attn_weights.reshape(-1, attn_weights.shape[-1])

    if dual:
        Aq_hi, Aq_lo, As_hi, As_lo = module.fp4_quantizer.dual_nvfp4(attn_weights_2d, search=False)

        Aq_hi = Aq_hi.reshape(original_shape)
        Aq_lo = Aq_lo.reshape(original_shape)
        As_hi = As_hi.reshape(*original_shape[:-1],1)
        As_lo = As_lo.reshape(*original_shape[:-1],1)


    else:

        Aq_hi,As_hi = module.fp4_quantizer.single_nvfp4(attn_weights_2d, search=False)
        Aq_hi = Aq_hi.reshape(original_shape)
        As_hi = As_hi.reshape(*original_shape[:-1],1)
        Aq_lo = torch.zeros_like(Aq_hi)
        As_lo = torch.zeros_like(As_hi)

    return Aq_hi, Aq_lo, As_hi, As_lo


def block_mask_with_first_block(N, m):

    N_pad = math.ceil(N / m) * m
    nb = N_pad // m 
    mask_first = torch.arange(N_pad).unsqueeze(0) < m 

  
    mask_pad = torch.kron(torch.eye(nb, dtype=torch.bool),
                          torch.ones((m, m), dtype=torch.bool))
    
    return (mask_pad | mask_first)[:N, :N]             




def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_uq: torch.Tensor,
    key_uq: torch.Tensor,
    value_uq: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    if key.shape[1]!=query.shape[1]:
        key_states = repeat_kv(key, module.num_key_value_groups)
    else:
        key_states=key

    value_states = repeat_kv(value, module.num_key_value_groups)

    key_states_uq=repeat_kv(key_uq, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    attn_weights_uq=torch.matmul(query_uq, key_states_uq.transpose(2, 3)) * scaling

    if hasattr(module, 'quantize') and module.fp_mask:
        full_prec_mask=block_mask_with_first_block(key_states.shape[-2], 64).unsqueeze(0).unsqueeze(0).to(key_states.device) 

        attn_weights = torch.where(full_prec_mask, attn_weights_uq, attn_weights) 


    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    if module.layer_idx != 0 and hasattr(module, 'quantize') and module.quantize == True:
        Aq_hi, Aq_lo, As_hi, As_lo = quantize_p(module, attn_weights, module.use_dual_quant_attn)
        attn_weights = (Aq_hi*As_hi+Aq_lo*As_lo)
    
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)



    attn_output = torch.matmul(attn_weights, value_states)
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
    if not hasattr(self, 'fp4_quantizer') and hasattr(llama_fp4_attention_forward, 'quantize_enabled'):
        self.quantize = True
        self.dequant_dtype = self.q_proj.weight.dtype
        


        self.use_dual_quant_q = os.getenv('FP4_USE_DUAL_QUANT_Q', 'true').lower() == 'true'
        self.use_dual_quant_attn = os.getenv('FP4_USE_DUAL_QUANT_ATTN', 'true').lower() == 'true'
        self.mean_before_rope = os.getenv('MEAN_BEFORE_ROPE', 'false').lower() == 'true'
        self.fp_mask = os.getenv('FP_MASK', 'true').lower() == 'true'
        self.ip = os.getenv('IP', 'true').lower() == 'true'
        self.qk_mean_averages_before_rope=torch.load("qk_mean_averages_before_rope.pt")
        self.q_hessian=torch.load("q_hessians.pt")
        # Initialize FP4Quantizer with appropriate parameters
        self.fp4_quantizer = FP4Quantizer(global_sf_max=1536, device=self.q_proj.weight.device)
        # Debug: print env var-driven configuration
        # print(
        #     f"FP4_USE_Q_SEARCH={self.use_Q_search}, "
        #     f"FP4_USE_P_SEARCH={self.use_P_search}, "
        #     f"FP4_USE_KV_SEARCH={self.use_KV_search}, "
        #     f"FP4_USE_DUAL_QUANT_Q={self.use_dual_quant_q}, "
        #     f"FP4_USE_DUAL_QUANT_ATTN={self.use_dual_quant_attn}, "
        #     f"ZERO_POINT_KV={self.zero_point_KV}, "
        #     f"ZERO_POINT_Q={self.zero_point_Q}, "
        #     f"SHIFTED_SM={self.with_shift}, "
        #     f"MEAN_BEFORE_ROPE={self.mean_before_rope}, "
        #     f"IP={self.ip}"
        # )




    
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    B, H_q, T, D = query_states.shape  
    _, H_kv, _, _ = key_states.shape    

    cos, sin = position_embeddings


    
    query_states_uq, key_states_uq = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if self.layer_idx != 0 and hasattr(self, 'quantize') and self.quantize:

        key_mean=torch.zeros_like(key_states).to(key_states.device)

        if self.mean_before_rope:

            
            key_mean=self.qk_mean_averages_before_rope[f'layer_{self.layer_idx}']['k_mean_avg'].to(key_states.device)
            key_states=key_states-key_mean


            key_mean, key_states = apply_rotary_pos_emb(key_mean, key_states, cos, sin)

            query_states=query_states_uq
        
        else:
            query_states=query_states_uq
            key_states=key_states_uq



        if self.ip:

            hessian=self.q_hessian[f'layer_{self.layer_idx}']['q_hessian'].to(query_states.device)
            query_states, key_states, key_mean= incoherence_processing(query_states.to(torch.float64), key_states.to(torch.float64), hessian.to(torch.float64), key_mean.to(torch.float64))



        Qq_hi, Qq_lo, Qs_hi, Qs_lo = quantize_q(self, query_states, dual=self.use_dual_quant_q)
        query_states = Qq_hi*Qs_hi + Qq_lo*Qs_lo 


        Kq, Ks = quantize_k(self, key_states)
        key_states = Kq*Ks 
        if self.mean_before_rope:
            key_states=key_states+(key_mean)


        Vq, Vs = quantize_v(self, value_states)
        value_states = Vq*Vs 

        

    else:

        query_states=query_states_uq
        key_states=key_states_uq

        if hasattr(llama_fp4_attention_forward, 'store_hessian') and llama_fp4_attention_forward.store_hessian:
            store_hessian(query_states, key_states, self.layer_idx)


    attention_interface: Callable = eager_attention_forward


    ##### UNCOMMENT THIS FOR EAGER ATTENTION AND P QUANTIZATION #####

    self.config._attn_implementation = "eager"

    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = eager_attention_forward(
        self,
        query_states.to(torch.float32),
        key_states.to(torch.float32),
        value_states.to(torch.float32),
        query_states_uq,
        key_states_uq,
        None,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )



    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights