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
from quant_kernel.nvfp4_fast import FastFP4Quantizer
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
    'k_means': {},  # layer_idx -> running average of K means (1 x H_k x 1 x D)
    'counts': {}    # layer_idx -> count of samples
}

means_running_averages = {
    'k_means': {},  # layer_idx -> running average of K means (1 x H_k x 1 x D) 
    'counts': {}    # layer_idx -> count of samples
}

# Global variable to store the hessian folder path
hessian_folder = None


def store_hessian(query_states, key_states, layer_idx):

    query_states=query_states.to(torch.float32)
    key_states=key_states.to(torch.float32)


    B, H_q, T, D = query_states.shape
    B, H_k, T, D = key_states.shape

    H_all_q = torch.einsum("bhtd,bhte->hde", query_states, query_states)
    H_all_k = torch.einsum("bhtd,bhte->hde", key_states, key_states)

    n_rep=H_q//H_k

    H_grouped = H_all_q.view(H_q // n_rep, n_rep, D, D).mean(dim=1)
    

    if layer_idx not in hessian_running_averages['counts']:
        hessian_running_averages['q_means'][layer_idx] = torch.zeros_like(H_grouped).cpu()
        hessian_running_averages['k_means'][layer_idx] = torch.zeros_like(H_all_k).cpu()
        hessian_running_averages['counts'][layer_idx] = 0
    

    count = hessian_running_averages['counts'][layer_idx]
    hessian_running_averages['q_means'][layer_idx] = (
        (hessian_running_averages['q_means'][layer_idx] * count + 
         H_grouped.detach().cpu()) / (count + 1)
    )
    hessian_running_averages['k_means'][layer_idx] = (
        (hessian_running_averages['k_means'][layer_idx] * count + 
         H_all_k.detach().cpu()) / (count + 1)
    )
    hessian_running_averages['counts'][layer_idx] += 1
    
    return


def store_means(key_states, layer_idx):
    """Store running average of K means per layer.
    Computes mean over batch and tokens, resulting in shape [1, H_k, 1, D]."""
    
    key_states = key_states.to(torch.float32)
    B, H_k, T, D = key_states.shape
    
    # Compute mean over batch and tokens: [B, H_k, T, D] -> [1, H_k, 1, D]
    k_mean = key_states.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True)  # [1, H_k, 1, D]
    
    if layer_idx not in means_running_averages['counts']:
        means_running_averages['k_means'][layer_idx] = torch.zeros_like(k_mean).cpu()
        means_running_averages['counts'][layer_idx] = 0
    
    count = means_running_averages['counts'][layer_idx]
    means_running_averages['k_means'][layer_idx] = (
        (means_running_averages['k_means'][layer_idx] * count + 
         k_mean.detach().cpu()) / (count + 1)
    )
    means_running_averages['counts'][layer_idx] += 1
    
    return


def batch_matrix_sqrt_and_inv_sqrt(H, eps=1e-10):

    L = torch.linalg.cholesky(H.cpu())  
    
    H_sqrt = L

    L_inv = torch.inverse(L)          
    H_invsqrt = L_inv 
    
    return H_sqrt, H_invsqrt

def get_R(QH,KH):

    def batch_sqrtm(H):
       
        eigvals, eigvecs = torch.linalg.eigh(H)  
        sqrtH = eigvecs @ torch.diag_embed(eigvals.clamp(min=0).sqrt()) @ eigvecs.transpose(-2, -1)
        return sqrtH


    KH = KH / KH.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
    QH = QH / QH.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)


    # Square roots
    sKH = batch_sqrtm(KH)   
    sQH = batch_sqrtm(QH)  

    # Batched SVD
    U, S, Vh = torch.linalg.svd(sKH @ sQH)   

    # Build R and its inverse per head
    R = sQH @ Vh.transpose(-2, -1) @ torch.diag_embed(S.rsqrt())
    invR = torch.linalg.inv(R)

    return R, invR

def incoherence_processing(self, Q,K, R, invR):

    B, H_q, T, D = Q.shape
    B, H_k, T, D = K.shape
    
    if self.randomization == 'hadamard':
        M = (torch.tensor(hadamard(D), dtype=torch.float32) / math.sqrt(D)).to(Q.device)
    else:
        M=torch.tensor(ortho_group.rvs(dim=D), dtype=torch.float32).to(Q.device)

    if self.hessians:

        invR=invR.repeat_interleave(dim=0, repeats=H_q//H_k)

        Q = torch.einsum("bhtd,hed->bhte", Q, invR)
        K = torch.einsum("bhtd,hde->bhte", K, R)


    Q = torch.einsum("bhtd,de->bhte", Q, M)

    
    K = torch.einsum("bhtd,de->bhte", K, M)


    return Q, K

def get_block_indices(T, t_block):

    n_blocks = (T + t_block - 1) // t_block  

    first_slice = slice(0, min(t_block, T))
    last_start = (n_blocks - 1) * t_block

    last_block_size = T - last_start
    is_last_incomplete = (last_block_size < t_block)

    if n_blocks > 2:
        middle_slice = slice(t_block, last_start)
    else:
        middle_slice = slice(0, 0) 

    last_slice = slice(last_start, T) if is_last_incomplete else None

    return first_slice, middle_slice, last_slice


def quantize_q(module, q_orig, dual=True):
    if not module.q_quant_log:
        print("Quantizing Q")   
        module.q_quant_log=True

    if dual:
        Qq_hi, Qq_lo, Qs_hi, Qs_lo = module.fast_fp4_quantizer.dual_nvfp4(q_orig, search=False)
    else:
        Qq_hi, Qs_hi = module.fast_fp4_quantizer.single_nvfp4(q_orig, search=True)
        Qq_lo = torch.zeros_like(Qq_hi)
        Qs_lo = torch.zeros_like(Qs_hi)
    
    if module.fast_fp4_quantizer.global_sf_max is not None:
        Qs_hi = Qs_hi[:, :, :, None]
        Qs_lo = Qs_lo[:, :, :, None]

    return Qq_hi, Qq_lo, Qs_hi, Qs_lo


def quantize_k(module, k_orig):

    if not module.k_quant_log:
        print("Quantizing K")
        module.k_quant_log=True

    Kq_hi, Ks_hi = module.fast_fp4_quantizer.single_nvfp4(k_orig, search=True)

    if module.fast_fp4_quantizer.global_sf_max is not None:
        Ks_hi = Ks_hi[:, :, :, None]

    return Kq_hi, Ks_hi


def quantize_v(module, v_orig):

    if not module.v_quant_log:
        print("Quantizing V")
        module.v_quant_log=True

    B, H_v, T, D = v_orig.shape
    v_ = v_orig

    Vq_hi, Vs_hi = module.fast_fp4_quantizer.single_nvfp4(v_orig, search=True, transpose=True)
    Vq_lo = torch.zeros_like(Vq_hi)

    if module.fast_fp4_quantizer.global_sf_max is not None:
        Vs_hi = Vs_hi[:, :, :, None]

    return Vq_hi, Vs_hi

def quantize_p(module, attn_weights, dual=True):

    if not module.p_quant_log:
        print("Quantizing P")
        module.p_quant_log=True


    if dual:
        Aq_hi, Aq_lo, As_hi, As_lo = module.fast_fp4_quantizer.dual_nvfp4(attn_weights, search=False)

    else:

        Aq_hi,As_hi = module.fast_fp4_quantizer.single_nvfp4(attn_weights, search=False)
        Aq_lo = torch.zeros_like(Aq_hi)
        As_lo = torch.zeros_like(As_hi)

    if module.fast_fp4_quantizer.global_sf_max is not None:
        As_hi = As_hi[:, :, :, None]
        As_lo = As_lo[:, :, :, None]

    return Aq_hi, Aq_lo, As_hi, As_lo


def block_mask_with_first_block(N, m):

    N_pad = math.ceil(N / m) * m
    nb = N_pad // m 
    mask_first = torch.arange(N_pad).unsqueeze(0) < m 

  
    mask_pad = torch.kron(torch.eye(nb, dtype=torch.bool),
                          torch.ones((m, m), dtype=torch.bool))
    
    return (mask_pad | mask_first)[:N, :N]             

@torch.no_grad()
def _causal_block_mask(q_start, q_end, k_start, k_end, device):
    qi = torch.arange(q_start, q_end, device=device)[:, None]
    ki = torch.arange(k_start, k_end, device=device)[None, :]
    return ki <= qi

def flash_style_attention(
    module: nn.Module,
    q_q: torch.Tensor,
    k_q: torch.Tensor,
    v_q: torch.Tensor,
    q_uq: torch.Tensor,
    k_uq: torch.Tensor,
    v_uq: torch.Tensor,
    attn_mask=None,       
    causal=False,
    block_q: int = 256,
    block_k: int = 256,
):


    if (attn_mask is not None) and (not torch.is_tensor(attn_mask)):
        raise TypeError(f"attn_mask must be a Tensor or None, got {type(attn_mask)}")
    assert q_q.ndim == k_q.ndim == v_q.ndim == 4
    B, H, Qlen, D  = q_q.shape
    _, _, Klen, Dk = k_q.shape
    _, _, Kv,  Dv  = v_q.shape
    assert D == Dk and Klen == Kv

    if k_q.shape[1]!=q_q.shape[1]:
        k_q = repeat_kv(k_q, module.num_key_value_groups)
        k_uq = repeat_kv(k_uq, module.num_key_value_groups)
    if v_q.shape[1]!=q_q.shape[1]:
        v_q = repeat_kv(v_q, module.num_key_value_groups)
        v_uq = repeat_kv(v_uq, module.num_key_value_groups)

    compute_dtype = torch.float32 
    scale = 1.0 / math.sqrt(D)
    out = torch.empty((B, H, Qlen, Dv), dtype=q_q.dtype, device=q_q.device)

    for q_start in range(0, Qlen, block_q):
        q_block_idx = q_start // block_q
        q_end = min(q_start + block_q, Qlen)
        q_blk = q_q[:, :, q_start:q_end, :].to(compute_dtype)
        q_blk_uq = q_uq[:, :, q_start:q_end, :].to(compute_dtype)


        m   = torch.full((B, H, q_end - q_start, 1), -float('inf'), dtype=compute_dtype, device=q_q.device)
        l   = torch.zeros((B, H, q_end - q_start, 1), dtype=compute_dtype, device=q_q.device)
        acc = torch.zeros((B, H, q_end - q_start, Dv), dtype=compute_dtype, device=q_q.device)

        for k_start in range(0, Klen, block_k):
            k_block_idx = k_start // block_k
            k_end = min(k_start + block_k, Klen)
            k_blk = k_q[:, :, k_start:k_end, :].to(compute_dtype)
            k_blk_uq = k_uq[:, :, k_start:k_end, :].to(compute_dtype)
            v_blk = v_q[:, :, k_start:k_end, :].to(compute_dtype)
            v_blk_uq = v_uq[:, :, k_start:k_end, :].to(compute_dtype)
            q=q_blk
            k=k_blk
            v=v_blk

            if module.layer_idx != 0 and hasattr(module, 'quantize') and module.quantize and module.fp_mask:

                if module.mode=="prefill" and (k_block_idx==0 or k_block_idx==q_block_idx):
                    k=k_blk_uq
                    q=q_blk_uq
                    v=v_blk_uq

                if module.mode=="decode" and k_block_idx==0:
                    k=repeat_kv(module.first_block["K"], module.num_key_value_groups) if module.first_block["K"].shape[1]!=q_blk_uq.shape[1] else module.first_block["K"]
                    q=q_blk_uq
                    v=repeat_kv(module.first_block["V"], module.num_key_value_groups) if module.first_block["V"].shape[1]!=q_blk_uq.shape[1] else module.first_block["V"]

                if module.mode=="decode" and k_end==Klen:
                    k=repeat_kv(module.unquantized_cache["K"], module.num_key_value_groups) if module.unquantized_cache["K"].shape[1]!=q_blk_uq.shape[1] else module.unquantized_cache["K"]
                    q=q_blk_uq
                    v=repeat_kv(module.unquantized_cache["V"], module.num_key_value_groups) if module.unquantized_cache["V"].shape[1]!=q_blk_uq.shape[1] else module.unquantized_cache["V"]
                
            scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale  # [B,H,qb,kb]

            if causal:
                keep = _causal_block_mask(q_start, q_end, k_start, k_end, q.device)  # [qb,kb]
                scores = scores.masked_fill(~keep[None, None, :, :], float("-inf"))

            if attn_mask is not None:
                scores = scores + attn_mask[:, :, q_start:q_end, k_start:k_end]

            m_block, _ = torch.max(scores, dim=-1, keepdim=True)
            m_new = torch.maximum(m, m_block)
            exp_scale = torch.exp(m - m_new)

            p = torch.exp(scores - m_new)

            if module.layer_idx != 0 and hasattr(module, 'quantize') and module.quantize == True:
                p_norm = p / (torch.sum(p, dim=-1, keepdim=True) + 1e-10)
                Aq_hi, Aq_lo, As_hi, As_lo = quantize_p(module, p_norm, module.use_dual_quant_attn)
                p_quant = (Aq_hi*As_hi+Aq_lo*As_lo)
                p = p_quant * (torch.sum(p, dim=-1, keepdim=True) + 1e-10)
            
            l   = exp_scale * l   + torch.sum(p, dim=-1, keepdim=True)
            acc = exp_scale * acc + torch.einsum("bhqk,bhkd->bhqd", p, v)
            m = m_new

            del scores, p, m_block, m_new, exp_scale, k_blk, v_blk

        out[:, :, q_start:q_end, :] = (acc / l).to(q.dtype)
        del q_blk, m, l, acc

    return out.transpose(1, 2).contiguous().to(q_q.dtype),None


def qwen3_fp4_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

    
    self.attention_block_size=int(os.getenv('ATTENTION_BLOCK_SIZE', '256'))
    # Initialize FastFP4Quantizer if not already present
    if hasattr(qwen3_fp4_attention_forward, 'quantize_enabled') and qwen3_fp4_attention_forward.quantize_enabled and not hasattr(self, 'fast_fp4_quantizer'):
        self.quantize = True
        self.dequant_dtype = self.q_proj.weight.dtype
        self.use_dual_quant_q = os.getenv('FP4_USE_DUAL_QUANT_Q', 'false').lower() == 'true'
        self.use_dual_quant_attn = os.getenv('FP4_USE_DUAL_QUANT_ATTN', 'false').lower() == 'true'
        self.fp_mask = os.getenv('FP_MASK', 'true').lower() == 'true'
        self.ip = os.getenv('IP', 'true').lower() == 'true'
        self.randomization = os.getenv('RANDOMIZATION', 'hadamard').lower()
        self.hessians=os.getenv('HESSIANS', 'true').lower() == 'true'
        self.first_block={"K":None, "V":None}
        self.unquantized_cache={"K":None, "V":None}
        
        
        # Load from the correct folder based on hessian_folder
        if self.hessians:
            global hessian_folder
            qk_hessian_file = f"{hessian_folder}/mag_reduce.pt"
            if os.path.exists(qk_hessian_file):
                self.mag_reduce=torch.load(qk_hessian_file)
            else:
                raise ValueError(f"QK Hessian file not found: {qk_hessian_file}")
    
            
        # Initialize FastFP4Quantizer with appropriate parameters
        self.fast_fp4_quantizer = FastFP4Quantizer(global_sf_max=1536, device=self.q_proj.weight.device)
        self.q_quant_log=False
        self.k_quant_log=False
        self.v_quant_log=False
        self.p_quant_log=False
        # Debug: print env var-driven configuration
        print(
            f"FP4_USE_DUAL_QUANT_Q={self.use_dual_quant_q}, "
            f"FP4_USE_DUAL_QUANT_ATTN={self.use_dual_quant_attn}, "
            f"FP_MASK={self.fp_mask}, "
            f"IP={self.ip}, "
            f"RANDOMIZATION={self.randomization}, "
            f"HESSIANS={self.hessians}"
        )




    
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    B, H_q, T_q, D = query_states.shape  
    _, H_kv, T_kv, _ = key_states.shape

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


    if self.layer_idx != 0 and hasattr(self, 'quantize') and self.quantize:

        if self.ip:
            R=None
            invR=None
            if self.hessians:
                R=self.mag_reduce[f'layer_{self.layer_idx}']['R'].to(query_states.device).to(torch.float32)
                invR=self.mag_reduce[f'layer_{self.layer_idx}']['invR'].to(key_states.device).to(torch.float32)
            query_states, key_states = incoherence_processing(self, query_states.to(torch.float32), key_states.to(torch.float32), R, invR)

        value_states = value_states.to(torch.float32)
        query_states_uq, key_states_uq , value_states_uq = query_states, key_states, value_states
        


        if past_key_value is None or len(past_key_value)<self.config.num_hidden_layers: #Prefill stage for this layer

            first_slice, middle_slice, remaining_slice = get_block_indices(T_kv, self.attention_block_size)

            self.first_block["K"] = key_states[:, :, first_slice, :]
            self.first_block["V"] = value_states[:, :, first_slice, :]
            if remaining_slice is not None:
                self.unquantized_cache["K"] = key_states[:, :, remaining_slice, :]
                self.unquantized_cache["V"] = value_states[:, :, remaining_slice, :]
            else:
                self.unquantized_cache["K"] = None
                self.unquantized_cache["V"] = None
            
            self.mode="prefill"


        else: #Decode stage for this layer

            if self.unquantized_cache["K"] is None or self.unquantized_cache["K"].shape[2]==self.attention_block_size:
                self.unquantized_cache["K"] = key_states
                self.unquantized_cache["V"] = value_states
            else:
                self.unquantized_cache["K"] = torch.cat([self.unquantized_cache["K"], key_states], dim=2)
                self.unquantized_cache["V"] = torch.cat([self.unquantized_cache["V"], value_states], dim=2)


            self.mode="decode"           

        
        Qq_hi, Qq_lo, Qs_hi, Qs_lo = quantize_q(self, query_states, dual=self.use_dual_quant_q)
        query_states = Qq_hi*Qs_hi + Qq_lo*Qs_lo 


        Kq, Ks = quantize_k(self, key_states)
        key_states = Kq*Ks 


        Vq, Vs = quantize_v(self, value_states)
        value_states = Vq*Vs 

    else:
       
        query_states_uq=query_states
        key_states_uq=key_states
        value_states_uq=value_states

        if hasattr(qwen3_fp4_attention_forward, 'store_hessian') and qwen3_fp4_attention_forward.store_hessian:
            store_hessian(query_states, key_states, self.layer_idx)


    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = flash_style_attention


    ##### UNCOMMENT THIS FOR EAGER ATTENTION AND P QUANTIZATION #####

    # self.config._attn_implementation = "eager"

    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]


    #default attention interface
    # attn_output, attn_weights = attention_interface(
    #     self,
    #     query_states,
    #     key_states,
    #     value_states,
    #     attention_mask,
    #     dropout=0.0 if not self.training else self.attention_dropout,
    #     scaling=self.scaling,
    #     **kwargs,
    # )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        query_states_uq,
        key_states_uq,
        value_states_uq,
        attn_mask=attention_mask,
        block_q=self.attention_block_size,
        block_k=self.attention_block_size
    )
    if hasattr(self, 'quantize') and self.quantize:
        attn_output = attn_output.to(self.dequant_dtype)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights