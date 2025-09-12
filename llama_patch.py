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
from visualize import collect_qkv, collect_qkv_diff
import time
import math
import numpy as np
from scipy.linalg import hadamard
from scipy.stats import ortho_group

# Global storage for running averages of Q and K means per layer
qk_running_averages = {
    'q_means': {},  # layer_idx -> running average of Q means (1 x H_q x 1 x D)
    'k_means': {},  # layer_idx -> running average of K means (1 x H_k x 1 x D)
    'counts': {}    # layer_idx -> count of samples
}


def compute_and_store_qk_means(query_states, key_states, layer_idx):
    """Compute means of Q and K states and update running averages.
    
    Args:
        query_states: Query tensor of shape [B, H_q, T, D]
        key_states: Key tensor of shape [B, H_k, T, D]
        layer_idx: Layer index for storing averages
        
    Returns:
        query_mean: Mean of query states with shape [1, H_q, 1, D]
        key_mean: Mean of key states with shape [1, H_k, 1, D]
    """
    query_states=query_states.to(torch.float32)
    key_states=key_states.to(torch.float32)
    
  
    # Compute means
    # query_mean = query_states.mean(dim=(0,2), keepdim=True)  # [1, H_q, 1, D]
    # key_mean = key_states.mean(dim=(0,2), keepdim=True)  # [1, H_k, 1, D]

    QT_Q_all_heads = []
    B, H, T, D = query_states.shape

    for h in range(H):
        Q_h = query_states[:, h, :, :]         # (B, T, D)
        Q_h = Q_h.reshape(B * T, D) # (B*T, D)
        QT_Q = Q_h.T @ Q_h          # (D, D)
        QT_Q_all_heads.append(QT_Q)

    query_mean = torch.stack(QT_Q_all_heads)  # (H, D, D)

    query_mean = query_mean.view(4, 8, D, D) 
    query_mean = query_mean.permute(1, 0, 2, 3)
    query_mean = query_mean.mean(dim=1)


    KT_K_all_heads = []
    B, H, T, D = key_states.shape

    for h in range(H):
        K_h = key_states[:, h, :, :]         # (B, T, D)
        K_h = K_h.reshape(B * T, D) # (B*T, D)
        KT_K = K_h.T @ K_h          # (D, D)
        KT_K_all_heads.append(KT_K)

    key_mean = torch.stack(KT_K_all_heads)  # (H, D, D)
    
    # Update running averages (proper incremental averaging) - keep 4D shape
    if layer_idx not in qk_running_averages['counts']:
        qk_running_averages['q_means'][layer_idx] = torch.zeros_like(query_mean).cpu()  # [1, H_q, 1, D]
        qk_running_averages['k_means'][layer_idx] = torch.zeros_like(key_mean).cpu()  # [1, H_k, 1, D]
        qk_running_averages['counts'][layer_idx] = 0
    
    # Compute running average: new_avg = (old_avg * count + new_value) / (count + 1)
    count = qk_running_averages['counts'][layer_idx]
    qk_running_averages['q_means'][layer_idx] = (
        (qk_running_averages['q_means'][layer_idx] * count + 
         query_mean.detach().cpu()) / (count + 1)
    )
    qk_running_averages['k_means'][layer_idx] = (
        (qk_running_averages['k_means'][layer_idx] * count + 
         key_mean.detach().cpu()) / (count + 1)
    )
    qk_running_averages['counts'][layer_idx] += 1
    
    return query_mean, key_mean

# Define TransformersKwargs if not available
try:
    from transformers.modeling_utils import TransformersKwargs
except ImportError:
    # Fallback for older versions
    class TransformersKwargs:
        pass

def permute_adjacent_pairs(Q):
    B,H,T,D = Q.shape
    D=D//2
    d = 2 * D
    P = torch.zeros(d, d).to(Q.device).to(torch.float32)
    for i in range(d):
        if i % 2 == 0:
            P[i, i // 2] = 1          # From first half (real)
        else:
            P[i, D + (i // 2)] = 1    # From second half (imag)
    P=P.transpose(0, 1)
    P_broadcasted = P.unsqueeze(0).expand(H, -1, -1)  # (H, D, D)
    Q_permuted = torch.einsum("bhtd,hde->bhte", Q, P_broadcasted)  # (B, H, T, D)
    return Q_permuted

def batch_matrix_sqrt_and_inv_sqrt(H, eps=1e-10):
    # eigvals, eigvecs = torch.linalg.eigh(H)  # (H, D), (H, D, D)

    # sqrt_vals = torch.sqrt(torch.clamp(eigvals, min=eps))
    # inv_sqrt_vals = 1.0 / sqrt_vals

    # H_sqrt = eigvecs @ torch.diag_embed(sqrt_vals) @ eigvecs.transpose(-1, -2)
    # H_inv_sqrt = eigvecs @ torch.diag_embed(inv_sqrt_vals) @ eigvecs.transpose(-1, -2)
    L = torch.linalg.cholesky(H.cpu())   # (H, D, D), lower triangular
    
    # Square root: take L (triangular square root)
    H_sqrt = L
    
    # Inverse square root: L^{-T}
    # torch.cholesky_inverse gives H^{-1}, but we want H^{-1/2}
    # So directly compute L^{-T}
    L_inv = torch.inverse(L)           # (H, D, D)
    H_invsqrt = L_inv #.transpose(-1, -2)
    
    return H_sqrt, H_invsqrt

def incoherence_processing(Q,K, H, K_mean=None):

    B, H_q, T, D = Q.shape
    B, H_k, T, D = K.shape

    # M = (torch.tensor(hadamard(D), dtype=torch.float32) / math.sqrt(D)).to(Q.device)

    M=torch.tensor(ortho_group.rvs(dim=D), dtype=torch.float32).to(Q.device)
    reg_scale=1e-2

    H.div_(H.diagonal(dim1=-2, dim2=-1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1))

    H.diagonal(dim1=-2, dim2=-1).add_(reg_scale)



    # S = (torch.randn(D) > 0).to(torch.float32) * 2 - 1
    # S=S.to(Q.device)
    
    # scale1 = S.view(1, 1, 1, D)
    C_sqrt, C_inv_sqrt=batch_matrix_sqrt_and_inv_sqrt(H)

    C_sqrt=C_sqrt.to(Q.device)
    C_inv_sqrt=C_inv_sqrt.to(Q.device)

    

    C_inv_sqrt=C_inv_sqrt.repeat(4, 1, 1)

    # for i in range(4):
    #     print(C_inv_sqrt[i,:,:]@C_sqrt)
    # raise
    
    Q = torch.einsum("bhtd,hde->bhte", Q, C_inv_sqrt)
    # Q =  Q* scale1.squeeze(-1)  # (B, H, T, D)
    Q = torch.einsum("bhtd,de->bhte", Q, M)


    
    K = torch.einsum("bhtd,hde->bhte", K, C_sqrt)
    # K =  K* scale1.squeeze(-1)  # (B, H, T, D)
    K = torch.einsum("bhtd,de->bhte", K, M)

    if K_mean is not None:

        # K_mean = torch.einsum("bhtd,hde->bhte", K_mean, C_sqrt)
        # K_mean =  K_mean* scale1.squeeze(-1)  # (B, H, T, D)
        K_mean = torch.einsum("bhtd,de->bhte", K_mean, M)

    return Q, K, K_mean

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

def quantize_q(module, q_orig, dual=True, search=True, permute=False, zero_point=None):

    permute=False

    B, H_q, T, D = q_orig.shape

    q_ = q_orig
    sorted_mean=torch.zeros(H_q,D).to(q_orig.device).to(torch.float32)[None,:,None,:]
    perm=None

    if zero_point=="mean":
        zp = module.qk_mean_averages_after_rope[f'layer_{module.layer_idx}']['q_mean_avg'].to(q_orig.device) #q_orig.mean(dim=(0, 2))
    elif zero_point=="min":
        zp = torch.amin(q_orig.abs(), dim = (0,2)).expand_as(q_orig)

    # if permute:
    #     zp = q_orig.mean(dim=(0, 2))
    #     zp, perm = torch.sort(zp, dim=1)
    #     for h in range(H_q):
    #         q_[:, h, :, :] = q_orig[:, h, :, :][:, :, perm[h]]
    

    if zero_point:
        sorted_mean=zp.to(torch.float32)
        q_ = q_ - sorted_mean

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


    sorted_mean=sorted_mean

    

    return Qq_hi, Qq_lo, Qs_hi, Qs_lo, sorted_mean, perm


def quantize_k(module, k_orig, perm=None, dual=False, search=True, permute=False, zero_point=None):

    permute=False

    if permute:
        k_orig=repeat_kv(k_orig, module.num_key_value_groups)

    B, H_k, T, D = k_orig.shape

    k_ = k_orig
    sorted_mean=torch.zeros(H_k,D).to(k_orig.device).to(torch.float32)[None,:,None,:]


    if zero_point=="mean":
        zp = module.qk_mean_averages_after_rope[f'layer_{module.layer_idx}']['k_mean_avg'].to(k_orig.device) #k_orig.mean(dim=(0, 2))
        if zp.shape[1]!=H_k:
            zp=repeat_kv(zp, module.num_key_value_groups)
    elif zero_point=="min":
        zp = torch.amin(k_orig.abs(), dim = (0,2)).expand_as(k_orig)

    # if permute:
    #     zp = k_orig.mean(dim=(0, 2))
    #     zp, _ = torch.sort(zp, dim=1)
    #     for h in range(H_k):
    #         k_[:, h, :, :] = k_orig[:, h, :, :][:, :, perm[h]]

    if zero_point:
        sorted_mean=zp.to(torch.float32)
        k_ = k_ - sorted_mean

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
    if module.fp4_quantizer.global_sf_max is not None:
        Ks_hi = Ks_hi.reshape(B, T, H_k, 1).permute(0, 2, 1, 3)
        Ks_lo = Ks_lo.reshape(B, T, H_k, 1).permute(0, 2, 1, 3)


    sorted_mean=sorted_mean



    return Kq_hi, Ks_hi, sorted_mean


def quantize_v(module, v_orig, dual=False, search=True, permute=False, zero_point=None):

    permute=False

    B, H_v, T, D = v_orig.shape


    v_ = v_orig
    sorted_mean=torch.zeros(B,T).to(v_orig.device).to(torch.float32)


    if zero_point=="mean":
        zp = v_.mean(dim=(1,3))
    elif zero_point=="min":
        zp = torch.amin(v_.abs(), dim = (1,3))

    # if permute:
    #     zp = v_.mean(dim=(1,3))
    #     zp, perm = torch.sort(zp, dim=1)
    #     inv_perm = torch.argsort(perm, dim=1) 
    #     for h in range(H_v):
    #         v_[:, h, :, :] = v_orig[:, h, :, :][:, :, perm[h]]

    if zero_point:

        sorted_mean=zp.to(torch.float32)
        v_ = v_ - sorted_mean[:,None,:, None]

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
    if module.fp4_quantizer.global_sf_max is not None:
        Vs_hi = Vs_hi.reshape(B, T, H_v, 1).permute(0, 2, 1, 3)
        Vs_lo = Vs_lo.reshape(B, T, H_v, 1).permute(0, 2, 1, 3)


    sorted_mean=sorted_mean[:,None,:, None]

 
    return Vq_hi, Vs_hi, sorted_mean

def quantize_p(module, attn_weights, dual=True, search=True, with_shift=True):


    if not with_shift:    
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

        attn_weights = nn.functional.dropout(attn_weights, p=module.attention_dropout, training=module.training)



        denom = torch.ones(*([1] * attn_weights.ndim),device=attn_weights.device)

    else:
 
        max_val, _ = torch.max(attn_weights, dim=-1, keepdim=True)
        
        num = attn_weights - max_val + 6

        attn_weights = torch.exp(num)

        denom = torch.sum(attn_weights, dim=-1, keepdim=True)

        attn_weights = nn.functional.dropout(attn_weights, p=module.attention_dropout, training=module.training)


    

    original_shape = attn_weights.shape
    attn_weights_2d = attn_weights.reshape(-1, attn_weights.shape[-1])

    if dual:
        Aq_hi, Aq_lo, As_hi, As_lo = module.fp4_quantizer.dual_nvfp4(attn_weights_2d, search=search)

        Aq_hi = Aq_hi.reshape(original_shape)
        Aq_lo = Aq_lo.reshape(original_shape)
        As_hi = As_hi.reshape(*original_shape[:-1],1)
        As_lo = As_lo.reshape(*original_shape[:-1],1)


    else:

        Aq_hi,As_hi = module.fp4_quantizer.single_nvfp4(attn_weights_2d, search=search)
        Aq_hi = Aq_hi.reshape(original_shape)
        As_hi = As_hi.reshape(*original_shape[:-1],1)
        Aq_lo = torch.zeros_like(Aq_hi)
        As_lo = torch.zeros_like(As_hi)

    return Aq_hi, Aq_lo, As_hi, As_lo, denom

def first_block_mask(N, m):

    N_pad = math.ceil(N / m) * m
    nb = N_pad // m 
    mask_first = torch.arange(N_pad).unsqueeze(0) < m 


    
    return ( mask_first.expand(N_pad, N_pad))[:N, :N] 


def block_mask_with_first_block(N, m):

    N_pad = math.ceil(N / m) * m
    nb = N_pad // m 
    mask_first = torch.arange(N_pad).unsqueeze(0) < m 

  
    mask_pad = torch.kron(torch.eye(nb, dtype=torch.bool),
                          torch.ones((m, m), dtype=torch.bool))
    
    return (mask_pad | mask_first)[:N, :N]             


def backward_window_with_first_block(T, m):

    col = torch.arange(T).unsqueeze(0)   
    row = torch.arange(T).unsqueeze(1)  


    mask_window = (col <= row) & (col >= row - (m - 1))


    mask_first = (col < m)

    return mask_window | mask_first


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

    if module.fp_mask and module.quantize:
        full_prec_mask=block_mask_with_first_block(key_states.shape[-2], 64).unsqueeze(0).unsqueeze(0).to(key_states.device)   #can change to backward_window_with_first_block or block_mask_with_first_block also

        attn_weights = torch.where(full_prec_mask, attn_weights_uq, attn_weights) 


    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    if hasattr(module, 'quantize_P') and module.quantize_P == True:
        Aq_hi, Aq_lo, As_hi, As_lo, denom = quantize_p(module, attn_weights, module.use_dual_quant_attn, module.use_P_search, with_shift=module.with_shift)
        attn_weights = (Aq_hi*As_hi+Aq_lo*As_lo)/denom
    else:

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)



    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output.to(torch.bfloat16), attn_weights.to(torch.bfloat16)


def eager_attention_forward_quantized(
    module: nn.Module,
    Qq_hi: torch.Tensor,
    Qq_lo: torch.Tensor,
    Qs_hi: torch.Tensor,
    Qs_lo: torch.Tensor,
    Q_mean: torch.Tensor,
    Kq: torch.Tensor,
    Ks: torch.Tensor,
    K_mean: torch.Tensor,
    Vq: torch.Tensor,
    Vs: torch.Tensor,
    V_mean: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    if Kq.shape != Qq_hi.shape:
        Kq=repeat_kv(Kq, module.num_key_value_groups)
        Ks=repeat_kv(Ks, module.num_key_value_groups)
        K_mean=repeat_kv(K_mean, module.num_key_value_groups)
    Vq=repeat_kv(Vq, module.num_key_value_groups)
    Vs=repeat_kv(Vs, module.num_key_value_groups)



    attn_weights = ((Qs_hi * (Qq_hi @ Kq.transpose(2, 3)) *Ks.transpose(2, 3)) +  (Qs_lo * (Qq_lo @ Kq.transpose(2, 3)) *Ks.transpose(2, 3))  + (Q_mean @ (Kq.transpose(2, 3)*Ks.transpose(2, 3)))  + ((Q_mean + Qq_hi*Qs_hi+Qq_lo*Qs_lo) @ K_mean.transpose(2, 3)) )* scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : (Ks*Kq).shape[-2]]
        attn_weights = attn_weights + causal_mask


    Aq_hi, Aq_lo, As_hi, As_lo, denom = quantize_p(module, attn_weights, Vs, module.use_dual_quant_attn, module.use_P_search, with_shift=module.with_shift)


    attn_output = ((As_hi* (Aq_hi @ Vq)) + (As_lo* (Aq_lo @ Vq)))/denom
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
        self.use_Q_search = os.getenv('FP4_USE_Q_SEARCH', 'true').lower() == 'true'
        self.use_P_search = os.getenv('FP4_USE_P_SEARCH', 'true').lower() == 'true'
        self.use_KV_search = os.getenv('FP4_USE_KV_SEARCH', 'true').lower() == 'true'
        self.use_dual_quant_q = os.getenv('FP4_USE_DUAL_QUANT_Q', 'true').lower() == 'true'
        self.use_dual_quant_attn = os.getenv('FP4_USE_DUAL_QUANT_ATTN', 'true').lower() == 'true'
        self.zero_point_KV  = os.getenv('ZERO_POINT_KV') and os.getenv('ZERO_POINT_KV').lower()
        self.zero_point_Q  = os.getenv("ZERO_POINT_Q", "mean")
        if self.zero_point_Q == "None":
            self.zero_point_Q = None
        self.with_shift = os.getenv('SHIFTED_SM', 'false').lower() == 'true'
        self.mean_before_rope = os.getenv('MEAN_BEFORE_ROPE', 'false').lower() == 'true'
        self.fp_mask = os.getenv('FP_MASK', 'true').lower() == 'true'
        self.ip = os.getenv('IP', 'false').lower() == 'true'
        # Initialize FP4Quantizer with appropriate parameters
        self.fp4_quantizer = FP4Quantizer(global_sf_max=1536, device=self.q_proj.weight.device)
        # Debug: print env var-driven configuration
        print(
            f"FP4_USE_Q_SEARCH={self.use_Q_search}, "
            f"FP4_USE_P_SEARCH={self.use_P_search}, "
            f"FP4_USE_KV_SEARCH={self.use_KV_search}, "
            f"FP4_USE_DUAL_QUANT_Q={self.use_dual_quant_q}, "
            f"FP4_USE_DUAL_QUANT_ATTN={self.use_dual_quant_attn}, "
            f"ZERO_POINT_KV={self.zero_point_KV}, "
            f"ZERO_POINT_Q={self.zero_point_Q}, "
            f"SHIFTED_SM={self.with_shift}, "
            f"MEAN_BEFORE_ROPE={self.mean_before_rope}, "
            f"IP={self.ip}"
        )




    
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    B, H_q, T, D = query_states.shape  
    _, H_kv, _, _ = key_states.shape    

    cos, sin = position_embeddings


    self.quantize = False
    perm = None
    quantize_spec = None
    if hasattr(llama_fp4_attention_forward, 'quantize_enabled'):
        quantize_spec = llama_fp4_attention_forward.quantize_enabled
        self.quantize = True

        letters = "QKVP" if isinstance(quantize_spec, bool) and quantize_spec else str(quantize_spec).upper()

        # Per-tensor toggles; P controls attention-weight quantization inside eager path
        self.quantize_Q = ('Q' in letters)
        self.quantize_K = ('K' in letters)
        self.quantize_V = ('V' in letters)
        self.quantize_P = ('P' in letters)


        self.qk_mean_averages_before_rope=torch.load("qk_mean_averages_before_rope.pt")
        print("Loaded qk_mean_averages_before_rope.pt")
        self.qk_mean_averages_after_rope=torch.load("qk_mean_averages_after_rope.pt")
        print("Loaded qk_mean_averages_after_rope.pt")
        self.q_hessian=torch.load("qk_hessians.pt")

    key_mean=torch.zeros_like(key_states).to(key_states.device)

    if self.layer_idx != 0 and self.mean_before_rope and self.quantize:
        query_states_uq, key_states_uq = apply_rotary_pos_emb(query_states, key_states, cos, sin)


        query_mean=self.qk_mean_averages_before_rope[f'layer_{self.layer_idx}']['q_mean_avg'].to(query_states.device)
        query_states=query_states-query_mean

        
        key_mean=self.qk_mean_averages_before_rope[f'layer_{self.layer_idx}']['k_mean_avg'].to(key_states.device)
        key_states=key_states-key_mean


        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_mean,  key_mean = apply_rotary_pos_emb(query_mean, key_mean, cos, sin)


        query_states=query_states_uq


        if not hasattr(self, 'quantize_K') or (hasattr(self, 'quantize_K') and not self.quantize_K):
            key_states=key_states+key_mean.expand_as(key_states)

        

    else:
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # compute_and_store_qk_means(query_states, key_states, self.layer_idx)
        query_states_uq=query_states
        key_states_uq=key_states



    if hasattr(llama_fp4_attention_forward, 'visualize') and llama_fp4_attention_forward.visualize:
        collect_qkv(query_states, key_states, value_states, self.layer_idx)


    
    if self.layer_idx != 0 and quantize_spec:

        # query_states=permute_adjacent_pairs(query_states.to(torch.float32))
        # key_states=permute_adjacent_pairs(key_states.to(torch.float32))


        if self.ip:
            query_states=query_states.to(torch.float32)
            key_states=key_states.to(torch.float32)
            key_mean=key_mean.to(torch.float32)


            hessian=self.q_hessian[f'layer_{self.layer_idx}']['q_mean_avg'].to(query_states.device).to(torch.float32)
            # QT_Q_all_heads=[]
            # B, H, T, D = query_states.shape
            # Q=query_states
            # for h in range(H):
            #     Q_h = Q[:, h, :, :]         # (B, T, D)
            #     Q_h = Q_h.reshape(B * T, D) # (B*T, D)
            #     QT_Q = Q_h.T @ Q_h          # (D, D)
            #     QT_Q_all_heads.append(QT_Q)

            # hessian = torch.stack(QT_Q_all_heads).to(torch.float32)  # (H, D, D)


            # print((query_states@(key_states_repeated.transpose(2, 3)))[0,0,0:3,0:16])

            query_states, key_states, key_mean= incoherence_processing(query_states, key_states, hessian.clone(), key_mean)


        if self.quantize_Q:
            Qq_hi, Qq_lo, Qs_hi, Qs_lo, Q_mean, perm = quantize_q(self, query_states, dual=True, search=self.use_Q_search, zero_point=self.zero_point_Q)
            query_states = Qq_hi*Qs_hi + Qq_lo*Qs_lo + Q_mean
            # if self.mean_before_rope:   
            #     query_states=query_states+query_mean

        if self.quantize_K:
            Kq, Ks, K_mean = quantize_k(self, key_states, perm, search=True, zero_point=self.zero_point_KV)
            key_states = Kq*Ks + K_mean
            if self.mean_before_rope:

                key_states=key_states+(key_mean)

        if self.quantize_V:
            Vq, Vs, V_mean = quantize_v(self, value_states, search=True, zero_point=None)
            value_states = Vq*Vs + V_mean

    # if past_key_value is not None:
    #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
    #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        
    

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

    # print(torch.mean((attn_output_ref-attn_output)**2))
    # print(torch.mean((attn_weights_ref-attn_weights)**2))
    # raise

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights