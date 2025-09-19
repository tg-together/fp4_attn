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

def incoherence_processing(self, Q,K, QH, KH, K_mean=None):

    B, H_q, T, D = Q.shape
    B, H_k, T, D = K.shape
    
    if self.randomization == 'hadamard':
        M = (torch.tensor(hadamard(D), dtype=torch.float64) / math.sqrt(D)).to(Q.device)
    else:
        M=torch.tensor(ortho_group.rvs(dim=D), dtype=torch.float64).to(Q.device)

    if self.hessians:
        reg_scale=1e-2

        QH.div_(QH.diagonal(dim1=-2, dim2=-1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1))

        QH.diagonal(dim1=-2, dim2=-1).add_(reg_scale)

        KH.div_(KH.diagonal(dim1=-2, dim2=-1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1))
        KH.diagonal(dim1=-2, dim2=-1).add_(reg_scale)

        R, invR=get_R(QH,KH)

        invR=invR.repeat_interleave(dim=0, repeats=H_q//H_k)

        Q = torch.einsum("bhtd,hed->bhte", Q, invR)
        K = torch.einsum("bhtd,hde->bhte", K, R)
        if K_mean is not None:
            K_mean = torch.einsum("bhtd,hde->bhte", K_mean, R)


    Q = torch.einsum("bhtd,de->bhte", Q, M)

    
    K = torch.einsum("bhtd,de->bhte", K, M)

    if K_mean is not None:

        K_mean = torch.einsum("bhtd,de->bhte", K_mean, M)

    return Q, K, K_mean



def quantize_q(module, q_orig, dual=True):
    if not module.q_quant_log:
        print("Quantizing Q")   
        module.q_quant_log=True

    B, H_q, T, D = q_orig.shape

    q_ = q_orig

    q_=q_.permute(0, 2, 1, 3)

    q_=q_.reshape(B * T, H_q, D)

    if dual:
        Qq_hi, Qq_lo, Qs_hi, Qs_lo = module.fp4_quantizer.dual_nvfp4(q_, search=True)
    else:
        Qq_hi, Qs_hi = module.fp4_quantizer.single_nvfp4(q_, search=True)
        Qq_lo = torch.zeros_like(Qq_hi)
        Qs_lo = torch.zeros_like(Qs_hi)

    Qq_hi=Qq_hi.reshape(B, T, H_q, D).permute(0, 2, 1, 3)
    Qq_lo = Qq_lo.reshape(B, T, H_q, D).permute(0, 2, 1, 3)

    if module.fp4_quantizer.global_sf_max is not None:
        Qs_hi = Qs_hi.reshape(B, T, H_q, 1).permute(0, 2, 1, 3)
        Qs_lo = Qs_lo.reshape(B, T, H_q, 1).permute(0, 2, 1, 3)


    return Qq_hi, Qq_lo, Qs_hi, Qs_lo


def quantize_k(module, k_orig):

    if not module.k_quant_log:
        print("Quantizing K")
        module.k_quant_log=True

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

    if not module.v_quant_log:
        print("Quantizing V")
        module.v_quant_log=True

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

    if not module.p_quant_log:
        print("Quantizing P")
        module.p_quant_log=True

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

@torch.no_grad()
def _causal_block_mask(q_start, q_end, k_start, k_end, device):
    qi = torch.arange(q_start, q_end, device=device)[:, None]
    ki = torch.arange(k_start, k_end, device=device)[None, :]
    return ki <= qi

def flash_style_attention(
    module: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_uq: torch.Tensor,
    k_uq: torch.Tensor,
    attn_mask=None,       
    causal=False,
    block_q: int = 128,
    block_k: int = 256,
):

    if (attn_mask is not None) and (not torch.is_tensor(attn_mask)):
        raise TypeError(f"attn_mask must be a Tensor or None, got {type(attn_mask)}")
    assert q.ndim == k.ndim == v.ndim == 4
    B, H, Qlen, D  = q.shape
    _, _, Klen, Dk = k.shape
    _, _, Kv,  Dv  = v.shape
    assert D == Dk and Klen == Kv

    if k.shape[1]!=q.shape[1]:
        k = repeat_kv(k, module.num_key_value_groups)
        k_uq = repeat_kv(k_uq, module.num_key_value_groups)
    if v.shape[1]!=q.shape[1]:
        v = repeat_kv(v, module.num_key_value_groups)

    compute_dtype = torch.float32 
    scale = 1.0 / math.sqrt(D)
    out = torch.empty((B, H, Qlen, Dv), dtype=q.dtype, device=q.device)
    
    # Create full precision mask if needed
    full_prec_mask = None
    if hasattr(module, 'quantize') and module.fp_mask:
        full_prec_mask = block_mask_with_first_block(Klen, 64).to(q.device)

    for q_start in range(0, Qlen, block_q):
        q_end = min(q_start + block_q, Qlen)
        q_blk = q[:, :, q_start:q_end, :].to(compute_dtype)
        q_blk_uq = q_uq[:, :, q_start:q_end, :].to(compute_dtype)

        m   = torch.full((B, H, q_end - q_start, 1), -float('inf'), dtype=compute_dtype, device=q.device)
        l   = torch.zeros((B, H, q_end - q_start, 1), dtype=compute_dtype, device=q.device)
        acc = torch.zeros((B, H, q_end - q_start, Dv), dtype=compute_dtype, device=q.device)

        for k_start in range(0, Klen, block_k):
            k_end = min(k_start + block_k, Klen)
            k_blk = k[:, :, k_start:k_end, :].to(compute_dtype)
            k_blk_uq = k_uq[:, :, k_start:k_end, :].to(compute_dtype)
            v_blk = v[:, :, k_start:k_end, :].to(compute_dtype)

            scores = torch.einsum("bhqd,bhkd->bhqk", q_blk, k_blk) * scale  # [B,H,qb,kb]
            
            # Apply full precision mask if enabled
            if full_prec_mask is not None:
                scores_uq = torch.einsum("bhqd,bhkd->bhqk", q_blk_uq, k_blk_uq) * scale
                mask_block = full_prec_mask[q_start:q_end, k_start:k_end].unsqueeze(0).unsqueeze(0)
                scores = torch.where(mask_block, scores_uq, scores)

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
            acc = exp_scale * acc + torch.einsum("bhqk,bhkd->bhqd", p, v_blk)
            m = m_new

            del scores, p, m_block, m_new, exp_scale, k_blk, v_blk

        out[:, :, q_start:q_end, :] = (acc / l).to(q.dtype)
        del q_blk, m, l, acc

    return out.transpose(1, 2).contiguous().to(q.dtype),None


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
    if hasattr(llama_fp4_attention_forward, 'quantize_enabled') and llama_fp4_attention_forward.quantize_enabled and not hasattr(self, 'fp4_quantizer'):
        self.quantize = True
        self.dequant_dtype = self.q_proj.weight.dtype
        self.use_dual_quant_q = os.getenv('FP4_USE_DUAL_QUANT_Q', 'true').lower() == 'true'
        self.use_dual_quant_attn = os.getenv('FP4_USE_DUAL_QUANT_ATTN', 'true').lower() == 'true'
        self.mean_before_rope = os.getenv('MEAN_BEFORE_ROPE', 'false').lower() == 'true'
        self.fp_mask = os.getenv('FP_MASK', 'true').lower() == 'true'
        self.ip = os.getenv('IP', 'true').lower() == 'true'
        self.randomization = os.getenv('RANDOMIZATION', 'hadamarad').lower()
        self.hessians=os.getenv('HESSIANS', 'true').lower() == 'true'
        
        # Load from the correct folder based on hessian_folder
        if self.hessians:
            global hessian_folder
            qk_hessian_file = f"{hessian_folder}/qk_hessians.pt"
            if os.path.exists(qk_hessian_file):
                self.qk_hessians=torch.load(qk_hessian_file)
            else:
                raise ValueError(f"QK Hessian file not found: {qk_hessian_file}")
    
        
        if self.mean_before_rope:
            k_mean_file = f"{hessian_folder}/k_mean_averages_before_rope.pt"
            if os.path.exists(k_mean_file):
                self.qk_mean_averages_before_rope=torch.load(k_mean_file)
            else:
                raise ValueError(f"K mean file not found: {k_mean_file}")
        
            
        # Initialize FP4Quantizer with appropriate parameters
        self.fp4_quantizer = FP4Quantizer(global_sf_max=1536, device=self.q_proj.weight.device)
        self.q_quant_log=False
        self.k_quant_log=False
        self.v_quant_log=False
        self.p_quant_log=False
        # Debug: print env var-driven configuration
        print(
            f"FP4_USE_DUAL_QUANT_Q={self.use_dual_quant_q}, "
            f"FP4_USE_DUAL_QUANT_ATTN={self.use_dual_quant_attn}, "
            f"MEAN_BEFORE_ROPE={self.mean_before_rope}, "
            f"FP_MASK={self.fp_mask}, "
            f"IP={self.ip}, "
            f"RANDOMIZATION={self.randomization}, "
            f"HESSIANS={self.hessians}"
        )




    
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    B, H_q, T, D = query_states.shape  
    _, H_kv, _, _ = key_states.shape    

    cos, sin = position_embeddings

    if hasattr(llama_fp4_attention_forward, 'store_means') and llama_fp4_attention_forward.store_means:
        store_means(key_states, self.layer_idx)
    
    query_states_uq, key_states_uq = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if self.layer_idx != 0 and hasattr(self, 'quantize') and self.quantize:

        key_mean=torch.zeros_like(key_states).to(key_states.device)

        if self.mean_before_rope:
            # Check if k_mean data exists
            key_mean=self.qk_mean_averages_before_rope[f'layer_{self.layer_idx}']['k_mean_avg'].to(key_states.device)
            key_states=key_states-key_mean
            key_mean, key_states = apply_rotary_pos_emb(key_mean, key_states, cos, sin)
            query_states=query_states_uq
        
        else:
            query_states=query_states_uq
            key_states=key_states_uq



        if self.ip:
            q_hessian=None
            k_hessian=None
            if self.hessians:
                q_hessian=self.qk_hessians[f'layer_{self.layer_idx}']['q_hessian'].to(query_states.device).to(torch.float64)
                k_hessian=self.qk_hessians[f'layer_{self.layer_idx}']['k_hessian'].to(key_states.device).to(torch.float64)
            query_states, key_states, key_mean= incoherence_processing(self, query_states.to(torch.float64), key_states.to(torch.float64), q_hessian, k_hessian, key_mean.to(torch.float64))
        
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
        attn_mask=attention_mask
    )
    if hasattr(self, 'quantize') and self.quantize:
        attn_output = attn_output.to(self.dequant_dtype)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights