#!/usr/bin/env python
#This is simulation code based on the conversation betwen tri and Chris
# --------------------------------------------------
#  NVFP4 simulation playground - CORRECTED VERSION
#  - baseline FP16
#  - single-slice FP4
#  - dual-slice FP4 (Qh + Ql)  + FP4-K/V
# --------------------------------------------------
import torch
import math
from torch import nn, Tensor
from typing import Tuple
import nvfp4sim
import time


class FP4Quantizer(nn.Module):
    """FP4 Quantizer module that handles FP4 quantization and dequantization."""
    
    def __init__(self, 
                 dequant_dtype: torch.dtype = torch.float32,
                 block_size: int = 16,
                 float4_e2m1_max: float = 6.0,
                 float8_e4m3_max: float = 448.0,
                 global_sf_max = 448.0*6.0,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.block_size = block_size
        self.float4_e2m1_max = torch.tensor(float4_e2m1_max, dtype=dequant_dtype, device=device)
        self.float8_e4m3_max = torch.tensor(float8_e4m3_max, dtype=dequant_dtype, device=device)
        if global_sf_max is not None:
            self.global_sf_max = torch.tensor(global_sf_max, dtype=dequant_dtype, device=device)
        else:
            self.global_sf_max = None
        self.zero_tensor = torch.tensor(0.0, dtype=dequant_dtype, device=device)
        self.one_tensor = torch.tensor(1.0, dtype=dequant_dtype, device=device)
        self.dequant_dtype = dequant_dtype

 

    def get_reciprocal(self, x):
        """Safe reciprocal that handles zeros."""
        if isinstance(x, torch.Tensor):
            return torch.where(x == 0, 0.0, self.one_tensor / x)
        else:
            raise TypeError("Input must be a torch.Tensor.")


    def single_nvfp4(self, x:Tensor, search: bool = False, transpose: bool = False):

        #input should be of shape T, H, D
        x=x.to(torch.float32)
        _reshaped=False

        if self.global_sf_max is not None:

    
            global_sf = torch.max(abs(x), dim=-1, keepdim=True)[0].to(torch.float32)

            global_sf = global_sf * self.get_reciprocal(self.global_sf_max)

            x=x*self.get_reciprocal(global_sf)

        if x.dim() == 3:
            T, H, D = x.shape
            x=x.reshape(T,H*D)
            _reshaped=True

        if transpose:
            x=x.T

        m_orig, n_orig = x.shape
        m=m_orig
        n=n_orig
        pad_cols = (self.block_size - n_orig % self.block_size) % self.block_size
        pad_rows = m_orig % 2 

        if pad_cols != 0 or pad_rows != 0:
            x = torch.nn.functional.pad(x, (0, pad_cols, 0, pad_rows), value=0.0)
            n = x.shape[1]
            m = x.shape[0]
        x=x.contiguous()
  


        x = x.view(m * (n // self.block_size), self.block_size)


        M = x.shape[0]  # total number of groups


        quantized_data = torch.empty(M, dtype=torch.int64, device=x.device)
        scales = torch.empty(M, dtype=torch.float8_e4m3fn, device=x.device)
        

        if search:
            nvfp4sim.f32_to_nvf4(quantized_data, scales, x)
        else:
            nvfp4sim.f32_to_nvf4_nosearch(quantized_data, scales, x)



        reconstructed_f32 = torch.empty(m * (n // self.block_size), self.block_size, dtype=torch.float32, device=x.device)
        nvfp4sim.nvf4_to_f32(reconstructed_f32, quantized_data, scales)

        # 

        reconstructed_f32 = reconstructed_f32.view(m, n)


        if n != n_orig or m != m_orig:
            reconstructed_f32 = reconstructed_f32[:m_orig, :n_orig]

        if transpose:
            reconstructed_f32 = reconstructed_f32.T


        
        if _reshaped:
            reconstructed_f32 = reconstructed_f32.reshape(T, H, D)


        if self.global_sf_max is not None:
            return reconstructed_f32, global_sf
        else:
            return reconstructed_f32, torch.ones_like(reconstructed_f32)



    def dual_nvfp4(self, x:Tensor, search: bool = False, transpose: bool = False):



        x_hi_q, scales_hi = self.single_nvfp4(x, search, transpose) 
        

        x_lo_q, scales_lo = self.single_nvfp4(x - x_hi_q*scales_hi, search, transpose)

        return x_hi_q, x_lo_q, scales_hi, scales_lo


        