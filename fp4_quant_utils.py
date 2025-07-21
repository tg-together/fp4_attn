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
                 global_sf_max: float = 448.0*6.0):
        super().__init__()
        self.block_size = block_size
        self.float4_e2m1_max = torch.tensor(float4_e2m1_max, dtype=dequant_dtype)
        self.float8_e4m3_max = torch.tensor(float8_e4m3_max, dtype=dequant_dtype)
        self.global_sf_max = torch.tensor(global_sf_max, dtype=dequant_dtype)
        self.zero_tensor = torch.tensor(0.0, dtype=dequant_dtype)
        self.one_tensor = torch.tensor(1.0, dtype=dequant_dtype)
        self.dequant_dtype = dequant_dtype

        # print(f"FP4Quantizer initialized with:")
        # print(f"block_size: {block_size}")
        # print(f"float4_e2m1_max: {float4_e2m1_max}")
        # print(f"global_sf_max: {global_sf_max}")
        # print(f"dequant_dtype: {dequant_dtype}")
        # print(f"use_search: {use_search}")

    

    def get_reciprocal(self, x):
        """Safe reciprocal that handles zeros."""
        if isinstance(x, torch.Tensor):
            return torch.where(x == 0, 0.0, self.one_tensor / x)
        else:
            raise TypeError("Input must be a torch.Tensor.")


    def single_nvfp4(self, x:Tensor, global_scale_aligned : bool = True, search: bool = False):


        x=x.to(torch.float32)

        m_orig, n_orig = x.shape
        m=m_orig
        n=n_orig
        pad_cols = (self.block_size - n_orig % self.block_size) % self.block_size
        pad_rows = m_orig % 2  # Will be 1 if odd, 0 if even

        if pad_cols != 0 or pad_rows != 0:
            x = torch.nn.functional.pad(x, (0, pad_cols, 0, pad_rows), value=0.0)
            n = x.shape[1]
            m = x.shape[0]
        x=x.contiguous()

        if global_scale_aligned:
            global_sf = torch.max(abs(x), dim=1, keepdim=True)[0].to(torch.float32)
        else:
            global_sf = torch.max(abs(x), dim=0, keepdim=True)[0].to(torch.float32)

        global_sf = global_sf * self.get_reciprocal(self.global_sf_max)

        x=x*self.get_reciprocal(global_sf)

        x = x.view(m * (n // self.block_size), self.block_size)


        M = x.shape[0]  # total number of groups


        quantized_data = torch.empty(M, dtype=torch.int64, device=x.device)
        scales = torch.empty(M, dtype=torch.float8_e4m3fn, device=x.device)
        

        if search:
            nvfp4sim.f32_to_nvf4(quantized_data, scales, x)
        else:
            print("no search")
            nvfp4sim.f32_to_nvf4_nosearch(quantized_data, scales, x)



        reconstructed_f32 = torch.empty(m * (n // self.block_size), self.block_size, dtype=torch.float32, device=x.device)
        nvfp4sim.nvf4_to_f32(reconstructed_f32, quantized_data, scales)

        # 

        reconstructed_f32 = reconstructed_f32.view(m, n)


        if n != n_orig or m != m_orig:
            reconstructed_f32 = reconstructed_f32[:m_orig, :n_orig]
            global_sf = global_sf[:m_orig, :n_orig]

        return reconstructed_f32, global_sf



    def dual_nvfp4(self, x:Tensor, global_scale_aligned : bool = True, search: bool = False):



        x_hi_q, scales_hi = self.single_nvfp4(x, global_scale_aligned, search) #self.single_nvfp4_dequant(q_hi, s_hi, global_sf)

        x_lo_q, scales_lo = self.single_nvfp4(x - x_hi_q*scales_hi, global_scale_aligned, search)

        return x_hi_q, x_lo_q, scales_hi, scales_lo


        