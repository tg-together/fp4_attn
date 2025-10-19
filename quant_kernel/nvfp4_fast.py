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
from kernels.fp32_to_fp4sim import quantize_single
import time


class FastFP4Quantizer(nn.Module):
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


    def single_nvfp4(self, x:Tensor, search: bool = False):

        assert x.dim() == 4, "Input must be 4D: (b, h, n, d)"
        b, h, n, d = x.shape
        assert d % self.block_size == 0, "d must be divisible by block_size"

        x=x.to(torch.float32)

        if self.global_sf_max is not None:
            global_sf = torch.max(abs(x.permute(0, 2, 1, 3).reshape(b, n, h*d)), dim=-1)[0].to(torch.float32)

            global_sf = global_sf * self.get_reciprocal(self.global_sf_max)
            x=x*self.get_reciprocal(global_sf)[:, None, :, None]


        n_orig = x.shape[-1]
        n=n_orig
        pad_cols = (self.block_size - n_orig % self.block_size) % self.block_size

        if pad_cols != 0:
            x = torch.nn.functional.pad(x, (0, pad_cols), value=0.0)
        x=x.contiguous()

        if search:
            assert False, "Search not implemented"
        else:
            reconstructed_f32 = quantize_single(x)

        if n != n_orig:
            reconstructed_f32 = reconstructed_f32[:, :, :, :n_orig]

        if self.global_sf_max is not None:
            return reconstructed_f32, global_sf
        else:
            return reconstructed_f32, torch.ones(*([1] * reconstructed_f32.ndim),device=reconstructed_f32.device)



    def dual_nvfp4(self, x:Tensor, search: bool = False):



        x_hi_q, scales_hi = self.single_nvfp4(x, search) 
        

        x_lo_q, scales_lo = self.single_nvfp4(x - x_hi_q*scales_hi[:, None, :, None], search)

        return x_hi_q, x_lo_q, scales_hi, scales_lo

if __name__ == "__main__":
    quantizer = FastFP4Quantizer()
    x = torch.randn(1, 1, 1024, 128, device="cuda", dtype=torch.float32)
    x_hi_q, x_lo_q, scales_hi, scales_lo = quantizer.dual_nvfp4(x, search=False)
    print(x_hi_q.shape)
    print(x_lo_q.shape)
    print(scales_hi.shape)
    print(scales_lo.shape)


        