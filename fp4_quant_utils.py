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
                 block_size: int,
                 float4_e2m1_max: float,
                 float8_e4m3_max: float,
                 global_sf: float,
                 dequant_dtype: torch.dtype = torch.bfloat16,
                 use_search: bool = False):
        super().__init__()
        self.block_size = block_size
        self.float4_e2m1_max = torch.tensor(float4_e2m1_max, dtype=dequant_dtype)
        self.float8_e4m3_max = torch.tensor(float8_e4m3_max, dtype=dequant_dtype)
        self.global_sf = torch.tensor(global_sf, dtype=dequant_dtype)
        self.zero_tensor = torch.tensor(0.0, dtype=dequant_dtype)
        self.one_tensor = torch.tensor(1.0, dtype=dequant_dtype)
        self.dequant_dtype = dequant_dtype
        self.use_search = use_search

        # print(f"FP4Quantizer initialized with:")
        # print(f"block_size: {block_size}")
        # print(f"float4_e2m1_max: {float4_e2m1_max}")
        # print(f"global_sf: {global_sf}")
        # print(f"dequant_dtype: {dequant_dtype}")
        # print(f"use_search: {use_search}")

    
    def cast_to_fp4(self, x: Tensor) -> Tensor:
        """Cast tensor values to FP4 E2M1 format representation."""
        sign = torch.sign(x)
        x = torch.abs(x)
        
        # Use nested torch.where operations with scalar values
        x = torch.where((x >= 0.0) & (x <= 0.25), 0.0, x)
        x = torch.where((x > 0.25) & (x < 0.75), 0.5, x)
        x = torch.where((x >= 0.75) & (x <= 1.25), 1.0, x)
        x = torch.where((x > 1.25) & (x < 1.75), 1.5, x)
        x = torch.where((x >= 1.75) & (x <= 2.5), 2.0, x)
        x = torch.where((x > 2.5) & (x < 3.5), 3.0, x)
        x = torch.where((x >= 3.5) & (x <= 5.0), 4.0, x)
        x = torch.where(x > 5.0, 6.0, x)
        
        return x * sign

    def get_reciprocal(self, x):
        """Safe reciprocal that handles zeros."""
        if isinstance(x, torch.Tensor):
            return torch.where(x == 0, 0.0, self.one_tensor / x)
        else:
            raise TypeError("Input must be a torch.Tensor.")

    # def single_nvfp4_quant(self, x: Tensor, global_sf: float = None) -> Tuple[Tensor, Tensor]:
    #     """Block-float nvFP4 PTQ (single slice)."""
    #     if global_sf is None:
    #         global_sf = self.global_sf
            
    #     assert x.ndim == 2
    #     m, n = x.shape
    #     n_orig = n

    #     # Ensure n is divisible by block_size
    #     if n % self.block_size != 0:
    #         # Pad to make it divisible
    #         pad_size = self.block_size - (n % self.block_size)
    #         x = torch.nn.functional.pad(x, (0, pad_size), value=0.0)
    #         n = x.shape[1]

    #     x_3d = x.reshape(m, n // self.block_size, self.block_size) n 

    #     # Find max absolute value per block
    #     vec_max = torch.max(torch.abs(x_3d), dim=-1, keepdim=True)[0].to(self.dequant_dtype)

    #     # Calculate scale factor per block
    #     scale = global_sf * (vec_max * self.get_reciprocal(self.float4_e2m1_max))
    #     # Quantize scale to FP8 for realistic simulation
    #     scale = scale.to(torch.float8_e4m3fn).to(self.dequant_dtype)

    #     # Calculate output scale factor (reciprocal for dequantization)
    #     out_sf = self.get_reciprocal(scale * self.get_reciprocal(global_sf))

    #     # Scale input into FP4 range and quantize
    #     scaled = x_3d.to(self.dequant_dtype) * out_sf
    #     clipped = torch.clamp(scaled, -self.float4_e2m1_max, self.float4_e2m1_max)
    #     q = self.cast_to_fp4(clipped).reshape(m, n)

    #     # Return original size if we padded
        # if n != n_orig:
        #     q = q[:, :n_orig]

    #     return q, scale.squeeze(-1)


    def single_nvfp4_quant_dequant_fused(self, x: Tensor, global_sf: float = None, global_scale_aligned : bool = True):
        """Block-float nvFP4 PTQ (single slice)."""
        torch.cuda.empty_cache()
        x = x.to(self.dequant_dtype)
  
        
        assert x.ndim == 2
        m, n = x.shape
        n_orig = n

        # Ensure n is divisible by block_size
        if n % self.block_size != 0:
            # Pad to make it divisible
            pad_size = self.block_size - (n % self.block_size)
            x = torch.nn.functional.pad(x, (0, pad_size), value=0.0)
            n = x.shape[1]

        if global_scale_aligned:
            global_sf = torch.max(abs(x), dim=1, keepdim=True)[0].to(self.dequant_dtype)
        else:
            global_sf = torch.max(abs(x), dim=0, keepdim=True)[0].to(self.dequant_dtype)

        global_sf = global_sf * self.get_reciprocal(self.float4_e2m1_max*self.float8_e4m3_max)

        x=x*self.get_reciprocal(global_sf)


        x = x.reshape(m, n // self.block_size, self.block_size)

        # Find max absolute value per block
        scale = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]

        # Calculate scale factor per block
        scale = (scale)* self.get_reciprocal(self.float4_e2m1_max)
        # Quantize scale to FP8 for realistic simulation

        scale = scale.to(torch.float8_e4m3fn).to(self.dequant_dtype)

        

        # Scale input into FP4 range and quantize
        x = x * self.get_reciprocal(scale) 
        x = torch.clamp(x, -self.float4_e2m1_max, self.float4_e2m1_max)
        x = self.cast_to_fp4(x)
        x = x * scale
        x = x.view(m, n)

        x=x*global_sf
        # Return original size if we padded
        if n != n_orig:
            x = x[:, :n_orig]
        
        torch.cuda.empty_cache()
        return x 

    # def single_nvfp4_dequant(self, q: Tensor, scale: Tensor, global_sf: float = None) -> Tensor:
    #     """Inverse of ref_nvfp4_quant (single slice)."""
    #     if global_sf is None:
    #         global_sf = self.global_sf
            
    #     m, n = q.shape

    #     # Handle padding if needed
    #     orig_n = n
    #     if n % self.block_size != 0:
    #         pad_size = self.block_size - (n % self.block_size)
    #         q = torch.nn.functional.pad(q, (0, pad_size), value=0.0)
    #         n = q.shape[1]

    #     q_3d = q.reshape(m, n // self.block_size, self.block_size)
    #     scale3 = scale.view(m, -1, 1).to(self.dequant_dtype)
    #     x = q_3d * scale3 / global_sf
    #     result = x.reshape(m, n).to(self.dequant_dtype)

    #     # Return original size if we padded
    #     if orig_n != n:
    #         result = result[:, :orig_n]

    #     return result
    def single_nvfp4_qd_nosearch(self, x:Tensor, global_sf: float = None, global_scale_aligned : bool = True):

        torch.cuda.empty_cache()

        x=x.to(torch.float32)

        m_orig, n_orig = x.shape
        n=n_orig
        m=m_orig

        if n_orig % self.block_size != 0 or m_orig % 2 != 0:
            pad_cols = self.block_size - (n_orig % self.block_size) if n_orig % self.block_size != 0 else 0
            pad_rows = 1 if m_orig % 2 != 0 else 0
            x = torch.nn.functional.pad(x, (0, pad_cols, 0, pad_rows), value=0.0)
            n = x.shape[1]
            m = x.shape[0]
        x=x.contiguous()

        if global_scale_aligned:
            global_sf = torch.max(abs(x), dim=1, keepdim=True)[0].to(self.dequant_dtype)
        else:
            global_sf = torch.max(abs(x), dim=0, keepdim=True)[0].to(self.dequant_dtype)

        global_sf = global_sf * self.get_reciprocal(self.float4_e2m1_max*self.float8_e4m3_max)

        x=x*self.get_reciprocal(global_sf)

        x = x.view(m * (n // self.block_size), self.block_size)


        M = x.shape[0]  # total number of groups


        quantized_data = torch.empty(M, dtype=torch.int64, device=x.device)
        scales = torch.empty(M, dtype=torch.float8_e4m3fn, device=x.device)
        

        
        nvfp4sim.f32_to_nvf4_nosearch(quantized_data, scales, x)



        reconstructed_f32 = torch.empty(m * (n // self.block_size), self.block_size, dtype=torch.float32, device=x.device)
        nvfp4sim.nvf4_to_f32(reconstructed_f32, quantized_data, scales)

        # 

        reconstructed_f32 = reconstructed_f32.view(m, n)
        reconstructed_f32=reconstructed_f32*global_sf

        if n != n_orig or m != m_orig:
            reconstructed_f32 = reconstructed_f32[:m_orig, :n_orig]

        torch.cuda.empty_cache()
        return reconstructed_f32.to(torch.bfloat16)


    def single_nvfp4_qd_searched(self, x:Tensor, global_sf: float = None, global_scale_aligned : bool = True):

        torch.cuda.empty_cache()

        x=x.to(torch.float32)

        m_orig, n_orig = x.shape
        n=n_orig
        m=m_orig

        if n_orig % self.block_size != 0 or m_orig % 2 != 0:
            pad_cols = self.block_size - (n_orig % self.block_size) if n_orig % self.block_size != 0 else 0
            pad_rows = 1 if m_orig % 2 != 0 else 0
            x = torch.nn.functional.pad(x, (0, pad_cols, 0, pad_rows), value=0.0)
            n = x.shape[1]
            m = x.shape[0]
        x=x.contiguous()

        if global_scale_aligned:
            global_sf = torch.max(abs(x), dim=1, keepdim=True)[0].to(self.dequant_dtype)
        else:
            global_sf = torch.max(abs(x), dim=0, keepdim=True)[0].to(self.dequant_dtype)

        global_sf = global_sf * self.get_reciprocal(self.float4_e2m1_max*self.float8_e4m3_max)

        x=x*self.get_reciprocal(global_sf)

        x = x.view(m * (n // self.block_size), self.block_size)


        M = x.shape[0]  # total number of groups


        quantized_data = torch.empty(M, dtype=torch.int64, device=x.device)
        scales = torch.empty(M, dtype=torch.float8_e4m3fn, device=x.device)
        

        
        nvfp4sim.f32_to_nvf4(quantized_data, scales, x)



        reconstructed_f32 = torch.empty(m * (n // self.block_size), self.block_size, dtype=torch.float32, device=x.device)
        nvfp4sim.nvf4_to_f32(reconstructed_f32, quantized_data, scales)

        # 

        reconstructed_f32 = reconstructed_f32.view(m, n)
        reconstructed_f32=reconstructed_f32*global_sf

        if n != n_orig or m != m_orig:
            reconstructed_f32 = reconstructed_f32[:m_orig, :n_orig]

        torch.cuda.empty_cache()
        return reconstructed_f32.to(torch.bfloat16)

    def dual_nvfp4_fake_quant(self, x: Tensor, global_sf: float = None, global_scale_aligned : bool = True):
        """Return (q_hi, s_hi, q_lo, s_lo) such that
           x ≈ s_hi · q_hi + s_lo · q_lo
           Both q_hi and q_lo are nvFP4 tensors.
        """

        if not self.use_search:
            x_hi_reconstructed = self.single_nvfp4_qd_nosearch(x, global_sf, global_scale_aligned) #self.single_nvfp4_dequant(q_hi, s_hi, global_sf)
            return x_hi_reconstructed + self.single_nvfp4_qd_nosearch(x - x_hi_reconstructed, global_sf, global_scale_aligned)

        else:
            x_hi_reconstructed = self.single_nvfp4_qd_searched(x, global_sf, global_scale_aligned) 
            return x_hi_reconstructed + self.single_nvfp4_qd_searched(x - x_hi_reconstructed, global_sf, global_scale_aligned)

        
    # def dual_nvfp4_fake_quant(self, x: Tensor, global_sf: float = None):
    #     """Dual-slice FP4 quantization with fake quantization."""
    #     if global_sf is None:
    #         global_sf = self.global_sf
        
    #     # First slice (coarse approximation)
    #     xH, sH, xL, sL = self.dual_nvfp4_quant(x)
    #     xh = self.single_nvfp4_dequant(xH, sH)
    #     xl = self.single_nvfp4_dequant(xL, sL)

    #     # Reconstruct Q from both slices
    #     xd = xh + xl

    #     return xd
    def single_nvfp4_fake_quant(self, x: Tensor, global_sf: float = None, global_scale_aligned : bool = True):

        if not self.use_search:
            return self.single_nvfp4_qd_nosearch(x, global_sf, global_scale_aligned)
        else:
            return self.single_nvfp4_qd_searched(x, global_sf, global_scale_aligned)


